import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.training.dataset import VocasetDataset, collate_fn, VOCA_ALL_SUBJECTS, VOCA_TRAIN_SUBJECTS, VOCA_VAL_SUBJECTS, VOCA_TEST_SUBJECTS
from src.training.model import SpeechToCoefficientsModel
from src.training.config import TrainingConfig

try:
    from gnm.shape.gnm_numpy import GNM
    from gnm.shape.data.versions.gnm_specs import GNMMajorVersion, GNMVariant
    _GNM_AVAILABLE = True
except ImportError:
    _GNM_AVAILABLE = False

log = logging.getLogger('training')

def compute_velocity_loss(pred, target):
    if pred.size(1) < 2:
        return torch.tensor(0.0, device=pred.device)
    pred_vel = pred[:, 1:] - pred[:, :-1]
    target_vel = target[:, 1:] - target[:, :-1]
    return nn.functional.l1_loss(pred_vel, target_vel)

def compute_acceleration_loss(pred, target):
    if pred.size(1) < 3:
        return torch.tensor(0.0, device=pred.device)
    pred_acc = pred[:, 2:] - 2.0 * pred[:, 1:-1] + pred[:, :-2]
    target_acc = target[:, 2:] - 2.0 * target[:, 1:-1] + target[:, :-2]
    return nn.functional.l1_loss(pred_acc, target_acc)

def compute_regularization_loss(pred):
    return torch.mean(torch.abs(pred))

def compute_edge_loss(pred, target):
    if pred.size(1) < 2:
        return torch.tensor(0.0, device=pred.device)
    pred_edges = pred[:, 1:] - pred[:, :-1]
    target_edges = target[:, 1:] - target[:, :-1]
    edge_diff = torch.abs(pred_edges - target_edges)
    return edge_diff.mean()

_gnm_model = None
def compute_vertex_loss(pred_coeffs, target_coeffs, identity=None):
    global _gnm_model
    if not _GNM_AVAILABLE:
        return torch.tensor(0.0, device=pred_coeffs.device)
    if _gnm_model is None:
        _gnm_model = GNM.from_local(GNMMajorVersion.V3, GNMVariant.HEAD)
    device = pred_coeffs.device
    identity_np = np.zeros(_gnm_model.identity_dim, dtype=np.float32) if identity is None else identity
    rotations = np.zeros((_gnm_model.num_joints, 3), dtype=np.float32)
    translation = np.zeros(3, dtype=np.float32)
    total_loss = 0.0
    B, T, D = pred_coeffs.shape
    n_vertices = _gnm_model.num_vertices
    for b in range(B):
        for t in range(T):
            expr_pred = np.zeros(_gnm_model.expression_dim, dtype=np.float32)
            expr_target = np.zeros(_gnm_model.expression_dim, dtype=np.float32)
            expr_pred[200:382] = pred_coeffs[b, t].detach().cpu().numpy()
            expr_target[200:382] = target_coeffs[b, t].detach().cpu().numpy()
            v_pred = _gnm_model(identity_np, expr_pred, rotations, translation)
            v_target = _gnm_model(identity_np, expr_target, rotations, translation)
            total_loss += float(np.mean(np.abs(v_pred - v_target)))
    return torch.tensor(total_loss / (B * T), device=device)

def train(epochs=50, batch_size=8, lr=1e-4, hidden_dim=256, config: TrainingConfig = None):
    if config is None:
        config = TrainingConfig()
        config.epochs = epochs
        config.batch_size = batch_size
        config.learning_rate = lr
        config.hidden_dim = hidden_dim

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    log.info(f"Training on device: {device}")
    log.info(f"VOCA split: {len(VOCA_TRAIN_SUBJECTS)} train, {len(VOCA_VAL_SUBJECTS)} val, {len(VOCA_TEST_SUBJECTS)} test")
    log.info(f"  Train: {VOCA_TRAIN_SUBJECTS}")
    log.info(f"  Val:   {VOCA_VAL_SUBJECTS}")
    log.info(f"  Test:  {VOCA_TEST_SUBJECTS}")
    log.info(f"Config: feature={config.feature_type}, vertex_loss={config.use_vertex_loss}")
    try:
        train_dataset = VocasetDataset(split="train")
        val_dataset = VocasetDataset(split="val")
    except FileNotFoundError as e:
        log.error(f"Dataset error: {e}")
        log.error("Run reproject_vocaset.py first.")
        sys.exit(1)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    model = SpeechToCoefficientsModel(hidden_dim=config.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    l1_loss_fn = nn.L1Loss()
    checkpoint_dir = "voca/model/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_l1 = 0.0
        train_vel = 0.0
        train_accel = 0.0
        train_edge = 0.0
        train_reg = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} (Train)"):
            features = batch["audio_features"].to(device)
            targets = batch["coefficients"].to(device)
            speaker_ids = batch["speaker_ids"].to(device)
            padding_mask = torch.zeros(features.size(0), features.size(1), dtype=torch.bool, device=device)
            for idx, seq_len in enumerate(batch["seq_lens"]):
                padding_mask[idx, seq_len:] = True
            optimizer.zero_grad()
            preds = model(features, speaker_ids=speaker_ids, src_key_padding_mask=padding_mask)
            loss_mask = (~padding_mask).unsqueeze(-1).float()
            l1_loss = l1_loss_fn(preds * loss_mask, targets * loss_mask)
            vel_loss = compute_velocity_loss(preds * loss_mask, targets * loss_mask)
            accel_loss = compute_acceleration_loss(preds * loss_mask, targets * loss_mask)
            edge_loss = compute_edge_loss(preds * loss_mask, targets * loss_mask)
            reg_loss = compute_regularization_loss(preds * loss_mask)
            total_loss = (config.loss_position_weight * l1_loss +
                          config.loss_velocity_weight * vel_loss +
                          config.loss_acceleration_weight * accel_loss +
                          config.loss_edge_weight * edge_loss +
                          config.loss_regularization_weight * reg_loss)
            if config.use_vertex_loss and _GNM_AVAILABLE:
                vertex_loss = compute_vertex_loss(preds, targets)
                total_loss = total_loss + config.loss_vertex_weight * vertex_loss
                train_vertex = train_vertex + vertex_loss.item() if 'train_vertex' in dir() else vertex_loss.item()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip)
            optimizer.step()
            train_loss += total_loss.item()
            train_l1 += l1_loss.item()
            train_vel += vel_loss.item()
            train_accel += accel_loss.item()
            train_edge += edge_loss.item()
            train_reg += reg_loss.item()
        model.eval()
        val_loss = 0.0
        val_l1 = 0.0
        val_vel = 0.0
        val_accel = 0.0
        val_edge = 0.0
        val_reg = 0.0
        with torch.no_grad():
            for batch in val_loader:
                features = batch["audio_features"].to(device)
                targets = batch["coefficients"].to(device)
                speaker_ids = batch["speaker_ids"].to(device)
                padding_mask = torch.zeros(features.size(0), features.size(1), dtype=torch.bool, device=device)
                for idx, seq_len in enumerate(batch["seq_lens"]):
                    padding_mask[idx, seq_len:] = True
                preds = model(features, speaker_ids=speaker_ids, src_key_padding_mask=padding_mask)
                loss_mask = (~padding_mask).unsqueeze(-1).float()
                l1_loss = l1_loss_fn(preds * loss_mask, targets * loss_mask)
                vel_loss = compute_velocity_loss(preds * loss_mask, targets * loss_mask)
                accel_loss = compute_acceleration_loss(preds * loss_mask, targets * loss_mask)
                edge_loss = compute_edge_loss(preds * loss_mask, targets * loss_mask)
                reg_loss = compute_regularization_loss(preds * loss_mask)
                total_loss = l1_loss + 0.5 * vel_loss + 0.2 * accel_loss + 0.1 * edge_loss + 1e-4 * reg_loss
                val_loss += total_loss.item()
                val_l1 += l1_loss.item()
                val_vel += vel_loss.item()
                val_accel += accel_loss.item()
                val_edge += edge_loss.item()
                val_reg += reg_loss.item()
        train_loss /= len(train_loader)
        train_l1 /= len(train_loader)
        train_vel /= len(train_loader)
        train_accel /= len(train_loader)
        train_edge /= len(train_loader)
        train_reg /= len(train_loader)
        val_loss /= len(val_loader)
        val_l1 /= len(val_loader)
        val_vel /= len(val_loader)
        val_accel /= len(val_loader)
        val_edge /= len(val_loader)
        val_reg /= len(val_loader)
        log.info(f"Epoch {epoch:2d} — train: {train_loss:.6f} (L1:{train_l1:.4f} Vel:{train_vel:.4f} Accel:{train_accel:.4f} Edge:{train_edge:.4f})")
        log.info(f"               val:   {val_loss:.6f} (L1:{val_l1:.4f} Vel:{val_vel:.4f} Accel:{val_accel:.4f} Edge:{val_edge:.4f})")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "speaker_labels": VOCA_ALL_SUBJECTS,
            }, checkpoint_path)
            log.info(f"  --> Saved best checkpoint (epoch {epoch}, val loss: {val_loss:.6f})")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'
    )
    import argparse
    parser = argparse.ArgumentParser(description="Train SpeechToCoefficientsModel on reprojected VOCASET.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch, lr=args.lr)
