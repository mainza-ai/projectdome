import os
import sys
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.training.dataset import VocasetDataset, collate_fn, VOCA_ALL_SUBJECTS, VOCA_TRAIN_SUBJECTS, VOCA_VAL_SUBJECTS, VOCA_TEST_SUBJECTS
from src.training.model import SpeechToCoefficientsModel

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

def train(epochs=50, batch_size=8, lr=1e-4, hidden_dim=256):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    log.info(f"Training on device: {device}")
    log.info(f"VOCA split: {len(VOCA_TRAIN_SUBJECTS)} train, {len(VOCA_VAL_SUBJECTS)} val, {len(VOCA_TEST_SUBJECTS)} test")
    log.info(f"  Train: {VOCA_TRAIN_SUBJECTS}")
    log.info(f"  Val:   {VOCA_VAL_SUBJECTS}")
    log.info(f"  Test:  {VOCA_TEST_SUBJECTS}")
    try:
        train_dataset = VocasetDataset(split="train")
        val_dataset = VocasetDataset(split="val")
    except FileNotFoundError as e:
        log.error(f"Dataset error: {e}")
        log.error("Run reproject_vocaset.py first.")
        sys.exit(1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    model = SpeechToCoefficientsModel(hidden_dim=hidden_dim).to(device)
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
            total_loss = l1_loss + 0.5 * vel_loss + 0.2 * accel_loss + 0.1 * edge_loss + 1e-4 * reg_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
