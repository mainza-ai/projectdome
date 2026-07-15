import os
import sys
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.training.dataset import VocasetDataset, collate_fn
from src.training.model import SpeechToCoefficientsModel

log = logging.getLogger('training')

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    log.info(f"Evaluating on device: {device}")
    checkpoint_path = "voca/model/checkpoints/best_model.pt"
    if not os.path.exists(checkpoint_path):
        log.error(f"Checkpoint not found at {checkpoint_path}. Train the model first.")
        sys.exit(1)
    log.info("Loading test split (VOCA subjects: FaceTalk_170809_00138_TA, FaceTalk_170731_00024_TA)...")
    test_dataset = VocasetDataset(split="test")
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = SpeechToCoefficientsModel().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    epoch_info = checkpoint.get('epoch', '?')
    val_loss = checkpoint.get('val_loss', '?')
    speaker_labels = checkpoint.get('speaker_labels', 'N/A')
    log.info(f"Loaded checkpoint from epoch {epoch_info} (val loss: {val_loss})")
    if speaker_labels != 'N/A':
        log.info(f"Speaker labels in checkpoint: {speaker_labels}")
    total_l1 = 0.0
    total_vel = 0.0
    total_edge = 0.0
    l1_loss_fn = nn.L1Loss()
    with torch.no_grad():
        for batch in test_loader:
            features = batch["audio_features"].to(device)
            targets = batch["coefficients"].to(device)
            speaker_ids = batch["speaker_ids"].to(device)
            padding_mask = torch.zeros(features.size(0), features.size(1), dtype=torch.bool, device=device)
            for idx, seq_len in enumerate(batch["seq_lens"]):
                padding_mask[idx, seq_len:] = True
            preds = model(features, speaker_ids=speaker_ids, src_key_padding_mask=padding_mask)
            loss_mask = (~padding_mask).unsqueeze(-1).float()
            l1_loss = l1_loss_fn(preds * loss_mask, targets * loss_mask)
            total_l1 += l1_loss.item()
            if preds.size(1) >= 2:
                pred_vel = preds[:, 1:] - preds[:, :-1]
                target_vel = targets[:, 1:] - targets[:, :-1]
                vel_loss = nn.functional.l1_loss(pred_vel * loss_mask[:, 1:], target_vel * loss_mask[:, 1:])
                total_vel += vel_loss.item()
                pred_edge = preds[:, 1:] - preds[:, :-1]
                target_edge = targets[:, 1:] - targets[:, :-1]
                edge_loss = nn.functional.l1_loss(pred_edge * loss_mask[:, 1:], target_edge * loss_mask[:, 1:])
                total_edge += edge_loss.item()
    avg_l1 = total_l1 / len(test_loader)
    avg_vel = total_vel / len(test_loader)
    avg_edge = total_edge / len(test_loader)
    log.info(f"Evaluation results — L1 pos: {avg_l1:.6f}, vel: {avg_vel:.6f}, edge: {avg_edge:.6f}")
    log.info("Evaluation completed successfully.")

if __name__ == "__main__":
    evaluate()
