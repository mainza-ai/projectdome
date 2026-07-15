import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.training.dataset import VocasetDataset, collate_fn
from src.training.model import SpeechToCoefficientsModel

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Evaluating on device: {device}")

    checkpoint_path = "voca/model/checkpoints/best_model.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}. Train the model first.")
        sys.exit(1)

    # 1. Load Dataset
    print("Loading test split...")
    test_dataset = VocasetDataset(split="test")
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    # 2. Load Model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SpeechToCoefficientsModel().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded model checkpoint from epoch {checkpoint['epoch']} (val loss: {checkpoint['val_loss']:.6f})")

    # 3. Evaluate L1 loss
    total_l1 = 0.0
    l1_loss_fn = nn.L1Loss()

    with torch.no_grad():
        for batch in test_loader:
            features = batch["audio_features"].to(device)
            targets = batch["coefficients"].to(device)
            
            padding_mask = torch.zeros(features.size(0), features.size(1), dtype=torch.bool, device=device)
            for idx, seq_len in enumerate(batch["seq_lens"]):
                padding_mask[idx, seq_len:] = True

            preds = model(features, src_key_padding_mask=padding_mask)
            
            loss_mask = (~padding_mask).unsqueeze(-1).float()
            l1_loss = l1_loss_fn(preds * loss_mask, targets * loss_mask)
            total_l1 += l1_loss.item()

    avg_l1 = total_l1 / len(test_loader)
    print("\n=== Evaluation Results ===")
    print(f"Test L1 Position Error (Coefficients): {avg_l1:.6f}")
    print("Evaluation completed successfully.")

if __name__ == "__main__":
    evaluate()
