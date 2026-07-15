import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.training.dataset import VocasetDataset, collate_fn
from src.training.model import SpeechToCoefficientsModel

def compute_velocity_loss(pred, target):
    """L1 loss on the temporal differences (velocity) to promote smooth animation."""
    if pred.size(1) < 2:
        return torch.tensor(0.0, device=pred.device)
    pred_vel = pred[:, 1:] - pred[:, :-1]
    target_vel = target[:, 1:] - target[:, :-1]
    return nn.functional.l1_loss(pred_vel, target_vel)

def train(epochs=50, batch_size=8, lr=1e-4, hidden_dim=256):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Training on device: {device}")

    # 1. Setup Datasets & DataLoaders
    print("Loading datasets...")
    try:
        train_dataset = VocasetDataset(split="train")
        val_dataset = VocasetDataset(split="val")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run python src/training/reproject_vocaset.py first.")
        sys.exit(1)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # 2. Setup Model, Optimizer, Loss
    model = SpeechToCoefficientsModel(hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    l1_loss_fn = nn.L1Loss()

    checkpoint_dir = "voca/model/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_val_loss = float("inf")

    # 3. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_l1 = 0.0
        train_vel = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} (Train)"):
            features = batch["audio_features"].to(device)
            targets = batch["coefficients"].to(device)
            
            # Mask for padded items (True means padded / ignore)
            padding_mask = torch.zeros(features.size(0), features.size(1), dtype=torch.bool, device=device)
            for idx, seq_len in enumerate(batch["seq_lens"]):
                padding_mask[idx, seq_len:] = True

            optimizer.zero_grad()
            
            # Forward pass
            preds = model(features, src_key_padding_mask=padding_mask)

            # Mask out loss on padded elements
            loss_mask = (~padding_mask).unsqueeze(-1).float()
            
            # L1 position loss
            l1_loss = l1_loss_fn(preds * loss_mask, targets * loss_mask)
            
            # L1 velocity loss
            vel_loss = compute_velocity_loss(preds * loss_mask, targets * loss_mask)
            
            # Total loss (balanced weights)
            total_loss = l1_loss + 0.5 * vel_loss

            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()
            train_l1 += l1_loss.item()
            train_vel += vel_loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        val_l1 = 0.0
        val_vel = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch["audio_features"].to(device)
                targets = batch["coefficients"].to(device)
                
                padding_mask = torch.zeros(features.size(0), features.size(1), dtype=torch.bool, device=device)
                for idx, seq_len in enumerate(batch["seq_lens"]):
                    padding_mask[idx, seq_len:] = True

                preds = model(features, src_key_padding_mask=padding_mask)
                loss_mask = (~padding_mask).unsqueeze(-1).float()
                
                l1_loss = l1_loss_fn(preds * loss_mask, targets * loss_mask)
                vel_loss = compute_velocity_loss(preds * loss_mask, targets * loss_mask)
                total_loss = l1_loss + 0.5 * vel_loss

                val_loss += total_loss.item()
                val_l1 += l1_loss.item()
                val_vel += vel_loss.item()

        # Average metrics
        train_loss /= len(train_loader)
        train_l1 /= len(train_loader)
        train_vel /= len(train_loader)
        val_loss /= len(val_loader)
        val_l1 /= len(val_loader)
        val_vel /= len(val_loader)

        print(f"Epoch {epoch}: Train Loss: {train_loss:.6f} (L1: {train_l1:.6f}, Vel: {train_vel:.6f})")
        print(f"          Val Loss:   {val_loss:.6f} (L1: {val_l1:.6f}, Vel: {val_vel:.6f})")

        # Save checkpoint if val loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, checkpoint_path)
            print(f"  --> Saved new best checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train SpeechToCoefficientsModel on reprojected VOCASET.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch, lr=args.lr)
