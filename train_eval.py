# train_eval.py
# Train LinearEvaluator using Texel tuning
# Run this first

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import sys

from models import LinearEvaluator

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


class TensorChessDataset(Dataset):
    def __init__(self, pt_file):
        print(f"Loading {pt_file}...")
        data           = torch.load(pt_file, weights_only=False)
        self.positions = data["positions"].float()
        self.scores    = data["scores"].float()
        s = self.scores.numpy()
        decisive = (np.abs(s) > 0.1).sum()
        print(f"Loaded {len(self.positions):,} positions")
        print(f"Decisive: {decisive:,} ({decisive/len(s)*100:.1f}%)")

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        return self.positions[idx], self.scores[idx]


class TexelLoss(nn.Module):
    """
    Texel tuning — the standard way to train chess evaluators
    
    Instead of MSE on raw centipawn values:
      MSE( model_score, sf_score )
    
    We use cross-entropy on win probabilities:
      sigmoid(score/K) → win probability
      cross_entropy( model_prob, sf_prob )
    
    Why better:
      Directly optimizes for predicting game outcomes
      Handles outliers better (a +5.0 position
      and a +50.0 position are both "winning" —
      MSE penalizes the difference, Texel doesn't)
      How Stockfish and most strong engines train
    """
    def __init__(self, K=400.0):
        super().__init__()
        self.K = K

    def forward(self, pred, target):
        pred_prob   = torch.sigmoid(pred   * self.K / 1000.0)
        target_prob = torch.sigmoid(target * self.K / 1000.0)
        return F.binary_cross_entropy(pred_prob, target_prob)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset  = TensorChessDataset("sf_dataset.pt")
    train_n  = int(0.9 * len(dataset))
    val_n    = len(dataset) - train_n
    train_ds, val_ds = random_split(dataset, [train_n, val_n])

    train_loader = DataLoader(train_ds, batch_size=16384,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=16384,
                              shuffle=False, num_workers=0)

    model   = LinearEvaluator().to(device)
    params  = sum(p.numel() for p in model.parameters())
    print(f"LinearEvaluator parameters: {params:,}")

    criterion = TexelLoss(K=400)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-4
    )
    scaler    = torch.amp.GradScaler(device.type)

    epochs      = 30
    total_steps = epochs * len(train_loader)
    scheduler   = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3, total_steps=total_steps,
        pct_start=0.05, anneal_strategy="cos",
        div_factor=3.0, final_div_factor=1000.0,
    )

    best_val   = float("inf")
    train_hist = []
    val_hist   = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader,
                    desc=f"Eval Epoch {epoch+1}/{epochs}", unit="batch")
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= len(train_loader)
        train_hist.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.amp.autocast(device.type):
                    val_loss += criterion(model(x), y).item()
        val_loss /= len(val_loader)
        val_hist.append(val_loss)

        print(f"\nEpoch {epoch+1:>2} | "
              f"train={train_loss:.4f} | val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "linear_evaluator.pt")
            print("  -> saved linear_evaluator.pt")

    print(f"\nDone. Best val loss: {best_val:.4f}")

    plt.figure(figsize=(10, 4))
    plt.plot(train_hist, label="Train")
    plt.plot(val_hist,   label="Val")
    plt.title("LinearEvaluator — Texel Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("eval_training.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()