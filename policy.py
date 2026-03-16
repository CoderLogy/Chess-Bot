# train_policy.py
# Train PolicyNetwork to predict good moves
# Uses actual moves played in grandmaster games as targets
# Run after train_eval.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import chess
import chess.pgn
import matplotlib.pyplot as plt

from models import PolicyNetwork, encode_board, encode_move

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


class PolicyDataset(Dataset):
    """
    For each position in a grandmaster game:
      Input:  board position (encoded tensor)
      Target: the move that was actually played (as index)
    
    The policy learns to predict human moves
    This gives us strong move ordering for free
    Grandmasters play good moves — policy learns what good moves look like
    """
    def __init__(self, pgn_path, max_games=50000):
        self.positions = []
        self.moves     = []

        result_map = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}
        games_read = 0

        print(f"Building policy dataset from {pgn_path}...")
        with open(pgn_path) as f:
            while games_read < max_games:
                game = chess.pgn.read_game(f)
                if game is None:
                    break

                result = result_map.get(game.headers.get("Result", ""))
                if result is None:
                    continue

                board = game.board()
                for i, move in enumerate(game.mainline_moves()):
                    if i < 5:
                        board.push(move)
                        continue  # skip opening moves

                    # Encode position
                    t = encode_board(board)
                    # Encode the move that was played
                    m_idx = encode_move(move, board)

                    self.positions.append(t)
                    self.moves.append(m_idx)

                    board.push(move)

                games_read += 1
                if games_read % 5000 == 0:
                    print(f"  {games_read} games, "
                          f"{len(self.positions):,} positions")

        self.positions = torch.tensor(
            np.array(self.positions), dtype=torch.float32
        )
        self.moves = torch.tensor(self.moves, dtype=torch.long)
        print(f"Done. {games_read} games, "
              f"{len(self.positions):,} positions")

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        return self.positions[idx], self.moves[idx]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset     = PolicyDataset("filtered.pgn", max_games=50000)
    train_n     = int(0.9 * len(dataset))
    val_n       = len(dataset) - train_n
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_n, val_n]
    )

    train_loader = DataLoader(train_ds, batch_size=4096,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=4096,
                              shuffle=False, num_workers=0)

    model   = PolicyNetwork().to(device)
    params  = sum(p.numel() for p in model.parameters())
    print(f"PolicyNetwork parameters: {params:,}")

    # Cross entropy — predict which move was played
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-4
    )
    scaler    = torch.amp.GradScaler(device.type)

    epochs      = 20
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
    acc_hist   = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader,
                    desc=f"Policy Epoch {epoch+1}/{epochs}", unit="batch")
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type):
                logits = model(x)
                loss   = criterion(logits, y)
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

        # Validation — measure top-1 accuracy (did it predict exact move?)
        # and top-5 accuracy (was correct move in top 5?)
        model.eval()
        val_loss = 0.0
        top1 = top5 = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.amp.autocast(device.type):
                    logits = model(x)
                    loss   = criterion(logits, y)
                val_loss += loss.item()

                # Accuracy
                _, top5_pred = logits.topk(5, dim=1)
                top1 += (top5_pred[:, 0] == y).sum().item()
                top5 += (top5_pred == y.unsqueeze(1)).any(dim=1).sum().item()
                total += y.size(0)

        val_loss /= len(val_loader)
        val_hist.append(val_loss)
        acc = top1 / total * 100
        acc5 = top5 / total * 100
        acc_hist.append(acc)

        print(f"\nEpoch {epoch+1:>2} | "
              f"train={train_loss:.4f} | val={val_loss:.4f} | "
              f"top1={acc:.1f}% | top5={acc5:.1f}%")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "policy_network.pt")
            print("  -> saved policy_network.pt")

    print(f"\nDone. Best val loss: {best_val:.4f}")
    print(f"Top-1 accuracy: {acc_hist[-1]:.1f}%")
    print("(30-50% top-1 is good for chess policy)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    ax1.plot(train_hist, label="Train")
    ax1.plot(val_hist,   label="Val")
    ax1.set_title("Policy Network — Loss")
    ax1.legend()
    ax1.grid(True)
    ax2.plot(acc_hist, color="green", label="Top-1 Accuracy %")
    ax2.set_title("Policy Network — Move Prediction Accuracy")
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig("policy_training.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()