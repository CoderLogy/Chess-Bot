# retrain.py
# Run after generate_sf_labels.py finishes
# Reads sf_dataset.pt
# Retrains on accurate Stockfish evaluations
# Starts from existing evaluator.pt weights
# Output: evaluator.pt (final best version)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import sys

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


class TensorChessDataset(Dataset):
    def __init__(self, pt_file: str):
        print(f"Loading dataset from {pt_file}...")
        data = torch.load(pt_file, weights_only=False)
        self.positions = data["positions"].float()
        self.scores    = data["scores"].float()
        print(f"Loaded {len(self.positions):,} positions")
        print(f"Tensor shape: {self.positions.shape}")

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        return self.positions[idx], self.scores[idx]


class CUDAPrefetcher:
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream()

    def __iter__(self):
        loader_iter = iter(self.loader)
        try:
            next_x, next_y = next(loader_iter)
        except StopIteration:
            return
        with torch.cuda.stream(self.stream):
            next_x = next_x.to(self.device, non_blocking=True)
            next_y = next_y.to(self.device, non_blocking=True)
        while True:
            torch.cuda.current_stream().wait_stream(self.stream)
            x, y = next_x, next_y
            try:
                next_x, next_y = next(loader_iter)
                with torch.cuda.stream(self.stream):
                    next_x = next_x.to(self.device, non_blocking=True)
                    next_y = next_y.to(self.device, non_blocking=True)
            except StopIteration:
                next_x = next_y = None
            yield x, y
            if next_x is None:
                break

    def __len__(self):
        return len(self.loader)


# Must match train.py exactly
class Evaluator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),

            nn.Linear(12 * 8 * 8, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 64),
            nn.ReLU(),

            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)



class CombinedLoss(nn.Module):
    """
    Standard MSE + extra penalty for extreme positions
    predicted too conservatively.

    extreme_weight controls how hard — 0.5 = moderate
    increase to 1.0 or 2.0 if predictions still too narrow
    """
    def __init__(self, extreme_threshold=0.5, extreme_weight=0.5):
        super().__init__()
        self.extreme_threshold = extreme_threshold
        self.extreme_weight    = extreme_weight

    def forward(self, pred, target):
        # Base MSE loss — same as before
        base_loss = F.mse_loss(pred, target)

        # Extra penalty on extreme positions
        # extreme_mask = positions where SF says clearly winning/losing
        extreme_mask = target.abs() > self.extreme_threshold

        if extreme_mask.sum() > 0:
            extreme_loss = F.mse_loss(
                pred[extreme_mask],
                target[extreme_mask]
            )
            return base_loss + self.extreme_weight * extreme_loss

        return base_loss


def collect_predictions(model, val_prefetcher):
    model.eval()
    all_preds  = []
    all_labels = []
    with torch.no_grad():
        for x, y in val_prefetcher:
            with torch.amp.autocast("cuda"):
                pred = model(x)
            all_preds.extend(pred.cpu().float().numpy())
            all_labels.extend(y.cpu().numpy())
    return np.array(all_preds), np.array(all_labels)


def plot_results(train_losses, val_losses, model, val_loader, device):
    val_prefetcher        = CUDAPrefetcher(val_loader, device)
    all_preds, all_labels = collect_predictions(model, val_prefetcher)
    errors                = all_preds - all_labels
    best_epoch            = int(np.argmin(val_losses))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Plot 1 — Loss curves
    ax = axes[0, 0]
    ax.plot(train_losses, label="Train Loss", marker="o")
    ax.plot(val_losses,   label="Val Loss",   marker="o")
    ax.axvline(best_epoch, color="red", linestyle="--",
               label=f"Best epoch ({best_epoch+1})")
    ax.set_title("Loss Curves (SF Labels)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    ax.grid(True)

    # Plot 2 — Predicted vs Actual
    ax = axes[0, 1]
    ax.scatter(all_labels, all_preds, alpha=0.05, s=1, color="steelblue")
    ax.plot([-1, 1], [-1, 1], "r--", linewidth=2, label="Perfect prediction")
    ax.set_title("Predicted vs Actual (should be diagonal scatter)")
    ax.set_xlabel("Actual Score")
    ax.set_ylabel("Predicted Score")
    ax.legend()
    ax.grid(True)

    # Plot 3 — Prediction distribution
    # After conservative fix — blue should spread wider matching orange
    ax = axes[0, 2]
    ax.hist(all_preds,  bins=60, alpha=0.7, label="Predictions", color="steelblue")
    ax.hist(all_labels, bins=60, alpha=0.7, label="Actual",      color="orange")
    ax.set_title("Prediction Distribution (blue should match orange width)")
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True)

    # Plot 4 — Error distribution
    ax = axes[1, 0]
    ax.hist(errors, bins=60, color="purple", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", label="Zero error")
    ax.axvline(errors.mean(), color="orange", linestyle="--",
               label=f"Mean error: {errors.mean():.4f}")
    ax.set_title("Error Distribution (Pred - Actual)")
    ax.set_xlabel("Error")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True)

    # Plot 5 — Outcome accuracy
    ax = axes[1, 1]

    def classify(scores):
        result = np.full(len(scores), "draw", dtype=object)
        result[scores >  0.5] = "white"
        result[scores < -0.5] = "black"
        return result

    true_outcomes = classify(all_labels)
    pred_outcomes = classify(all_preds)
    correct       = true_outcomes == pred_outcomes
    categories    = ["white", "draw", "black"]
    accuracies, counts = [], []
    for cat in categories:
        mask = true_outcomes == cat
        accuracies.append(correct[mask].mean() * 100 if mask.sum() > 0 else 0)
        counts.append(mask.sum())

    bars = ax.bar(
        categories, accuracies,
        color=["white", "gray", "black"],
        edgecolor="black", linewidth=1.2
    )
    for bar, acc, cnt in zip(bars, accuracies, counts):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 1,
            f"{acc:.1f}%\n(n={cnt:,})",
            ha="center", va="bottom", fontsize=9
        )
    ax.set_title("Outcome Classification Accuracy")
    ax.set_xlabel("True Outcome")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 110)
    ax.grid(True, axis="y")

    # Plot 6 — Summary stats
    ax = axes[1, 2]
    ax.axis("off")
    mae         = np.abs(errors).mean()
    rmse        = np.sqrt((errors**2).mean())
    overall_acc = correct.mean() * 100

    # Check if conservative prediction is fixed
    pred_std   = all_preds.std()
    label_std  = all_labels.std()
    spread_ratio = pred_std / label_std if label_std > 0 else 0
    spread_status = (
        "FIXED" if spread_ratio > 0.7
        else "IMPROVING" if spread_ratio > 0.4
        else "STILL CONSERVATIVE"
    )

    summary = (
        f"Dataset:          sf_dataset.pt\n"
        f"Dataset size:     {len(all_labels) * 10:,} (val={len(all_labels):,})\n"
        f"Best epoch:       {best_epoch + 1}\n"
        f"Best val loss:    {min(val_losses):.4f}\n\n"
        f"MAE:              {mae:.4f}\n"
        f"RMSE:             {rmse:.4f}\n"
        f"Bias:             {errors.mean():.4f}\n\n"
        f"Pred std:         {pred_std:.4f}\n"
        f"Label std:        {label_std:.4f}\n"
        f"Spread ratio:     {spread_ratio:.2f} ({spread_status})\n\n"
        f"Outcome accuracy: {overall_acc:.1f}%\n"
        f"  White:          {accuracies[0]:.1f}%\n"
        f"  Draw:           {accuracies[1]:.1f}%\n"
        f"  Black:          {accuracies[2]:.1f}%\n\n"
        f"Target val loss:  < 0.25\n"
        f"Status:           {'GOOD' if min(val_losses) < 0.25 else 'NEEDS MORE TRAINING'}"
    )

    ax.text(
        0.1, 0.9, summary, transform=ax.transAxes,
        fontsize=11, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8)
    )
    ax.set_title("Summary Statistics")

    plt.suptitle(
        "Chess Evaluator — Retrain Results (Stockfish Labels)",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig("training_results_retrain.png", dpi=150)
    plt.show()
    print("Saved training_results_retrain.png")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}  VRAM: {props.total_memory/1e9:.1f} GB")

    dataset  = TensorChessDataset("sf_dataset.pt")
    train_n  = int(0.9 * len(dataset))
    val_n    = len(dataset) - train_n
    train_ds, val_ds = random_split(dataset, [train_n, val_n])

    LOADER_KWARGS = dict(
        batch_size         = 16384,
        num_workers        = 0,
        pin_memory         = False,
        persistent_workers = False,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **LOADER_KWARGS)
    val_loader   = DataLoader(val_ds,   shuffle=False, **LOADER_KWARGS)

    model = Evaluator().to(device)

    if sys.platform != "win32":
        try:
            model = torch.compile(model)
            print("torch.compile: enabled")
        except Exception:
            print("torch.compile: not available, skipping")
    else:
        print("torch.compile: skipped (Windows)")

    try:
        state = torch.load(
            "evaluator.pt", map_location=device, weights_only=True
        )
        (model._orig_mod if hasattr(model, "_orig_mod") else model
         ).load_state_dict(state)
        print("Starting from existing evaluator.pt weights")
        lr = 5e-5
    except FileNotFoundError:
        print("No existing weights — starting fresh")
        lr = 1e-3

    epochs    = 30
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max   = epochs * len(train_loader),  # total steps
    eta_min = 1e-7
    )

    criterion = CombinedLoss(extreme_threshold=0.5, extreme_weight=0.5)

    scaler = torch.amp.GradScaler("cuda")

    best_val_loss = float("inf")
    train_losses  = []
    val_losses    = []

    for epoch in range(epochs):
        model.train()
        train_loss       = 0.0
        train_prefetcher = CUDAPrefetcher(train_loader, device)
        pbar = tqdm(
            train_prefetcher,
            total = len(train_loader),
            desc  = f"Epoch {epoch+1}/{epochs}",
            unit  = "batch"
        )
        for x, y in pbar:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                pred = model(x)
                loss = criterion(pred, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss += loss.item()
            pbar.set_postfix(
                loss = f"{loss.item():.4f}",
                lr   = f"{scheduler.get_last_lr()[0]:.2e}"
            )

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss       = 0.0
        val_prefetcher = CUDAPrefetcher(val_loader, device)
        with torch.no_grad():
            for x, y in val_prefetcher:
                with torch.amp.autocast("cuda"):
                    pred = model(x)
                    # Use plain MSE for val loss — comparable across runs
                    loss = F.mse_loss(pred, y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"\nEpoch {epoch+1:>2} | train={train_loss:.4f} | val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state = (
                model._orig_mod if hasattr(model, "_orig_mod")
                else model
            ).state_dict()
            torch.save(state, "evaluator.pt")
            print("  -> saved new best evaluator.pt")

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print("evaluator.pt is ready for bot.py")

    # Load best model before plotting
    state = torch.load("evaluator.pt", weights_only=True)
    (model._orig_mod if hasattr(model, "_orig_mod") else model
     ).load_state_dict(state)
    plot_results(train_losses, val_losses, model, val_loader, device)


if __name__ == "__main__":
    main()