import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")   # TF32 → ~2× throughput vs FP32



class TensorChessDataset(Dataset):
    def __init__(self, pt_file: str):
        print(f"Loading dataset from {pt_file} ...")
        data = torch.load(pt_file, weights_only=False)

        self.positions = data["positions"].float()
        self.scores    = data["scores"].float()
        self.fens      = data.get("fens",  None)
        self.turns     = data.get("turns", None)

        ram_gb = (self.positions.element_size() * self.positions.nelement() +
                  self.scores.element_size()    * self.scores.nelement()) / 1e9
        print(f"Dataset loaded  -  {len(self.positions):,} positions  "
              f"shape={tuple(self.positions.shape[1:])}  "
              f"RAM: {ram_gb:.2f} GB")

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        return self.positions[idx], self.scores[idx]


# ── CUDA prefetch wrapper ─────────────────────────────────────────────────────
class CUDAPrefetcher:
    """
    Overlaps the next CPU->GPU transfer with the current GPU compute step.
    The GPU never stalls waiting for data -- PCIe latency is completely hidden.
    """

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


# ── Model ─────────────────────────────────────────────────────────────────────
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



def collect_predictions(model, val_prefetcher):
    model.eval()
    all_preds, all_labels = [], []
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

    # Loss curves
    ax = axes[0, 0]
    ax.plot(train_losses, label="Train Loss", marker="o")
    ax.plot(val_losses,   label="Val Loss",   marker="o")
    ax.axvline(best_epoch, color="red", linestyle="--",
               label=f"Best epoch ({best_epoch+1})")
    ax.set_title("Loss Curves"); ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.legend(); ax.grid(True)

    # Predicted vs Actual
    ax = axes[0, 1]
    ax.scatter(all_labels, all_preds, alpha=0.05, s=1, color="steelblue")
    ax.plot([-1, 1], [-1, 1], "r--", linewidth=2, label="Perfect prediction")
    ax.set_title("Predicted vs Actual")
    ax.set_xlabel("Actual Score"); ax.set_ylabel("Predicted Score")
    ax.legend(); ax.grid(True)

    # Prediction distribution
    ax = axes[0, 2]
    ax.hist(all_preds,  bins=60, alpha=0.7, label="Predictions", color="steelblue")
    ax.hist(all_labels, bins=60, alpha=0.7, label="Actual",      color="orange")
    ax.set_title("Prediction Distribution")
    ax.set_xlabel("Score"); ax.set_ylabel("Count")
    ax.legend(); ax.grid(True)

    # Error distribution
    ax = axes[1, 0]
    ax.hist(errors, bins=60, color="purple", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", label="Zero error")
    ax.axvline(errors.mean(), color="orange", linestyle="--",
               label=f"Mean error: {errors.mean():.4f}")
    ax.set_title("Error Distribution (Pred - Actual)")
    ax.set_xlabel("Error"); ax.set_ylabel("Count")
    ax.legend(); ax.grid(True)

    # Win/Draw/Loss accuracy
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

    bars = ax.bar(categories, accuracies,
                  color=["white", "gray", "black"],
                  edgecolor="black", linewidth=1.2)
    for bar, acc, cnt in zip(bars, accuracies, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{acc:.1f}%\n(n={cnt:,})",
                ha="center", va="bottom", fontsize=9)
    ax.set_title("Outcome Classification Accuracy")
    ax.set_xlabel("True Outcome"); ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 110); ax.grid(True, axis="y")

    # Summary stats
    ax = axes[1, 2]; ax.axis("off")
    mae         = np.abs(errors).mean()
    rmse        = np.sqrt((errors**2).mean())
    overall_acc = correct.mean() * 100
    summary = (
        f"Dataset size:     {len(all_labels):,}\n"
        f"Best epoch:       {best_epoch + 1}\n"
        f"Best val loss:    {min(val_losses):.4f}\n\n"
        f"MAE:              {mae:.4f}\n"
        f"RMSE:             {rmse:.4f}\n"
        f"Bias:             {errors.mean():.4f}\n\n"
        f"Outcome accuracy: {overall_acc:.1f}%\n"
        f"  White:          {accuracies[0]:.1f}%\n"
        f"  Draw:           {accuracies[1]:.1f}%\n"
        f"  Black:          {accuracies[2]:.1f}%"
    )
    ax.text(0.1, 0.9, summary, transform=ax.transAxes,
            fontsize=11, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax.set_title("Summary Statistics")

    plt.suptitle("Chess Evaluator - Training Results",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("training_results.png", dpi=150)
    plt.show()
    print("Saved training_results.png")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}  VRAM: {props.total_memory / 1e9:.1f} GB")

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset    = TensorChessDataset("dataset.pt")
    train_size = int(0.9 * len(dataset))
    val_size   = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    
    # num_workers=0  -> Windows-safe (no shared-memory fork / error 1450)
    # pin_memory=False -> large dataset OOMs the CUDA pinned allocator
    # CUDAPrefetcher handles async transfers instead
    LOADER_KWARGS = dict(
        batch_size         = 16384,
        num_workers        = 0,
        pin_memory         = False,
        persistent_workers = False,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **LOADER_KWARGS)
    val_loader   = DataLoader(val_ds,   shuffle=False, **LOADER_KWARGS)

   
    model = Evaluator().to(device)
    # torch.compile requires Triton which is Linux-only; skip on Windows
    import sys
    if sys.platform != "win32":
        try:
            model = torch.compile(model)
            print("torch.compile: enabled")
        except Exception:
            print("torch.compile: not available, skipping")
    else:
        print("torch.compile: skipped (Windows — Triton not supported)")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs    = 20
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr          = 1e-3,
        epochs          = epochs,
        steps_per_epoch = len(train_loader),
        pct_start       = 0.3,
        anneal_strategy = "cos",
    )
    criterion = nn.MSELoss()
    scaler    = torch.amp.GradScaler("cuda")

    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss       = 0.0
        train_prefetcher = CUDAPrefetcher(train_loader, device)
        pbar = tqdm(train_prefetcher, total=len(train_loader),
                    desc=f"Epoch {epoch+1}/{epochs}", unit="batch")

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
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             lr=f"{scheduler.get_last_lr()[0]:.2e}")

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss       = 0.0
        val_prefetcher = CUDAPrefetcher(val_loader, device)
        with torch.no_grad():
            for x, y in val_prefetcher:
                with torch.amp.autocast("cuda"):
                    pred = model(x)
                    loss = criterion(pred, y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"\nEpoch {epoch+1:>2} | train={train_loss:.4f} | val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state = (model._orig_mod if hasattr(model, "_orig_mod")
                     else model).state_dict()
            torch.save(state, "evaluator.pt")
            print("  -> saved new best model")

    print(f"\nTraining complete - best val loss: {best_val_loss:.4f}")
    print("Model saved as evaluator.pt")

    state = torch.load("evaluator.pt", weights_only=True)
    (model._orig_mod if hasattr(model, "_orig_mod") else model).load_state_dict(state)
    plot_results(train_losses, val_losses, model, val_loader, device)


if __name__ == "__main__":
    main()