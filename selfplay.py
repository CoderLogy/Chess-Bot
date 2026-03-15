# selfplay.py
# Run AFTER retrain.py finishes
# Bot plays itself at depth 4
# Each position labeled by Stockfish depth 8
# Same label quality as sf_dataset.pt — won't undo retrain gains
# Continuously improves evaluator.pt
# Ctrl+C saves and exits cleanly

import chess
import chess.engine
import torch
import torch.nn as nn
import numpy as np
import signal
import sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

SF_PATH = "stockfish.exe"  
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

def board_to_tensor(board):
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    piece_idx = {
        (chess.PAWN,   True):  0,
        (chess.KNIGHT, True):  1,
        (chess.BISHOP, True):  2,
        (chess.ROOK,   True):  3,
        (chess.QUEEN,  True):  4,
        (chess.KING,   True):  5,
        (chess.PAWN,   False): 6,
        (chess.KNIGHT, False): 7,
        (chess.BISHOP, False): 8,
        (chess.ROOK,   False): 9,
        (chess.QUEEN,  False): 10,
        (chess.KING,   False): 11,
    }
    for sq, p in board.piece_map().items():
        r, c = divmod(sq, 8)
        tensor[piece_idx[(p.piece_type, p.color)]][r][c] = 1.0
    return tensor

def encode_board(board):
    t = board_to_tensor(board)
    if board.turn == chess.BLACK:
        t = np.flip(t, axis=2).copy()
        t = t[[6,7,8,9,10,11,0,1,2,3,4,5]]
    return t

# ── Setup ─────────────────────────────────────────────────────────────────────
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = Evaluator().to(device)

# Very low lr — fine-tuning already trained weights
# 1e-6 not 1e-5 — selfplay labels still noisier than retrain.py
# because positions come from bot vs itself not 2200+ ELO games
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr           = 1e-6,
    weight_decay = 1e-4
)
criterion = nn.MSELoss()

try:
    model.load_state_dict(torch.load(
        "evaluator.pt", map_location=device, weights_only=True
    ))
    print(f"Loaded evaluator.pt — running on {device}")
except FileNotFoundError:
    print("ERROR: evaluator.pt not found — run retrain.py first")
    sys.exit(1)

# Open Stockfish once at startup
try:
    sf = chess.engine.SimpleEngine.popen_uci(SF_PATH)
    sf.configure({"Threads": 4, "Hash": 256})
    print("Stockfish opened for position labeling at depth 8")
except Exception as e:
    print(f"ERROR: could not open Stockfish — {e}")
    sys.exit(1)

# ── Chart state ───────────────────────────────────────────────────────────────
chart_losses       = []
chart_avg_losses   = []
chart_game_nums    = []
chart_all_preds    = []
chart_all_labels   = []
chart_pos_per_game = []

def plot_selfplay_charts(game_num):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Plot 1 — Loss over games
    ax = axes[0, 0]
    ax.plot(chart_game_nums, chart_losses,
            alpha=0.3, color="steelblue", linewidth=0.8, label="Loss per game")
    ax.plot(chart_game_nums, chart_avg_losses,
            color="steelblue", linewidth=2, label="Rolling avg (10 games)")
    ax.set_title("Self-Play Loss Over Games")
    ax.set_xlabel("Game")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    ax.grid(True)

    # Plot 2 — Predicted vs Actual
    ax = axes[0, 1]
    if chart_all_preds:
        recent_preds  = chart_all_preds[-5000:]
        recent_labels = chart_all_labels[-5000:]
        ax.scatter(recent_labels, recent_preds,
                   alpha=0.05, s=1, color="steelblue")
        ax.plot([-1, 1], [-1, 1], "r--", linewidth=2, label="Perfect")
    ax.set_title("Predicted vs Actual (last 5000 positions)")
    ax.set_xlabel("SF Score depth 8 (actual)")
    ax.set_ylabel("Model Score (predicted)")
    ax.legend()
    ax.grid(True)

    # Plot 3 — Score distribution
    ax = axes[0, 2]
    if chart_all_preds:
        recent_preds  = chart_all_preds[-5000:]
        recent_labels = chart_all_labels[-5000:]
        ax.hist(recent_preds,  bins=60, alpha=0.7,
                label="Predictions", color="steelblue")
        ax.hist(recent_labels, bins=60, alpha=0.7,
                label="SF labels d8", color="orange")
    ax.set_title("Score Distribution (last 5000)")
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True)

    # Plot 4 — Error distribution
    ax = axes[1, 0]
    if chart_all_preds:
        recent_preds  = np.array(chart_all_preds[-5000:])
        recent_labels = np.array(chart_all_labels[-5000:])
        errors        = recent_preds - recent_labels
        ax.hist(errors, bins=60, color="purple", alpha=0.8)
        ax.axvline(0, color="red", linestyle="--", label="Zero error")
        ax.axvline(errors.mean(), color="orange", linestyle="--",
                   label=f"Mean: {errors.mean():.4f}")
        ax.set_title("Error Distribution (last 5000)")
        ax.set_xlabel("Error")
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(True)

    # Plot 5 — Positions per game
    ax = axes[1, 1]
    ax.plot(chart_game_nums, chart_pos_per_game,
            color="green", alpha=0.5, linewidth=0.8)
    if len(chart_pos_per_game) >= 10:
        rolling = np.convolve(
            chart_pos_per_game, np.ones(10)/10, mode="valid"
        )
        ax.plot(chart_game_nums[9:], rolling,
                color="green", linewidth=2, label="Rolling avg")
    ax.set_title("Positions per Game")
    ax.set_xlabel("Game")
    ax.set_ylabel("Positions")
    ax.legend()
    ax.grid(True)

    # Plot 6 — Summary
    ax = axes[1, 2]
    ax.axis("off")

    recent_loss = np.mean(chart_losses[-10:]) if chart_losses else 0
    best_loss   = min(chart_losses) if chart_losses else 0
    total_pos   = sum(chart_pos_per_game)

    if chart_all_preds:
        recent_preds  = np.array(chart_all_preds[-5000:])
        recent_labels = np.array(chart_all_labels[-5000:])
        errors        = recent_preds - recent_labels
        mae           = np.abs(errors).mean()
        rmse          = np.sqrt((errors**2).mean())
        bias          = errors.mean()
        pred_std      = recent_preds.std()
        label_std     = recent_labels.std()
        spread        = pred_std / label_std if label_std > 0 else 0
    else:
        mae = rmse = bias = spread = 0

    summary = (
        f"Games played:     {game_num}\n"
        f"Total positions:  {total_pos:,}\n\n"
        f"Current loss:     {recent_loss:.4f}\n"
        f"Best loss:        {best_loss:.4f}\n\n"
        f"MAE (recent):     {mae:.4f}\n"
        f"RMSE (recent):    {rmse:.4f}\n"
        f"Bias (recent):    {bias:.4f}\n"
        f"Spread ratio:     {spread:.2f}\n\n"
        f"SF label depth:   8\n"
        f"Search depth:     4\n"
        f"LR:               1e-6\n\n"
        f"Chart updates:    every 10 games\n"
        f"Saves:            every 100 games"
    )

    ax.text(
        0.1, 0.9, summary, transform=ax.transAxes,
        fontsize=11, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8)
    )
    ax.set_title("Self-Play Summary")

    plt.suptitle(
        f"Chess Bot — Self-Play Progress (Game {game_num})",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig("selfplay_progress.png", dpi=150)
    plt.close(fig)
    print(f"  Chart saved → selfplay_progress.png")

# ── Scoring ───────────────────────────────────────────────────────────────────
def nn_score(board):
    """Used for move selection — not for labels"""
    if board.is_checkmate():             return -100000
    if board.is_stalemate():             return 0
    if board.is_insufficient_material(): return 0
    t = encode_board(board)
    x = torch.tensor(t).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        return model(x).item() * 1000

def sf_score(board):
    """
    Stockfish labels at depth 8
    More accurate than depth 4 used in generate_sf_labels.py
    Returns score from current player's perspective
    Matches encode_board() convention exactly
    """
    if board.is_game_over():
        return None
    try:
        info  = sf.analyse(board, chess.engine.Limit(depth=8))
        score = info["score"].white().score(mate_score=10000)
        if score is None:
            return None
        # Normalize to -1 to +1
        score = float(max(-1.0, min(1.0, score / 1000.0)))
        # Flip to current player's perspective
        # matches encode_board() which flips Black positions
        if board.turn == chess.BLACK:
            score = -score
        return score
    except Exception:
        return None

# ── Search ────────────────────────────────────────────────────────────────────
def quiescence(board, alpha, beta, maximizing):
    """
    Extends search on captures only after depth runs out
    Prevents horizon effect — missing obvious recaptures
    Example without quiescence:
      Bot captures queen, search stops, says +900
      Next move opponent recaptures, real result -200
    With quiescence:
      Keeps searching captures until quiet position
      Sees the recapture coming, avoids blunder
    """
    stand_pat = nn_score(board)

    if maximizing:
        if stand_pat >= beta:
            return beta
        alpha = max(alpha, stand_pat)
        captures = sorted(
            [m for m in board.legal_moves if board.is_capture(m)],
            key=lambda m: board.is_capture(m),
            reverse=True
        )
        for move in captures:
            board.push(move)
            score = quiescence(board, alpha, beta, False)
            board.pop()
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return alpha
    else:
        if stand_pat <= alpha:
            return alpha
        beta = min(beta, stand_pat)
        captures = sorted(
            [m for m in board.legal_moves if board.is_capture(m)],
            key=lambda m: board.is_capture(m),
            reverse=True
        )
        for move in captures:
            board.push(move)
            score = quiescence(board, alpha, beta, True)
            board.pop()
            beta = min(beta, score)
            if beta <= alpha:
                break
        return beta

def alpha_beta(board, depth, alpha, beta, maximizing):
    if board.is_game_over():
        return nn_score(board)
    if depth == 0:
        # Call quiescence instead of nn_score directly
        # this is the key fix for tactical blunders
        return quiescence(board, alpha, beta, maximizing)

    moves = sorted(
        board.legal_moves,
        key=lambda m: (board.is_capture(m), board.gives_check(m)),
        reverse=True
    )
    if maximizing:
        value = -float('inf')
        for move in moves:
            board.push(move)
            value = max(value, alpha_beta(board, depth-1, alpha, beta, False))
            board.pop()
            alpha = max(alpha, value)
            if beta <= alpha: break
        return value
    else:
        value = float('inf')
        for move in moves:
            board.push(move)
            value = min(value, alpha_beta(board, depth-1, alpha, beta, True))
            board.pop()
            beta  = min(beta, value)
            if beta <= alpha: break
        return value

def pick_move(board, depth=4):
    """Search at depth 4 — stronger than depth 2"""
    best, best_val = None, -float('inf')
    for move in board.legal_moves:
        board.push(move)
        val = alpha_beta(board, depth-1, -float('inf'), float('inf'), False)
        board.pop()
        if val > best_val:
            best_val, best = val, move
    return best

# ── Self play ─────────────────────────────────────────────────────────────────
def play_one_game(depth=4, max_moves=150):
    """
    Bot plays itself at depth 4
    Each position labeled by SF depth 8
    NOT by game outcome — key fix from old version
    """
    board   = chess.Board()
    samples = []
    moves   = 0

    while not board.is_game_over() and moves < max_moves:
        move = pick_move(board, depth=depth)
        if move is None:
            break

        # Label BEFORE pushing move
        encoded = encode_board(board)
        label   = sf_score(board)   # SF depth 8 label

        if label is not None:
            samples.append((encoded, label))

        board.push(move)
        moves += 1

    return samples

def train_on_game(samples):
    if not samples:
        return 0.0, [], []

    xs = torch.tensor(
        np.array([s[0] for s in samples]),
        dtype=torch.float32
    ).to(device)
    ys = torch.tensor(
        [s[1] for s in samples],
        dtype=torch.float32
    ).to(device)

    # Collect predictions before training for charts
    model.eval()
    with torch.no_grad():
        preds = model(xs).cpu().numpy().tolist()

    # Train
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(xs), ys)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss.item(), preds, ys.cpu().numpy().tolist()

# ── Ctrl+C handler ────────────────────────────────────────────────────────────
game_num = 0

def save_and_exit(sig, frame):
    print("\nCtrl+C — saving evaluator.pt...")
    torch.save(model.state_dict(), "evaluator.pt")
    if chart_game_nums:
        plot_selfplay_charts(game_num)
    sf.quit()
    print("Saved. Goodbye.")
    sys.exit(0)

signal.signal(signal.SIGINT, save_and_exit)

# ── Main loop ─────────────────────────────────────────────────────────────────
print(f"Self-play running on {device}")
print(f"Search depth:    4")
print(f"SF label depth:  8")
print(f"LR:              1e-6")
print(f"Charts saved to: selfplay_progress.png every 10 games")
print("Press Ctrl+C to stop and save\n")

total_samples = 0

while True:
    game_num += 1
    samples              = play_one_game(depth=4)
    loss, preds, labels  = train_on_game(samples)
    total_samples       += len(samples)

    # Update chart data
    chart_losses.append(loss)
    chart_game_nums.append(game_num)
    chart_pos_per_game.append(len(samples))
    chart_all_preds.extend(preds)
    chart_all_labels.extend(labels)

    window = chart_losses[-10:]
    chart_avg_losses.append(np.mean(window))

    print(
        f"Game {game_num:>6} | "
        f"positions: {len(samples):>4} | "
        f"total: {total_samples:>8} | "
        f"loss: {loss:.4f} | "
        f"avg10: {chart_avg_losses[-1]:.4f}"
    )

    if game_num % 10 == 0:
        plot_selfplay_charts(game_num)

    if game_num % 100 == 0:
        torch.save(model.state_dict(), "evaluator.pt")
        print(f"--- Checkpoint saved at game {game_num} ---")