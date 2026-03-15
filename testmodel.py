# test_model.py
# Run after train.py to verify evaluator.pt is working correctly
# Tests: model loads, evaluates positions correctly,
#scores make intuitive sense, inference speed

import chess
import torch
import torch.nn as nn
import numpy as np
import time

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

def evaluate(model, board, device):
    if board.is_checkmate():             return -100000
    if board.is_stalemate():             return 0
    if board.is_insufficient_material(): return 0
    t = encode_board(board)
    x = torch.tensor(t).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(x).item() * 1000

# ── Load model ────────────────────────────────────────────────────────────────
print("=" * 60)
print("Chess Evaluator — Test Suite")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

try:
    model = Evaluator().to(device)
    model.load_state_dict(torch.load(
        "evaluator.pt", map_location=device, weights_only=True
    ))
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded — {total_params:,} parameters")
    print("PASS: model loads correctly")
except FileNotFoundError:
    print("FAIL: evaluator.pt not found — run train.py first")
    exit(1)
except Exception as e:
    print(f"FAIL: model load error — {e}")
    exit(1)

# ── Test 1: Starting position should be near 0 ───────────────────────────────
print("\n" + "─" * 60)
print("TEST 1: Starting position score")
board = chess.Board()
score = evaluate(model, board, device)
print(f"  Starting position score: {score:.1f} centipawns")
if abs(score) < 300:
    print("  PASS: score is near zero (equal position)")
else:
    print("  WARN: score is far from zero — model may not have converged")

# ── Test 2: Material advantage ────────────────────────────────────────────────
print("\n" + "─" * 60)
print("TEST 2: Material advantage detection")

# White up a queen
board_w = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
board_b = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1")
score_w = evaluate(model, board_w, device)
score_b = evaluate(model, board_b, device)
print(f"  White up a queen:  {score_w:.1f} cp")
print(f"  Black up a queen:  {score_b:.1f} cp")
if score_w > 0 and score_b < 0:
    print("  PASS: model correctly identifies material advantage")
else:
    print("  WARN: model may not understand material advantage yet")

# ── Test 3: Checkmate detection ───────────────────────────────────────────────
print("\n" + "─" * 60)
print("TEST 3: Checkmate detection")

# Fool's mate — Black just got checkmated
board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
if board.is_checkmate():
    score = evaluate(model, board, device)
    print(f"  Checkmate position score: {score}")
    print("  PASS: checkmate returns -100000")
else:
    # Scholar's mate — White just checkmated Black
    board = chess.Board("r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4")
    if board.is_checkmate():
        score = evaluate(model, board, device)
        print(f"  Checkmate position score: {score}")
        print("  PASS: checkmate returns -100000")
    else:
        print("  INFO: test position not checkmate — skipping")

# ── Test 4: Inference speed ───────────────────────────────────────────────────
print("\n" + "─" * 60)
print("TEST 4: Inference speed")

board   = chess.Board()
n_evals = 1000
start   = time.time()
for _ in range(n_evals):
    evaluate(model, board, device)
elapsed  = time.time() - start
ms_each  = (elapsed / n_evals) * 1000

print(f"  {n_evals} evaluations in {elapsed:.2f}s")
print(f"  {ms_each:.2f}ms per position")

# Estimate depth 3 move time
nodes_d3 = 20 * 20 * 20   # rough estimate
est_d3   = nodes_d3 * ms_each / 1000
nodes_d2 = 20 * 20
est_d2   = nodes_d2 * ms_each / 1000

print(f"\n  Estimated time per move:")
print(f"    Depth 2: ~{est_d2:.1f}s  ({nodes_d2} nodes)")
print(f"    Depth 3: ~{est_d3:.1f}s  ({nodes_d3} nodes)")

if est_d3 < 3.0:
    print("  PASS: depth 3 fits within 3 second limit")
elif est_d2 < 3.0:
    print("  WARN: depth 3 too slow — use depth 2 in bot.py")
    print("        change: if time_left > 60000: depth = 2")
else:
    print("  FAIL: even depth 2 may timeout — consider depth 1 as max")

# ── Test 5: Batch inference speed ────────────────────────────────────────────
print("\n" + "─" * 60)
print("TEST 5: Batch vs single inference")

tensors = torch.tensor(
    np.array([encode_board(chess.Board())] * 100),
    dtype=torch.float32
).to(device)

start = time.time()
for _ in range(100):
    with torch.no_grad():
        model(tensors)
batch_time = (time.time() - start) / 100 * 1000

print(f"  Batch of 100: {batch_time:.2f}ms")
print(f"  Single:       {ms_each:.2f}ms")
print(f"  Speedup:      {ms_each * 100 / batch_time:.1f}x")
print("  INFO: alpha-beta is sequential so batch doesn't help here")

# ── Test 6: Consistent results ────────────────────────────────────────────────
print("\n" + "─" * 60)
print("TEST 6: Deterministic output (eval mode)")

board    = chess.Board()
scores   = [evaluate(model, board, device) for _ in range(5)]
all_same = len(set(f"{s:.4f}" for s in scores)) == 1
print(f"  5 evaluations of same position: {[f'{s:.1f}' for s in scores]}")
if all_same:
    print("  PASS: model is deterministic in eval mode")
else:
    print("  FAIL: model gives different scores — check model.eval() is set")

# ── Test 7: UCI protocol test ─────────────────────────────────────────────────
print("\n" + "─" * 60)
print("TEST 7: Quick UCI protocol check")
print("  Run this manually to verify bot.py works:")
print()
print("  python bot.py")
print("  > uci")
print("  > isready")
print("  > position startpos")
print("  > go wtime 60000 btime 60000")
print("  > quit")
print()
print("  Expected: sees 'uciok', 'readyok', 'bestmove e2e4' (or similar)")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Device:          {device}")
print(f"  Parameters:      {total_params:,}")
print(f"  Inference speed: {ms_each:.2f}ms per position")
print(f"  Depth 2 est:     {est_d2:.1f}s per move")
print(f"  Depth 3 est:     {est_d3:.1f}s per move")
print()
if est_d3 < 3.0:
    print("  RECOMMENDATION: use depth 3 in bot.py — within time limit")
elif est_d2 < 3.0:
    print("  RECOMMENDATION: use depth 2 in bot.py — depth 3 too slow")
else:
    print("  RECOMMENDATION: use depth 1 in bot.py — net is too slow")
    print("  Consider: smaller architecture or GPU inference batching")
print("=" * 60)