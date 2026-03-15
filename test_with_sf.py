# Tests how often your net agrees with Stockfish
# on which move is best

import chess
import chess.engine
import torch
import torch.nn as nn
import numpy as np

SF_PATH = "stockfish.exe"

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
        (chess.PAWN,   True):  0, (chess.PAWN,   False): 6,
        (chess.KNIGHT, True):  1, (chess.KNIGHT, False): 7,
        (chess.BISHOP, True):  2, (chess.BISHOP, False): 8,
        (chess.ROOK,   True):  3, (chess.ROOK,   False): 9,
        (chess.QUEEN,  True):  4, (chess.QUEEN,  False): 10,
        (chess.KING,   True):  5, (chess.KING,   False): 11,
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = Evaluator().to(device)
model.load_state_dict(torch.load("evaluator.pt", map_location=device, weights_only=True))
model.eval()

def nn_score(board):
    t = encode_board(board)
    x = torch.tensor(t).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(x).item() * 1000

# Test positions — mix of opening, middlegame, endgame
TEST_POSITIONS = [
    ("Starting position",
     "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("After e4",
     "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),
    ("Sicilian",
     "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2"),
    ("Italian game",
     "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
    ("Middlegame",
     "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQR1K1 w - - 0 9"),
    ("Endgame — White winning",
     "8/5pk1/6p1/7p/7P/6P1/5PK1/8 w - - 0 1"),
    ("Rook endgame",
     "8/8/4k3/8/8/4K3/4R3/8 w - - 0 1"),
    ("Queen vs pawn",
     "8/8/8/8/8/k7/p7/K1Q5 w - - 0 1"),
]

print("=" * 70)
print("SF vs Your Net — Move Agreement Test")
print("=" * 70)

sf = chess.engine.SimpleEngine.popen_uci(SF_PATH)
sf.configure({"Threads": 2})

total_positions = 0
top1_agrees     = 0
top3_agrees     = 0

for name, fen in TEST_POSITIONS:
    board = chess.Board(fen)
    legal = list(board.legal_moves)
    if not legal:
        continue

    print(f"\n{'─'*70}")
    print(f"Position: {name}")
    print(f"FEN: {fen}")
    print(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}")

    # Get SF top 5 moves
    n_pv = min(5, len(legal))
    sf_result = sf.analyse(
        board,
        chess.engine.Limit(depth=15),  # deeper for accuracy
        multipv=n_pv
    )
    sf_moves  = [info["pv"][0] for info in sf_result if "pv" in info]
    sf_scores = [info["score"].white().score(mate_score=10000)
                 for info in sf_result if "score" in info]

    # Get your net scores for all legal moves
    net_scores = {}
    for move in legal:
        board.push(move)
        net_scores[move] = nn_score(board)
        board.pop()

    # Sort by your net score
    # For White to move: higher = better
    # For Black to move: lower = better (net scores from current player view)
    net_sorted = sorted(
        net_scores.items(),
        key=lambda x: x[1],
        reverse=True   # always descending — net always from current player view
    )

    net_top1 = net_sorted[0][0]
    net_top3 = [m for m, _ in net_sorted[:3]]

    print(f"\nSF top moves:   {[m.uci() for m in sf_moves[:3]]}")
    print(f"Net top moves:  {[m.uci() for m, _ in net_sorted[:3]]}")

    # Agreement
    agrees_top1 = net_top1 == sf_moves[0] if sf_moves else False
    agrees_top3 = net_top1 in sf_moves[:3] if sf_moves else False

    if agrees_top1:
        top1_agrees += 1
        print(f"Agreement: ✅ EXACT MATCH (both pick {net_top1.uci()})")
    elif agrees_top3:
        top3_agrees += 1
        print(f"Agreement: ⚠️  NET top1 in SF top3")
        print(f"  SF best: {sf_moves[0].uci()}  Net best: {net_top1.uci()}")
    else:
        print(f"Agreement: ❌ DISAGREE")
        print(f"  SF best: {sf_moves[0].uci()}  Net best: {net_top1.uci()}")

    # Show score comparison
    print(f"\nScore comparison (top 3):")
    print(f"  {'Move':<8} {'SF score':>10} {'Net score':>12} {'Match':>6}")
    for j, (move, net_s) in enumerate(net_sorted[:5]):
        sf_s = sf_scores[j] if j < len(sf_scores) and \
               sf_moves and move == sf_moves[j] else "—"
        match = "✅" if sf_moves and move == sf_moves[0] else ""
        print(f"  {move.uci():<8} {str(sf_s):>10} {net_s:>12.1f} {match:>6}")

    total_positions += 1

sf.quit()

print(f"\n{'='*70}")
print(f"RESULTS SUMMARY")
print(f"{'='*70}")
print(f"Positions tested:     {total_positions}")
print(f"Exact agreement:      {top1_agrees}/{total_positions} "
      f"({top1_agrees/total_positions*100:.1f}%)")
print(f"Top-3 agreement:      {top1_agrees+top3_agrees}/{total_positions} "
      f"({(top1_agrees+top3_agrees)/total_positions*100:.1f}%)")
print()
if top1_agrees / total_positions >= 0.5:
    print("VERDICT: Net and SF mostly agree ✅")
    print("SF ordering is helping your bot find good moves")
elif (top1_agrees + top3_agrees) / total_positions >= 0.5:
    print("VERDICT: Net partially agrees with SF ⚠️")
    print("SF ordering helps with pruning but bot makes own decisions")
else:
    print("VERDICT: Net and SF mostly disagree ❌")
    print("SF ordering may actually hurt — bot searches wrong branches first")
    print("Consider removing SF ordering and using pure net evaluation")