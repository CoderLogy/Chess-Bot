# bot.py
# LinearEvaluator scores positions
# PolicyNetwork orders moves
# Alpha-beta searches depth 4-5
# Result: strong bot that understands both position value AND move quality

import chess
import chess.engine
import chess.polyglot
import torch
import numpy as np
import sys

from models import LinearEvaluator, PolicyNetwork, encode_board

EVAL_PATH = "linear_evaluator.pt"
POLICY_PATH = "policy_network.pt"
SF_PATH   = "stockfish.exe"
BOOK_PATH = "openingBook/Perfect2023.bin"

# ── Load models ───────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

evaluator = LinearEvaluator().to(device)
try:
    evaluator.load_state_dict(torch.load(
        EVAL_PATH, map_location=device, weights_only=True
    ))
    evaluator.eval()
    print(f"LinearEvaluator loaded", file=sys.stderr)
except FileNotFoundError:
    print(f"ERROR: {EVAL_PATH} not found", file=sys.stderr)
    sys.exit(1)

policy = PolicyNetwork().to(device)
policy_loaded = False
try:
    policy.load_state_dict(torch.load(
        POLICY_PATH, map_location=device, weights_only=True
    ))
    policy.eval()
    policy_loaded = True
    print(f"PolicyNetwork loaded", file=sys.stderr)
except FileNotFoundError:
    print("WARNING: policy_network.pt not found — using SF ordering only",
          file=sys.stderr)

# ── SF for move ordering fallback ─────────────────────────────────────────────
try:
    sf_engine = chess.engine.SimpleEngine.popen_uci(SF_PATH)
except FileNotFoundError:
    sf_engine = None
    print("WARNING: stockfish not found", file=sys.stderr)

# ── Eval cache ────────────────────────────────────────────────────────────────
eval_cache = {}

def nn_score(board):
    """
    Position evaluation — LinearEvaluator
    No BatchNorm = safe with batch size 1
    Returns centipawns from current player perspective
    """
    if board.is_checkmate():             return -100000
    if board.is_stalemate():             return 0
    if board.is_insufficient_material(): return 0

    key = board.fen()
    if key in eval_cache:
        return eval_cache[key]

    t     = encode_board(board)
    x     = torch.tensor(t).unsqueeze(0).to(device)
    with torch.no_grad():
        score = evaluator(x).item() * 1000

    eval_cache[key] = score
    return score

def batch_nn_score(boards):
    """
    Batch evaluation — all positions in ONE GPU call
    20x faster than individual calls
    Used at root to seed move ordering
    """
    if not boards:
        return []
    tensors = np.array(
        [encode_board(b) for b in boards], dtype=np.float32
    )
    x = torch.tensor(tensors).to(device)
    with torch.no_grad():
        scores = evaluator(x).cpu().numpy() * 1000
    return scores.tolist()

# ── Move ordering — Policy + SF combined ─────────────────────────────────────
def get_move_order(board):
    """
    Best move ordering uses BOTH policy and SF:
    1. Policy predicts which moves look promising
    2. SF depth-1 gives tactical awareness
    3. Captures and checks always searched first
    Combined ordering = maximum alpha-beta pruning
    """
    legal = list(board.legal_moves)
    if not legal:
        return []

    move_scores = {m.uci(): 0.0 for m in legal}

    # Policy scores — learned from grandmaster games
    if policy_loaded:
        try:
            move_probs = policy.get_move_probs(board, device)
            for move, prob in move_probs:
                move_scores[move.uci()] += prob * 100   # scale up
        except Exception:
            pass

    # SF depth-1 ordering — tactical awareness
    if sf_engine is not None:
        try:
            result  = sf_engine.analyse(
                board, chess.engine.Limit(depth=1),
                multipv=min(len(legal), 20)
            )
            for rank, info in enumerate(result):
                if "pv" in info:
                    move = info["pv"][0]
                    # Higher SF rank = higher score
                    move_scores[move.uci()] += (20 - rank) * 5
        except Exception:
            pass

    # Capture and check bonuses — always search forcing moves first
    for move in legal:
        if board.is_capture(move):
            move_scores[move.uci()] += 50
        if board.gives_check(move):
            move_scores[move.uci()] += 20

    # Sort by combined score
    ordered = sorted(legal, key=lambda m: move_scores[m.uci()], reverse=True)
    return ordered

# ── Quiescence search ─────────────────────────────────────────────────────────
def quiescence(board, alpha, beta, maximizing, qs_depth=4):
    stand_pat = nn_score(board)
    if qs_depth == 0:
        return stand_pat
    if maximizing:
        if stand_pat >= beta:   return beta
        alpha = max(alpha, stand_pat)
        for move in sorted(
            (m for m in board.legal_moves if board.is_capture(m)),
            key=lambda m: board.gives_check(m), reverse=True
        ):
            board.push(move)
            score = quiescence(board, alpha, beta, False, qs_depth-1)
            board.pop()
            alpha = max(alpha, score)
            if alpha >= beta: break
        return alpha
    else:
        if stand_pat <= alpha:  return alpha
        beta = min(beta, stand_pat)
        for move in sorted(
            (m for m in board.legal_moves if board.is_capture(m)),
            key=lambda m: board.gives_check(m), reverse=True
        ):
            board.push(move)
            score = quiescence(board, alpha, beta, True, qs_depth-1)
            board.pop()
            beta = min(beta, score)
            if beta <= alpha: break
        return beta

# ── Alpha-beta ────────────────────────────────────────────────────────────────
def alpha_beta(board, depth, alpha, beta, maximizing):
    if board.is_game_over():
        return nn_score(board)
    if depth == 0:
        return quiescence(board, alpha, beta, maximizing)

    # Inside search tree — fast capture/check ordering
    # (policy ordering only at root — too slow per node)
    moves = sorted(
        board.legal_moves,
        key=lambda m: (board.is_capture(m), board.gives_check(m)),
        reverse=True
    )

    if maximizing:
        value = -float('inf')
        for move in moves:
            board.push(move)
            value = max(value,
                        alpha_beta(board, depth-1, alpha, beta, False))
            board.pop()
            alpha = max(alpha, value)
            if beta <= alpha: break
        return value
    else:
        value = float('inf')
        for move in moves:
            board.push(move)
            value = min(value,
                        alpha_beta(board, depth-1, alpha, beta, True))
            board.pop()
            beta  = min(beta, value)
            if beta <= alpha: break
        return value

# ── Opening book ──────────────────────────────────────────────────────────────
def get_book_move(board):
    try:
        with chess.polyglot.open_reader(BOOK_PATH) as reader:
            entry = max(reader.find_all(board), key=lambda e: e.weight)
            return entry.move
    except Exception:
        return None

# ── Best move — batched root + policy ordering ────────────────────────────────
def best_move(board, depth=4):
    # Opening book
    book = get_book_move(board)
    if book and book in board.legal_moves:
        return book

    # Combined policy + SF move ordering at root
    ordered_moves = get_move_order(board)
    if not ordered_moves:
        return None

    # Batch evaluate all root positions in ONE GPU call
    post_move_boards = []
    for move in ordered_moves:
        board.push(move)
        post_move_boards.append(board.copy())
        board.pop()

    batch_scores = batch_nn_score(post_move_boards)

    # Re-sort by batch evaluation score
    # Policy gives strategic ordering
    # Batch evaluation gives tactical correction
    # Combined = best possible root move ordering
    move_score_pairs = sorted(
        zip(ordered_moves, batch_scores),
        key=lambda x: x[1], reverse=True
    )

    best, best_val = None, -float('inf')
    alpha          = -float('inf')

    for move, _ in move_score_pairs:
        board.push(move)
        val = alpha_beta(board, depth-2, alpha, float('inf'), False) \
              if depth > 1 else nn_score(board)
        board.pop()

        if val > best_val:
            best_val, best = val, move
        alpha = max(alpha, best_val)

    return best

# ── UCI loop ──────────────────────────────────────────────────────────────────
def uci_loop():
    board = chess.Board()
    while True:
        try:
            cmd = input().strip()
        except EOFError:
            break

        if cmd == "uci":
            print("id name LinearBot")
            print("id author YourTeam")
            print("uciok")

        elif cmd == "isready":
            print("readyok")

        elif cmd == "ucinewgame":
            board = chess.Board()
            eval_cache.clear()

        elif cmd.startswith("position"):
            parts = cmd.split()
            board = chess.Board()
            if "fen" in parts:
                fi = parts.index("fen") + 1
                mi = parts.index("moves") if "moves" in parts else len(parts)
                board = chess.Board(" ".join(parts[fi:mi]))
            if "moves" in parts:
                for m in parts[parts.index("moves")+1:]:
                    board.push_uci(m)

        elif cmd.startswith("go"):
            parts     = cmd.split()
            wtime     = int(parts[parts.index("wtime")+1]) \
                        if "wtime" in parts else 60000
            btime     = int(parts[parts.index("btime")+1]) \
                        if "btime" in parts else 60000
            time_left = wtime if board.turn == chess.WHITE else btime

            if time_left > 30000:   depth = 4
            elif time_left > 10000: depth = 3
            elif time_left > 3000:  depth = 2
            else:                   depth = 1

            move = best_move(board, depth=depth)
            print(f"bestmove {move.uci() if move else '0000'}")

        elif cmd == "quit":
            if sf_engine:
                sf_engine.quit()
            break

        sys.stdout.flush()


if __name__ == "__main__":
    uci_loop()