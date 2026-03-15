#built search and sorting by hand but let ai handle rest
import chess
import chess.engine
import chess.polyglot
import torch
import torch.nn as nn
import numpy as np
import sys


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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = Evaluator().to(device)

try:
    model.load_state_dict(torch.load(
        "evaluator.pt", map_location=device, weights_only=True
    ))
    # CRITICAL — must be eval() for two reasons:
    # 1. BatchNorm uses running stats in eval, live stats in train
    #    single-sample inference with train mode = wrong results
    # 2. Dropout is disabled in eval — you want deterministic play
    model.eval()
except FileNotFoundError:
    print("ERROR: evaluator.pt not found — run train.py first")
    sys.exit(1)

# Open SF once at startup
try:
    sf_engine = chess.engine.SimpleEngine.popen_uci("stockfish.exe")
except FileNotFoundError:
    sf_engine = None
    print("WARNING: stockfish not found — move ordering disabled")

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
    """
    Same perspective flip as convert_pgn.py encode()
    Must match training exactly or evaluations are wrong
    """
    t = board_to_tensor(board)
    if board.turn == chess.BLACK:
        t = np.flip(t, axis=2).copy()
        t = t[[6,7,8,9,10,11,0,1,2,3,4,5]]
    return t

eval_cache = {}

def nn_score(board):
    if board.is_checkmate():             return -100000
    if board.is_stalemate():             return 0
    if board.is_insufficient_material(): return 0

    key = board.fen()
    if key in eval_cache:
        return eval_cache[key]

    t      = encode_board(board)
    x      = torch.tensor(t).unsqueeze(0).to(device)
    with torch.no_grad():
        score = model(x).item() * 1000

    eval_cache[key] = score
    return score

def get_sf_move_order(board):
    if sf_engine is None:
        return sorted(
            board.legal_moves,
            key=lambda m: (board.is_capture(m), board.gives_check(m)),
            reverse=True
        )
    legal = list(board.legal_moves)
    if not legal:
        return []
    try:
        result  = sf_engine.analyse(
            board,
            chess.engine.Limit(depth=1),
            multipv=min(len(legal), 20)
        )
        ordered = [info["pv"][0] for info in result if "pv" in info]
        seen    = {m.uci() for m in ordered}
        for m in legal:
            if m.uci() not in seen:
                ordered.append(m)
        return ordered
    except Exception:
        return sorted(
            legal,
            key=lambda m: (board.is_capture(m), board.gives_check(m)),
            reverse=True
        )

def quiescence(board, alpha, beta, maximizing, qs_depth=3):
    """Search captures until the position is quiet."""
    stand_pat = nn_score(board)

    if qs_depth == 0:
        return stand_pat

    if maximizing:
        if stand_pat >= beta:
            return stand_pat
        alpha = max(alpha, stand_pat)

        for move in sorted(
            (m for m in board.legal_moves if board.is_capture(m)),
            key=lambda m: board.gives_check(m), reverse=True
        ):
            board.push(move)
            score = quiescence(board, alpha, beta, False, qs_depth - 1)
            board.pop()
            if score >= beta:
                return score
            alpha = max(alpha, score)
        return alpha
    else:
        if stand_pat <= alpha:
            return stand_pat
        beta = min(beta, stand_pat)

        for move in sorted(
            (m for m in board.legal_moves if board.is_capture(m)),
            key=lambda m: board.gives_check(m), reverse=True
        ):
            board.push(move)
            score = quiescence(board, alpha, beta, True, qs_depth - 1)
            board.pop()
            if score <= alpha:
                return score
            beta = min(beta, score)
        return beta

def alpha_beta(board, depth, alpha, beta, maximizing):
    if board.is_game_over():
        return nn_score(board)
    if depth == 0:
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

BOOK_PATH = "openingBook/Perfect2023.bin"

def get_book_move(board):
    """
    Looks up current position in opening book.
    weighted_choice picks moves proportionally to their weight —
    plays varied openings, not always the same line.
    """
    try:
        with chess.polyglot.open_reader(BOOK_PATH) as reader:
            entry = reader.weighted_choice(board)
            return entry.move
    except (IndexError, FileNotFoundError, Exception):
        return None

def get_best_book_move(board):
    """
    Always picks the highest weighted move.
    More consistent but less varied — better for bot vs bot competition.
    """
    try:
        with chess.polyglot.open_reader(BOOK_PATH) as reader:
            entry = max(reader.find_all(board), key=lambda e: e.weight)
            return entry.move
    except Exception:
        return None

def best_move(board, depth=3):
    # 1. Check opening book first — always strongest in opening
    book_move = get_best_book_move(board)
    if book_move and book_move in board.legal_moves:
        return book_move

    # 2. SF move ordering at root
    ordered_moves = get_sf_move_order(board)
    if not ordered_moves:
        return None

    # 3. Alpha-beta with the neural net
    best, best_val = None, -float('inf')
    alpha          = -float('inf')
    for move in ordered_moves:
        board.push(move)
        val = alpha_beta(board, depth-1, alpha, float('inf'), False)
        board.pop()
        if val > best_val:
            best_val, best = val, move
        alpha = max(alpha, best_val)
    return best

def uci_loop():
    board = chess.Board()
    while True:
        try:
            cmd = input().strip()
        except EOFError:
            break

        if cmd == "uci":
            print("id name HackathonBot")
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
                fen_idx   = parts.index("fen") + 1
                moves_idx = parts.index("moves") if "moves" in parts else len(parts)
                board     = chess.Board(" ".join(parts[fen_idx:moves_idx]))
            if "moves" in parts:
                for m in parts[parts.index("moves")+1:]:
                    board.push_uci(m)

        elif cmd.startswith("go"):
            parts     = cmd.split()
            wtime     = int(parts[parts.index("wtime")+1]) if "wtime" in parts else 60000
            btime     = int(parts[parts.index("btime")+1]) if "btime" in parts else 60000
            time_left = wtime if board.turn == chess.WHITE else btime

            if time_left > 60000:   depth = 3
            elif time_left > 20000: depth = 2
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