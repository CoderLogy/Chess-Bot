# convert_pgn.py
# Optimized for winning — bot learns to WIN not draw
# Strategy:
#   - Only decisive games for full position extraction
#   - Draws included minimally just for balance
#   - Late game winning positions oversampled
#     (these are most instructive for winning technique)

import chess
import chess.pgn
import torch
import numpy as np
from tqdm import tqdm
import random

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

def encode(board, result):
    t = board_to_tensor(board)
    if board.turn == chess.BLACK:
        t      = np.flip(t, axis=2).copy()
        t      = t[[6,7,8,9,10,11,0,1,2,3,4,5]]
        result = -result
    return t, result

def build_dataset(pgn_path="filtered.pgn", output="dataset.pt"):

    # Three buckets — treated very differently
    winning_positions  = []   # from decisive games — FULL game kept
    winning_scores     = []
    winning_fens       = []
    winning_turns      = []

    # Late game winning positions — extra valuable
    # when a player converts a won endgame
    # model learns winning technique specifically
    endgame_positions  = []
    endgame_scores     = []
    endgame_fens       = []
    endgame_turns      = []

    draw_positions     = []   # minimal inclusion for stability
    draw_scores        = []
    draw_fens          = []
    draw_turns         = []

    result_map = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}

    print(f"Building win-focused dataset from {pgn_path}...")

    n_white = n_black = n_draw = 0

    with open(pgn_path) as f:
        game_count = 0
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            result = result_map.get(game.headers.get("Result", ""))
            if result is None:
                continue

            is_decisive = result != 0.0
            if result == 1.0:  n_white += 1
            if result == -1.0: n_black += 1
            if result == 0.0:  n_draw  += 1

            # Count total moves for endgame detection
            moves_list = list(game.mainline_moves())
            total_moves = len(moves_list)

            board = game.board()
            for i, move in enumerate(moves_list):
                board.push(move)

                if i < 10:      continue
                if i % 3 != 0: continue
                if board.is_game_over(): continue

                is_white  = board.turn == chess.WHITE
                t, label  = encode(board, result)

                if is_decisive:
                    # Is this a late game position?
                    # Last 30% of moves in a decisive game
                    # = winning technique phase
                    is_endgame = i > total_moves * 0.7

                    if is_endgame:
                        # Store separately for heavy oversampling
                        endgame_positions.append(t)
                        endgame_scores.append(label)
                        endgame_fens.append(board.fen())
                        endgame_turns.append(1.0 if is_white else 0.0)
                    else:
                        winning_positions.append(t)
                        winning_scores.append(label)
                        winning_fens.append(board.fen())
                        winning_turns.append(1.0 if is_white else 0.0)

                else:
                    # Draws: only keep opening/early middlegame
                    # move 10-30 only — structural understanding
                    # nothing from drawn endgames
                    if 10 <= i <= 30:
                        draw_positions.append(t)
                        draw_scores.append(label)
                        draw_fens.append(board.fen())
                        draw_turns.append(1.0 if is_white else 0.0)

            game_count += 1
            if game_count % 500 == 0:
                print(f"  {game_count} games | "
                      f"winning: {len(winning_positions):,} | "
                      f"endgame: {len(endgame_positions):,} | "
                      f"draws: {len(draw_positions):,}")

    print(f"\nGames:  total={game_count}  "
          f"white={n_white}  black={n_black}  draw={n_draw}")
    print(f"\nRaw positions:")
    print(f"  Winning (mid):    {len(winning_positions):,}")
    print(f"  Winning (end):    {len(endgame_positions):,}")
    print(f"  Draws (early):    {len(draw_positions):,}")

    # ── Sampling strategy ─────────────────────────────────────────────────────
    #
    # Target composition:
    #   50% — regular winning positions (middlegame of decisive games)
    #   30% — endgame winning positions (oversampled 3x)
    #          model really needs to learn how to convert wins
    #   20% — early draw positions (just enough for opening structure)
    #
    # Why oversample endgames:
    #   Converting a won position is the hardest thing for bots
    #   They often let wins slip into draws
    #   Seeing more "this was +0.8 and they converted it" teaches
    #   the bot that winning positions require active play not shuffling

    n_winning  = len(winning_positions)
    n_endgame  = len(endgame_positions)

    # Cap draws at 40% of winning total
    # enough to learn openings, not enough to dominate
    max_draws  = int(n_winning * 0.4)
    if len(draw_positions) > max_draws:
        indices        = random.sample(range(len(draw_positions)), max_draws)
        draw_positions = [draw_positions[i] for i in indices]
        draw_scores    = [draw_scores[i]    for i in indices]
        draw_fens      = [draw_fens[i]      for i in indices]
        draw_turns     = [draw_turns[i]     for i in indices]

    # Oversample endgame positions 3x
    # repeat them so model sees them more often during training
    endgame_positions = endgame_positions * 3
    endgame_scores    = endgame_scores    * 3
    endgame_fens      = endgame_fens      * 3
    endgame_turns     = endgame_turns     * 3

    # Combine all
    all_positions = winning_positions + endgame_positions + draw_positions
    all_scores    = winning_scores    + endgame_scores    + draw_scores
    all_fens      = winning_fens      + endgame_fens      + draw_fens
    all_turns     = winning_turns     + endgame_turns     + draw_turns

    # Shuffle
    combined = list(zip(all_positions, all_scores, all_fens, all_turns))
    random.shuffle(combined)
    all_positions, all_scores, all_fens, all_turns = zip(*combined)

    # Stats
    n_total    = len(all_positions)
    n_dec_tot  = len(winning_positions) + len(endgame_positions)
    n_draw_tot = len(draw_positions)

    print(f"\nFinal dataset composition:")
    print(f"  Total:            {n_total:,}")
    print(f"  Decisive (mid):   {len(winning_positions):,}  "
          f"({len(winning_positions)/n_total*100:.1f}%)")
    print(f"  Decisive (end×3): {len(endgame_positions):,}  "
          f"({len(endgame_positions)/n_total*100:.1f}%)")
    print(f"  Draws (capped):   {n_draw_tot:,}  "
          f"({n_draw_tot/n_total*100:.1f}%)")
    print(f"\n  Decisive total:   {n_dec_tot:,}  "
          f"({n_dec_tot/n_total*100:.1f}%)")
    print(f"  Draw total:       {n_draw_tot:,}  "
          f"({n_draw_tot/n_total*100:.1f}%)")

    print("\nConverting to tensors...")
    data = {
        "positions": torch.tensor(
            np.array(all_positions), dtype=torch.float32
        ),
        "scores": torch.tensor(list(all_scores), dtype=torch.float32),
        "fens":   list(all_fens),
        "turns":  torch.tensor(list(all_turns),  dtype=torch.float32)
    }

    torch.save(data, output)
    print(f"Saved {output}")
    print(f"Total positions: {n_total:,}")
    print("Done.")

if __name__ == "__main__":
    build_dataset()
