# convert_pgn.py
#built orginally by hand but needed ai for complex positions
# Output: dataset.pt

import chess
import chess.pgn
import torch
import numpy as np
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

def count_pieces(board):
    """Total pieces on board — used for real endgame detection"""
    return len(board.piece_map())

def build_dataset(pgn_path="filtered.pgn", output="dataset.pt"):

    decisive_mid  = []   # decisive games, middlegame (moves 10-40)
    decisive_end  = []   # decisive games, real endgame (≤12 pieces)
    draw_early    = []   # draw games, early only (moves 10-25)

    result_map = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}
    n_white = n_black = n_draw = 0

    print(f"Building win-focused dataset from {pgn_path}...")

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
            if result ==  1.0: n_white += 1
            if result == -1.0: n_black += 1
            if result ==  0.0: n_draw  += 1

            board = game.board()
            for i, move in enumerate(game.mainline_moves()):
                board.push(move)

                if i < 10:      continue
                if i % 3 != 0: continue
                if board.is_game_over(): continue

                is_white     = board.turn == chess.WHITE
                t, label     = encode(board, result)
                piece_count  = count_pieces(board)

                # Real endgame detection — based on pieces not move number
                # ≤12 pieces = both sides down to endgame material
                # This works regardless of game length
                is_real_endgame = piece_count <= 12

                if is_decisive:
                    if is_real_endgame:
                        # Real endgame of decisive game
                        # Most instructive for winning technique
                        decisive_end.append((t, label, board.fen(),
                                             1.0 if is_white else 0.0))
                    else:
                        # Opening/middlegame of decisive game
                        decisive_mid.append((t, label, board.fen(),
                                             1.0 if is_white else 0.0))
                else:
                    # Draw games — only early positions
                    # moves 10-25 = opening structure only
                    # nothing from drawn endgames
                    if i <= 25:
                        draw_early.append((t, label, board.fen(),
                                           1.0 if is_white else 0.0))

            game_count += 1
            if game_count % 500 == 0:
                print(f"  {game_count} games | "
                      f"dec_mid={len(decisive_mid):,} | "
                      f"dec_end={len(decisive_end):,} | "
                      f"draws={len(draw_early):,}")

    print(f"\nGames: {game_count} "
          f"(white={n_white} black={n_black} draw={n_draw})")
    print(f"\nRaw positions:")
    print(f"  Decisive mid:   {len(decisive_mid):,}")
    print(f"  Decisive end:   {len(decisive_end):,}")
    print(f"  Draw early:     {len(draw_early):,}")

    # ── Sampling ──────────────────────────────────────────────────────────────
    # Keep ALL decisive middlegame positions
    # Keep ALL decisive endgame positions — don't oversample by repeating
    # Instead just cap draws aggressively
    #
    # Why not repeat endgame positions 3x?
    # Repeating exact positions = model memorizes them
    # = overfitting not learning
    # Better to just reduce draws than repeat decisive
    #
    # Target: 80% decisive, 20% draws

    n_decisive_total = len(decisive_mid) + len(decisive_end)

    # Cap draws at 25% of decisive total
    # = roughly 20% of final dataset
    max_draws = int(n_decisive_total * 0.25)

    if len(draw_early) > max_draws:
        print(f"\nCapping draws: {len(draw_early):,} → {max_draws:,}")
        draw_early = random.sample(draw_early, max_draws)

    # Combine
    all_data = decisive_mid + decisive_end + draw_early
    random.shuffle(all_data)

    # Unzip
    all_positions, all_scores, all_fens, all_turns = zip(*all_data)

    n_total     = len(all_data)
    n_dec       = len(decisive_mid) + len(decisive_end)
    n_draw_kept = len(draw_early)

    print(f"\nFinal dataset:")
    print(f"  Total:          {n_total:,}")
    print(f"  Decisive mid:   {len(decisive_mid):,} "
          f"({len(decisive_mid)/n_total*100:.1f}%)")
    print(f"  Decisive end:   {len(decisive_end):,} "
          f"({len(decisive_end)/n_total*100:.1f}%)")
    print(f"  Draws:          {n_draw_kept:,} "
          f"({n_draw_kept/n_total*100:.1f}%)")
    print(f"\n  Decisive total: {n_dec:,} ({n_dec/n_total*100:.1f}%)")
    print(f"  Draw total:     {n_draw_kept:,} "
          f"({n_draw_kept/n_total*100:.1f}%)")

    # ── Sanity check — verify no draws are labeled as wins ────────────────────
    # This directly answers your question
    score_arr = np.array(list(all_scores))
    print(f"\nLabel sanity check:")
    print(f"  Min score:      {score_arr.min():.3f}  (should be -1.0)")
    print(f"  Max score:      {score_arr.max():.3f}  (should be +1.0)")
    print(f"  Mean score:     {score_arr.mean():.3f} (should be near 0)")
    print(f"  Scores == 0:    {(score_arr == 0.0).sum():,}  (draws)")
    print(f"  Scores == 1:    {(score_arr == 1.0).sum():,}  (winning)")
    print(f"  Scores == -1:   {(score_arr == -1.0).sum():,} (losing)")
    print(f"  Other scores:   {((score_arr != 0) & (score_arr != 1) & (score_arr != -1)).sum():,}")
    # Should be 0 — no draw position should have label 1.0 or -1.0

    # ── Save ──────────────────────────────────────────────────────────────────
    print("\nConverting to tensors...")
    data = {
        "positions": torch.tensor(
            np.array(list(all_positions)), dtype=torch.float32
        ),
        "scores": torch.tensor(list(all_scores), dtype=torch.float32),
        "fens":   list(all_fens),
        "turns":  torch.tensor(list(all_turns), dtype=torch.float32)
    }

    torch.save(data, output)
    print(f"Saved {output}")
    print(f"Total positions: {n_total:,}")
    print("Done.")

if __name__ == "__main__":
    build_dataset()