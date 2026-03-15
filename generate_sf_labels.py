#built with ai
# Random sampling — not just first 500k positions

import chess
import chess.engine
import torch
import random
from tqdm import tqdm

def generate(
    input_file    = "dataset.pt",
    output_file   = "sf_dataset.pt",
    sf_depth      = 8,
    max_positions = 1000000
):
    print(f"Loading {input_file}...")
    data      = torch.load(input_file, weights_only=False)
    positions = data["positions"]
    fens      = data["fens"]
    turns     = data["turns"]

    # Random sample — not first N
    # ensures diverse positions across all game phases
    all_indices = list(range(len(positions)))
    if len(positions) > max_positions:
        indices = sorted(random.sample(all_indices, max_positions))
    else:
        indices = all_indices
    total = len(indices)

    # Estimate time — depth 8 takes ~0.15s per position
    secs_per_pos = 0.05 if sf_depth <= 4 else 0.15
    est_mins    = total * secs_per_pos / 3600

    print(f"Loaded {len(positions):,} positions total")
    print(f"Will evaluate {total:,} positions")
    print(f"Stockfish depth:  {sf_depth}")
    print(f"Estimated time:   {est_mins:.1f} mins")
    print(f"  (depth 8 is ~3x slower than depth 4)\n")

    sf_positions = []
    sf_scores    = []
    skipped      = 0

    print("Opening Stockfish...")
    with chess.engine.SimpleEngine.popen_uci("stockfish.exe") as engine:
        engine.configure({"Threads": 4, "Hash": 256})

        pbar = tqdm(range(total), desc="Evaluating", unit="pos")

        for i in pbar:
            real_i = indices[i]
            try:
                board = chess.Board(fens[real_i])

                if board.is_game_over():
                    skipped += 1
                    continue

                info  = engine.analyse(
                    board,
                    chess.engine.Limit(depth=sf_depth)
                )
                score = info["score"].white().score(mate_score=10000)

                if score is None:
                    skipped += 1
                    continue

                # Normalize to -1 to +1
                score = float(max(-1.0, min(1.0, score / 1000.0)))

                # Flip to current player's perspective
                # SF always returns from White's view
                # encode_board() flips Black positions
                # so label must also be from current player's view
                if turns[real_i].item() == 0.0:   # Black's turn
                    score = -score

                sf_positions.append(positions[real_i])
                sf_scores.append(score)

                pbar.set_postfix(
                    evaluated = len(sf_scores),
                    skipped   = skipped
                )

            except Exception:
                skipped += 1
                continue

        pbar.close()

    print(f"\nDone evaluating")
    print(f"Successfully evaluated: {len(sf_scores):,}")
    print(f"Skipped:                {skipped:,}")

    if len(sf_scores) == 0:
        print("ERROR: No positions evaluated — check Stockfish is installed")
        return

    print(f"\nSaving {output_file}...")

    pos_tensor   = torch.stack(sf_positions)
    score_tensor = torch.tensor(sf_scores, dtype=torch.float32)

    # Sanity check — verify win-focused distribution carried through
    decisive = (score_tensor.abs() > 0.1).sum().item()
    draws    = (score_tensor.abs() <= 0.1).sum().item()
    print(f"\nSanity check:")
    print(f"  Decisive positions: {decisive:,} ({decisive/len(sf_scores)*100:.1f}%)")
    print(f"  Near-draw positions:{draws:,}  ({draws/len(sf_scores)*100:.1f}%)")
    if decisive / len(sf_scores) > 0.7:
        print("  ✅ Win-focused distribution confirmed")
    else:
        print("  ⚠️  Still draw-heavy — check dataset.pt composition")

    torch.save({
        "positions": pos_tensor,
        "scores":    score_tensor,
    }, output_file)

    print(f"\nSaved {output_file}")
    print(f"Shape:       {pos_tensor.shape}")
    print(f"Score range: {score_tensor.min():.3f} to {score_tensor.max():.3f}")
    print(f"Score mean:  {score_tensor.mean():.3f}")
    print("\nReady for retrain.py")


if __name__ == "__main__":
    generate()