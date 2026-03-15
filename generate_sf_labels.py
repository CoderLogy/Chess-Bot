# Reads dataset.pt directly
# Uses stored FENs for accurate board reconstruction
# Uses stored turns to flip SF score to current player's perspective


import chess
import chess.engine
import torch
from tqdm import tqdm

def generate(
    input_file    = "dataset.pt",
    output_file   = "sf_dataset.pt",
    sf_depth      = 8,
    max_positions = 500000    # 500k
):
    print(f"Loading {input_file}...")
   
    data      = torch.load(input_file, weights_only=False)
    positions = data["positions"]   # (N, 12, 8, 8) float32
    fens      = data["fens"]        # list of FEN strings
    turns     = data["turns"]       # 1.0=white, 0.0=black

    total = min(len(positions), max_positions)

    print(f"Loaded {len(positions)} positions total")
    print(f"Will evaluate {total} positions")
    print(f"Stockfish depth: {sf_depth}")

    sf_positions = []
    sf_scores    = []
    skipped      = 0

    print("Opening Stockfish...")
    with chess.engine.SimpleEngine.popen_uci("stockfish.exe") as engine:
        engine.configure({"Threads": 4, "Hash": 256})

        pbar = tqdm(range(total), desc="Evaluating", unit="pos")

        for i in pbar:
            try:
                # Reconstruct board from FEN
                # FEN stores the real board state including whose turn it is
                board = chess.Board(fens[i])

                if board.is_game_over():
                    skipped += 1
                    continue

                # Get Stockfish evaluation
                info  = engine.analyse(
                    board,
                    chess.engine.Limit(depth=sf_depth)
                )
                score = info["score"].white().score(mate_score=10000)

                if score is None:
                    skipped += 1
                    continue

                # Step 1 normalize raw centipawns to -1 to +1
                # matches Tanh output range of your net
                score = float(max(-1.0, min(1.0, score / 1000.0)))

                # Step 2  flip score to current player's perspective
                # SF always returns score from White's perspective
                if turns[i].item() == 0.0:   # Black's turn
                    score = -score

                sf_positions.append(positions[i])
                sf_scores.append(score)

                pbar.set_postfix(
                    evaluated = len(sf_scores),
                    skipped   = skipped
                )

            except Exception as e:
                skipped += 1
                continue

        pbar.close()

    print(f"\nDone evaluating")
    print(f"Successfully evaluated: {len(sf_scores)}")
    print(f"Skipped:                {skipped}")

    if len(sf_scores) == 0:
        print("ERROR: No positions evaluated — check Stockfish is installed")
        print("Install: https://stockfishchess.org/download/")
        return

    print(f"\nSaving {output_file}...")

    # Stack list of tensors into one tensor
    pos_tensor   = torch.stack(sf_positions)
    score_tensor = torch.tensor(sf_scores, dtype=torch.float32)

    # Save in exact same format as dataset.pt
    torch.save({
        "positions": pos_tensor,
        "scores":    score_tensor,
    }, output_file)

    print(f"Saved {output_file}")
    print(f"Shape: {pos_tensor.shape}")
    print(f"Score range: {score_tensor.min():.3f} to {score_tensor.max():.3f}")
    print(f"Score mean:  {score_tensor.mean():.3f}")
    print("\nReady for retrain.py")


if __name__ == "__main__":
    generate()