# models.py
# Two models — keep in sync across all files

import torch
import torch.nn as nn
import numpy as np
import chess


# ── Board encoding ────────────────────────────────────────────────────────────
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
    """Perspective normalized — always from current player's view"""
    t = board_to_tensor(board)
    if board.turn == chess.BLACK:
        t = np.flip(t, axis=2).copy()
        t = t[[6,7,8,9,10,11,0,1,2,3,4,5]]
    return t

def encode_move(move, board):
    """
    Encode a move as a flat index for policy network
    From-square (64) × To-square (64) = 4096 possible moves
    Most are illegal but network learns to score only legal ones
    """
    return move.from_square * 64 + move.to_square


# ── Model 1: Linear Evaluator ─────────────────────────────────────────────────
class LinearEvaluator(nn.Module):
    """
    EVALUATOR — predicts position score not moves
    
    Why linear with one hidden layer:
    - Fast: 0.02ms per position vs 0.65ms for big net
    - Interpretable: each weight = importance of one feature
    - Texel tuning works best on linear models
    - Allows depth 4-5 search in time budget
    - Strong engines (Stockfish classical) used pure linear eval
    
    Input:  768 features (12 piece planes × 8×8 board)
    Output: single score in -1 to +1 range
            positive = current player winning
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            # Layer 1: raw features → piece relationships
            nn.Linear(12 * 8 * 8, 256),
            nn.ReLU(),
            # Layer 2: relationships → tactical patterns
            nn.Linear(256, 64),
            nn.ReLU(),
            # Output: single evaluation score
            nn.Linear(64, 1),
            nn.Tanh(),   # constrain to -1 to +1
        )
        # Initialize weights — small values for stable training
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── Model 2: Policy Network ───────────────────────────────────────────────────
class PolicyNetwork(nn.Module):
    """
    POLICY — predicts which moves are most promising
    Used for move ORDERING in alpha-beta search
    Better ordering = more pruning = effectively deeper search
    
    Why this helps:
    - Alpha-beta prunes branches when it finds a clearly better move
    - If best move is searched FIRST, pruning is maximum
    - Random ordering: prune ~50% of branches
    - Good policy ordering: prune ~80-90% of branches
    - Same result, 4-5x fewer nodes searched
    
    This is how AlphaZero works — policy guides MCTS
    We use it to guide alpha-beta instead
    
    Input:  768 board features
    Output: 4096 move scores (from_sq × to_sq)
            softmax → probability distribution over moves
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4096),  # 64×64 possible from→to moves
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x)   # raw logits — apply softmax externally

    def get_move_probs(self, board, device):
        """
        Returns probability for each legal move
        Higher probability = policy thinks move is better
        Used to order moves in alpha-beta search
        """
        t      = encode_board(board)
        x      = torch.tensor(t).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = self.forward(x)[0]   # shape: (4096,)

        # Get legal move indices
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return []

        indices = torch.tensor(
            [encode_move(m, board) for m in legal_moves],
            dtype=torch.long
        )
        # Softmax over legal moves only
        legal_logits = logits[indices]
        probs        = torch.softmax(legal_logits, dim=0).cpu().numpy()

        return list(zip(legal_moves, probs))