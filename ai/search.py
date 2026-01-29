from __future__ import annotations

from math import inf

from onitama.pieces import Player
from onitama.rules import apply_action, generate_legal_actions, is_terminal
from onitama.state import GameState

from ai.evaluate import evaluate


def alphabeta(
    state: GameState,
    depth: int,
    alpha: float,
    beta: float,
    perspective: Player,
) -> int:
    """
    Negamax alpha-beta search.
    Returns a score from the point of view of `perspective`.
    """
    # Terminal or depth limit -> evaluate
    if depth <= 0 or is_terminal(state):
        return evaluate(state, perspective)

    actions = generate_legal_actions(state)
    assert actions, "Non-terminal state must have legal actions (Move or Pass)."

    # If it's not perspective's turn, we still use negamax trick:
    # we always compute values from `perspective`, and flip sign on recursion.

    best = -inf

    for action in actions:
        child = apply_action(state, action)
        score = -alphabeta(child, depth - 1, -beta, -alpha, perspective)

        best = max(best, score)
        alpha = max(alpha, best)
        
        if alpha >= beta:
            break  # beta cut-off

    return int(best)