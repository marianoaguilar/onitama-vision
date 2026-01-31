from __future__ import annotations

from math import inf

from onitama.pieces import Player
from onitama.rules import apply_action, generate_legal_actions, is_terminal, Action
from onitama.state import GameState
from onitama.moves import Move

from ai.evaluate import evaluate


def _is_capture(state: GameState, action: Action, mover: Player) -> bool:
    """
    Cheap move ordering heuristic:
    True if this action captures an opponent piece.
    """
    if not isinstance(action, Move):
        return False

    tr, tc = action.to_pos
    target = state.board[tr][tc]
    return target is not None and target.owner != mover


def _color(state: GameState, perspective: Player) -> int:
    return 1 if state.to_move == perspective else -1



def alphabeta(
    state: GameState,
    depth: int,
    alpha: float,
    beta: float,
    perspective: Player,
) -> int:
    """
    Negamax alpha-beta search.
    Returns a score for the player to move: higher is better for the side to play.
    """

    # Terminal or depth limit
    if depth <= 0 or is_terminal(state):
        return _color(state, perspective) * evaluate(state, perspective)

    actions = generate_legal_actions(state)
    assert actions, "Non-terminal state must have legal actions (Move or Pass)."

    # Move ordering inside the search: captures first
    mover = state.to_move
    actions.sort(key=lambda a: _is_capture(state, a, mover), reverse=True)

    best = -inf

    for action in actions:
        child = apply_action(state, action)
        score = -alphabeta(child, depth - 1, -beta, -alpha, perspective)

        best = max(best, score)
        alpha = max(alpha, best)
        
        if alpha >= beta:
            break  # beta cut-off

    return int(best)