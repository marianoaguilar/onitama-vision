from __future__ import annotations

from math import inf
from onitama.pieces import Player
from onitama.rules import apply_action, generate_legal_actions, is_terminal, Action
from onitama.state import GameState
from onitama.moves import Move

from ai.types import Evaluator, TranspositionTable, TT_EXACT, TT_LOWER, TT_UPPER


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
    evaluator: Evaluator,
    tt: TranspositionTable | None = None,
) -> int:
    """
    Negamax alpha-beta search.
    Returns a score for the player to move: higher is better for the side to play.
    """

    # Transposition table lookup (if any)
    tt_action: Action | None = None
    if tt is not None:
        entry = tt.get(state)
        if entry is not None:
            entry_depth, entry_value, entry_flag, entry_action = entry
            tt_action = entry_action
            if entry_depth >= depth:
                if entry_flag == TT_EXACT:
                    return entry_value
                if entry_flag == TT_LOWER:
                    alpha = max(alpha, entry_value)
                elif entry_flag == TT_UPPER:
                    beta = min(beta, entry_value)
                if alpha >= beta:
                    return entry_value

    # Terminal or depth limit
    if depth <= 0 or is_terminal(state):
        return _color(state, perspective) * evaluator(state, perspective)

    actions = generate_legal_actions(state)
    assert actions, "Non-terminal state must have legal actions (Move or Pass)."

    # Move ordering inside the search: TT best_action first, then captures
    mover = state.to_move
    actions.sort(
        key=lambda a: (a == tt_action, _is_capture(state, a, mover)),
        reverse=True,
    )

    best = -inf
    orig_alpha = alpha
    orig_beta = beta
    best_action: Action | None = None

    for action in actions:
        child = apply_action(state, action)
        score = -alphabeta(child, depth - 1, -beta, -alpha, perspective, evaluator, tt)

        if score > best:
            best = score
            best_action = action
        alpha = max(alpha, best)
        
        if alpha >= beta:
            break  # beta cut-off

    if tt is not None:
        if best <= orig_alpha:
            flag = TT_UPPER
        elif best >= orig_beta:
            flag = TT_LOWER
        else:
            flag = TT_EXACT
        tt[state] = (depth, int(best), flag, best_action)

    return int(best)
