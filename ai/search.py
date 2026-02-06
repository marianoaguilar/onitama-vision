from __future__ import annotations

from math import inf
from onitama.pieces import Player
from onitama.rules import apply_action, generate_legal_actions, is_terminal, Action
from onitama.state import GameState
from onitama.moves import Move

from ai.types import Evaluator, TranspositionTable, TT_EXACT, TT_LOWER, TT_UPPER

HistoryTable = dict[tuple[int, int, int, int], int]  # (fr, fc, tr, tc)


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


def _history_key(action: Action) -> tuple[int, int, int, int] | None:
    if not isinstance(action, Move):
        return None
    fr, fc = action.from_pos
    tr, tc = action.to_pos
    return (fr, fc, tr, tc)


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
    killer_moves: list[list[Action | None]] | None = None,
    history: HistoryTable | None = None,
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

    # Move ordering inside the search: TT best_action, then killers, then captures, then history
    mover = state.to_move
    killers = None
    if killer_moves is not None and depth < len(killer_moves):
        killers = killer_moves[depth]

    # History heuristic: prefer moves that have been good in past cutoffs.
    def _history_score(action: Action) -> int:
        if history is None:
            return 0
        key = _history_key(action)
        return 0 if key is None else history.get(key, 0)

    actions.sort(
        key=lambda a: (
            a == tt_action,
            (killers is not None and a in killers),
            _is_capture(state, a, mover),
            _history_score(a),
        ),
        reverse=True,
    )

    best = -inf
    orig_alpha = alpha
    orig_beta = beta
    best_action: Action | None = None

    for action in actions:
        child = apply_action(state, action)
        score = -alphabeta(
            child,
            depth - 1,
            -beta,
            -alpha,
            perspective,
            evaluator,
            tt,
            killer_moves,
            history,
        )

        if score > best:
            best = score
            best_action = action
            # Reward moves that improve the best score at this node.
            if history is not None:
                key = _history_key(action)
                if key is not None:
                    history[key] = history.get(key, 0) + (depth * depth)
        alpha = max(alpha, best)
        
        if alpha >= beta:
            # Beta cutoff: store the move as a killer at this depth.
            if killer_moves is not None and depth < len(killer_moves):
                if killers is not None and action not in killers:
                    killers[1] = killers[0]
                    killers[0] = action
            break

    if tt is not None:
        if best <= orig_alpha:
            flag = TT_UPPER
        elif best >= orig_beta:
            flag = TT_LOWER
        else:
            flag = TT_EXACT
        tt[state] = (depth, int(best), flag, best_action)

    return int(best)
