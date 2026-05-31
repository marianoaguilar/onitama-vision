from __future__ import annotations

from math import inf
from onitama.engine.pieces import Player
from onitama.engine.rules import apply_action, generate_legal_actions, is_terminal, Action
from onitama.engine.state import GameState
from onitama.engine.moves import Move

from onitama.ai.types import Evaluator, SearchStats, TranspositionTable, TT_EXACT, TT_LOWER, TT_UPPER

# History table:
# key   -> move (from, to, card)
# value -> score used to sort moves (bigger = usually better)
HistoryTable = dict[tuple[int, int, int, int, int], int]  # (fr, fc, tr, tc, card_index)


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


def _history_key(action: Action) -> tuple[int, int, int, int, int] | None:
    """
    Convert an action to a history table key.
    """
    if not isinstance(action, Move):
        return None
    fr, fc = action.from_pos
    tr, tc = action.to_pos
    return (fr, fc, tr, tc, action.card_index)


def _history_score(action: Action, history: HistoryTable | None) -> int:
    """
    Get the history score for an action, used for move ordering. Higher is better.
    """
    if history is None:
        return 0
    key = _history_key(action)
    return 0 if key is None else history.get(key, 0)


def _color(state: GameState, perspective: Player) -> int:
    return 1 if state.to_move == perspective else -1


def _captures_only(state: GameState, actions: list[Action], mover: Player) -> list[Action]:
    return [a for a in actions if _is_capture(state, a, mover)]


def quiescence(
    state: GameState,
    alpha: float,
    beta: float,
    perspective: Player,
    evaluator: Evaluator,
    q_depth: int,
    tt: TranspositionTable | None = None,
    killer_moves: list[list[Action | None]] | None = None,
    history: HistoryTable | None = None,
    stats: SearchStats | None = None,
    use_move_ordering: bool = True,
) -> int:
    """
    Quiescence search: a limited extension of leaf nodes to reduce horizon effect.
    """
    if stats is not None:
        stats.q_nodes += 1

    # Stand-pat evaluation.
    # If this static score is already good enough for beta, stop now.
    stand_pat = _color(state, perspective) * evaluator(state, perspective)
    if stand_pat >= beta:
        return int(stand_pat)
    if stand_pat > alpha:
        alpha = stand_pat

    if q_depth <= 0 or is_terminal(state):
        return int(stand_pat)

    actions = generate_legal_actions(state)
    mover = state.to_move
    actions = _captures_only(state, actions, mover)
    if not actions:
        return int(stand_pat)

    # Use the same move-order hints here too (TT, killers, history).
    tt_action: Action | None = None
    if tt is not None:
        if stats is not None:
            stats.tt_probes += 1
        entry = tt.get(state)
        if entry is not None:
            if stats is not None:
                stats.tt_hits += 1
            tt_action = entry[3]

    killers = None
    if killer_moves is not None and q_depth < len(killer_moves):
        killers = killer_moves[q_depth]

    if use_move_ordering:
        actions.sort(
            # Sort priority (high to low):
            # 1) TT move, 2) killer move, 3) history score.
            key=lambda a: (
                a == tt_action,
                (killers is not None and a in killers),
                _history_score(a, history),
            ),
            reverse=True,
        )

    for action in actions:
        child = apply_action(state, action)
        score = -quiescence(
            child,
            -beta,
            -alpha,
            perspective,
            evaluator,
            q_depth - 1,
            tt,
            killer_moves,
            history,
            stats,
            use_move_ordering,
        )
        if score >= beta:
            if stats is not None:
                stats.beta_cutoffs += 1
            return int(score)
        if score > alpha:
            alpha = score

    return int(alpha)


# -----------------------------------------------------------------------------

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
    q_depth: int = 0,
    stats: SearchStats | None = None,
    use_move_ordering: bool = True,
) -> int:
    """
    Negamax alpha-beta search.
    Uses:
        - Transposition table (TT) with exact/lower/upper bounds
        - Quiescence search at leaf nodes to reduce horizon effect
        - Move ordering with TT move, killer moves, captures, and history heuristic
        
    Returns a score for the player to move: higher is better for the side to play.
    """

    if stats is not None:
        stats.nodes += 1

    # Read TT entry (if any):
    # - EXACT: can return score directly (if depth is enough)
    # - LOWER/UPPER: can tighten alpha/beta
    tt_action: Action | None = None
    if tt is not None:
        if stats is not None:
            stats.tt_probes += 1
        entry = tt.get(state)
        if entry is not None:
            if stats is not None:
                stats.tt_hits += 1
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
                    if stats is not None:
                        stats.tt_cutoffs += 1
                    return entry_value

    # At depth limit (or terminal), switch to quiescence to reduce horizon effects.
    if depth <= 0 or is_terminal(state):
        return quiescence(
            state,
            alpha,
            beta,
            perspective,
            evaluator,
            q_depth,
            tt,
            killer_moves,
            history,
            stats,
            use_move_ordering,
        )

    actions = generate_legal_actions(state)
    assert actions, "Non-terminal state must have legal actions (Move or Pass)."

    # Move ordering in main search:
    # 1) TT move, 2) killer moves, 3) captures, 4) history
    mover = state.to_move
    killers = None
    if killer_moves is not None and depth < len(killer_moves):
        killers = killer_moves[depth]

    if use_move_ordering:
        actions.sort(
            key=lambda a: (
                a == tt_action,
                (killers is not None and a in killers),
                _is_capture(state, a, mover),
                _history_score(a, history),
            ),
            reverse=True,
        )

    best = -inf
    # Save original window to set the right TT flag later.
    orig_alpha = alpha
    orig_beta = beta
    best_action: Action | None = None

    # Main alpha-beta loop.
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
            q_depth,
            stats,
            use_move_ordering,
        )

        if score > best:
            best = score
            best_action = action
            # Reward moves that improve the best score at this node.
            # Deeper nodes add more history bonus.
            if history is not None:
                key = _history_key(action)
                if key is not None:
                    history[key] = history.get(key, 0) + (depth * depth)
        alpha = max(alpha, best)
        
        if alpha >= beta:
            if stats is not None:
                stats.beta_cutoffs += 1
            # Beta cutoff: store the move as a killer at this depth.
            if use_move_ordering and killer_moves is not None and depth < len(killer_moves):
                if killers is not None and action not in killers:
                    # Keep two killer moves per depth.
                    killers[1] = killers[0]
                    killers[0] = action
            break

    if tt is not None:
        # Store TT flag based on where best falls in the original window.
        if best <= orig_alpha:
            flag = TT_UPPER
        elif best >= orig_beta:
            flag = TT_LOWER
        else:
            flag = TT_EXACT
        tt[state] = (depth, int(best), flag, best_action)
        if stats is not None:
            stats.tt_stores += 1

    return int(best)
