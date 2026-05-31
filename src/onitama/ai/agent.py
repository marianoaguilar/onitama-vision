from __future__ import annotations

from typing import Optional
from math import inf

from onitama.engine.pieces import Player
from onitama.engine.rules import Action, apply_action, generate_legal_actions, is_terminal, winner
from onitama.engine.state import GameState
from onitama.engine.moves import Move
from onitama.ai.search import alphabeta
from onitama.ai.evaluate import get_evaluator
from onitama.ai.types import Evaluator, SearchStats, TranspositionTable


def _action_priority(state: GameState, action: Action, perspective: Player) -> tuple[int, int]:
    """
    Higher tuple means higher priority.
    First key: immediate win (1/0)
    Second key: capture (1/0)
    """
    # Cheap 1-ply tactical hint for root ordering.
    child = apply_action(state, action)
    w = winner(child)
    immediate_win = 1 if (w is not None and w[0] == perspective) else 0

    capture = 0
    if isinstance(action, Move):
        tr, tc = action.to_pos
        target_piece = state.board[tr][tc]
        if target_piece is not None and target_piece.owner != perspective:
            capture = 1

    return (immediate_win, capture)



def choose_action(
    state: GameState,
    depth: int = 3,
    evaluator: Evaluator | None = None,
    use_tt: bool = True,
    tt: TranspositionTable | None = None,
    use_iterative_deepening: bool = True,
    aspiration_window: int | None = 100,
    q_depth: int = 2,
    use_move_ordering: bool = True,
    stats: SearchStats | None = None,
) -> Optional[Action]:
    """
    Choose the best action for the player to move in `state` using alpha-beta search.

    Returns:
        - An Action (Move or Pass) if the game is not terminal
        - None if the state is terminal
    """
    if depth < 1:
        raise ValueError("depth must be >= 1")

    if is_terminal(state):
        return None

    actions = generate_legal_actions(state)
    assert actions, "Non-terminal state must have legal actions (Move or Pass)."
    
    # Root side: scores are interpreted from this player's perspective.
    perspective: Player = state.to_move
    
    # Initial root ordering before any search information is available.
    if use_move_ordering:
        actions = sorted(actions, key=lambda a: _action_priority(state, a, perspective), reverse=True)

    if evaluator is None:
        evaluator = get_evaluator("v1")

    if tt is None:
        tt = {} if use_tt else None

    # Per-search move ordering helpers shared across deepening iterations.
    killer_moves = [[None, None] for _ in range(depth + 1)] if use_move_ordering else None
    history = {} if use_move_ordering else None

    # -------------------
    def _search_at_depth(d: int, alpha: float, beta: float, root_actions: list[Action]) -> tuple[Action, int]:
        """
        Search at a given depth with alpha-beta and return the best action and its score.
        """
        best_action_local = root_actions[0]
        best_score_local = -inf
        a = alpha
        b = beta

        for action in root_actions:
            child = apply_action(state, action)

            # After making one move, it's opponent's turn. Negamax handles this via sign flip.
            score = -alphabeta(
                child,
                d - 1,
                -b,
                -a,
                perspective,
                evaluator,
                tt,
                killer_moves,
                history,
                q_depth,
                stats,
                use_move_ordering,
            )

            if score > best_score_local:
                best_score_local = score
                best_action_local = action

            # Root alpha update can cut siblings early.
            a = max(a, best_score_local)
            if a >= b:
                break

        return best_action_local, int(best_score_local)
    # -------------------

    if not use_iterative_deepening:
        best_action, best_score = _search_at_depth(depth, -inf, inf, actions)
        if stats is not None:
            stats.depth_completed = depth
            stats.value = best_score
        return best_action

    best_action = actions[0]
    best_score = -inf
    last_best_action: Action | None = None

    # Iterative deepening driver
    for d in range(1, depth + 1):
        if use_move_ordering and last_best_action is not None:
            # Principal variation move from previous iteration first.
            actions = sorted(
                actions,
                key=lambda a: (a == last_best_action, _action_priority(state, a, perspective)),
                reverse=True,
            )

        if aspiration_window is None or d == 1:
            # Full window for the first iteration (or when aspiration is disabled).
            best_action, best_score = _search_at_depth(d, -inf, inf, actions)
        else:
            # Narrow window around the previous score, often increasing cutoffs.
            a0 = best_score - aspiration_window
            b0 = best_score + aspiration_window
            best_action, best_score = _search_at_depth(d, a0, b0, actions)
            # Fail-low / fail-high: re-search with full window for correctness.
            if best_score <= a0 or best_score >= b0:
                best_action, best_score = _search_at_depth(d, -inf, inf, actions)

        last_best_action = best_action
        if stats is not None:
            stats.depth_completed = d
            stats.value = best_score

    return best_action
