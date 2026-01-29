from __future__ import annotations

from typing import Optional
from math import inf

from onitama.rules import Action
from onitama.pieces import Player
from onitama.rules import apply_action, generate_legal_actions, is_terminal, winner
from onitama.state import GameState
from onitama.moves import Move, Pass

from ai.search import alphabeta


def _action_priority(state: GameState, action: Action, perspective: Player) -> tuple[int, int]:
    """
    Higher tuple means higher priority.
    First key: immediate win (1/0)
    Second key: capture (1/0)
    """
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



def choose_action(state: GameState, depth: int = 2) -> Optional[Action]:
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
    
    perspective: Player = state.to_move
    
    actions = sorted(actions, key=lambda a: _action_priority(state, a, perspective), reverse=True)

    best_action = actions[0]
    best_score = -inf

    alpha = -inf
    beta = inf

    for action in actions:
        child = apply_action(state, action)

        # After making one move, it's opponent's turn. Negamax handles this via sign flip.
        score = -alphabeta(child, depth - 1, -beta, -alpha, perspective)

        if score > best_score:
            best_score = score
            best_action = action

        alpha = max(alpha, best_score)

    return best_action
