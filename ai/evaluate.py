# ai/evaluate.py

from __future__ import annotations

from onitama.pieces import PieceType, Player
from onitama.rules import winner
from onitama.state import GameState

WIN_SCORE = 100_000

# Weights (simple on purpose; we can tune later)
STUDENT_WEIGHT = 100
MASTER_DISTANCE_WEIGHT = 10


def _target_temple_for(player: Player) -> tuple[int, int]:
    """
    Temple square the player wants to reach with their own Master
    to win by Way of the Stream.
    """
    # From your description / rules:
    # RED wins when its master reaches (0, 2)
    # BLUE wins when its master reaches (4, 2)
    return (0, 2) if player == Player.RED else (4, 2)


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _find_master_pos(state: GameState, owner: Player) -> tuple[int, int] | None:
    for r in range(5):
        for c in range(5):
            piece = state.board[r][c]
            if piece is None:
                continue
            if piece.owner == owner and piece.kind == PieceType.MASTER:
                return (r, c)
    return None


def _count_students(state: GameState, owner: Player) -> int:
    count = 0
    for r in range(5):
        for c in range(5):
            piece = state.board[r][c]
            if piece is None:
                continue
            if piece.owner == owner and piece.kind == PieceType.STUDENT:
                count += 1
    return count


def evaluate(state: GameState, perspective: Player) -> int:
    """
    Returns an integer score from the point of view of `perspective`.
    Higher is better for `perspective`.
    """
    w = winner(state)
    if w is not None:
        winner_player, _reason = w
        return WIN_SCORE if winner_player == perspective else -WIN_SCORE

    opponent = perspective.opponent()

    # Material: students difference
    my_students = _count_students(state, perspective)
    opp_students = _count_students(state, opponent)
    material_score = (my_students - opp_students) * STUDENT_WEIGHT

    # Master progress: closer to target temple is better
    target = _target_temple_for(perspective)
    my_master = _find_master_pos(state, perspective)

    # If master is missing (should be terminal by rules, but just in case):
    if my_master is None:
        return -WIN_SCORE

    dist = _manhattan(my_master, target)
    master_progress_score = -dist * MASTER_DISTANCE_WEIGHT

    return material_score + master_progress_score
