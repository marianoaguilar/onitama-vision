# ai/evaluate.py

from __future__ import annotations

from dataclasses import replace, dataclass

from onitama.moves import Move
from onitama.pieces import PieceType, Player
from onitama.rules import generate_legal_actions, winner
from onitama.state import GameState

from typing import Callable

Evaluator = Callable[[GameState, Player], int]


WIN_SCORE = 100_000

# ----------------------------------------------------------------------------
# V1 Weights
STUDENT_WEIGHT = 300
MASTER_DISTANCE_WEIGHT = 8

# New weights for tactical safety / positioning
MASTER_THREAT_PENALTY = 8_000
HANGING_STUDENT_PENALTY = 600

MOBILITY_WEIGHT = 0

# ----------------------------------------------------------------------------
# V2: generic heuristic with tunable weights (variants: v2a, v2b, v2c, ...)

@dataclass(frozen=True)
class EvalWeightsV2:
    student_weight: int
    master_distance_weight: int

    master_threat_penalty: int
    master_threat_bonus: int

    hanging_student_penalty: int
    hanging_student_bonus: int

    mobility_weight: int  # counts only Move actions (Pass excluded)

    centrality_weight: int
    advancement_weight: int


V2A = EvalWeightsV2(
    student_weight=300,
    master_distance_weight=10,
    master_threat_penalty=8000,
    master_threat_bonus=7000,
    hanging_student_penalty=550,
    hanging_student_bonus=450,
    mobility_weight=8,
    centrality_weight=25,
    advancement_weight=12,
)

# Slightly more "tactical": punish hanging more, reward threats more.
V2B = EvalWeightsV2(
    student_weight=300,
    master_distance_weight=10,
    master_threat_penalty=8500,
    master_threat_bonus=7500,
    hanging_student_penalty=650,
    hanging_student_bonus=500,
    mobility_weight=8,
    centrality_weight=20,
    advancement_weight=10,
)

# Slightly more "positional": emphasize centrality/advancement a bit more.
V2C = EvalWeightsV2(
    student_weight=280,
    master_distance_weight=10,
    master_threat_penalty=8000,
    master_threat_bonus=6500,
    hanging_student_penalty=550,
    hanging_student_bonus=450,
    mobility_weight=10,
    centrality_weight=35,
    advancement_weight=18,
)


# ----------------------------------------------------------------------------
# Helpers

def _target_temple_for(player: Player) -> tuple[int, int]:
    """
    Temple square the player wants to reach with their own Master
    to win by Way of the Stream.
    """
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


def _opponent_can_capture_master_next(state: GameState, perspective: Player) -> bool:
    """
    Returns True if the opponent has any legal Move that lands on perspective's Master square
    on their next turn (1-ply threat).
    """
    master_pos = _find_master_pos(state, perspective)
    if master_pos is None:
        # Should be terminal, but if not, treat as immediate danger.
        return True

    opponent = perspective.opponent()
    opp_state = replace(state, to_move=opponent)
    opp_actions = generate_legal_actions(opp_state)

    for a in opp_actions:
        if isinstance(a, Move) and a.to_pos == master_pos:
            return True

    return False


def _count_legal_actions_for(state: GameState, player: Player) -> int:
    s = replace(state, to_move=player)
    return len(generate_legal_actions(s))


def _count_hanging_students_next(state: GameState, perspective: Player) -> int:
    """
    Counts how many of perspective's STUDENT pieces can be captured by the opponent
    on their next move (1-ply).
    """
    opponent = perspective.opponent()
    opp_state = replace(state, to_move=opponent)
    opp_actions = generate_legal_actions(opp_state)

    hanging = 0
    for a in opp_actions:
        if not isinstance(a, Move):
            continue
        tr, tc = a.to_pos
        target = state.board[tr][tc]
        if target is None:
            continue
        if target.owner != perspective:
            continue
        if target.kind == PieceType.STUDENT:
            hanging += 1

    return hanging

# ----------------------------------------------------------------------------
# Helpers V2

def _actions_for(state: GameState, player: Player) -> list:
    """Legal actions as if `player` were to move now."""
    s = replace(state, to_move=player)
    return list(generate_legal_actions(s))


def _move_actions(actions: list) -> list[Move]:
    """Filter only real moves (exclude Pass)."""
    return [a for a in actions if isinstance(a, Move)]


def _iter_positions(state: GameState, owner: Player, kind: PieceType) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for r in range(5):
        for c in range(5):
            p = state.board[r][c]
            if p is None:
                continue
            if p.owner == owner and p.kind == kind:
                out.append((r, c))
    return out


def _centrality_value(pos: tuple[int, int]) -> int:
    """Closer to center (2,2) is better. Range: 0..4"""
    r, c = pos
    return 4 - (abs(r - 2) + abs(c - 2))


def _advancement_value(pos: tuple[int, int], owner: Player) -> int:
    """How advanced a piece is towards the opponent side."""
    r, _c = pos
    return (4 - r) if owner == Player.RED else r



# ----------------------------------------------------------------------------

def evaluate_v1(state: GameState, perspective: Player) -> int:
    """
    Returns an integer score from the point of view of perspective player.
    Higher is better for perspective player.
    """
    w = winner(state)
    if w is not None:
        winner_player, _reason = w
        return WIN_SCORE if winner_player == perspective else -WIN_SCORE

    opponent = perspective.opponent()

    # 1) Material: students difference
    my_students = _count_students(state, perspective)
    opp_students = _count_students(state, opponent)
    material_score = (my_students - opp_students) * STUDENT_WEIGHT

    # 2) Master progress: closer to target temple is better
    my_master = _find_master_pos(state, perspective)
    if my_master is None:
        # Should be terminal (loss), but just in case:
        return -WIN_SCORE

    target = _target_temple_for(perspective)
    dist = _manhattan(my_master, target)
    master_progress_score = -dist * MASTER_DISTANCE_WEIGHT

    # 3) Tactical safety: avoid positions where opponent can capture our Master next move
    threat_penalty = -MASTER_THREAT_PENALTY if _opponent_can_capture_master_next(state, perspective) else 0

    # 4) Mobility: prefer having more legal actions than opponent
    my_moves = _count_legal_actions_for(state, perspective)
    opp_moves = _count_legal_actions_for(state, opponent)
    mobility_score = (my_moves - opp_moves) * MOBILITY_WEIGHT
    
    # 5) Hanging students: penalize having students that can be captured next move
    hanging_students = _count_hanging_students_next(state, perspective)
    hanging_penalty = -hanging_students * HANGING_STUDENT_PENALTY

    return material_score + master_progress_score + threat_penalty + mobility_score + hanging_penalty


# ----------------------------------------------------------------------------

def evaluate_v2_generic(state: GameState, perspective: Player, w2: EvalWeightsV2) -> int:
    """
    Generic V2 heuristic driven by weights.
    """
    w = winner(state)
    if w is not None:
        winner_player, _reason = w
        return WIN_SCORE if winner_player == perspective else -WIN_SCORE

    opponent = perspective.opponent()

    # Actions (as if each player were to move now)
    my_actions = _actions_for(state, perspective)
    opp_actions = _actions_for(state, opponent)
    my_moves = _move_actions(my_actions)
    opp_moves = _move_actions(opp_actions)

    # 1) Material
    my_students = _count_students(state, perspective)
    opp_students = _count_students(state, opponent)
    material_score = (my_students - opp_students) * w2.student_weight

    # 2) Master progress (relative)
    my_master = _find_master_pos(state, perspective)
    opp_master = _find_master_pos(state, opponent)
    if my_master is None:
        return -WIN_SCORE
    if opp_master is None:
        return WIN_SCORE

    my_dist = _manhattan(my_master, _target_temple_for(perspective))
    opp_dist = _manhattan(opp_master, _target_temple_for(opponent))
    master_progress_score = (opp_dist - my_dist) * w2.master_distance_weight

    # 3) Master threats (symmetric 1-ply)
    threat_penalty = -w2.master_threat_penalty if any(m.to_pos == my_master for m in opp_moves) else 0
    threat_bonus = w2.master_threat_bonus if any(m.to_pos == opp_master for m in my_moves) else 0

    # 4) Hanging students (unique threatened pieces)
    my_students_pos = set(_iter_positions(state, perspective, PieceType.STUDENT))
    opp_students_pos = set(_iter_positions(state, opponent, PieceType.STUDENT))

    my_attack_squares = {m.to_pos for m in my_moves}
    opp_attack_squares = {m.to_pos for m in opp_moves}

    my_hanging = len(my_students_pos & opp_attack_squares)
    opp_hanging = len(opp_students_pos & my_attack_squares)

    hanging_penalty = -my_hanging * w2.hanging_student_penalty
    hanging_bonus = opp_hanging * w2.hanging_student_bonus

    # 5) Mobility (Moves only)
    mobility_score = (len(my_moves) - len(opp_moves)) * w2.mobility_weight

    # 6) Positional: centrality + advancement (students only)
    positional_score = 0
    for pos in my_students_pos:
        positional_score += _centrality_value(pos) * w2.centrality_weight
        positional_score += _advancement_value(pos, perspective) * w2.advancement_weight
    for pos in opp_students_pos:
        positional_score -= _centrality_value(pos) * w2.centrality_weight
        positional_score -= _advancement_value(pos, opponent) * w2.advancement_weight

    return (
        material_score
        + master_progress_score
        + threat_penalty
        + threat_bonus
        + hanging_penalty
        + hanging_bonus
        + mobility_score
        + positional_score
    )


def evaluate_v2a(state: GameState, perspective: Player) -> int:
    return evaluate_v2_generic(state, perspective, V2A)


def evaluate_v2b(state: GameState, perspective: Player) -> int:
    return evaluate_v2_generic(state, perspective, V2B)


def evaluate_v2c(state: GameState, perspective: Player) -> int:
    return evaluate_v2_generic(state, perspective, V2C)


# ----------------------------------------------------------------------------

EVALUATORS: dict[str, Evaluator] = {
    "v1": evaluate_v1,
    "v2a": evaluate_v2a,
    "v2b": evaluate_v2b,
    "v2c": evaluate_v2c,
}

def get_evaluator(name: str) -> Evaluator:
    try:
        return EVALUATORS[name]
    except KeyError as e:
        raise ValueError(f"Unknown evaluator '{name}'. Available: {sorted(EVALUATORS)}") from e
    
