# ai/evaluate.py

from __future__ import annotations

from dataclasses import replace, dataclass

from onitama.engine.moves import Move
from onitama.engine.pieces import PieceType, Player, Piece
from onitama.engine.rules import generate_legal_actions, winner
from onitama.engine.state import GameState

from onitama.ai.types import Evaluator


WIN_SCORE = 100_000

# ----------------------------------------------------------------------------
# V1 Weights
STUDENT_WEIGHT = 300
MASTER_DISTANCE_WEIGHT = 8

MASTER_THREAT_PENALTY = 8_000
HANGING_STUDENT_PENALTY = 600

# ----------------------------------------------------------------------------
# V2: generic heuristic with tunable weights

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

V2 = EvalWeightsV2(
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


# ----------------------------------------------------------------------------
# V3

# Simple PSTs (defined from RED perspective).
STUDENT_PST: list[list[int]] = [
    [4, 5, 6, 5, 4],
    [3, 4, 5, 4, 3],
    [2, 3, 4, 3, 2],
    [1, 2, 3, 2, 1],
    [0, 1, 2, 1, 0],
]

MASTER_PST: list[list[int]] = [
    [0, 1, 2, 1, 0],
    [1, 2, 3, 2, 1],
    [1, 2, 4, 2, 1],
    [1, 2, 3, 2, 1],
    [0, 1, 2, 1, 0],
]

@dataclass(frozen=True)
class EvalWeightsV3:
    student_weight: int
    master_distance_weight_opening: int
    master_distance_weight_endgame: int
    master_threat_penalty: int
    master_threat_bonus: int
    hanging_student_penalty: int
    hanging_student_bonus: int
    mobility_weight_opening: int
    mobility_weight_endgame: int
    pst_weight_opening: int
    pst_weight_endgame: int
    

V3 = EvalWeightsV3(
    student_weight=300,
    master_distance_weight_opening=10,
    master_distance_weight_endgame=25,
    master_threat_penalty=8_500,
    master_threat_bonus=5_500,
    hanging_student_penalty=600,
    hanging_student_bonus=400,
    mobility_weight_opening=12,
    mobility_weight_endgame=4,
    pst_weight_opening=90,
    pst_weight_endgame=25,
)


# ----------------------------------------------------------------------------
# Helpers V1 (Shared helpers)

def _target_temple_for(player: Player) -> tuple[int, int]:
    """
    Temple square the player wants to reach with their own Master
    to win by Way of the Stream.
    """
    return (0, 2) if player == Player.RED else (4, 2)


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _find_master_pos(state: GameState, owner: Player) -> tuple[int, int] | None:
    return state.red_master_pos if owner is Player.RED else state.blue_master_pos


# ----------------------------------------------------------------------------
# Helpers V2

CENTRALITY_TABLE: tuple[tuple[int, ...], ...] = tuple(
    tuple(4 - (abs(r - 2) + abs(c - 2)) for c in range(5))
    for r in range(5)
)

ADVANCEMENT_RED_TABLE: tuple[tuple[int, ...], ...] = tuple(
    tuple(4 - r for _c in range(5))
    for r in range(5)
)

ADVANCEMENT_BLUE_TABLE: tuple[tuple[int, ...], ...] = tuple(
    tuple(r for _c in range(5))
    for r in range(5)
)


# ----------------------------------------------------------------------------
# Helpers V3

def _rotate_180(pos: tuple[int, int]) -> tuple[int, int]:
    r, c = pos
    return (4 - r, 4 - c)


def _pst_value(table: list[list[int]], pos: tuple[int, int], owner: Player) -> int:
    """
    PSTs are defined from RED's perspective (row 0 is RED-forward).
    For BLUE pieces we rotate 180 degrees.
    """
    r, c = pos if owner == Player.RED else _rotate_180(pos)
    return table[r][c]


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
    board = state.board

    # 1) Material: students difference (single board scan).
    my_students = 0
    opp_students = 0
    my_students_pos: set[tuple[int, int]] = set() # used in step 5
    for r in range(5):
        for c in range(5):
            piece = board[r][c]
            if piece is None or piece.kind != PieceType.STUDENT:
                continue
            if piece.owner == perspective:
                my_students += 1
                my_students_pos.add((r, c))
            else:
                opp_students += 1
    material_score = (my_students - opp_students) * STUDENT_WEIGHT

    # 2) Master progress (relative): better if we are closer than opponent.
    my_master = _find_master_pos(state, perspective)
    opp_master = _find_master_pos(state, opponent)
    if my_master is None:
        # Should be terminal (loss), but just in case:
        return -WIN_SCORE
    if opp_master is None:
        # Should be terminal (win), but just in case:
        return WIN_SCORE

    my_target = _target_temple_for(perspective)
    opp_target = _target_temple_for(opponent)
    my_dist = _manhattan(my_master, my_target)
    opp_dist = _manhattan(opp_master, opp_target)
    master_progress_score = (opp_dist - my_dist) * MASTER_DISTANCE_WEIGHT

    # Build opponent actions once and reuse for both tactical terms.
    opp_actions = generate_legal_actions(replace(state, to_move=opponent))

    # 3) Tactical safety: avoid positions where opponent can capture our Master next move
    opp_can_capture_master = any(isinstance(a, Move) and a.to_pos == my_master for a in opp_actions)
    threat_penalty = -MASTER_THREAT_PENALTY if opp_can_capture_master else 0

    # 4) Hanging students: penalize own students capturable next move.
    opp_attack_squares: set[tuple[int, int]] = set()
    for a in opp_actions:
        if not isinstance(a, Move):
            continue
        opp_attack_squares.add(a.to_pos)
    hanging_students = len(my_students_pos & opp_attack_squares)
    hanging_penalty = -hanging_students * HANGING_STUDENT_PENALTY

    return material_score + master_progress_score + threat_penalty + hanging_penalty


# ----------------------------------------------------------------------------
# Differences in V2 compared to V1:
# - More tunable weights, including for new features.
# - New features:
#   - Master threat bonus: reward positions where we can capture opponent's Master next move.
#   - Hanging students counted as unique threatened pieces (set intersection, no multiplicity).
#   - Positional: centrality + advancement (students only, Master is all-or-nothing).

def evaluate_v2_generic(state: GameState, perspective: Player, w2: EvalWeightsV2) -> int:
    """
    Generic V2 heuristic driven by weights.
    """
    w = winner(state)
    if w is not None:
        winner_player, _reason = w
        return WIN_SCORE if winner_player == perspective else -WIN_SCORE

    opponent = perspective.opponent()
    board = state.board

    # Actions (as if each player were to move now)
    my_actions = generate_legal_actions(replace(state, to_move=perspective))
    opp_actions = generate_legal_actions(replace(state, to_move=opponent))

    # 1) Material + student positions (single board scan)
    my_students = 0
    opp_students = 0
    my_students_pos: set[tuple[int, int]] = set()
    opp_students_pos: set[tuple[int, int]] = set()
    for r in range(5):
        for c in range(5):
            p = board[r][c]
            if p is None or p.kind != PieceType.STUDENT:
                continue
            pos = (r, c)
            if p.owner == perspective:
                my_students += 1
                my_students_pos.add(pos)
            else:
                opp_students += 1
                opp_students_pos.add(pos)
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

    # Build move-derived features in one pass per player.
    my_move_count = 0
    opp_move_count = 0
    my_attack_squares: set[tuple[int, int]] = set()
    opp_attack_squares: set[tuple[int, int]] = set()

    for a in my_actions:
        if not isinstance(a, Move):
            continue
        my_move_count += 1
        my_attack_squares.add(a.to_pos)

    for a in opp_actions:
        if not isinstance(a, Move):
            continue
        opp_move_count += 1
        opp_attack_squares.add(a.to_pos)

    # 3) Master threats (symmetric 1-ply)
    threat_score = 0
    if my_master in opp_attack_squares:
        threat_score -= w2.master_threat_penalty
    if opp_master in my_attack_squares:
        threat_score += w2.master_threat_bonus
        
    # 4) Hanging students (unique threatened pieces)
    my_hanging = len(my_students_pos & opp_attack_squares)
    opp_hanging = len(opp_students_pos & my_attack_squares)

    hanging_score = (
        -my_hanging * w2.hanging_student_penalty
        + opp_hanging * w2.hanging_student_bonus
    )

    # 5) Mobility (Moves only)
    mobility_score = (my_move_count - opp_move_count) * w2.mobility_weight

    # 6) Positional: centrality + advancement (students only)
    positional_score = 0
    for pos in my_students_pos:
        r, c = pos
        positional_score += CENTRALITY_TABLE[r][c] * w2.centrality_weight
        if perspective == Player.RED:
            positional_score += ADVANCEMENT_RED_TABLE[r][c] * w2.advancement_weight
        else:
            positional_score += ADVANCEMENT_BLUE_TABLE[r][c] * w2.advancement_weight
    for pos in opp_students_pos:
        r, c = pos
        positional_score -= CENTRALITY_TABLE[r][c] * w2.centrality_weight
        if opponent == Player.RED:
            positional_score -= ADVANCEMENT_RED_TABLE[r][c] * w2.advancement_weight
        else:
            positional_score -= ADVANCEMENT_BLUE_TABLE[r][c] * w2.advancement_weight

    return (
        material_score
        + master_progress_score
        + threat_score
        + hanging_score
        + mobility_score
        + positional_score
    )


def evaluate_v2(state: GameState, perspective: Player) -> int:
    return evaluate_v2_generic(state, perspective, V2)


# ----------------------------------------------------------------------------
# Differences in V3 compared to V2:
# - Endgame-aware blending for key weights (master progress, mobility, PST).
# - Hanging students uses unique threatened pieces (same criterion as V2).
# - Mobility includes phase-dependent weights.
# - Positional model upgraded from centrality/advancement to PST (students + master).

def evaluate_v3_generic(
    state: GameState,
    perspective: Player,
    w3: EvalWeightsV3,
) -> int:
    """
    Generic V3 heuristic driven by weights.
    """
    w = winner(state)
    if w is not None:
        winner_player, _reason = w
        return WIN_SCORE if winner_player == perspective else -WIN_SCORE

    opponent = perspective.opponent()
    
    my_actions = generate_legal_actions(replace(state, to_move=perspective))
    opp_actions = generate_legal_actions(replace(state, to_move=opponent))
    
    # Single board scan to collect piece list and student counts.
    my_students = 0
    opp_students = 0
    my_students_pos: set[tuple[int, int]] = set()
    opp_students_pos: set[tuple[int, int]] = set()
    pieces: list[tuple[int, int, Piece]] = []

    board = state.board
    for r in range(5):
        for c in range(5):
            p = board[r][c]
            if p is None:
                continue
            pieces.append((r, c, p))
            if p.owner == perspective:
                if p.kind == PieceType.STUDENT:
                    my_students += 1
                    my_students_pos.add((r, c))
            else:
                if p.kind == PieceType.STUDENT:
                    opp_students += 1
                    opp_students_pos.add((r, c))

    total_students = my_students + opp_students
    endgame = 1.0 - (total_students / 8.0)  # 0: opening, 1: endgame

    # 1) Material
    material_score = (my_students - opp_students) * w3.student_weight

    # 2) Master progress (relative, phase-blended)
    my_master = _find_master_pos(state, perspective)
    opp_master = _find_master_pos(state, opponent)
    if my_master is None:
        return -WIN_SCORE
    if opp_master is None:
        return WIN_SCORE

    my_dist = _manhattan(my_master, _target_temple_for(perspective))
    opp_dist = _manhattan(opp_master, _target_temple_for(opponent))

    master_progress_weight = int(
        w3.master_distance_weight_opening * (1.0 - endgame)
        + w3.master_distance_weight_endgame * endgame
    )
    master_progress_score = (opp_dist - my_dist) * master_progress_weight

    # Build move-derived tactical features in one pass per player.
    my_move_count = 0
    opp_move_count = 0

    my_attack_squares: set[tuple[int, int]] = set()
    opp_attack_squares: set[tuple[int, int]] = set()

    for a in opp_actions:
        if not isinstance(a, Move):
            continue
        opp_move_count += 1
        tr, tc = a.to_pos
        opp_attack_squares.add((tr, tc))

    for a in my_actions:
        if not isinstance(a, Move):
            continue
        my_move_count += 1
        tr, tc = a.to_pos
        my_attack_squares.add((tr, tc))

    # 3) Master threats (symmetric 1-ply)
    threat_score = 0
    if my_master in opp_attack_squares:
        threat_score -= w3.master_threat_penalty
    if opp_master in my_attack_squares:
        threat_score += w3.master_threat_bonus

    # 4) Hanging students (unique threatened pieces, as in V2)
    my_hanging = len(my_students_pos & opp_attack_squares)
    opp_hanging = len(opp_students_pos & my_attack_squares)

    hanging_score = (
        -my_hanging * w3.hanging_student_penalty
        + opp_hanging * w3.hanging_student_bonus
    )
    
    # 5) Mobility (move-count difference, phase-blended)
    mobility_weight = int(
        w3.mobility_weight_opening * (1.0 - endgame)
        + w3.mobility_weight_endgame * endgame
    )
    mobility_score = (my_move_count - opp_move_count) * mobility_weight

    # 6) Positional: PST (students + master)
    pst_weight = int(
        w3.pst_weight_opening * (1.0 - endgame)
        + w3.pst_weight_endgame * endgame
    )

    pst_score = 0
    for r, c, p in pieces:
        pos = (r, c)
        if p.kind == PieceType.STUDENT:
            v = _pst_value(STUDENT_PST, pos, p.owner)
        else:
            v = _pst_value(MASTER_PST, pos, p.owner)

        if p.owner == perspective:
            pst_score += v * pst_weight
        else:
            pst_score -= v * pst_weight


    return material_score + master_progress_score + threat_score + hanging_score + mobility_score + pst_score


def evaluate_v3(state: GameState, perspective: Player) -> int:
    return evaluate_v3_generic(state, perspective, V3)


# ----------------------------------------------------------------------------

EVALUATORS: dict[str, Evaluator] = {
    "v1": evaluate_v1,
    "v2": evaluate_v2,
    "v3": evaluate_v3,
}

def get_evaluator(name: str) -> Evaluator:
    try:
        return EVALUATORS[name]
    except KeyError as e:
        raise ValueError(f"Unknown evaluator '{name}'. Available: {sorted(EVALUATORS)}") from e
