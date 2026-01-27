from __future__ import annotations

from onitama.pieces import PieceType
from onitama.state import GameState


def _cell_to_token(piece) -> str:
    if piece is None:
        return "."
    if piece.kind is PieceType.MASTER:
        return "M" if piece.owner.value == "RED" else "m"
    return "R" if piece.owner.value == "RED" else "b"


def render_board(state: GameState) -> str:
    """
    Convert the 5x5 board into a printable string.

    Legend:
    - . = empty
    - M = RED master, m = BLUE master
    - R = RED student, b = BLUE student
    """
    lines: list[str] = []
    lines.append(f"To move: {state.to_move.value}")
    lines.append("  a b c d e")
    for r in range(5):
        row_tokens = [_cell_to_token(state.board[r][c]) for c in range(5)]
        lines.append(f"{5 - r} " + " ".join(row_tokens))
    return "\n".join(lines)
