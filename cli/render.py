from __future__ import annotations

from onitama.pieces import PieceType
from onitama.state import GameState


def _cell_to_token(piece) -> str:
    if piece is None:
        return "."
    if piece.kind is PieceType.MASTER:
        return "M" if piece.owner.value == "RED" else "m"
    return "R" if piece.owner.value == "RED" else "b"


def pos_to_coord(pos: tuple[int, int]) -> str:
    """Convert (row, col) into chess-like coordinates a1..e5."""
    r, c = pos
    file_char = chr(ord("a") + c)   # 0->a, 1->b, ...
    rank_num = 5 - r                # row 4->1, row 0->5
    return f"{file_char}{rank_num}"



def render_state(state: GameState) -> str:
    """
    Render the current state to a printable string.

    Board legend:
    - . = empty
    - M = RED master, m = BLUE master
    - R = RED student, b = BLUE student
    """
    lines: list[str] = []
    lines.append(f"To move: {state.to_move.value}")
    lines.append(f"RED cards:  {state.red_cards[0].name}, {state.red_cards[1].name}")
    lines.append(f"BLUE cards: {state.blue_cards[0].name}, {state.blue_cards[1].name}")
    lines.append(f"SIDE card:  {state.side_card.name}")
    lines.append("")
    lines.append("  a b c d e")
    for r in range(5):
        row_tokens = [_cell_to_token(state.board[r][c]) for c in range(5)]
        lines.append(f"{5 - r} " + " ".join(row_tokens))
    return "\n".join(lines)
