from __future__ import annotations

from onitama.pieces import PieceType, Player
from onitama.state import GameState
from onitama.moves import Move


def _cell_to_token(piece) -> str:
    if piece is None:
        return "-"
    if piece.owner.value == "RED":
        return "R" if piece.kind is PieceType.MASTER else "r"
    return "B" if piece.kind is PieceType.MASTER else "b"


def pos_to_coord(pos: tuple[int, int]) -> str:
    """Convert (row, col) into chess-like coordinates a1..e5."""
    r, c = pos
    file_char = chr(ord("a") + c)   # 0->a, 1->b, ...
    rank_num = 5 - r                # row 4->1, row 0->5
    return f"{file_char}{rank_num}"

def format_move(state: GameState, mv: Move) -> str:
    card = state.red_cards[mv.card_index] if state.to_move is Player.RED else state.blue_cards[mv.card_index]
    return f"{pos_to_coord(mv.from_pos)} -> {pos_to_coord(mv.to_pos)} ({card.name})"


def render_state(state: GameState) -> str:
    """
    Render the current state to a printable string.

    Board legend:
      - = empty
      R = RED master, r = RED student
      B = BLUE master, b = BLUE student
    """
    lines: list[str] = []
    lines.append(f"To move: {state.to_move.value}")
    
    def _fmt_card(card) -> str:
        return f"{card.name}({card.stamp.value})"
    
    lines.append(f"RED cards:  {_fmt_card(state.red_cards[0])}, {_fmt_card(state.red_cards[1])}")
    lines.append(f"BLUE cards: {_fmt_card(state.blue_cards[0])}, {_fmt_card(state.blue_cards[1])}")
    lines.append(f"SIDE card:  {_fmt_card(state.side_card)}")
    
    lines.append("")
    lines.append("    a  b  c  d  e")
    lines.append("")

    for r in range(5):
        row_tokens = [_cell_to_token(state.board[r][c]) for c in range(5)]
        lines.append(f"{5 - r}   " + "  ".join(row_tokens))
    return "\n".join(lines)
