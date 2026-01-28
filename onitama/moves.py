from __future__ import annotations

from dataclasses import dataclass

Position = tuple[int, int]  # (row, col)


@dataclass(frozen=True)
class Move:
    from_pos: Position
    to_pos: Position
    card_index: int  # 0 or 1 (which active card was used)


@dataclass(frozen=True)
class Pass:
    """
    Used when the player has no legal piece moves.
    The player must still choose which card to rotate/swap (0 or 1).
    """
    card_index: int  # 0 or 1