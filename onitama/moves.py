from __future__ import annotations

from dataclasses import dataclass

Position = tuple[int, int]  # (row, col)


@dataclass(frozen=True)
class Move:
    from_pos: Position
    to_pos: Position
    card_index: int  # 0 or 1 (which active card was used)
