from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


BOARD_SIZE = 5


class VisionPiece(str, Enum):
    RED_MASTER = "red_master"
    RED_STUDENT = "red_student"
    BLUE_MASTER = "blue_master"
    BLUE_STUDENT = "blue_student"

    @property
    def short(self) -> str:
        if self is VisionPiece.RED_MASTER:
            return "RM"
        if self is VisionPiece.RED_STUDENT:
            return "RS"
        if self is VisionPiece.BLUE_MASTER:
            return "BM"
        return "BS"


CellValue = Optional[VisionPiece]
VisionBoardGrid = tuple[tuple[CellValue, ...], ...]


def _parse_cell_value(value: CellValue | str) -> CellValue:
    if value is None:
        return None
    if isinstance(value, VisionPiece):
        return value
    if isinstance(value, str):
        return VisionPiece(value)
    raise TypeError("Cell value must be None, VisionPiece, or str.")


def _normalize_board(board: VisionBoardGrid | list[list[CellValue | str]]) -> VisionBoardGrid:
    if len(board) != BOARD_SIZE:
        raise ValueError(f"Board must have {BOARD_SIZE} rows.")

    rows: list[tuple[CellValue, ...]] = []
    for row in board:
        if len(row) != BOARD_SIZE:
            raise ValueError(f"Each row must have {BOARD_SIZE} columns.")
        rows.append(tuple(_parse_cell_value(cell) for cell in row))

    return tuple(rows)


@dataclass(frozen=True)
class VisionBoard:
    """
    Discrete 5x5 board observation aligned with engine coordinates:
    - row 0 is top (BLUE side in canonical warp)
    - row 4 is bottom (RED side in canonical warp)
    """

    board: VisionBoardGrid

    def __post_init__(self) -> None:
        object.__setattr__(self, "board", _normalize_board(self.board))

    @classmethod
    def empty(cls) -> "VisionBoard":
        return cls(
            board=tuple(tuple(None for _ in range(BOARD_SIZE)) for _ in range(BOARD_SIZE))
        )

    @classmethod
    def from_board_tokens(cls, board: list[list[str | None]]) -> "VisionBoard":
        return cls(board=board)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "VisionBoard":
        board = data.get("board")
        if not isinstance(board, list):
            raise ValueError("Invalid vision board dict: 'board' must be a 5x5 list.")
        return cls.from_board_tokens(board=board)

    @classmethod
    def load_json(cls, path: str | Path) -> "VisionBoard":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Invalid vision board JSON: root must be an object.")
        return cls.from_dict(data)

    
    def with_cell(self, row: int, col: int, value: CellValue | str) -> "VisionBoard":
        """
        Create a new VisionBoard with a modified cell value at the specified position.
        """
        if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
            raise ValueError(f"Cell ({row}, {col}) out of range [0..{BOARD_SIZE - 1}].")

        parsed = _parse_cell_value(value)
        rows = [list(r) for r in self.board]
        rows[row][col] = parsed
        return type(self)(board=tuple(tuple(r) for r in rows))


    def to_board_tokens(self) -> list[list[str | None]]:
        return [
            [None if cell is None else cell.value for cell in row]
            for row in self.board
        ]

    def pretty(self) -> str:
        lines: list[str] = []
        for row in self.board:
            row_tokens = [".." if cell is None else cell.short for cell in row]
            lines.append(" ".join(row_tokens))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "board_size": BOARD_SIZE,
            "board": self.to_board_tokens(),
        }

    def save_json(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
