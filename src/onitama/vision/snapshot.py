from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from onitama.engine.pieces import Player
from onitama.vision.board import VisionBoard


def _parse_player(value: Player | str) -> Player:
    if isinstance(value, Player):
        return value
    if isinstance(value, str):
        upper = value.strip().upper()
        if upper == "RED":
            return Player.RED
        if upper == "BLUE":
            return Player.BLUE
    raise ValueError("to_move must be Player.RED, Player.BLUE, 'RED' or 'BLUE'.")


def _parse_card_pair(value: tuple[str, str] | list[str], field_name: str) -> tuple[str, str]:
    if len(value) != 2:
        raise ValueError(f"{field_name} must contain exactly 2 card names.")
    first = str(value[0]).strip()
    second = str(value[1]).strip()
    if not first or not second:
        raise ValueError(f"{field_name} contains empty card names.")
    return (first, second)


@dataclass(frozen=True)
class VisionSnapshot:
    """
    Minimal vision payload to reconstruct an engine GameState.
    """

    board: VisionBoard
    to_move: Player
    red_cards: tuple[str, str]
    blue_cards: tuple[str, str]
    side_card: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "to_move", _parse_player(self.to_move))
        object.__setattr__(self, "red_cards", _parse_card_pair(self.red_cards, "red_cards"))
        object.__setattr__(self, "blue_cards", _parse_card_pair(self.blue_cards, "blue_cards"))

        side = str(self.side_card).strip()
        if not side:
            raise ValueError("side_card cannot be empty.")
        object.__setattr__(self, "side_card", side)

    @staticmethod
    def from_dict(data: dict[str, object]) -> "VisionSnapshot":
        board = data.get("board")
        if isinstance(board, list):
            parsed_board = VisionBoard.from_board_tokens(board=board)
        elif isinstance(board, dict):
            parsed_board = VisionBoard.from_dict(board)
        else:
            raise ValueError("Invalid snapshot: 'board' must be a 5x5 list or a board object.")

        to_move = data.get("to_move")
        red_cards = data.get("red_cards")
        blue_cards = data.get("blue_cards")
        side_card = data.get("side_card")

        if not isinstance(red_cards, (list, tuple)):
            raise ValueError("Invalid snapshot: 'red_cards' must be a 2-item list.")
        if not isinstance(blue_cards, (list, tuple)):
            raise ValueError("Invalid snapshot: 'blue_cards' must be a 2-item list.")
        if not isinstance(side_card, str):
            raise ValueError("Invalid snapshot: 'side_card' must be a string.")

        return VisionSnapshot(
            board=parsed_board,
            to_move=_parse_player(to_move),
            red_cards=_parse_card_pair(red_cards, "red_cards"),
            blue_cards=_parse_card_pair(blue_cards, "blue_cards"),
            side_card=side_card,
        )

    @staticmethod
    def load_json(path: str | Path) -> "VisionSnapshot":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Invalid snapshot JSON: root must be an object.")
        return VisionSnapshot.from_dict(data)


    def to_dict(self) -> dict[str, object]:
        return {
            "board": self.board.to_board_tokens(),
            "to_move": self.to_move.value,
            "red_cards": [self.red_cards[0], self.red_cards[1]],
            "blue_cards": [self.blue_cards[0], self.blue_cards[1]],
            "side_card": self.side_card,
        }

    def save_json(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
