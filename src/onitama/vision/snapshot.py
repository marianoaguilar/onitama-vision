from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from onitama.engine.cards import CARD_BY_NAME, Card
from onitama.engine.pieces import Piece, PieceType, Player
from onitama.engine.state import Board, GameState
from onitama.errors import VisionObservationError, VisionObservationKind, VisionPipelineError
from onitama.vision.board import VisionBoard, VisionPiece
from onitama.vision.card_classifier import CardClassificationResult


def _raise_invalid_board(debug_message: str) -> None:
    raise VisionObservationError(VisionObservationKind.INVALID_BOARD_PIECE_COUNT, debug_message=debug_message)


def _parse_player(value: Player | str) -> Player:
    """Normalize a player value coming from code or JSON."""
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
    """Validate a pair of card names."""
    if len(value) != 2:
        raise ValueError(f"{field_name} must contain exactly 2 card names.")
    first = str(value[0]).strip()
    second = str(value[1]).strip()
    if not first or not second:
        raise ValueError(f"{field_name} contains empty card names.")
    return (first, second)


def _vision_piece_to_engine_piece(piece: VisionPiece | None) -> Piece | None:
    """Convert one visual piece to the engine piece type."""
    if piece is None:
        return None

    if piece is VisionPiece.RED_MASTER:
        return Piece(owner=Player.RED, kind=PieceType.MASTER)
    if piece is VisionPiece.RED_STUDENT:
        return Piece(owner=Player.RED, kind=PieceType.STUDENT)
    if piece is VisionPiece.BLUE_MASTER:
        return Piece(owner=Player.BLUE, kind=PieceType.MASTER)
    return Piece(owner=Player.BLUE, kind=PieceType.STUDENT)


def _vision_board_to_engine_board(board: VisionBoard) -> Board:
    """Convert the visual board into the engine board."""
    return tuple(
        tuple(_vision_piece_to_engine_piece(cell) for cell in row)
        for row in board.board
    )


def _resolve_card(card_name: str) -> Card:
    """Resolve a card name to the engine Card object."""
    resolved = CARD_BY_NAME.get(card_name)
    if resolved is None:
        raise VisionPipelineError(f"Unknown card name from vision pipeline: {card_name!r}")
    return resolved


def _resolve_card_pair(cards: Sequence[str]) -> tuple[Card, Card]:
    """Resolve a pair of card names to engine cards."""
    first = _resolve_card(cards[0])
    second = _resolve_card(cards[1])
    return (first, second)


def _validate_piece_counts(board: Board) -> None:
    """Reject boards that exceed Onitama piece limits for either player."""
    red_masters = 0
    red_students = 0
    blue_masters = 0
    blue_students = 0

    for row in board:
        for piece in row:
            if piece is None:
                continue

            if piece.owner is Player.RED:
                if piece.kind is PieceType.MASTER:
                    red_masters += 1
                else:
                    red_students += 1
            else:
                if piece.kind is PieceType.MASTER:
                    blue_masters += 1
                else:
                    blue_students += 1

    invalid_limits = (
        (red_masters > 1, "Invalid board: more than one RED master detected."),
        (blue_masters > 1, "Invalid board: more than one BLUE master detected."),
        (red_students > 4, "Invalid board: more than four RED students detected."),
        (blue_students > 4, "Invalid board: more than four BLUE students detected."),
        (red_masters + red_students > 5, "Invalid board: more than five RED pieces detected."),
        (blue_masters + blue_students > 5, "Invalid board: more than five BLUE pieces detected."),
    )
    for condition, debug_message in invalid_limits:
        if condition:
            _raise_invalid_board(debug_message)


@dataclass(frozen=True)
class VisionSnapshot:
    """
    Minimal visual observation of the board and cards.
    """

    board: VisionBoard
    red_cards: tuple[str, str]
    blue_cards: tuple[str, str]
    side_card: str

    def __post_init__(self) -> None:
        """Normalize card fields after construction."""
        object.__setattr__(self, "red_cards", _parse_card_pair(self.red_cards, "red_cards"))
        object.__setattr__(self, "blue_cards", _parse_card_pair(self.blue_cards, "blue_cards"))

        side = str(self.side_card).strip()
        if not side:
            raise ValueError("side_card cannot be empty.")
        object.__setattr__(self, "side_card", side)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "VisionSnapshot":
        board = data.get("board")
        if isinstance(board, list):
            parsed_board = VisionBoard.from_board_tokens(board=board)
        elif isinstance(board, dict):
            parsed_board = VisionBoard.from_dict(board)
        else:
            raise ValueError("Invalid snapshot: 'board' must be a 5x5 list or a board object.")

        red_cards = data.get("red_cards")
        blue_cards = data.get("blue_cards")
        side_card = data.get("side_card")

        if not isinstance(red_cards, (list, tuple)):
            raise ValueError("Invalid snapshot: 'red_cards' must be a 2-item list.")
        if not isinstance(blue_cards, (list, tuple)):
            raise ValueError("Invalid snapshot: 'blue_cards' must be a 2-item list.")
        if not isinstance(side_card, str):
            raise ValueError("Invalid snapshot: 'side_card' must be a string.")

        return cls(
            board=parsed_board,
            red_cards=_parse_card_pair(red_cards, "red_cards"),
            blue_cards=_parse_card_pair(blue_cards, "blue_cards"),
            side_card=side_card,
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "VisionSnapshot":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Invalid snapshot JSON: root must be an object.")
        return cls.from_dict(data)

    @classmethod
    def from_board_and_cards(
        cls,
        *,
        board: VisionBoard,
        card_result: CardClassificationResult,
    ) -> "VisionSnapshot":
        """Build a snapshot from an inferred board and inferred cards."""
        red_cards, blue_cards, side_card = card_result.cards_layout()
        return cls(
            board=board,
            red_cards=red_cards,
            blue_cards=blue_cards,
            side_card=side_card,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "board": self.board.to_board_tokens(),
            "red_cards": [self.red_cards[0], self.red_cards[1]],
            "blue_cards": [self.blue_cards[0], self.blue_cards[1]],
            "side_card": self.side_card,
        }

    def save_json(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    # Main conversion method
    def to_game_state(self, to_move: Player | str) -> GameState:
        """Convert the visual snapshot into an engine GameState."""
        
        engine_board = _vision_board_to_engine_board(self.board)

        # Validate the board does not exceed Onitama piece limits.
        _validate_piece_counts(engine_board)

        red_pair = _resolve_card_pair(self.red_cards)
        blue_pair = _resolve_card_pair(self.blue_cards)
        side = _resolve_card(self.side_card)

        # Enforced unique cards
        used = [red_pair[0].name, red_pair[1].name, blue_pair[0].name, blue_pair[1].name, side.name]
        if len(set(used)) != 5:
            raise VisionObservationError(
                VisionObservationKind.LOW_CONFIDENCE_CARD,
                debug_message="Cards must be 5 unique cards across red, blue and side.",
            )

        return GameState(
            board=engine_board,
            to_move=_parse_player(to_move),
            red_cards=red_pair,
            blue_cards=blue_pair,
            side_card=side,
        )
