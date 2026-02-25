from __future__ import annotations

from collections.abc import Sequence

from onitama.engine.cards import CARD_BY_NAME, Card
from onitama.engine.pieces import Piece, PieceType, Player
from onitama.engine.state import Board, GameState
from onitama.vision.board import VisionBoard, VisionPiece
from onitama.vision.snapshot import VisionSnapshot


def vision_piece_to_engine_piece(piece: VisionPiece | None) -> Piece | None:
    if piece is None:
        return None

    if piece is VisionPiece.RED_MASTER:
        return Piece(owner=Player.RED, kind=PieceType.MASTER)
    if piece is VisionPiece.RED_STUDENT:
        return Piece(owner=Player.RED, kind=PieceType.STUDENT)
    if piece is VisionPiece.BLUE_MASTER:
        return Piece(owner=Player.BLUE, kind=PieceType.MASTER)
    return Piece(owner=Player.BLUE, kind=PieceType.STUDENT)


def vision_board_to_engine_board(board: VisionBoard) -> Board:
    """
    Convert a discrete vision board (5x5) into engine Board coordinates.
    Coordinates are preserved as-is:
    - row 0 top (BLUE side in canonical warp)
    - row 4 bottom (RED side in canonical warp)
    """
    return tuple(
        tuple(vision_piece_to_engine_piece(cell) for cell in row)
        for row in board.board
    )


def _resolve_card(card_name: str) -> Card:
    resolved = CARD_BY_NAME.get(card_name)
    if resolved is None:
        raise ValueError(f"Unknown card name: {card_name!r}")
    return resolved


def _resolve_card_pair(cards: Sequence[str], field_name: str) -> tuple[Card, Card]:
    if len(cards) != 2:
        raise ValueError(f"{field_name} must contain exactly 2 cards.")
    first = _resolve_card(cards[0])
    second = _resolve_card(cards[1])
    return (first, second)


def _validate_master_count(board: Board) -> None:
    red_masters = 0
    blue_masters = 0
    for row in board:
        for piece in row:
            if piece is None or piece.kind is not PieceType.MASTER:
                continue
            if piece.owner is Player.RED:
                red_masters += 1
            else:
                blue_masters += 1

    if red_masters > 1:
        raise ValueError("Invalid board: more than one RED master detected.")
    if blue_masters > 1:
        raise ValueError("Invalid board: more than one BLUE master detected.")


def snapshot_to_game_state(
    snapshot: VisionSnapshot,
    *,
    enforce_unique_cards: bool = True,
    validate_master_count: bool = True,
) -> GameState:
    """
    Build an engine GameState from a vision snapshot payload.
    """
    engine_board = vision_board_to_engine_board(snapshot.board)
    if validate_master_count:
        _validate_master_count(engine_board)

    red_pair = _resolve_card_pair(snapshot.red_cards, "red_cards")
    blue_pair = _resolve_card_pair(snapshot.blue_cards, "blue_cards")
    side = _resolve_card(snapshot.side_card)

    if enforce_unique_cards:
        used = [red_pair[0].name, red_pair[1].name, blue_pair[0].name, blue_pair[1].name, side.name]
        if len(set(used)) != 5:
            raise ValueError("Cards must be 5 unique cards across red, blue and side.")

    return GameState(
        board=engine_board,
        to_move=snapshot.to_move,
        red_cards=red_pair,
        blue_cards=blue_pair,
        side_card=side,
    )
