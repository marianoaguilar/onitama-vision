import pytest

from onitama.engine.pieces import PieceType, Player
from onitama.vision.board import VisionBoard, VisionPiece
from onitama.vision.snapshot import VisionSnapshot


def test_snapshot_to_game_state_with_card_names() -> None:
    board = VisionBoard.empty()
    board = board.with_cell(0, 2, VisionPiece.BLUE_MASTER)
    board = board.with_cell(4, 2, VisionPiece.RED_MASTER)
    board = board.with_cell(4, 0, VisionPiece.RED_STUDENT)

    snapshot = VisionSnapshot(
        board=board,
        red_cards=("Tiger", "Horse"),
        blue_cards=("Crab", "Boar"),
        side_card="Rabbit",
    )
    game_state = snapshot.to_game_state(to_move="RED")

    assert game_state.to_move is Player.RED
    assert game_state.red_cards[0].name == "Tiger"
    assert game_state.blue_cards[1].name == "Boar"
    assert game_state.side_card.name == "Rabbit"
    assert game_state.board[0][2] is not None
    assert game_state.board[0][2].owner is Player.BLUE
    assert game_state.board[0][2].kind is PieceType.MASTER


def test_snapshot_to_game_state_rejects_duplicate_cards() -> None:
    snapshot = VisionSnapshot(
        board=VisionBoard.empty(),
        red_cards=("Tiger", "Horse"),
        blue_cards=("Crab", "Boar"),
        side_card="Tiger",
    )
    with pytest.raises(ValueError, match="Cards must be 5 unique cards"):
        snapshot.to_game_state(to_move=Player.RED)


def test_snapshot_to_game_state_rejects_multiple_red_masters() -> None:
    board = VisionBoard.empty()
    board = board.with_cell(4, 2, VisionPiece.RED_MASTER)
    board = board.with_cell(3, 2, VisionPiece.RED_MASTER)
    snapshot = VisionSnapshot(
        board=board,
        red_cards=("Tiger", "Horse"),
        blue_cards=("Crab", "Boar"),
        side_card="Rabbit",
    )

    with pytest.raises(ValueError, match="more than one RED master"):
        snapshot.to_game_state(to_move=Player.RED)


def test_snapshot_to_game_state_rejects_too_many_red_students() -> None:
    board = VisionBoard.empty()
    for col in range(5):
        board = board.with_cell(4, col, VisionPiece.RED_STUDENT)
    snapshot = VisionSnapshot(
        board=board,
        red_cards=("Tiger", "Horse"),
        blue_cards=("Crab", "Boar"),
        side_card="Rabbit",
    )

    with pytest.raises(ValueError, match="more than four RED students"):
        snapshot.to_game_state(to_move=Player.RED)


def test_snapshot_to_game_state_rejects_too_many_red_pieces() -> None:
    board = VisionBoard.empty()
    board = board.with_cell(4, 2, VisionPiece.RED_MASTER)
    for row, col in ((4, 0), (4, 1), (4, 3), (4, 4), (3, 2)):
        board = board.with_cell(row, col, VisionPiece.RED_STUDENT)
    snapshot = VisionSnapshot(
        board=board,
        red_cards=("Tiger", "Horse"),
        blue_cards=("Crab", "Boar"),
        side_card="Rabbit",
    )

    with pytest.raises(ValueError, match="more than four RED students|more than five RED pieces"):
        snapshot.to_game_state(to_move=Player.RED)
