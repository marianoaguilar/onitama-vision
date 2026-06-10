import pytest

from onitama.engine.pieces import PieceType, Player
from onitama.vision.board import BOARD_SIZE, VisionBoard, VisionPiece
from onitama.vision.card_classifier import CardClassificationResult, CardSlotPrediction
from onitama.vision.snapshot import VisionSnapshot


def _snapshot(
    board: VisionBoard | None = None,
    *,
    red_cards: tuple[str, str] = ("Tiger", "Horse"),
    blue_cards: tuple[str, str] = ("Crab", "Boar"),
    side_card: str = "Rabbit",
) -> VisionSnapshot:
    return VisionSnapshot(
        board=VisionBoard.empty() if board is None else board,
        red_cards=red_cards,
        blue_cards=blue_cards,
        side_card=side_card,
    )


def test_empty_board_has_5x5_none() -> None:
    board = VisionBoard.empty()

    assert len(board.board) == BOARD_SIZE
    assert all(len(row) == BOARD_SIZE for row in board.board)
    assert all(cell is None for row in board.board for cell in row)


def test_board_with_cell_accepts_enum_and_string() -> None:
    board = VisionBoard.empty()
    board = board.with_cell(0, 0, VisionPiece.BLUE_MASTER)
    board = board.with_cell(4, 2, "red_master")

    assert board.board[0][0] is VisionPiece.BLUE_MASTER
    assert board.board[4][2] is VisionPiece.RED_MASTER
    assert board.to_board_tokens()[4][2] == "red_master"


def test_board_rejects_invalid_shape_and_cell() -> None:
    with pytest.raises(ValueError, match="Board must have"):
        VisionBoard(board=((None,),))

    with pytest.raises(ValueError, match="out of range"):
        VisionBoard.empty().with_cell(5, 0, VisionPiece.RED_STUDENT)


def test_board_json_roundtrip(tmp_path) -> None:
    board = VisionBoard.empty().with_cell(2, 3, VisionPiece.BLUE_STUDENT)
    out_path = tmp_path / "vision_state.json"

    board.save_json(out_path)

    assert VisionBoard.load_json(out_path).board == board.board


def test_snapshot_roundtrip_json(tmp_path) -> None:
    board = VisionBoard.empty().with_cell(0, 2, VisionPiece.BLUE_MASTER)
    snapshot = _snapshot(board)

    out_path = tmp_path / "snapshot.json"
    snapshot.save_json(out_path)
    loaded = VisionSnapshot.load_json(out_path)

    assert loaded.red_cards == ("Tiger", "Horse")
    assert loaded.board.board == snapshot.board.board


def test_snapshot_from_dict_accepts_board_tokens() -> None:
    data = {
        "board": [
            [None, None, "blue_master", None, None],
            [None, None, None, None, None],
            [None, None, None, None, None],
            [None, None, None, None, None],
            [None, None, "red_master", None, None],
        ],
        "red_cards": ["Tiger", "Horse"],
        "blue_cards": ["Crab", "Boar"],
        "side_card": "Rabbit",
    }
    snapshot = VisionSnapshot.from_dict(data)
    assert snapshot.board.board[0][2] is VisionPiece.BLUE_MASTER


def test_snapshot_from_board_and_cards() -> None:
    board = VisionBoard.empty().with_cell(0, 2, VisionPiece.BLUE_MASTER)
    card_result = CardClassificationResult(
        predictions=(
            CardSlotPrediction(slot="red_0", class_name="Tiger", confidence=0.91),
            CardSlotPrediction(slot="red_1", class_name="Horse", confidence=0.88),
            CardSlotPrediction(slot="side", class_name="Rabbit", confidence=0.86),
            CardSlotPrediction(slot="blue_0", class_name="Crab", confidence=0.92),
            CardSlotPrediction(slot="blue_1", class_name="Boar", confidence=0.87),
        )
    )

    snapshot = VisionSnapshot.from_board_and_cards(
        board=board,
        card_result=card_result,
    )

    assert snapshot.red_cards == ("Tiger", "Horse")
    assert snapshot.blue_cards == ("Crab", "Boar")
    assert snapshot.side_card == "Rabbit"


def test_snapshot_to_game_state_with_card_names() -> None:
    board = VisionBoard.empty()
    board = board.with_cell(0, 2, VisionPiece.BLUE_MASTER)
    board = board.with_cell(4, 2, VisionPiece.RED_MASTER)
    board = board.with_cell(3, 1, VisionPiece.RED_STUDENT)

    game_state = _snapshot(board).to_game_state(to_move="RED")

    assert game_state.to_move is Player.RED
    assert game_state.red_cards[0].name == "Tiger"
    assert game_state.blue_cards[1].name == "Boar"
    assert game_state.side_card.name == "Rabbit"
    assert game_state.board[0][2] is not None
    assert game_state.board[0][2].owner is Player.BLUE
    assert game_state.board[0][2].kind is PieceType.MASTER
    assert game_state.board[4][2] is not None
    assert game_state.board[4][2].owner is Player.RED
    assert game_state.board[4][2].kind is PieceType.MASTER
    assert game_state.board[3][1] is not None
    assert game_state.board[3][1].owner is Player.RED
    assert game_state.board[3][1].kind is PieceType.STUDENT


def test_snapshot_to_game_state_rejects_duplicate_cards() -> None:
    with pytest.raises(ValueError, match="Cards must be 5 unique cards"):
        _snapshot(side_card="Tiger").to_game_state(to_move=Player.RED)


def test_snapshot_to_game_state_rejects_multiple_red_masters() -> None:
    board = VisionBoard.empty()
    board = board.with_cell(4, 2, VisionPiece.RED_MASTER)
    board = board.with_cell(3, 2, VisionPiece.RED_MASTER)

    with pytest.raises(ValueError, match="more than one RED master"):
        _snapshot(board).to_game_state(to_move=Player.RED)


def test_snapshot_to_game_state_rejects_too_many_red_students() -> None:
    board = VisionBoard.empty()
    for col in range(5):
        board = board.with_cell(4, col, VisionPiece.RED_STUDENT)

    with pytest.raises(ValueError, match="more than four RED students"):
        _snapshot(board).to_game_state(to_move=Player.RED)


def test_snapshot_to_game_state_rejects_too_many_red_pieces() -> None:
    board = VisionBoard.empty()
    board = board.with_cell(4, 2, VisionPiece.RED_MASTER)
    for row, col in ((4, 0), (4, 1), (4, 3), (4, 4), (3, 2)):
        board = board.with_cell(row, col, VisionPiece.RED_STUDENT)

    with pytest.raises(ValueError, match="more than four RED students|more than five RED pieces"):
        _snapshot(board).to_game_state(to_move=Player.RED)
