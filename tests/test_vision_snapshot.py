import pytest

from onitama.engine.pieces import Player
from onitama.vision.board import VisionBoard, VisionPiece
from onitama.vision.snapshot import VisionSnapshot


def test_snapshot_roundtrip_json(tmp_path) -> None:
    board = VisionBoard.empty().with_cell(0, 2, VisionPiece.BLUE_MASTER)
    snapshot = VisionSnapshot(
        board=board,
        to_move=Player.RED,
        red_cards=("Tiger", "Horse"),
        blue_cards=("Crab", "Boar"),
        side_card="Rabbit",
    )

    out_path = tmp_path / "snapshot.json"
    snapshot.save_json(out_path)
    loaded = VisionSnapshot.load_json(out_path)

    assert loaded.to_move is Player.RED
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
        "to_move": "BLUE",
        "red_cards": ["Tiger", "Horse"],
        "blue_cards": ["Crab", "Boar"],
        "side_card": "Rabbit",
    }
    snapshot = VisionSnapshot.from_dict(data)
    assert snapshot.to_move is Player.BLUE
    assert snapshot.board.board[0][2] is VisionPiece.BLUE_MASTER


def test_snapshot_rejects_invalid_to_move() -> None:
    with pytest.raises(ValueError, match="to_move"):
        VisionSnapshot(
            board=VisionBoard.empty(),
            to_move="GREEN",
            red_cards=("Tiger", "Horse"),
            blue_cards=("Crab", "Boar"),
            side_card="Rabbit",
        )
