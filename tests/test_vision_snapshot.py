import pytest

from onitama.vision.board import VisionBoard, VisionPiece
from onitama.vision.card_classifier import CardClassificationResult, CardSlotPrediction
from onitama.vision.snapshot import VisionSnapshot


def test_snapshot_roundtrip_json(tmp_path) -> None:
    board = VisionBoard.empty().with_cell(0, 2, VisionPiece.BLUE_MASTER)
    snapshot = VisionSnapshot(
        board=board,
        red_cards=("Tiger", "Horse"),
        blue_cards=("Crab", "Boar"),
        side_card="Rabbit",
    )

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
