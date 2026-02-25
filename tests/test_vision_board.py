import pytest

from onitama.vision.board import BOARD_SIZE, VisionBoard, VisionPiece


def test_empty_state_has_5x5_none() -> None:
    state = VisionBoard.empty()
    assert len(state.board) == BOARD_SIZE
    assert all(len(row) == BOARD_SIZE for row in state.board)
    assert all(cell is None for row in state.board for cell in row)


def test_with_cell_accepts_enum_and_string() -> None:
    state = VisionBoard.empty()
    state = state.with_cell(0, 0, VisionPiece.BLUE_MASTER)
    state = state.with_cell(4, 2, "red_master")

    assert state.board[0][0] is VisionPiece.BLUE_MASTER
    assert state.board[4][2] is VisionPiece.RED_MASTER


def test_to_board_tokens_serializes_values() -> None:
    state = VisionBoard.empty()
    state = state.with_cell(1, 1, VisionPiece.BLUE_STUDENT)
    tokens = state.to_board_tokens()

    assert tokens[1][1] == "blue_student"
    assert tokens[0][0] is None


def test_invalid_board_shape_raises() -> None:
    with pytest.raises(ValueError, match="Board must have"):
        VisionBoard(board=((None,),))


def test_out_of_range_cell_raises() -> None:
    with pytest.raises(ValueError, match="out of range"):
        VisionBoard.empty().with_cell(5, 0, VisionPiece.RED_STUDENT)


def test_json_roundtrip(tmp_path) -> None:
    state = VisionBoard.empty().with_cell(2, 3, VisionPiece.BLUE_STUDENT)
    out_path = tmp_path / "vision_state.json"
    state.save_json(out_path)

    loaded = VisionBoard.load_json(out_path)
    assert loaded.board == state.board
