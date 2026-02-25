from onitama.engine.pieces import PieceType, Player
from onitama.vision.board import VisionBoard, VisionPiece
from onitama.vision.bridge import vision_piece_to_engine_piece, vision_board_to_engine_board


def test_vision_piece_to_engine_piece_mapping() -> None:
    red_master = vision_piece_to_engine_piece(VisionPiece.RED_MASTER)
    assert red_master is not None
    assert red_master.owner is Player.RED
    assert red_master.kind is PieceType.MASTER

    blue_student = vision_piece_to_engine_piece(VisionPiece.BLUE_STUDENT)
    assert blue_student is not None
    assert blue_student.owner is Player.BLUE
    assert blue_student.kind is PieceType.STUDENT

    assert vision_piece_to_engine_piece(None) is None


def test_vision_board_to_engine_board_preserves_coordinates() -> None:
    state = VisionBoard.empty()
    state = state.with_cell(0, 2, VisionPiece.BLUE_MASTER)
    state = state.with_cell(4, 2, VisionPiece.RED_MASTER)
    state = state.with_cell(3, 1, VisionPiece.RED_STUDENT)

    board = vision_board_to_engine_board(state)

    assert board[0][2] is not None
    assert board[0][2].owner is Player.BLUE
    assert board[0][2].kind is PieceType.MASTER

    assert board[4][2] is not None
    assert board[4][2].owner is Player.RED
    assert board[4][2].kind is PieceType.MASTER

    assert board[3][1] is not None
    assert board[3][1].owner is Player.RED
    assert board[3][1].kind is PieceType.STUDENT
