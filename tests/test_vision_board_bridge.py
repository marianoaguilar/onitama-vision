from onitama.engine.pieces import PieceType, Player
from onitama.vision.board import VisionBoard, VisionPiece
from onitama.vision.snapshot import VisionSnapshot


def test_snapshot_to_game_state_preserves_board_coordinates() -> None:
    state = VisionBoard.empty()
    state = state.with_cell(0, 2, VisionPiece.BLUE_MASTER)
    state = state.with_cell(4, 2, VisionPiece.RED_MASTER)
    state = state.with_cell(3, 1, VisionPiece.RED_STUDENT)

    snapshot = VisionSnapshot(
        board=state,
        red_cards=("Tiger", "Horse"),
        blue_cards=("Crab", "Boar"),
        side_card="Rabbit",
    )
    board = snapshot.to_game_state(to_move=Player.RED).board

    assert board[0][2] is not None
    assert board[0][2].owner is Player.BLUE
    assert board[0][2].kind is PieceType.MASTER

    assert board[4][2] is not None
    assert board[4][2].owner is Player.RED
    assert board[4][2].kind is PieceType.MASTER

    assert board[3][1] is not None
    assert board[3][1].owner is Player.RED
    assert board[3][1].kind is PieceType.STUDENT
