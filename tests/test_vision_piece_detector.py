from onitama.vision.board import VisionPiece
from onitama.vision.piece_detector import PieceDetection, YoloPieceDetector


def _detection(
    *,
    confidence: float,
    cell: tuple[int, int] | None,
    vision_piece: VisionPiece | None,
) -> PieceDetection:
    return PieceDetection(
        class_index=0,
        class_name="" if vision_piece is None else vision_piece.value,
        confidence=confidence,
        bbox_xyxy=(0.0, 0.0, 1.0, 1.0),
        anchor_xy=(0.5, 0.5),
        cell=cell,
        in_board=cell is not None,
        vision_piece=vision_piece,
    )


def test_detections_to_board_keeps_best_detection_per_cell() -> None:
    detector = object.__new__(YoloPieceDetector)

    board = detector.detections_to_board(
        [
            _detection(confidence=0.80, cell=(2, 3), vision_piece=VisionPiece.RED_STUDENT),
            _detection(confidence=0.95, cell=(2, 3), vision_piece=VisionPiece.BLUE_MASTER),
            _detection(confidence=0.70, cell=(4, 2), vision_piece=VisionPiece.RED_MASTER),
            _detection(confidence=0.99, cell=None, vision_piece=VisionPiece.BLUE_STUDENT),
            _detection(confidence=0.99, cell=(0, 0), vision_piece=None),
        ]
    )

    assert board.board[2][3] is VisionPiece.BLUE_MASTER
    assert board.board[4][2] is VisionPiece.RED_MASTER
    assert board.board[0][0] is None
