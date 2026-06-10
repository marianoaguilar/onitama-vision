import numpy as np
from onitama.vision.board import VisionBoard, VisionPiece
from onitama.vision.card_classifier import CardClassificationResult, CardSlotPrediction
from onitama.vision.vision_pipeline import VisionPipeline


class _StubPieceDetector:
    def __init__(self, board: VisionBoard) -> None:
        self.board = board
        self.received_frame = None

    def detect_from_frame(self, frame):
        self.received_frame = frame
        return ["stub-detection"]

    def detections_to_board(self, detections):
        assert detections == ["stub-detection"]
        return self.board


class _StubCardClassifier:
    def __init__(self, result: CardClassificationResult) -> None:
        self.result = result
        self.received_frame = None

    def classify_from_frame(self, frame):
        self.received_frame = frame
        return self.result


def _sample_card_result() -> CardClassificationResult:
    return CardClassificationResult(
        predictions=(
            CardSlotPrediction(slot="red_0", class_name="Tiger", confidence=0.91),
            CardSlotPrediction(slot="red_1", class_name="Horse", confidence=0.88),
            CardSlotPrediction(slot="side", class_name="Rabbit", confidence=0.86),
            CardSlotPrediction(slot="blue_0", class_name="Crab", confidence=0.92),
            CardSlotPrediction(slot="blue_1", class_name="Boar", confidence=0.87),
        )
    )


def test_pipeline_snapshot_from_frame_assembles_board_and_cards() -> None:
    board = VisionBoard.empty().with_cell(4, 2, VisionPiece.RED_MASTER)
    pipeline = VisionPipeline(
        piece_detector=_StubPieceDetector(board),
        card_classifier=_StubCardClassifier(_sample_card_result()),
    )
    frame = np.zeros((32, 32, 3), dtype=np.uint8)

    snapshot = pipeline.snapshot_from_frame(frame)

    assert pipeline.piece_detector.received_frame is frame
    assert pipeline.card_classifier.received_frame is frame
    assert snapshot.board.board == board.board
    assert snapshot.red_cards == ("Tiger", "Horse")
    assert snapshot.blue_cards == ("Crab", "Boar")
    assert snapshot.side_card == "Rabbit"

