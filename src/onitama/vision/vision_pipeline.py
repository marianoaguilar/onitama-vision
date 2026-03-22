from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from onitama.engine.pieces import Player
from onitama.engine.state import GameState
from onitama.vision.card_classifier import YoloCardClassifier
from onitama.vision.piece_detector import YoloPieceDetector
from onitama.vision.snapshot import VisionSnapshot

if TYPE_CHECKING:
    import numpy as np


@dataclass
class VisionPipeline:
    """High-level vision flow from frame to snapshot or GameState."""

    # Vision components
    piece_detector: YoloPieceDetector
    card_classifier: YoloCardClassifier

    def __init__(
        self,
        *,
        piece_detector: YoloPieceDetector | None = None,
        card_classifier: YoloCardClassifier | None = None,
    ) -> None:
        """Create the pipeline with default or injected vision components."""
        self.piece_detector = piece_detector if piece_detector is not None else YoloPieceDetector()
        self.card_classifier = card_classifier if card_classifier is not None else YoloCardClassifier()


    def snapshot_from_frame(self, frame: "np.ndarray",) -> VisionSnapshot:
        """Run the full vision pipeline and return a visual snapshot."""
        detections = self.piece_detector.detect_from_frame(frame)
        board = self.piece_detector.detections_to_board(detections)
        card_result = self.card_classifier.classify_from_frame(frame)
        
        return VisionSnapshot.from_board_and_cards(
            board=board,
            card_result=card_result,
        )

    def game_state_from_frame(self, frame: "np.ndarray", to_move: Player | str) -> GameState:
        """Run the full pipeline and return the engine GameState.
        """
        snapshot = self.snapshot_from_frame(frame)
        return snapshot.to_game_state(to_move)
