from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from onitama.app.errors import (
    VisionConfigurationError,
    VisionDependencyError,
)
from onitama.vision.board import BOARD_SIZE, VisionBoard, VisionPiece
from onitama.vision.homography import (
    HomographyCalibration,
    apply_rotation,
    build_padded_homography,
    rotate_roi,
    xy_to_cell,
)


@dataclass(frozen=True)
class PieceDetection:
    """One YOLO piece detection mapped to board coordinates when possible."""

    class_index: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    anchor_xy: tuple[float, float]
    cell: tuple[int, int] | None
    in_board: bool
    vision_piece: VisionPiece | None


class YoloPieceDetector:
    """Run piece detection on the warped board view."""

    def __init__(
        self,
        *,
        model_path: str | Path = "models/pieces_yolov8s_640_best.pt",
        calibration_path: str | Path = "data/vision/calibration.json",
        padding_ratio: float = 0.25,
        imgsz: int = 640,
        conf: float = 0.50,
        iou: float = 0.45,
        max_det: int = 25,
        yolo_device: str = "cpu",
        anchor_x_ratio: float = 0.30,
    ) -> None:
        model_path = Path(model_path)
        calibration_path = Path(calibration_path)
        if not model_path.exists():
            raise VisionConfigurationError(f"YOLO model not found: {model_path}")
        if not calibration_path.exists():
            raise VisionConfigurationError(f"Calibration file not found: {calibration_path}")
        if padding_ratio < 0.0:
            raise VisionConfigurationError("padding_ratio must be >= 0.0")
        if not (0.0 <= conf <= 1.0):
            raise VisionConfigurationError("conf must be in [0.0, 1.0]")
        if not (0.0 <= iou <= 1.0):
            raise VisionConfigurationError("iou must be in [0.0, 1.0]")
        if not (0.0 <= anchor_x_ratio <= 1.0):
            raise VisionConfigurationError("anchor_x_ratio must be in [0.0, 1.0]")

        self.model_path = model_path
        self.calibration_path = calibration_path
        self.padding_ratio = float(padding_ratio)
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_det = int(max_det)
        self.yolo_device = yolo_device
        self.anchor_x_ratio = float(anchor_x_ratio)

        self.calib = HomographyCalibration.load(calibration_path)
        self.matrix, self.output_size, self.board_roi = build_padded_homography(self.calib, self.padding_ratio)
        raw_w, raw_h = self.output_size
        self.rotated_roi = rotate_roi(
            x=self.board_roi[0],
            y=self.board_roi[1],
            w=self.board_roi[2],
            h=self.board_roi[3],
            width=raw_w,
            height=raw_h,
            rotate=self.calib.rotate,
        )

        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise VisionDependencyError(
                "Could not import ultralytics. Install it in your .venv with: "
                ".venv/bin/python -m pip install ultralytics"
            ) from exc

        self._model = YOLO(str(model_path))
        names = self._model.names if isinstance(self._model.names, dict) else {}
        self.class_names = {int(idx): str(name) for idx, name in names.items()}

    def warp_frame(self, frame: np.ndarray) -> np.ndarray:
        """Warp the camera frame to the canonical board view."""
        raw_warp = cv2.warpPerspective(frame, self.matrix, self.output_size)
        return apply_rotation(raw_warp, self.calib.rotate)

    def detect_on_warped(self, warped: np.ndarray) -> list[PieceDetection]:
        """Run YOLO on an already warped board image."""
        results = self._model.predict(
            source=warped,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            device=self.yolo_device,
            verbose=False,
        )
        result = results[0]

        detections: list[PieceDetection] = []
        if result.boxes is None:
            return detections

        roi_x, roi_y, roi_w, roi_h = self.rotated_roi
        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        for i in range(len(result.boxes)):
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])
            class_index = int(classes[i])
            class_name = self.class_names.get(class_index, str(class_index))
            confidence = float(confs[i])

            # This anchor works better than the box center for deciding the board cell.
            anchor_x = float(x1 + self.anchor_x_ratio * (x2 - x1))
            anchor_y = float((y1 + y2) * 0.5)
            local_x = anchor_x - float(roi_x)
            local_y = anchor_y - float(roi_y)
            in_board = (0.0 <= local_x < float(roi_w)) and (0.0 <= local_y < float(roi_h))

            cell: tuple[int, int] | None = None
            if in_board:
                row, col = xy_to_cell(local_x, local_y, board_size=BOARD_SIZE, dst_size=(roi_w, roi_h))
                cell = (row, col)

            vision_piece: VisionPiece | None
            try:
                vision_piece = VisionPiece(class_name)
            except ValueError:
                vision_piece = None

            detections.append(
                PieceDetection(
                    class_index=class_index,
                    class_name=class_name,
                    confidence=confidence,
                    bbox_xyxy=(x1, y1, x2, y2),
                    anchor_xy=(anchor_x, anchor_y),
                    cell=cell,
                    in_board=in_board,
                    vision_piece=vision_piece,
                )
            )
        return detections

    def detect_from_frame(self, frame: np.ndarray) -> list[PieceDetection]:
        """Warp a frame and run piece detection on it."""
        return self.detect_on_warped(self.warp_frame(frame))

    def detections_to_board(self, detections: list[PieceDetection]) -> VisionBoard:
        """Convert raw detections into a discrete 5x5 board."""
        best_conf = [[-1.0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        tokens: list[list[str | None]] = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

        for detection in detections:
            if detection.cell is None or detection.vision_piece is None:
                continue
            row, col = detection.cell
            # Keep only the strongest detection for each cell.
            if detection.confidence <= best_conf[row][col]:
                continue
            best_conf[row][col] = detection.confidence
            tokens[row][col] = detection.vision_piece.value

        return VisionBoard.from_board_tokens(tokens)
