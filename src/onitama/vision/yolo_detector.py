from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from onitama.vision.board import BOARD_SIZE, VisionBoard, VisionPiece
from onitama.vision.homography import HomographyCalibration, xy_to_cell


def _order_points_clockwise(pts: np.ndarray) -> np.ndarray:
    pts = pts.astype(np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)

    top_left = pts[np.argmin(sums)]
    bottom_right = pts[np.argmax(sums)]
    top_right = pts[np.argmin(diffs)]
    bottom_left = pts[np.argmax(diffs)]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def _apply_rotation(img: np.ndarray, rotate: int) -> np.ndarray:
    rotate = rotate % 360
    if rotate == 0:
        return img
    if rotate == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if rotate == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if rotate == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError("rotate must be one of {0, 90, 180, 270}.")


def _rotate_point(x: int, y: int, width: int, height: int, rotate: int) -> Tuple[int, int]:
    rotate = rotate % 360
    if rotate == 0:
        return x, y
    if rotate == 90:
        return height - 1 - y, x
    if rotate == 180:
        return width - 1 - x, height - 1 - y
    if rotate == 270:
        return y, width - 1 - x
    raise ValueError("rotate must be one of {0, 90, 180, 270}.")


def _rotate_roi(
    x: int,
    y: int,
    w: int,
    h: int,
    width: int,
    height: int,
    rotate: int,
) -> Tuple[int, int, int, int]:
    corners = [
        (x, y),
        (x + w - 1, y),
        (x + w - 1, y + h - 1),
        (x, y + h - 1),
    ]
    rotated_corners = [_rotate_point(px, py, width, height, rotate) for px, py in corners]
    xs = [pt[0] for pt in rotated_corners]
    ys = [pt[1] for pt in rotated_corners]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return min_x, min_y, max_x - min_x + 1, max_y - min_y + 1


def _build_padded_homography(
    calib: HomographyCalibration,
    padding_ratio: float,
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int, int, int]]:
    board_w, board_h = calib.dst_size
    pad_x = int(round(board_w * padding_ratio))
    pad_y = int(round(board_h * padding_ratio))
    out_w = board_w + 2 * pad_x
    out_h = board_h + 2 * pad_y

    src = _order_points_clockwise(np.array(calib.src_points, dtype=np.float32))
    dst = np.array(
        [
            (pad_x, pad_y),
            (pad_x + board_w - 1, pad_y),
            (pad_x + board_w - 1, pad_y + board_h - 1),
            (pad_x, pad_y + board_h - 1),
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    board_roi = (pad_x, pad_y, board_w, board_h)
    return matrix, (out_w, out_h), board_roi


@dataclass(frozen=True)
class PieceDetection:
    class_index: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    anchor_xy: tuple[float, float]
    cell: tuple[int, int] | None
    in_board: bool
    vision_piece: VisionPiece | None


class YoloPieceDetector:
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
            raise FileNotFoundError(f"YOLO model not found: {model_path}")
        if not calibration_path.exists():
            raise FileNotFoundError(f"Calibration file not found: {calibration_path}")
        if padding_ratio < 0.0:
            raise ValueError("padding_ratio must be >= 0.0")
        if not (0.0 <= conf <= 1.0):
            raise ValueError("conf must be in [0.0, 1.0]")
        if not (0.0 <= iou <= 1.0):
            raise ValueError("iou must be in [0.0, 1.0]")
        if not (0.0 <= anchor_x_ratio <= 1.0):
            raise ValueError("anchor_x_ratio must be in [0.0, 1.0]")

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
        self.matrix, self.output_size, self.board_roi = _build_padded_homography(self.calib, self.padding_ratio)
        raw_w, raw_h = self.output_size
        self.rotated_roi = _rotate_roi(
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
            raise RuntimeError(
                "Could not import ultralytics. Install it in your .venv with: "
                ".venv/bin/python -m pip install ultralytics"
            ) from exc

        self._model = YOLO(str(model_path))
        names = self._model.names if isinstance(self._model.names, dict) else {}
        self.class_names = {int(idx): str(name) for idx, name in names.items()}

    def warp_frame(self, frame: np.ndarray) -> np.ndarray:
        raw_warp = cv2.warpPerspective(frame, self.matrix, self.output_size)
        return _apply_rotation(raw_warp, self.calib.rotate)

    def detect_on_warped(self, warped: np.ndarray) -> tuple[list[PieceDetection], float]:
        t0 = time.perf_counter()
        results = self._model.predict(
            source=warped,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            device=self.yolo_device,
            verbose=False,
        )
        infer_ms = (time.perf_counter() - t0) * 1000.0
        result = results[0]

        detections: list[PieceDetection] = []
        if result.boxes is None:
            return detections, infer_ms

        roi_x, roi_y, roi_w, roi_h = self.rotated_roi
        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        for i in range(len(result.boxes)):
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])
            class_index = int(classes[i])
            class_name = self.class_names.get(class_index, str(class_index))
            confidence = float(confs[i])

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
        return detections, infer_ms

    def detect_from_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[PieceDetection], float]:
        warped = self.warp_frame(frame)
        detections, infer_ms = self.detect_on_warped(warped)
        return warped, detections, infer_ms

    def detections_to_board(self, detections: list[PieceDetection]) -> VisionBoard:
        best_conf = [[-1.0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        tokens: list[list[str | None]] = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

        for detection in detections:
            if detection.cell is None or detection.vision_piece is None:
                continue
            row, col = detection.cell
            if detection.confidence <= best_conf[row][col]:
                continue
            best_conf[row][col] = detection.confidence
            tokens[row][col] = detection.vision_piece.value

        return VisionBoard.from_board_tokens(tokens)
