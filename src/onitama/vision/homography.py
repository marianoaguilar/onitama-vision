import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class HomographyCalibration:
    """
    Calibration for mapping camera frames to a canonical top-down board view.
    src_points: 4 points in camera frame (float x,y).
    dst_size: output size (width,height), e.g. (500,500).
    rotate: rotation applied AFTER warp (0/90/180/270).
    """
    src_points: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    dst_size: Tuple[int, int]
    rotate: int = 0

    @staticmethod
    def load(path: str | Path) -> "HomographyCalibration":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        src_points = tuple(tuple(p) for p in data["src_points"])
        dst_size = tuple(data["dst_size"])
        rotate = int(data.get("rotate", 0))
        if len(src_points) != 4:
            raise ValueError("src_points must have exactly 4 points.")
        if rotate not in (0, 90, 180, 270):
            raise ValueError("rotate must be one of {0, 90, 180, 270}.")
        return HomographyCalibration(src_points=src_points, dst_size=dst_size, rotate=rotate)


def _order_points_clockwise(pts: np.ndarray) -> np.ndarray:
    """
    Returns points in order: top-left, top-right, bottom-right, bottom-left.
    Works even if clicked in random order.
    """
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def compute_homography_matrix(calib: HomographyCalibration) -> np.ndarray:
    """
    Compute the 3x3 homography matrix that maps camera coordinates to the canonical board image.
    Note: rotation is applied after warp (not included in the matrix).
    """
    dst_w, dst_h = calib.dst_size
    src = _order_points_clockwise(np.array(calib.src_points, dtype=np.float32))
    dst = np.array(
        [(0, 0), (dst_w - 1, 0), (dst_w - 1, dst_h - 1), (0, dst_h - 1)],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(src, dst)


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


def warp_board(frame: np.ndarray, calib: HomographyCalibration, M: np.ndarray | None = None) -> np.ndarray:
    """
    Warp a camera frame into a canonical top-down board view.
    Returns an image of size calib.dst_size, rotated according to calib.rotate.
    """
    dst_w, dst_h = calib.dst_size
    if M is None:
        M = compute_homography_matrix(calib)
    warped = cv2.warpPerspective(frame, M, (dst_w, dst_h))
    warped = _apply_rotation(warped, calib.rotate)
    return warped


def draw_grid(img: np.ndarray, cells: int = 5) -> np.ndarray:
    """
    Draw a cells x cells grid on top of img for debugging.
    """
    h, w = img.shape[:2]
    out = img.copy()
    for i in range(1, cells):
        x = int(i * w / cells)
        y = int(i * h / cells)
        cv2.line(out, (x, 0), (x, h - 1), (0, 255, 0), 1)
        cv2.line(out, (0, y), (w - 1, y), (0, 255, 0), 1)
    return out


def xy_to_cell(x: float, y: float, board_size: int = 5, dst_size: tuple[int, int] = (500, 500)) -> tuple[int, int]:
    """
    Map a point (x,y) in warped board coordinates to a (row, col) cell index.
    Assumes 0<=x<w and 0<=y<h. Returns (row, col) in [0..board_size-1].
    """
    w, h = dst_size
    cell_w = w / board_size
    cell_h = h / board_size

    col = int(x // cell_w)
    row = int(y // cell_h)

    # Clamp (safety for boundary points)
    col = max(0, min(board_size - 1, col))
    row = max(0, min(board_size - 1, row))

    return row, col