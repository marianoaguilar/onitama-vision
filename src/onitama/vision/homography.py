import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from onitama.app.errors import VisionConfigurationError


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
            raise VisionConfigurationError("src_points must have exactly 4 points.")
        if rotate not in (0, 90, 180, 270):
            raise VisionConfigurationError("rotate must be one of {0, 90, 180, 270}.")
        
        return HomographyCalibration(src_points=src_points, dst_size=dst_size, rotate=rotate)

    def to_dict(self) -> dict[str, object]:
        return {
            "src_points": [[float(round(x)), float(round(y))] for x, y in self.src_points],
            "dst_size": [int(self.dst_size[0]), int(self.dst_size[1])],
            "rotate": int(self.rotate),
        }

    def save(self, path: str | Path) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def order_points_clockwise(pts: np.ndarray) -> np.ndarray:
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
    src = order_points_clockwise(np.array(calib.src_points, dtype=np.float32))
    dst = np.array(
        [(0, 0), (dst_w - 1, 0), (dst_w - 1, dst_h - 1), (0, dst_h - 1)],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(src, dst)


def apply_rotation(img: np.ndarray, rotate: int) -> np.ndarray:
    """Rotate an image by 0, 90, 180 or 270 degrees."""
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


def rotate_point(x: int, y: int, width: int, height: int, rotate: int) -> tuple[int, int]:
    """Rotate one point inside an image."""
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


def rotate_roi(
    x: int,
    y: int,
    w: int,
    h: int,
    width: int,
    height: int,
    rotate: int,
) -> tuple[int, int, int, int]:
    """Rotate a rectangular ROI and return its new bounding box.
    
    Args:
        - x, y, w, h: input ROI (top-left corner and size).
        - width, height: dimensions of the image containing the ROI.
        - rotate: rotation in degrees (0, 90, 180, 270).
    Returns:
        - new_x, new_y, new_w, new_h: bounding box of the rotated ROI.
    """
    corners = [
        (x, y),
        (x + w - 1, y),
        (x + w - 1, y + h - 1),
        (x, y + h - 1),
    ]
    rotated = [rotate_point(px, py, width, height, rotate) for px, py in corners]
    xs = [p[0] for p in rotated]
    ys = [p[1] for p in rotated]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return min_x, min_y, max_x - min_x + 1, max_y - min_y + 1


def build_padded_homography(
    calib: HomographyCalibration,
    padding_ratio: float,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int, int, int]]:
    """Build a warp that keeps extra padding around the board."""
    
    board_w, board_h = calib.dst_size
    pad_x = int(round(board_w * padding_ratio))
    pad_y = int(round(board_h * padding_ratio))
    out_w = board_w + 2 * pad_x
    out_h = board_h + 2 * pad_y

    src = order_points_clockwise(np.array(calib.src_points, dtype=np.float32))
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


def warp_board(frame: np.ndarray, calib: HomographyCalibration, M: np.ndarray | None = None) -> np.ndarray:
    """
    Warp a camera frame into a canonical top-down board view.
    Returns an image of size calib.dst_size, rotated according to calib.rotate.
    """
    dst_w, dst_h = calib.dst_size
    if M is None:
        M = compute_homography_matrix(calib)
    warped = cv2.warpPerspective(frame, M, (dst_w, dst_h))
    warped = apply_rotation(warped, calib.rotate)
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
    Assumes 0<=x<w and 0<=y<h. 
    
    Returns (row, col) in [0..board_size-1].
    """
    w, h = dst_size
    cell_w = w / board_size
    cell_h = h / board_size

    col = int(x // cell_w)
    row = int(y // cell_h)

    # Clamp boundary points to the board.
    col = max(0, min(board_size - 1, col))
    row = max(0, min(board_size - 1, row))

    return row, col
