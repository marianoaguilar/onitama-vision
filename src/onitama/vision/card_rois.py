from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from onitama.errors import VisionConfigurationError
from onitama.vision.homography import order_points_clockwise


SlotName = str
Point = Tuple[float, float]
Quad = List[Point]

SLOT_ORDER: Tuple[SlotName, ...] = ("red_0", "red_1", "side", "blue_0", "blue_1")
SLOT_LABEL: Dict[SlotName, str] = {
    "red_0": "RED 0",
    "red_1": "RED 1",
    "side": "SIDE",
    "blue_0": "BLUE 0",
    "blue_1": "BLUE 1",
}
SLOT_COLOR: Dict[SlotName, Tuple[int, int, int]] = {
    "red_0": (0, 0, 255),
    "red_1": (0, 0, 255),
    "side": (0, 255, 255),
    "blue_0": (255, 0, 0),
    "blue_1": (255, 0, 0),
}

def _parse_quad(value: object, *, slot: SlotName) -> Quad:
    """
    Validate and normalize a 4-point ROI.
    
    Args:
        value: the raw JSON value for this ROI, expected to be a list of 4 (x, y) points.
        slot: the name of the card slot (for error messages).
    Returns:
        A list of 4 (x, y) points in clockwise order starting from top-left.
    """
    
    if not isinstance(value, list) or len(value) != 4:
        raise VisionConfigurationError(f"Invalid card ROI for {slot}: 'src_points' must have exactly 4 points.")

    points: Quad = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise VisionConfigurationError(f"Invalid point in {slot}.")
        points.append((float(point[0]), float(point[1])))

    ordered = order_points_clockwise(np.array(points, dtype=np.float32))
    return [(float(x), float(y)) for x, y in ordered.tolist()]


def load_card_rois(path: str | Path, *, allow_missing: bool = False) -> Dict[SlotName, Quad]:
    """Load the 5 card ROIs from JSON."""
    
    roi_path = Path(path)
    if not roi_path.exists():
        if allow_missing:
            return {slot: [] for slot in SLOT_ORDER}
        raise VisionConfigurationError(f"Card ROI file not found: {roi_path}")

    raw = json.loads(roi_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise VisionConfigurationError("Invalid card ROI file: root must be an object.")

    rois: Dict[SlotName, Quad] = {}
    for slot in SLOT_ORDER:
        entry = raw.get(slot)
        if entry is None:
            if allow_missing:
                rois[slot] = []
                continue
            raise VisionConfigurationError(f"Invalid card ROI for {slot}: expected object with 'src_points'.")
        if not isinstance(entry, dict):
            raise VisionConfigurationError(f"Invalid card ROI for {slot}: expected object with 'src_points'.")
        rois[slot] = _parse_quad(entry.get("src_points"), slot=slot)
    return rois


def save_card_rois(path: str | Path, rois: Dict[SlotName, Quad]) -> None:
    """Save the 5 card ROIs to JSON."""
    
    payload: Dict[str, Dict[str, List[List[float]]]] = {}
    for slot in SLOT_ORDER:
        points = rois.get(slot)
        if points is None or len(points) != 4:
            raise ValueError(f"Cannot save: slot '{slot}' does not have 4 points.")
        ordered = order_points_clockwise(np.array(points, dtype=np.float32))
        payload[slot] = {
            "src_points": [[float(round(x)), float(round(y))] for x, y in ordered.tolist()],
        }

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def quad_centroid(points: Quad) -> Point:
    """Return the center point of a quad."""
    
    if len(points) == 0:
        raise ValueError("Cannot compute centroid of an empty quad.")
    return (
        float(sum(x for x, _ in points) / len(points)),
        float(sum(y for _, y in points) / len(points)),
    )


def extract_polygon_crop(frame: np.ndarray, points: Quad, *, mask_polygon: bool = True) -> np.ndarray:
    """Crop the ROI bounding box and optionally mask the outside of the polygon."""
    
    pts = np.array(points, dtype=np.float32)
    if pts.shape != (4, 2):
        raise VisionConfigurationError("points must contain exactly 4 (x, y) vertices.")

    min_x = int(np.floor(float(np.min(pts[:, 0]))))
    max_x = int(np.ceil(float(np.max(pts[:, 0]))))
    min_y = int(np.floor(float(np.min(pts[:, 1]))))
    max_y = int(np.ceil(float(np.max(pts[:, 1]))))

    frame_h, frame_w = frame.shape[:2]
    src_x0 = max(0, min_x)
    src_y0 = max(0, min_y)
    src_x1 = min(frame_w - 1, max_x)
    src_y1 = min(frame_h - 1, max_y)
    if src_x0 > src_x1 or src_y0 > src_y1:
        raise VisionConfigurationError("ROI polygon falls completely outside frame.")

    # Clip the ROI to the visible frame.
    out = frame[src_y0:src_y1 + 1, src_x0:src_x1 + 1].copy()
    if not mask_polygon:
        return out

    # Keep a rectangular crop, but black out pixels outside the polygon.
    shifted = np.array(
        [[int(round(x - src_x0)), int(round(y - src_y0))] for x, y in points],
        dtype=np.int32,
    ).reshape((-1, 1, 2))
    mask = np.zeros((out.shape[0], out.shape[1]), dtype=np.uint8)
    cv2.fillPoly(mask, [shifted], 255)
    return cv2.bitwise_and(out, out, mask=mask)


def extract_card_crops(
    frame: np.ndarray,
    rois: Dict[SlotName, Quad],
    *,
    mask_polygon: bool = True,
) -> Dict[SlotName, np.ndarray]:
    """Extract one crop per card slot."""
    
    crops: Dict[SlotName, np.ndarray] = {}
    for slot in SLOT_ORDER:
        points = rois.get(slot)
        if points is None or len(points) != 4:
            raise VisionConfigurationError(f"Slot '{slot}' does not contain a valid 4-point ROI.")
        crops[slot] = extract_polygon_crop(frame, points, mask_polygon=mask_polygon)
    return crops


def draw_card_rois_overlay(
    frame: np.ndarray,
    rois: Dict[SlotName, Quad],
    *,
    show_vertices: bool = False,
) -> np.ndarray:
    """Draw the configured card ROIs on top of an image."""
    
    preview = frame.copy()
    for slot in SLOT_ORDER:
        points = rois.get(slot, [])
        if len(points) == 0:
            continue

        color = SLOT_COLOR[slot]
        if len(points) >= 2:
            pts_i = np.array(
                [[int(round(x)), int(round(y))] for x, y in points],
                dtype=np.int32,
            ).reshape((-1, 1, 2))
            cv2.polylines(preview, [pts_i], isClosed=(len(points) == 4), color=color, thickness=2)

        if show_vertices:
            for vx, vy in points:
                cv2.circle(preview, (int(round(vx)), int(round(vy))), 5, color, -1)
                cv2.circle(preview, (int(round(vx)), int(round(vy))), 6, (0, 0, 0), 1)

        cx, cy = quad_centroid(points)
        cv2.putText(
            preview,
            SLOT_LABEL[slot],
            (int(round(cx - 40.0)), int(round(cy))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
        )
    return preview
