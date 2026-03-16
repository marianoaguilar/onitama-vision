import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


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


def open_camera(device: int = 0, width: int = 1280, height: int = 720, fps: int = 30) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera device {device}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def _order_points_clockwise(pts: np.ndarray) -> np.ndarray:
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def load_card_rois(path: Path) -> Dict[SlotName, Quad]:
    if not path.exists():
        raise FileNotFoundError(f"Card ROI file not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Invalid card ROI file: root must be an object.")

    rois: Dict[SlotName, Quad] = {}
    for slot in SLOT_ORDER:
        entry = raw.get(slot)
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid card ROI for {slot}: expected object with 'src_points'.")
        src_points = entry.get("src_points")
        if not isinstance(src_points, list) or len(src_points) != 4:
            raise ValueError(f"Invalid card ROI for {slot}: 'src_points' must have exactly 4 points.")

        pts: Quad = []
        for p in src_points:
            if not isinstance(p, (list, tuple)) or len(p) != 2:
                raise ValueError(f"Invalid point in {slot}.")
            pts.append((float(p[0]), float(p[1])))

        ordered = _order_points_clockwise(np.array(pts, dtype=np.float32))
        rois[slot] = [(float(x), float(y)) for x, y in ordered.tolist()]

    return rois


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview calibrated card ROIs on real camera feed.")
    parser.add_argument("--rois-path", type=Path, default=Path("data/vision/card_rois.json"))
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--show-vertices", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rois = load_card_rois(args.rois_path)

    cap = open_camera(device=args.camera_id, width=args.width, height=args.height, fps=args.fps)
    frozen_frame: np.ndarray | None = None
    show_vertices = args.show_vertices

    print(f"Loaded ROIs from: {args.rois_path}")
    print("Controls: q quit | f freeze/live | r reload roi file | v toggle vertices")

    while True:
        if frozen_frame is None:
            ok, frame = cap.read()
            if not ok:
                print("Could not read frame from camera.")
                break
            current = frame
        else:
            current = frozen_frame

        h, _ = current.shape[:2]
        preview = current.copy()

        for slot in SLOT_ORDER:
            color = SLOT_COLOR[slot]
            points = rois[slot]
            pts_i = np.array([[int(round(x)), int(round(y))] for x, y in points], dtype=np.int32).reshape((-1, 1, 2))

            cv2.polylines(preview, [pts_i], isClosed=True, color=color, thickness=2)
            if show_vertices:
                for vx, vy in points:
                    cv2.circle(preview, (int(round(vx)), int(round(vy))), 5, color, -1)
                    cv2.circle(preview, (int(round(vx)), int(round(vy))), 6, (0, 0, 0), 1)

            cx = int(round(sum(x for x, _ in points) / 4.0))
            cy = int(round(sum(y for _, y in points) / 4.0))
            cv2.putText(preview, SLOT_LABEL[slot], (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)

        mode = "FROZEN" if frozen_frame is not None else "LIVE"
        help_text = f"{mode} | q quit | f freeze/live | r reload rois | v vertices"
        cv2.putText(
            preview,
            help_text,
            (12, h - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )
        cv2.imshow("preview_card_rois_real", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("f"):
            if frozen_frame is None:
                frozen_frame = current.copy()
            else:
                frozen_frame = None
        if key == ord("r"):
            try:
                rois = load_card_rois(args.rois_path)
                print(f"Reloaded ROIs from: {args.rois_path}")
            except Exception as exc:  # noqa: BLE001
                print(f"Reload failed: {exc}")
        if key == ord("v"):
            show_vertices = not show_vertices

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
