import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from onitama.vision.card_rois import SLOT_COLOR, SLOT_LABEL, SLOT_ORDER, Quad, SlotName, load_card_rois, save_card_rois


def open_camera(device: int = 0, width: int = 1280, height: int = 720, fps: int = 30) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera device {device}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def nearest_vertex(points: Quad, x: float, y: float, max_dist_px: float) -> int | None:
    best_idx: int | None = None
    best_d2 = max_dist_px * max_dist_px
    for idx, (px, py) in enumerate(points):
        d2 = (px - x) * (px - x) + (py - y) * (py - y)
        if d2 <= best_d2:
            best_d2 = d2
            best_idx = idx
    return best_idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual calibration for 5 Onitama card ROIs using draggable 4-point polygons.")
    parser.add_argument("--out", type=Path, default=Path("data/vision/card_rois.json"))
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--canvas-padding",
        type=int,
        default=80,
        help="Extra visible margin around frame. Enables dragging points outside the image bounds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rois = load_card_rois(args.out, allow_missing=True)

    cap = open_camera(device=args.camera_id, width=args.width, height=args.height, fps=args.fps)
    active_slot_idx = 0
    dragging_idx: int | None = None
    frozen_frame: np.ndarray | None = None
    frame_w = 0
    frame_h = 0

    window = "calibrate_card_rois"
    cv2.namedWindow(window)

    def active_slot() -> SlotName:
        return SLOT_ORDER[active_slot_idx]

    def to_world(mx: int, my: int) -> Tuple[float, float]:
        return float(mx - args.canvas_padding), float(my - args.canvas_padding)

    def on_mouse(event: int, x: int, y: int, flags: int, param) -> None:
        nonlocal dragging_idx
        if frame_w <= 0 or frame_h <= 0:
            return

        slot = active_slot()
        points = rois.get(slot)

        wx, wy = to_world(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                points.append((wx, wy))
                dragging_idx = len(points) - 1
                print(f"{slot}: added point {len(points)}/4 at ({wx:.1f}, {wy:.1f})")
            else:
                dragging_idx = nearest_vertex(points, wx, wy, max_dist_px=24.0)
        elif event == cv2.EVENT_MOUSEMOVE and dragging_idx is not None:
            points[dragging_idx] = (wx, wy)
        elif event == cv2.EVENT_LBUTTONUP:
            dragging_idx = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            idx = nearest_vertex(points, wx, wy, max_dist_px=24.0)
            if idx is not None:
                points.pop(idx)
                dragging_idx = None
                print(f"{slot}: removed point, now {len(points)}/4")

    cv2.setMouseCallback(window, on_mouse)

    print(f"Output file: {args.out}")
    print("Controls:")
    print("  1..5 select slot")
    print("  left click add point (until 4) | drag vertex to edit")
    print("  right click on vertex to remove")
    print("  f freeze/resume live view")
    print("  r clear active slot | a clear all slots")
    print("  s save | q quit")

    while True:
        if frozen_frame is None:
            ok, frame = cap.read()
            if not ok:
                print("Could not read frame from camera.")
                break
            current = frame
        else:
            current = frozen_frame

        frame_h, frame_w = current.shape[:2]

        canvas_h = frame_h + 2 * args.canvas_padding
        canvas_w = frame_w + 2 * args.canvas_padding
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:] = (18, 18, 18)
        pad = args.canvas_padding
        canvas[pad:pad + frame_h, pad:pad + frame_w] = current
        cv2.rectangle(canvas, (pad, pad), (pad + frame_w - 1, pad + frame_h - 1), (220, 220, 220), 1)

        for idx, slot in enumerate(SLOT_ORDER):
            points = rois.get(slot)
            color = SLOT_COLOR[slot]
            active = idx == active_slot_idx
            thick = 2 if active else 1

            if len(points) >= 2:
                pts_i = np.array(
                    [[int(round(x + pad)), int(round(y + pad))] for x, y in points],
                    dtype=np.int32,
                ).reshape((-1, 1, 2))
                cv2.polylines(canvas, [pts_i], isClosed=(len(points) == 4), color=color, thickness=thick)
            for v_idx, (vx, vy) in enumerate(points):
                px = int(round(vx + pad))
                py = int(round(vy + pad))
                radius = 7 if active and dragging_idx == v_idx else 5
                cv2.circle(canvas, (px, py), radius, color, -1)
                cv2.circle(canvas, (px, py), radius + 1, (0, 0, 0), 1)

            if points:
                cx = int(round(sum(x for x, _ in points) / len(points) + pad))
                cy = int(round(sum(y for _, y in points) / len(points) + pad))
            else:
                cx = 12
                cy = 0
            label = SLOT_LABEL[slot]
            if active:
                label = f"[{label}]"
            if points:
                cv2.putText(canvas, label, (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            status = f"{idx + 1}:{SLOT_LABEL[slot]} {len(points)}/4"
            cv2.putText(
                canvas,
                status,
                (16, 92 + idx * 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                color if active else (190, 190, 190),
                2 if active else 1,
            )

        mode = "FROZEN" if frozen_frame is not None else "LIVE"
        help_1 = f"mode={mode}  active={active_slot_idx + 1}:{SLOT_LABEL[active_slot()]}"
        help_2 = "click:add, drag:edit, right-click:delete | f freeze | r clear active | a clear all | s save | q quit"
        cv2.putText(canvas, help_1, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 1)
        cv2.putText(canvas, help_2, (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 1)
        cv2.imshow(window, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("1"):
            active_slot_idx = 0
            dragging_idx = None
        if key == ord("2"):
            active_slot_idx = 1
            dragging_idx = None
        if key == ord("3"):
            active_slot_idx = 2
            dragging_idx = None
        if key == ord("4"):
            active_slot_idx = 3
            dragging_idx = None
        if key == ord("5"):
            active_slot_idx = 4
            dragging_idx = None
        if key == ord("f"):
            if frozen_frame is None:
                frozen_frame = current.copy()
            else:
                frozen_frame = None
        if key == ord("r"):
            slot = active_slot()
            rois[slot] = []
            dragging_idx = None
            print(f"Cleared slot: {slot}")
        if key == ord("a"):
            for slot in SLOT_ORDER:
                rois[slot] = []
            dragging_idx = None
            print("Cleared all slots.")
        if key == ord("s"):
            try:
                save_card_rois(args.out, rois)
                print(f"Saved card ROIs to: {args.out}")
            except Exception as exc:  # noqa: BLE001
                print(f"Save failed: {exc}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
