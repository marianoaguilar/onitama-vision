import argparse
from pathlib import Path

import cv2
import numpy as np

from onitama.vision.card_rois import SLOT_ORDER, draw_card_rois_overlay, load_card_rois


def open_camera(device: int = 0, width: int = 1280, height: int = 720, fps: int = 30) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera device {device}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


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
        preview = draw_card_rois_overlay(current, rois, show_vertices=show_vertices)

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
