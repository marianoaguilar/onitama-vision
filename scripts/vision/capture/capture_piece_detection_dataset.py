import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from onitama.vision.homography import HomographyCalibration, apply_rotation, build_padded_homography, rotate_roi


def open_camera(device: int = 0, width: int = 1280, height: int = 720, fps: int = 30) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera device {device}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap

def draw_grid_roi(img: np.ndarray, roi: Tuple[int, int, int, int], cells: int = 5) -> np.ndarray:
    x, y, w, h = roi
    out = img.copy()
    cv2.rectangle(out, (x, y), (x + w - 1, y + h - 1), (0, 255, 255), 2)
    for i in range(1, cells):
        gx = x + int(i * w / cells)
        gy = y + int(i * h / cells)
        cv2.line(out, (gx, y), (gx, y + h - 1), (0, 255, 0), 1)
        cv2.line(out, (x, gy), (x + w - 1, gy), (0, 255, 0), 1)
    return out


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture padded warped board images for the piece detection dataset."
    )
    parser.add_argument("--calibration", type=Path, default=Path("data/vision/board_calibration.json"))
    parser.add_argument("--output-root", type=Path, default=Path("data/vision/raw_pieces"))
    parser.add_argument("--session-id", type=str, default="")
    parser.add_argument("--prefix", type=str, default="img")
    parser.add_argument("--image-format", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--jpg-quality", type=int, default=95)
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--padding-ratio", type=float, default=0.25)
    parser.add_argument("--interval-ms", type=int, default=900)
    parser.add_argument("--max-images", type=int, default=0, help="0 means unlimited.")
    parser.add_argument("--show-grid", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.calibration.exists():
        raise FileNotFoundError(f"Calibration file not found: {args.calibration}")
    if args.padding_ratio < 0.0:
        raise ValueError("padding_ratio must be >= 0.0")
    if args.interval_ms < 50:
        raise ValueError("interval_ms must be >= 50")

    session_id = args.session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = args.output_root / session_id
    images_dir = session_dir / "images"
    manifest_path = session_dir / "manifest.jsonl"
    meta_path = session_dir / "session_meta.json"
    images_dir.mkdir(parents=True, exist_ok=True)

    calib = HomographyCalibration.load(args.calibration)
    matrix, output_size, board_roi = build_padded_homography(calib, args.padding_ratio)
    raw_w, raw_h = output_size
    rotated_roi = rotate_roi(
        x=board_roi[0],
        y=board_roi[1],
        w=board_roi[2],
        h=board_roi[3],
        width=raw_w,
        height=raw_h,
        rotate=calib.rotate,
    )

    rotated_w = raw_w if calib.rotate in (0, 180) else raw_h
    rotated_h = raw_h if calib.rotate in (0, 180) else raw_w

    metadata = {
        "session_id": session_id,
        "created_utc": utc_now_iso(),
        "calibration_path": str(args.calibration),
        "padding_ratio": args.padding_ratio,
        "board_roi": {
            "x": rotated_roi[0],
            "y": rotated_roi[1],
            "w": rotated_roi[2],
            "h": rotated_roi[3],
        },
        "warped_size": {"w": rotated_w, "h": rotated_h},
        "camera": {
            "id": args.camera_id,
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
        },
        "capture": {
            "interval_ms": args.interval_ms,
            "image_format": args.image_format,
            "jpg_quality": args.jpg_quality,
            "max_images": args.max_images,
        },
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    cap = open_camera(device=args.camera_id, width=args.width, height=args.height, fps=args.fps)
    auto_capture = True
    show_grid = args.show_grid
    capture_count = 0
    last_capture_ts = 0.0
    interval_s = args.interval_ms / 1000.0
    manifest_file = manifest_path.open("a", encoding="utf-8")

    print(f"Session: {session_id}")
    print(f"Output:  {session_dir}")
    print(f"Padding: {args.padding_ratio:.3f}")
    print(f"ROI:     {rotated_roi}")
    print("Controls: q quit | p pause/resume auto | s save now | g toggle grid")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read frame from camera.")
                break

            raw_warp = cv2.warpPerspective(frame, matrix, output_size)
            warped = apply_rotation(raw_warp, calib.rotate)

            now = time.monotonic()
            should_capture = auto_capture and (now - last_capture_ts) >= interval_s
            if should_capture:
                capture_count += 1
                image_name = f"{args.prefix}_{capture_count:06d}.{args.image_format}"
                image_path = images_dir / image_name

                if args.image_format == "jpg":
                    write_ok = cv2.imwrite(str(image_path), warped, [cv2.IMWRITE_JPEG_QUALITY, args.jpg_quality])
                else:
                    write_ok = cv2.imwrite(str(image_path), warped)

                if write_ok:
                    entry = {
                        "session_id": session_id,
                        "capture_index": capture_count,
                        "image": f"images/{image_name}",
                        "captured_utc": utc_now_iso(),
                        "padding_ratio": args.padding_ratio,
                        "board_roi": {
                            "x": rotated_roi[0],
                            "y": rotated_roi[1],
                            "w": rotated_roi[2],
                            "h": rotated_roi[3],
                        },
                        "warped_size": {"w": warped.shape[1], "h": warped.shape[0]},
                    }
                    manifest_file.write(json.dumps(entry) + "\n")
                    manifest_file.flush()
                    last_capture_ts = now
                    print(f"[{capture_count}] saved {image_name}")
                else:
                    capture_count -= 1
                    print(f"Failed to save image: {image_path}")

            if not args.no_preview:
                preview = warped
                if show_grid:
                    preview = draw_grid_roi(preview, rotated_roi, cells=5)
                status = "AUTO" if auto_capture else "PAUSED"
                cv2.putText(
                    preview,
                    f"{status}  count={capture_count}  size={preview.shape[1]}x{preview.shape[0]}",
                    (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    preview,
                    "q quit | p pause/resume | s save now | g toggle grid",
                    (10, preview.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    1,
                )
                cv2.imshow("capture_piece_detection_dataset", preview)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("p"):
                auto_capture = not auto_capture
            if key == ord("g"):
                show_grid = not show_grid
            if key == ord("s"):
                now = time.monotonic()
                capture_count += 1
                image_name = f"{args.prefix}_{capture_count:06d}.{args.image_format}"
                image_path = images_dir / image_name
                if args.image_format == "jpg":
                    write_ok = cv2.imwrite(str(image_path), warped, [cv2.IMWRITE_JPEG_QUALITY, args.jpg_quality])
                else:
                    write_ok = cv2.imwrite(str(image_path), warped)
                if write_ok:
                    entry = {
                        "session_id": session_id,
                        "capture_index": capture_count,
                        "image": f"images/{image_name}",
                        "captured_utc": utc_now_iso(),
                        "padding_ratio": args.padding_ratio,
                        "board_roi": {
                            "x": rotated_roi[0],
                            "y": rotated_roi[1],
                            "w": rotated_roi[2],
                            "h": rotated_roi[3],
                        },
                        "warped_size": {"w": warped.shape[1], "h": warped.shape[0]},
                    }
                    manifest_file.write(json.dumps(entry) + "\n")
                    manifest_file.flush()
                    last_capture_ts = now
                    print(f"[{capture_count}] saved {image_name} (manual)")
                else:
                    capture_count -= 1
                    print(f"Failed to save image: {image_path}")

            if args.max_images > 0 and capture_count >= args.max_images:
                print(f"Reached max_images={args.max_images}.")
                break
    finally:
        manifest_file.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
