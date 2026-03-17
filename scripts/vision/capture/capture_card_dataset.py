import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import cv2

from onitama.vision.card_rois import (
    SLOT_ORDER,
    SlotName,
    draw_card_rois_overlay,
    extract_polygon_crop,
    load_card_rois,
)


def open_camera(device: int = 0, width: int = 1280, height: int = 720, fps: int = 30) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera device {device}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture card crops dataset from calibrated card ROI polygons.")
    parser.add_argument("--rois-path", type=Path, default=Path("data/vision/card_rois.json"))
    parser.add_argument("--output-root", type=Path, default=Path("data/vision/raw_cards"))
    parser.add_argument("--session-id", type=str, default="")
    parser.add_argument("--prefix", type=str, default="img")
    parser.add_argument("--image-format", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--jpg-quality", type=int, default=95)
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--interval-ms", type=int, default=800)
    parser.add_argument("--max-captures", type=int, default=0, help="0 means unlimited.")
    parser.add_argument(
        "--mask-polygon",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply polygon mask inside each ROI crop (enabled by default). Use --no-mask-polygon to disable.",
    )
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--show-rois", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.interval_ms < 50:
        raise ValueError("interval_ms must be >= 50")

    rois = load_card_rois(args.rois_path)
    session_id = args.session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = args.output_root / session_id
    manifest_path = session_dir / "manifest.jsonl"
    meta_path = session_dir / "session_meta.json"

    slot_dirs: Dict[SlotName, Path] = {}
    for slot in SLOT_ORDER:
        slot_dir = session_dir / "images" / slot
        slot_dir.mkdir(parents=True, exist_ok=True)
        slot_dirs[slot] = slot_dir

    meta = {
        "session_id": session_id,
        "created_utc": utc_now_iso(),
        "rois_path": str(args.rois_path),
        "image_format": args.image_format,
        "jpg_quality": args.jpg_quality,
        "interval_ms": args.interval_ms,
        "max_captures": args.max_captures,
        "mask_polygon": bool(args.mask_polygon),
        "camera": {
            "id": args.camera_id,
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
        },
        "slots": {
            slot: {
                "src_points": [[float(x), float(y)] for x, y in rois[slot]],
            }
            for slot in SLOT_ORDER
        },
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    cap = open_camera(device=args.camera_id, width=args.width, height=args.height, fps=args.fps)

    auto_capture = True
    show_rois = bool(args.show_rois)
    capture_count = 0
    interval_s = args.interval_ms / 1000.0
    last_capture_ts = 0.0

    manifest_file = manifest_path.open("a", encoding="utf-8")

    print(f"Session: {session_id}")
    print(f"Output:  {session_dir}")
    print(f"ROIs:    {args.rois_path}")
    print("Controls: q quit | p pause/resume auto | s save now | g toggle roi overlay")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read frame from camera.")
                break

            now = time.monotonic()
            should_capture = auto_capture and (now - last_capture_ts) >= interval_s

            if should_capture:
                capture_count += 1
                slot_paths: Dict[SlotName, str] = {}

                for slot in SLOT_ORDER:
                    crop = extract_polygon_crop(frame, rois[slot], mask_polygon=args.mask_polygon)
                    image_name = f"{args.prefix}_{capture_count:06d}.{args.image_format}"
                    image_path = slot_dirs[slot] / image_name
                    if args.image_format == "jpg":
                        ok_write = cv2.imwrite(str(image_path), crop, [cv2.IMWRITE_JPEG_QUALITY, args.jpg_quality])
                    else:
                        ok_write = cv2.imwrite(str(image_path), crop)
                    if not ok_write:
                        raise RuntimeError(f"Failed to save crop image: {image_path}")
                    slot_paths[slot] = f"images/{slot}/{image_name}"

                entry = {
                    "session_id": session_id,
                    "capture_index": capture_count,
                    "captured_utc": utc_now_iso(),
                    "slots": slot_paths,
                }
                manifest_file.write(json.dumps(entry) + "\n")
                manifest_file.flush()
                last_capture_ts = now
                print(f"[{capture_count}] saved 5 card crops")

                if args.max_captures > 0 and capture_count >= args.max_captures:
                    print("Reached max_captures. Stopping.")
                    break

            if not args.no_preview:
                preview = frame
                if show_rois:
                    preview = draw_card_rois_overlay(preview, rois)
                status = "AUTO" if auto_capture else "PAUSED"
                cv2.putText(
                    preview,
                    f"{status} count={capture_count}",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    preview,
                    "q quit | p pause/resume | s save | g overlay",
                    (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.56,
                    (0, 255, 255),
                    1,
                )
                cv2.imshow("capture_card_dataset", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                auto_capture = not auto_capture
            if key == ord("g"):
                show_rois = not show_rois
            if key == ord("s"):
                capture_count += 1
                slot_paths: Dict[SlotName, str] = {}
                for slot in SLOT_ORDER:
                    crop = extract_polygon_crop(frame, rois[slot], mask_polygon=args.mask_polygon)
                    image_name = f"{args.prefix}_{capture_count:06d}.{args.image_format}"
                    image_path = slot_dirs[slot] / image_name
                    if args.image_format == "jpg":
                        ok_write = cv2.imwrite(str(image_path), crop, [cv2.IMWRITE_JPEG_QUALITY, args.jpg_quality])
                    else:
                        ok_write = cv2.imwrite(str(image_path), crop)
                    if not ok_write:
                        raise RuntimeError(f"Failed to save crop image: {image_path}")
                    slot_paths[slot] = f"images/{slot}/{image_name}"

                entry = {
                    "session_id": session_id,
                    "capture_index": capture_count,
                    "captured_utc": utc_now_iso(),
                    "slots": slot_paths,
                    "manual": True,
                }
                manifest_file.write(json.dumps(entry) + "\n")
                manifest_file.flush()
                print(f"[{capture_count}] manual save 5 card crops")
                if args.max_captures > 0 and capture_count >= args.max_captures:
                    print("Reached max_captures. Stopping.")
                    break

    finally:
        manifest_file.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
