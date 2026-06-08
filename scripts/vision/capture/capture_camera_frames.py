import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2


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
    parser = argparse.ArgumentParser(description="Capture raw camera frames manually.")
    parser.add_argument("--output-root", type=Path, default=Path("data/vision/raw_frames"))
    parser.add_argument("--session-id", type=str, default="")
    parser.add_argument("--prefix", type=str, default="img")
    parser.add_argument("--image-format", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--jpg-quality", type=int, default=95)
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session_id = args.session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = args.output_root / session_id
    images_dir = session_dir / "images"
    manifest_path = session_dir / "manifest.jsonl"
    meta_path = session_dir / "session_meta.json"
    images_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "session_id": session_id,
        "created_utc": utc_now_iso(),
        "image_format": args.image_format,
        "jpg_quality": args.jpg_quality,
        "camera": {
            "id": args.camera_id,
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    cap = open_camera(device=args.camera_id, width=args.width, height=args.height, fps=args.fps)
    capture_count = 0
    manifest_file = manifest_path.open("a", encoding="utf-8")

    print(f"Session: {session_id}")
    print(f"Output:  {session_dir}")
    print("Controls: q quit | s save frame")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read frame from camera.")
                break

            preview = frame.copy()
            cv2.putText(
                preview,
                f"RAW count={capture_count}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                preview,
                "q quit | s save frame",
                (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                (0, 255, 255),
                1,
            )
            cv2.imshow("capture_camera_frames", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                capture_count += 1
                image_name = f"{args.prefix}_{capture_count:06d}.{args.image_format}"
                image_path = images_dir / image_name
                if args.image_format == "jpg":
                    write_ok = cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpg_quality])
                else:
                    write_ok = cv2.imwrite(str(image_path), frame)

                if write_ok:
                    entry = {
                        "session_id": session_id,
                        "capture_index": capture_count,
                        "image": f"images/{image_name}",
                        "captured_utc": utc_now_iso(),
                        "frame_size": {"w": frame.shape[1], "h": frame.shape[0]},
                    }
                    manifest_file.write(json.dumps(entry) + "\n")
                    manifest_file.flush()
                    print(f"[{capture_count}] saved {image_name}")
                else:
                    capture_count -= 1
                    print(f"Failed to save image: {image_path}")
    finally:
        manifest_file.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
