import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from onitama.vision.homography import HomographyCalibration


def open_camera(device: int = 0, width: int = 1280, height: int = 720, fps: int = 30) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera device {device}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def order_points_clockwise(pts: np.ndarray) -> np.ndarray:
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def apply_rotation(img: np.ndarray, rotate: int) -> np.ndarray:
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


def rotate_point(x: int, y: int, width: int, height: int, rotate: int) -> Tuple[int, int]:
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


def rotate_roi(x: int, y: int, w: int, h: int, width: int, height: int, rotate: int) -> Tuple[int, int, int, int]:
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
    calib: HomographyCalibration, padding_ratio: float
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int, int, int]]:
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
    m = cv2.getPerspectiveTransform(src, dst)
    board_roi = (pad_x, pad_y, board_w, board_h)
    return m, (out_w, out_h), board_roi


def draw_roi_grid(img: np.ndarray, roi: Tuple[int, int, int, int], cells: int = 5) -> np.ndarray:
    x, y, w, h = roi
    out = img.copy()
    cv2.rectangle(out, (x, y), (x + w - 1, y + h - 1), (0, 255, 255), 2)

    for i in range(1, cells):
        gx = x + int(i * w / cells)
        gy = y + int(i * h / cells)
        cv2.line(out, (gx, y), (gx, y + h - 1), (0, 255, 0), 1)
        cv2.line(out, (x, gy), (x + w - 1, gy), (0, 255, 0), 1)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview and capture one padded board warp.")
    parser.add_argument("--calibration", type=Path, default=Path("data/vision/calibration.json"))
    parser.add_argument("--output", type=Path, default=Path("data/vision/padded_capture.jpg"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--padding-ratio", type=float, default=0.10)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--max-padding-ratio", type=float, default=0.30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.calibration.exists():
        raise FileNotFoundError(f"Calibration file not found: {args.calibration}")

    calib = HomographyCalibration.load(args.calibration)
    cap = open_camera(device=args.device, width=args.width, height=args.height, fps=args.fps)

    padding_ratio = max(0.0, min(args.max_padding_ratio, float(args.padding_ratio)))
    step = max(0.001, float(args.step))
    cached_ratio: float | None = None
    cached_m: np.ndarray | None = None
    cached_size: Tuple[int, int] | None = None
    cached_roi: Tuple[int, int, int, int] | None = None
    last_preview: np.ndarray | None = None

    print("Controls: '+' increase padding | '-' decrease padding | 's' save | 'q' quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read frame from camera.")
            break

        if cached_ratio != padding_ratio:
            cached_m, cached_size, cached_roi = build_padded_homography(calib, padding_ratio)
            cached_ratio = padding_ratio

        assert cached_m is not None and cached_size is not None and cached_roi is not None
        raw_warp = cv2.warpPerspective(frame, cached_m, cached_size)
        raw_w, raw_h = cached_size

        rotated = apply_rotation(raw_warp, calib.rotate)
        rotated_roi = rotate_roi(
            x=cached_roi[0],
            y=cached_roi[1],
            w=cached_roi[2],
            h=cached_roi[3],
            width=raw_w,
            height=raw_h,
            rotate=calib.rotate,
        )
        preview = draw_roi_grid(rotated, rotated_roi, cells=5)

        cv2.putText(
            preview,
            f"padding={padding_ratio:.3f}  size={preview.shape[1]}x{preview.shape[0]}",
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            preview,
            "Keys: +/- adjust | s save image | q quit",
            (10, preview.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1,
        )

        cv2.imshow("padded_warp_preview", preview)
        last_preview = rotated

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key in (ord("+"), ord("=")):
            padding_ratio = min(args.max_padding_ratio, padding_ratio + step)
        if key in (ord("-"), ord("_")):
            padding_ratio = max(0.0, padding_ratio - step)
        if key == ord("s"):
            if last_preview is None:
                continue
            args.output.parent.mkdir(parents=True, exist_ok=True)
            ok = cv2.imwrite(str(args.output), last_preview)
            if not ok:
                print(f"Failed to save image: {args.output}")
            else:
                print(f"Saved: {args.output}")
                print(f"padding_ratio={padding_ratio:.3f}, board_roi={rotated_roi}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
