import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class Calibration:
    src_points: List[Tuple[float, float]]  # 4 points in camera image
    dst_size: Tuple[int, int]              # (width, height)
    rotate: int = 0                        # 0/90/180/270

    def to_dict(self) -> dict:
        return {
            "src_points": self.src_points,
            "dst_size": list(self.dst_size),
            "rotate": self.rotate,
        }


def decode_fourcc(v: float) -> str:
    v = int(v)
    return "".join([chr((v >> (8 * i)) & 0xFF) for i in range(4)])


def order_points_clockwise(pts: np.ndarray) -> np.ndarray:
    """
    Returns points in order: top-left, top-right, bottom-right, bottom-left.
    Works even if the user clicks in random order.
    """
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def compute_homography(src_points: List[Tuple[float, float]], dst_w: int, dst_h: int) -> np.ndarray:
    src = np.array(src_points, dtype=np.float32)
    src = order_points_clockwise(src)

    dst = np.array(
        [(0, 0), (dst_w - 1, 0), (dst_w - 1, dst_h - 1), (0, dst_h - 1)],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(src, dst)


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
    raise ValueError("rotate must be one of {0, 90, 180, 270}")


def draw_grid(img: np.ndarray, cells: int = 5) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy()
    for i in range(1, cells):
        x = int(i * w / cells)
        y = int(i * h / cells)
        cv2.line(out, (x, 0), (x, h - 1), (0, 255, 0), 1)
        cv2.line(out, (0, y), (w - 1, y), (0, 255, 0), 1)
    return out


def open_camera(device: int = 0, width: int = 1280, height: int = 720, fps: int = 30) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera device {device}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))
    reported_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera opened: {actual_w}x{actual_h} FOURCC={fourcc} FPS={reported_fps}")
    return cap


def main() -> None:
    out_path = Path("data/vision/calibration.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dst_w, dst_h = 500, 500
    rotate = 90   # it depends on how your camera is mounted. Adjust if the output looks rotated.

    cap = open_camera()

    clicked: List[Tuple[float, float]] = []
    frozen = None

    def on_mouse(event, x, y, flags, param):
        nonlocal clicked
        if frozen is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked) < 4:
            clicked.append((float(x), float(y)))
            print(f"Clicked {len(clicked)}/4: ({x}, {y})")

    cv2.namedWindow("camera")
    cv2.setMouseCallback("camera", on_mouse)

    while True:
        if frozen is None:
            ret, frame = cap.read()
            if not ret:
                break

            display = frame.copy()
            cv2.putText(display, "Press 'c' to freeze. 'q' quit.",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.imshow("camera", display)
        else:
            display = frozen.copy()
            for i, (px, py) in enumerate(clicked, start=1):
                cv2.circle(display, (int(px), int(py)), 6, (0, 0, 255), -1)
                cv2.putText(display, str(i), (int(px) + 8, int(py) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.putText(display, "Click 4 corners. 'r' reset, 's' test, 'q' quit.",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("camera", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        if key == ord("c") and frozen is None:
            frozen = frame.copy()
            clicked = []

        if key == ord("r") and frozen is not None:
            clicked = []

        if key == ord("s") and frozen is not None and len(clicked) == 4:
            M = compute_homography(clicked, dst_w, dst_h)
            warped = cv2.warpPerspective(frozen, M, (dst_w, dst_h))
            warped = apply_rotation(warped, rotate)
            warped = draw_grid(warped, cells=5)

            cv2.imshow("warped", warped)
            print("Check alignment. Press any key to save, or close window and recalibrate.")
            cv2.waitKey(0)
            cv2.destroyWindow("warped")

            calib = Calibration(src_points=clicked, dst_size=(dst_w, dst_h), rotate=rotate)
            out_path.write_text(json.dumps(calib.to_dict(), indent=2), encoding="utf-8")
            print(f"Saved calibration to: {out_path}")

            frozen = None
            clicked = []

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()