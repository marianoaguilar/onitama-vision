from pathlib import Path
from typing import List, Tuple

import cv2
from onitama.vision.homography import HomographyCalibration, compute_homography_matrix, draw_grid, warp_board


def decode_fourcc(v: float) -> str:
    v = int(v)
    return "".join([chr((v >> (8 * i)) & 0xFF) for i in range(4)])


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

    # Load existing calibration if available
    calib = None
    try:
        calib = HomographyCalibration.load(out_path)
        print(f"Loaded existing calibration from: {out_path}")
    except FileNotFoundError:
        print("No existing calibration found. Press 'c' to calibrate.")

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
            if calib is not None:
                points = calib.src_points
                for i, (px, py) in enumerate(points, start=1):
                    cv2.circle(display, (int(px), int(py)), 8, (0, 255, 255), -1)
                    cv2.putText(display, str(i), (int(px) + 10, int(py) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                # Draw lines connecting the points
                if len(points) == 4:
                    for i in range(4):
                        pt1 = (int(points[i][0]), int(points[i][1]))
                        pt2 = (int(points[(i+1) % 4][0]), int(points[(i+1) % 4][1]))
                        cv2.line(display, pt1, pt2, (0, 255, 255), 2)
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
            calib = HomographyCalibration(
                src_points=tuple(clicked),
                dst_size=(dst_w, dst_h),
                rotate=rotate,
            )
            M = compute_homography_matrix(calib)
            warped = warp_board(frozen, calib, M=M)
            warped = draw_grid(warped, cells=5)

            cv2.imshow("warped", warped)
            print("Check alignment. Press any key to save, or close window and recalibrate.")
            cv2.waitKey(0)
            cv2.destroyWindow("warped")

            calib.save(out_path)
            print(f"Saved calibration to: {out_path}")

            frozen = None
            clicked = []

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
