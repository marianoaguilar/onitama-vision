from pathlib import Path

import cv2

from onitama.vision.homography import HomographyCalibration, compute_homography_matrix, draw_grid, warp_board


def open_camera(device: int = 0, width: int = 1280, height: int = 720, fps: int = 30) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera device {device}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def main() -> None:
    calib_path = Path("data/vision/calibration.json")
    calib = HomographyCalibration.load(calib_path)
    M = compute_homography_matrix(calib)

    cap = open_camera()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        warped = warp_board(frame, calib, M=M)
        warped = draw_grid(warped, cells=5)

        cv2.imshow("camera", frame)
        cv2.imshow("warped", warped)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()