from pathlib import Path

import cv2

from onitama.vision.homography import (
    HomographyCalibration,
    compute_homography_matrix,
    draw_grid,
    warp_board,
    xy_to_cell,
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


def main() -> None:
    calib = HomographyCalibration.load(Path("data/vision/calibration.json"))
    M = compute_homography_matrix(calib)

    cap = open_camera()

    last_warped = None
    click_pos = None
    click_cell = None

    def on_mouse(event, x, y, flags, param):
        nonlocal click_pos, click_cell
        if event == cv2.EVENT_LBUTTONDOWN:
            click_pos = (x, y)
            click_cell = xy_to_cell(x, y, board_size=5, dst_size=calib.dst_size)
            print(f"Clicked warped (x={x}, y={y}) -> cell row={click_cell[0]}, col={click_cell[1]}")

    cv2.namedWindow("warped")
    cv2.setMouseCallback("warped", on_mouse)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        warped = warp_board(frame, calib, M=M)
        warped = draw_grid(warped, cells=5)

        if click_pos is not None:
            cv2.circle(warped, click_pos, 6, (0, 0, 255), -1)
            cv2.putText(
                warped,
                f"row={click_cell[0]} col={click_cell[1]}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )

        cv2.imshow("warped", warped)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()