import json
from pathlib import Path

import cv2

from onitama.vision.board import VisionBoard, VisionPiece
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


def _piece_color(piece: VisionPiece) -> tuple[int, int, int]:
    if piece in (VisionPiece.RED_MASTER, VisionPiece.RED_STUDENT):
        return (0, 0, 255)
    return (255, 0, 0)


def draw_state_overlay(img, state: VisionBoard, board_size: int = 5):
    out = img.copy()
    h, w = out.shape[:2]
    cell_w = w / board_size
    cell_h = h / board_size

    for r in range(board_size):
        for c in range(board_size):
            piece = state.board[r][c]
            if piece is None:
                continue

            x = int((c + 0.5) * cell_w)
            y = int((r + 0.5) * cell_h)

            cv2.putText(
                out,
                piece.short,
                (x - 20, y + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                _piece_color(piece),
                2,
            )
    return out


def main() -> None:
    calib = HomographyCalibration.load(Path("assets/vision/calibration.json"))
    M = compute_homography_matrix(calib)

    cap = open_camera()
    state = VisionBoard.empty()
    selected: VisionPiece | None = VisionPiece.RED_STUDENT
    frozen_frame = None

    def selected_label() -> str:
        if selected is None:
            return "EMPTY"
        return selected.value.upper()

    def on_mouse(event, x, y, flags, param):
        nonlocal state
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        row, col = xy_to_cell(x, y, board_size=5, dst_size=calib.dst_size)
        state = state.with_cell(row, col, selected)
        print(f"Set cell ({row}, {col}) -> {selected_label()}")

    cv2.namedWindow("manual_state")
    cv2.setMouseCallback("manual_state", on_mouse)

    help_1 = "1:RM 2:RS 3:BM 4:BS 0:EMPTY | click: set"
    help_2 = "p:print json  j:save json  c:clear  f:freeze  q:quit"
    out_path = Path("assets/vision/manual_piece_board.json")

    while True:
        if frozen_frame is None:
            ret, frame = cap.read()
            if not ret:
                break
            current = frame
        else:
            current = frozen_frame

        warped = warp_board(current, calib, M=M)
        warped = draw_grid(warped, cells=5)
        warped = draw_state_overlay(warped, state)

        cv2.putText(warped, f"Selected: {selected_label()}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(warped, help_1, (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(warped, help_2, (10, 492), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow("manual_state", warped)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("0"):
            selected = None
        if key == ord("1"):
            selected = VisionPiece.RED_MASTER
        if key == ord("2"):
            selected = VisionPiece.RED_STUDENT
        if key == ord("3"):
            selected = VisionPiece.BLUE_MASTER
        if key == ord("4"):
            selected = VisionPiece.BLUE_STUDENT
        if key == ord("c"):
            state = VisionBoard.empty()
            print("Board cleared.")
        if key == ord("f"):
            if frozen_frame is None:
                frozen_frame = current.copy()
                print("Frame frozen.")
            else:
                frozen_frame = None
                print("Live view resumed.")
        if key == ord("p"):
            print("Current state (pretty):")
            print(state.pretty())
            print("Current state (json):")
            print(json.dumps(state.to_dict(), indent=2))
        if key == ord("j"):
            state.save_json(out_path)
            print(f"Saved manual state to: {out_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
