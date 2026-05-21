import argparse
import json
from pathlib import Path
import time

import cv2

from onitama.vision.board import VisionBoard, VisionPiece
from onitama.vision.piece_detector import PieceDetection, YoloPieceDetector


def open_camera(device: int = 0, width: int = 1280, height: int = 720, fps: int = 30) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera device {device}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def draw_grid_roi(img, roi: tuple[int, int, int, int], cells: int = 5):
    x, y, w, h = roi
    out = img.copy()
    cv2.rectangle(out, (x, y), (x + w - 1, y + h - 1), (0, 255, 255), 2)
    for i in range(1, cells):
        gx = x + int(i * w / cells)
        gy = y + int(i * h / cells)
        cv2.line(out, (gx, y), (gx, y + h - 1), (0, 255, 0), 1)
        cv2.line(out, (x, gy), (x + w - 1, gy), (0, 255, 0), 1)
    return out


def class_style(name: str) -> tuple[tuple[int, int, int], int, int]:
    if name == "red_master":
        return (0, 0, 210), 2, 1
    if name == "red_student":
        return (90, 90, 255), 2, 1
    if name == "blue_master":
        return (220, 70, 0), 2, 1
    if name == "blue_student":
        return (255, 200, 80), 2, 1
    return (255, 255, 255), 2, 1


def piece_overlay_style(piece: VisionPiece | None) -> tuple[str, tuple[int, int, int]]:
    if piece is VisionPiece.RED_MASTER:
        return "RM", (0, 0, 210)
    if piece is VisionPiece.RED_STUDENT:
        return "RS", (90, 90, 255)
    if piece is VisionPiece.BLUE_MASTER:
        return "BM", (220, 70, 0)
    if piece is VisionPiece.BLUE_STUDENT:
        return "BS", (255, 200, 80)
    return "..", (160, 160, 160)


def draw_board_overlay(img, board: VisionBoard, roi: tuple[int, int, int, int]):
    out = img.copy()
    x, y, w, h = roi
    cell_w = w / 5.0
    cell_h = h / 5.0
    for row in range(5):
        for col in range(5):
            piece = board.board[row][col]
            token, color = piece_overlay_style(piece)
            px = int(round(x + (col + 0.5) * cell_w))
            py = int(round(y + (row + 0.5) * cell_h))
            cv2.putText(out, token, (px - 14, py + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return out


def draw_detections(img, detections: list[PieceDetection]):
    out = img.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox_xyxy
        color, box_thickness, text_thickness = class_style(detection.class_name)
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(out, p1, p2, color, box_thickness)

        if detection.cell is None:
            cell_text = "offboard"
        else:
            row, col = detection.cell
            cell_text = f"r{row}c{col}"

        label = f"{detection.class_name} {detection.confidence:.2f} {cell_text}"
        cv2.putText(out, label, (p1[0], max(18, p1[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, text_thickness)
        ax, ay = detection.anchor_xy
        cv2.circle(out, (int(round(ax)), int(round(ay))), 2, color, -1)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live debug: YOLO detections -> VisionBoard. Saves/prints discrete board before game session."
    )
    parser.add_argument("--model", type=Path, default=Path("models/pieces_yolov8s_640_best.pt"))
    parser.add_argument("--calibration", type=Path, default=Path("data/vision/calibration.json"))
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--padding-ratio", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=25)
    parser.add_argument("--yolo-device", type=str, default="cpu")
    parser.add_argument("--anchor-x-ratio", type=float, default=0.30)
    parser.add_argument("--out-path", type=Path, default=Path("data/vision/auto_piece_board.json"))
    parser.add_argument("--show-grid", action="store_true")
    parser.add_argument("--show-dets", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = YoloPieceDetector(
        model_path=args.model,
        calibration_path=args.calibration,
        padding_ratio=args.padding_ratio,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        yolo_device=args.yolo_device,
        anchor_x_ratio=args.anchor_x_ratio,
    )
    cap = open_camera(device=args.camera_id, width=args.width, height=args.height, fps=args.fps)

    print(f"Loaded model: {detector.model_path}")
    print(f"Class names:   {detector.class_names}")
    print(f"Board ROI:     {detector.rotated_roi} (x, y, w, h)")
    print("Controls: q quit | p print board/json | j save board json | g toggle grid | d toggle detections")

    show_grid = bool(args.show_grid)
    show_dets = bool(args.show_dets)
    last_board_json = ""

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Could not read frame from camera.")
            break

        warped = detector.warp_frame(frame)
        t0 = time.perf_counter()
        detections = detector.detect_on_warped(warped)
        infer_ms = (time.perf_counter() - t0) * 1000.0
        board = detector.detections_to_board(detections)

        preview = draw_board_overlay(warped, board, detector.rotated_roi)
        if show_grid:
            preview = draw_grid_roi(preview, detector.rotated_roi, cells=5)
        if show_dets:
            preview = draw_detections(preview, detections)

        cv2.putText(
            preview,
            f"dets={len(detections)} infer={infer_ms:.1f}ms conf={args.conf:.2f}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            preview,
            "q quit | p print | j save | g grid | d detections",
            (10, preview.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1,
        )
        cv2.imshow("onitama_yolo_board_live", preview)

        board_json = json.dumps(board.to_dict(), sort_keys=True)
        if board_json != last_board_json:
            last_board_json = board_json
            print("Board changed:")
            print(board.pretty())

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("g"):
            show_grid = not show_grid
        if key == ord("d"):
            show_dets = not show_dets
        if key == ord("p"):
            print("Current board (pretty):")
            print(board.pretty())
            print("Current board (json):")
            print(json.dumps(board.to_dict(), indent=2))
        if key == ord("j"):
            board.save_json(args.out_path)
            print(f"Saved board JSON to: {args.out_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
