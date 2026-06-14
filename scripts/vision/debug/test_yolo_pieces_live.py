import argparse
from pathlib import Path
import time

import cv2

from onitama.vision.piece_detector import YoloPieceDetector


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


def class_style(name: str) -> tuple[tuple[int, int, int], int, int, int]:
    # BGR colors tuned for clearer contrast between master and student.
    if name == "red_master":
        return (0, 0, 210), 2, 2, 1
    if name == "red_student":
        return (90, 90, 255), 2, 2, 1
    if name == "blue_master":
        return (220, 70, 0), 2, 2, 1
    if name == "blue_student":
        return (255, 200, 80), 2, 2, 1
    return (255, 255, 255), 2, 2, 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live YOLO test for Onitama pieces (camera + padded board warp).")
    parser.add_argument("--model", type=Path, default=Path("models/pieces_yolo11s_640_best.pt"))
    parser.add_argument("--calibration", type=Path, default=Path("data/vision/board_calibration.json"))
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--padding-ratio",
        type=float,
        default=0.25,
        help="Warp padding ratio. Tuned default for your dataset capture is 0.25 (500 board -> 750 image).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO inference size. Tuned default is 640 (same size used in training dataset).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.50,
        help="Confidence threshold. Tuned default for your setup is 0.50.",
    )
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--max-det", type=int, default=25)
    parser.add_argument("--yolo-device", type=str, default="cpu", help="YOLO device (cpu, cuda:0, ...).")
    parser.add_argument(
        "--anchor-x-ratio",
        type=float,
        default=0.30,
        help="Horizontal anchor inside bbox [0..1]. Tuned default is 0.30 (left-shifted from center).",
    )
    parser.add_argument("--show-grid", action="store_true")
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
    raw_w, raw_h = detector.output_size
    print(f"Loaded model: {detector.model_path}")
    print(f"Class names:   {detector.class_names}")
    print(f"Warp size:     {raw_w}x{raw_h} (before rotate)")
    print(f"Board ROI:     {detector.rotated_roi} (x, y, w, h)")

    cap = open_camera(device=args.camera_id, width=args.width, height=args.height, fps=args.fps)
    show_grid = bool(args.show_grid)

    print("Controls: q quit | g toggle grid")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Could not read frame from camera.")
            break

        warped = detector.warp_frame(frame)
        t0 = time.perf_counter()
        detections = detector.detect_on_warped(warped)
        infer_ms = (time.perf_counter() - t0) * 1000.0

        preview = warped.copy()
        if show_grid:
            preview = draw_grid_roi(preview, detector.rotated_roi, cells=5)

        det_count = len(detections)
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox_xyxy
            color, box_thickness, marker_radius, text_thickness = class_style(detection.class_name)

            p1 = (int(round(x1)), int(round(y1)))
            p2 = (int(round(x2)), int(round(y2)))
            cv2.rectangle(preview, p1, p2, color, box_thickness)

            if detection.cell is not None:
                row, col = detection.cell
                cell_text = f"r{row}c{col}"
            else:
                cell_text = "offboard"

            text = f"{detection.class_name} {detection.confidence:.2f} {cell_text}"
            text_org = (p1[0], max(18, p1[1] - 8))
            cv2.putText(preview, text, text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, text_thickness)

            anchor_x, anchor_y = detection.anchor_xy
            cv2.circle(preview, (int(round(anchor_x)), int(round(anchor_y))), marker_radius, color, -1)

        cv2.putText(
            preview,
            f"dets={det_count}  infer={infer_ms:.1f}ms  imgsz={args.imgsz}  conf={args.conf:.2f}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            preview,
            "q quit | g toggle grid",
            (10, preview.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1,
        )

        cv2.imshow("onitama_yolo_live", preview)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("g"):
            show_grid = not show_grid

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
