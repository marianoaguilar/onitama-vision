import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from onitama.cli.render import render_state
from onitama.engine.pieces import Player
from onitama.engine.state import GameState
from onitama.vision.board import VisionBoard, VisionPiece
from onitama.vision.card_classifier import YoloCardClassifier
from onitama.vision.card_rois import SLOT_COLOR, draw_card_rois_overlay, quad_centroid
from onitama.vision.piece_detector import YoloPieceDetector
from onitama.vision.snapshot import VisionSnapshot
from onitama.vision.vision_pipeline import VisionPipeline


_MONO_FONT = ImageFont.truetype("DejaVuSansMono.ttf", 16)


def open_camera(device: int = 0, width: int = 1280, height: int = 720, fps: int = 30) -> cv2.VideoCapture:
    """Open the camera with the same settings used in the other live scripts."""
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera device {device}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def _parse_player(value: str) -> Player:
    """Parse RED or BLUE from the command line."""
    upper = value.strip().upper()
    if upper == "RED":
        return Player.RED
    if upper == "BLUE":
        return Player.BLUE
    raise ValueError("to-move must be RED or BLUE.")


def _toggle_player(player: Player) -> Player:
    """Swap the current player to move."""
    return Player.BLUE if player is Player.RED else Player.RED


def draw_grid_roi(img: np.ndarray, roi: tuple[int, int, int, int], cells: int = 5) -> np.ndarray:
    """Draw the 5x5 board grid over the board ROI."""
    x, y, w, h = roi
    out = img.copy()
    cv2.rectangle(out, (x, y), (x + w - 1, y + h - 1), (0, 255, 255), 2)
    for i in range(1, cells):
        gx = x + int(i * w / cells)
        gy = y + int(i * h / cells)
        cv2.line(out, (gx, y), (gx, y + h - 1), (0, 255, 0), 1)
        cv2.line(out, (x, gy), (x + w - 1, gy), (0, 255, 0), 1)
    return out


def resize_for_display(image: np.ndarray, scale: float) -> np.ndarray:
    """Resize an image for display keeping its aspect ratio."""
    if scale <= 0.0:
        raise ValueError("display scale must be > 0.")
    if scale == 1.0:
        return image
    h, w = image.shape[:2]
    out_w = max(1, int(round(w * scale)))
    out_h = max(1, int(round(h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, (out_w, out_h), interpolation=interpolation)


def piece_overlay_style(piece: VisionPiece | None) -> tuple[str, tuple[int, int, int]]:
    """Return a short token and color for one board cell."""
    if piece is VisionPiece.RED_MASTER:
        return "RM", (0, 0, 210)
    if piece is VisionPiece.RED_STUDENT:
        return "RS", (90, 90, 255)
    if piece is VisionPiece.BLUE_MASTER:
        return "BM", (220, 70, 0)
    if piece is VisionPiece.BLUE_STUDENT:
        return "BS", (255, 200, 80)
    return "..", (160, 160, 160)


def draw_board_overlay(img: np.ndarray, board: VisionBoard, roi: tuple[int, int, int, int]) -> np.ndarray:
    """Draw the discrete VisionBoard on top of the warped board image."""
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


def draw_card_predictions(frame: np.ndarray, snapshot: VisionSnapshot, classifier: YoloCardClassifier) -> np.ndarray:
    """Draw card ROIs and the predicted card name for each slot."""
    preview = draw_card_rois_overlay(frame, classifier.rois)
    slot_to_name = {
        "red_0": snapshot.red_cards[0],
        "red_1": snapshot.red_cards[1],
        "side": snapshot.side_card,
        "blue_0": snapshot.blue_cards[0],
        "blue_1": snapshot.blue_cards[1],
    }

    for slot, card_name in slot_to_name.items():
        cx, cy = quad_centroid(classifier.rois[slot])
        cv2.putText(
            preview,
            card_name,
            (int(round(cx - 55.0)), int(round(cy + 24.0))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            SLOT_COLOR[slot],
            2,
        )
    return preview


def build_state_panel(
    state: GameState | None,
    *,
    snapshot: VisionSnapshot | None,
    to_move: Player,
    mode: str,
    pipeline_ms: float,
    error_message: str | None,
    width: int = 720,
    height: int = 430,
) -> np.ndarray:
    """Render the current GameState in a monospace text panel."""
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (18, 18, 18)
    pil_image = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    y = 64
    draw.text(
        (16, 10),
        f"{mode}  pipeline={pipeline_ms:.1f}ms  to_move={to_move.value}",
        font=_MONO_FONT,
        fill=(255, 255, 0),
    )

    if error_message:
        draw.text((16, y), "Last error:", font=_MONO_FONT, fill=(255, 150, 0))
        y += 30
        for line in error_message.splitlines():
            draw.text((16, y), line[:80], font=_MONO_FONT, fill=(220, 220, 220))
            y += 26
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    if snapshot is not None:
        draw.text(
            (16, y),
            f"Cards: RED={snapshot.red_cards}  BLUE={snapshot.blue_cards}  SIDE={snapshot.side_card}",
            font=_MONO_FONT,
            fill=(220, 220, 220),
        )
        y += 32

    if state is None:
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    for line in render_state(state).splitlines():
        if not line:
            y += 14
            continue
        draw.text((16, y), line, font=_MONO_FONT, fill=(220, 220, 220))
        y += 24
        if y >= height - 12:
            break
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the live pipeline test."""
    parser = argparse.ArgumentParser(
        description="Live test for the full Onitama VisionPipeline: frame -> snapshot -> GameState."
    )
    parser.add_argument("--piece-model", type=Path, default=Path("models/pieces_yolov8s_640_best.pt"))
    parser.add_argument("--card-model", type=Path, default=Path("models/cards_yolo11n-cls_320_best.pt"))
    parser.add_argument("--calibration", type=Path, default=Path("data/vision/calibration.json"))
    parser.add_argument("--rois-path", type=Path, default=Path("data/vision/card_rois.json"))
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--to-move", default="RED")
    parser.add_argument("--out-path", type=Path, default=Path("data/vision/snapshot.json"))
    parser.add_argument("--display-scale", type=float, default=0.8)
    parser.add_argument("--padding-ratio", type=float, default=0.25)
    parser.add_argument("--piece-imgsz", type=int, default=640)
    parser.add_argument("--card-imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=25)
    parser.add_argument("--yolo-device", type=str, default="cpu")
    parser.add_argument("--anchor-x-ratio", type=float, default=0.30)
    parser.add_argument(
        "--mask-polygon",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply polygon masking before card classification.",
    )
    parser.add_argument("--show-grid", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    current_to_move = _parse_player(args.to_move)

    piece_detector = YoloPieceDetector(
        model_path=args.piece_model,
        calibration_path=args.calibration,
        padding_ratio=args.padding_ratio,
        imgsz=args.piece_imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        yolo_device=args.yolo_device,
        anchor_x_ratio=args.anchor_x_ratio,
    )
    card_classifier = YoloCardClassifier(
        model_path=args.card_model,
        rois_path=args.rois_path,
        imgsz=args.card_imgsz,
        yolo_device=args.yolo_device,
        mask_polygon=args.mask_polygon,
    )
    pipeline = VisionPipeline(piece_detector=piece_detector, card_classifier=card_classifier)

    print(f"Piece model: {piece_detector.model_path}")
    print(f"Card model:  {card_classifier.model_path}")
    print(f"To move:     {current_to_move.value}")
    print("Controls: q quit | f freeze/live | j save snapshot | t toggle to_move | g toggle grid")

    cap = open_camera(device=args.camera_id, width=args.width, height=args.height, fps=args.fps)
    frozen_frame: np.ndarray | None = None
    show_grid = bool(args.show_grid)

    while True:
        if frozen_frame is None:
            ok, frame = cap.read()
            if not ok:
                print("Could not read frame from camera.")
                break
            current = frame
        else:
            current = frozen_frame

        mode = "FROZEN" if frozen_frame is not None else "LIVE"
        error_message: str | None = None
        snapshot: VisionSnapshot | None = None
        game_state: GameState | None = None

        t0 = time.perf_counter()
        try:
            snapshot = pipeline.snapshot_from_frame(current)
            game_state = snapshot.to_game_state(to_move=current_to_move)
        except Exception as exc:
            error_message = str(exc)
        pipeline_ms = (time.perf_counter() - t0) * 1000.0

        warped = piece_detector.warp_frame(current)
        if snapshot is not None:
            board_preview = draw_board_overlay(warped, snapshot.board, piece_detector.rotated_roi)
            cards_preview = draw_card_predictions(current, snapshot, card_classifier)
        else:
            board_preview = warped.copy()
            cards_preview = current.copy()

        if show_grid:
            board_preview = draw_grid_roi(board_preview, piece_detector.rotated_roi, cells=5)

        cv2.putText(
            board_preview,
            f"{mode} pipeline={pipeline_ms:.1f}ms",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            board_preview,
            "q quit | f freeze/live | j save | t toggle to_move | g grid",
            (10, board_preview.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1,
        )
        cv2.putText(
            cards_preview,
            f"{mode} to_move={current_to_move.value}",
            (10, cards_preview.shape[0] - 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            cards_preview,
            "q quit | f freeze/live | j save | t toggle to_move",
            (10, cards_preview.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1,
        )

        state_panel = build_state_panel(
            game_state,
            snapshot=snapshot,
            to_move=current_to_move,
            mode=mode,
            pipeline_ms=pipeline_ms,
            error_message=error_message,
        )

        cv2.imshow("onitama_pipeline_board", resize_for_display(board_preview, args.display_scale))
        cv2.imshow("onitama_pipeline_cards", resize_for_display(cards_preview, args.display_scale))
        cv2.imshow("onitama_pipeline_state", resize_for_display(state_panel, args.display_scale))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("f"):
            if frozen_frame is None:
                frozen_frame = current.copy()
                print("Frame frozen.")
            else:
                frozen_frame = None
                print("Live view resumed.")
        if key == ord("g"):
            show_grid = not show_grid
        if key == ord("t"):
            current_to_move = _toggle_player(current_to_move)
            print(f"to_move -> {current_to_move.value}")
        if key == ord("j"):
            if snapshot is None:
                print("No valid snapshot to save.")
            else:
                snapshot.save_json(args.out_path)
                print(f"Saved snapshot JSON to: {args.out_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
