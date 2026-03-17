import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from onitama.vision.card_classifier import CardClassificationResult, CardSlotPrediction, YoloCardClassifier
from onitama.vision.card_rois import SLOT_COLOR, SLOT_LABEL, SLOT_ORDER, draw_card_rois_overlay, quad_centroid


def open_camera(device: int = 0, width: int = 1280, height: int = 720, fps: int = 30) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera device {device}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def _fit_thumbnail(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.size == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)

    src_h, src_w = image.shape[:2]
    scale = min(width / float(src_w), height / float(src_h))
    out_w = max(1, int(round(src_w * scale)))
    out_h = max(1, int(round(src_h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (out_w, out_h), interpolation=interpolation)

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    offset_x = (width - out_w) // 2
    offset_y = (height - out_h) // 2
    canvas[offset_y:offset_y + out_h, offset_x:offset_x + out_w] = resized
    return canvas


def _to_float_list(values: Any) -> list[float]:
    if hasattr(values, "cpu"):
        return [float(v) for v in values.cpu().tolist()]
    return [float(v) for v in values]


def build_debug_batch(
    frame: np.ndarray,
    classifier: YoloCardClassifier,
    *,
    topk: int,
) -> dict[str, object]:
    raw_crops = classifier.extract_card_crops(frame)
    prepared_crops = classifier.prepare_crops(raw_crops)

    t0 = time.perf_counter()
    raw_results = classifier.predict_prepared_crops(prepared_crops)
    infer_ms = (time.perf_counter() - t0) * 1000.0

    predictions: list[CardSlotPrediction] = []
    top_candidates_by_slot: dict[str, list[tuple[str, float]]] = {}
    for slot, raw_result in zip(SLOT_ORDER, raw_results):
        probs = raw_result.probs
        if probs is None:
            raise RuntimeError(f"Classification result for slot '{slot}' does not contain probabilities.")

        top1_index = int(probs.top1)
        top1_conf = float(probs.top1conf.item() if hasattr(probs.top1conf, "item") else probs.top1conf)
        predictions.append(
            CardSlotPrediction(
                slot=slot,
                class_name=classifier.class_names.get(top1_index, str(top1_index)),
                confidence=top1_conf,
            )
        )

        top_indices = [int(idx) for idx in list(probs.top5)[:topk]]
        top_confidences = _to_float_list(probs.top5conf)[:topk]
        top_candidates_by_slot[slot] = [
            (classifier.class_names.get(class_index, str(class_index)), float(top_confidences[pos]))
            for pos, class_index in enumerate(top_indices)
        ]

    return {
        "raw_crops": raw_crops,
        "prepared_crops": prepared_crops,
        "result": CardClassificationResult(predictions=tuple(predictions)),
        "infer_ms": infer_ms,
        "top_candidates": top_candidates_by_slot,
    }


def draw_prediction_overlay(frame: np.ndarray, classifier: YoloCardClassifier, batch: dict[str, object]) -> np.ndarray:
    preview = draw_card_rois_overlay(frame, classifier.rois)
    result = batch["result"]
    assert isinstance(result, CardClassificationResult)
    by_slot = result.by_slot()

    for slot in SLOT_ORDER:
        prediction = by_slot[slot]
        color = SLOT_COLOR[slot]
        cx, cy = quad_centroid(classifier.rois[slot])
        label = f"{prediction.class_name} {prediction.confidence:.2f}"
        cv2.putText(
            preview,
            label,
            (int(round(cx - 55.0)), int(round(cy + 24.0))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
        )
    return preview


def build_dashboard(batch: dict[str, object]) -> np.ndarray:
    result = batch["result"]
    infer_ms = batch["infer_ms"]
    raw_crops = batch["raw_crops"]
    prepared_crops = batch["prepared_crops"]
    top_candidates = batch["top_candidates"]
    assert isinstance(result, CardClassificationResult)
    assert isinstance(infer_ms, float)
    assert isinstance(raw_crops, dict)
    assert isinstance(prepared_crops, dict)
    assert isinstance(top_candidates, dict)

    row_h = 180
    header_h = 38
    width = 980
    dashboard = np.zeros((header_h + row_h * len(SLOT_ORDER), width, 3), dtype=np.uint8)
    dashboard[:] = (20, 20, 20)

    cv2.putText(
        dashboard,
        f"Top-1 predictions | infer={infer_ms:.1f}ms",
        (16, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 255, 255),
        2,
    )

    by_slot = result.by_slot()
    for idx, slot in enumerate(SLOT_ORDER):
        y0 = header_h + idx * row_h
        y1 = y0 + row_h
        panel = dashboard[y0:y1]
        panel[:] = (28, 28, 28)

        color = SLOT_COLOR[slot]
        prediction = by_slot[slot]

        raw_thumb = _fit_thumbnail(raw_crops[slot], width=280, height=150)
        prep_thumb = _fit_thumbnail(prepared_crops[slot], width=150, height=150)
        panel[15:165, 12:292] = raw_thumb
        panel[15:165, 320:470] = prep_thumb
        cv2.rectangle(panel, (11, 14), (292, 165), (90, 90, 90), 1)
        cv2.rectangle(panel, (319, 14), (470, 165), (90, 90, 90), 1)

        cv2.putText(panel, "raw ROI", (105, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)
        cv2.putText(panel, "320x320 input", (326, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)

        cv2.putText(panel, SLOT_LABEL[slot], (500, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)
        cv2.putText(
            panel,
            f"top1: {prediction.class_name} ({prediction.confidence:.3f})",
            (500, 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (235, 235, 235),
            2,
        )
        cv2.putText(
            panel,
            f"crop={raw_crops[slot].shape[1]}x{raw_crops[slot].shape[0]}",
            (500, 96),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (190, 190, 190),
            1,
        )

        for line_idx, (class_name, confidence) in enumerate(top_candidates[slot][:3], start=1):
            text = f"top{line_idx}: {class_name} ({confidence:.3f})"
            cv2.putText(
                panel,
                text,
                (500, 96 + line_idx * 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                (190, 190, 190),
                1,
            )

        cv2.line(panel, (0, row_h - 1), (width - 1, row_h - 1), (55, 55, 55), 1)

    return dashboard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live YOLO classification test for Onitama cards using calibrated card ROIs."
    )
    parser.add_argument("--model", type=Path, default=Path("models/cards_yolo11n-cls_320_best.pt"))
    parser.add_argument("--rois-path", type=Path, default=Path("data/vision/card_rois.json"))
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=320, help="Input size used by the classifier.")
    parser.add_argument("--topk", type=int, default=5, help="Number of top candidates shown in the debug dashboard.")
    parser.add_argument("--yolo-device", type=str, default="cpu", help="YOLO device (cpu, cuda:0, ...).")
    parser.add_argument(
        "--mask-polygon",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply polygon masking before classification. Enabled by default to match dataset capture.",
    )
    parser.add_argument("--out-path", type=Path, default=Path("data/vision/auto_card_predictions.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classifier = YoloCardClassifier(
        model_path=args.model,
        rois_path=args.rois_path,
        imgsz=args.imgsz,
        yolo_device=args.yolo_device,
        mask_polygon=args.mask_polygon,
    )

    print(f"Loaded model: {classifier.model_path}")
    print(f"Class names:   {classifier.class_names}")
    print(f"ROIs path:     {classifier.rois_path}")
    print(f"Mask polygon:  {classifier.mask_polygon}")
    print("Controls: q quit | f freeze/live | p print predictions | j save predictions json")

    cap = open_camera(device=args.camera_id, width=args.width, height=args.height, fps=args.fps)
    frozen_frame: np.ndarray | None = None

    while True:
        if frozen_frame is None:
            ok, frame = cap.read()
            if not ok:
                print("Could not read frame from camera.")
                break
            current = frame
        else:
            current = frozen_frame

        batch = build_debug_batch(current, classifier, topk=args.topk)
        preview = draw_prediction_overlay(current, classifier, batch)
        dashboard = build_dashboard(batch)
        result = batch["result"]
        infer_ms = batch["infer_ms"]
        assert isinstance(result, CardClassificationResult)
        assert isinstance(infer_ms, float)

        mode = "FROZEN" if frozen_frame is not None else "LIVE"
        cv2.putText(
            preview,
            f"{mode} infer={infer_ms:.1f}ms imgsz={classifier.imgsz} mask={int(classifier.mask_polygon)}",
            (10, preview.shape[0] - 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            preview,
            "q quit | f freeze/live | p print | j save json",
            (10, preview.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 255, 255),
            1,
        )

        cv2.imshow("onitama_yolo_cards_live", preview)
        cv2.imshow("onitama_yolo_cards_dashboard", dashboard)

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
        if key == ord("p"):
            red_cards, blue_cards, side_card = result.cards_layout()
            print("Current card predictions:")
            print(json.dumps(result.to_dict(), indent=2))
            print(f"Snapshot layout: red={red_cards} blue={blue_cards} side={side_card}")
        if key == ord("j"):
            result.save_json(args.out_path)
            print(f"Saved card predictions JSON to: {args.out_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
