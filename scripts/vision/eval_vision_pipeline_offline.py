from __future__ import annotations

"""Offline evaluation for the full Onitama vision pipeline.

The script evaluates fixed-camera full-frame images with VisionPipeline only:
frame -> VisionSnapshot. It intentionally does not convert snapshots to
GameState, stabilize observations, synchronize legal moves, or open a camera.
"""

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from onitama.engine.cards import CARD_BY_NAME
from onitama.vision.board import BOARD_SIZE, VisionBoard, VisionPiece
from onitama.vision.card_classifier import YoloCardClassifier
from onitama.vision.piece_detector import YoloPieceDetector
from onitama.vision.snapshot import VisionSnapshot
from onitama.vision.vision_pipeline import VisionPipeline


LIGHTING_LEVELS = ("low_light", "good_light")
CARD_SLOTS = ("red_0", "red_1", "side", "blue_0", "blue_1")
CSV_COLUMNS = (
    "state_id",
    "lighting",
    "image",
    "status",
    "pieces_gt",
    "pieces_pred",
    "pieces_correct",
    "pieces_missing",
    "pieces_extra",
    "pieces_wrong_class",
    "cards_correct",
    "board_exact",
    "cards_exact",
    "snapshot_exact",
    "pipeline_ms",
    "mismatch_details",
    "error_type",
    "error_message",
)


class AnnotationValidationError(ValueError):
    """Raised when the annotation file cannot be evaluated safely."""


@dataclass(frozen=True)
class ExpectedSample:
    state_id: str
    lighting: str
    image_path: Path
    board: VisionBoard
    red_cards: tuple[str, str]
    side_card: str
    blue_cards: tuple[str, str]

    def card_slots(self) -> tuple[str, str, str, str, str]:
        return (
            self.red_cards[0],
            self.red_cards[1],
            self.side_card,
            self.blue_cards[0],
            self.blue_cards[1],
        )


@dataclass(frozen=True)
class BoardComparison:
    pieces_gt: int
    pieces_pred: int
    pieces_correct: int
    pieces_missing: int
    pieces_extra: int
    pieces_wrong_class: int
    board_exact: bool
    cell_mismatches: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class CardComparison:
    cards_correct: int
    cards_exact: bool
    card_mismatches: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class SnapshotComparison:
    board: BoardComparison
    cards: CardComparison
    snapshot_exact: bool
    mismatch_details: str


@dataclass(frozen=True)
class ResultRow:
    state_id: str
    lighting: str
    image: str
    status: str
    pieces_gt: int
    pieces_pred: int
    pieces_correct: int
    pieces_missing: int
    pieces_extra: int
    pieces_wrong_class: int
    cards_correct: int
    board_exact: bool
    cards_exact: bool
    snapshot_exact: bool
    pipeline_ms: float
    mismatch_details: str
    error_type: str
    error_message: str


def _require_object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise AnnotationValidationError(f"{label} must be an object.")
    return value


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AnnotationValidationError(f"{label} must be a non-empty string.")
    return value.strip()


def _parse_card_pair(value: object, label: str) -> tuple[str, str]:
    if not isinstance(value, list) or len(value) != 2:
        raise AnnotationValidationError(f"{label} must contain exactly two cards.")
    first = _require_card_name(value[0], f"{label}[0]")
    second = _require_card_name(value[1], f"{label}[1]")
    return (first, second)


def _require_card_name(value: object, label: str) -> str:
    name = _require_string(value, label)
    if name not in CARD_BY_NAME:
        raise AnnotationValidationError(f"{label} has unknown card name: {name!r}.")
    return name


def _parse_board(value: object, label: str) -> VisionBoard:
    if not isinstance(value, list) or len(value) != BOARD_SIZE:
        raise AnnotationValidationError(f"{label} must be a {BOARD_SIZE}x{BOARD_SIZE} matrix.")

    valid_piece_tokens = {piece.value for piece in VisionPiece}
    rows: list[list[str | None]] = []
    for row_idx, row in enumerate(value):
        if not isinstance(row, list) or len(row) != BOARD_SIZE:
            raise AnnotationValidationError(f"{label}[{row_idx}] must contain {BOARD_SIZE} cells.")
        parsed_row: list[str | None] = []
        for col_idx, cell in enumerate(row):
            if cell is None:
                parsed_row.append(None)
                continue
            if not isinstance(cell, str) or cell not in valid_piece_tokens:
                raise AnnotationValidationError(
                    f"{label}[{row_idx}][{col_idx}] has invalid piece class: {cell!r}."
                )
            parsed_row.append(cell)
        rows.append(parsed_row)
    return VisionBoard.from_board_tokens(rows)


def _read_image_for_validation(path: Path) -> None:
    try:
        import cv2
    except Exception as exc:
        raise AnnotationValidationError("Could not import cv2 to validate evaluation images.") from exc

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise AnnotationValidationError(f"Image cannot be opened with OpenCV: {path}")


def load_and_validate_annotations(path: Path) -> list[ExpectedSample]:
    """Load annotations and validate all samples before model initialization."""
    if not path.exists():
        raise AnnotationValidationError(f"Annotation file not found: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AnnotationValidationError(f"Invalid annotation JSON: {exc}") from exc

    root = _require_object(data, "Annotation root")
    states = root.get("states")
    if not isinstance(states, list):
        raise AnnotationValidationError("'states' must be a list.")

    annotation_dir = path.parent
    seen_state_ids: set[str] = set()
    samples: list[ExpectedSample] = []

    for state_idx, raw_state in enumerate(states):
        state = _require_object(raw_state, f"states[{state_idx}]")
        state_id = _require_string(state.get("id"), f"states[{state_idx}].id")
        if state_id in seen_state_ids:
            raise AnnotationValidationError(f"Duplicate state id: {state_id!r}.")
        seen_state_ids.add(state_id)

        board = _parse_board(state.get("board"), f"state {state_id}.board")
        red_cards = _parse_card_pair(state.get("red_cards"), f"state {state_id}.red_cards")
        blue_cards = _parse_card_pair(state.get("blue_cards"), f"state {state_id}.blue_cards")
        side_card = _require_card_name(state.get("side_card"), f"state {state_id}.side_card")

        visible_cards = [red_cards[0], red_cards[1], side_card, blue_cards[0], blue_cards[1]]
        if len(set(visible_cards)) != len(visible_cards):
            raise AnnotationValidationError(f"state {state_id} must contain five different visible cards.")

        images = _require_object(state.get("images"), f"state {state_id}.images")
        for lighting in LIGHTING_LEVELS:
            rel_path = _require_string(images.get(lighting), f"state {state_id}.images.{lighting}")
            image_path = (annotation_dir / rel_path).resolve()
            if not image_path.exists():
                raise AnnotationValidationError(f"Image file not found: {image_path}")
            _read_image_for_validation(image_path)
            samples.append(
                ExpectedSample(
                    state_id=state_id,
                    lighting=lighting,
                    image_path=image_path,
                    board=board,
                    red_cards=red_cards,
                    side_card=side_card,
                    blue_cards=blue_cards,
                )
            )

    return samples


def compare_boards(expected: VisionBoard, predicted: VisionBoard) -> BoardComparison:
    """Compare the 25 board cells without double-counting wrong classes."""
    pieces_gt = 0
    pieces_pred = 0
    pieces_correct = 0
    pieces_missing = 0
    pieces_extra = 0
    pieces_wrong_class = 0
    mismatches: list[dict[str, object]] = []

    for row_idx in range(BOARD_SIZE):
        for col_idx in range(BOARD_SIZE):
            gt_cell = expected.board[row_idx][col_idx]
            pred_cell = predicted.board[row_idx][col_idx]

            if gt_cell is not None:
                pieces_gt += 1
            if pred_cell is not None:
                pieces_pred += 1

            if gt_cell == pred_cell:
                if gt_cell is not None:
                    pieces_correct += 1
                continue

            mismatch = {
                "row": row_idx,
                "col": col_idx,
                "expected": None if gt_cell is None else gt_cell.value,
                "predicted": None if pred_cell is None else pred_cell.value,
            }
            if gt_cell is not None and pred_cell is None:
                pieces_missing += 1
                mismatch["kind"] = "missing"
            elif gt_cell is None and pred_cell is not None:
                pieces_extra += 1
                mismatch["kind"] = "extra"
            else:
                pieces_wrong_class += 1
                mismatch["kind"] = "wrong_class"
            mismatches.append(mismatch)

    return BoardComparison(
        pieces_gt=pieces_gt,
        pieces_pred=pieces_pred,
        pieces_correct=pieces_correct,
        pieces_missing=pieces_missing,
        pieces_extra=pieces_extra,
        pieces_wrong_class=pieces_wrong_class,
        board_exact=not mismatches,
        cell_mismatches=tuple(mismatches),
    )


def compare_cards(expected: ExpectedSample, predicted: VisionSnapshot) -> CardComparison:
    """Compare card slots in fixed visual order."""
    expected_cards = expected.card_slots()
    predicted_cards = (
        predicted.red_cards[0],
        predicted.red_cards[1],
        predicted.side_card,
        predicted.blue_cards[0],
        predicted.blue_cards[1],
    )

    correct = 0
    mismatches: list[dict[str, object]] = []
    for slot, expected_name, predicted_name in zip(CARD_SLOTS, expected_cards, predicted_cards):
        if expected_name == predicted_name:
            correct += 1
        else:
            mismatches.append(
                {
                    "slot": slot,
                    "expected": expected_name,
                    "predicted": predicted_name,
                }
            )

    return CardComparison(
        cards_correct=correct,
        cards_exact=not mismatches,
        card_mismatches=tuple(mismatches),
    )


def compare_snapshot(expected: ExpectedSample, predicted: VisionSnapshot) -> SnapshotComparison:
    board = compare_boards(expected.board, predicted.board)
    cards = compare_cards(expected, predicted)
    payload: dict[str, object] = {}
    if board.cell_mismatches:
        payload["cells"] = list(board.cell_mismatches)
    if cards.card_mismatches:
        payload["cards"] = list(cards.card_mismatches)
    return SnapshotComparison(
        board=board,
        cards=cards,
        snapshot_exact=board.board_exact and cards.cards_exact,
        mismatch_details=json.dumps(payload, separators=(",", ":"), ensure_ascii=False),
    )


def _read_image(path: Path) -> Any:
    import cv2

    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Image cannot be opened with OpenCV: {path}")
    return frame


def _build_pipeline(args: argparse.Namespace) -> VisionPipeline:
    piece_detector = YoloPieceDetector(
        model_path=args.piece_model,
        calibration_path=args.calibration,
        padding_ratio=0.25,
        imgsz=640,
        conf=0.50,
        iou=0.45,
        max_det=25,
        yolo_device=args.device,
        anchor_x_ratio=0.30,
    )
    card_classifier = YoloCardClassifier(
        model_path=args.card_model,
        rois_path=args.rois_path,
        imgsz=320,
        yolo_device=args.device,
        mask_polygon=True,
    )
    return VisionPipeline(piece_detector=piece_detector, card_classifier=card_classifier)


def evaluate_sample(sample: ExpectedSample, pipeline: VisionPipeline) -> ResultRow:
    frame = _read_image(sample.image_path)
    started = time.perf_counter()
    try:
        snapshot = pipeline.snapshot_from_frame(frame)
        pipeline_ms = (time.perf_counter() - started) * 1000.0
        comparison = compare_snapshot(sample, snapshot)
        return ResultRow(
            state_id=sample.state_id,
            lighting=sample.lighting,
            image=str(sample.image_path),
            status="ok",
            pieces_gt=comparison.board.pieces_gt,
            pieces_pred=comparison.board.pieces_pred,
            pieces_correct=comparison.board.pieces_correct,
            pieces_missing=comparison.board.pieces_missing,
            pieces_extra=comparison.board.pieces_extra,
            pieces_wrong_class=comparison.board.pieces_wrong_class,
            cards_correct=comparison.cards.cards_correct,
            board_exact=comparison.board.board_exact,
            cards_exact=comparison.cards.cards_exact,
            snapshot_exact=comparison.snapshot_exact,
            pipeline_ms=pipeline_ms,
            mismatch_details=comparison.mismatch_details,
            error_type="",
            error_message="",
        )
    except Exception as exc:
        pipeline_ms = (time.perf_counter() - started) * 1000.0
        return ResultRow(
            state_id=sample.state_id,
            lighting=sample.lighting,
            image=str(sample.image_path),
            status="error",
            pieces_gt=count_pieces(sample.board),
            pieces_pred=0,
            pieces_correct=0,
            pieces_missing=0,
            pieces_extra=0,
            pieces_wrong_class=0,
            cards_correct=0,
            board_exact=False,
            cards_exact=False,
            snapshot_exact=False,
            pipeline_ms=pipeline_ms,
            mismatch_details="{}",
            error_type=type(exc).__name__,
            error_message=str(exc),
        )


def count_pieces(board: VisionBoard) -> int:
    return sum(1 for row in board.board for cell in row if cell is not None)


def build_summary(rows: list[ResultRow]) -> dict[str, object]:
    return {
        "global": _summarize_rows(rows),
        "by_lighting": {
            lighting: _summarize_lighting([row for row in rows if row.lighting == lighting])
            for lighting in LIGHTING_LEVELS
        },
    }


def _summarize_rows(rows: list[ResultRow]) -> dict[str, int | float]:
    samples_total = len(rows)
    pieces_gt = sum(row.pieces_gt for row in rows)
    cards_total = 5 * samples_total
    times = [row.pipeline_ms for row in rows]

    return {
        "samples_total": samples_total,
        "samples_success": sum(1 for row in rows if row.status == "ok"),
        "samples_error": sum(1 for row in rows if row.status == "error"),
        "pieces_gt": pieces_gt,
        "pieces_correct": sum(row.pieces_correct for row in rows),
        "pieces_accuracy": sum(row.pieces_correct for row in rows) / pieces_gt if pieces_gt else 0.0,
        "pieces_missing": sum(row.pieces_missing for row in rows),
        "pieces_extra": sum(row.pieces_extra for row in rows),
        "pieces_wrong_class": sum(row.pieces_wrong_class for row in rows),
        "cards_total": cards_total,
        "cards_correct": sum(row.cards_correct for row in rows),
        "cards_accuracy": sum(row.cards_correct for row in rows) / cards_total if cards_total else 0.0,
        "boards_exact": sum(1 for row in rows if row.board_exact),
        "boards_exact_rate": _rate(sum(1 for row in rows if row.board_exact), samples_total),
        "cards_exact": sum(1 for row in rows if row.cards_exact),
        "cards_exact_rate": _rate(sum(1 for row in rows if row.cards_exact), samples_total),
        "snapshots_exact": sum(1 for row in rows if row.snapshot_exact),
        "snapshots_exact_rate": _rate(sum(1 for row in rows if row.snapshot_exact), samples_total),
        "pipeline_ms_mean": statistics.mean(times) if times else 0.0,
        "pipeline_ms_median": statistics.median(times) if times else 0.0,
        "pipeline_ms_min": min(times) if times else 0.0,
        "pipeline_ms_max": max(times) if times else 0.0,
    }


def _summarize_lighting(rows: list[ResultRow]) -> dict[str, int | float]:
    samples = len(rows)
    pieces_gt = sum(row.pieces_gt for row in rows)
    cards_total = 5 * samples
    times = [row.pipeline_ms for row in rows]
    return {
        "samples": samples,
        "pieces_accuracy": sum(row.pieces_correct for row in rows) / pieces_gt if pieces_gt else 0.0,
        "cards_accuracy": sum(row.cards_correct for row in rows) / cards_total if cards_total else 0.0,
        "boards_exact_rate": _rate(sum(1 for row in rows if row.board_exact), samples),
        "cards_exact_rate": _rate(sum(1 for row in rows if row.cards_exact), samples),
        "snapshots_exact_rate": _rate(sum(1 for row in rows if row.snapshot_exact), samples),
        "pipeline_ms_mean": statistics.mean(times) if times else 0.0,
        "pipeline_ms_median": statistics.median(times) if times else 0.0,
    }


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def write_results_csv(path: Path, rows: list[ResultRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            payload = asdict(row)
            payload["pipeline_ms"] = f"{row.pipeline_ms:.6f}"
            writer.writerow(payload)


def write_summary_json(path: Path, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def print_summary(summary: dict[str, object], results_path: Path, summary_path: Path) -> None:
    global_summary = _require_summary_object(summary["global"])
    by_lighting = _require_summary_object(summary["by_lighting"])

    print("Offline vision pipeline evaluation")
    print(f"Captures evaluated: {global_summary['samples_total']}")
    print(
        "Pieces correct: "
        f"{global_summary['pieces_correct']}/{global_summary['pieces_gt']} "
        f"({global_summary['pieces_accuracy']:.3f})"
    )
    print(
        "Cards correct: "
        f"{global_summary['cards_correct']}/{global_summary['cards_total']} "
        f"({global_summary['cards_accuracy']:.3f})"
    )
    print(
        "Exact boards: "
        f"{global_summary['boards_exact']}/{global_summary['samples_total']} "
        f"({global_summary['boards_exact_rate']:.3f})"
    )
    print(
        "Exact snapshots: "
        f"{global_summary['snapshots_exact']}/{global_summary['samples_total']} "
        f"({global_summary['snapshots_exact_rate']:.3f})"
    )
    print(
        "Pipeline time: "
        f"mean={global_summary['pipeline_ms_mean']:.2f} ms "
        f"median={global_summary['pipeline_ms_median']:.2f} ms"
    )
    print("Results by lighting:")
    for lighting in LIGHTING_LEVELS:
        item = _require_summary_object(by_lighting[lighting])
        print(
            f"  {lighting}: samples={item['samples']} "
            f"pieces={item['pieces_accuracy']:.3f} "
            f"cards={item['cards_accuracy']:.3f} "
            f"boards={item['boards_exact_rate']:.3f} "
            f"snapshots={item['snapshots_exact_rate']:.3f} "
            f"mean={item['pipeline_ms_mean']:.2f} ms "
            f"median={item['pipeline_ms_median']:.2f} ms"
        )
    print(f"results.csv: {results_path}")
    print(f"summary.json: {summary_path}")


def _require_summary_object(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("summary payload is malformed.")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the Onitama VisionPipeline offline on annotated full-frame images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--annotations", type=Path, default=Path("data/vision/pipeline_eval/annotations.json"))
    parser.add_argument("--piece-model", type=Path, default=Path("models/pieces_yolo11s_640_best.pt"))
    parser.add_argument("--card-model", type=Path, default=Path("models/cards_yolo11n-cls_320_best.pt"))
    parser.add_argument("--calibration", type=Path, default=Path("data/vision/calibration.json"))
    parser.add_argument("--rois-path", type=Path, default=Path("data/vision/card_rois.json"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("scripts/results/vision_pipeline_eval"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        samples = load_and_validate_annotations(args.annotations)
    except AnnotationValidationError as exc:
        print(f"Invalid annotations: {exc}", file=sys.stderr)
        return 2

    if not samples:
        print("Invalid annotations: no evaluation samples found.", file=sys.stderr)
        return 2

    pipeline = _build_pipeline(args)

    try:
        warmup_frame = _read_image(samples[0].image_path)
        pipeline.snapshot_from_frame(warmup_frame)
    except Exception as exc:
        print(f"Warmup failed and will not be included in results: {type(exc).__name__}: {exc}", file=sys.stderr)

    rows = [evaluate_sample(sample, pipeline) for sample in samples]
    summary = build_summary(rows)

    results_path = args.output_dir / "results.csv"
    summary_path = args.output_dir / "summary.json"
    write_results_csv(results_path, rows)
    write_summary_json(summary_path, summary)
    print_summary(summary, results_path, summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
