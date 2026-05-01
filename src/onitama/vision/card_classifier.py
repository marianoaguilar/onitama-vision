from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from onitama.app.errors import (
    VisionConfigurationError,
    VisionDependencyError,
    VisionObservationError,
    VisionObservationKind,
    VisionPipelineError,
)
from onitama.vision.card_rois import SLOT_ORDER, SlotName, extract_card_crops, load_card_rois


@dataclass(frozen=True)
class CardSlotPrediction:
    """Top-1 card prediction for one slot."""

    slot: SlotName
    class_name: str
    confidence: float

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "CardSlotPrediction":
        return cls(
            slot=str(data["slot"]),
            class_name=str(data["class_name"]),
            confidence=float(data["confidence"]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "slot": self.slot,
            "class_name": self.class_name,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class CardClassificationResult:
    """Card predictions for the 5 fixed slots."""

    predictions: tuple[CardSlotPrediction, ...]
    
    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "CardClassificationResult":
        predictions = data.get("predictions", [])
        if not isinstance(predictions, list):
            raise ValueError("Invalid card predictions JSON: 'predictions' must be a list.")
        return cls(
            predictions=tuple(CardSlotPrediction.from_dict(item) for item in predictions),
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "CardClassificationResult":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Invalid card predictions JSON: root must be an object.")
        return cls.from_dict(data)

    def by_slot(self) -> dict[SlotName, CardSlotPrediction]:
        """Return predictions indexed by slot name."""
        return {prediction.slot: prediction for prediction in self.predictions}

    # Conexion with VisionSnapshot
    def cards_layout(self) -> tuple[tuple[str, str], tuple[str, str], str]:
        """Return cards grouped as red pair, blue pair and side card."""
        by_slot = self.by_slot()
        missing = [slot for slot in SLOT_ORDER if slot not in by_slot]
        if missing:
            raise VisionPipelineError(f"Missing predictions for slots: {missing}")

        red_cards = (by_slot["red_0"].class_name, by_slot["red_1"].class_name)
        blue_cards = (by_slot["blue_0"].class_name, by_slot["blue_1"].class_name)
        side_card = by_slot["side"].class_name
        return red_cards, blue_cards, side_card

    def to_dict(self) -> dict[str, object]:
        return {
            "predictions": [prediction.to_dict() for prediction in self.predictions],
        }

    def save_json(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")



class YoloCardClassifier:
    """Run card classification on the 5 configured ROIs."""

    def __init__(
        self,
        *,
        model_path: str | Path = "models/cards_yolo11n-cls_320_best.pt",
        rois_path: str | Path = "data/vision/card_rois.json",
        imgsz: int = 320,
        yolo_device: str = "cpu",
        mask_polygon: bool = True,
        min_card_confidence: float = 0.50,
    ) -> None:
        model_path = Path(model_path)
        rois_path = Path(rois_path)
        if not model_path.exists():
            raise VisionConfigurationError(f"YOLO card model not found: {model_path}")
        if not rois_path.exists():
            raise VisionConfigurationError(f"Card ROI file not found: {rois_path}")
        if imgsz <= 0:
            raise VisionConfigurationError("imgsz must be > 0.")
        if not (0.0 <= min_card_confidence <= 1.0):
            raise VisionConfigurationError("min_card_confidence must be in [0.0, 1.0]")

        self.model_path = model_path
        self.rois_path = rois_path
        self.imgsz = int(imgsz)
        self.yolo_device = yolo_device
        self.mask_polygon = bool(mask_polygon)
        self.min_card_confidence = float(min_card_confidence)
        self.rois = load_card_rois(rois_path)

        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise VisionDependencyError(
                "Could not import ultralytics. Install it in your .venv with: "
                ".venv/bin/python -m pip install ultralytics"
            ) from exc

        self._model = YOLO(str(model_path))
        raw_names = self._model.names
        if isinstance(raw_names, dict):
            self.class_names = {int(idx): str(name) for idx, name in raw_names.items()}
        else:
            self.class_names = {int(idx): str(name) for idx, name in enumerate(raw_names)}
    
    def extract_card_crops(self, frame: np.ndarray) -> dict[SlotName, np.ndarray]:
        """Extract the 5 card crops from a frame."""
        return extract_card_crops(frame, self.rois, mask_polygon=self.mask_polygon)

    def prepare_crop(self, crop: np.ndarray) -> np.ndarray:
        """Resize one crop to the model input size with black padding."""
        if crop.size == 0:
            raise VisionPipelineError("crop must be non-empty.")
        if crop.ndim != 3:
            raise VisionPipelineError("crop must be a color image with shape (H, W, C).")

        target_size = self.imgsz
        target_w = target_h = int(target_size)
        src_h, src_w = crop.shape[:2]
        scale = min(target_w / float(src_w), target_h / float(src_h))
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(crop, (new_w, new_h), interpolation=interpolation)

        canvas = np.zeros((target_h, target_w, crop.shape[2]), dtype=crop.dtype)
        offset_x = (target_w - new_w) // 2
        offset_y = (target_h - new_h) // 2
        canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized
        return canvas

    def prepare_crops(self, crops: dict[SlotName, np.ndarray]) -> dict[SlotName, np.ndarray]:
        """Prepare the 5 crops in a stable slot order."""
        return {slot: self.prepare_crop(crops[slot]) for slot in SLOT_ORDER}

    def predict_prepared_crops(self, prepared_crops: dict[SlotName, np.ndarray]) -> list[Any]:
        """Run YOLO on already prepared crops."""
        inputs = [prepared_crops[slot] for slot in SLOT_ORDER]
        return self._model.predict(
            source=inputs,
            imgsz=self.imgsz,
            device=self.yolo_device,
            verbose=False,
        )

    def _classify_prepared_crops(self, prepared_crops: dict[SlotName, np.ndarray]) -> CardClassificationResult:
        
        # Run the model and check we got a result for each slot.
        results = self.predict_prepared_crops(prepared_crops)
        if len(results) != len(SLOT_ORDER):
            raise VisionPipelineError(
                f"Expected {len(SLOT_ORDER)} classification results, got {len(results)}."
            )

        # Extract the top-1 prediction for each slot and build the final result.
        predictions: list[CardSlotPrediction] = []
        for slot, result in zip(SLOT_ORDER, results):
            probs = result.probs
            if probs is None:
                raise VisionPipelineError(
                    f"Classification result for slot '{slot}' does not contain probabilities."
                )

            top1_conf = float(probs.top1conf.item() if hasattr(probs.top1conf, "item") else probs.top1conf)
            if top1_conf < self.min_card_confidence:
                raise VisionObservationError(
                    VisionObservationKind.LOW_CONFIDENCE_CARD,
                    debug_message=(
                        f"Low-confidence card prediction for slot '{slot}': "
                        f"{top1_conf:.2f} < {self.min_card_confidence:.2f}."
                    ),
                )
            top1_index = int(probs.top1)

            predictions.append(
                CardSlotPrediction(
                    slot=slot,
                    class_name=self.class_names.get(top1_index, str(top1_index)),
                    confidence=top1_conf,
                )
            )

        return CardClassificationResult(predictions=tuple(predictions))

    def classify_crops(self, crops: dict[SlotName, np.ndarray]) -> CardClassificationResult:
        """Prepare and classify a set of already extracted crops."""
        prepared_crops = self.prepare_crops(crops)
        return self._classify_prepared_crops(prepared_crops)

    def classify_from_frame(self, frame: np.ndarray) -> CardClassificationResult:
        """Extract, prepare and classify cards from a frame."""
        return self.classify_crops(self.extract_card_crops(frame))
