import math

import numpy as np

from onitama.vision.card_classifier import (
    CardClassificationResult,
    CardSlotPrediction,
    YoloCardClassifier,
)


def test_prepare_crop_preserves_aspect_ratio() -> None:
    image = np.full((100, 200, 3), 50, dtype=np.uint8)

    classifier = object.__new__(YoloCardClassifier)
    classifier.imgsz = 320

    prepared = classifier.prepare_crop(image)

    assert prepared.shape == (320, 320, 3)
    scale = 320.0 / 200.0
    assert math.isclose(scale, 1.6)
    assert np.all(prepared[:80, :, :] == 0)
    assert np.all(prepared[80:240, :, :] == 50)
    assert np.all(prepared[240:, :, :] == 0)


def test_card_classification_result_roundtrip_and_layout(tmp_path) -> None:
    predictions = (
        CardSlotPrediction(
            slot="red_0",
            class_name="Tiger",
            confidence=0.99,
        ),
        CardSlotPrediction(
            slot="red_1",
            class_name="Horse",
            confidence=0.98,
        ),
        CardSlotPrediction(
            slot="side",
            class_name="Rabbit",
            confidence=0.97,
        ),
        CardSlotPrediction(
            slot="blue_0",
            class_name="Crab",
            confidence=0.96,
        ),
        CardSlotPrediction(
            slot="blue_1",
            class_name="Boar",
            confidence=0.95,
        ),
    )
    result = CardClassificationResult(
        predictions=predictions,
    )

    out_path = tmp_path / "cards.json"
    result.save_json(out_path)
    loaded = CardClassificationResult.load_json(out_path)

    assert loaded.cards_layout() == (("Tiger", "Horse"), ("Crab", "Boar"), "Rabbit")
    assert loaded.predictions[0].class_name == "Tiger"


def test_card_classification_result_loads_legacy_debug_fields() -> None:
    legacy = {
        "predictions": [
            {
                "slot": "red_0",
                "class_index": 15,
                "class_name": "Tiger",
                "confidence": 0.99,
                "top_candidates": [{"class_index": 15, "class_name": "Tiger", "confidence": 0.99}],
                "crop_shape": [250, 362],
                "prepared_shape": [320, 320],
            },
            {"slot": "red_1", "class_index": 9, "class_name": "Horse", "confidence": 0.98},
            {"slot": "side", "class_index": 13, "class_name": "Rabbit", "confidence": 0.97},
            {"slot": "blue_0", "class_index": 2, "class_name": "Crab", "confidence": 0.96},
            {"slot": "blue_1", "class_index": 0, "class_name": "Boar", "confidence": 0.95},
        ],
        "model_path": "models/cards_yolo11n-cls_320_best.pt",
        "imgsz": 320,
        "mask_polygon": True,
    }

    loaded = CardClassificationResult.from_dict(legacy)

    assert loaded.cards_layout() == (("Tiger", "Horse"), ("Crab", "Boar"), "Rabbit")
    assert loaded.predictions[0].confidence == 0.99
