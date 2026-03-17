import json

import numpy as np
import pytest

from onitama.vision.card_rois import SLOT_ORDER, extract_polygon_crop, load_card_rois


def _roi_payload(points) -> dict[str, object]:
    return {
        slot: {
            "src_points": points,
        }
        for slot in SLOT_ORDER
    }


def test_load_card_rois_orders_points_clockwise(tmp_path) -> None:
    path = tmp_path / "card_rois.json"
    raw_points = [[4.0, 4.0], [1.0, 1.0], [4.0, 1.0], [1.0, 4.0]]
    path.write_text(json.dumps(_roi_payload(raw_points)), encoding="utf-8")

    rois = load_card_rois(path)

    assert rois["red_0"] == [(1.0, 1.0), (4.0, 1.0), (4.0, 4.0), (1.0, 4.0)]


def test_load_card_rois_allow_missing_returns_empty_slots(tmp_path) -> None:
    rois = load_card_rois(tmp_path / "missing.json", allow_missing=True)
    assert all(rois[slot] == [] for slot in SLOT_ORDER)


def test_extract_polygon_crop_clips_to_visible_frame() -> None:
    frame = np.full((5, 5, 3), 255, dtype=np.uint8)
    points = [(-2.0, -1.0), (3.0, -1.0), (3.0, 2.0), (-2.0, 2.0)]

    crop = extract_polygon_crop(frame, points, mask_polygon=False)

    assert crop.shape == (3, 4, 3)
    assert np.all(crop == 255)


def test_extract_polygon_crop_applies_polygon_mask() -> None:
    frame = np.full((6, 6, 3), 255, dtype=np.uint8)
    points = [(1.0, 0.0), (4.0, 1.0), (3.0, 4.0), (0.0, 3.0)]

    crop = extract_polygon_crop(frame, points, mask_polygon=True)

    assert crop.shape == (5, 5, 3)
    assert np.all(crop[0, 4] == 0)
    assert np.all(crop[2, 2] == 255)


def test_extract_polygon_crop_rejects_invalid_vertices() -> None:
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="exactly 4"):
        extract_polygon_crop(frame, [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)], mask_polygon=True)
