import cv2
import numpy as np

from onitama.vision.homography import (
    HomographyCalibration,
    build_padded_homography,
    rotate_point,
    rotate_roi,
    xy_to_cell,
)


def test_xy_to_cell_maps_points_and_clamps_edges() -> None:
    assert xy_to_cell(0, 0, board_size=5, dst_size=(500, 500)) == (0, 0)
    assert xy_to_cell(250, 350, board_size=5, dst_size=(500, 500)) == (3, 2)
    assert xy_to_cell(500, 500, board_size=5, dst_size=(500, 500)) == (4, 4)


def test_rotate_point_and_roi_for_quarter_turn() -> None:
    assert rotate_point(2, 1, width=10, height=6, rotate=90) == (4, 2)
    assert rotate_roi(2, 1, 3, 2, width=10, height=6, rotate=90) == (3, 2, 2, 3)


def test_build_padded_homography_returns_expected_output_and_roi() -> None:
    calib = HomographyCalibration(
        src_points=((0.0, 0.0), (99.0, 0.0), (99.0, 99.0), (0.0, 99.0)),
        dst_size=(100, 100),
    )

    matrix, output_size, board_roi = build_padded_homography(calib, padding_ratio=0.25)
    corners = np.array(
        [[[0.0, 0.0]], [[99.0, 0.0]], [[99.0, 99.0]], [[0.0, 99.0]]],
        dtype=np.float32,
    )
    transformed = cv2.perspectiveTransform(corners, matrix).reshape(4, 2)

    assert np.allclose(
        transformed,
        [(25.0, 25.0), (124.0, 25.0), (124.0, 124.0), (25.0, 124.0)],
    )
    assert output_size == (150, 150)
    assert board_roi == (25, 25, 100, 100)
