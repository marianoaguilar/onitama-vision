from __future__ import annotations
from pathlib import Path

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QMessageBox, QPushButton, QWidget

from onitama.gui.calibration.calibration_common import (
    BOARD_DEFAULT_ROTATION,
    BOARD_LINE_BGR,
    BOARD_POINT_BGR,
    CARD_VERTEX_PICK_RADIUS,
    CalibrationDialogBase,
    draw_polyline,
    draw_vertex,
    fill_polygon,
    nearest_point,
)
from onitama.vision.homography import HomographyCalibration


BOARD_DST_SIZE = (500, 500)


class BoardCalibrationDialog(CalibrationDialogBase):
    saved = Signal()
    frame_title = "Calibracion del tablero"
    header_text = "Captura y marca 4 esquinas."

    def __init__(self, calibration_path: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.frame_title)
        self.resize(1320, 850)

        self._calibration_path = calibration_path
        self._points: list[tuple[float, float]] = []
        self._dragging_index: int | None = None

        self._clear_button = QPushButton("Borrar puntos")
        self._save_button = QPushButton("Guardar")

        self._build_layout()
        self._connect_signals()
        self._load_existing()
        self._update_actions()
        self._camera.start()

    def _build_layout(self) -> None:
        layout = self._build_base_layout()
        layout.addLayout(self._build_actions_row(self._freeze_button, self._clear_button, self._save_button))

    def _connect_signals(self) -> None:
        self._connect_base_signals()
        self._view.left_pressed.connect(self._on_left_pressed)
        self._view.left_dragged.connect(self._on_left_dragged)
        self._view.left_released.connect(self._on_left_released)
        self._view.right_pressed.connect(self._on_right_pressed)
        self._clear_button.clicked.connect(self._clear_points)
        self._save_button.clicked.connect(self._save)

    def _load_existing(self) -> None:
        try:
            calib = HomographyCalibration.load(self._calibration_path)
        except Exception:  # noqa: BLE001
            return
        self._points = list(calib.src_points)

    def _clear_points(self) -> None:
        self._points = []
        self._dragging_index = None
        self._render()
        self._update_actions()

    def _on_left_pressed(self, x: float, y: float) -> None:
        if self._frozen_frame is None:
            return
        if len(self._points) < 4:
            self._points.append((x, y))
            self._dragging_index = len(self._points) - 1
        else:
            self._dragging_index = nearest_point(self._points, x, y, CARD_VERTEX_PICK_RADIUS)
        self._render()
        self._update_actions()

    def _on_left_dragged(self, x: float, y: float) -> None:
        if self._dragging_index is None:
            return
        self._points[self._dragging_index] = (x, y)
        self._render()

    def _on_left_released(self, _x: float, _y: float) -> None:
        self._dragging_index = None

    def _on_right_pressed(self, x: float, y: float) -> None:
        if self._frozen_frame is None:
            return
        idx = nearest_point(self._points, x, y, CARD_VERTEX_PICK_RADIUS)
        if idx is not None:
            self._points.pop(idx)
            self._render()
            self._update_actions()

    def _clear_drag_state(self) -> None:
        self._dragging_index = None

    def _save(self) -> None:
        if len(self._points) != 4:
            QMessageBox.warning(self, self.frame_title, "Faltan esquinas.")
            return
        calib = HomographyCalibration(
            src_points=tuple((float(x), float(y)) for x, y in self._points),
            dst_size=(BOARD_DST_SIZE[0], BOARD_DST_SIZE[1]),
            rotate=BOARD_DEFAULT_ROTATION,
        )
        calib.save(self._calibration_path)
        self.saved.emit()
        self._show_feedback("Calibracion guardada.")

    def _render(self) -> None:
        frame = self._display_frame()
        if frame is None:
            return
        canvas = frame.copy()
        if len(self._points) >= 2:
            pts = np.array([[int(round(x)), int(round(y))] for x, y in self._points], dtype=np.int32).reshape((-1, 1, 2))
            if len(self._points) == 4:
                fill_polygon(canvas, pts, BOARD_LINE_BGR, alpha=0.14)
            draw_polyline(canvas, pts, BOARD_LINE_BGR, closed=len(self._points) == 4, thickness=2)
        for idx, (x, y) in enumerate(self._points):
            draw_vertex(
                canvas,
                (int(round(x)), int(round(y))),
                BOARD_POINT_BGR,
                selected=idx == self._dragging_index,
            )
        self._view.set_bgr_frame(canvas)
    def _update_actions(self) -> None:
        self._save_button.setEnabled(len(self._points) == 4)
