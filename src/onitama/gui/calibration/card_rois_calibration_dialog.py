from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QButtonGroup, QMessageBox, QPushButton, QHBoxLayout, QWidget

from onitama.gui.calibration.calibration_common import (
    CARD_ACTIVE_COLOR_BGR,
    CARD_CANVAS_PADDING,
    CARD_INACTIVE_BGR,
    CARD_MARGIN_BGR,
    CARD_VERTEX_PICK_RADIUS,
    CalibrationDialogBase,
    SLOT_BUTTON_ORDER,
    SLOT_DISPLAY_LABEL,
    draw_badge,
    draw_polyline,
    draw_vertex,
    fill_polygon,
    nearest_point,
)
from onitama.vision.card_rois import SLOT_ORDER, SlotName, load_card_rois, save_card_rois


class CardRoisCalibrationDialog(CalibrationDialogBase):
    saved = Signal()
    frame_title = "Calibracion de cartas"
    header_text = "Selecciona una carta y marca 4 puntos."

    def __init__(self, rois_path: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.frame_title)
        self.resize(1320, 850)

        self._rois_path = rois_path
        try:
            self._rois = load_card_rois(rois_path, allow_missing=True)
        except Exception:  # noqa: BLE001
            self._rois = {slot: [] for slot in SLOT_ORDER}
        self._active_slot_index = 0
        self._dragging_index: int | None = None

        self._clear_slot_button = QPushButton("Borrar carta")
        self._clear_all_button = QPushButton("Borrar todo")
        self._save_button = QPushButton("Guardar")
        self._slot_group = QButtonGroup(self)

        self._build_layout()
        self._connect_signals()
        self._update_actions()
        self._camera.start()

    def _build_layout(self) -> None:
        layout = self._build_base_layout()

        slot_row = QHBoxLayout()
        slot_row.setSpacing(8)
        for slot in SLOT_BUTTON_ORDER:
            idx = SLOT_ORDER.index(slot)
            button = QPushButton(SLOT_DISPLAY_LABEL[slot])
            if slot.startswith("red"):
                button.setObjectName("redCardSlot")
            elif slot.startswith("blue"):
                button.setObjectName("blueCardSlot")
            else:
                button.setObjectName("sideCardSlot")
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, i=idx: self._select_slot(i))
            self._slot_group.addButton(button, idx)
            slot_row.addWidget(button)
        active_button = self._slot_group.button(self._active_slot_index)
        if active_button is not None:
            active_button.setChecked(True)
        slot_row.setContentsMargins(0, 8, 0, 2)
        layout.addLayout(slot_row)
        layout.addLayout(
            self._build_actions_row(
                self._freeze_button,
                self._clear_slot_button,
                self._clear_all_button,
                self._save_button,
            )
        )

    def _connect_signals(self) -> None:
        self._connect_base_signals()
        self._view.left_pressed.connect(self._on_left_pressed)
        self._view.left_dragged.connect(self._on_left_dragged)
        self._view.left_released.connect(self._on_left_released)
        self._view.right_pressed.connect(self._on_right_pressed)
        self._clear_slot_button.clicked.connect(self._clear_active_slot)
        self._clear_all_button.clicked.connect(self._clear_all)
        self._save_button.clicked.connect(self._save)

    def _select_slot(self, index: int) -> None:
        self._active_slot_index = index
        self._dragging_index = None
        self._render()
        self._update_actions()

    def _clear_active_slot(self) -> None:
        self._rois[self._active_slot()] = []
        self._dragging_index = None
        self._render()
        self._update_actions()

    def _clear_all(self) -> None:
        for slot in SLOT_ORDER:
            self._rois[slot] = []
        self._dragging_index = None
        self._render()
        self._update_actions()

    def _on_left_pressed(self, x: float, y: float) -> None:
        x, y = self._to_world(x, y)
        points = self._rois[self._active_slot()]
        if len(points) < 4:
            points.append((x, y))
            self._dragging_index = len(points) - 1
        else:
            self._dragging_index = nearest_point(points, x, y, CARD_VERTEX_PICK_RADIUS)
        self._render()
        self._update_actions()

    def _on_left_dragged(self, x: float, y: float) -> None:
        if self._dragging_index is None:
            return
        x, y = self._to_world(x, y)
        self._rois[self._active_slot()][self._dragging_index] = (x, y)
        self._render()

    def _on_left_released(self, _x: float, _y: float) -> None:
        self._dragging_index = None

    def _on_right_pressed(self, x: float, y: float) -> None:
        x, y = self._to_world(x, y)
        points = self._rois[self._active_slot()]
        idx = nearest_point(points, x, y, CARD_VERTEX_PICK_RADIUS)
        if idx is not None:
            points.pop(idx)
            self._dragging_index = None
            self._render()
            self._update_actions()

    def _clear_drag_state(self) -> None:
        self._dragging_index = None

    def _save(self) -> None:
        try:
            save_card_rois(self._rois_path, self._rois)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, self.frame_title, f"No se pudo guardar: {exc}")
            return
        self.saved.emit()
        self._show_feedback("Calibracion guardada.")

    def _render(self) -> None:
        frame = self._display_frame()
        if frame is None:
            return
        frame_h, frame_w = frame.shape[:2]
        pad = CARD_CANVAS_PADDING
        canvas = np.zeros((frame_h + 2 * pad, frame_w + 2 * pad, 3), dtype=np.uint8)
        canvas[:] = CARD_MARGIN_BGR
        canvas[pad:pad + frame_h, pad:pad + frame_w] = frame

        active_slot = self._active_slot()
        render_order = [slot for slot in SLOT_ORDER if slot != active_slot] + [active_slot]
        for slot in render_order:
            points = self._rois.get(slot, [])
            active = slot == active_slot
            color = CARD_ACTIVE_COLOR_BGR[slot] if active else CARD_INACTIVE_BGR
            thickness = 2 if active else 1
            if len(points) >= 2:
                pts = np.array(
                    [[int(round(x + pad)), int(round(y + pad))] for x, y in points],
                    dtype=np.int32,
                ).reshape((-1, 1, 2))
                if len(points) == 4:
                    fill_polygon(canvas, pts, color, alpha=0.12 if active else 0.06)
                draw_polyline(canvas, pts, color, closed=len(points) == 4, thickness=thickness)
            for vertex_idx, (x, y) in enumerate(points):
                draw_vertex(
                    canvas,
                    (int(round(x + pad)), int(round(y + pad))),
                    color,
                    selected=active and vertex_idx == self._dragging_index,
                )
            if active and points:
                cx = int(round(sum(x for x, _ in points) / len(points) + pad))
                cy = int(round(sum(y for _, y in points) / len(points) + pad))
                draw_badge(canvas, SLOT_DISPLAY_LABEL[slot], (cx, cy - 18), color)

        self._view.set_bgr_frame(canvas)

    def _update_actions(self) -> None:
        self._save_button.setEnabled(all(len(self._rois.get(slot, [])) == 4 for slot in SLOT_ORDER))

    def _active_slot(self) -> SlotName:
        return SLOT_ORDER[self._active_slot_index]
    def _to_world(self, x: float, y: float) -> tuple[float, float]:
        return x - CARD_CANVAS_PADDING, y - CARD_CANVAS_PADDING
