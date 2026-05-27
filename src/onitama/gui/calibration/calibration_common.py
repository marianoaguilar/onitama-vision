from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPixmap, QWheelEvent
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from onitama.gui import theme
from onitama.vision.card_rois import Quad, SlotName


CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
BOARD_LINE_BGR = theme.BOARD_LINE_BGR
BOARD_POINT_BGR = theme.BOARD_POINT_BGR
BOARD_DEFAULT_ROTATION = 90
CARD_VERTEX_PICK_RADIUS = 24.0
CARD_CANVAS_PADDING = 80
CARD_MARGIN_BGR = theme.CARD_MARGIN_BGR
OVERLAY_WHITE_BGR = theme.OVERLAY_WHITE_BGR
OVERLAY_DARK_BGR = theme.OVERLAY_DARK_BGR
CARD_ACTIVE_COLOR_BGR = theme.CARD_ACTIVE_COLOR_BGR
CARD_INACTIVE_BGR = theme.CARD_INACTIVE_BGR
SLOT_DISPLAY_LABEL: dict[SlotName, str] = {
    "red_0": "Roja Arriba",
    "red_1": "Roja Abajo",
    "side": "Lateral",
    "blue_0": "Azul Arriba",
    "blue_1": "Azul Abajo",
}
SLOT_BUTTON_ORDER: tuple[SlotName, ...] = ("blue_0", "blue_1", "side", "red_0", "red_1")


class CameraCaptureThread(QThread):
    frame_changed = Signal(object)
    failed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._stop_requested = False

    def run(self) -> None:
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.failed.emit("No se pudo abrir la cámara 0.")
            return

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

        try:
            while not self._stop_requested:
                ok, frame = cap.read()
                if not ok:
                    self.failed.emit("No se pudo leer una imagen de la cámara.")
                    break
                self.frame_changed.emit(frame.copy())
                self.msleep(33)
        finally:
            cap.release()

    def request_stop(self) -> None:
        self._stop_requested = True


class CameraImageView(QLabel):
    left_pressed = Signal(float, float)
    left_dragged = Signal(float, float)
    left_released = Signal(float, float)
    right_pressed = Signal(float, float)

    def __init__(self, placeholder: str, parent: QWidget | None = None) -> None:
        super().__init__(placeholder, parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(900, 520)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(f"background: {theme.TRANSPARENT}; color: {theme.TEXT};")
        self._image = QImage()
        self._pixmap = QPixmap()
        self._zoom = 1.0
        self._pan = QPoint(0, 0)
        self._panning = False
        self._pan_start = QPoint(0, 0)
        self._pan_origin = QPoint(0, 0)

    def set_bgr_frame(self, frame: np.ndarray) -> None:
        self._image = qimage_from_bgr_frame(frame)
        self._rescale()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._pixmap.isNull():
            super().paintEvent(event)
            return
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(theme.APP_BG))
        painter.drawPixmap(self._pixmap_rect().topLeft(), self._pixmap)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._rescale()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.MouseButton.LeftButton and event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self._panning = True
            self._pan_start = event.position().toPoint()
            self._pan_origin = QPoint(self._pan)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        point = self._image_point(event.position().toPoint())
        if point is None:
            return
        x, y = point
        if event.button() == Qt.MouseButton.LeftButton:
            self.left_pressed.emit(x, y)
        elif event.button() == Qt.MouseButton.RightButton:
            self.right_pressed.emit(x, y)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        if self._panning:
            self._pan = self._pan_origin + event.position().toPoint() - self._pan_start
            self._clamp_pan()
            self.update()
            return

        if not event.buttons() & Qt.MouseButton.LeftButton:
            return
        point = self._image_point(event.position().toPoint())
        if point is None:
            return
        self.left_dragged.emit(point[0], point[1])

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        if self._panning and event.button() == Qt.MouseButton.LeftButton:
            self._panning = False
            self.unsetCursor()
            return

        if event.button() != Qt.MouseButton.LeftButton:
            return
        point = self._image_point(event.position().toPoint())
        if point is None:
            return
        self.left_released.emit(point[0], point[1])

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt API
        if self._image.isNull():
            return

        cursor = event.position().toPoint()
        anchor = self._image_point(cursor)
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self._set_zoom(self._zoom * factor)
        if anchor is None:
            return

        rect = self._pixmap_rect()
        target_x = anchor[0] * rect.width() / self._image.width()
        target_y = anchor[1] * rect.height() / self._image.height()
        self._pan += cursor - QPoint(int(round(rect.x() + target_x)), int(round(rect.y() + target_y)))
        self._clamp_pan()
        self.update()

    def _rescale(self) -> None:
        if self._image.isNull():
            return
        base = QPixmap.fromImage(self._image).scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        if self._zoom == 1.0:
            self._pixmap = base
        else:
            scaled_size = base.size()
            scaled_size.setWidth(max(1, int(round(scaled_size.width() * self._zoom))))
            scaled_size.setHeight(max(1, int(round(scaled_size.height() * self._zoom))))
            self._pixmap = base.scaled(
                scaled_size,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        self._clamp_pan()
        self.update()

    def _image_point(self, pos: QPoint) -> tuple[float, float] | None:
        if self._image.isNull() or self._pixmap.isNull():
            return None

        rect = self._pixmap_rect()
        if not rect.contains(pos):
            return None

        rel_x = pos.x() - rect.x()
        rel_y = pos.y() - rect.y()
        x = rel_x * self._image.width() / rect.width()
        y = rel_y * self._image.height() / rect.height()
        return float(x), float(y)

    def _pixmap_rect(self) -> QRect:
        x = (self.width() - self._pixmap.width()) // 2
        y = (self.height() - self._pixmap.height()) // 2
        return QRect(x + self._pan.x(), y + self._pan.y(), self._pixmap.width(), self._pixmap.height())

    def _set_zoom(self, value: float) -> None:
        self._zoom = max(1.0, min(6.0, value))
        if self._zoom == 1.0:
            self._pan = QPoint(0, 0)
        self._rescale()

    def _clamp_pan(self) -> None:
        if self._pixmap.isNull() or self._zoom == 1.0:
            self._pan = QPoint(0, 0)
            return

        base_x = (self.width() - self._pixmap.width()) // 2
        base_y = (self.height() - self._pixmap.height()) // 2
        self._pan.setX(clamp_axis(self._pan.x(), base_x, self._pixmap.width(), self.width()))
        self._pan.setY(clamp_axis(self._pan.y(), base_y, self._pixmap.height(), self.height()))


class CalibrationDialogBase(QDialog):
    frame_title = "Calibracion"
    header_text = ""
    freeze_live_text = "En vivo"
    freeze_capture_text = "Capturar"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._current_frame: np.ndarray | None = None
        self._frozen_frame: np.ndarray | None = None
        self._feedback_timer = QTimer(self)
        self._feedback_timer.setSingleShot(True)

        self._camera = CameraCaptureThread(self)
        self._view = CameraImageView("Abriendo cámara...")
        self._freeze_button = QPushButton(self.freeze_capture_text)
        self._close_button = QPushButton("Cerrar")
        self._feedback_label = QLabel()
        self._feedback_label.setObjectName("feedbackLabel")
        self._feedback_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._feedback_label.hide()

        self._feedback_timer.timeout.connect(self._feedback_label.hide)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        stop_camera_thread(self._camera)
        super().closeEvent(event)

    def _on_frame(self, frame: np.ndarray) -> None:
        self._current_frame = frame
        if self._frozen_frame is None:
            self._render()

    def _on_camera_error(self, message: str) -> None:
        QMessageBox.warning(self, self.frame_title, message)

    def _toggle_freeze(self) -> None:
        if self._frozen_frame is None:
            if self._current_frame is None:
                return
            self._frozen_frame = self._current_frame.copy()
            self._freeze_button.setText(self.freeze_live_text)
        else:
            self._frozen_frame = None
            self._freeze_button.setText(self.freeze_capture_text)
            self._clear_drag_state()
        self._render()
        self._update_actions()

    def _display_frame(self) -> np.ndarray | None:
        return self._frozen_frame if self._frozen_frame is not None else self._current_frame

    def _build_base_layout(self) -> QVBoxLayout:
        self.setStyleSheet(dialog_stylesheet())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        header = QLabel(self.header_text)
        header.setObjectName("helpText")
        header.setWordWrap(True)
        layout.addWidget(header)
        layout.addWidget(self._view, stretch=1)
        return layout

    def _build_actions_row(self, *buttons: QPushButton) -> QHBoxLayout:
        actions = QHBoxLayout()
        actions.setContentsMargins(0, 8, 0, 4)
        actions.setSpacing(14)
        for button in buttons:
            actions.addWidget(button)
        actions.addWidget(self._feedback_label)
        actions.addStretch(1)
        actions.addWidget(self._close_button)
        return actions

    def _connect_base_signals(self) -> None:
        self._camera.frame_changed.connect(self._on_frame)
        self._camera.failed.connect(self._on_camera_error)
        self._freeze_button.clicked.connect(self._toggle_freeze)
        self._close_button.clicked.connect(self.close)

    def _show_feedback(self, text: str, *, tone: str = "success", timeout_ms: int = 2200) -> None:
        styles = {
            "success": (
                f"color: {theme.SUCCESS_TEXT}; background: {theme.SUCCESS_BG}; "
                f"border: 1px solid {theme.SUCCESS_BORDER};"
            ),
            "warning": (
                f"color: {theme.WARNING_TEXT}; background: {theme.WARNING_BG}; "
                f"border: 1px solid {theme.WARNING_BORDER};"
            ),
            "error": (
                f"color: {theme.RED_DARK}; background: {theme.ERROR_BG}; "
                f"border: 1px solid {theme.ERROR_BORDER};"
            ),
        }
        self._feedback_timer.stop()
        self._feedback_label.setText(text)
        self._feedback_label.setStyleSheet(
            "font-size: 14px; font-weight: 700; border-radius: 8px; padding: 8px 12px; "
            + styles.get(tone, styles["success"])
        )
        self._feedback_label.show()
        self._feedback_timer.start(timeout_ms)

    def _clear_drag_state(self) -> None:
        pass

    def _render(self) -> None:
        raise NotImplementedError

    def _update_actions(self) -> None:
        raise NotImplementedError


def nearest_point(points: Quad, x: float, y: float, max_dist_px: float) -> int | None:
    best_idx: int | None = None
    best_d2 = max_dist_px * max_dist_px
    for idx, (px, py) in enumerate(points):
        d2 = (px - x) * (px - x) + (py - y) * (py - y)
        if d2 <= best_d2:
            best_d2 = d2
            best_idx = idx
    return best_idx


def stop_camera_thread(camera: CameraCaptureThread) -> None:
    camera.request_stop()
    camera.wait(1200)
    if camera.isRunning():
        camera.terminate()
        camera.wait(500)


def clamp_axis(pan: int, base_pos: int, pixmap_size: int, widget_size: int) -> int:
    if pixmap_size <= widget_size:
        return 0
    min_pan = widget_size - pixmap_size - base_pos
    max_pan = -base_pos
    return max(min_pan, min(max_pan, pan))


def fill_polygon(
    canvas: np.ndarray,
    pts: np.ndarray,
    color: tuple[int, int, int],
    *,
    alpha: float,
) -> None:
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [pts], color, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, dst=canvas)


def draw_polyline(
    canvas: np.ndarray,
    pts: np.ndarray,
    color: tuple[int, int, int],
    *,
    closed: bool,
    thickness: int,
) -> None:
    cv2.polylines(
        canvas,
        [pts],
        isClosed=closed,
        color=OVERLAY_WHITE_BGR,
        thickness=thickness + 1,
        lineType=cv2.LINE_AA,
    )
    cv2.polylines(canvas, [pts], isClosed=closed, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def draw_vertex(
    canvas: np.ndarray,
    center: tuple[int, int],
    color: tuple[int, int, int],
    *,
    selected: bool,
) -> None:
    radius = 6 if selected else 4
    cv2.circle(canvas, center, radius + 2, OVERLAY_WHITE_BGR, -1, lineType=cv2.LINE_AA)
    cv2.circle(canvas, center, radius, color, -1, lineType=cv2.LINE_AA)
    cv2.circle(canvas, center, radius + 1, OVERLAY_DARK_BGR, 1, lineType=cv2.LINE_AA)


def draw_badge(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
    center_x, baseline_y = origin
    pad_x = 9
    pad_y = 6
    box_w = text_size[0] + 2 * pad_x
    box_h = text_size[1] + baseline + 2 * pad_y
    x = int(round(center_x - box_w / 2))
    y = int(round(baseline_y - text_size[1] - pad_y))
    box_start = (x, y)
    box_end = (x + box_w, y + box_h)
    overlay = canvas.copy()
    rounded_box(overlay, box_start, box_end, OVERLAY_DARK_BGR, radius=10, thickness=-1)
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, dst=canvas)
    rounded_box(canvas, box_start, box_end, OVERLAY_WHITE_BGR, radius=10, thickness=1)
    rounded_box(canvas, (x + 1, y + 1), (x + box_w - 1, y + box_h - 1), color, radius=9, thickness=1)
    text_org = (x + pad_x, y + pad_y + text_size[1])
    cv2.putText(canvas, text, text_org, font, scale, OVERLAY_WHITE_BGR, thickness, cv2.LINE_AA)


def rounded_box(
    canvas: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    *,
    radius: int,
    thickness: int,
) -> None:
    x1, y1 = start
    x2, y2 = end
    radius = max(1, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    if thickness < 0:
        cv2.rectangle(canvas, (x1 + radius, y1), (x2 - radius, y2), color, -1, lineType=cv2.LINE_AA)
        cv2.rectangle(canvas, (x1, y1 + radius), (x2, y2 - radius), color, -1, lineType=cv2.LINE_AA)
        corners = (
            (x1 + radius, y1 + radius),
            (x2 - radius, y1 + radius),
            (x1 + radius, y2 - radius),
            (x2 - radius, y2 - radius),
        )
        for cx, cy in corners:
            cv2.circle(canvas, (cx, cy), radius, color, -1, lineType=cv2.LINE_AA)
        return

    cv2.line(canvas, (x1 + radius, y1), (x2 - radius, y1), color, thickness, lineType=cv2.LINE_AA)
    cv2.line(canvas, (x1 + radius, y2), (x2 - radius, y2), color, thickness, lineType=cv2.LINE_AA)
    cv2.line(canvas, (x1, y1 + radius), (x1, y2 - radius), color, thickness, lineType=cv2.LINE_AA)
    cv2.line(canvas, (x2, y1 + radius), (x2, y2 - radius), color, thickness, lineType=cv2.LINE_AA)
    cv2.ellipse(canvas, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(canvas, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(canvas, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(canvas, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)


def qimage_from_bgr_frame(frame: np.ndarray) -> QImage:
    if frame.ndim == 2:
        height, width = frame.shape
        return QImage(frame.data, width, height, width, QImage.Format.Format_Grayscale8).copy()

    rgb = frame[..., ::-1].copy()
    height, width, channels = rgb.shape
    bytes_per_line = channels * width
    return QImage(rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).copy()


def dialog_stylesheet() -> str:
    return f"""
        QDialog {{
            background: {theme.APP_BG};
        }}
        QLabel {{
            color: {theme.TEXT};
            font-size: 15px;
        }}
        QLabel#helpText {{
            font-size: 17px;
            font-weight: 700;
            color: {theme.TEXT_SUBTLE};
        }}
        QPushButton {{
            background: {theme.SURFACE};
            color: {theme.TEXT};
            border: 1px solid {theme.TEXT_MUTED};
            border-radius: 7px;
            padding: 10px 14px;
            font-weight: 800;
            font-size: 15px;
        }}
        QPushButton:checked {{
            background: {theme.BLUE};
            color: {theme.SURFACE};
        }}
        QPushButton#redCardSlot:checked {{
            background: {theme.RED};
            color: {theme.SURFACE};
            border-color: {theme.RED_DARK};
        }}
        QPushButton#sideCardSlot:checked {{
            background: {theme.LINE};
            color: {theme.SURFACE};
            border-color: {theme.TEXT_MUTED};
        }}
        QPushButton#blueCardSlot:checked {{
            background: {theme.BLUE};
            color: {theme.SURFACE};
            border-color: {theme.BLUE_DARK};
        }}
        QPushButton:disabled {{
            background: {theme.SURFACE_ALT};
            color: {theme.TEXT_MUTED};
        }}
        QLabel#feedbackLabel {{
            margin-top: 2px;
            margin-bottom: 2px;
        }}
    """
