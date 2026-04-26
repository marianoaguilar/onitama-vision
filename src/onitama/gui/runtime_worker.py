from __future__ import annotations

import threading

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from onitama.app.vision_models import VisionRuntimeConfig, VisionRuntimeState
from onitama.app.vision_runtime import VisionGameRuntime


class RuntimeWorker(QThread):
    state_changed = Signal(object)
    frame_changed = Signal(QImage)
    failed = Signal(str)

    def __init__(self, config: VisionRuntimeConfig, parent=None) -> None:
        super().__init__(parent)
        self._config = config
        self._runtime: VisionGameRuntime | None = None
        self._lock = threading.Lock()
        self._stop_requested = False
        self._camera_stream_enabled = False

    def run(self) -> None:
        try:
            runtime = VisionGameRuntime(self._config)
            with self._lock:
                self._runtime = runtime
            runtime.start()
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return

        try:
            while not self._stop_requested:
                with self._lock:
                    state = runtime.step()
                    stream_frame = self._camera_stream_enabled
                    frame = runtime.get_latest_frame() if stream_frame else None

                self.state_changed.emit(state)
                if frame is not None:
                    self.frame_changed.emit(_qimage_from_bgr_frame(frame))

                self.msleep(120)
        finally:
            with self._lock:
                runtime.stop()
                self._runtime = None

    def request_stop(self) -> None:
        self._stop_requested = True

    def request_reset(self) -> None:
        with self._lock:
            if self._runtime is not None:
                self._runtime.reset()

    def set_camera_stream_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._camera_stream_enabled = enabled


def _qimage_from_bgr_frame(frame) -> QImage:
    """Convert an OpenCV BGR frame into an owned QImage."""
    if frame.ndim == 2:
        height, width = frame.shape
        return QImage(frame.data, width, height, width, QImage.Format.Format_Grayscale8).copy()

    rgb = frame[..., ::-1].copy()
    height, width, channels = rgb.shape
    bytes_per_line = channels * width
    return QImage(rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).copy()

