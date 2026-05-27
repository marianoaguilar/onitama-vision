from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from onitama.runtime.vision_models import VisionRuntimeConfig, VisionRuntimeState
from onitama.engine.moves import Move
from onitama.engine.pieces import Player
from onitama.gui.camera_window import CameraWindow
from onitama.gui.calibration.board_calibration_dialog import BoardCalibrationDialog
from onitama.gui.calibration.card_rois_calibration_dialog import CardRoisCalibrationDialog
from onitama.gui.runtime_worker import RuntimeWorker
from onitama.gui.setup_page import SetupPage
from onitama.gui import theme
from onitama.gui.view_logic import StatusView, build_status_view
from onitama.gui.widgets import BoardWidget, CardWidget, MessageBanner


_CALIBRATION_PATH = Path("data/vision/calibration.json")
_CARD_ROIS_PATH = Path("data/vision/card_rois.json")
_CARD_ROI_SLOTS = ("red_0", "red_1", "side", "blue_0", "blue_1")
_DEFAULT_REQUIRED_REPEATS = 3
_DEFAULT_AI_DEPTH = 5


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Onitama")
        self.resize(1180, 820)

        self._worker: RuntimeWorker | None = None
        self._latest_state: VisionRuntimeState | None = None
        self._camera_window: CameraWindow | None = None
        self._calibration_dialog: QDialog | None = None
        self._close_requested = False

        self._finish_button = QPushButton("Finalizar partida")
        self._reset_button = QPushButton("Reiniciar")
        self._camera_button = QPushButton("Cámara")
        self._game_quit_button = QPushButton("Salir")
        
        self._game_quit_button.setStyleSheet(
            f"""
            QPushButton {{
                background: {theme.BUTTON_SECONDARY};
            }}
            """
        )

        self._setup_page = SetupPage()
        self._board = BoardWidget()
        self._red_cards = [CardWidget("Carta roja 2"), CardWidget("Carta roja 1")]
        self._blue_cards = [CardWidget("Carta azul 2"), CardWidget("Carta azul 1")]
        self._red_cards_label: QLabel | None = None
        self._blue_cards_label: QLabel | None = None
        self._side_card = CardWidget("Carta lateral")
        self._message = MessageBanner()
        self._stack = QStackedWidget()

        self._build_layout()
        self._connect_signals()
        self._set_running_controls(False)
        self._refresh_calibration_status()
        self._update_message_size()
        self._apply_state(None)

    def _build_layout(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        root.setStyleSheet(
            f"""
            QWidget#root {{
                background: {theme.APP_BG};
            }}
            QLabel {{
                color: {theme.TEXT};
            }}
            QPushButton {{
                background: {theme.TEXT};
                color: {theme.WHITE};
                border: 0;
                border-radius: 7px;
                padding: 12px 18px;
                font-weight: 700;
                font-size: 15px;
            }}
            QPushButton:disabled {{
                background: {theme.DISABLED};
            }}
            """
        )
        self.setCentralWidget(root)

        main = QVBoxLayout(root)
        main.setContentsMargins(18, 14, 18, 16)
        main.setSpacing(14)
        main.addWidget(self._stack)

        self._stack.addWidget(self._setup_page)
        self._stack.addWidget(self._build_game_page())

    def _build_game_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        layout.addWidget(self._message)
        layout.addWidget(self._build_game_area(), stretch=1)
        layout.addWidget(self._build_side_card_row())
        layout.addLayout(self._build_toolbar())
        return page

    def _build_side_card_row(self) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch(3)
        layout.addWidget(self._side_card, stretch=2)
        layout.addStretch(3)
        return row

    def _build_toolbar(self) -> QHBoxLayout:
        toolbar = QHBoxLayout()
        toolbar.setSpacing(10)

        toolbar.addWidget(self._camera_button)
        toolbar.addStretch(1)
        toolbar.addWidget(self._finish_button)
        toolbar.addWidget(self._reset_button)
        toolbar.addWidget(self._game_quit_button)
        return toolbar

    def _build_game_area(self) -> QWidget:
        area = QWidget()
        layout = QHBoxLayout(area)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        left_cards, self._red_cards_label = self._build_card_column(self._red_cards)
        right_cards, self._blue_cards_label = self._build_card_column(self._blue_cards)
        layout.addWidget(left_cards, stretch=1)
        layout.addWidget(self._board, stretch=2)
        layout.addWidget(right_cards, stretch=1)
        return area

    def _build_card_column(self, cards: list[CardWidget]) -> tuple[QWidget, QLabel]:
        frame = QFrame()
        frame.setMinimumWidth(220)
        frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 16, 0, 16)
        layout.setSpacing(12)

        label = QLabel()
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("font-size: 18px; font-weight: 800;")
        layout.addWidget(label)
        layout.addStretch(1)
        for card in cards:
            layout.addWidget(card, stretch=3)
        layout.addStretch(1)
        return frame, label

    def _connect_signals(self) -> None:
        self._setup_page.start_requested.connect(self._start_runtime_from_setup)
        self._setup_page.calibrate_board_requested.connect(self._calibrate_board)
        self._setup_page.calibrate_cards_requested.connect(self._calibrate_cards)
        self._setup_page.refresh_calibration_requested.connect(self._refresh_calibration_status)
        self._setup_page.quit_requested.connect(QApplication.instance().quit)
        self._finish_button.clicked.connect(self._finish_game)
        self._reset_button.clicked.connect(self._reset_runtime)
        self._camera_button.clicked.connect(self._show_camera)
        self._game_quit_button.clicked.connect(QApplication.instance().quit)

    @Slot()
    def _finish_game(self) -> None:
        self._stop_runtime()
        self._latest_state = None
        self._apply_state(None)
        self._refresh_calibration_status()
        self._stack.setCurrentIndex(0)

    @Slot()
    def _start_runtime_from_setup(self) -> None:
        if not self._refresh_calibration_status():
            return
        self._latest_state = None
        self._apply_state(None)
        self._update_card_column_labels()
        self._stack.setCurrentIndex(1)
        self._start_runtime()

    def _start_runtime(self) -> None:
        if self._worker is not None:
            return
        config = self._build_config()
        worker = RuntimeWorker(config, self)
        worker.state_changed.connect(self._on_state_changed)
        worker.frame_changed.connect(self._on_frame_changed)
        worker.failed.connect(self._on_worker_failed)
        worker.finished.connect(self._on_worker_finished)
        self._worker = worker
        if self._camera_window is not None:
            worker.set_camera_stream_enabled(True)
        self._set_running_controls(True)
        worker.start()

    def _build_config(self) -> VisionRuntimeConfig:
        return VisionRuntimeConfig(
            human_player=self._setup_page.human_player(),
            required_repeats=_DEFAULT_REQUIRED_REPEATS,
            ai_depth=_DEFAULT_AI_DEPTH,
            ai_evaluator=self._setup_page.ai_evaluator(),
        )

    def _update_card_column_labels(self) -> None:
        human_player = self._setup_page.human_player()
        if self._red_cards_label is not None:
            self._red_cards_label.setText("HUMANO" if human_player is Player.RED else "IA")
        if self._blue_cards_label is not None:
            self._blue_cards_label.setText("HUMANO" if human_player is Player.BLUE else "IA")

    @Slot()
    def _reset_runtime(self) -> None:
        if self._worker is not None:
            self._worker.request_reset()
        self._latest_state = None
        self._apply_state(None)

    def _stop_runtime(self) -> None:
        if self._worker is None:
            return
        self._worker.request_stop()
        self._worker.wait(2500)
        if self._worker is not None and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait(1000)
        self._worker = None
        self._set_running_controls(False)

    def _request_runtime_stop_for_close(self) -> None:
        if self._worker is None:
            return
        self._worker.request_stop()
        self._set_running_controls(False)

    def _close_calibration_dialog(self) -> None:
        if self._calibration_dialog is None:
            return
        self._calibration_dialog.close()
        self._calibration_dialog = None

    @Slot()
    def _show_camera(self) -> None:
        if self._camera_window is None:
            self._camera_window = CameraWindow(self)
            self._camera_window.destroyed.connect(self._on_camera_closed)
        self._camera_window.show()
        self._camera_window.raise_()
        self._camera_window.activateWindow()
        if self._worker is not None:
            self._worker.set_camera_stream_enabled(True)

    @Slot(object)
    def _on_camera_closed(self, _obj=None) -> None:
        self._camera_window = None
        if self._worker is not None:
            self._worker.set_camera_stream_enabled(False)

    @Slot(object)
    def _on_state_changed(self, state: VisionRuntimeState) -> None:
        self._latest_state = state
        self._apply_state(state)

    @Slot(object)
    def _on_frame_changed(self, image) -> None:
        if self._camera_window is not None:
            self._camera_window.set_frame(image)

    @Slot(str)
    def _on_worker_failed(self, message: str) -> None:
        self._message.apply(StatusView("Error de ejecucion", message, "error"))
        self._set_running_controls(False)
        self._worker = None
        if self._close_requested:
            self.close()

    @Slot()
    def _on_worker_finished(self) -> None:
        self._set_running_controls(False)
        self._worker = None
        if self._close_requested:
            self.close()

    def _apply_state(self, state: VisionRuntimeState | None) -> None:
        current_state = state.current_state if state is not None else None
        highlight = state.ai_action if state is not None and isinstance(state.ai_action, Move) else None
        highlighted_player = current_state.to_move if current_state is not None and state is not None and state.ai_action is not None else None
        highlighted_card_index = state.ai_action.card_index if state is not None and state.ai_action is not None else None

        self._board.set_state(current_state, highlight)
        if current_state is None:
            for card in self._red_cards + self._blue_cards:
                card.set_card(None)
            self._side_card.set_card(None)
        else:
            self._red_cards[0].set_card(
                current_state.red_cards[1],
                "Carta roja 2",
                Player.RED,
                highlighted=highlighted_player is Player.RED and highlighted_card_index == 1,
            )
            self._red_cards[1].set_card(
                current_state.red_cards[0],
                "Carta roja 1",
                Player.RED,
                highlighted=highlighted_player is Player.RED and highlighted_card_index == 0,
            )
            self._blue_cards[0].set_card(
                current_state.blue_cards[1],
                "Carta azul 2",
                Player.BLUE,
                highlighted=highlighted_player is Player.BLUE and highlighted_card_index == 1,
            )
            self._blue_cards[1].set_card(
                current_state.blue_cards[0],
                "Carta azul 1",
                Player.BLUE,
                highlighted=highlighted_player is Player.BLUE and highlighted_card_index == 0,
            )
            self._side_card.set_card(current_state.side_card, "Carta lateral")

        self._message.apply(build_status_view(state))

    def _set_running_controls(self, running: bool) -> None:
        self._finish_button.setEnabled(running)
        self._setup_page.set_controls_enabled(not running)
        self._reset_button.setEnabled(running)

    def _refresh_calibration_status(self) -> bool:
        board_ok, board_message = self._check_board_calibration()
        cards_ok, cards_message = self._check_card_rois()
        ready = board_ok and cards_ok

        self._setup_page.set_calibration_status(
            ready=ready,
            board_message=board_message,
            cards_message=cards_message,
            start_enabled=ready and self._calibration_dialog is None,
        )
        return ready

    def _check_board_calibration(self) -> tuple[bool, str]:
        if not _CALIBRATION_PATH.exists():
            return False, f"falta {_CALIBRATION_PATH}"
        try:
            data = json.loads(_CALIBRATION_PATH.read_text(encoding="utf-8"))
            src_points = data["src_points"]
            dst_size = data["dst_size"]
            rotate = int(data.get("rotate", 0))
            if len(src_points) != 4:
                return False, "las esquinas del tablero no son validas"
            if len(dst_size) != 2 or int(dst_size[0]) < 1 or int(dst_size[1]) < 1:
                return False, "el tamano de destino no es valido"
            if rotate not in (0, 90, 180, 270):
                return False, "la rotacion no es valida"
        except Exception as exc:  # noqa: BLE001
            return False, f"archivo no valido ({type(exc).__name__}: {exc})"
        return True, f"cargado {_CALIBRATION_PATH}"

    def _check_card_rois(self) -> tuple[bool, str]:
        if not _CARD_ROIS_PATH.exists():
            return False, f"falta {_CARD_ROIS_PATH}"
        try:
            data = json.loads(_CARD_ROIS_PATH.read_text(encoding="utf-8"))
            for slot in _CARD_ROI_SLOTS:
                points = data[slot]["src_points"]
                if len(points) != 4:
                    return False, f"{slot} necesita 4 puntos"
        except Exception as exc:  # noqa: BLE001
            return False, f"archivo no valido ({type(exc).__name__}: {exc})"
        return True, f"cargado {_CARD_ROIS_PATH}"

    @Slot()
    def _calibrate_board(self) -> None:
        if not self._can_open_calibration_dialog():
            return
        self._open_calibration_dialog(BoardCalibrationDialog(_CALIBRATION_PATH, self))

    @Slot()
    def _calibrate_cards(self) -> None:
        if not self._can_open_calibration_dialog():
            return
        self._open_calibration_dialog(CardRoisCalibrationDialog(_CARD_ROIS_PATH, self))

    def _can_open_calibration_dialog(self) -> bool:
        if self._calibration_dialog is not None:
            return False
        if self._worker is not None:
            self._message.apply(StatusView("Para la partida primero", "No se puede calibrar mientras la visión está activa.", "warning"))
            return False
        return True

    def _open_calibration_dialog(self, dialog: QDialog) -> None:
        self._calibration_dialog = dialog
        self._setup_page.set_calibration_buttons_enabled(False)
        self._refresh_calibration_status()
        if isinstance(dialog, (BoardCalibrationDialog, CardRoisCalibrationDialog)):
            dialog.saved.connect(self._refresh_calibration_status)
        dialog.finished.connect(self._on_calibration_dialog_finished)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    @Slot()
    def _on_calibration_dialog_finished(self) -> None:
        if self._calibration_dialog is not None:
            self._calibration_dialog.deleteLater()
        self._calibration_dialog = None
        self._setup_page.set_calibration_buttons_enabled(True)
        self._refresh_calibration_status()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        if self._worker is not None:
            self._close_requested = True
            self._request_runtime_stop_for_close()
            self._close_calibration_dialog()
            event.ignore()
            return
        self._close_calibration_dialog()
        super().closeEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._update_message_size()

    def _update_message_size(self) -> None:
        self._message.set_responsive_size(self.height())
