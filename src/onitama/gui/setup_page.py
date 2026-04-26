from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QComboBox, QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from onitama.ai.evaluate import EVALUATORS
from onitama.engine.pieces import Player


_DIFFICULTIES = (
    ("Facil - Heuristica 1", "v1"),
    ("Media - Heuristica 2", "v2"),
    ("Dificil - Heuristica 3", "v3"),
)


class SetupPreview(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(340, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def sizeHint(self) -> QSize:
        return QSize(430, 430)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        bounds = QRectF(self.rect()).adjusted(18, 18, -18, -18)
        side = min(bounds.width(), bounds.height()) * 0.76
        board = QRectF(
            bounds.center().x() - side / 2,
            bounds.center().y() - side / 2,
            side,
            side,
        )
        cell = side / 5

        painter.setPen(QPen(QColor("#6f4e2c"), 2))
        for row in range(5):
            for col in range(5):
                fill = QColor("#f2dfb3") if (row + col) % 2 == 0 else QColor("#d6b479")
                painter.setBrush(fill)
                painter.drawRect(QRectF(board.left() + col * cell, board.top() + row * cell, cell, cell))

        painter.setPen(QPen(QColor("#8a6b3f"), 2))
        painter.setBrush(QColor("#fff7df"))
        card_w = side * 0.34
        card_h = side * 0.20
        painter.drawRoundedRect(QRectF(board.left() - card_w * 0.70, board.top() + cell * 0.6, card_w, card_h), 8, 8)
        painter.drawRoundedRect(QRectF(board.right() - card_w * 0.30, board.bottom() - cell * 1.5, card_w, card_h), 8, 8)
        painter.drawRoundedRect(QRectF(board.center().x() - card_w / 2, board.top() - card_h * 0.65, card_w, card_h), 8, 8)

        pieces = (
            (0, 0, QColor("#c2413b"), QColor("#7f1d1d")),
            (2, 0, QColor("#c2413b"), QColor("#7f1d1d")),
            (4, 0, QColor("#c2413b"), QColor("#7f1d1d")),
            (0, 4, QColor("#2563eb"), QColor("#1e3a8a")),
            (2, 4, QColor("#2563eb"), QColor("#1e3a8a")),
            (4, 4, QColor("#2563eb"), QColor("#1e3a8a")),
        )
        for row, col, color, outline in pieces:
            cx = board.left() + col * cell + cell / 2
            cy = board.top() + row * cell + cell / 2
            painter.setPen(QPen(outline, 3))
            painter.setBrush(color)
            painter.drawEllipse(QPointF(cx, cy), cell * 0.25, cell * 0.25)

        font = QFont(self.font())
        font.setPointSize(13)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(QColor("#4b3621")))
        painter.drawText(QRectF(bounds.left(), bounds.bottom() - 28, bounds.width(), 24), Qt.AlignmentFlag.AlignCenter, "Camara + tablero fisico")


class SetupPage(QWidget):
    start_requested = Signal()
    calibrate_board_requested = Signal()
    calibrate_cards_requested = Signal()
    refresh_calibration_requested = Signal()
    quit_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._human_combo = QComboBox()
        self._human_combo.addItem("Rojo", Player.RED)
        self._human_combo.addItem("Azul", Player.BLUE)
        self._human_combo.setMinimumWidth(280)

        self._evaluator_combo = QComboBox()
        for label, evaluator_name in _DIFFICULTIES:
            if evaluator_name in EVALUATORS:
                self._evaluator_combo.addItem(label, evaluator_name)
        if self._evaluator_combo.count() == 0:
            for evaluator_name in sorted(EVALUATORS.keys()):
                self._evaluator_combo.addItem(evaluator_name, evaluator_name)
        self._evaluator_combo.setCurrentIndex(self._evaluator_combo.count() - 1)
        self._evaluator_combo.setMinimumWidth(320)

        self._start_button = QPushButton("Empezar partida")
        self._quit_button = QPushButton("Salir")
        self._calibrate_board_button = QPushButton("Calibrar tablero")
        self._calibrate_cards_button = QPushButton("Calibrar cartas")
        self._refresh_calibration_button = QPushButton("Actualizar")
        self._calibration_status = QLabel()
        self._calibration_detail = QLabel()

        self._start_button.setMinimumHeight(48)
        self._quit_button.setMinimumHeight(48)
        self._start_button.setStyleSheet(
            """
            QPushButton {
                background: #1f2933;
                color: white;
                border: 0;
                border-radius: 8px;
                padding: 12px 28px;
                font-size: 17px;
                font-weight: 900;
            }
            QPushButton:disabled {
                background: #9ca3af;
            }
            """
        )
        self._quit_button.setStyleSheet(
            """
            QPushButton {
                background: #6b7280;
                color: white;
                border: 0;
                border-radius: 8px;
                padding: 12px 22px;
                font-size: 16px;
                font-weight: 800;
            }
            """
        )

        self._build_layout()
        self._connect_signals()

    def human_player(self) -> Player:
        player = self._human_combo.currentData()
        return player if player in {Player.RED, Player.BLUE} else Player.RED

    def ai_evaluator(self) -> str:
        evaluator_name = self._evaluator_combo.currentData()
        if isinstance(evaluator_name, str):
            return evaluator_name
        return "v3" if "v3" in EVALUATORS else next(iter(EVALUATORS))

    def set_controls_enabled(self, enabled: bool) -> None:
        self._human_combo.setEnabled(enabled)
        self._evaluator_combo.setEnabled(enabled)

    def set_calibration_buttons_enabled(self, enabled: bool) -> None:
        self._calibrate_board_button.setEnabled(enabled)
        self._calibrate_cards_button.setEnabled(enabled)
        self._refresh_calibration_button.setEnabled(enabled)

    def set_calibration_status(
        self,
        *,
        ready: bool,
        board_message: str,
        cards_message: str,
        start_enabled: bool,
    ) -> None:
        self._calibration_status.setText("Listo para jugar" if ready else "Hace falta calibrar")
        self._calibration_status.setStyleSheet(
            "font-size: 18px; font-weight: 800; "
            f"color: {'#059669' if ready else '#dc2626'};"
        )
        self._calibration_detail.setText(
            f"Tablero: {board_message}\nCartas: {cards_message}"
        )
        self._start_button.setEnabled(start_enabled)

    def set_calibration_running(self, script: Path) -> None:
        self._calibration_status.setText("Calibracion en curso")
        self._calibration_status.setStyleSheet("font-size: 18px; font-weight: 800; color: #2563eb;")
        self._calibration_detail.setText(f"Ejecutando {script}. Usa los controles de la ventana de OpenCV y cierrala al terminar.")
        self._start_button.setEnabled(False)

    def set_calibration_script_missing(self, script: Path) -> None:
        self._calibration_status.setText("No se encontro el script de calibracion")
        self._calibration_status.setStyleSheet("font-size: 18px; font-weight: 800; color: #dc2626;")
        self._calibration_detail.setText(str(script))
        self._start_button.setEnabled(False)

    def set_calibration_exit_error(self, exit_code: int, stderr: str) -> None:
        self._calibration_detail.setText(f"La calibracion termino con codigo {exit_code}.\n{stderr}")

    def set_calibration_error(self, message: str) -> None:
        self._calibration_status.setText("La calibracion fallo")
        self._calibration_status.setStyleSheet("font-size: 18px; font-weight: 800; color: #dc2626;")
        self._calibration_detail.setText(message)

    def _build_layout(self) -> None:
        page_layout = QHBoxLayout(self)
        page_layout.setContentsMargins(64, 46, 64, 46)
        page_layout.setSpacing(34)

        intro = QFrame()
        intro.setObjectName("introPanel")
        intro.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        intro.setStyleSheet(
            """
            QFrame#introPanel {
                background: #fff7df;
                border: 2px solid #b08a4d;
                border-radius: 14px;
            }
            """
        )
        intro_layout = QVBoxLayout(intro)
        intro_layout.setContentsMargins(34, 30, 34, 28)
        intro_layout.setSpacing(18)

        title = QLabel("Onitama")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 54px; font-weight: 900; color: #1f2933;")
        intro_layout.addWidget(title)

        subtitle = QLabel("Vision asistida para jugar contra la IA")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("font-size: 20px; font-weight: 700; color: #4b5563;")
        intro_layout.addWidget(subtitle)
        intro_layout.addWidget(SetupPreview(), stretch=1)

        controls = QFrame()
        controls.setObjectName("controlsPanel")
        controls.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        controls.setStyleSheet(
            """
            QFrame#controlsPanel {
                background: white;
                border: 2px solid #d1b17a;
                border-radius: 14px;
            }
            """
        )
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(34, 30, 34, 30)
        controls_layout.setSpacing(22)

        settings_title = QLabel("Configura la partida")
        settings_title.setStyleSheet("font-size: 30px; font-weight: 900; color: #1f2933;")
        controls_layout.addWidget(settings_title)

        controls_layout.addWidget(self._build_field("Tu color", self._human_combo))
        controls_layout.addWidget(self._build_field("Dificultad de la IA", self._evaluator_combo))

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet("color: #d1b17a;")
        controls_layout.addWidget(separator)

        calibration_title = QLabel("Calibracion")
        calibration_title.setStyleSheet("font-size: 24px; font-weight: 800;")
        controls_layout.addWidget(calibration_title)

        self._calibration_status.setStyleSheet("font-size: 18px; font-weight: 800;")
        self._calibration_detail.setWordWrap(True)
        self._calibration_detail.setStyleSheet("font-size: 15px; color: #4b5563;")
        controls_layout.addWidget(self._calibration_status)
        controls_layout.addWidget(self._calibration_detail)

        calibration_buttons = QHBoxLayout()
        calibration_buttons.setSpacing(10)
        calibration_buttons.addWidget(self._calibrate_board_button)
        calibration_buttons.addWidget(self._calibrate_cards_button)
        calibration_buttons.addWidget(self._refresh_calibration_button)
        calibration_buttons.addStretch(1)
        controls_layout.addLayout(calibration_buttons)

        controls_layout.addStretch(1)
        actions = QHBoxLayout()
        actions.addWidget(self._start_button)
        actions.addWidget(self._quit_button)
        controls_layout.addLayout(actions)

        page_layout.addWidget(intro, stretch=5)
        page_layout.addWidget(controls, stretch=4)

    def _connect_signals(self) -> None:
        self._start_button.clicked.connect(self.start_requested.emit)
        self._calibrate_board_button.clicked.connect(self.calibrate_board_requested.emit)
        self._calibrate_cards_button.clicked.connect(self.calibrate_cards_requested.emit)
        self._refresh_calibration_button.clicked.connect(self.refresh_calibration_requested.emit)
        self._quit_button.clicked.connect(self.quit_requested.emit)

    def _build_field(self, label: str, widget: QWidget) -> QWidget:
        field = QWidget()
        layout = QVBoxLayout(field)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(7)
        text = QLabel(label)
        text.setStyleSheet("font-size: 18px; font-weight: 800;")
        layout.addWidget(text)
        layout.addWidget(widget)
        return field
