from __future__ import annotations

from functools import partial
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from onitama.ai.profiles import AI_PROFILES, DEFAULT_AI_PROFILE_ID
from onitama.engine.pieces import Player
from onitama.gui import theme


_STATUS_BASE_STYLE = """
font-size: 18px;
font-weight: 900;
padding: 8px 14px;
border-radius: 14px;
"""

_DETAIL_STYLE = f"""
font-size: 16px;
font-weight: 700;
color: {theme.TEXT_MUTED};
background: {theme.TRANSPARENT};
border: 0;
padding: 6px 0;
"""


class SetupPage(QWidget):
    start_requested = Signal()
    calibrate_board_requested = Signal()
    calibrate_cards_requested = Signal()
    refresh_calibration_requested = Signal()
    quit_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("setupPage")

        self._human_player = Player.RED
        self._profile_options = [(profile.label, profile.id) for profile in AI_PROFILES.values()]
        self._ai_profile_id = DEFAULT_AI_PROFILE_ID

        self._color_group = QButtonGroup(self)
        self._color_group.setExclusive(True)
        self._difficulty_group = QButtonGroup(self)
        self._difficulty_group.setExclusive(True)
        self._color_buttons: list[QPushButton] = []
        self._difficulty_buttons: list[QPushButton] = []

        self._start_button = QPushButton("Iniciar partida")
        self._quit_button = QPushButton("Salir")
        self._calibrate_board_button = QPushButton("Calibrar tablero")
        self._calibrate_cards_button = QPushButton("Calibrar cartas")
        self._refresh_calibration_button = QPushButton("Actualizar")
        self._calibration_status = QLabel()
        self._calibration_detail = QLabel()

        self._start_button.setObjectName("primaryAction")
        self._quit_button.setObjectName("quietAction")
        self._refresh_calibration_button.setObjectName("refreshAction")
        self._start_button.setMinimumHeight(64)
        self._quit_button.setMinimumHeight(48)
        self._calibrate_board_button.setMinimumHeight(50)
        self._calibrate_cards_button.setMinimumHeight(50)
        self._refresh_calibration_button.setMinimumHeight(50)

        self._build_layout()
        self._connect_signals()

    def human_player(self) -> Player:
        return self._human_player

    def ai_profile_id(self) -> str:
        return self._ai_profile_id

    def set_controls_enabled(self, enabled: bool) -> None:
        for button in self._color_buttons + self._difficulty_buttons:
            button.setEnabled(enabled)

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
            _STATUS_BASE_STYLE
            + (
                f"color: {theme.SUCCESS_TEXT}; background: {theme.SUCCESS_BG}; border: 1px solid {theme.SUCCESS_BORDER};"
                if ready
                else f"color: {theme.RED_DARK}; background: {theme.ERROR_BG}; border: 1px solid {theme.ERROR_BORDER};"
            )
        )
        self._calibration_detail.setText(
            f"{self._format_calibration_detail('Tablero', board_message)}\n"
            f"{self._format_calibration_detail('Cartas', cards_message)}"
        )
        self._start_button.setEnabled(start_enabled)

    def set_calibration_running(self, script: Path) -> None:
        self._calibration_status.setText("Calibracion en curso")
        self._calibration_status.setStyleSheet(
            _STATUS_BASE_STYLE
            + f"color: {theme.BLUE_DARK}; background: {theme.INFO_BG}; border: 1px solid {theme.INFO_BORDER};"
        )
        self._calibration_detail.setText(f"Ejecutando {script.name}. Cierra la ventana de OpenCV al terminar.")
        self._start_button.setEnabled(False)

    def set_calibration_script_missing(self, script: Path) -> None:
        self._calibration_status.setText("No se encontró el script de calibración")
        self._calibration_status.setStyleSheet(
            _STATUS_BASE_STYLE
            + f"color: {theme.RED_DARK}; background: {theme.ERROR_BG}; border: 1px solid {theme.ERROR_BORDER};"
        )
        self._calibration_detail.setText(f"No se encuentra {script.name}")
        self._start_button.setEnabled(False)

    def set_calibration_exit_error(self, exit_code: int, stderr: str) -> None:
        self._calibration_detail.setText(f"La calibración terminó con código {exit_code}.\n{stderr}")

    def set_calibration_error(self, message: str) -> None:
        self._calibration_status.setText("La calibración falló")
        self._calibration_status.setStyleSheet(
            _STATUS_BASE_STYLE
            + f"color: {theme.RED_DARK}; background: {theme.ERROR_BG}; border: 1px solid {theme.ERROR_BORDER};"
        )
        self._calibration_detail.setText(message)

    def _build_layout(self) -> None:
        self.setStyleSheet(
            f"""
            QWidget#setupPage {{
                background: {theme.APP_BG};
            }}
            QFrame#pageHeader {{
                background: {theme.TRANSPARENT};
            }}
            QFrame#setupPanel {{
                background: {theme.SURFACE_ALT};
                border: 2px solid {theme.TEXT_MUTED};
                border-radius: 18px;
            }}
            QLabel#headerKicker {{
                color: {theme.RED_DARK};
                font-size: 24px;
                font-weight: 900;
            }}
            QLabel#headerTitle {{
                color: {theme.TEXT};
                font-size: 76px;
                font-weight: 900;
            }}
            QLabel#stepBadge {{
                background: {theme.RED_ACTION_HOVER};
                color: {theme.SURFACE};
                border-radius: 17px;
                font-size: 18px;
                font-weight: 900;
                min-width: 34px;
                min-height: 34px;
            }}
            QLabel#fieldLabel {{
                color: {theme.TEXT};
                font-size: 22px;
                font-weight: 900;
            }}
            QFrame#line {{
                background: {theme.LINE};
                max-height: 1px;
            }}
            QPushButton#redChoice,
            QPushButton#blueChoice {{
                background: {theme.SURFACE};
                color: {theme.TEXT};
                border: 2px solid {theme.LINE};
                border-radius: 12px;
                padding: 14px 18px;
                font-size: 20px;
                font-weight: 900;
                text-align: center;
                min-height: 52px;
            }}
            QPushButton#redChoice:checked {{
                color: {theme.RED_ACTION_HOVER};
                border-color: {theme.RED};
                background: {theme.RED_SOFT};
            }}
            QPushButton#blueChoice:checked {{
                color: {theme.BLUE_DARK};
                border-color: {theme.BLUE_DARK};
                background: {theme.BLUE_SOFT};
            }}
            QPushButton#choiceButton {{
                background: {theme.TRANSPARENT};
                color: {theme.TEXT};
                border: 0;
                border-radius: 21px;
                padding: 12px 20px;
                font-size: 20px;
                font-weight: 900;
                min-height: 42px;
            }}
            QPushButton#choiceButton:checked {{
                background: {theme.BLUE};
                color: {theme.SURFACE};
                border-radius: 21px;
            }}
            QFrame#segmentedControl {{
                background: {theme.SURFACE};
                border: 1px solid {theme.TEXT_MUTED};
                border-radius: 23px;
            }}
            QPushButton {{
                background: {theme.SURFACE};
                color: {theme.TEXT};
                border: 1px solid {theme.TEXT_MUTED};
                border-radius: 7px;
                padding: 12px 16px;
                font-weight: 900;
                font-size: 18px;
            }}
            QPushButton:disabled {{
                background: {theme.SURFACE_ALT};
                color: {theme.TEXT_MUTED};
                border-color: {theme.LINE};
            }}
            QPushButton:hover,
            QPushButton#primaryAction:hover,
            QPushButton#quietAction:hover,
            QPushButton#refreshAction:hover {{
                background: {theme.LINE};
                color: {theme.SURFACE};
            }}
            QPushButton#redChoice:hover {{
                background: {theme.SURFACE};
                color: {theme.TEXT};
            }}
            QPushButton#redChoice:checked:hover {{
                color: {theme.RED_ACTION_HOVER};
                background: {theme.RED_SOFT};
            }}
            QPushButton#blueChoice:hover {{
                background: {theme.SURFACE};
                color: {theme.TEXT};
            }}
            QPushButton#blueChoice:checked:hover {{
                color: {theme.BLUE_DARK};
                background: {theme.BLUE_SOFT};
            }}
            QPushButton#choiceButton:hover {{
                background: {theme.TRANSPARENT};
                color: {theme.TEXT};
            }}
            QPushButton#choiceButton:checked:hover {{
                background: {theme.BLUE};
                color: {theme.SURFACE};
            }}
            QPushButton#primaryAction {{
                background: {theme.RED};
                color: {theme.SURFACE};
                border: 2px solid {theme.RED_DARK};
                border-radius: 14px;
                font-size: 22px;
                padding: 16px 24px;
            }}
            QPushButton#primaryAction:hover {{
                background: {theme.RED_ACTION_HOVER};
                color: {theme.SURFACE};
            }}
            QPushButton#quietAction {{
                background: {theme.SURFACE};
                color: {theme.TEXT_MUTED};
                border: 1px solid {theme.TEXT_MUTED};
            }}
            QPushButton#refreshAction {{
                background: {theme.SURFACE};
                color: {theme.TEXT_MUTED};
            }}
            """
        )

        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(28, 22, 28, 28)
        page_layout.setSpacing(12)

        header = QFrame()
        header.setObjectName("pageHeader")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(34, 6, 34, 4)
        header_layout.setSpacing(0)

        title = QLabel("Onitama")
        title.setObjectName("headerTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(title)

        kicker = QLabel("Configuración de partida")
        kicker.setObjectName("headerKicker")
        kicker.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(kicker)

        page_layout.addWidget(header)

        controls = QFrame()
        controls.setObjectName("setupPanel")
        controls.setMinimumWidth(800)
        controls.setMaximumWidth(1000)
        controls.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        shadow = QGraphicsDropShadowEffect(controls)
        shadow.setBlurRadius(40)
        shadow.setColor(QColor(*theme.SHADOW_RGBA))
        shadow.setOffset(0, 12)
        controls.setGraphicsEffect(shadow)

        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(38, 28, 38, 28)
        controls_layout.setSpacing(14)

        controls_layout.addWidget(
            self._build_setup_card(
                "1",
                "Elige tu color",
                self._build_color_options(),
            )
        )
        controls_layout.addWidget(self._separator())
        controls_layout.addWidget(
            self._build_setup_card(
                "2",
                "Dificultad de la IA",
                self._build_difficulty_options(),
            )
        )
        controls_layout.addWidget(self._separator())

        calibration_card = self._build_calibration_card()
        calibration_card.setContentsMargins(0, 8, 0, 0)
        controls_layout.addWidget(calibration_card)
        controls_layout.addWidget(self._separator())

        actions = QHBoxLayout()
        actions.setSpacing(18)
        actions.setContentsMargins(0, 20, 0, 0)
        actions.addWidget(self._start_button)
        actions.addStretch(1)
        actions.addWidget(self._quit_button)
        controls_layout.addLayout(actions)

        centered_controls = QHBoxLayout()
        centered_controls.setContentsMargins(0, 0, 0, 0)
        centered_controls.addStretch(1)
        centered_controls.addWidget(controls, stretch=8)
        centered_controls.addStretch(1)
        page_layout.addLayout(centered_controls, stretch=1)

    def _connect_signals(self) -> None:
        self._start_button.clicked.connect(self.start_requested.emit)
        self._calibrate_board_button.clicked.connect(self.calibrate_board_requested.emit)
        self._calibrate_cards_button.clicked.connect(self.calibrate_cards_requested.emit)
        self._refresh_calibration_button.clicked.connect(self.refresh_calibration_requested.emit)
        self._quit_button.clicked.connect(self.quit_requested.emit)

    def _build_setup_card(self, step: str, label: str, widget: QWidget) -> QWidget:
        field = QWidget()
        layout = QVBoxLayout(field)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        layout.addLayout(self._build_step_header(step, label))
        layout.addWidget(widget)
        return field

    def _build_color_options(self) -> QWidget:
        wrapper = QWidget()
        layout = QHBoxLayout(wrapper)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(16)

        red_button = self._build_choice_button("Rojo", "redChoice")
        blue_button = self._build_choice_button("Azul", "blueChoice")
        self._color_group.addButton(red_button, 0)
        self._color_group.addButton(blue_button, 1)
        self._color_buttons = [red_button, blue_button]
        red_button.setChecked(True)

        red_button.clicked.connect(lambda _checked=False: self._set_human_player(Player.RED))
        blue_button.clicked.connect(lambda _checked=False: self._set_human_player(Player.BLUE))
        layout.addWidget(red_button)
        layout.addWidget(blue_button)
        return wrapper

    def _build_difficulty_options(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("segmentedControl")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(0)

        self._difficulty_buttons = []
        for index, (label, profile_id) in enumerate(self._profile_options):
            button = self._build_choice_button(label, "choiceButton")
            self._difficulty_group.addButton(button, index)
            button.clicked.connect(partial(self._set_ai_profile, profile_id))
            self._difficulty_buttons.append(button)
            layout.addWidget(button)

        default_index = next(
            (
                index
                for index, (_label, profile_id) in enumerate(self._profile_options)
                if profile_id == DEFAULT_AI_PROFILE_ID
            ),
            len(self._difficulty_buttons) - 1,
        )
        if 0 <= default_index < len(self._difficulty_buttons):
            self._difficulty_buttons[default_index].setChecked(True)
        return frame

    def _build_calibration_card(self) -> QWidget:
        frame = QWidget()
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        header = self._build_step_header("3", "Calibración")
        header.addStretch(1)
        self._calibration_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._calibration_status.setStyleSheet(_STATUS_BASE_STYLE)
        header.addWidget(self._calibration_status)
        layout.addLayout(header)

        self._calibration_detail.setWordWrap(True)
        self._calibration_detail.setStyleSheet(_DETAIL_STYLE)
        layout.addWidget(self._calibration_detail)

        calibration_buttons = QHBoxLayout()
        calibration_buttons.setSpacing(16)
        calibration_buttons.addWidget(self._calibrate_board_button)
        calibration_buttons.addWidget(self._calibrate_cards_button)
        calibration_buttons.addWidget(self._refresh_calibration_button)
        layout.addLayout(calibration_buttons)
        return frame

    def _build_step_header(self, step: str, label: str) -> QHBoxLayout:
        header = QHBoxLayout()
        header.setSpacing(12)

        badge = QLabel(step)
        badge.setObjectName("stepBadge")
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setFixedSize(34, 34)
        header.addWidget(badge)

        title = QLabel(label)
        title.setObjectName("fieldLabel")
        header.addWidget(title)
        return header

    def _build_choice_button(self, text: str, object_name: str) -> QPushButton:
        button = QPushButton(text)
        button.setObjectName(object_name)
        button.setCheckable(True)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        return button

    def _set_human_player(self, player: Player) -> None:
        self._human_player = player

    def _set_ai_profile(self, profile_id: str, _checked: bool = False) -> None:
        self._ai_profile_id = profile_id

    def _separator(self) -> QFrame:
        line = QFrame()
        line.setObjectName("line")
        line.setFixedHeight(1)
        return line

    def _format_calibration_detail(self, label: str, message: str) -> str:
        if message.startswith("cargado "):
            return f"{label}: calibrado"
        if message.startswith("falta "):
            return f"{label}: pendiente"
        return f"{label}: {message}"
