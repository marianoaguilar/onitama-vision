from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QFrame, QLabel, QSizePolicy, QVBoxLayout, QWidget

from onitama.engine.cards import Card
from onitama.engine.moves import Move
from onitama.engine.pieces import PieceType, Player
from onitama.engine.state import GameState
from onitama.gui import theme
from onitama.gui.view_logic import INITIAL_STATUS, StatusView


RED = QColor(theme.RED)
RED_DARK = QColor(theme.RED_DARK)
RED_TEMPLE = QColor(*theme.RED_TEMPLE_RGBA)
BLUE = QColor(theme.BLUE)
BLUE_DARK = QColor(theme.BLUE_DARK)
BLUE_TEMPLE = QColor(*theme.BLUE_TEMPLE_RGBA)
BOARD_LIGHT = QColor(theme.BOARD_LIGHT)
BOARD_DARK = QColor(theme.BOARD_DARK)
BOARD_LINE = QColor(theme.BOARD_LINE)
SURFACE = QColor(theme.SURFACE)
TEXT = QColor(theme.TEXT)


class BoardWidget(QWidget):
    """Draw a horizontal Onitama board: red on the left, blue on the right."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state: GameState | None = None
        self._highlight_move: Move | None = None
        self.setMinimumSize(440, 440)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_state(self, state: GameState | None, highlight_move: Move | None = None) -> None:
        self._state = state
        self._highlight_move = highlight_move
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(560, 560)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(theme.APP_BG))

        side = min(self.width(), self.height()) - 48
        side = max(120, side)
        left = (self.width() - side) / 2
        top = (self.height() - side) / 2
        board_rect = QRectF(left, top, side, side)
        cell = side / 5

        self._draw_cells(painter, board_rect, cell)
        self._draw_highlight(painter, board_rect, cell)
        self._draw_labels(painter, board_rect, cell)
        self._draw_pieces(painter, board_rect, cell)

    def _draw_cells(self, painter: QPainter, board_rect: QRectF, cell: float) -> None:
        painter.setPen(QPen(BOARD_LINE, 2))
        for visual_row in range(5):
            for visual_col in range(5):
                x = board_rect.left() + visual_col * cell
                y = board_rect.top() + visual_row * cell
                row, col = _visual_to_canonical(visual_row, visual_col)
                fill = BOARD_LIGHT if (visual_row + visual_col) % 2 == 0 else BOARD_DARK
                if (row, col) == (4, 2):
                    fill = RED_TEMPLE
                elif (row, col) == (0, 2):
                    fill = BLUE_TEMPLE
                painter.fillRect(
                    QRectF(x, y, cell, cell),
                    fill,
                )
                painter.drawRect(QRectF(x, y, cell, cell))

    def _draw_labels(self, painter: QPainter, board_rect: QRectF, cell: float) -> None:
        painter.save()
        painter.setPen(QPen(QColor(theme.BOARD_LABEL)))
        font = QFont(self.font())
        font.setPointSize(max(9, min(16, int(cell * 0.16))))
        font.setBold(True)
        painter.setFont(font)

        for visual_col in range(5):
            rank = str(visual_col + 1)
            x = board_rect.left() + visual_col * cell
            painter.drawText(QRectF(x, board_rect.top() - 24, cell, 18), Qt.AlignmentFlag.AlignCenter, rank)
            painter.drawText(QRectF(x, board_rect.bottom() + 6, cell, 18), Qt.AlignmentFlag.AlignCenter, rank)

        for visual_row in range(5):
            file_name = chr(ord("a") + visual_row)
            y = board_rect.top() + visual_row * cell
            painter.drawText(QRectF(board_rect.left() - 24, y, 18, cell), Qt.AlignmentFlag.AlignCenter, file_name)
            painter.drawText(QRectF(board_rect.right() + 6, y, 18, cell), Qt.AlignmentFlag.AlignCenter, file_name)
        painter.restore()

    def _draw_highlight(self, painter: QPainter, board_rect: QRectF, cell: float) -> None:
        if self._highlight_move is None:
            return

        painter.save()
        painter.setPen(QPen(QColor(theme.HIGHLIGHT), 4))
        painter.setBrush(QColor(*theme.HIGHLIGHT_RGBA))
        for pos in (self._highlight_move.from_pos, self._highlight_move.to_pos):
            visual_row, visual_col = _canonical_to_visual(pos)
            painter.drawRoundedRect(
                QRectF(
                    board_rect.left() + visual_col * cell + 5,
                    board_rect.top() + visual_row * cell + 5,
                    cell - 10,
                    cell - 10,
                ),
                8,
                8,
            )
        painter.restore()

    def _draw_pieces(self, painter: QPainter, board_rect: QRectF, cell: float) -> None:
        if self._state is None:
            self._draw_empty_hint(painter, board_rect)
            return

        for visual_row in range(5):
            for visual_col in range(5):
                row, col = _visual_to_canonical(visual_row, visual_col)
                piece = self._state.board[row][col]
                if piece is None:
                    continue

                cx = board_rect.left() + visual_col * cell + cell / 2
                cy = board_rect.top() + visual_row * cell + cell / 2
                is_master = piece.kind is PieceType.MASTER
                radius = cell * (0.39 if is_master else 0.32)
                color = RED if piece.owner is Player.RED else BLUE
                outline = RED_DARK if piece.owner is Player.RED else BLUE_DARK

                painter.setPen(QPen(outline, 4 if is_master else 3))
                painter.setBrush(color)
                painter.drawEllipse(QPointF(cx, cy), radius, radius)

                if is_master:
                    painter.setPen(QPen(outline, 2))
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawEllipse(QPointF(cx, cy), radius * 0.66, radius * 0.66)

                painter.setPen(QPen(QColor(theme.WHITE)))
                font = QFont(self.font())
                font.setPointSize(max(12, int(cell * (0.2 if is_master else 0.18))))
                font.setBold(True)
                painter.setFont(font)
                label = "M" if is_master else "S"
                painter.drawText(
                    QRectF(cx - radius, cy - radius, radius * 2, radius * 2),
                    Qt.AlignmentFlag.AlignCenter,
                    label,
                )

    def _draw_empty_hint(self, painter: QPainter, board_rect: QRectF) -> None:
        painter.save()
        painter.setPen(QPen(QColor(theme.BUTTON_SECONDARY)))
        font = QFont(self.font())
        font.setPointSize(14)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(board_rect, Qt.AlignmentFlag.AlignCenter, "Empieza la partida")
        painter.restore()


class CardWidget(QFrame):
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._card: Card | None = None
        self._title = title
        self._owner: Player | None = None
        self._highlighted = False
        self.setMinimumSize(200, 150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFrameShape(QFrame.Shape.NoFrame)

    def set_card(
        self,
        card: Card | None,
        title: str | None = None,
        owner: Player | None = None,
        highlighted: bool = False,
    ) -> None:
        self._card = card
        self._owner = owner
        self._highlighted = highlighted
        if title is not None:
            self._title = title
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(240, 180)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect().adjusted(4, 4, -4, -4)
        border_color = QColor(theme.HIGHLIGHT) if self._highlighted else QColor(theme.TEXT_MUTED)
        fill_color = QColor(theme.HIGHLIGHT_SOFT) if self._highlighted else SURFACE
        border_width = 4 if self._highlighted else 2
        painter.setPen(QPen(border_color, border_width))
        painter.setBrush(fill_color)
        painter.drawRoundedRect(rect, 8, 8)
        content_y_offset = -8.0

        if self._card is None:
            painter.setPen(QPen(QColor(theme.DISABLED)))
            painter.drawText(rect.adjusted(8, 36, -8, -8), Qt.AlignmentFlag.AlignCenter, "Sin carta")
            return

        name_font = QFont(self.font())
        name_font.setPointSize(max(18, min(24, int(rect.height() * 0.13))))
        name_font.setBold(True)
        painter.setFont(name_font)
        painter.setPen(QPen(TEXT))
        name_h = max(30.0, rect.height() * 0.24)
        painter.drawText(
            QRectF(
                rect.left() + 12,
                rect.top() + rect.height() * 0.09 + content_y_offset,
                rect.width() - 24,
                name_h,
            ),
            Qt.AlignmentFlag.AlignCenter,
            self._card.name,
        )

        stamp_color = RED if self._card.stamp is Player.RED else BLUE
        painter.setPen(QPen(stamp_color, 2))
        painter.setBrush(stamp_color)
        stamp_radius = max(8.0, min(14.0, rect.height() * 0.07))
        painter.drawEllipse(
            QPointF(
                rect.left() + rect.width() * 0.10,
                rect.top() + rect.height() * 0.16 + content_y_offset + 5.0,
            ),
            stamp_radius,
            stamp_radius,
        )

        grid_side = min(rect.width() * 0.62, rect.height() * 0.58)
        self._draw_mini_grid(
            painter,
            QRectF(
                rect.center().x() - grid_side / 2,
                rect.bottom() - grid_side - rect.height() * 0.04 + content_y_offset,
                grid_side,
                grid_side,
            ),
        )

    def _draw_mini_grid(self, painter: QPainter, rect: QRectF) -> None:
        assert self._card is not None
        cell = min(rect.width(), rect.height()) / 5
        left = rect.center().x() - cell * 2.5
        top = rect.center().y() - cell * 2.5

        painter.setPen(QPen(QColor(theme.TEXT_MUTED), 1))
        painter.setBrush(QColor(theme.APP_BG))
        for r in range(5):
            for c in range(5):
                painter.drawRect(QRectF(left + c * cell, top + r * cell, cell, cell))

        painter.setPen(QPen(QColor(theme.TEXT_MUTED), 1))
        painter.setBrush(QColor(theme.CARD_CENTER))
        painter.drawRect(QRectF(left + 2 * cell, top + 2 * cell, cell, cell))

        painter.setBrush(QColor(theme.MOVE_TARGET))
        player = self._owner or Player.RED
        for dr, dc in self._card.deltas:
            r = 2 + dr
            c = 2 + dc
            if 0 <= r < 5 and 0 <= c < 5:
                painter.drawRect(
                    QRectF(
                        left + c * cell,
                        top + r * cell,
                        cell,
                        cell,
                    )
                )


class MessageBanner(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._title = QLabel()
        self._detail = QLabel()
        self._detail.setWordWrap(True)
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._detail.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title_font = QFont(self.font())
        title_font.setPointSize(18)
        title_font.setBold(True)
        self._title.setFont(title_font)

        detail_font = QFont(self.font())
        detail_font.setPointSize(13)
        self._detail.setFont(detail_font)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 12, 18, 12)
        layout.setSpacing(3)
        layout.addWidget(self._title)
        layout.addWidget(self._detail)
        self.apply(INITIAL_STATUS)

    def set_responsive_size(self, window_height: int) -> None:
        banner_height = max(78, min(124, int(window_height * 0.095)))
        self.setFixedHeight(banner_height)

        title_font = QFont(self.font())
        title_font.setPointSize(max(18, min(24, int(banner_height * 0.22))))
        title_font.setBold(True)
        self._title.setFont(title_font)

        detail_font = QFont(self.font())
        detail_font.setPointSize(max(13, min(17, int(banner_height * 0.155))))
        self._detail.setFont(detail_font)

    def apply(self, view: StatusView) -> None:
        self._title.setText(view.title)
        self._detail.setText(view.detail)
        colors = {
            "neutral": (theme.SURFACE, theme.TEXT_MUTED),
            "success": (theme.SUCCESS_SOFT_BG, theme.SUCCESS_STRONG),
            "warning": (theme.WARNING_SOFT_BG, theme.WARNING_STRONG),
            "error": (theme.ERROR_SOFT_BG, theme.ERROR_STRONG),
        }
        bg, border = colors.get(view.tone, colors["neutral"])
        self.setStyleSheet(
            f"""
            MessageBanner {{
                background: {bg};
                border: 2px solid {border};
                border-radius: 10px;
            }}
            QLabel {{
                color: {theme.TEXT};
            }}
            """
        )


def _visual_to_canonical(visual_row: int, visual_col: int) -> tuple[int, int]:
    return 4 - visual_col, visual_row


def _canonical_to_visual(pos: tuple[int, int]) -> tuple[int, int]:
    row, col = pos
    return col, 4 - row
