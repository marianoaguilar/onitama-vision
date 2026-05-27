from __future__ import annotations

# Core surface colors.
APP_BG = "#f6f0e4"
SURFACE = "#fffaf0"
SURFACE_ALT = "#f2ead8"
TEXT = "#1f2933"
TEXT_MUTED = "#7f6642"
TEXT_SUBTLE = "#5f4a2f"
TRANSPARENT = "transparent"
WHITE = "white"

# Brand and player colors.
RED = "#c34a3e"
RED_ACTION_HOVER = "#ad3a31"
RED_DARK = "#8f2b23"
RED_SOFT = "#f7dbd6"
BLUE = "#2f7bb8"
BLUE_DARK = "#245789"
BLUE_SOFT = "#d3ebfa"

# Lines, borders, and neutral UI colors.
LINE = "#aa8350"
BOARD_LINE = "#6a4a2c"
BOARD_LABEL = "#4a321f"
BUTTON_SECONDARY = "#6b7280"
DISABLED = "#9ca3af"
CARD_CENTER = "#374151"
MOVE_TARGET = "#d8b46a"
HIGHLIGHT = "#2f8f5a"
HIGHLIGHT_SOFT = "#d8efe1"
CAMERA_BG = "#111827"

# Board colors.
BOARD_LIGHT = "#f6e7c5"
BOARD_DARK = "#dfc38a"
RED_TEMPLE_RGBA = (221, 102, 83, 82)
BLUE_TEMPLE_RGBA = (76, 149, 207, 82)
HIGHLIGHT_RGBA = (122, 201, 157, 135)
SHADOW_RGBA = (75, 54, 33, 70)

# Status colors.
SUCCESS_TEXT = "#1f6a43"
SUCCESS_BG = "#cfe8d7"
SUCCESS_BORDER = "#78b08d"
SUCCESS_SOFT_BG = "#ecfdf5"
SUCCESS_STRONG = "#059669"
INFO_BG = "#dbeafe"
INFO_BORDER = "#93c5fd"
WARNING_TEXT = "#8a5a10"
WARNING_BG = "#fff4d6"
WARNING_BORDER = "#e6c77a"
WARNING_SOFT_BG = "#fffbeb"
WARNING_STRONG = "#d97706"
ERROR_BG = "#fee2e2"
ERROR_BORDER = "#fca5a5"
ERROR_SOFT_BG = "#fef2f2"
ERROR_STRONG = "#dc2626"

# Calibration overlay colors use OpenCV BGR tuples.
BOARD_LINE_BGR = (11, 158, 245)
BOARD_POINT_BGR = (49, 58, 173)
CARD_MARGIN_BGR = (185, 200, 210)
OVERLAY_WHITE_BGR = (240, 250, 255)
OVERLAY_DARK_BGR = (81, 65, 55)
CARD_ACTIVE_COLOR_BGR: dict[str, tuple[int, int, int]] = {
    "red_0": (49, 58, 173),
    "red_1": (49, 58, 173),
    "side": (11, 158, 245),
    "blue_0": (184, 123, 47),
    "blue_1": (184, 123, 47),
}
CARD_INACTIVE_BGR = (175, 163, 156)
