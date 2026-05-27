from __future__ import annotations

# Core surface colors.
APP_BG = "#f6f0e4"
SURFACE = "#fffaf0"
SURFACE_ALT = "#f2ead8"
PAPER = "#fff7df"
TEXT = "#1f2933"
TEXT_INVERTED = "#fffaf0"
TEXT_MUTED = "#8a6b3f"
TEXT_SUBTLE = "#5f4c2d"
TRANSPARENT = "transparent"
WHITE = "white"

# Brand and player colors.
RED = "#c2413b"
RED_ACTION = "#c72920"
RED_ACTION_HOVER = "#a7251d"
RED_DARK = "#7f1d1d"
RED_DEEP = "#8b1f18"
RED_BORDER = "#d32920"
BLUE = "#2563eb"
BLUE_ACTION = "#2b5d8a"
BLUE_DARK = "#1e3a8a"
BLUE_DEEP = "#17385e"
BLUE_SOFT = "#eef5fb"

# Lines, borders, and neutral UI colors.
BORDER = "#8a6b3f"
LINE = "#b08a4d"
BOARD_LINE = "#6f4e2c"
BOARD_LABEL = "#4b3621"
BUTTON_DARK = "#1f2933"
BUTTON_SECONDARY = "#6b7280"
DISABLED = "#9ca3af"
EMPTY_TEXT = "#6b7280"
CARD_CENTER = "#374151"
MOVE_TARGET = "#f8d36a"
HIGHLIGHT = "#f59e0b"
CAMERA_BG = "#111827"

# Board colors.
BOARD_LIGHT = "#f2dfb3"
BOARD_DARK = "#d6b479"
RED_TEMPLE_RGBA = (194, 65, 59, 55)
BLUE_TEMPLE_RGBA = (37, 99, 235, 55)
HIGHLIGHT_RGBA = (245, 158, 11, 55)
SHADOW_RGBA = (75, 54, 33, 70)

# Status colors.
SUCCESS_TEXT = "#24513a"
SUCCESS_BG = "#dcefe2"
SUCCESS_BORDER = "#9cc8ad"
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
BOARD_LINE_BGR = (88, 185, 240)
BOARD_POINT_BGR = (74, 86, 224)
CARD_MARGIN_BGR = (185, 200, 210)
OVERLAY_WHITE_BGR = (246, 244, 238)
OVERLAY_DARK_BGR = (58, 62, 70)
CARD_ACTIVE_COLOR_BGR: dict[str, tuple[int, int, int]] = {
    "red_0": (78, 88, 218),
    "red_1": (78, 88, 218),
    "side": (74, 174, 224),
    "blue_0": (190, 122, 50),
    "blue_1": (190, 122, 50),
}
CARD_INACTIVE_BGR = (164, 166, 170)
