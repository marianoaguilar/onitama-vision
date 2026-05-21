from __future__ import annotations

from enum import Enum


class VisionObservationKind(str, Enum):
    """Stable categories for recoverable vision observation failures."""

    INVALID_BOARD_PIECE_COUNT = "invalid_board_piece_count"
    LOW_CONFIDENCE_CARD = "low_confidence_card"
    GENERIC = "generic"


class VisionObservationError(ValueError):
    """The current visual observation is invalid but the runtime can continue."""

    def __init__(
        self,
        kind: VisionObservationKind,
        *,
        debug_message: str | None = None,
    ) -> None:
        self.kind = kind
        self.debug_message = debug_message or kind.value
        super().__init__(self.debug_message)


class VisionFatalError(RuntimeError):
    """Base class for fatal vision/runtime errors that should surface to the GUI."""


class VisionConfigurationError(VisionFatalError):
    """Static configuration or resource loading failed."""


class VisionDependencyError(VisionFatalError):
    """A required runtime dependency is missing or unusable."""


class VisionCameraError(VisionFatalError):
    """Camera access failed."""


class VisionPipelineError(VisionFatalError):
    """The vision pipeline produced an invalid internal result."""


class VisionInternalError(VisionFatalError):
    """The runtime reached an impossible internal state."""
