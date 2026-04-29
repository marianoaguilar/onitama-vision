from __future__ import annotations


class VisionObservationError(ValueError):
    """The current visual observation is invalid but the runtime can continue."""


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
