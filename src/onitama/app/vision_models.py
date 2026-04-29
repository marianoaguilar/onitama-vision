from __future__ import annotations

from dataclasses import dataclass

from onitama.app.errors import VisionConfigurationError
from onitama.engine.pieces import Player
from onitama.engine.rules import Action
from onitama.engine.state import GameState
from onitama.integration.session import SessionOutcome, SessionPhase


@dataclass(frozen=True)
class VisionRuntimeConfig:
    """
    Configuration shared by any frontend that runs the vision-assisted game.

    The runtime owns the game loop, camera access and session orchestration.
    """

    human_player: Player
    required_repeats: int
    ai_depth: int
    ai_evaluator: str
    camera_device: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: int = 30

    @property
    def ai_player(self) -> Player:
        return self.human_player.opponent()

    def __post_init__(self) -> None:
        if self.required_repeats < 1:
            raise VisionConfigurationError("required_repeats must be >= 1.")
        if self.ai_depth < 1:
            raise VisionConfigurationError("ai_depth must be >= 1.")
        if self.camera_width < 1:
            raise VisionConfigurationError("camera_width must be >= 1.")
        if self.camera_height < 1:
            raise VisionConfigurationError("camera_height must be >= 1.")
        if self.camera_fps < 1:
            raise VisionConfigurationError("camera_fps must be >= 1.")
        if not self.ai_evaluator.strip():
            raise VisionConfigurationError("ai_evaluator cannot be empty.")


@dataclass(frozen=True)
class VisionRuntimeState:
    """
    Frontend-facing state snapshot produced by the runtime.
    """

    running: bool
    phase: SessionPhase
    last_outcome: SessionOutcome | None
    current_state: GameState | None
    expected_state: GameState | None
    ai_action: Action | None
    error_message: str | None
    winner_player: Player | None
    winner_reason: str | None
