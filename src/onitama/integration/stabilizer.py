from __future__ import annotations

from dataclasses import dataclass, field

from onitama.app.errors import VisionConfigurationError
from onitama.engine.state import GameState


@dataclass
class StateStabilizer:
    """
    Confirm a state only after it has been observed enough consecutive times.

    This is a temporal filter only. It does not know anything about Onitama
    legality; it only tracks repeated equal observations.
    """

    required_repeats: int = 3
    candidate_state: GameState | None = field(init=False, default=None)
    repeat_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.required_repeats < 1:
            raise VisionConfigurationError("required_repeats must be >= 1")

    def push(self, observed_state: GameState) -> GameState | None:
        """Consume one observation and return the stable state when the threshold is reached."""
        if self.candidate_state is None or self.candidate_state != observed_state:
            self.candidate_state = observed_state
            self.repeat_count = 1
        else:
            self.repeat_count += 1

        if self.repeat_count >= self.required_repeats:
            return self.candidate_state

        return None

    def reset(self) -> None:
        """Forget the current candidate and repetition counter."""
        self.candidate_state = None
        self.repeat_count = 0
