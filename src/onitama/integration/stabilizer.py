from __future__ import annotations

from dataclasses import dataclass, field

from onitama.engine.state import GameState


@dataclass(frozen=True)
class StabilizerResult:
    """Current stabilization status after consuming one observation."""

    stable: bool
    state: GameState
    repeat_count: int
    required_count: int


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
            raise ValueError("required_repeats must be >= 1")

    def push(self, observed_state: GameState) -> StabilizerResult:
        """Consume one observation and report whether it is already stable."""
        if self.candidate_state is None or self.candidate_state != observed_state:
            self.candidate_state = observed_state
            self.repeat_count = 1
        else:
            self.repeat_count += 1

        stable = self.repeat_count >= self.required_repeats
        return StabilizerResult(
            stable=stable,
            state=self.candidate_state,
            repeat_count=self.repeat_count,
            required_count=self.required_repeats,
        )

    def reset(self) -> None:
        """Forget the current candidate and repetition counter."""
        self.candidate_state = None
        self.repeat_count = 0
