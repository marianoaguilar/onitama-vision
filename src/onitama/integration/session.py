from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from onitama.ai.controllers import AIController
from onitama.engine.pieces import Player
from onitama.engine.rules import Action, apply_action, winner
from onitama.engine.state import GameState
from onitama.integration.stabilizer import StateStabilizer
from onitama.integration.synchronizer import SyncResult, match_observed_state


class SessionPhase(str, Enum):
    BOOTSTRAP = "BOOTSTRAP"
    WAITING_HUMAN_MOVE = "WAITING_HUMAN_MOVE"
    READY_FOR_AI = "READY_FOR_AI"
    WAITING_AI_EXECUTION = "WAITING_AI_EXECUTION"
    FINISHED = "FINISHED"


class SessionOutcome(str, Enum):
    FINISHED_ALREADY = "finished_already"
    AWAITING_AI = "awaiting_ai"
    COLLECTING = "collecting"
    AI_NOT_READY = "ai_not_ready"
    AI_UNAVAILABLE = "ai_unavailable"
    BOOTSTRAPPED = "bootstrapped"
    UNCHANGED_OBSERVATION = "unchanged_observation"
    HUMAN_MOVE_REJECTED = "human_move_rejected"
    HUMAN_MOVE_ACCEPTED = "human_move_accepted"
    AI_ACTION_SELECTED = "ai_action_selected"
    AWAITING_AI_EXECUTION = "awaiting_ai_execution"
    AI_EXECUTION_MISMATCH = "ai_execution_mismatch"
    AI_EXECUTION_CONFIRMED = "ai_execution_confirmed"


@dataclass(frozen=True)
class SessionStepResult:
    """Outcome of processing one step inside the session."""

    phase: SessionPhase
    outcome: SessionOutcome
    current_state: GameState | None
    expected_state: GameState | None = None
    ai_action: Action | None = None
    sync_result: SyncResult | None = None
    message: str | None = None


@dataclass
class VisionGameSession:
    """
    Coordinate the high-level flow of a vision-assisted game session.

    This session works on already-observed `GameState` values. It does not talk
    to the camera or the vision pipeline directly yet.
    """

    human_player: Player
    ai_player: Player
    ai_controller: AIController | None = None
    stabilizer: StateStabilizer = field(default_factory=StateStabilizer)
    current_state: GameState | None = field(init=False, default=None)
    phase: SessionPhase = field(init=False, default=SessionPhase.BOOTSTRAP)
    expected_state: GameState | None = field(init=False, default=None)
    last_ai_action: Action | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.human_player is self.ai_player:
            raise ValueError("human_player and ai_player must be different players.")


    def process_observation(self, observed_state: GameState) -> SessionStepResult:
        """Consume one observed state and advance the session when possible."""
        if self.phase is SessionPhase.FINISHED:
            return SessionStepResult(
                phase=self.phase,
                outcome=SessionOutcome.FINISHED_ALREADY,
                current_state=self.current_state,
                expected_state=self.expected_state,
                ai_action=self.last_ai_action,
                message="Game already finished.",
            )

        if self.phase is SessionPhase.READY_FOR_AI:
            return SessionStepResult(
                phase=self.phase,
                outcome=SessionOutcome.AWAITING_AI,
                current_state=self.current_state,
                expected_state=self.expected_state,
                ai_action=self.last_ai_action,
                message="It is the AI turn; call run_ai_turn() before processing more observations.",
            )

        stable_state = self.stabilizer.push(observed_state)
        if stable_state is None:
            return SessionStepResult(
                phase=self.phase,
                outcome=SessionOutcome.COLLECTING,
                current_state=self.current_state,
                expected_state=self.expected_state,
                ai_action=self.last_ai_action,
                message="Observation not stable yet.",
            )

        if self.phase is SessionPhase.BOOTSTRAP:
            return self._bootstrap_from_stable_state(stable_state)

        if self.phase is SessionPhase.WAITING_HUMAN_MOVE:
            return self._process_stable_human_observation(stable_state)

        if self.phase is SessionPhase.WAITING_AI_EXECUTION:
            return self._process_stable_ai_execution(stable_state)

        raise ValueError(f"Unknown session phase: {self.phase!r}")


    def run_ai_turn(self) -> SessionStepResult:
        """Select the AI action and start waiting for the expected physical state."""
        if self.phase is SessionPhase.FINISHED:
            return SessionStepResult(
                phase=self.phase,
                outcome=SessionOutcome.FINISHED_ALREADY,
                current_state=self.current_state,
                expected_state=self.expected_state,
                ai_action=self.last_ai_action,
                message="Game already finished.",
            )

        if self.phase is not SessionPhase.READY_FOR_AI:
            return SessionStepResult(
                phase=self.phase,
                outcome=SessionOutcome.AI_NOT_READY,
                current_state=self.current_state,
                expected_state=self.expected_state,
                ai_action=self.last_ai_action,
                message="AI turn requested while session is not ready for it.",
            )

        if self.ai_controller is None:
            return SessionStepResult(
                phase=self.phase,
                outcome=SessionOutcome.AI_UNAVAILABLE,
                current_state=self.current_state,
                message="No AI controller has been configured.",
            )

        assert self.current_state is not None, "READY_FOR_AI requires a confirmed current_state."

        ai_action = self.ai_controller.select_action(self.current_state)
        expected_state = apply_action(self.current_state, ai_action)

        self.last_ai_action = ai_action
        self.expected_state = expected_state
        self.phase = SessionPhase.WAITING_AI_EXECUTION

        return SessionStepResult(
            phase=self.phase,
            outcome=SessionOutcome.AI_ACTION_SELECTED,
            current_state=self.current_state,
            expected_state=self.expected_state,
            ai_action=self.last_ai_action,
            message="AI action selected; waiting for the physical board to match the expected state.",
        )


    def _phase_for_confirmed_state(self, state: GameState) -> SessionPhase:
        """Decide the next session phase from a confirmed game state."""
        if winner(state) is not None:
            return SessionPhase.FINISHED
        if state.to_move is self.human_player:
            return SessionPhase.WAITING_HUMAN_MOVE
        if state.to_move is self.ai_player:
            return SessionPhase.READY_FOR_AI
        raise ValueError(f"Unexpected player to move: {state.to_move!r}")


    def _bootstrap_from_stable_state(
        self,
        stable_state: GameState,
    ) -> SessionStepResult:
        """Adopt the first stable observation as the confirmed current state."""
        self.current_state = stable_state
        self.expected_state = None
        self.last_ai_action = None
        self.phase = self._phase_for_confirmed_state(stable_state)

        if self.phase is SessionPhase.FINISHED:
            message = "Stable initial observation adopted and the game is already terminal."
        elif self.phase is SessionPhase.READY_FOR_AI:
            message = "Stable initial observation adopted; AI has the first turn."
        else:
            message = "Stable initial observation adopted; human has the first turn."

        return SessionStepResult(
            phase=self.phase,
            outcome=SessionOutcome.BOOTSTRAPPED,
            current_state=self.current_state,
            message=message,
        )


    def _process_stable_human_observation(
        self,
        stable_state: GameState,
    ) -> SessionStepResult:
        """Validate one stable human observation against the confirmed state."""
        assert self.current_state is not None, "WAITING_HUMAN_MOVE requires a confirmed current_state."

        if stable_state == self.current_state:
            return SessionStepResult(
                phase=self.phase,
                outcome=SessionOutcome.UNCHANGED_OBSERVATION,
                current_state=self.current_state,
                message="Stable observation matches the current confirmed state.",
            )

        sync_result = match_observed_state(self.current_state, stable_state)

        if not sync_result.accepted:
            return SessionStepResult(
                phase=self.phase,
                outcome=SessionOutcome.HUMAN_MOVE_REJECTED,
                current_state=self.current_state,
                sync_result=sync_result,
                message="Stable observation was rejected by the legal-state synchronizer.",
            )

        self.current_state = stable_state
        self.expected_state = None
        self.last_ai_action = None
        self.phase = self._phase_for_confirmed_state(stable_state)

        if self.phase is SessionPhase.FINISHED:
            message = "Stable human move accepted and the game is now terminal."
        elif self.phase is SessionPhase.READY_FOR_AI:
            message = "Stable human move accepted; session is ready for the AI step."
        else:
            raise ValueError("Invalid session state: a confirmed human move must hand control to the AI or finish the game.")

        return SessionStepResult(
            phase=self.phase,
            outcome=SessionOutcome.HUMAN_MOVE_ACCEPTED,
            current_state=self.current_state,
            sync_result=sync_result,
            message=message,
        )


    def _process_stable_ai_execution(
        self,
        stable_state: GameState,
    ) -> SessionStepResult:
        """Check whether the physical board now matches the expected AI state."""
        assert self.current_state is not None, "WAITING_AI_EXECUTION requires a confirmed current_state."
        assert self.expected_state is not None, "WAITING_AI_EXECUTION requires an expected_state."
        assert self.last_ai_action is not None, "WAITING_AI_EXECUTION requires a last_ai_action."

        if stable_state == self.expected_state:
            self.current_state = stable_state
            self.expected_state = None
            ai_action = self.last_ai_action
            self.last_ai_action = None
            self.phase = self._phase_for_confirmed_state(stable_state)

            if self.phase is SessionPhase.FINISHED:
                message = "AI move confirmed on the physical board and the game is now terminal."
            elif self.phase is SessionPhase.WAITING_HUMAN_MOVE:
                message = "AI move confirmed on the physical board; waiting for the human turn."
            else:
                raise ValueError("Invalid session state: a confirmed AI move must hand control to the human player or finish the game.")

            return SessionStepResult(
                phase=self.phase,
                outcome=SessionOutcome.AI_EXECUTION_CONFIRMED,
                current_state=self.current_state,
                expected_state=None,
                ai_action=ai_action,
                message=message,
            )

        if stable_state == self.current_state:
            return SessionStepResult(
                phase=self.phase,
                outcome=SessionOutcome.AWAITING_AI_EXECUTION,
                current_state=self.current_state,
                expected_state=self.expected_state,
                ai_action=self.last_ai_action,
                message="Stable observation still matches the pre-AI state; waiting for the board to change.",
            )

        return SessionStepResult(
            phase=self.phase,
            outcome=SessionOutcome.AI_EXECUTION_MISMATCH,
            current_state=self.current_state,
            expected_state=self.expected_state,
            ai_action=self.last_ai_action,
            message="Stable observation does not match the expected AI result.",
        )
