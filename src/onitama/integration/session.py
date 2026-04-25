from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from onitama.ai.controllers import AIController
from onitama.engine.pieces import Player
from onitama.engine.rules import Action, apply_action, winner
from onitama.engine.state import GameState
from onitama.integration.stabilizer import StateStabilizer
from onitama.integration.synchronizer import SyncStatus, match_observed_state


class SessionPhase(str, Enum):
    BOOTSTRAP = "BOOTSTRAP"
    WAITING_HUMAN_MOVE = "WAITING HUMAN MOVE"
    READY_FOR_AI = "READY FOR AI"
    WAITING_AI_EXECUTION = "WAITING AI EXECUTION"
    FINISHED = "FINISHED"


class SessionOutcome(str, Enum):
    COLLECTING = "collecting"
    BOOTSTRAPPED = "bootstrapped"
    UNCHANGED_OBSERVATION = "unchanged observation"
    HUMAN_MOVE_REJECTED = "human move rejected"
    HUMAN_MOVE_ACCEPTED = "human move accepted"
    AI_ACTION_SELECTED = "ai action selected"
    AWAITING_AI_EXECUTION = "awaiting ai execution"
    AI_EXECUTION_MISMATCH = "ai execution mismatch"
    AI_EXECUTION_CONFIRMED = "ai execution confirmed"

@dataclass
class VisionGameSession:
    """
    Coordinate the high-level flow of a vision-assisted game session.

    This session works on already-observed `GameState` values. It does not talk
    to the camera or the vision pipeline directly yet.
    """

    human_player: Player
    ai_player: Player
    ai_controller: AIController
    stabilizer: StateStabilizer = field(default_factory=StateStabilizer)
    current_state: GameState | None = field(init=False, default=None)
    phase: SessionPhase = field(init=False, default=SessionPhase.BOOTSTRAP)
    expected_state: GameState | None = field(init=False, default=None)
    last_ai_action: Action | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.human_player is self.ai_player:
            raise ValueError("human_player and ai_player must be different players.")
        if self.ai_controller is None:
            raise ValueError("ai_controller cannot be None.")


    def process_observation(self, observed_state: GameState) -> SessionOutcome:
        """Consume one observed state and advance the session when possible."""
        if self.phase is SessionPhase.FINISHED:
            raise ValueError("Cannot process observations after the session has finished.")

        if self.phase is SessionPhase.READY_FOR_AI:
            raise ValueError("Cannot process observations while the session is ready for the AI turn.")

        stable_state = self.stabilizer.push(observed_state)
        if stable_state is None:
            return SessionOutcome.COLLECTING

        if self.phase is SessionPhase.BOOTSTRAP:
            return self._bootstrap_from_stable_state(stable_state)

        if self.phase is SessionPhase.WAITING_HUMAN_MOVE:
            return self._process_stable_human_observation(stable_state)

        if self.phase is SessionPhase.WAITING_AI_EXECUTION:
            return self._process_stable_ai_execution(stable_state)

        raise ValueError(f"Unknown session phase: {self.phase!r}")


    def run_ai_turn(self) -> SessionOutcome:
        """Select the AI action and start waiting for the expected physical state."""
        if self.phase is SessionPhase.FINISHED:
            raise ValueError("Cannot run an AI turn after the session has finished.")

        if self.phase is not SessionPhase.READY_FOR_AI:
            raise ValueError("Cannot run an AI turn unless the session is READY_FOR_AI.")

        assert self.current_state is not None, "READY_FOR_AI requires a confirmed current_state."

        ai_action = self.ai_controller.select_action(self.current_state)
        expected_state = apply_action(self.current_state, ai_action)

        self.last_ai_action = ai_action
        self.expected_state = expected_state
        self.phase = SessionPhase.WAITING_AI_EXECUTION

        return SessionOutcome.AI_ACTION_SELECTED


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
    ) -> SessionOutcome:
        """Adopt the first stable observation as the confirmed current state."""
        self.current_state = stable_state
        self.expected_state = None
        self.last_ai_action = None
        self.stabilizer.reset()
        self.phase = self._phase_for_confirmed_state(stable_state)

        return SessionOutcome.BOOTSTRAPPED


    def _process_stable_human_observation(
        self,
        stable_state: GameState,
    ) -> SessionOutcome:
        """Validate one stable human observation against the confirmed state."""
        assert self.current_state is not None, "WAITING_HUMAN_MOVE requires a confirmed current_state."

        sync_status = match_observed_state(self.current_state, stable_state)

        if sync_status is SyncStatus.UNCHANGED:
            return SessionOutcome.UNCHANGED_OBSERVATION

        if sync_status is SyncStatus.REJECTED:
            return SessionOutcome.HUMAN_MOVE_REJECTED

        self.current_state = stable_state
        self.expected_state = None
        self.last_ai_action = None
        self.stabilizer.reset()
        self.phase = self._phase_for_confirmed_state(stable_state)

        if self.phase not in {SessionPhase.FINISHED, SessionPhase.READY_FOR_AI}:
            raise ValueError("Invalid session state: a confirmed human move must hand control to the AI or finish the game.")

        return SessionOutcome.HUMAN_MOVE_ACCEPTED


    def _process_stable_ai_execution(
        self,
        stable_state: GameState,
    ) -> SessionOutcome:
        """Check whether the physical board now matches the expected AI state."""
        assert self.current_state is not None, "WAITING_AI_EXECUTION requires a confirmed current_state."
        assert self.expected_state is not None, "WAITING_AI_EXECUTION requires an expected_state."
        assert self.last_ai_action is not None, "WAITING_AI_EXECUTION requires a last_ai_action."

        if stable_state == self.expected_state:
            self.current_state = stable_state
            self.expected_state = None
            self.last_ai_action = None
            self.stabilizer.reset()
            self.phase = self._phase_for_confirmed_state(stable_state)

            if self.phase is not SessionPhase.FINISHED and self.phase is not SessionPhase.WAITING_HUMAN_MOVE:
                raise ValueError("Invalid session state: a confirmed AI move must hand control to the human player or finish the game.")

            return SessionOutcome.AI_EXECUTION_CONFIRMED

        if stable_state == self.current_state:
            return SessionOutcome.AWAITING_AI_EXECUTION

        return SessionOutcome.AI_EXECUTION_MISMATCH
