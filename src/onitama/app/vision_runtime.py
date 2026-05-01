from __future__ import annotations

from typing import TYPE_CHECKING

from onitama.ai.controllers import AIController
from onitama.app.errors import (
    VisionCameraError,
    VisionFatalError,
    VisionInternalError,
    VisionObservationError,
    VisionObservationKind,
    VisionPipelineError,
)
from onitama.app.vision_models import VisionRuntimeConfig, VisionRuntimeState
from onitama.engine.cards import CARD_BY_NAME
from onitama.engine.pieces import Player
from onitama.engine.rules import winner
from onitama.engine.state import GameState
from onitama.integration.session import (
    SessionOutcome,
    SessionPhase,
    VisionGameSession,
)
from onitama.integration.stabilizer import StateStabilizer
from onitama.vision.snapshot import VisionSnapshot
from onitama.vision.vision_pipeline import VisionPipeline

if TYPE_CHECKING:
    import numpy as np

class VisionGameRuntime:
    """
    Frontend-agnostic runtime for the vision-assisted game mode.

    This class owns camera access, vision inference, session orchestration and
    the translation into a raw runtime state consumable by different frontends.
    """

    _BOOTSTRAP_OBSERVATION_WARNING_THRESHOLD = 2
    _IN_GAME_OBSERVATION_WARNING_THRESHOLD = 10
    _RESET_FRAME_DISCARD_COUNT = 4

    def __init__(
        self,
        config: VisionRuntimeConfig,
        *,
        pipeline: VisionPipeline | None = None,
    ) -> None:
        
        self.config = config
        self.pipeline = pipeline if pipeline is not None else VisionPipeline()
        self.session: VisionGameSession = self._build_session(config)

        # Keep the latest raw runtime error.
        self._tracked_error_message: str | None = None

        # Keep the latest recoverable observation warning.
        self._observation_warning_kind: VisionObservationKind | None = None
        self._pending_observation_warning_kind: VisionObservationKind | None = None
        self._pending_observation_warning_count = 0
        
        # Keep the latest emitted session outcome.
        self._last_outcome: SessionOutcome | None = None

        # Runtime state.
        self.running = False
        self._camera = None
        self._latest_frame: np.ndarray | None = None
        self._pending_reset_frame_discards = 0

    def start(self) -> None:
        """Open the camera and start the runtime."""
        if self.running:
            return

        self._camera = self._open_camera(
            device=self.config.camera_device,
            width=self.config.camera_width,
            height=self.config.camera_height,
            fps=self.config.camera_fps,
        )
        self.running = True

    def stop(self) -> None:
        """Stop the runtime and release the camera."""
        if self._camera is not None:
            self._camera.release()
            self._camera = None

        self.running = False
        self._latest_frame = None

    def reset(self) -> None:
        """Reset the game session while keeping the configured pipeline and camera."""
        self.session = self._build_session(self.config)
        self._clear_error()
        self._clear_observation_warning()
        self._last_outcome = None
        self._pending_reset_frame_discards = self._RESET_FRAME_DISCARD_COUNT

    def step(self) -> VisionRuntimeState:
        """
        Advance the runtime by one logical step.

        - If the AI is ready, select its action immediately.
        - Otherwise read one frame, run vision and feed the session.
        """
        if not self.running:
            return self._build_state()

        # Once the game is finished, keep exposing the last known state without
        # reading more frames or calling the session again.
        if self.session.phase is SessionPhase.FINISHED:
            return self._build_state()

        # When the session says the AI should move, no camera frame is needed.
        if self.session.phase is SessionPhase.READY_FOR_AI:
            self._last_outcome = self.session.run_ai_turn()
            return self._build_state()

        # Otherwise the runtime advances from one observed camera frame.
        if self._camera is None:
            raise VisionInternalError("Runtime is running without an initialized camera.")

        ok, frame = self._camera.read()
        if not ok:
            self._record_error(VisionCameraError("Could not read a frame from the camera."))
            return self._build_state()

        self._latest_frame = frame

        # After a reset, drop a short burst of buffered frames before trusting
        # new observations again.
        if self._pending_reset_frame_discards > 0:
            self._pending_reset_frame_discards -= 1
            return self._build_state()

        try:
            # 1) Rebuild the visual snapshot.
            # 2) Infer the correct player-to-move for the current session phase.
            snapshot = self.pipeline.snapshot_from_frame(frame)
            observed_state = self._state_from_snapshot_for_session(snapshot)
        except VisionObservationError as exc:
            self._record_observation_warning(exc)
            return self._build_state()
        except VisionFatalError as exc:
            self._clear_observation_warning()
            self._record_error(exc)
            return self._build_state()

        self._clear_error()
        self._clear_observation_warning()

        # Let the session decide whether the observation is stable, legal and actionable.
        self._last_outcome = self.session.process_observation(observed_state)
        return self._build_state()

    def get_state(self) -> VisionRuntimeState:
        """Return the current runtime state snapshot without advancing the loop."""
        return self._build_state()

    def get_latest_frame(self) -> np.ndarray | None:
        """Return the most recent camera frame, if available."""
        return self._latest_frame

    def _build_session(self, config: VisionRuntimeConfig) -> VisionGameSession:
        ai_controller = AIController(depth=config.ai_depth, evaluator_name=config.ai_evaluator)
        return VisionGameSession(
            human_player=config.human_player,
            ai_player=config.ai_player,
            ai_controller=ai_controller,
            stabilizer=StateStabilizer(required_repeats=config.required_repeats),
        )

    def _state_from_snapshot_for_session(self, snapshot: VisionSnapshot) -> GameState:
        """
        Convert a snapshot into the most plausible GameState for the current phase.
        """
        # During bootstrap we do not have a confirmed state yet, so the side card
        # is the only reliable source for who should move first.
        if self.session.current_state is None or self.session.phase is SessionPhase.BOOTSTRAP:
            return snapshot.to_game_state(self._initial_player_from_snapshot(snapshot))

        # First interpret the snapshot assuming the turn did not change yet.
        current_candidate = snapshot.to_game_state(self.session.current_state.to_move)

        if self.session.phase is SessionPhase.WAITING_HUMAN_MOVE:
            if current_candidate == self.session.current_state:
                return current_candidate
            # If the board changed, the human move most likely consumed the turn.
            return snapshot.to_game_state(self.session.current_state.to_move.opponent())

        if self.session.phase is SessionPhase.WAITING_AI_EXECUTION:
            if current_candidate == self.session.current_state:
                return current_candidate

            if self.session.expected_state is None:
                raise VisionInternalError("WAITING_AI_EXECUTION requires an expected_state.")

            # After the AI move, interpret the snapshot with the expected next player.
            return snapshot.to_game_state(self.session.expected_state.to_move)

        return current_candidate

    def _initial_player_from_snapshot(self, snapshot: VisionSnapshot) -> Player:
        """Infer the initial player to move from the observed side card."""
        side_card = CARD_BY_NAME.get(snapshot.side_card)
        if side_card is None:
            raise VisionPipelineError(f"Unknown side card from vision pipeline: {snapshot.side_card!r}")
        return side_card.stamp

    def _record_error(self, exc: Exception) -> None:
        self._tracked_error_message = f"{type(exc).__name__}: {exc}"

    def _clear_error(self) -> None:
        self._tracked_error_message = None

    def _record_observation_warning(self, exc: VisionObservationError) -> None:
        if exc.kind is not self._pending_observation_warning_kind:
            self._pending_observation_warning_kind = exc.kind
            self._pending_observation_warning_count = 1
            self._observation_warning_kind = None
            return

        self._pending_observation_warning_count += 1
        if self._pending_observation_warning_count >= self._observation_warning_threshold():
            self._observation_warning_kind = exc.kind

    def _clear_observation_warning(self) -> None:
        self._observation_warning_kind = None
        self._pending_observation_warning_kind = None
        self._pending_observation_warning_count = 0

    def _observation_warning_threshold(self) -> int:
        if self.session.phase is SessionPhase.BOOTSTRAP:
            return self._BOOTSTRAP_OBSERVATION_WARNING_THRESHOLD
        return self._IN_GAME_OBSERVATION_WARNING_THRESHOLD

    def _build_state(self) -> VisionRuntimeState:
        current_state = self.session.current_state
        winner_info = winner(current_state) if current_state is not None else None
        winner_player, winner_reason = winner_info if winner_info is not None else (None, None)

        return VisionRuntimeState(
            running=self.running,
            phase=self.session.phase,
            last_outcome=self._last_outcome,
            current_state=current_state,
            expected_state=self.session.expected_state,
            ai_action=self.session.last_ai_action,
            error_message=self._tracked_error_message,
            observation_kind=self._observation_warning_kind,
            winner_player=winner_player,
            winner_reason=winner_reason,
        )

    @staticmethod
    def _open_camera(device: int = 0, width: int = 1280, height: int = 720, fps: int = 30):
        """Open the camera using the same settings as the working live debug tools."""
        import cv2

        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise VisionCameraError(f"Could not open camera index {device}.")

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        return cap
