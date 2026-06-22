import numpy as np
import pytest

from onitama.ai.controllers import AIController
from onitama.errors import VisionObservationError, VisionObservationKind
from onitama.runtime.vision_models import VisionRuntimeConfig
from onitama.runtime.vision_runtime import VisionGameRuntime
from onitama.engine.moves import Move
from onitama.engine.pieces import Piece, PieceType, Player
from onitama.engine.rules import apply_action, generate_legal_actions
from onitama.engine.state import GameState
from onitama.gui.view_logic import build_status_view
from onitama.session.vision_session import SessionOutcome, SessionPhase, VisionGameSession
from onitama.session.stabilizer import StateStabilizer
from onitama.vision.snapshot import VisionSnapshot


def _snapshot_from_state(state: GameState) -> VisionSnapshot:
    board_tokens: list[list[str | None]] = []
    for row in state.board:
        out_row: list[str | None] = []
        for piece in row:
            if piece is None:
                out_row.append(None)
            elif piece.owner is Player.RED and piece.kind is PieceType.MASTER:
                out_row.append("red_master")
            elif piece.owner is Player.RED:
                out_row.append("red_student")
            elif piece.owner is Player.BLUE and piece.kind is PieceType.MASTER:
                out_row.append("blue_master")
            else:
                out_row.append("blue_student")
        board_tokens.append(out_row)

    return VisionSnapshot.from_dict(
        {
            "board": board_tokens,
            "red_cards": [state.red_cards[0].name, state.red_cards[1].name],
            "blue_cards": [state.blue_cards[0].name, state.blue_cards[1].name],
            "side_card": state.side_card.name,
        }
    )


def _make_runtime(human_player: Player) -> VisionGameRuntime:
    return VisionGameRuntime(
        VisionRuntimeConfig(
            human_player=human_player,
            required_repeats=2,
            ai_depth=1,
            ai_evaluator="v1",
        )
    )


def _make_runtime_for_session(human_player: Player, session: VisionGameSession) -> VisionGameRuntime:
    """Build a VisionGameRuntime and inject the given session for testing."""
    runtime = _make_runtime(human_player)
    runtime.session = session
    return runtime


def _publish_session_warning(runtime: VisionGameRuntime, outcome: SessionOutcome) -> None:
    for _ in range(runtime._SESSION_WARNING_THRESHOLD):
        runtime._record_session_outcome(outcome)


def _confirm_observation_warning(
    runtime: VisionGameRuntime,
    kind: VisionObservationKind = VisionObservationKind.LOW_CONFIDENCE_CARD,
) -> None:
    warning = VisionObservationError(kind)
    for _ in range(runtime._IN_GAME_OBSERVATION_WARNING_THRESHOLD):
        runtime._reset_pending_session_warning()
        runtime._record_observation_warning(warning)


def test_initial_player_is_inferred_from_side_card_stamp():
    state = GameState.initial(seed=1)
    snapshot = _snapshot_from_state(state)

    config = VisionRuntimeConfig(
        human_player=Player.RED,
        required_repeats=2,
        ai_depth=1,
        ai_evaluator="v1",
    )
    runtime = VisionGameRuntime(config)

    assert runtime._initial_player_from_snapshot(snapshot) is state.side_card.stamp


@pytest.mark.parametrize(
    "state_selector",
    [
        lambda state: state,
        lambda state: apply_action(state, generate_legal_actions(state)[0]),
    ],
)
def test_state_from_snapshot_uses_expected_turn_while_waiting_for_human_move(
    state_selector,
):
    state = GameState.initial(seed=1)
    session = VisionGameSession(
        human_player=state.to_move,
        ai_player=state.to_move.opponent(),
        ai_controller=AIController(depth=1, evaluator_name="v1"),
        stabilizer=StateStabilizer(required_repeats=2),
    )
    session.current_state = state
    session.phase = SessionPhase.WAITING_HUMAN_MOVE

    runtime = _make_runtime_for_session(state.to_move, session)
    expected_state = state_selector(state)
    observed = runtime._state_from_snapshot_for_session(_snapshot_from_state(expected_state))

    assert observed == expected_state


def test_state_from_snapshot_uses_expected_turn_while_waiting_for_ai_execution():
    board = [[None for _ in range(5)] for _ in range(5)]
    board[3][2] = Piece(owner=Player.BLUE, kind=PieceType.MASTER)
    board[0][4] = Piece(owner=Player.RED, kind=PieceType.MASTER)
    state = GameState(
        board=board,
        to_move=Player.BLUE,
        red_cards=(GameState.initial(seed=1).red_cards[0], GameState.initial(seed=1).red_cards[1]),
        blue_cards=(GameState.initial(seed=1).blue_cards[0], GameState.initial(seed=1).blue_cards[1]),
        side_card=GameState.initial(seed=1).side_card,
    )
    action = Move(from_pos=(3, 2), to_pos=(4, 2), card_index=0)
    expected_state = apply_action(state, action)

    session = VisionGameSession(
        human_player=Player.RED,
        ai_player=Player.BLUE,
        ai_controller=AIController(depth=1, evaluator_name="v1"),
        stabilizer=StateStabilizer(required_repeats=2),
    )
    session.current_state = state
    session.expected_state = expected_state
    session.phase = SessionPhase.WAITING_AI_EXECUTION

    runtime = _make_runtime_for_session(Player.RED, session)
    observed = runtime._state_from_snapshot_for_session(_snapshot_from_state(expected_state))

    assert observed == expected_state


def test_step_ignores_transient_observation_errors() -> None:
    class _FakePipeline:
        def snapshot_from_frame(self, frame):
            raise VisionObservationError(
                VisionObservationKind.GENERIC,
                debug_message="transient invalid observation",
            )

    class _FakeCamera:
        def read(self):
            return True, np.zeros((4, 4, 3), dtype=np.uint8)

    runtime = VisionGameRuntime(
        VisionRuntimeConfig(
            human_player=Player.RED,
            required_repeats=2,
            ai_depth=1,
            ai_evaluator="v1",
        ),
        pipeline=_FakePipeline(),
    )
    runtime.running = True
    runtime._camera = _FakeCamera()

    state = runtime.step()

    assert state.error_message is None
    assert state.observation_warning_kind is None


def test_step_reports_repeated_bootstrap_observation_errors() -> None:
    class _FakePipeline:
        def snapshot_from_frame(self, frame):
            raise VisionObservationError(
                VisionObservationKind.LOW_CONFIDENCE_CARD,
                debug_message="Cards must be 5 unique cards across red, blue and side.",
            )

    class _FakeCamera:
        def read(self):
            return True, np.zeros((4, 4, 3), dtype=np.uint8)

    runtime = VisionGameRuntime(
        VisionRuntimeConfig(
            human_player=Player.RED,
            required_repeats=2,
            ai_depth=1,
            ai_evaluator="v1",
        ),
        pipeline=_FakePipeline(),
    )
    runtime.running = True
    runtime._camera = _FakeCamera()

    first_state = runtime.step()
    second_state = runtime.step()

    assert first_state.observation_warning_kind is None
    assert second_state.observation_warning_kind is VisionObservationKind.LOW_CONFIDENCE_CARD
    assert second_state.error_message is None
    assert build_status_view(second_state).title == "No se puede confirmar la posición inicial"
    assert build_status_view(second_state).detail == "Revisa que las cartas estén bien colocadas y visibles."


def test_step_clears_observation_warning_after_valid_observation() -> None:
    initial_state = GameState.initial(seed=1)
    snapshot = _snapshot_from_state(initial_state)

    class _FakePipeline:
        def __init__(self):
            self.calls = 0

        def snapshot_from_frame(self, frame):
            self.calls += 1
            if self.calls <= 2:
                raise VisionObservationError(
                    VisionObservationKind.LOW_CONFIDENCE_CARD,
                    debug_message="side_card cannot be empty.",
                )
            return snapshot

    class _FakeCamera:
        def read(self):
            return True, np.zeros((4, 4, 3), dtype=np.uint8)

    runtime = VisionGameRuntime(
        VisionRuntimeConfig(
            human_player=initial_state.to_move,
            required_repeats=2,
            ai_depth=1,
            ai_evaluator="v1",
        ),
        pipeline=_FakePipeline(),
    )
    runtime.running = True
    runtime._camera = _FakeCamera()

    warned_state = runtime.step()
    warned_state = runtime.step()
    recovered_state = runtime.step()

    assert warned_state.observation_warning_kind is VisionObservationKind.LOW_CONFIDENCE_CARD
    assert build_status_view(warned_state).title == "No se puede confirmar la posición inicial"
    assert build_status_view(warned_state).detail == "Revisa que las cartas estén bien colocadas y visibles."
    assert recovered_state.observation_warning_kind is None


def test_step_reports_repeated_in_game_observation_errors_after_higher_threshold() -> None:
    initial_state = GameState.initial(seed=1)
    session = VisionGameSession(
        human_player=initial_state.to_move,
        ai_player=initial_state.to_move.opponent(),
        ai_controller=AIController(depth=1, evaluator_name="v1"),
        stabilizer=StateStabilizer(required_repeats=2),
    )
    session.current_state = initial_state
    session.phase = SessionPhase.WAITING_HUMAN_MOVE

    class _FakePipeline:
        def __init__(self) -> None:
            self.calls = 0

        def snapshot_from_frame(self, frame):
            self.calls += 1
            if self.calls == 1:
                raise VisionObservationError(
                    VisionObservationKind.LOW_CONFIDENCE_CARD,
                    debug_message="Low-confidence card prediction for slot 'side': 0.14 < 0.50.",
                )
            if self.calls == 2:
                raise VisionObservationError(
                    VisionObservationKind.LOW_CONFIDENCE_CARD,
                    debug_message="Low-confidence card prediction for slot 'red_0': 0.29 < 0.50.",
                )
            raise VisionObservationError(
                VisionObservationKind.LOW_CONFIDENCE_CARD,
                debug_message="Low-confidence card prediction for slot 'blue_1': 0.21 < 0.50.",
            )

    class _FakeCamera:
        def read(self):
            return True, np.zeros((4, 4, 3), dtype=np.uint8)

    runtime = _make_runtime_for_session(initial_state.to_move, session)
    runtime.pipeline = _FakePipeline()
    runtime.running = True
    runtime._camera = _FakeCamera()

    states = [runtime.step() for _ in range(runtime._IN_GAME_OBSERVATION_WARNING_THRESHOLD)]

    for state in states[:-1]:
        assert state.observation_warning_kind is None

    warned_state = states[-1]
    assert warned_state.observation_warning_kind is VisionObservationKind.LOW_CONFIDENCE_CARD
    assert build_status_view(warned_state).title == "Lectura inválida"
    assert build_status_view(warned_state).detail == "Revisa que las cartas estén bien colocadas y visibles."


@pytest.mark.parametrize(
    "outcome",
    [
        SessionOutcome.HUMAN_MOVE_REJECTED,
        SessionOutcome.AI_EXECUTION_MISMATCH,
    ],
)
def test_runtime_delays_rejected_session_outcomes(outcome: SessionOutcome) -> None:
    initial_state = GameState.initial(seed=1)
    runtime = _make_runtime(initial_state.to_move)

    for _ in range(runtime._SESSION_WARNING_THRESHOLD - 1):
        runtime._record_session_outcome(outcome)
        assert runtime.get_state().last_outcome is None

    runtime._record_session_outcome(outcome)
    assert runtime.get_state().last_outcome is outcome


def test_runtime_requires_consecutive_rejected_session_outcomes() -> None:
    initial_state = GameState.initial(seed=1)
    runtime = _make_runtime(initial_state.to_move)

    runtime._record_session_outcome(SessionOutcome.HUMAN_MOVE_REJECTED)
    runtime._record_session_outcome(SessionOutcome.HUMAN_MOVE_REJECTED)
    runtime._record_session_outcome(SessionOutcome.COLLECTING)
    runtime._record_session_outcome(SessionOutcome.HUMAN_MOVE_REJECTED)
    runtime._record_session_outcome(SessionOutcome.HUMAN_MOVE_REJECTED)

    assert runtime.get_state().last_outcome is None


def test_runtime_respects_single_outcome_warning_threshold() -> None:
    initial_state = GameState.initial(seed=1)
    runtime = _make_runtime(initial_state.to_move)
    runtime._SESSION_WARNING_THRESHOLD = 1

    runtime._record_session_outcome(SessionOutcome.HUMAN_MOVE_REJECTED)

    assert runtime.get_state().last_outcome is SessionOutcome.HUMAN_MOVE_REJECTED


def test_transient_observation_warning_keeps_published_session_warning() -> None:
    initial_state = GameState.initial(seed=1)
    runtime = _make_runtime(initial_state.to_move)
    runtime.session.current_state = initial_state
    runtime.session.phase = SessionPhase.WAITING_HUMAN_MOVE
    _publish_session_warning(runtime, SessionOutcome.HUMAN_MOVE_REJECTED)

    runtime._reset_pending_session_warning()
    runtime._record_observation_warning(
        VisionObservationError(VisionObservationKind.LOW_CONFIDENCE_CARD)
    )

    state = runtime.get_state()
    assert state.last_outcome is SessionOutcome.HUMAN_MOVE_REJECTED
    assert state.observation_warning_kind is None


def test_published_session_warning_stays_visible_while_same_outcome_rebuilds_evidence() -> None:
    initial_state = GameState.initial(seed=1)
    runtime = _make_runtime(initial_state.to_move)
    _publish_session_warning(runtime, SessionOutcome.HUMAN_MOVE_REJECTED)

    runtime._reset_pending_session_warning()

    for _ in range(runtime._SESSION_WARNING_THRESHOLD - 1):
        runtime._record_session_outcome(SessionOutcome.HUMAN_MOVE_REJECTED)
        assert runtime.get_state().last_outcome is SessionOutcome.HUMAN_MOVE_REJECTED


@pytest.mark.parametrize(
    ("phase", "outcome"),
    [
        (SessionPhase.WAITING_HUMAN_MOVE, SessionOutcome.HUMAN_MOVE_REJECTED),
        (SessionPhase.WAITING_AI_EXECUTION, SessionOutcome.AI_EXECUTION_MISMATCH),
    ],
)
def test_confirmed_observation_warning_takes_priority_over_session_warning(
    phase: SessionPhase,
    outcome: SessionOutcome,
) -> None:
    initial_state = GameState.initial(seed=1)
    runtime = _make_runtime(initial_state.to_move)
    runtime.session.current_state = initial_state
    runtime.session.phase = phase
    _publish_session_warning(runtime, outcome)
    _confirm_observation_warning(runtime)

    state = runtime.get_state()
    assert state.last_outcome is outcome
    assert state.observation_warning_kind is VisionObservationKind.LOW_CONFIDENCE_CARD
    assert build_status_view(state).title == "Lectura inválida"


def test_reset_discards_buffered_frames_before_bootstrap() -> None:
    initial_state = GameState.initial(seed=1)
    snapshot = _snapshot_from_state(initial_state)

    class _FakePipeline:
        def __init__(self) -> None:
            self.calls = 0

        def snapshot_from_frame(self, frame):
            self.calls += 1
            return snapshot

    class _FakeCamera:
        def read(self):
            return True, np.zeros((4, 4, 3), dtype=np.uint8)

    runtime = VisionGameRuntime(
        VisionRuntimeConfig(
            human_player=initial_state.to_move,
            required_repeats=1,
            ai_depth=1,
            ai_evaluator="v1",
        ),
        pipeline=_FakePipeline(),
    )
    runtime.running = True
    runtime._camera = _FakeCamera()

    runtime.reset()

    for _ in range(runtime._RESET_FRAME_DISCARD_COUNT):
        state = runtime.step()
        assert state.current_state is None
        assert runtime.pipeline.calls == 0

    bootstrapped_state = runtime.step()

    assert runtime.pipeline.calls == 1
    assert bootstrapped_state.current_state == initial_state
