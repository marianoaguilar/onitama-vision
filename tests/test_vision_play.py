import numpy as np

from onitama.ai.controllers import AIController
from onitama.app.errors import VisionObservationError, VisionObservationKind
from onitama.app.vision_models import VisionRuntimeConfig
from onitama.app.vision_runtime import VisionGameRuntime
from onitama.engine.moves import Move
from onitama.engine.pieces import Piece, PieceType, Player
from onitama.engine.rules import apply_action, generate_legal_actions
from onitama.engine.state import GameState
from onitama.gui.view_logic import build_status_view
from onitama.integration.session import SessionPhase, VisionGameSession
from onitama.integration.stabilizer import StateStabilizer
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


def _make_runtime_for_session(human_player: Player, session: VisionGameSession) -> VisionGameRuntime:
    """Build a VisionGameRuntime and inject the given session for testing."""
    config = VisionRuntimeConfig(
        human_player=human_player,
        required_repeats=2,
        ai_depth=1,
        ai_evaluator="v1",
    )
    runtime = VisionGameRuntime(config)
    runtime.session = session
    return runtime


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


def test_state_from_snapshot_uses_current_turn_for_unchanged_human_observation():
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
    observed = runtime._state_from_snapshot_for_session(_snapshot_from_state(state))

    assert observed == state


def test_state_from_snapshot_uses_opponent_turn_after_human_move():
    state = GameState.initial(seed=1)
    next_state = apply_action(state, generate_legal_actions(state)[0])
    session = VisionGameSession(
        human_player=state.to_move,
        ai_player=state.to_move.opponent(),
        ai_controller=AIController(depth=1, evaluator_name="v1"),
        stabilizer=StateStabilizer(required_repeats=2),
    )
    session.current_state = state
    session.phase = SessionPhase.WAITING_HUMAN_MOVE

    runtime = _make_runtime_for_session(state.to_move, session)
    observed = runtime._state_from_snapshot_for_session(_snapshot_from_state(next_state))

    assert observed == next_state


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
    assert state.observation_kind is None


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

    assert first_state.observation_kind is None
    assert second_state.observation_kind is VisionObservationKind.LOW_CONFIDENCE_CARD
    assert second_state.error_message is None
    assert build_status_view(second_state).title == "No se puede confirmar la posición inicial"
    assert build_status_view(second_state).detail == "Revisa que las cartas estén bien colocadas y visibles."


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
        def snapshot_from_frame(self, frame):
            raise VisionObservationError(
                VisionObservationKind.LOW_CONFIDENCE_CARD,
                debug_message="side_card cannot be empty.",
            )

    class _FakeCamera:
        def read(self):
            return True, np.zeros((4, 4, 3), dtype=np.uint8)

    runtime = _make_runtime_for_session(initial_state.to_move, session)
    runtime.pipeline = _FakePipeline()
    runtime.running = True
    runtime._camera = _FakeCamera()

    first_state = runtime.step()
    second_state = runtime.step()
    third_state = runtime.step()

    assert first_state.observation_kind is None
    assert second_state.observation_kind is None
    assert third_state.observation_kind is VisionObservationKind.LOW_CONFIDENCE_CARD
    assert build_status_view(third_state).title == "Lectura inválida"
    assert build_status_view(third_state).detail == "Revisa que las cartas estén bien colocadas y visibles."


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

    assert warned_state.observation_kind is VisionObservationKind.LOW_CONFIDENCE_CARD
    assert build_status_view(warned_state).title == "No se puede confirmar la posición inicial"
    assert build_status_view(warned_state).detail == "Revisa que las cartas estén bien colocadas y visibles."
    assert recovered_state.observation_kind is None


def test_step_groups_low_confidence_card_warnings_with_varying_details() -> None:
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

    first_state = runtime.step()
    second_state = runtime.step()
    third_state = runtime.step()

    assert first_state.observation_kind is None
    assert second_state.observation_kind is None
    assert third_state.observation_kind is VisionObservationKind.LOW_CONFIDENCE_CARD
    assert build_status_view(third_state).title == "Lectura inválida"
    assert build_status_view(third_state).detail == "Revisa que las cartas estén bien colocadas y visibles."
