from onitama.ai.controllers import AIController
from onitama.engine.cards import ALL_CARDS, BOAR, CRAB
from onitama.engine.moves import Move
from onitama.engine.pieces import Piece, PieceType, Player
from onitama.engine.rules import Action, apply_action, generate_legal_actions
from onitama.engine.state import GameState
from onitama.integration.session import SessionOutcome, SessionPhase, VisionGameSession
from onitama.integration.stabilizer import StateStabilizer


class StubAIController(AIController):
    forced_action: Action

    def __init__(self, forced_action: Action) -> None:
        super().__init__(depth=1, evaluator_name="v1")
        self.forced_action = forced_action

    def select_action(self, state: GameState) -> Action:
        return self.forced_action


def _session(
    *,
    human_player: Player,
    ai_player: Player,
    required_repeats: int = 2,
) -> VisionGameSession:
    return VisionGameSession(
        human_player=human_player,
        ai_player=ai_player,
        ai_controller=AIController(depth=1, evaluator_name="v1"),
        stabilizer=StateStabilizer(required_repeats=required_repeats),
    )


def _stabilize(session: VisionGameSession, observed_state: GameState):
    session.process_observation(observed_state)
    return session.process_observation(observed_state)


def test_session_bootstrap_waits_until_state_is_stable():
    initial_state = GameState.initial(seed=1)
    session = _session(
        human_player=initial_state.to_move,
        ai_player=initial_state.to_move.opponent(),
        required_repeats=2,
    )

    result = session.process_observation(initial_state)

    assert result.phase is SessionPhase.BOOTSTRAP
    assert result.outcome is SessionOutcome.COLLECTING
    assert result.current_state is None


def test_session_bootstrap_routes_to_human_turn_when_human_starts():
    initial_state = GameState.initial(seed=1)
    session = _session(
        human_player=initial_state.to_move,
        ai_player=initial_state.to_move.opponent(),
        required_repeats=2,
    )

    result = _stabilize(session, initial_state)

    assert result.phase is SessionPhase.WAITING_HUMAN_MOVE
    assert result.outcome is SessionOutcome.BOOTSTRAPPED
    assert result.current_state == initial_state
    assert session.current_state == initial_state


def test_session_bootstrap_routes_to_ai_turn_when_ai_starts():
    initial_state = GameState.initial(seed=1)
    session = _session(
        human_player=initial_state.to_move.opponent(),
        ai_player=initial_state.to_move,
        required_repeats=2,
    )

    result = _stabilize(session, initial_state)

    assert result.phase is SessionPhase.READY_FOR_AI
    assert result.outcome is SessionOutcome.BOOTSTRAPPED
    assert result.current_state == initial_state


def test_session_reports_unchanged_stable_observation_during_human_turn():
    initial_state = GameState.initial(seed=1)
    session = _session(
        human_player=initial_state.to_move,
        ai_player=initial_state.to_move.opponent(),
        required_repeats=2,
    )
    _stabilize(session, initial_state)

    result = _stabilize(session, initial_state)

    assert result.phase is SessionPhase.WAITING_HUMAN_MOVE
    assert result.outcome is SessionOutcome.UNCHANGED_OBSERVATION
    assert result.current_state == initial_state


def test_session_accepts_stable_legal_human_move_and_switches_to_ai_turn():
    initial_state = GameState.initial(seed=1)
    next_state = apply_action(initial_state, generate_legal_actions(initial_state)[0])
    session = _session(
        human_player=initial_state.to_move,
        ai_player=initial_state.to_move.opponent(),
        required_repeats=2,
    )
    _stabilize(session, initial_state)

    result = _stabilize(session, next_state)

    assert result.phase is SessionPhase.READY_FOR_AI
    assert result.outcome is SessionOutcome.HUMAN_MOVE_ACCEPTED
    assert result.current_state == next_state
    assert result.sync_result is not None
    assert result.sync_result.accepted is True
    assert session.current_state == next_state


def test_session_rejects_stable_illegal_human_move():
    initial_state = GameState.initial(seed=1)
    illegal_state = GameState(
        board=initial_state.board,
        to_move=initial_state.to_move.opponent(),
        red_cards=initial_state.red_cards,
        blue_cards=initial_state.blue_cards,
        side_card=initial_state.side_card,
    )
    session = _session(
        human_player=initial_state.to_move,
        ai_player=initial_state.to_move.opponent(),
        required_repeats=2,
    )
    _stabilize(session, initial_state)

    result = _stabilize(session, illegal_state)

    assert result.phase is SessionPhase.WAITING_HUMAN_MOVE
    assert result.outcome is SessionOutcome.HUMAN_MOVE_REJECTED
    assert result.current_state == initial_state
    assert result.sync_result is not None
    assert result.sync_result.accepted is False


def test_session_moves_to_finished_when_accepted_human_move_is_terminal():
    board = [[None for _ in range(5)] for _ in range(5)]
    board[1][2] = Piece(owner=Player.RED, kind=PieceType.MASTER)
    board[4][4] = Piece(owner=Player.BLUE, kind=PieceType.MASTER)

    initial_state = GameState(
        board=board,
        to_move=Player.RED,
        red_cards=(BOAR, CRAB),
        blue_cards=(ALL_CARDS[0], ALL_CARDS[1]),
        side_card=ALL_CARDS[2],
    )
    terminal_action = Move(from_pos=(1, 2), to_pos=(0, 2), card_index=0)
    terminal_state = apply_action(initial_state, terminal_action)
    session = _session(
        human_player=Player.RED,
        ai_player=Player.BLUE,
        required_repeats=2,
    )
    _stabilize(session, initial_state)

    result = _stabilize(session, terminal_state)

    assert result.phase is SessionPhase.FINISHED
    assert result.outcome is SessionOutcome.HUMAN_MOVE_ACCEPTED
    assert result.current_state == terminal_state


def test_run_ai_turn_selects_action_and_waits_for_expected_state():
    initial_state = GameState.initial(seed=1)
    session = _session(
        human_player=initial_state.to_move.opponent(),
        ai_player=initial_state.to_move,
        required_repeats=2,
    )
    _stabilize(session, initial_state)

    result = session.run_ai_turn()

    assert result.phase is SessionPhase.WAITING_AI_EXECUTION
    assert result.outcome is SessionOutcome.AI_ACTION_SELECTED
    assert result.current_state == initial_state
    assert result.expected_state is not None
    assert result.ai_action is not None
    assert session.expected_state == result.expected_state
    assert session.last_ai_action == result.ai_action


def test_session_waits_for_physical_ai_execution_after_action_selection():
    initial_state = GameState.initial(seed=1)
    session = _session(
        human_player=initial_state.to_move.opponent(),
        ai_player=initial_state.to_move,
        required_repeats=2,
    )
    _stabilize(session, initial_state)
    session.run_ai_turn()

    result = _stabilize(session, initial_state)

    assert result.phase is SessionPhase.WAITING_AI_EXECUTION
    assert result.outcome is SessionOutcome.AWAITING_AI_EXECUTION
    assert result.current_state == initial_state
    assert result.expected_state == session.expected_state


def test_session_confirms_expected_ai_state_and_returns_to_human_turn():
    initial_state = GameState.initial(seed=1)
    session = _session(
        human_player=initial_state.to_move.opponent(),
        ai_player=initial_state.to_move,
        required_repeats=2,
    )
    _stabilize(session, initial_state)
    ai_step = session.run_ai_turn()
    assert ai_step.expected_state is not None

    result = _stabilize(session, ai_step.expected_state)

    assert result.phase is SessionPhase.WAITING_HUMAN_MOVE
    assert result.outcome is SessionOutcome.AI_EXECUTION_CONFIRMED
    assert result.current_state == ai_step.expected_state
    assert result.ai_action == ai_step.ai_action
    assert session.expected_state is None
    assert session.last_ai_action is None


def test_session_moves_to_finished_when_confirmed_ai_state_is_terminal():
    board = [[None for _ in range(5)] for _ in range(5)]
    board[3][2] = Piece(owner=Player.BLUE, kind=PieceType.MASTER)
    board[0][4] = Piece(owner=Player.RED, kind=PieceType.MASTER)

    initial_state = GameState(
        board=board,
        to_move=Player.BLUE,
        red_cards=(ALL_CARDS[0], ALL_CARDS[1]),
        blue_cards=(BOAR, CRAB),
        side_card=ALL_CARDS[2],
    )
    winning_action = Move(from_pos=(3, 2), to_pos=(4, 2), card_index=1)

    session = VisionGameSession(
        human_player=Player.RED,
        ai_player=Player.BLUE,
        ai_controller=StubAIController(winning_action),
        stabilizer=StateStabilizer(required_repeats=2),
    )
    _stabilize(session, initial_state)
    ai_step = session.run_ai_turn()
    assert ai_step.expected_state is not None

    result = _stabilize(session, ai_step.expected_state)

    assert result.phase is SessionPhase.FINISHED
    assert result.outcome is SessionOutcome.AI_EXECUTION_CONFIRMED
