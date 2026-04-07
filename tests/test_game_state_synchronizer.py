from onitama.engine.cards import ALL_CARDS
from onitama.engine.pieces import Piece, PieceType, Player
from onitama.engine.rules import apply_action, generate_legal_actions
from onitama.engine.state import GameState
from onitama.integration.synchronizer import SyncStatus, match_observed_state


def test_synchronizer_accepts_unique_legal_successor():
    previous_state = GameState.initial(seed=1)
    action = generate_legal_actions(previous_state)[0]
    observed_state = apply_action(previous_state, action)

    result = match_observed_state(previous_state, observed_state)

    assert result.status is SyncStatus.ACCEPTED
    assert result.accepted is True
    assert result.match_count == 1
    assert result.matched_action == action


def test_synchronizer_marks_identical_state_as_unchanged():
    previous_state = GameState.initial(seed=1)

    result = match_observed_state(previous_state, previous_state)

    assert result.status is SyncStatus.UNCHANGED
    assert result.accepted is False
    assert result.match_count == 0
    assert result.matched_action is None
    assert "identical" in (result.reason or "")


def test_synchronizer_rejects_non_successor_state():
    previous_state = GameState.initial(seed=1)
    observed_state = GameState(
        board=previous_state.board,
        to_move=previous_state.to_move.opponent(),
        red_cards=previous_state.red_cards,
        blue_cards=previous_state.blue_cards,
        side_card=previous_state.side_card,
    )

    result = match_observed_state(previous_state, observed_state)

    assert result.status is SyncStatus.REJECTED
    assert result.accepted is False
    assert result.match_count == 0
    assert result.matched_action is None
    assert "does not match any legal successor" in (result.reason or "")


def test_synchronizer_rejects_when_previous_state_is_terminal():
    board = [[None for _ in range(5)] for _ in range(5)]
    board[0][2] = Piece(owner=Player.RED, kind=PieceType.MASTER)
    board[4][2] = Piece(owner=Player.BLUE, kind=PieceType.MASTER)

    previous_state = GameState(
        board=board,
        to_move=Player.RED,
        red_cards=(ALL_CARDS[0], ALL_CARDS[1]),
        blue_cards=(ALL_CARDS[2], ALL_CARDS[3]),
        side_card=ALL_CARDS[4],
    )
    observed_state = previous_state

    result = match_observed_state(previous_state, observed_state)

    assert result.status is SyncStatus.TERMINAL_PREVIOUS
    assert result.accepted is False
    assert result.match_count == 0
    assert "terminal" in (result.reason or "")
