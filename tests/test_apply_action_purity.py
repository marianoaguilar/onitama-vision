from onitama.state import GameState
from onitama.rules import generate_legal_actions, apply_action
from onitama.moves import Move, Pass


def test_apply_action_does_not_mutate_original_state_board():
    s = GameState.initial(seed=1)

    # Take a snapshot of the original board contents
    original_snapshot = tuple(tuple(cell for cell in row) for row in s.board)

    actions = generate_legal_actions(s)
    assert actions, "Initial state should have legal actions"

    a = actions[0]
    s2 = apply_action(s, a)

    # Original state's board contents should be identical after applying the action
    after_snapshot = tuple(tuple(cell for cell in row) for row in s.board)
    assert after_snapshot == original_snapshot

    # And the boards must not be the same object
    assert s2.board is not s.board
    for r in range(5):
        assert s2.board[r] is not s.board[r]

    # The new state should change for a Move; for Pass, the board should not change
    if isinstance(a, Move):
        diff_cells = sum(
            (s.board[r][c] != s2.board[r][c])
            for r in range(5)
            for c in range(5)
        )
        assert diff_cells == 2
    elif isinstance(a, Pass):
        assert s2.board == s.board
    else:
        raise AssertionError("Unknown action type")
