from onitama.state import GameState
from onitama.pieces import Player, PieceType


def test_initial_board_setup_and_unique_cards():
    s = GameState.initial(seed=1)

    # --- Board setup ---
    # Blue row 0: 5 pieces, master at column 2
    assert all(s.board[0][c] is not None for c in range(5))
    assert s.board[0][2].owner == Player.BLUE
    assert s.board[0][2].kind == PieceType.MASTER
    for c in range(5):
        if c == 2:
            continue
        assert s.board[0][c].owner == Player.BLUE
        assert s.board[0][c].kind == PieceType.STUDENT

    # Red row 4: 5 pieces, master at column 2
    assert all(s.board[4][c] is not None for c in range(5))
    assert s.board[4][2].owner == Player.RED
    assert s.board[4][2].kind == PieceType.MASTER
    for c in range(5):
        if c == 2:
            continue
        assert s.board[4][c].owner == Player.RED
        assert s.board[4][c].kind == PieceType.STUDENT

    # Middle rows should be empty in the initial setup
    for r in range(1, 4):
        assert all(s.board[r][c] is None for c in range(5))

    # --- Cards uniqueness ---
    all_cards = list(s.red_cards) + list(s.blue_cards) + [s.side_card]
    assert len(all_cards) == 5
    assert len({c.name for c in all_cards}) == 5
