from onitama.state import GameState
from onitama.pieces import Player, Piece, PieceType
from onitama.rules import winner
from onitama.cards import ALL_CARDS


def test_winner_stone_when_opponent_master_missing():
    board = [[None for _ in range(5)] for _ in range(5)]
    # Only red master on the board
    board[4][2] = Piece(owner=Player.RED, kind=PieceType.MASTER)

    s = GameState(
        board=board,
        to_move=Player.RED,
        red_cards=(ALL_CARDS[0], ALL_CARDS[1]),
        blue_cards=(ALL_CARDS[2], ALL_CARDS[3]),
        side_card=ALL_CARDS[4],
    )

    assert winner(s) == (Player.RED, "Capture of Master")


def test_winner_stream_when_red_master_reaches_temple():
    board = [[None for _ in range(5)] for _ in range(5)]

    # Red master in the blue temple (row 0, col 2) => RED wins by Stream
    board[0][2] = Piece(owner=Player.RED, kind=PieceType.MASTER)

    # Also place the blue master somewhere so it's not a Stone win
    board[4][2] = Piece(owner=Player.BLUE, kind=PieceType.MASTER)

    s = GameState(
        board=board,
        to_move=Player.RED,
        red_cards=(ALL_CARDS[0], ALL_CARDS[1]),
        blue_cards=(ALL_CARDS[2], ALL_CARDS[3]),
        side_card=ALL_CARDS[4],
    )

    assert winner(s) == (Player.RED, "Reach Temple")
