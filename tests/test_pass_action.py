from onitama.state import GameState
from onitama.pieces import Player, Piece, PieceType
from onitama.rules import generate_legal_actions
from onitama.cards import HORSE, TIGER, CRAB, BOAR
from onitama.moves import Pass, Move


def test_generate_pass_when_no_moves_exist():
    # Board: entire column 0 occupied by red pieces
    board = [[None for _ in range(5)] for _ in range(5)]
    for r in range(5):
        kind = PieceType.MASTER if r == 4 else PieceType.STUDENT
        board[r][0] = Piece(owner=Player.RED, kind=kind)
        
    # Avoid terminal state: add blue master
    board[2][4] = Piece(owner=Player.BLUE, kind=PieceType.MASTER)

    # RED cards that try to move vertically or left (blocked or out of bounds)
    red_cards = (HORSE, TIGER)

    # BLUE cards / side card are irrelevant (only to build GameState)
    blue_cards = (CRAB, BOAR)
    side_card = CRAB

    s = GameState(
        board=board,
        to_move=Player.RED,
        red_cards=red_cards,
        blue_cards=blue_cards,
        side_card=side_card,
    )

    actions = generate_legal_actions(s)

    # There should be only PASS actions and exactly 2 options
    assert actions == [Pass(0), Pass(1)]
    assert not any(isinstance(a, Move) for a in actions)
