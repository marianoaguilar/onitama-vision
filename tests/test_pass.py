from onitama.state import GameState
from onitama.pieces import Player, PieceType, Piece
from onitama.cards import OX, TIGER
from onitama.rules import generate_legal_actions, apply_action
from onitama.moves import Pass
from cli.render import format_action, render_state
from cli.play import _prompt_int

def make_pass_test_state() -> GameState:
    state = GameState.empty(
        to_move=Player.RED,
        red_cards=(TIGER, TIGER),
        blue_cards=(TIGER, TIGER),
        side_card=OX,
    )
    
    # Place BLUE pieces somewhere
    state.board[0][4] = Piece(owner=Player.BLUE, kind=PieceType.MASTER)

    # Place pieces so that RED has no legal moves (vertical line and only TIGER cards)
    state.board[0][2] = Piece(owner=Player.RED, kind=PieceType.STUDENT)
    state.board[1][2] = Piece(owner=Player.RED, kind=PieceType.STUDENT)
    state.board[2][2] = Piece(owner=Player.RED, kind=PieceType.MASTER)
    state.board[3][2] = Piece(owner=Player.RED, kind=PieceType.STUDENT)
    state.board[4][2] = Piece(owner=Player.RED, kind=PieceType.STUDENT)
    
    return state


if __name__ == "__main__":
    s = make_pass_test_state()
    print("\nTest state:")
    print(render_state(s))
    actions = generate_legal_actions(s)
    
    print("\nLegal actions:")
    for i, act in enumerate(actions, start=1):
        print(f"{i:2d}) {format_action(s, act)}")
    
    choice = _prompt_int(f"\nChoose an action (1-{len(actions)}): ", 1, len(actions))
    action = actions[choice - 1]

    s = apply_action(s, action)
    print("\n" + "-" * 60 + "\n")

    print("\nTest state after passing:")
    print(render_state(s))
