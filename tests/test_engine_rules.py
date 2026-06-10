from onitama.engine.cards import ALL_CARDS, BOAR, CRAB, HORSE, RABBIT, TIGER
from onitama.engine.moves import Move, Pass
from onitama.engine.pieces import Piece, PieceType, Player
from onitama.engine.rules import apply_action, generate_legal_actions, winner
from onitama.engine.state import GameState


def _known_red_state() -> GameState:
    state = GameState.initial(seed=1)
    return GameState(
        board=state.board,
        to_move=Player.RED,
        red_cards=(TIGER, HORSE),
        blue_cards=(CRAB, BOAR),
        side_card=CRAB,
    )


def _state_from_board(board):
    return GameState(
        board=board,
        to_move=Player.RED,
        red_cards=(ALL_CARDS[0], ALL_CARDS[1]),
        blue_cards=(ALL_CARDS[2], ALL_CARDS[3]),
        side_card=ALL_CARDS[4],
    )


def _piece(owner: Player, kind: PieceType = PieceType.MASTER) -> Piece:
    return Piece(owner=owner, kind=kind)


def test_initial_board_setup_and_unique_cards():
    state = GameState.initial(seed=1)

    assert [(p.owner, p.kind) for p in state.board[0]] == [
        (Player.BLUE, PieceType.STUDENT),
        (Player.BLUE, PieceType.STUDENT),
        (Player.BLUE, PieceType.MASTER),
        (Player.BLUE, PieceType.STUDENT),
        (Player.BLUE, PieceType.STUDENT),
    ]
    assert [(p.owner, p.kind) for p in state.board[4]] == [
        (Player.RED, PieceType.STUDENT),
        (Player.RED, PieceType.STUDENT),
        (Player.RED, PieceType.MASTER),
        (Player.RED, PieceType.STUDENT),
        (Player.RED, PieceType.STUDENT),
    ]
    assert all(cell is None for row in state.board[1:4] for cell in row)

    all_cards = [*state.red_cards, *state.blue_cards, state.side_card]
    assert len(all_cards) == 5
    assert len({c.name for c in all_cards}) == 5


def test_rabbit_is_rotated_180_for_blue():
    red = RABBIT.deltas_for(Player.RED)

    assert RABBIT.deltas_for(Player.BLUE) == tuple((-dr, -dc) for dr, dc in red)


def test_apply_action_does_not_mutate_original_state_board():
    state = GameState.initial(seed=1)
    original_board = state.board
    move = next(a for a in generate_legal_actions(state) if isinstance(a, Move))

    next_state = apply_action(state, move)

    assert state.board == original_board
    assert next_state.board is not state.board
    assert sum(
        state.board[r][c] != next_state.board[r][c]
        for r in range(5)
        for c in range(5)
    ) == 2


def test_generate_pass_when_no_moves_exist():
    board = [[None for _ in range(5)] for _ in range(5)]
    for r in range(5):
        kind = PieceType.MASTER if r == 4 else PieceType.STUDENT
        board[r][0] = Piece(owner=Player.RED, kind=kind)
    board[2][4] = Piece(owner=Player.BLUE, kind=PieceType.MASTER)

    state = GameState(
        board=board,
        to_move=Player.RED,
        red_cards=(HORSE, TIGER),
        blue_cards=(CRAB, BOAR),
        side_card=CRAB,
    )

    assert generate_legal_actions(state) == [Pass(0), Pass(1)]


def test_swap_cards_after_pass_index_0():
    state = _known_red_state()

    next_state = apply_action(state, Pass(0))

    assert next_state.board == state.board
    assert next_state.to_move is Player.BLUE
    assert next_state.red_cards == (CRAB, HORSE)
    assert next_state.blue_cards == state.blue_cards
    assert next_state.side_card is TIGER


def test_swap_cards_after_a_move():
    state = _known_red_state()
    move = next(a for a in generate_legal_actions(state) if isinstance(a, Move))

    next_state = apply_action(state, move)

    assert next_state.to_move is Player.BLUE
    assert next_state.side_card is state.red_cards[move.card_index]
    assert next_state.red_cards[move.card_index] is state.side_card
    assert next_state.red_cards[1 - move.card_index] is state.red_cards[1 - move.card_index]
    assert sum(
        state.board[r][c] != next_state.board[r][c]
        for r in range(5)
        for c in range(5)
    ) == 2


def test_winner_stone_when_opponent_master_missing():
    board = [[None for _ in range(5)] for _ in range(5)]
    board[4][2] = _piece(Player.RED)

    assert winner(_state_from_board(board)) == (Player.RED, "Capture of Master")


def test_winner_stream_when_red_master_reaches_temple():
    board = [[None for _ in range(5)] for _ in range(5)]
    board[0][2] = _piece(Player.RED)
    board[4][2] = _piece(Player.BLUE)

    assert winner(_state_from_board(board)) == (Player.RED, "Reach Temple")
