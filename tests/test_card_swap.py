from onitama.state import GameState
from onitama.rules import apply_action, generate_legal_actions
from onitama.moves import Pass, Move
from onitama.cards import TIGER, HORSE, CRAB, BOAR
from onitama.pieces import Player


def test_swap_cards_after_pass_index_0():
    s = GameState.initial(seed=1)

    # Force known cards so the test is deterministic
    s = GameState(
        board=s.board,
        to_move=Player.RED,
        red_cards=(TIGER, HORSE),
        blue_cards=(CRAB, BOAR),
        side_card=CRAB,
    )

    s2 = apply_action(s, Pass(0))

    # Board unchanged
    assert s2.board == s.board

    # Turn changes
    assert s2.to_move == Player.BLUE

    # Swap: red_cards[0] <-> side_card
    assert s2.side_card.name == TIGER.name
    assert s2.red_cards[0].name == CRAB.name

    # The other RED card does not change
    assert s2.red_cards[1].name == HORSE.name

    # Blue cards do not change
    assert [c.name for c in s2.blue_cards] == [CRAB.name, BOAR.name]


def test_swap_cards_after_a_move():
    s0 = GameState.initial(seed=1)

    # Fix cards to control which swap we expect.
    # We need at least one legal move with card_index 0 or 1.
    s = GameState(
        board=s0.board,
        to_move=Player.RED,
        red_cards=(TIGER, HORSE),
        blue_cards=(CRAB, BOAR),
        side_card=CRAB,
    )

    actions = generate_legal_actions(s)
    move_actions = [a for a in actions if isinstance(a, Move)]
    assert move_actions, "Expected at least one legal Move in initial position"

    # Pick the first legal Move
    m = move_actions[0]
    used_card = s.red_cards[m.card_index]
    old_side = s.side_card

    s2 = apply_action(s, m)

    # Turn changes
    assert s2.to_move == Player.BLUE

    # Correct swap
    assert s2.side_card.name == used_card.name
    assert s2.red_cards[m.card_index].name == old_side.name

    # The unused card remains the same
    other = 1 - m.card_index
    assert s2.red_cards[other].name == s.red_cards[other].name

    # And the board must have changed (a real Move)
    diff_cells = sum(
        (s.board[r][c] != s2.board[r][c])
        for r in range(5)
        for c in range(5)
    )
    assert diff_cells == 2
