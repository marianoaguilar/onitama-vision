from onitama.cards import RABBIT
from onitama.pieces import Player


def test_rabbit_is_rotated_180_for_blue():
    red = RABBIT.deltas_for(Player.RED)
    blue = RABBIT.deltas_for(Player.BLUE)

    expected_blue = tuple((-dr, -dc) for dr, dc in red)

    assert blue == expected_blue
