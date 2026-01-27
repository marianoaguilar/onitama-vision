from __future__ import annotations

from dataclasses import dataclass

from onitama.pieces import Player

MoveDelta = tuple[int, int]  # (dr, dc)


@dataclass(frozen=True)
class Card:
    name: str
    deltas: tuple[MoveDelta, ...]

    def deltas_for(self, player: Player) -> tuple[MoveDelta, ...]:
        """
        Return movement deltas adjusted for the player's perspective.

        Convention:
        - We define deltas from RED's perspective.
        - For BLUE, we flip the row direction (dr -> -dr).
        """
        if player is Player.RED:
            return self.deltas
        return tuple((-dr, dc) for dr, dc in self.deltas)


# --- Temporary: one card to test the pipeline ---
TIGER = Card(
    name="Tiger",
    deltas=(
        (-2, 0),  # two forward
        (1, 0),   # one backward
    ),
)
