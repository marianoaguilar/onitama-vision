from __future__ import annotations

from dataclasses import dataclass

from onitama.engine.pieces import Player

MoveDelta = tuple[int, int]  # (dr, dc)


@dataclass(frozen=True)
class Card:
    """
    Onitama movement card.

    - deltas are defined from RED's perspective:
      * dr < 0 means "forward" for RED (towards row 0, since RED starts at row 4).
      * dc < 0 means "left", dc > 0 means "right".
    - For BLUE, we rotate 180 degrees (dr -> -dr, dc -> -dc).
    - stamp indicates the card's seal color (used later to decide who starts when this is the side card).
    """
    name: str
    deltas: tuple[MoveDelta, ...]
    stamp: Player  # Player.RED or Player.BLUE

    def deltas_for(self, player: Player) -> tuple[MoveDelta, ...]:
        """Return movement deltas adjusted for the player's perspective."""
        if player is Player.RED:
            return self.deltas
        return tuple((-dr, -dc) for dr, dc in self.deltas)


# --- 16 real base-game cards ---

RABBIT = Card(
    name="Rabbit",
    deltas=(
        (-1, 1),
        (0, 2),
        (1, -1),
    ),
    stamp=Player.BLUE,
)

MONKEY = Card(
    name="Monkey",
    deltas=(
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ),
    stamp=Player.BLUE,
)

ELEPHANT = Card(
    name="Elephant",
    deltas=(
        (-1, -1),
        (-1, 1),
        (0, -1),
        (0, 1),
    ),
    stamp=Player.RED,
)

DRAGON = Card(
    name="Dragon",
    deltas=(
        (-1, -2),
        (-1, 2),
        (1, -1),
        (1, 1),
    ),
    stamp=Player.RED,
)

GOOSE = Card(
    name="Goose",
    deltas=(
        (-1, -1),
        (0, -1),
        (0, 1),
        (1, 1),
    ),
    stamp=Player.BLUE,
)

MANTIS = Card(
    name="Mantis",
    deltas=(
        (-1, -1),
        (-1, 1),
        (1, 0),
    ),
    stamp=Player.RED,
)

ROOSTER = Card(
    name="Rooster",
    deltas=(
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
    ),
    stamp=Player.RED,
)

CRANE = Card(
    name="Crane",
    deltas=(
        (-1, 0),
        (1, -1),
        (1, 1),
    ),
    stamp=Player.BLUE,
)

HORSE = Card(
    name="Horse",
    deltas=(
        (-1, 0),
        (0, -1),
        (1, 0),
    ),
    stamp=Player.RED,
)

CRAB = Card(
    name="Crab",
    deltas=(
        (-1, 0),
        (0, -2),
        (0, 2),
    ),
    stamp=Player.BLUE,
)

OX = Card(
    name="Ox",
    deltas=(
        (-1, 0),
        (0, 1),
        (1, 0),
    ),
    stamp=Player.BLUE,
)

FROG = Card(
    name="Frog",
    deltas=(
        (-1, -1),
        (0, -2),
        (1, 1),
    ),
    stamp=Player.RED,
)

BOAR = Card(
    name="Boar",
    deltas=(
        (-1, 0),
        (0, -1),
        (0, 1),
    ),
    stamp=Player.RED,
)

COBRA = Card(
    name="Cobra",
    deltas=(
        (-1, 1),
        (0, -1),
        (1, 1),
    ),
    stamp=Player.RED,
)

TIGER = Card(
    name="Tiger",
    deltas=(
        (-2, 0),
        (1, 0),
    ),
    stamp=Player.BLUE,
)

EEL = Card(
    name="Eel",
    deltas=(
        (-1, -1),
        (0, 1),
        (1, -1),
    ),
    stamp=Player.BLUE,
)

ALL_CARDS: tuple[Card, ...] = (
    RABBIT, MONKEY, ELEPHANT, DRAGON,
    GOOSE, MANTIS, ROOSTER, CRANE,
    HORSE, CRAB, OX, FROG,
    BOAR, COBRA, TIGER, EEL,
)

CARD_BY_NAME: dict[str, Card] = {c.name: c for c in ALL_CARDS}
