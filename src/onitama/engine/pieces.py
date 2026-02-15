from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Player(Enum):
    RED = "RED"
    BLUE = "BLUE"

    def opponent(self) -> Player:
        return Player.BLUE if self is Player.RED else Player.RED


class PieceType(Enum):
    MASTER = "MASTER"
    STUDENT = "STUDENT"


@dataclass(frozen=True)
class Piece:
    owner: Player
    kind: PieceType
