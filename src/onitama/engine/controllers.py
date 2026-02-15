from __future__ import annotations

from typing import Protocol

from onitama.engine.rules import Action
from onitama.engine.state import GameState


class Controller(Protocol):
    """
    A controller decides an action given the current state.
    This keeps the game loop decoupled from CLI / AI / vision.
    """

    def select_action(self, state: GameState) -> Action:
        raise NotImplementedError
