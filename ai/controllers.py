from __future__ import annotations

from dataclasses import dataclass

from ai.agent import choose_action
from ai.evaluate import Evaluator, get_evaluator
from onitama.controllers import Controller
from onitama.rules import Action
from onitama.state import GameState


@dataclass(frozen=True)
class AIController(Controller):
    depth: int
    evaluator_name: str

    def select_action(self, state: GameState) -> Action:
        evaluator: Evaluator = get_evaluator(self.evaluator_name)
        action = choose_action(state, depth=self.depth, evaluator=evaluator)
        assert action is not None, "AIController was asked to move in a terminal state."
        return action
