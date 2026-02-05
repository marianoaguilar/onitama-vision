from __future__ import annotations

from dataclasses import dataclass, field

from ai.agent import choose_action
from ai.evaluate import get_evaluator
from ai.types import Evaluator, TranspositionTable
from onitama.controllers import Controller
from onitama.rules import Action
from onitama.state import GameState


@dataclass
class AIController(Controller):
    depth: int
    evaluator_name: str
    tt: TranspositionTable = field(default_factory=dict)

    def select_action(self, state: GameState) -> Action:
        evaluator: Evaluator = get_evaluator(self.evaluator_name)
        action = choose_action(state, depth=self.depth, evaluator=evaluator, tt=self.tt)
        assert action is not None, "AIController was asked to move in a terminal state."
        return action
