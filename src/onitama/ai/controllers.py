from __future__ import annotations

from dataclasses import dataclass, field

from onitama.ai.agent import choose_action
from onitama.ai.evaluate import get_evaluator
from onitama.ai.types import Evaluator
from onitama.engine.controllers import Controller
from onitama.engine.rules import Action
from onitama.engine.state import GameState


@dataclass
class AIController(Controller):
    depth: int
    evaluator_name: str
    q_depth: int = 2
    # TT support is implemented in the search layer, but disabled in the final agent.
    # tt: TranspositionTable = field(default_factory=dict)
    evaluator: Evaluator = field(init=False)

    def __post_init__(self) -> None:
        self.evaluator = get_evaluator(self.evaluator_name)

    def select_action(self, state: GameState) -> Action:
        action = choose_action(
            state,
            depth=self.depth,
            evaluator=self.evaluator,
            q_depth=self.q_depth,
            use_tt=False,
            use_iterative_deepening=False,
            aspiration_window=None,
            use_move_ordering=True,
        )
        if action is None:
            raise RuntimeError("AIController was asked to move in a terminal state.")
        return action
