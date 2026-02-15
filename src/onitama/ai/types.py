from __future__ import annotations

from typing import Callable

from onitama.engine.rules import Action
from onitama.engine.pieces import Player
from onitama.engine.state import GameState

Evaluator = Callable[[GameState, Player], int]

TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2

TTEntry = tuple[int, int, int, Action | None]  # (depth, value, flag, best_action)
TranspositionTable = dict[GameState, TTEntry]
