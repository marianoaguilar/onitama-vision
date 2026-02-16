from __future__ import annotations

from typing import Callable

from onitama.engine.rules import Action
from onitama.engine.pieces import Player
from onitama.engine.state import GameState

# Evaluator contract used by the AI:
# - input: game state + player perspective
# - output: integer score
Evaluator = Callable[[GameState, Player], int]

# Transposition table flags:
TT_EXACT = 0    # Exact score for this node/depth
TT_LOWER = 1    # Score is a lower bound (fail-high)
TT_UPPER = 2    # Score is an upper bound (fail-low)

# TT entry layout: [depth, value, flag, best_action]
TTEntry = tuple[int, int, int, Action | None]  

# Main TT container: board state -> cached search entry.
TranspositionTable = dict[GameState, TTEntry]
