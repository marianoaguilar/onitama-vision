from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class SearchStats:
    """Counters collected during one choose_action call."""

    nodes: int = 0
    q_nodes: int = 0
    beta_cutoffs: int = 0
    tt_probes: int = 0
    tt_hits: int = 0
    tt_cutoffs: int = 0
    tt_stores: int = 0
    depth_completed: int = 0
    value: int | None = None
