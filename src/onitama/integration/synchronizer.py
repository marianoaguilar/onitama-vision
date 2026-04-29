from __future__ import annotations

from enum import Enum

from onitama.app.errors import VisionInternalError
from onitama.engine.rules import apply_action, generate_legal_actions, is_terminal
from onitama.engine.state import GameState


class SyncStatus(str, Enum):
    ACCEPTED = "accepted"
    UNCHANGED = "unchanged"
    REJECTED = "rejected"


def legal_successors(previous_state: GameState) -> tuple[GameState, ...]:
    """Enumerate all legal one-ply successors from the previous state."""
    if is_terminal(previous_state):
        return ()

    return tuple(
        apply_action(previous_state, action)
        for action in generate_legal_actions(previous_state)
    )


def match_observed_state(previous_state: GameState, observed_state: GameState) -> SyncStatus:
    """
    Validate an observed state against the legal successors of `previous_state`.

    Possible outcomes are represented by `SyncStatus`.
    """
    if is_terminal(previous_state):
        raise VisionInternalError("previous_state must be non-terminal.")

    if previous_state == observed_state:
        return SyncStatus.UNCHANGED

    if any(successor == observed_state for successor in legal_successors(previous_state)):
        return SyncStatus.ACCEPTED

    return SyncStatus.REJECTED
