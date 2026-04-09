from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from onitama.engine.rules import Action, apply_action, generate_legal_actions, is_terminal
from onitama.engine.state import GameState


class SyncStatus(str, Enum):
    ACCEPTED = "accepted"
    UNCHANGED = "unchanged"
    REJECTED = "rejected"
    TERMINAL_PREVIOUS = "terminal_previous"


@dataclass(frozen=True)
class MatchedSuccessor:
    """One legal transition from a previous state."""

    action: Action
    state: GameState


@dataclass(frozen=True)
class SyncResult:
    """Result of checking an observed state against legal successors."""

    status: SyncStatus
    accepted: bool
    match_count: int
    matched_action: Action | None = None
    reason: str | None = None


def legal_successors(previous_state: GameState) -> tuple[MatchedSuccessor, ...]:
    """Enumerate all legal one-ply successors from the previous state."""
    if is_terminal(previous_state):
        return ()

    return tuple(
        MatchedSuccessor(action=action, state=apply_action(previous_state, action))
        for action in generate_legal_actions(previous_state)
    )


def match_observed_state(previous_state: GameState, observed_state: GameState) -> SyncResult:
    """
    Validate an observed state against the legal successors of `previous_state`.

    Possible outcomes are represented by `SyncStatus`.
    """
    if is_terminal(previous_state):
        return SyncResult(
            status=SyncStatus.TERMINAL_PREVIOUS,
            accepted=False,
            match_count=0,
            reason="Previous state is terminal; no further transition can be accepted.",
        )

    if previous_state == observed_state:
        return SyncResult(
            status=SyncStatus.UNCHANGED,
            accepted=False,
            match_count=0,
            reason="Observed state is identical to the previous confirmed state.",
        )

    matches = [
        successor
        for successor in legal_successors(previous_state)
        if successor.state == observed_state
    ]

    if len(matches) == 1:
        match = matches[0]
        return SyncResult(
            status=SyncStatus.ACCEPTED,
            accepted=True,
            match_count=1,
            matched_action=match.action,
        )

    return SyncResult(
        status=SyncStatus.REJECTED,
        accepted=False,
        match_count=len(matches),
        reason="Observed state does not match any legal successor.",
    )
