from __future__ import annotations

from dataclasses import dataclass

from onitama.engine.rules import Action, apply_action, generate_legal_actions, is_terminal
from onitama.engine.state import GameState


@dataclass(frozen=True)
class MatchedSuccessor:
    """One legal transition from a previous state."""

    action: Action
    state: GameState


@dataclass(frozen=True)
class SyncResult:
    """Result of checking an observed state against legal successors."""

    status: str
    accepted: bool
    previous_state: GameState
    observed_state: GameState
    match_count: int
    matched_action: Action | None = None
    matched_state: GameState | None = None
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

    Status values:
    - `accepted`: exactly one legal successor matches the observation.
    - `unchanged`: observation is identical to the previous confirmed state.
    - `rejected`: no legal successor matches the observation.
    - `ambiguous`: more than one legal successor matches the observation.
    - `terminal_previous`: previous state is terminal, so no transition is valid.
    """
    if is_terminal(previous_state):
        return SyncResult(
            status="terminal_previous",
            accepted=False,
            previous_state=previous_state,
            observed_state=observed_state,
            match_count=0,
            reason="Previous state is terminal; no further transition can be accepted.",
        )

    if previous_state == observed_state:
        return SyncResult(
            status="unchanged",
            accepted=False,
            previous_state=previous_state,
            observed_state=observed_state,
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
            status="accepted",
            accepted=True,
            previous_state=previous_state,
            observed_state=observed_state,
            match_count=1,
            matched_action=match.action,
            matched_state=match.state,
        )

    if len(matches) == 0:
        return SyncResult(
            status="rejected",
            accepted=False,
            previous_state=previous_state,
            observed_state=observed_state,
            match_count=0,
            reason="Observed state does not match any legal successor.",
        )

    return SyncResult(
        status="ambiguous",
        accepted=False,
        previous_state=previous_state,
        observed_state=observed_state,
        match_count=len(matches),
        reason="Observed state matches more than one legal successor.",
    )


def infer_action(previous_state: GameState, observed_state: GameState) -> Action | None:
    """Infer the human action when the observed transition has a unique match."""
    result = match_observed_state(previous_state, observed_state)
    return result.matched_action if result.accepted else None
