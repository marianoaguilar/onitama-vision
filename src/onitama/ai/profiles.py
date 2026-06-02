from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AIProfile:
    """User-facing AI difficulty profile."""

    id: str
    label: str
    evaluator_name: str
    depth: int
    q_depth: int


AI_PROFILES: dict[str, AIProfile] = {
    "easy": AIProfile(
        id="easy",
        label="Fácil",
        evaluator_name="v3",
        depth=1,
        q_depth=1,
    ),
    "medium": AIProfile(
        id="medium",
        label="Media",
        evaluator_name="v3",
        depth=3,
        q_depth=2,
    ),
    "hard": AIProfile(
        id="hard",
        label="Difícil",
        evaluator_name="v3",
        depth=5,
        q_depth=2,
    ),
}

DEFAULT_AI_PROFILE_ID = "hard"


def get_ai_profile(profile_id: str) -> AIProfile:
    try:
        return AI_PROFILES[profile_id]
    except KeyError as exc:
        raise ValueError(f"Unknown AI profile '{profile_id}'. Available: {sorted(AI_PROFILES)}") from exc
