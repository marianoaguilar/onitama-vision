from __future__ import annotations

from dataclasses import dataclass

from onitama.cli.render import format_action
from onitama.engine.pieces import Player
from onitama.integration.session import SessionOutcome, SessionPhase
from onitama.app.vision_models import VisionRuntimeState


@dataclass(frozen=True)
class StatusView:
    title: str
    detail: str
    tone: str = "neutral"


INITIAL_STATUS = StatusView(
    title="Preparando...",
    detail="",
)


def player_label(player: Player | None) -> str:
    if player is Player.RED:
        return "Rojo"
    if player is Player.BLUE:
        return "Azul"
    return "-"


def build_status_view(state: VisionRuntimeState | None) -> StatusView:
    """Translate raw runtime state into one clear UI message."""
    if state is None:
        return INITIAL_STATUS

    if state.error_message:
        return StatusView(
            title="Error de vision",
            detail=state.error_message,
            tone="error",
        )

    if state.phase is SessionPhase.FINISHED:
        winner = player_label(state.winner_player)
        reason = state.winner_reason or "partida terminada"
        return StatusView(
            title=f"Partida terminada: gana {winner}",
            detail=reason,
            tone="success",
        )

    if state.phase is SessionPhase.BOOTSTRAP:
        return StatusView(
            title="Leyendo posicion inicial",
            detail="Manten el tablero quieto hasta confirmar una posicion estable",
        )

    if state.phase is SessionPhase.WAITING_HUMAN_MOVE:
        if state.last_outcome is SessionOutcome.HUMAN_MOVE_REJECTED:
            return StatusView(
                title="Movimiento rechazado",
                detail="La posicion fisica no coincide con un movimiento legal",
                tone="warning",
            )
        if state.last_outcome is SessionOutcome.AI_EXECUTION_CONFIRMED:
            return StatusView(
                title="Movimiento de la IA confirmado",
                detail="Turno del humano",
                tone="success",
            )
        return StatusView(
            title="Turno del humano",
            detail="Haz tu movimiento en el tablero fisico",
        )

    if state.phase is SessionPhase.READY_FOR_AI:
        return StatusView(
            title="Turno de la IA",
            detail="La IA esta eligiendo un movimiento",
        )

    if state.phase is SessionPhase.WAITING_AI_EXECUTION:
        action_text = "el movimiento seleccionado"
        if state.current_state is not None and state.ai_action is not None:
            action_text = format_action(state.current_state, state.ai_action).replace("PASS", "PASA")

        if state.last_outcome is SessionOutcome.AI_EXECUTION_MISMATCH:
            return StatusView(
                title="El movimiento de la IA no coincide",
                detail=f"Esperado: {action_text}",
                tone="warning",
            )

        return StatusView(
            title="Ejecuta el movimiento de la IA",
            detail=f"Movimiento: {action_text}.",
        )

    return StatusView(
        title=state.phase.value,
        detail=state.last_outcome.value if state.last_outcome is not None else "",
    )
