from __future__ import annotations

from dataclasses import dataclass

from onitama.errors import VisionObservationKind
from onitama.engine.formatting import format_action
from onitama.engine.pieces import Player
from onitama.integration.session import SessionOutcome, SessionPhase
from onitama.runtime.vision_models import VisionRuntimeState


OBSERVATION_DETAILS = {
    VisionObservationKind.INVALID_BOARD_PIECE_COUNT: "Revisa que las piezas estén bien colocadas y visibles.",
    VisionObservationKind.LOW_CONFIDENCE_CARD: "Revisa que las cartas estén bien colocadas y visibles.",
    VisionObservationKind.GENERIC: "La observación visual no es válida.",
}


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


def observation_detail(kind: VisionObservationKind) -> str:
    return OBSERVATION_DETAILS.get(kind, OBSERVATION_DETAILS[VisionObservationKind.GENERIC])


def build_status_view(state: VisionRuntimeState | None) -> StatusView:
    """Translate raw runtime state into one clear UI message."""
    if state is None:
        return INITIAL_STATUS

    if state.error_message:
        return StatusView(
            title="Error de visión",
            detail=state.error_message,
            tone="error",
        )

    if state.phase is SessionPhase.FINISHED:
        winner = player_label(state.winner_player)
        reason = state.winner_reason or "partida terminada"
        return StatusView(
            title=f"¡Partida terminada: gana {winner}!",
            detail=reason,
            tone="success",
        )

    if state.phase is SessionPhase.BOOTSTRAP:
        if state.observation_kind is not None:
            return StatusView(
                title="No se puede confirmar la posición inicial",
                detail=observation_detail(state.observation_kind),
                tone="warning",
            )
        return StatusView(
            title="Leyendo posición inicial",
            detail="Mantén el tablero quieto hasta confirmar una posición estable",
        )

    if state.phase is SessionPhase.WAITING_HUMAN_MOVE:
        if state.last_outcome is SessionOutcome.HUMAN_MOVE_REJECTED:
            return StatusView(
                title="Movimiento rechazado",
                detail="La posición física no coincide con un movimiento legal",
                tone="warning",
            )
        if state.last_outcome is SessionOutcome.AI_EXECUTION_CONFIRMED:
            return StatusView(
                title="Movimiento de la IA confirmado",
                detail="Turno del humano",
                tone="success",
            )
        if state.observation_kind is not None:
            return StatusView(
                title="Lectura inválida",
                detail=observation_detail(state.observation_kind),
                tone="warning",
            )
        return StatusView(
            title="Turno del humano",
            detail="Haz tu movimiento en el tablero físico",
        )

    if state.phase is SessionPhase.READY_FOR_AI:
        return StatusView(
            title="Turno de la IA",
            detail="La IA está eligiendo un movimiento",
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
        if state.observation_kind is not None:
            return StatusView(
                title="Lectura inválida",
                detail=observation_detail(state.observation_kind),
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
