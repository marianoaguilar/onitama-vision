from __future__ import annotations

from dataclasses import dataclass

from onitama.ai.controllers import AIController
from onitama.ai.evaluate import EVALUATORS
from onitama.cli.prompts import prompt_choice, prompt_int
from onitama.cli.render import format_action, render_state
from onitama.engine.cards import CARD_BY_NAME
from onitama.engine.pieces import Player
from onitama.engine.rules import winner
from onitama.engine.state import GameState
from onitama.integration.session import SessionOutcome, SessionPhase, SessionStepResult, VisionGameSession
from onitama.integration.stabilizer import StateStabilizer
from onitama.vision.snapshot import VisionSnapshot
from onitama.vision.vision_pipeline import VisionPipeline

VISIBLE_OUTCOMES = {
    SessionOutcome.BOOTSTRAPPED,
    SessionOutcome.HUMAN_MOVE_ACCEPTED,
    SessionOutcome.AI_ACTION_SELECTED,
    SessionOutcome.AI_EXECUTION_CONFIRMED,
    SessionOutcome.HUMAN_MOVE_REJECTED,
    SessionOutcome.AI_EXECUTION_MISMATCH,
    SessionOutcome.AI_UNAVAILABLE,
}
FILTERED_WARNING_OUTCOMES = {
    SessionOutcome.HUMAN_MOVE_REJECTED,
    SessionOutcome.AI_EXECUTION_MISMATCH,
}
WARNING_REPORT_THRESHOLD = 2
VISION_ERROR_REPORT_THRESHOLD = 3


@dataclass(frozen=True)
class VisionPlayConfig:
    human_player: Player
    required_repeats: int
    ai_depth: int
    ai_evaluator: str


def prompt_vision_config() -> VisionPlayConfig:
    print("\n=== Onitama Vision Play ===\n")

    required_repeats = prompt_int("Stable repeats required", default=4, lo=1, hi=10)

    human_choice = prompt_choice(
        "\nHuman side:",
        options=["RED", "BLUE"],
        default_index=0,
    )
    human_player = Player.RED if human_choice == "RED" else Player.BLUE

    ai_depth = prompt_int("AI depth", default=5, lo=1, hi=8)
    eval_names = sorted(EVALUATORS.keys())
    default_eval_index = min(2, len(eval_names) - 1)
    ai_evaluator = prompt_choice(
        "AI evaluator:",
        options=eval_names,
        default_index=default_eval_index,
    )

    print("\nConfiguration:")
    print(f"  Human player : {human_player.value}")
    print(f"  AI player    : {human_player.opponent().value}")
    print(f"  Stable reps  : {required_repeats}")
    print(f"  AI depth     : {ai_depth}")
    print(f"  AI evaluator : {ai_evaluator}")
    input("\nPress Enter to start vision play...")

    return VisionPlayConfig(
        human_player=human_player,
        required_repeats=required_repeats,
        ai_depth=ai_depth,
        ai_evaluator=ai_evaluator,
    )


def _initial_player_from_snapshot(snapshot: VisionSnapshot) -> Player:
    """Infer the initial player to move from the observed side card."""
    side_card = CARD_BY_NAME.get(snapshot.side_card)
    if side_card is None:
        raise ValueError(f"Unknown side card from snapshot: {snapshot.side_card!r}")
    return side_card.stamp


def _state_from_snapshot_for_session(snapshot: VisionSnapshot, session: VisionGameSession) -> GameState:
    """
    Convert a snapshot into the most plausible GameState for the current session phase.

    The vision layer does not observe `to_move` directly, so we infer it from:
    - the side card during bootstrap,
    - the current confirmed state while waiting for a human move,
    - the current/expected states while waiting for AI execution.
    """
    if session.current_state is None or session.phase is SessionPhase.BOOTSTRAP:
        # At startup, the side card is the only reliable source for to_move.
        return snapshot.to_game_state(_initial_player_from_snapshot(snapshot))

    # First assume the board still reflects the currently confirmed turn.
    current_candidate = snapshot.to_game_state(session.current_state.to_move)

    if session.phase is SessionPhase.WAITING_HUMAN_MOVE:
        if current_candidate == session.current_state:
            return current_candidate
        # If the board changed, the human move likely consumed the turn.
        return snapshot.to_game_state(session.current_state.to_move.opponent())

    if session.phase is SessionPhase.WAITING_AI_EXECUTION:
        if current_candidate == session.current_state:
            return current_candidate

        if session.expected_state is None:
            raise ValueError("WAITING_AI_EXECUTION requires an expected_state.")
        # After the AI move, interpret the snapshot with the expected next player.
        return snapshot.to_game_state(session.expected_state.to_move)

    return current_candidate


def _print_step(step: SessionStepResult) -> None:
    """Print only high-signal session transitions."""
    if step.outcome not in VISIBLE_OUTCOMES:
        return

    print("")
    print("-" * 40)
    print(f"[{step.phase.value}]")
    print(f"-> {step.outcome.value}")

    # Also print the current board
    if step.outcome in {
        SessionOutcome.BOOTSTRAPPED,
        SessionOutcome.HUMAN_MOVE_ACCEPTED,
        SessionOutcome.AI_EXECUTION_CONFIRMED,
    } and step.current_state is not None:
        print(render_state(step.current_state))
        return

    # Also print the reason 
    if step.outcome is SessionOutcome.HUMAN_MOVE_REJECTED and step.sync_result is not None and step.sync_result.reason:
        print(f"Reason: {step.sync_result.reason}")
        return

    # Also print the AI action when selected.
    if step.outcome is SessionOutcome.AI_ACTION_SELECTED and step.current_state is not None and step.ai_action is not None:
        print(f"[AI {step.current_state.to_move.value}] {format_action(step.current_state, step.ai_action)}")


def _reset_warning_tracking() -> tuple[None, int, bool]:
    return None, 0, False


def _build_session(config: VisionPlayConfig) -> VisionGameSession:
    """Construct the game session with the appropriate AI controller and stabilizer based on the config."""
    ai_controller = AIController(depth=config.ai_depth, evaluator_name=config.ai_evaluator)
    return VisionGameSession(
        human_player=config.human_player,
        ai_player=config.human_player.opponent(),
        ai_controller=ai_controller,
        stabilizer=StateStabilizer(required_repeats=config.required_repeats),
    )


def _open_camera(device: int = 0, width: int = 1280, height: int = 720, fps: int = 30):
    """Open the camera using the same settings as the working live debug scripts."""
    import cv2

    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {device}.")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def run() -> None:
    config = prompt_vision_config()
    pipeline = VisionPipeline()
    session = _build_session(config)

    # Use default camera settings.
    cap = _open_camera()

    print("\nVision play started.")
    print("Keep the board still while the system stabilizes observations.")
    print("Press Ctrl+C to quit.\n")

    # Track the last printed session phase/outcome.
    last_printed_key: tuple[SessionPhase, SessionOutcome] | None = None

    # Track repeated vision errors
    current_error: str | None = None
    current_error_count = 0
    current_error_reported = False

    # Track repeated warnings
    current_warning_key: tuple[SessionPhase, SessionOutcome] | None = None
    current_warning_count = 0
    current_warning_reported = False

    try:
        while True:
            if session.phase is SessionPhase.READY_FOR_AI:
                # Move on without reading a new frame.
                ai_step = session.run_ai_turn()
                step_key = (ai_step.phase, ai_step.outcome)
                current_warning_key, current_warning_count, current_warning_reported = _reset_warning_tracking()
                if ai_step.outcome in VISIBLE_OUTCOMES and step_key != last_printed_key:
                    _print_step(ai_step)
                    last_printed_key = step_key
                continue

            # Read one live frame from the camera.
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Could not read a frame from the camera.")

            try:
                # 1. Vision reconstructs the board/cards snapshot from the frame.
                snapshot = pipeline.snapshot_from_frame(frame)
                # 2. Inject the most plausible "to_move" for this phase.
                observed_state = _state_from_snapshot_for_session(snapshot, session)
                
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                if message != current_error:
                    current_error = message
                    current_error_count = 1
                    current_error_reported = False
                else:
                    current_error_count += 1

                # Only report vision errors if they repeat several times.
                if not current_error_reported and current_error_count >= VISION_ERROR_REPORT_THRESHOLD:
                    print(f"\n[vision error] {message}")
                    current_error_reported = True
                continue
            
            # Reset error tracking on successful observation.
            current_error = None
            current_error_count = 0
            current_error_reported = False

            # Integration decides whether the observation is stable, legal and actionable.
            step = session.process_observation(observed_state)
            step_key = (step.phase, step.outcome)

            if step.outcome in FILTERED_WARNING_OUTCOMES:
                if step_key != current_warning_key:
                    current_warning_key = step_key
                    current_warning_count = 1
                    current_warning_reported = False
                else:
                    current_warning_count += 1

                # Only show warnings if they repeat several.
                if (not current_warning_reported
                    and current_warning_count >= WARNING_REPORT_THRESHOLD
                    and step_key != last_printed_key):
                    _print_step(step)
                    last_printed_key = step_key
                    current_warning_reported = True

            elif step.outcome in VISIBLE_OUTCOMES:
                # Show important non-warning events right away.
                current_warning_key, current_warning_count, current_warning_reported = _reset_warning_tracking()
                if step_key != last_printed_key:
                    _print_step(step)
                    last_printed_key = step_key

            else:
                # Hidden/internal outcomes still reset warning tracking.
                current_warning_key, current_warning_count, current_warning_reported = _reset_warning_tracking()

            # Once the session reaches a terminal state, announce the winner and stop.
            if step.phase is SessionPhase.FINISHED and step.current_state is not None:
                outcome = winner(step.current_state)
                if outcome is not None:
                    print("")
                    print(f"*** WINNER: {outcome[0].value} ***")
                    print(f"*** REASON: {outcome[1]} ***")
                break
    except KeyboardInterrupt:
        print("\nStopping vision play.")
    finally:
        cap.release()


if __name__ == "__main__":
    run()
