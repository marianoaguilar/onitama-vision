from __future__ import annotations

from onitama.ai.evaluate import EVALUATORS
from onitama.runtime.vision_models import VisionRuntimeConfig, VisionRuntimeState
from onitama.runtime.vision_runtime import VisionGameRuntime
from onitama.cli.prompts import prompt_choice, prompt_int
from onitama.cli.render import format_action, render_state
from onitama.engine.pieces import Player
from onitama.integration.session import SessionOutcome, SessionPhase

VISIBLE_OUTCOMES = {
    SessionOutcome.BOOTSTRAPPED,
    SessionOutcome.HUMAN_MOVE_ACCEPTED,
    SessionOutcome.AI_ACTION_SELECTED,
    SessionOutcome.AI_EXECUTION_CONFIRMED,
    SessionOutcome.HUMAN_MOVE_REJECTED,
    SessionOutcome.AI_EXECUTION_MISMATCH,
}
FILTERED_WARNING_OUTCOMES = {
    SessionOutcome.HUMAN_MOVE_REJECTED,
    SessionOutcome.AI_EXECUTION_MISMATCH,
}
WARNING_REPORT_THRESHOLD = 3


def prompt_vision_config() -> VisionRuntimeConfig:
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

    return VisionRuntimeConfig(
        human_player=human_player,
        required_repeats=required_repeats,
        ai_depth=ai_depth,
        ai_evaluator=ai_evaluator,
    )


def _print_step(state: VisionRuntimeState) -> None:
    """Print only high-signal session transitions."""
    outcome = state.last_outcome
    if outcome is None or outcome not in VISIBLE_OUTCOMES:
        return

    print("")
    print("-" * 40)
    print(f"[{state.phase.value}]")
    print(f"-> {outcome.value}")

    # Also print the current board
    if outcome in {
        SessionOutcome.BOOTSTRAPPED,
        SessionOutcome.HUMAN_MOVE_ACCEPTED,
        SessionOutcome.AI_EXECUTION_CONFIRMED,
    } and state.current_state is not None:
        print(render_state(state.current_state))
        return

    # Also print the reason
    if outcome is SessionOutcome.HUMAN_MOVE_REJECTED:
        print("Reason: Observed state does not match any legal successor.")
        return

    # Also print the AI action when selected.
    if outcome is SessionOutcome.AI_ACTION_SELECTED and state.current_state is not None and state.ai_action is not None:
        print(f"[AI {state.current_state.to_move.value}] {format_action(state.current_state, state.ai_action)}")


def _reset_warning_tracking() -> tuple[None, int, bool]:
    return None, 0, False


def run() -> None:
    config = prompt_vision_config()
    runtime = VisionGameRuntime(config)
    runtime.start()

    print("\nVision play started.")
    print("Keep the board still while the system stabilizes observations.")
    print("Press Ctrl+C to quit.\n")

    # Track the last printed session phase/outcome.
    last_printed_key: tuple[SessionPhase, SessionOutcome] | None = None

    # Keep the last hard runtime error to avoid printing the same one every loop.
    last_error_message: str | None = None

    # Track repeated warning outcomes from the session flow.
    current_outcome_warning_key: tuple[SessionPhase, SessionOutcome] | None = None
    current_outcome_warning_count = 0
    current_outcome_warning_reported = False

    # Observation warnings are already threshold-filtered by VisionGameRuntime.
    last_observation_warning_kind: str | None = None

    try:
        while True:
            state = runtime.step()

            # --- Handle vision / runtime errors ---
            if state.error_message is not None:
                if state.error_message != last_error_message:
                    print(f"\n[vision error] {state.error_message}")
                    last_error_message = state.error_message
                continue

            last_error_message = None

            if state.observation_kind is not None:
                if state.observation_kind.value != last_observation_warning_kind:
                    print(f"\n[vision warning] {state.observation_kind.value}")
                    last_observation_warning_kind = state.observation_kind.value
                continue

            outcome = state.last_outcome
            if outcome is None:
                continue

            step_key = (state.phase, outcome)

            if outcome in FILTERED_WARNING_OUTCOMES:
                if step_key != current_outcome_warning_key:
                    current_outcome_warning_key = step_key
                    current_outcome_warning_count = 1
                    current_outcome_warning_reported = False
                else:
                    current_outcome_warning_count += 1

                # Only show warnings if they repeat several.
                if (not current_outcome_warning_reported
                    and current_outcome_warning_count >= WARNING_REPORT_THRESHOLD
                    and step_key != last_printed_key):
                    _print_step(state)
                    last_printed_key = step_key
                    last_observation_warning_kind = None
                    current_outcome_warning_reported = True

            elif outcome in VISIBLE_OUTCOMES:
                # Show important non-warning events right away.
                (
                    current_outcome_warning_key,
                    current_outcome_warning_count,
                    current_outcome_warning_reported,
                ) = _reset_warning_tracking()
                if step_key != last_printed_key:
                    _print_step(state)
                    last_printed_key = step_key
                    last_observation_warning_kind = None

            else:
                # Hidden/internal outcomes still reset warning tracking.
                (
                    current_outcome_warning_key,
                    current_outcome_warning_count,
                    current_outcome_warning_reported,
                ) = _reset_warning_tracking()

            # Once the session reaches a terminal state, announce the winner and stop.
            if state.phase is SessionPhase.FINISHED and state.current_state is not None:
                if state.winner_player is not None:
                    print("")
                    print(f"*** WINNER: {state.winner_player.value} ***")
                    print(f"*** REASON: {state.winner_reason} ***")
                break
    except KeyboardInterrupt:
        print("\nStopping vision play.")
    finally:
        runtime.stop()


if __name__ == "__main__":
    run()
