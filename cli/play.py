from __future__ import annotations

from dataclasses import dataclass

from ai.controllers import AIController
from ai.evaluate import EVALUATORS
from cli.controllers import HumanCLIController
from cli.render import render_state, format_action
from onitama.controllers import Controller
from onitama.pieces import Player
from onitama.rules import apply_action, winner
from onitama.state import GameState


@dataclass(frozen=True)
class GameConfig:
    red_controller: Controller
    blue_controller: Controller


def _prompt_int(prompt: str, default: int, lo: int, hi: int) -> int:
    while True:
        raw = input(f"\n{prompt} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            n = int(raw)
        except ValueError:
            print("Please enter a valid integer.")
            continue
        if n < lo or n > hi:
            print(f"Please enter a number between {lo} and {hi}.")
            continue
        return n


def _prompt_choice(prompt: str, options: list[str], default_index: int = 0) -> str:
    assert options, "Options must not be empty."
    while True:
        print(prompt)
        for i, opt in enumerate(options, start=1):
            mark = " (default)" if (i - 1) == default_index else ""
            print(f"  {i}) {opt}{mark}")

        raw = input("Select an option: ").strip()
        if raw == "":
            return options[default_index]

        try:
            idx = int(raw)
        except ValueError:
            print("Please enter a number.")
            continue

        if idx < 1 or idx > len(options):
            print(f"Please enter a number between 1 and {len(options)}.")
            continue

        return options[idx - 1]


def _prompt_ai_settings(player_name: str) -> AIController:
    depth = _prompt_int(f"{player_name} AI depth", default=5, lo=1, hi=8)

    eval_names = sorted(EVALUATORS.keys())
    eval_name = _prompt_choice(
        f"{player_name} evaluator:",
        options=eval_names,
        default_index=2,
    )

    return AIController(depth=depth, evaluator_name=eval_name)


def prompt_game_config() -> GameConfig:
    print("\n=== Onitama CLI ===\n")
    mode = _prompt_choice(
        "Game mode:",
        options=[
            "Human vs Human",
            "Human (RED) vs AI (BLUE)",
            "AI (RED) vs Human (BLUE)",
            "AI vs AI",
        ],
        default_index=1,
    )

    red_is_ai = mode in ("AI (RED) vs Human (BLUE)", "AI vs AI")
    blue_is_ai = mode in ("Human (RED) vs AI (BLUE)", "AI vs AI")

    red_controller: Controller = HumanCLIController()
    blue_controller: Controller = HumanCLIController()

    if red_is_ai:
        red_controller = _prompt_ai_settings("RED")
    if blue_is_ai:
        blue_controller = _prompt_ai_settings("BLUE")

    print("\nConfiguration:")
    if isinstance(red_controller, AIController):
        print(f"  RED : AI | depth={red_controller.depth} | eval={red_controller.evaluator_name}")
    else:
        print("  RED : Human")

    if isinstance(blue_controller, AIController):
        print(f"  BLUE: AI | depth={blue_controller.depth} | eval={blue_controller.evaluator_name}")
    else:
        print("  BLUE: Human")

    input("\nPress Enter to start the game...")

    return GameConfig(red_controller=red_controller, blue_controller=blue_controller)


def _controller_for(config: GameConfig, player: Player) -> Controller:
    return config.red_controller if player == Player.RED else config.blue_controller


def main(config: GameConfig, seed: int | None = None) -> str:
    """
    Runs a single game session.
    Returns:
      'quit' when the user quits or the game ends,
      'restart' when the user requests restart.
    """
    state = GameState.initial(seed=seed)

    while True:
        print(render_state(state))

        outcome = winner(state)
        if outcome is not None:
            w, reason = outcome
            print("")
            print(f"*** WINNER: {w.value} ***")
            print(f"*** REASON: {reason} ***")
            return "quit"

        controller = _controller_for(config, state.to_move)

        try:
            action = controller.select_action(state)
        except RuntimeError as e:
            if str(e) == "quit":
                print("Bye!")
                return "quit"
            if str(e) == "restart":
                print("Restarting...\n")
                return "restart"
            raise

        # If it's an AI, print what it played (nice for demo/debug)
        if isinstance(controller, AIController):
            print(f"\n[AI {state.to_move.value}] {format_action(state, action)}")
            state = apply_action(state, action)
            print("\n" + "-" * 60 + "\n")
            input("Press Enter to continue...")
        else:
            state = apply_action(state, action)
            print("\n" + "-" * 60 + "\n")


def run() -> None:
    """
    Wrapper to allow restarting without exiting the program.
    """
    seed = None

    # Menu once; keep the same config across restarts.
    config = prompt_game_config()

    while True:
        result = main(config=config, seed=seed)
        if result == "restart":
            seed = None
            continue
        break


if __name__ == "__main__":
    run()
