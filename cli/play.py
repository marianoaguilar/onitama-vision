from __future__ import annotations

import os

from onitama.rules import apply_action, generate_legal_actions, winner
from onitama.state import GameState
from cli.render import render_state, format_action

from ai.agent import choose_action
from onitama.pieces import Player


# Determine AI mode and depth from environment variables
def _get_ai_mode() -> str:
    return os.environ.get("ONITAMA_AI", "none").strip().lower()


def _get_ai_depth() -> int:
    raw = os.environ.get("ONITAMA_AI_DEPTH", "3").strip()
    try:
        depth = int(raw)
    except ValueError:
        depth = 3
    return max(1, depth)


def _is_ai_turn(player: Player) -> bool:
    mode = _get_ai_mode()
    if mode == "both":
        return True
    if mode == "red":
        return player == Player.RED
    if mode == "blue":
        return player == Player.BLUE
    return False



def _print_help() -> None:
    print("\nCommands:")
    print("  <number>         Choose an action by its index")
    print("  h / help         Show this help")
    print("  q / quit         Quit the game")
    print("  r / restart      Restart the game\n")


def _prompt_command_or_int(prompt: str, lo: int, hi: int) -> int | str:
    """
    Prompt until user enters:
      - an integer in [lo, hi], or
      - a command: help / quit / restart (and their short forms).
    Returns:
      int for an action selection, or str for a command.
    """
    while True:
        raw = input(prompt).strip().lower()

        if raw in ("h", "help"):
            return "help"
        if raw in ("q", "quit"):
            return "quit"
        if raw in ("r", "restart"):
            return "restart"

        try:
            n = int(raw)
        except ValueError:
            print("Invalid input. Enter a number, or 'h' for help.")
            continue

        if n < lo or n > hi:
            print(f"Please enter a number between {lo} and {hi}.")
            continue

        return n


def main(seed: int | None = None) -> str:
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
        
        # AI turn (optional, controlled by env vars)
        if _is_ai_turn(state.to_move):
            depth = _get_ai_depth()
            action = choose_action(state, depth=depth)
            assert action is not None

            print(f"\n[AI {state.to_move.value}] {format_action(state, action)}")
            state = apply_action(state, action)
            print("\n" + "-" * 60 + "\n")
            input("Press Enter to continue...")
            continue
        
        actions = generate_legal_actions(state)

        print("\nLegal actions:")
        for i, act in enumerate(actions, start=1):
            print(f"{i:2d}) {format_action(state, act)}")

        choice = _prompt_command_or_int(
            f"\nChoose an action (1-{len(actions)}) or (h/q/r): ",
            1,
            len(actions),
        )

        if choice == "help":
            _print_help()
            input("Press Enter to continue...")
            continue

        if choice == "quit":
            print("Bye!")
            return "quit"

        if choice == "restart":
            print("Restarting...\n")
            return "restart"

        # choice is an int in [1, len(actions)]
        action = actions[choice - 1]

        state = apply_action(state, action)
        print("\n" + "-" * 60 + "\n")


def run() -> None:
    """
    Wrapper to allow restarting without exiting the program.
    """
    seed = None
    while True:
        result = main(seed=seed)
        if result == "restart":
            # For now, restart with a fresh random setup.
            # Later we can ask for a seed or reuse the same seed.
            seed = None
            continue
        break


if __name__ == "__main__":
    run()
