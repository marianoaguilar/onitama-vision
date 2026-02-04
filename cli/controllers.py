from __future__ import annotations

from dataclasses import dataclass

from cli.render import format_action
from onitama.controllers import Controller
from onitama.rules import Action, generate_legal_actions
from onitama.state import GameState


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


@dataclass
class HumanCLIController(Controller):
    """
    Human controller via terminal input.

    It signals quit/restart by raising RuntimeError("quit") / RuntimeError("restart").
    The outer game loop catches these and acts accordingly.
    """

    def select_action(self, state: GameState) -> Action:
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
            return self.select_action(state)

        if choice == "quit":
            raise RuntimeError("quit")

        if choice == "restart":
            raise RuntimeError("restart")

        return actions[choice - 1]
