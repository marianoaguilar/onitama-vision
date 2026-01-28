from __future__ import annotations

from onitama.rules import apply_action, generate_legal_actions, winner
from onitama.state import GameState
from cli.render import render_state, format_action


def _prompt_int(prompt: str, lo: int, hi: int) -> int:
    """Prompt until user enters an integer in [lo, hi]."""
    while True:
        raw = input(prompt).strip()
        try:
            n = int(raw)
        except ValueError:
            print("Please enter a valid integer.")
            continue

        if n < lo or n > hi:
            print(f"Please enter a number between {lo} and {hi}.")
            continue

        return n


def main() -> None:
    state = GameState.initial()

    while True:
        w = winner(state)
        if w is not None:
            print("")
            print(f"*** WINNER: {w.value} ***")
            break

        print(render_state(state))

        actions = generate_legal_actions(state)

        print("\nLegal actions:")
        for i, act in enumerate(actions, start=1):
            print(f"{i:2d}) {format_action(state, act)}")

        choice = _prompt_int(f"\nChoose an action (1-{len(actions)}): ", 1, len(actions))
        action = actions[choice - 1]

        state = apply_action(state, action)
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()
