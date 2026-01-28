from __future__ import annotations

from onitama.rules import apply_move, generate_legal_moves, winner
from onitama.state import GameState
from cli.render import render_state, format_move


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

        moves = generate_legal_moves(state)
        if not moves:
            # In theory this shouldn't happen in Onitama before winner(),
            # but it's a safe fallback.
            print("No legal moves available.")
            break

        print("\nLegal moves:")
        for i, mv in enumerate(moves, start=1):
            print(f"{i:2d}) {format_move(state, mv)}")

        choice = _prompt_int(f"\nChoose a move (1-{len(moves)}): ", 1, len(moves))
        move = moves[choice - 1]

        state = apply_move(state, move)
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()
