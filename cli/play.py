from onitama.state import GameState
from cli.render import render_board


def main() -> None:
    state = GameState.initial()
    print(render_board(state))


if __name__ == "__main__":
    main()

