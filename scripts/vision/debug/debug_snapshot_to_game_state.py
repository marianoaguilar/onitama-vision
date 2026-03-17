import argparse
from pathlib import Path

from onitama.cli.render import render_state
from onitama.engine.pieces import Player
from onitama.vision.snapshot import VisionSnapshot


def _parse_player(value: str) -> Player:
    upper = value.strip().upper()
    if upper == "RED":
        return Player.RED
    if upper == "BLUE":
        return Player.BLUE
    raise ValueError("to-move must be RED or BLUE.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and print a GameState from a vision snapshot JSON.")
    parser.add_argument("--snapshot-path", default="data/vision/snapshot.json")
    parser.add_argument("--to-move", default="RED")
    args = parser.parse_args()

    snapshot = VisionSnapshot.load_json(Path(args.snapshot_path))
    game_state = snapshot.to_game_state(to_move=_parse_player(args.to_move))
    print(render_state(game_state))


if __name__ == "__main__":
    main()
