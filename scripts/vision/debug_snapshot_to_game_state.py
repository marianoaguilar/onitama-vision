import argparse
from pathlib import Path

from onitama.cli.render import render_state
from onitama.vision.bridge import snapshot_to_game_state
from onitama.vision.snapshot import VisionSnapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and print a GameState from a vision snapshot JSON.")
    parser.add_argument("--snapshot-path", default="assets/vision/snapshot.json")
    args = parser.parse_args()

    snapshot = VisionSnapshot.load_json(Path(args.snapshot_path))
    game_state = snapshot_to_game_state(snapshot)
    print(render_state(game_state))


if __name__ == "__main__":
    main()
