import argparse
from pathlib import Path

from onitama.engine.pieces import Player
from onitama.vision.board import VisionBoard
from onitama.vision.snapshot import VisionSnapshot


def _parse_player(value: str) -> Player:
    upper = value.strip().upper()
    if upper == "RED":
        return Player.RED
    if upper == "BLUE":
        return Player.BLUE
    raise ValueError("to-move must be RED or BLUE.")


def _parse_card_pair(value: str) -> tuple[str, str]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError("Card pair must be provided as 'CardA,CardB'.")
    return (parts[0], parts[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a vision snapshot JSON from a vision board JSON + turn/cards.")
    parser.add_argument("--board-path", default="data/vision/manual_piece_board.json")
    parser.add_argument("--to-move", default="RED")
    parser.add_argument("--red-cards", default="Tiger,Horse")
    parser.add_argument("--blue-cards", default="Crab,Boar")
    parser.add_argument("--side-card", default="Rabbit")
    parser.add_argument("--out-path", default="data/vision/snapshot.json")
    args = parser.parse_args()

    board = VisionBoard.load_json(Path(args.board_path))
    snapshot = VisionSnapshot(
        board=board,
        to_move=_parse_player(args.to_move),
        red_cards=_parse_card_pair(args.red_cards),
        blue_cards=_parse_card_pair(args.blue_cards),
        side_card=args.side_card.strip(),
    )
    out_path = Path(args.out_path)
    snapshot.save_json(out_path)
    print(f"Saved snapshot to: {out_path}")


if __name__ == "__main__":
    main()
