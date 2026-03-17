import argparse
from pathlib import Path

from onitama.vision.board import VisionBoard
from onitama.vision.card_classifier import CardClassificationResult
from onitama.vision.snapshot import VisionSnapshot


def _parse_card_pair(value: str) -> tuple[str, str]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError("Card pair must be provided as 'CardA,CardB'.")
    return (parts[0], parts[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a vision snapshot JSON from a vision board JSON + cards.")
    parser.add_argument("--board-path", default="data/vision/auto_piece_board.json")
    parser.add_argument("--red-cards", default="Tiger,Horse")
    parser.add_argument("--blue-cards", default="Crab,Boar")
    parser.add_argument("--side-card", default="Rabbit")
    parser.add_argument(
        "--card-predictions-path",
        default="",
        help="Optional JSON saved from the live card classifier. If provided, inferred cards override manual card args.",
    )
    parser.add_argument("--out-path", default="data/vision/snapshot.json")
    args = parser.parse_args()

    board = VisionBoard.load_json(Path(args.board_path))
    red_cards = _parse_card_pair(args.red_cards)
    blue_cards = _parse_card_pair(args.blue_cards)
    side_card = args.side_card.strip()

    if args.card_predictions_path:
        card_result = CardClassificationResult.load_json(Path(args.card_predictions_path))
        snapshot = VisionSnapshot.from_board_and_cards(
            board=board,
            card_result=card_result,
        )
    else:
        snapshot = VisionSnapshot(
            board=board,
            red_cards=red_cards,
            blue_cards=blue_cards,
            side_card=side_card,
        )
    out_path = Path(args.out_path)
    snapshot.save_json(out_path)
    print(f"Saved snapshot to: {out_path}")


if __name__ == "__main__":
    main()
