from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from onitama.cards import ALL_CARDS, Card
from onitama.pieces import Piece, PieceType, Player

Board = tuple[tuple[Optional[Piece], ...], ...]
CardPair = tuple[Card, Card]


def _find_master_positions(board: Board) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    red_pos: tuple[int, int] | None = None
    blue_pos: tuple[int, int] | None = None
    for r in range(5):
        for c in range(5):
            p = board[r][c]
            if p is None or p.kind is not PieceType.MASTER:
                continue
            if p.owner is Player.RED:
                red_pos = (r, c)
            else:
                blue_pos = (r, c)
    return red_pos, blue_pos


@dataclass(frozen=True)
class GameState:
    """
    Minimal game state for Onitama.

    - board: 5x5 grid. Each cell is either None (empty) or a Piece.
    - to_move: which player must move next.
    - red_cards / blue_cards: the 2 active cards for each player.
    - side_card: the card to be swapped after a move.
    - red_master_pos / blue_master_pos: cached master positions (None if captured).
    """

    board: Board
    to_move: Player
    red_cards: CardPair
    blue_cards: CardPair
    side_card: Card
    red_master_pos: tuple[int, int] | None = None
    blue_master_pos: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        # Normalize board to an immutable tuple-of-tuples for hashing/caching.
        board = self.board
        if not isinstance(board, tuple) or (board and not isinstance(board[0], tuple)):
            board = tuple(tuple(row) for row in board)
            object.__setattr__(self, "board", board)

        # Fill cached master positions if not provided.
        if self.red_master_pos is None or self.blue_master_pos is None:
            red_pos, blue_pos = _find_master_positions(board)
            if self.red_master_pos is None:
                object.__setattr__(self, "red_master_pos", red_pos)
            if self.blue_master_pos is None:
                object.__setattr__(self, "blue_master_pos", blue_pos)

    @staticmethod
    def empty(
        to_move: Player = Player.RED,
        red_cards: Optional[CardPair] = None,
        blue_cards: Optional[CardPair] = None,
        side_card: Optional[Card] = None,
    ) -> GameState:
        """Create an empty 5x5 board (no pieces). Cards can be provided or defaulted."""
        board = [[None for _ in range(5)] for _ in range(5)]

        # Safe defaults (only used if you explicitly call empty()).
        if red_cards is None:
            red_cards = (ALL_CARDS[0], ALL_CARDS[1])
        if blue_cards is None:
            blue_cards = (ALL_CARDS[2], ALL_CARDS[3])
        if side_card is None:
            side_card = ALL_CARDS[4]

        return GameState(
            board=board,
            to_move=to_move,
            red_cards=red_cards,
            blue_cards=blue_cards,
            side_card=side_card,
            red_master_pos=None,
            blue_master_pos=None,
        )

    @staticmethod
    def initial(seed: Optional[int] = None) -> GameState:
        """
        Standard Onitama initial position + real card setup:

        - Choose 5 distinct cards from the 16.
        - Deal 2 to RED, 2 to BLUE, 1 as SIDE.
        - Starting player is determined by the SIDE card stamp.
        """
        rng = random.Random(seed)
        chosen = rng.sample(list(ALL_CARDS), k=5)

        red_cards: CardPair = (chosen[0], chosen[1])
        blue_cards: CardPair = (chosen[2], chosen[3])
        side_card: Card = chosen[4]

        # Who starts is determined by the side card's stamp.
        to_move = side_card.stamp

        board = [[None for _ in range(5)] for _ in range(5)]

        # Row 0: BLUE pieces
        for c in range(5):
            board[0][c] = Piece(owner=Player.BLUE, kind=PieceType.STUDENT)
        board[0][2] = Piece(owner=Player.BLUE, kind=PieceType.MASTER)

        # Row 4: RED pieces
        for c in range(5):
            board[4][c] = Piece(owner=Player.RED, kind=PieceType.STUDENT)
        board[4][2] = Piece(owner=Player.RED, kind=PieceType.MASTER)

        return GameState(
            board=board,
            to_move=to_move,
            red_cards=red_cards,
            blue_cards=blue_cards,
            side_card=side_card,
            red_master_pos=(4, 2),
            blue_master_pos=(0, 2),
        )
