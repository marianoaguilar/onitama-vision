from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from onitama.cards import ALL_CARDS, Card
from onitama.pieces import Piece, PieceType, Player

Board = list[list[Optional[Piece]]]
CardPair = tuple[Card, Card]


@dataclass(frozen=True)
class GameState:
    """
    Minimal game state for Onitama.

    - board: 5x5 grid. Each cell is either None (empty) or a Piece.
    - to_move: which player must move next.
    - red_cards / blue_cards: the 2 active cards for each player.
    - side_card: the card to be swapped after a move.
    """

    board: Board
    to_move: Player
    red_cards: CardPair
    blue_cards: CardPair
    side_card: Card

    @staticmethod
    def empty(
        to_move: Player = Player.RED,
        red_cards: Optional[CardPair] = None,
        blue_cards: Optional[CardPair] = None,
        side_card: Optional[Card] = None,
    ) -> GameState:
        """Create an empty 5x5 board (no pieces). Cards can be provided or defaulted."""
        board: Board = [[None for _ in range(5)] for _ in range(5)]

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

        state = GameState.empty(
            to_move=to_move,
            red_cards=red_cards,
            blue_cards=blue_cards,
            side_card=side_card,
        )

        # Row 0: BLUE pieces
        for c in range(5):
            state.board[0][c] = Piece(owner=Player.BLUE, kind=PieceType.STUDENT)
        state.board[0][2] = Piece(owner=Player.BLUE, kind=PieceType.MASTER)

        # Row 4: RED pieces
        for c in range(5):
            state.board[4][c] = Piece(owner=Player.RED, kind=PieceType.STUDENT)
        state.board[4][2] = Piece(owner=Player.RED, kind=PieceType.MASTER)

        return state
