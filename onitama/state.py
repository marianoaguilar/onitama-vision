from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from onitama.pieces import Piece, PieceType, Player


Board = list[list[Optional[Piece]]]


@dataclass(frozen=True)
class GameState:
    """
    Minimal game state for Onitama.

    - board: 5x5 grid. Each cell is either None (empty) or a Piece.
    - to_move: which player must move next.
    """

    board: Board
    to_move: Player

    @staticmethod
    def empty(to_move: Player = Player.RED) -> GameState:
        """Create an empty 5x5 board."""
        board: Board = [[None for _ in range(5)] for _ in range(5)]
        return GameState(board=board, to_move=to_move)

    @staticmethod
    def initial(to_move: Player = Player.RED) -> GameState:
        """
        Standard Onitama initial position:

        Top row (row 0): BLUE students, BLUE master in the center.
        Bottom row (row 4): RED students, RED master in the center.
        """
        state = GameState.empty(to_move=to_move)

        # Row 0: BLUE pieces
        for c in range(5):
            state.board[0][c] = Piece(owner=Player.BLUE, kind=PieceType.STUDENT)
        state.board[0][2] = Piece(owner=Player.BLUE, kind=PieceType.MASTER)

        # Row 4: RED pieces
        for c in range(5):
            state.board[4][c] = Piece(owner=Player.RED, kind=PieceType.STUDENT)
        state.board[4][2] = Piece(owner=Player.RED, kind=PieceType.MASTER)

        return state
