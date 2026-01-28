from __future__ import annotations

from onitama.moves import Move
from onitama.pieces import Player, PieceType
from onitama.state import GameState
from copy import deepcopy


def _in_bounds(r: int, c: int) -> bool:
    return 0 <= r < 5 and 0 <= c < 5


def _find_master(state: GameState, player: Player) -> tuple[int, int] | None:
    for r in range(5):
        for c in range(5):
            p = state.board[r][c]
            if p is None:
                continue
            if p.owner == player and p.kind is PieceType.MASTER:
                return (r, c)
    return None


def winner(state: GameState) -> Player | None:
    """
    Return the winning player if the position is terminal, else None.

    Win conditions:
    - Way of the Stone: opponent master is captured (missing from board)
    - Way of the Stream: your master reaches opponent temple
        * RED wins if red master is at (0, 2)
        * BLUE wins if blue master is at (4, 2)
    """
    red_master = _find_master(state, Player.RED)
    blue_master = _find_master(state, Player.BLUE)

    # Stone (capture master)
    if red_master is None and blue_master is not None:
        return Player.BLUE
    if blue_master is None and red_master is not None:
        return Player.RED

    # If both missing (invalid), treat as no winner
    if red_master is None or blue_master is None:
        return None

    # Stream (reach temple)
    if red_master == (0, 2):
        return Player.RED
    if blue_master == (4, 2):
        return Player.BLUE

    return None


def is_terminal(state: GameState) -> bool:
    return winner(state) is not None



def generate_legal_moves(state: GameState) -> list[Move]:
    """
    Generate all legal moves for the current player.

    Rules implemented:
    - Win conditions
    - A move must stay inside the 5x5 board.
    - A move cannot land on a square occupied by your own piece.
    """

    if is_terminal(state):
        return []

    player = state.to_move
    cards = state.red_cards if player is Player.RED else state.blue_cards

    legal_moves: list[Move] = []

    for r in range(5):
        for c in range(5):
            piece = state.board[r][c]
            if piece is None:
                continue
            if piece.owner != player:
                continue

            for card_index, card in enumerate(cards):
                for dr, dc in card.deltas_for(player):
                    rr = r + dr
                    cc = c + dc
                    if not _in_bounds(rr, cc):
                        continue

                    target = state.board[rr][cc]
                    if target is not None and target.owner == player:
                        continue

                    legal_moves.append(
                        Move(from_pos=(r, c), to_pos=(rr, cc), card_index=card_index)
                    )


    return legal_moves



def apply_move(state: GameState, move: Move) -> GameState:
    """
    Apply a move and return a NEW GameState.

    Implemented:
    - Move the piece from from_pos to to_pos (captures by replacement)
    - Switch turn to the opponent
    - Swap cards according to Onitama rules:
        * the used card becomes the new side card
        * the old side card replaces the used card in the player's hand
    """
    fr, fc = move.from_pos
    tr, tc = move.to_pos

    piece = state.board[fr][fc]
    if piece is None:
        raise ValueError("Invalid move: no piece at from_pos")

    if piece.owner != state.to_move:
        raise ValueError("Invalid move: piece does not belong to player to move")

    target = state.board[tr][tc]
    if target is not None and target.owner == state.to_move:
        raise ValueError("Invalid move: cannot capture your own piece")

    # Copy board so we don't mutate the previous state
    new_board = deepcopy(state.board)
    new_board[fr][fc] = None
    new_board[tr][tc] = piece

    player = state.to_move
    opponent = player.opponent()

    if player is Player.RED:
        active = state.red_cards
        other_active = state.blue_cards
    else:
        active = state.blue_cards
        other_active = state.red_cards

    used_card = active[move.card_index]
    incoming_card = state.side_card

    # Replace the used card with the incoming side card
    if move.card_index == 0:
        new_active = (incoming_card, active[1])
    elif move.card_index == 1:
        new_active = (active[0], incoming_card)
    else:
        raise ValueError("Invalid move: card_index must be 0 or 1")

    new_side = used_card

    # Build next state
    if player is Player.RED:
        return GameState(
            board=new_board,
            to_move=opponent,
            red_cards=new_active,
            blue_cards=other_active,
            side_card=new_side,
        )

    return GameState(
        board=new_board,
        to_move=opponent,
        red_cards=other_active,
        blue_cards=new_active,
        side_card=new_side,
    )

