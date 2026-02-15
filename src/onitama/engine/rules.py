from __future__ import annotations

from onitama.engine.moves import Move, Pass
from onitama.engine.pieces import Player, PieceType, Piece
from onitama.engine.state import GameState, CardPair, Board
from onitama.engine.cards import Card, ALL_CARDS

Action = Move | Pass


def _in_bounds(r: int, c: int) -> bool:
    return 0 <= r < 5 and 0 <= c < 5


def _precompute_moves() -> dict[Player, dict[Card, tuple[tuple[tuple[int, int], ...], ...]]]:
    table: dict[Player, dict[Card, tuple[tuple[tuple[int, int], ...], ...]]] = {
        Player.RED: {},
        Player.BLUE: {},
    }
    for player in Player:
        for card in ALL_CARDS:
            per_pos: list[tuple[tuple[int, int], ...]] = []
            deltas = card.deltas_for(player)
            for r in range(5):
                for c in range(5):
                    targets: list[tuple[int, int]] = []
                    for dr, dc in deltas:
                        rr = r + dr
                        cc = c + dc
                        if _in_bounds(rr, cc):
                            targets.append((rr, cc))
                    per_pos.append(tuple(targets))
            table[player][card] = tuple(per_pos)
    return table


_MOVE_TABLE = _precompute_moves()


def _board_after_move(
    board: Board, fr: int, fc: int, tr: int, tc: int, piece: Piece
) -> Board:
    rows = list(board)
    if fr == tr:
        row = list(rows[fr])
        row[fc] = None
        row[tc] = piece
        rows[fr] = tuple(row)
    else:
        row_from = list(rows[fr])
        row_from[fc] = None
        rows[fr] = tuple(row_from)

        row_to = list(rows[tr])
        row_to[tc] = piece
        rows[tr] = tuple(row_to)
    return tuple(rows)


def winner(state: GameState) -> tuple[Player, str] | None:
    """
    Return the winning player and the reason if the position is terminal, else None.

    Win conditions:
    - Way of the Stone: opponent master is captured (missing from board)
    - Way of the Stream: your master reaches opponent temple
        * RED wins if red master is at (0, 2)
        * BLUE wins if blue master is at (4, 2)
    """
    red_master = state.red_master_pos
    blue_master = state.blue_master_pos

    # Capture master
    if red_master is None and blue_master is not None:
        return (Player.BLUE, "Capture of Master")
    if blue_master is None and red_master is not None:
        return (Player.RED, "Capture of Master")

    # If both missing (invalid), treat as no winner
    if red_master is None or blue_master is None:
        return None

    # Reach temple
    if red_master == (0, 2):
        return (Player.RED, "Reach Temple")
    if blue_master == (4, 2):
        return (Player.BLUE, "Reach Temple")

    return None


def is_terminal(state: GameState) -> bool:
    return winner(state) is not None



def generate_legal_actions(state: GameState) -> list[Action]:
    """
    Generate all legal moves for the current player.

    Rules implemented:
    - Win conditions
    - A move must stay inside the 5x5 board.
    - A move cannot land on a square occupied by your own piece.
    - If no legal piece moves exist, the player must PASS by choosing a card to swap.
    """

    if is_terminal(state):
        return []

    player = state.to_move
    cards = state.red_cards if player is Player.RED else state.blue_cards

    legal_moves: list[Action] = []

    for r in range(5):
        for c in range(5):
            piece = state.board[r][c]
            if piece is None or piece.owner != player:
                continue

            pos_index = r * 5 + c
            for card_index, card in enumerate(cards):
                for rr, cc in _MOVE_TABLE[player][card][pos_index]:
                    target = state.board[rr][cc]
                    if target is not None and target.owner == player:
                        continue

                    legal_moves.append(
                        Move(from_pos=(r, c), to_pos=(rr, cc), card_index=card_index)
                    )
    
    if len(legal_moves) == 0:
        # No legal piece moves, must pass
        return [Pass(0), Pass(1)]


    return legal_moves


def _swap_cards(state: GameState, card_index: int) -> tuple[CardPair, CardPair, Card]:
    """
    Apply the Onitama card swap rule for the current player, returning:
      (new_red_cards, new_blue_cards, new_side_card)
    """
    if card_index not in (0, 1):
        raise ValueError("Invalid action: card_index must be 0 or 1")

    incoming = state.side_card
    player = state.to_move

    if player is Player.RED:
        active = state.red_cards
        used = active[card_index]
        if card_index == 0:
            new_red = (incoming, active[1])
        else:
            new_red = (active[0], incoming)
        return new_red, state.blue_cards, used

    # player is BLUE
    active = state.blue_cards
    used = active[card_index]
    if card_index == 0:
        new_blue = (incoming, active[1])
    else:
        new_blue = (active[0], incoming)

    return state.red_cards, new_blue, used


def apply_action(state: GameState, action: Action) -> GameState:
    """
    Apply an action and return a NEW GameState.

    Implemented:
    - If Move: move the piece (captures by replacement)
    - If Pass: do not move any piece
    - Switch turn to the opponent
    - Swap cards according to Onitama rules
    """
    
    if is_terminal(state):
        raise ValueError("Cannot apply move: game is already terminal")

    player = state.to_move
    opponent = player.opponent()

    # 1) Update board (or keep it unchanged for Pass)
    new_red_master = state.red_master_pos
    new_blue_master = state.blue_master_pos

    if isinstance(action, Pass):
        new_board = state.board
        chosen_index = action.card_index
    else:
        assert isinstance(action, Move)
        fr, fc = action.from_pos
        tr, tc = action.to_pos

        piece = state.board[fr][fc]
        if piece is None:
            raise ValueError("Invalid move: no piece at from_pos")
        if piece.owner != state.to_move:
            raise ValueError("Invalid move: piece does not belong to player to move")

        target = state.board[tr][tc]
        if target is not None and target.owner == state.to_move:
            raise ValueError("Invalid move: cannot capture your own piece")

        new_board = _board_after_move(state.board, fr, fc, tr, tc, piece)
        chosen_index = action.card_index

        # Update cached master positions.
        if piece.kind is PieceType.MASTER:
            if player is Player.RED:
                new_red_master = (tr, tc)
            else:
                new_blue_master = (tr, tc)
        if target is not None and target.kind is PieceType.MASTER:
            if target.owner is Player.RED:
                new_red_master = None
            else:
                new_blue_master = None

    # 2) Swap cards
    new_red_cards, new_blue_cards, new_side = _swap_cards(state, chosen_index)

    # 3) Next state
    return GameState(
        board=new_board,
        to_move=opponent,
        red_cards=new_red_cards,
        blue_cards=new_blue_cards,
        side_card=new_side,
        red_master_pos=new_red_master,
        blue_master_pos=new_blue_master,
    )
