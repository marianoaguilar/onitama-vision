from cli.render import pos_to_coord, render_state
from onitama.rules import apply_move, generate_legal_moves
from onitama.state import GameState


def main() -> None:
    state = GameState.initial()
    print("=== INITIAL STATE ===")
    print(render_state(state))
    print("")

    moves = generate_legal_moves(state)
    print(f"Legal moves: {len(moves)}")
    for m in moves[:10]:
        cards = state.red_cards if state.to_move.value == "RED" else state.blue_cards
        card_name = cards[m.card_index].name
        print(f"- {card_name}: {pos_to_coord(m.from_pos)} -> {pos_to_coord(m.to_pos)}")

    print("")
    first = moves[0]
    cards = state.red_cards if state.to_move.value == "RED" else state.blue_cards
    first_card_name = cards[first.card_index].name
    print(f"Applying first move: {first_card_name} {pos_to_coord(first.from_pos)} -> {pos_to_coord(first.to_pos)}")
    print("")

    next_state = apply_move(state, first)
    print("=== STATE AFTER MOVE ===")
    print(render_state(next_state))


if __name__ == "__main__":
    main()
