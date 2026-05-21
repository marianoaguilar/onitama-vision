from onitama.engine.rules import apply_action, generate_legal_actions
from onitama.engine.state import GameState
from onitama.session.stabilizer import StateStabilizer


def test_stabilizer_does_not_confirm_before_threshold():
    stabilizer = StateStabilizer(required_repeats=3)
    state = GameState.initial(seed=1)

    first = stabilizer.push(state)
    second = stabilizer.push(state)

    assert first is None
    assert second is None
    assert stabilizer.candidate_state == state
    assert stabilizer.repeat_count == 2


def test_stabilizer_confirms_when_threshold_is_reached():
    stabilizer = StateStabilizer(required_repeats=3)
    state = GameState.initial(seed=1)

    stabilizer.push(state)
    stabilizer.push(state)
    third = stabilizer.push(state)

    assert third == state
    assert stabilizer.repeat_count == 3


def test_stabilizer_resets_counter_when_state_changes():
    stabilizer = StateStabilizer(required_repeats=3)
    first_state = GameState.initial(seed=1)
    second_state = apply_action(first_state, generate_legal_actions(first_state)[0])

    stabilizer.push(first_state)
    stabilizer.push(first_state)
    changed = stabilizer.push(second_state)

    assert changed is None
    assert stabilizer.candidate_state == second_state
    assert stabilizer.repeat_count == 1


def test_stabilizer_reset_clears_internal_state():
    stabilizer = StateStabilizer(required_repeats=3)
    state = GameState.initial(seed=1)

    stabilizer.push(state)
    stabilizer.push(state)
    stabilizer.reset()

    assert stabilizer.candidate_state is None
    assert stabilizer.repeat_count == 0

    after_reset = stabilizer.push(state)

    assert after_reset is None
    assert stabilizer.repeat_count == 1


def test_stabilizer_keeps_reporting_stable_for_same_state():
    stabilizer = StateStabilizer(required_repeats=3)
    state = GameState.initial(seed=1)

    stabilizer.push(state)
    stabilizer.push(state)
    stabilizer.push(state)
    fourth = stabilizer.push(state)

    assert fourth == state
    assert stabilizer.repeat_count == 4
