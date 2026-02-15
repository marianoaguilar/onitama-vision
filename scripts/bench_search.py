from __future__ import annotations

import argparse
import statistics
import time
import sys
from pathlib import Path

# -----------------------------------------------------------------------------
# Make running from repo root robust (without requiring editable install)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from onitama.ai.agent import choose_action
from onitama.ai.evaluate import get_evaluator
from onitama.engine.rules import apply_action, generate_legal_actions
from onitama.engine.state import GameState


def _advance_state(state: GameState) -> tuple[GameState | None, int]:
    actions = generate_legal_actions(state)
    if not actions:
        return None, 0
    return apply_action(state, actions[0]), len(actions)  # deterministic progression


def _build_states(seed: int, plies: int, count: int, min_actions: int) -> list[GameState]:
    states: list[GameState] = []
    current_seed = seed
    state = GameState.initial(seed=current_seed)

    # Advance into a deterministic midgame.
    for _ in range(plies):
        nxt, _ = _advance_state(state)
        if nxt is None:
            current_seed += 1
            state = GameState.initial(seed=current_seed)
            continue
        state = nxt

    while len(states) < count:
        actions_now = generate_legal_actions(state)
        if len(actions_now) >= min_actions:
            states.append(state)

        nxt, _ = _advance_state(state)
        if nxt is None:
            current_seed += 1
            state = GameState.initial(seed=current_seed)
            # Advance again to reach midgame from the new seed.
            for _ in range(plies):
                nxt, _ = _advance_state(state)
                if nxt is None:
                    current_seed += 1
                    state = GameState.initial(seed=current_seed)
                    continue
                state = nxt
            continue
        state = nxt

    return states


def _bench_once(
    states: list[GameState],
    depth: int,
    evaluator_name: str,
    runs: int,
    warmup: int,
    tt_mode: str,
) -> dict[str, float]:
    evaluator = get_evaluator(evaluator_name)
    tt = {} if tt_mode == "persistent" else None
    use_tt = tt_mode in {"local", "persistent"}

    for i in range(warmup):
        _ = choose_action(states[i], depth=depth, evaluator=evaluator, use_tt=use_tt, tt=tt)

    times: list[float] = []
    for _ in range(runs):
        idx = warmup + _
        t0 = time.perf_counter()
        _ = choose_action(states[idx], depth=depth, evaluator=evaluator, use_tt=use_tt, tt=tt)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg = statistics.mean(times)
    stdev = statistics.pstdev(times)
    return {
        "avg": avg,
        "min": min(times),
        "max": max(times),
        "stdev": stdev,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Micro-benchmark for choose_action.")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--evaluator", type=str, default="v3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plies", type=int, default=8, help="Plies to advance before benchmarking.")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--min-actions",
        type=int,
        default=8,
        help="Minimum legal actions required for a state to be benchmarked.",
    )
    parser.add_argument(
        "--tt-mode",
        choices=["none", "local", "persistent"],
        default=None,
        help="TT mode: none, local (per call), persistent (reused across runs).",
    )
    parser.add_argument("--use-tt", action="store_true", help="(Legacy) Enable local TT.")
    parser.add_argument("--compare", action="store_true", help="Run all TT modes.")

    args = parser.parse_args()

    total = args.warmup + args.runs
    states = _build_states(seed=args.seed, plies=args.plies, count=total, min_actions=args.min_actions)

    if args.compare:
        modes = ["none", "local", "persistent"]
        results: dict[str, dict[str, float]] = {}
        for mode in modes:
            results[mode] = _bench_once(
                states,
                depth=args.depth,
                evaluator_name=args.evaluator,
                runs=args.runs,
                warmup=args.warmup,
                tt_mode=mode,
            )
        none_avg = results["none"]["avg"]
        for mode in modes:
            label = mode.capitalize()
            print(f"{label} TT:", results[mode])
        for mode in ["local", "persistent"]:
            speedup = none_avg / results[mode]["avg"] if results[mode]["avg"] > 0 else float("inf")
            print(f"Speedup (NoTT / {mode}): {speedup:.3f}x")
        return

    if args.tt_mode is None:
        tt_mode = "local" if args.use_tt else "none"
    else:
        tt_mode = args.tt_mode

    result = _bench_once(
        states,
        depth=args.depth,
        evaluator_name=args.evaluator,
        runs=args.runs,
        warmup=args.warmup,
        tt_mode=tt_mode,
    )
    print(f"{tt_mode.capitalize()} TT:", result)


if __name__ == "__main__":
    main()
