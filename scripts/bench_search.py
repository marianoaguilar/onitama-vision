from __future__ import annotations

"""Benchmark de optimizaciones de busqueda para choose_action.

Ejecuta una suite fija de ablacion:
  - full: configuracion completa.
  - no_tt: sin tabla de transposicion.
  - no_move_ordering: sin ordenacion de movimientos.
  - no_iterative_deepening: busqueda directa a la profundidad objetivo.
  - no_aspiration: sin ventana de aspiracion.
  - no_quiescence: sin busqueda quiescente.
  - minimal: sin TT, ordenacion, iterative deepening, aspiration ni quiescence.

Las posiciones se generan de forma determinista a partir de seed/plies para que
las comparaciones sean reproducibles.
"""

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from onitama.ai.agent import choose_action
from onitama.ai.evaluate import EVALUATORS, get_evaluator
from onitama.ai.types import SearchStats
from onitama.engine.formatting import format_action
from onitama.engine.rules import apply_action, generate_legal_actions
from onitama.engine.state import GameState


@dataclass(frozen=True)
class BenchConfig:
    label: str
    use_tt: bool
    use_move_ordering: bool
    use_iterative_deepening: bool
    aspiration_window: int | None
    q_depth: int


@dataclass
class BenchRow:
    config: str
    run_index: int
    state_index: int
    elapsed_sec: float
    depth: int
    evaluator: str
    action: str
    value: int | None
    nodes: int
    q_nodes: int
    total_nodes: int
    nodes_per_sec: float
    beta_cutoffs: int
    tt_probes: int
    tt_hits: int
    tt_hit_rate: float
    tt_cutoffs: int
    tt_stores: int
    depth_completed: int


def _advance_state(state: GameState) -> GameState | None:
    actions = generate_legal_actions(state)
    if not actions:
        return None
    return apply_action(state, actions[0])


def _build_states(seed: int, plies: int, count: int, min_actions: int) -> list[GameState]:
    states: list[GameState] = []
    current_seed = seed
    state = GameState.initial(seed=current_seed)

    for _ in range(plies):
        nxt = _advance_state(state)
        if nxt is None:
            current_seed += 1
            state = GameState.initial(seed=current_seed)
        else:
            state = nxt

    while len(states) < count:
        if len(generate_legal_actions(state)) >= min_actions:
            states.append(state)

        nxt = _advance_state(state)
        if nxt is None:
            current_seed += 1
            state = GameState.initial(seed=current_seed)
            for _ in range(plies):
                nxt = _advance_state(state)
                if nxt is None:
                    current_seed += 1
                    state = GameState.initial(seed=current_seed)
                else:
                    state = nxt
        else:
            state = nxt

    return states


def _bench_configs(q_depth: int, aspiration_window: int) -> list[BenchConfig]:
    return [
        BenchConfig("full", True, True, True, aspiration_window, q_depth),
        BenchConfig("no_tt", False, True, True, aspiration_window, q_depth),
        BenchConfig("no_move_ordering", True, False, True, aspiration_window, q_depth),
        BenchConfig("no_iterative_deepening", True, True, False, aspiration_window, q_depth),
        BenchConfig("no_aspiration", True, True, True, None, q_depth),
        BenchConfig("no_quiescence", True, True, True, aspiration_window, 0),
        BenchConfig("minimal", False, False, False, None, 0),
    ]


def _run_once(
    state: GameState,
    *,
    depth: int,
    evaluator_name: str,
    config: BenchConfig,
) -> tuple[str, int | None, SearchStats, float]:
    stats = SearchStats()
    evaluator = get_evaluator(evaluator_name)

    t0 = time.perf_counter()
    action = choose_action(
        state,
        depth=depth,
        evaluator=evaluator,
        use_tt=config.use_tt,
        use_iterative_deepening=config.use_iterative_deepening,
        aspiration_window=config.aspiration_window,
        q_depth=config.q_depth,
        use_move_ordering=config.use_move_ordering,
        stats=stats,
    )
    elapsed = time.perf_counter() - t0

    action_label = "<terminal>" if action is None else format_action(state, action)
    return action_label, stats.value, stats, elapsed


def _bench_config(
    states: list[GameState],
    *,
    depth: int,
    evaluator_name: str,
    runs: int,
    warmup: int,
    config: BenchConfig,
) -> tuple[dict[str, float | int | str | bool | None], list[BenchRow]]:
    for i in range(warmup):
        _run_once(states[i], depth=depth, evaluator_name=evaluator_name, config=config)

    rows: list[BenchRow] = []
    for run_index in range(runs):
        state_index = warmup + run_index
        action, value, stats, elapsed = _run_once(
            states[state_index],
            depth=depth,
            evaluator_name=evaluator_name,
            config=config,
        )
        total_nodes = stats.nodes + stats.q_nodes
        nodes_per_sec = total_nodes / elapsed if elapsed > 0 else 0.0
        tt_hit_rate = stats.tt_hits / stats.tt_probes if stats.tt_probes else 0.0
        rows.append(
            BenchRow(
                config=config.label,
                run_index=run_index,
                state_index=state_index,
                elapsed_sec=elapsed,
                depth=depth,
                evaluator=evaluator_name,
                action=action,
                value=value,
                nodes=stats.nodes,
                q_nodes=stats.q_nodes,
                total_nodes=total_nodes,
                nodes_per_sec=nodes_per_sec,
                beta_cutoffs=stats.beta_cutoffs,
                tt_probes=stats.tt_probes,
                tt_hits=stats.tt_hits,
                tt_hit_rate=tt_hit_rate,
                tt_cutoffs=stats.tt_cutoffs,
                tt_stores=stats.tt_stores,
                depth_completed=stats.depth_completed,
            )
        )

    times = [row.elapsed_sec for row in rows]
    total_nodes = sum(row.total_nodes for row in rows)
    total_elapsed = sum(times)
    tt_probes = sum(row.tt_probes for row in rows)
    tt_hits = sum(row.tt_hits for row in rows)
    summary = {
        "config": config.label,
        "depth": depth,
        "evaluator": evaluator_name,
        "runs": runs,
        "warmup": warmup,
        "avg_sec": statistics.mean(times),
        "min_sec": min(times),
        "max_sec": max(times),
        "stdev_sec": statistics.pstdev(times),
        "total_elapsed_sec": total_elapsed,
        "total_nodes": total_nodes,
        "avg_nodes": total_nodes / runs if runs else 0.0,
        "nodes_per_sec": total_nodes / total_elapsed if total_elapsed > 0 else 0.0,
        "beta_cutoffs": sum(row.beta_cutoffs for row in rows),
        "tt_probes": tt_probes,
        "tt_hits": tt_hits,
        "tt_hit_rate": tt_hits / tt_probes if tt_probes else 0.0,
        "tt_cutoffs": sum(row.tt_cutoffs for row in rows),
        "tt_stores": sum(row.tt_stores for row in rows),
        **asdict(config),
    }
    return summary, rows


def _write_csv(path: Path, rows: list[BenchRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(BenchRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark fixed search-optimization ablations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--q-depth", type=int, default=2)
    parser.add_argument("--aspiration-window", type=int, default=100)
    parser.add_argument("--evaluator", type=str, default="v3", choices=sorted(EVALUATORS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plies", type=int, default=8)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--min-actions", type=int, default=8)
    parser.add_argument("--out", type=str, default=None, help="Write per-run CSV.")
    parser.add_argument("--out-json", type=str, default=None, help="Write summary JSON.")

    args = parser.parse_args()
    if args.depth < 1:
        raise ValueError("depth must be >= 1")
    if args.q_depth < 0:
        raise ValueError("q_depth must be >= 0")
    if args.aspiration_window <= 0:
        raise ValueError("aspiration_window must be > 0")
    if args.runs < 1:
        raise ValueError("runs must be >= 1")
    if args.warmup < 0:
        raise ValueError("warmup must be >= 0")
    if args.plies < 0:
        raise ValueError("plies must be >= 0")
    if args.min_actions < 0:
        raise ValueError("min_actions must be >= 0")

    configs = _bench_configs(args.q_depth, args.aspiration_window)
    states = _build_states(
        seed=args.seed,
        plies=args.plies,
        count=args.warmup + args.runs,
        min_actions=args.min_actions,
    )

    summaries: list[dict[str, float | int | str | bool | None]] = []
    rows: list[BenchRow] = []
    for config in configs:
        summary, config_rows = _bench_config(
            states,
            depth=args.depth,
            evaluator_name=args.evaluator,
            runs=args.runs,
            warmup=args.warmup,
            config=config,
        )
        summaries.append(summary)
        rows.extend(config_rows)
        print(
            f"{config.label}: avg={summary['avg_sec']:.6f}s "
            f"nodes/s={summary['nodes_per_sec']:.0f} "
            f"nodes={summary['total_nodes']} "
            f"tt_hit_rate={summary['tt_hit_rate']:.3f}"
        )

    baseline_avg = summaries[0]["avg_sec"]
    for summary in summaries[1:]:
        speedup = baseline_avg / summary["avg_sec"] if summary["avg_sec"] > 0 else float("inf")
        print(f"Speedup (full / {summary['config']}): {speedup:.3f}x")

    if args.out:
        out = Path(args.out)
        _write_csv(out, rows)
        print(f"CSV written: {out}")

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "seed": args.seed,
            "plies": args.plies,
            "min_actions": args.min_actions,
            "depth": args.depth,
            "q_depth": args.q_depth,
            "aspiration_window": args.aspiration_window,
            "evaluator": args.evaluator,
            "runs": args.runs,
            "warmup": args.warmup,
            "configs": [asdict(config) for config in configs],
            "summaries": summaries,
            "rows": [asdict(row) for row in rows],
        }
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"JSON written: {out_json}")


if __name__ == "__main__":
    main()
