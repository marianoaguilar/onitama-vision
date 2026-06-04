from __future__ import annotations

"""Benchmark search optimizations on states from AI self-play games.

The script first generates self-play games with the complete agent and then
replays every decision state through a fixed ablation suite. This makes the
measurements representative of positions the agent actually reaches in play.

For TT-enabled configurations, one transposition table per player is kept for
the duration of each replayed game and cleared before the next game. This
matches the runtime/tournament model more closely than a single global table or
a fresh table per decision.
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
from onitama.ai.types import Evaluator, SearchStats, TranspositionTable
from onitama.engine.formatting import format_action
from onitama.engine.pieces import Player
from onitama.engine.rules import Action, apply_action, winner
from onitama.engine.state import GameState


@dataclass(frozen=True)
class BenchConfig:
    label: str
    use_tt: bool
    use_move_ordering: bool
    use_iterative_deepening: bool
    aspiration_window: int | None
    q_depth: int


@dataclass(frozen=True)
class ReplayState:
    game_index: int
    seed: int
    ply_index: int
    state: GameState


@dataclass
class GameMeta:
    game_index: int
    seed: int
    measured: bool
    plies: int
    winner: str
    reason: str


@dataclass
class BenchRow:
    config: str
    game_index: int
    seed: int
    ply_index: int
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


def _bench_configs(
    q_depth: int,
    aspiration_window: int,
    *,
    suite: str = "default",
) -> list[BenchConfig]:
    configs = [
        BenchConfig("full", True, True, True, aspiration_window, q_depth),
        BenchConfig("no_tt", False, True, True, aspiration_window, q_depth),
        BenchConfig("no_move_ordering", True, False, True, aspiration_window, q_depth),
        BenchConfig("no_iterative_deepening", True, True, False, aspiration_window, q_depth),
        BenchConfig("no_aspiration", True, True, True, None, q_depth),
        BenchConfig("minimal", False, False, False, None, q_depth),
    ]
    if suite == "default":
        configs.insert(5, BenchConfig("no_tt_no_iterative_deepening", False, True, False, None, q_depth))
    return configs


def _run_once(
    state: GameState,
    *,
    depth: int,
    evaluator: Evaluator,
    config: BenchConfig,
    tt: TranspositionTable | None,
) -> tuple[str, int | None, SearchStats, float, Action | None]:
    stats = SearchStats()

    t0 = time.perf_counter()
    action = choose_action(
        state,
        depth=depth,
        evaluator=evaluator,
        use_tt=config.use_tt,
        tt=tt,
        use_iterative_deepening=config.use_iterative_deepening,
        aspiration_window=config.aspiration_window,
        q_depth=config.q_depth,
        use_move_ordering=config.use_move_ordering,
        stats=stats,
    )
    elapsed = time.perf_counter() - t0

    action_label = "<terminal>" if action is None else format_action(state, action)
    return action_label, stats.value, stats, elapsed, action


def _generate_replay_games(
    *,
    seed_start: int,
    total_games: int,
    max_plies: int,
    depth: int,
    q_depth: int,
    evaluator_name: str,
    aspiration_window: int,
    warmup_games: int,
    progress_every: int,
) -> tuple[list[list[ReplayState]], list[GameMeta]]:
    evaluator = get_evaluator(evaluator_name)
    games: list[list[ReplayState]] = []
    metas: list[GameMeta] = []

    for game_index in range(total_games):
        seed = seed_start + game_index
        state = GameState.initial(seed=seed)
        tt_by_player: dict[Player, TranspositionTable] = {Player.RED: {}, Player.BLUE: {}}
        states: list[ReplayState] = []
        plies = 0

        while True:
            out = winner(state)
            if out is not None:
                player, reason = out
                metas.append(
                    GameMeta(
                        game_index=game_index,
                        seed=seed,
                        measured=game_index >= warmup_games,
                        plies=plies,
                        winner=player.value,
                        reason=reason,
                    )
                )
                break

            if plies >= max_plies:
                metas.append(
                    GameMeta(
                        game_index=game_index,
                        seed=seed,
                        measured=game_index >= warmup_games,
                        plies=plies,
                        winner="DRAW",
                        reason="Ply cap",
                    )
                )
                break

            states.append(ReplayState(game_index=game_index, seed=seed, ply_index=plies, state=state))
            action = choose_action(
                state,
                depth=depth,
                evaluator=evaluator,
                use_tt=True,
                tt=tt_by_player[state.to_move],
                use_iterative_deepening=True,
                aspiration_window=aspiration_window,
                q_depth=q_depth,
                use_move_ordering=True,
            )
            assert action is not None, "Full agent was asked to move in a terminal state."
            state = apply_action(state, action)
            plies += 1

        games.append(states)
        if progress_every > 0 and (game_index + 1) % progress_every == 0:
            print(f"Generated replay games: {game_index + 1}/{total_games}")

    return games, metas


def _bench_config(
    games: list[list[ReplayState]],
    *,
    depth: int,
    evaluator_name: str,
    config: BenchConfig,
    warmup_games: int,
) -> tuple[dict[str, float | int | str | bool | None], list[BenchRow]]:
    rows: list[BenchRow] = []
    evaluator = get_evaluator(evaluator_name)

    for game_index, game in enumerate(games):
        tt_by_player: dict[Player, TranspositionTable] | None = (
            {Player.RED: {}, Player.BLUE: {}} if config.use_tt else None
        )
        measured = game_index >= warmup_games

        for replay_state in game:
            state = replay_state.state
            action, value, stats, elapsed, _ = _run_once(
                state,
                depth=depth,
                evaluator=evaluator,
                config=config,
                tt=None if tt_by_player is None else tt_by_player[state.to_move],
            )

            if not measured:
                continue

            total_nodes = stats.nodes + stats.q_nodes
            nodes_per_sec = total_nodes / elapsed if elapsed > 0 else 0.0
            tt_hit_rate = stats.tt_hits / stats.tt_probes if stats.tt_probes else 0.0
            rows.append(
                BenchRow(
                    config=config.label,
                    game_index=replay_state.game_index,
                    seed=replay_state.seed,
                    ply_index=replay_state.ply_index,
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
        "decisions": len(rows),
        "avg_sec": statistics.mean(times) if times else 0.0,
        "min_sec": min(times) if times else 0.0,
        "max_sec": max(times) if times else 0.0,
        "stdev_sec": statistics.pstdev(times) if times else 0.0,
        "total_elapsed_sec": total_elapsed,
        "total_nodes": total_nodes,
        "avg_nodes": total_nodes / len(rows) if rows else 0.0,
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


def _add_decision_diffs(
    summaries: list[dict[str, float | int | str | bool | None]],
    rows: list[BenchRow],
) -> None:
    by_config: dict[str, dict[tuple[int, int], BenchRow]] = {}
    for row in rows:
        by_config.setdefault(row.config, {})[(row.game_index, row.ply_index)] = row

    full_rows = by_config.get("full", {})
    for summary in summaries:
        config = str(summary["config"])
        if config == "full":
            summary["action_diff_vs_full"] = 0
            summary["value_diff_vs_full"] = 0
            continue
        if not full_rows:
            summary["action_diff_vs_full"] = None
            summary["value_diff_vs_full"] = None
            continue

        action_diff = 0
        value_diff = 0
        for key, row in by_config.get(config, {}).items():
            full = full_rows.get(key)
            if full is None:
                continue
            if row.action != full.action:
                action_diff += 1
            if row.value != full.value:
                value_diff += 1
        summary["action_diff_vs_full"] = action_diff
        summary["value_diff_vs_full"] = value_diff


def _write_csv(path: Path, rows: list[BenchRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(BenchRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark search ablations on self-play decision states.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--q-depth", type=int, default=2)
    parser.add_argument("--aspiration-window", type=int, default=100)
    parser.add_argument("--evaluator", type=str, default="v3", choices=sorted(EVALUATORS))
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--games", type=int, default=20, help="Measured replay games.")
    parser.add_argument("--warmup-games", type=int, default=2)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument(
        "--config-suite",
        choices=["default", "standard"],
        default="default",
        help="Config set to run. standard excludes the extended combined ablations.",
    )
    parser.add_argument(
        "--only-config",
        choices=[
            "full",
            "no_tt",
            "no_move_ordering",
            "no_iterative_deepening",
            "no_aspiration",
            "no_tt_no_iterative_deepening",
            "minimal",
        ],
        default=None,
        help="Run only one ablation config after replay game generation.",
    )
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--out", type=str, default=None, help="Write per-decision CSV.")
    parser.add_argument("--out-json", type=str, default=None, help="Write summary JSON.")

    args = parser.parse_args()
    if args.depth < 1:
        raise ValueError("depth must be >= 1")
    if args.q_depth < 0:
        raise ValueError("q_depth must be >= 0")
    if args.aspiration_window <= 0:
        raise ValueError("aspiration_window must be > 0")
    if args.games < 1:
        raise ValueError("games must be >= 1")
    if args.warmup_games < 0:
        raise ValueError("warmup_games must be >= 0")
    if args.max_plies < 1:
        raise ValueError("max_plies must be >= 1")
    if args.progress_every < 0:
        raise ValueError("progress_every must be >= 0")

    total_games = args.warmup_games + args.games
    games, game_metas = _generate_replay_games(
        seed_start=args.seed_start,
        total_games=total_games,
        max_plies=args.max_plies,
        depth=args.depth,
        q_depth=args.q_depth,
        evaluator_name=args.evaluator,
        aspiration_window=args.aspiration_window,
        warmup_games=args.warmup_games,
        progress_every=args.progress_every,
    )
    measured_decisions = sum(len(game) for game in games[args.warmup_games :])
    print(f"Measured replay decisions: {measured_decisions}")

    configs = _bench_configs(
        args.q_depth,
        args.aspiration_window,
        suite=args.config_suite,
    )
    if args.only_config is not None:
        known_configs = {
            config.label: config
            for config in _bench_configs(
                args.q_depth,
                args.aspiration_window,
                suite="default",
            )
        }
        configs = [known_configs[args.only_config]]

    summaries: list[dict[str, float | int | str | bool | None]] = []
    rows: list[BenchRow] = []
    for config in configs:
        summary, config_rows = _bench_config(
            games,
            depth=args.depth,
            evaluator_name=args.evaluator,
            config=config,
            warmup_games=args.warmup_games,
        )
        summaries.append(summary)
        rows.extend(config_rows)
        print(
            f"{config.label}: avg={summary['avg_sec']:.6f}s "
            f"nodes/s={summary['nodes_per_sec']:.0f} "
            f"nodes={summary['total_nodes']} "
            f"tt_hit_rate={summary['tt_hit_rate']:.3f}"
        )

    _add_decision_diffs(summaries, rows)

    baseline_avg = summaries[0]["avg_sec"]
    for summary in summaries[1:]:
        relative_time = summary["avg_sec"] / baseline_avg if baseline_avg > 0 else float("inf")
        print(f"Relative time ({summary['config']} / full): {relative_time:.3f}x")

    if args.out:
        out = Path(args.out)
        _write_csv(out, rows)
        print(f"CSV written: {out}")

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mode": "replay",
            "seed_start": args.seed_start,
            "games": args.games,
            "warmup_games": args.warmup_games,
            "max_plies": args.max_plies,
            "measured_decisions": measured_decisions,
            "depth": args.depth,
            "q_depth": args.q_depth,
            "aspiration_window": args.aspiration_window,
            "evaluator": args.evaluator,
            "tt_scope": "per_player_per_game",
            "config_suite": args.config_suite,
            "only_config": args.only_config,
            "configs": [asdict(config) for config in configs],
            "games_meta": [asdict(meta) for meta in game_metas],
            "summaries": summaries,
            "rows": [asdict(row) for row in rows],
        }
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"JSON written: {out_json}")


if __name__ == "__main__":
    main()
