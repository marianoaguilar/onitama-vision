from __future__ import annotations

"""Onitama tournament runner for AI heuristic/profile evaluation.

The script supports two experiment shapes:
  - match: head-to-head A vs B.
  - roundrobin: every configured agent against every other agent.

Agent tokens are explicit in the command line:
  - v1@3      -> evaluator v1, depth 3, default q-depth.
  - v3@5q2    -> evaluator v3, depth 5, q-depth 2.

Every seed is paired with a color swap: A as RED vs B as BLUE, then B as RED
vs A as BLUE. This keeps heuristic/profile comparisons reproducible and less
dependent on color assignment.
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from onitama.ai.agent import choose_action
from onitama.ai.evaluate import EVALUATORS, get_evaluator
from onitama.ai.types import TranspositionTable
from onitama.engine.pieces import Player
from onitama.engine.rules import Action, apply_action, winner
from onitama.engine.state import GameState


@dataclass(frozen=True)
class AgentSpec:
    label: str
    evaluator_name: str
    depth: int
    q_depth: int

    def validate(self) -> None:
        if self.depth < 1:
            raise ValueError("depth must be >= 1")
        if self.q_depth < 0:
            raise ValueError("q_depth must be >= 0")
        get_evaluator(self.evaluator_name)


@dataclass
class CompiledAgent:
    label: str
    evaluator_name: str
    depth: int
    q_depth: int
    evaluator: Callable[[GameState, Player], int]
    tt: TranspositionTable

    def select_action(self, state: GameState) -> tuple[Action, float]:
        t0 = time.perf_counter()
        action = choose_action(
            state,
            depth=self.depth,
            evaluator=self.evaluator,
            q_depth=self.q_depth,
            tt=self.tt,
        )
        elapsed = time.perf_counter() - t0
        assert action is not None, "Agent asked to move in a terminal state."
        return action, elapsed


@dataclass
class GameRecord:
    game_id: int
    pair_id: int
    seed: int
    swapped_colors: bool
    red_agent: str
    blue_agent: str
    winner: str
    reason: str
    plies: int
    starting_player: str
    elapsed_sec: float
    red_search_time_sec: float
    red_decisions: int
    blue_search_time_sec: float
    blue_decisions: int

    @property
    def search_time_sec(self) -> float:
        return self.red_search_time_sec + self.blue_search_time_sec


@dataclass
class Aggregate:
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    points: float = 0.0
    total_plies: int = 0
    elapsed_sec: float = 0.0
    search_time_sec: float = 0.0
    decisions: int = 0

    def add(self, result: str, game: GameRecord, color: str) -> None:
        self.games += 1
        self.total_plies += game.plies
        self.elapsed_sec += game.elapsed_sec

        if color == "RED":
            self.search_time_sec += game.red_search_time_sec
            self.decisions += game.red_decisions
        else:
            self.search_time_sec += game.blue_search_time_sec
            self.decisions += game.blue_decisions

        if result == "WIN":
            self.wins += 1
            self.points += 1.0
        elif result == "LOSS":
            self.losses += 1
        elif result == "DRAW":
            self.draws += 1
            self.points += 0.5
        else:
            raise ValueError(f"Unknown result: {result}")

    def avg_plies(self) -> float:
        return self.total_plies / self.games if self.games else 0.0

    def score_rate(self) -> float:
        return self.points / self.games if self.games else 0.0

    def avg_decision_sec(self) -> float:
        return self.search_time_sec / self.decisions if self.decisions else 0.0


def _aggregate_payload(agg: Aggregate) -> dict[str, float | int]:
    return {
        "games": agg.games,
        "wins": agg.wins,
        "draws": agg.draws,
        "losses": agg.losses,
        "points": agg.points,
        "score_rate": agg.score_rate(),
        "avg_plies": agg.avg_plies(),
        "elapsed_sec": agg.elapsed_sec,
        "search_time_sec": agg.search_time_sec,
        "decisions": agg.decisions,
        "avg_decision_sec": agg.avg_decision_sec(),
    }


def _parse_agent_token(token: str, default_depth: int, default_q_depth: int) -> AgentSpec:
    """Parse evaluator tokens such as v1, v1@3, or v3@5q2."""
    token = token.strip()
    if not token:
        raise ValueError("Empty agent token")

    q_depth = default_q_depth
    if "@" in token:
        name, depth_part = token.split("@", 1)
        name = name.strip()
        depth_part = depth_part.strip()
        if "q" in depth_part:
            depth_text, q_text = depth_part.split("q", 1)
            depth = int(depth_text)
            q_depth = int(q_text)
        else:
            depth = int(depth_part)
    else:
        name = token
        depth = default_depth

    label = f"{name}@{depth}q{q_depth}"
    spec = AgentSpec(label=label, evaluator_name=name, depth=depth, q_depth=q_depth)
    spec.validate()
    return spec


def _compile_agent(spec: AgentSpec) -> CompiledAgent:
    return CompiledAgent(
        label=spec.label,
        evaluator_name=spec.evaluator_name,
        depth=spec.depth,
        q_depth=spec.q_depth,
        evaluator=get_evaluator(spec.evaluator_name),
        tt={},
    )


def _git_commit_hash() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def play_game(
    *,
    seed: int,
    red: CompiledAgent,
    blue: CompiledAgent,
    max_plies: int,
    game_id: int,
    pair_id: int,
    swapped_colors: bool,
) -> GameRecord:
    red.tt.clear()
    blue.tt.clear()

    state = GameState.initial(seed=seed)
    starting_player = "RED" if state.to_move == Player.RED else "BLUE"

    plies = 0
    red_search_time_sec = 0.0
    red_decisions = 0
    blue_search_time_sec = 0.0
    blue_decisions = 0
    game_t0 = time.perf_counter()

    def _record(winner_str: str, reason: str) -> GameRecord:
        return GameRecord(
            game_id=game_id,
            pair_id=pair_id,
            seed=seed,
            swapped_colors=swapped_colors,
            red_agent=red.label,
            blue_agent=blue.label,
            winner=winner_str,
            reason=reason,
            plies=plies,
            starting_player=starting_player,
            elapsed_sec=time.perf_counter() - game_t0,
            red_search_time_sec=red_search_time_sec,
            red_decisions=red_decisions,
            blue_search_time_sec=blue_search_time_sec,
            blue_decisions=blue_decisions,
        )

    while True:
        out = winner(state)
        if out is not None:
            w, reason = out
            return _record("RED" if w == Player.RED else "BLUE", reason)

        if plies >= max_plies:
            return _record("DRAW", "Ply cap")

        mover = state.to_move
        agent = red if mover == Player.RED else blue
        action, elapsed = agent.select_action(state)

        if mover == Player.RED:
            red_search_time_sec += elapsed
            red_decisions += 1
        else:
            blue_search_time_sec += elapsed
            blue_decisions += 1

        state = apply_action(state, action)
        plies += 1


def _update_pair_aggregates(
    game: GameRecord,
    left_label: str,
    right_label: str,
    left: Aggregate,
    right: Aggregate,
) -> None:
    left_color = "RED" if game.red_agent == left_label else "BLUE"
    right_color = "RED" if game.red_agent == right_label else "BLUE"

    if game.winner == "DRAW":
        left.add("DRAW", game, left_color)
        right.add("DRAW", game, right_color)
        return

    left.add("WIN" if game.winner == left_color else "LOSS", game, left_color)
    right.add("WIN" if game.winner == right_color else "LOSS", game, right_color)


def _merge_aggregate(dst: Aggregate, src: Aggregate) -> None:
    dst.games += src.games
    dst.wins += src.wins
    dst.losses += src.losses
    dst.draws += src.draws
    dst.points += src.points
    dst.total_plies += src.total_plies
    dst.elapsed_sec += src.elapsed_sec
    dst.search_time_sec += src.search_time_sec
    dst.decisions += src.decisions


def _write_csv(path: Path, records: Iterable[GameRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "game_id",
                "pair_id",
                "seed",
                "swapped_colors",
                "red_agent",
                "blue_agent",
                "winner",
                "reason",
                "plies",
                "starting_player",
                "elapsed_sec",
                "search_time_sec",
                "red_search_time_sec",
                "red_decisions",
                "blue_search_time_sec",
                "blue_decisions",
            ]
        )
        for r in records:
            writer.writerow(
                [
                    r.game_id,
                    r.pair_id,
                    r.seed,
                    int(r.swapped_colors),
                    r.red_agent,
                    r.blue_agent,
                    r.winner,
                    r.reason,
                    r.plies,
                    r.starting_player,
                    f"{r.elapsed_sec:.9f}",
                    f"{r.search_time_sec:.9f}",
                    f"{r.red_search_time_sec:.9f}",
                    r.red_decisions,
                    f"{r.blue_search_time_sec:.9f}",
                    r.blue_decisions,
                ]
            )


def _agent_payload(agent: AgentSpec) -> dict[str, int | str]:
    return {
        "label": agent.label,
        "eval": agent.evaluator_name,
        "depth": agent.depth,
        "q_depth": agent.q_depth,
    }


def _print_match_summary(
    *,
    a: AgentSpec,
    b: AgentSpec,
    agg_a: Aggregate,
    agg_b: Aggregate,
    elapsed: float,
    pairs: int,
    max_plies: int,
    seed_start: int,
) -> None:
    total_games = agg_a.games
    assert total_games == 2 * pairs

    print("")
    print("=== Match summary (paired + color swap) ===")
    print(f"Pairs       : {pairs} (2 games per pair => {total_games} games)")
    print(f"Seed start  : {seed_start}")
    print(f"Max plies   : {max_plies}")
    print("")
    print(f"A: {a.label}  (eval={a.evaluator_name}, depth={a.depth}, q={a.q_depth})")
    print(f"B: {b.label}  (eval={b.evaluator_name}, depth={b.depth}, q={b.q_depth})")
    print("")
    print("Results (win=1, draw=0.5):")
    print(f"  A points  : {agg_a.points:.1f} / {total_games}  (rate={agg_a.score_rate():.3f})")
    print(f"  B points  : {agg_b.points:.1f} / {total_games}  (rate={agg_b.score_rate():.3f})")
    print("")
    print(f"  A W/D/L   : {agg_a.wins}/{agg_a.draws}/{agg_a.losses}")
    print(f"  B W/D/L   : {agg_b.wins}/{agg_b.draws}/{agg_b.losses}")
    print("")
    print(f"Avg plies   : {agg_a.avg_plies():.1f}")
    print(f"Elapsed     : {elapsed:.2f}s")
    if elapsed > 0:
        print(f"Games/sec   : {total_games / elapsed:.2f}")
    print(f"A search    : {agg_a.search_time_sec:.2f}s  avg decision={agg_a.avg_decision_sec():.4f}s")
    print(f"B search    : {agg_b.search_time_sec:.2f}s  avg decision={agg_b.avg_decision_sec():.4f}s")


def run_match(
    *,
    a: AgentSpec,
    b: AgentSpec,
    pairs: int,
    seed_start: int,
    max_plies: int,
    out_csv: Path | None,
    out_json: Path | None,
    progress_every: int,
    git_commit: str | None,
) -> None:
    if pairs < 1:
        raise ValueError("pairs must be >= 1")

    agent_a = _compile_agent(a)
    agent_b = _compile_agent(b)

    records: list[GameRecord] = []
    agg_a = Aggregate()
    agg_b = Aggregate()

    t0 = time.perf_counter()
    game_id = 0
    for pair_idx in range(pairs):
        seed = seed_start + pair_idx

        g1 = play_game(
            seed=seed,
            red=agent_a,
            blue=agent_b,
            max_plies=max_plies,
            game_id=game_id,
            pair_id=pair_idx,
            swapped_colors=False,
        )
        game_id += 1

        g2 = play_game(
            seed=seed,
            red=agent_b,
            blue=agent_a,
            max_plies=max_plies,
            game_id=game_id,
            pair_id=pair_idx,
            swapped_colors=True,
        )
        game_id += 1

        records.extend([g1, g2])
        for game in (g1, g2):
            _update_pair_aggregates(game, agent_a.label, agent_b.label, agg_a, agg_b)

        if progress_every > 0 and (pair_idx + 1) % progress_every == 0:
            done_pairs = pair_idx + 1
            print(
                f"Progress: {done_pairs}/{pairs} pairs "
                f"(games={2 * done_pairs}) | A rate={agg_a.score_rate():.3f}"
            )

    elapsed = time.perf_counter() - t0

    _print_match_summary(
        a=a,
        b=b,
        agg_a=agg_a,
        agg_b=agg_b,
        elapsed=elapsed,
        pairs=pairs,
        max_plies=max_plies,
        seed_start=seed_start,
    )

    if out_csv is not None:
        _write_csv(out_csv, records)
        print(f"\nCSV written: {out_csv}")

    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mode": "match",
            "pairs": pairs,
            "seed_start": seed_start,
            "max_plies": max_plies,
            "agent_a": _agent_payload(a),
            "agent_b": _agent_payload(b),
            "git_commit": git_commit,
            "summary": {
                "a": _aggregate_payload(agg_a),
                "b": _aggregate_payload(agg_b),
                "elapsed_sec": elapsed,
            },
        }
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"JSON written: {out_json}")


def run_roundrobin(
    *,
    agents: list[AgentSpec],
    pairs: int,
    seed_start: int,
    max_plies: int,
    out_csv: Path | None,
    out_json: Path | None,
    progress_every: int,
    git_commit: str | None,
    offset_seeds_by_matchup: bool,
) -> None:
    if len(agents) < 2:
        raise ValueError("Need at least 2 agents for round-robin")
    labels = [agent.label for agent in agents]
    if len(set(labels)) != len(labels):
        raise ValueError("Round-robin agents must have unique labels")

    compiled = {agent.label: _compile_agent(agent) for agent in agents}
    scoreboard: dict[str, Aggregate] = {agent.label: Aggregate() for agent in agents}
    matchups: list[dict[str, object]] = []
    records: list[GameRecord] = []

    t0 = time.perf_counter()
    game_id = 0
    matchup_idx = 0

    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            a = agents[i]
            b = agents[j]
            matchup_idx += 1

            local_a = Aggregate()
            local_b = Aggregate()

            for pair_idx in range(pairs):
                seed = (
                    seed_start + (matchup_idx - 1) * pairs + pair_idx
                    if offset_seeds_by_matchup
                    else seed_start + pair_idx
                )
                pair_id = (matchup_idx - 1) * pairs + pair_idx

                g1 = play_game(
                    seed=seed,
                    red=compiled[a.label],
                    blue=compiled[b.label],
                    max_plies=max_plies,
                    game_id=game_id,
                    pair_id=pair_id,
                    swapped_colors=False,
                )
                game_id += 1

                g2 = play_game(
                    seed=seed,
                    red=compiled[b.label],
                    blue=compiled[a.label],
                    max_plies=max_plies,
                    game_id=game_id,
                    pair_id=pair_id,
                    swapped_colors=True,
                )
                game_id += 1

                records.extend([g1, g2])
                for game in (g1, g2):
                    _update_pair_aggregates(game, a.label, b.label, local_a, local_b)

                if progress_every > 0 and (pair_idx + 1) % progress_every == 0:
                    total_matchups = len(agents) * (len(agents) - 1) // 2
                    done_pairs = pair_idx + 1
                    print(
                        f"Progress: matchup {matchup_idx}/{total_matchups} "
                        f"({a.label} vs {b.label}) | pairs={done_pairs}/{pairs} "
                        f"| {a.label} rate={local_a.score_rate():.3f}"
                    )

            _merge_aggregate(scoreboard[a.label], local_a)
            _merge_aggregate(scoreboard[b.label], local_b)

            matchups.append(
                {
                    "a": a.label,
                    "b": b.label,
                    "pairs": pairs,
                    "games": 2 * pairs,
                    "a_points": local_a.points,
                    "b_points": local_b.points,
                    "a_rate": local_a.score_rate(),
                    "b_rate": local_b.score_rate(),
                    "a_search_time_sec": local_a.search_time_sec,
                    "b_search_time_sec": local_b.search_time_sec,
                    "a_avg_decision_sec": local_a.avg_decision_sec(),
                    "b_avg_decision_sec": local_b.avg_decision_sec(),
                }
            )

    elapsed = time.perf_counter() - t0

    print("")
    print("=== Round-robin summary (paired + color swap per matchup) ===")
    print(f"Agents     : {len(agents)}")
    print(f"Pairs/match: {pairs} (2 games per pair)")
    print(f"Seed start : {seed_start}")
    print(f"Max plies  : {max_plies}")
    print(f"Elapsed    : {elapsed:.2f}s")
    print("")

    ranking = sorted(scoreboard.items(), key=lambda kv: kv[1].points, reverse=True)
    for rank, (label, agg) in enumerate(ranking, start=1):
        print(
            f"{rank:>2}. {label:<12} "
            f"points={agg.points:>6.1f}  rate={agg.score_rate():.3f}  "
            f"W/D/L={agg.wins}/{agg.draws}/{agg.losses}  "
            f"avg_plies={agg.avg_plies():.1f}  "
            f"avg_decision={agg.avg_decision_sec():.4f}s"
        )

    if out_csv is not None:
        _write_csv(out_csv, records)
        print(f"\nCSV written: {out_csv}")

    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mode": "roundrobin",
            "pairs_per_match": pairs,
            "seed_start": seed_start,
            "offset_seeds_by_matchup": offset_seeds_by_matchup,
            "max_plies": max_plies,
            "git_commit": git_commit,
            "agents": [_agent_payload(agent) for agent in agents],
            "scoreboard": {label: _aggregate_payload(agg) for label, agg in scoreboard.items()},
            "matchups": matchups,
            "elapsed_sec": elapsed,
        }
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"JSON written: {out_json}")


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--pairs", type=int, default=200)
    parser.add_argument("--out", type=str, default=None, help="Write per-game CSV.")
    parser.add_argument("--out-json", type=str, default=None, help="Write summary JSON.")
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--default-depth", type=int, default=3)
    parser.add_argument("--default-q-depth", type=int, default=2)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Onitama tournament runner for heuristic/profile evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_match = sub.add_parser("match", help="Run a head-to-head match between two agents.")
    _add_common_args(p_match)
    p_match.add_argument("--a", required=True, help="Agent A, e.g. v1@3 or v3@5q2.")
    p_match.add_argument("--b", required=True, help="Agent B, e.g. v2@3 or v3@1q0.")

    p_rr = sub.add_parser("roundrobin", help="Run a round-robin across multiple agents.")
    _add_common_args(p_rr)
    p_rr.set_defaults(progress_every=2)
    p_rr.add_argument("--agents", nargs="+", required=True, help="Agent specs, e.g. v1@3 v2@3 v3@3.")
    p_rr.add_argument(
        "--offset-seeds-by-matchup",
        action="store_true",
        help="Use different seed ranges per matchup.",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.default_depth < 1:
        raise ValueError("default_depth must be >= 1")
    if args.default_q_depth < 0:
        raise ValueError("default_q_depth must be >= 0")
    if args.pairs < 1:
        raise ValueError("pairs must be >= 1")
    if args.max_plies < 1:
        raise ValueError("max_plies must be >= 1")
    if args.progress_every < 0:
        raise ValueError("progress_every must be >= 0")

    _ = sorted(EVALUATORS)
    git_commit = _git_commit_hash()
    out_csv = Path(args.out) if args.out else None
    out_json = Path(args.out_json) if args.out_json else None

    if args.cmd == "match":
        a = _parse_agent_token(args.a, args.default_depth, args.default_q_depth)
        b = _parse_agent_token(args.b, args.default_depth, args.default_q_depth)
        run_match(
            a=a,
            b=b,
            pairs=args.pairs,
            seed_start=args.seed_start,
            max_plies=args.max_plies,
            out_csv=out_csv,
            out_json=out_json,
            progress_every=args.progress_every,
            git_commit=git_commit,
        )
        return

    if args.cmd == "roundrobin":
        agents = [
            _parse_agent_token(token, args.default_depth, args.default_q_depth)
            for token in args.agents
        ]
        run_roundrobin(
            agents=agents,
            pairs=args.pairs,
            seed_start=args.seed_start,
            max_plies=args.max_plies,
            out_csv=out_csv,
            out_json=out_json,
            progress_every=args.progress_every,
            git_commit=git_commit,
            offset_seeds_by_matchup=args.offset_seeds_by_matchup,
        )
        return

    raise RuntimeError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
