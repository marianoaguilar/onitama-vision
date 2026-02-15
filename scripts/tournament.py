from __future__ import annotations

"""Onitama tournament runner (AI vs AI) focused on heuristic evaluation.

Key features:
  - Paired seeds + color swap (fair comparison): for each seed we play two games
    (A as RED vs B as BLUE, then B as RED vs A as BLUE).
  - CSV export of every game (reproducible experiments).
  - Optional round-robin across multiple agents.

Examples
--------
Head-to-head (paired, recommended):

  python scripts/tournament.py match \
      --a v2b@3 --b v2c@3 \
      --pairs 200 --seed-start 0 --max-plies 300 \
      --out results_v2b_vs_v2c.csv

Round-robin:

  python scripts/tournament.py roundrobin \
      --agents v1@3 v2a@3 v2b@3 v2c@3 \
      --pairs 150 --seed-start 0 --max-plies 300 \
      --out rr_depth3.csv
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from collections.abc import Callable
from pathlib import Path
from typing import Iterable, Optional


# -----------------------------------------------------------------------------
# Make running from repo root robust (without requiring editable install)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


# -----------------------------------------------------------------------------

from onitama.ai.agent import choose_action
from onitama.ai.evaluate import get_evaluator, EVALUATORS
from onitama.ai.types import TranspositionTable
from onitama.engine.pieces import Player
from onitama.engine.rules import Action, apply_action, winner
from onitama.engine.state import GameState


# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentSpec:
    """Configuration for an AI agent."""

    label: str
    evaluator_name: str
    depth: int

    def validate(self) -> None:
        if self.depth < 1:
            raise ValueError("depth must be >= 1")
        # This will raise a nice error if unknown.
        get_evaluator(self.evaluator_name)


@dataclass(frozen=True)
class SearchConfig:
    use_tt: bool
    use_iterative_deepening: bool
    aspiration_window: int | None
    q_depth: int


@dataclass(frozen=True)
class CompiledAgent:
    """Runtime-ready agent (evaluator cached)."""

    label: str
    depth: int
    evaluator_name: str
    evaluator: Callable[[GameState, Player], int]
    search: SearchConfig
    tt: TranspositionTable

    def select_action(self, state: GameState) -> Action:
        action = choose_action(
            state,
            depth=self.depth,
            evaluator=self.evaluator,
            use_tt=self.search.use_tt,
            tt=self.tt,
            use_iterative_deepening=self.search.use_iterative_deepening,
            aspiration_window=self.search.aspiration_window,
            q_depth=self.search.q_depth,
        )
        assert action is not None, "Agent asked to move in a terminal state."
        return action


@dataclass
class GameRecord:
    game_id: int
    pair_id: int
    seed: int
    swapped_colors: bool
    red_agent: str
    blue_agent: str
    winner: str  # "RED" / "BLUE" / "DRAW"
    reason: str  # winner() reason or "Ply cap"
    plies: int
    starting_player: str  # "RED" or "BLUE" (from side card stamp)


@dataclass
class Aggregate:
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    points: float = 0.0  # win=1, draw=0.5
    total_plies: int = 0

    def add(self, result: str, plies: int) -> None:
        self.games += 1
        self.total_plies += plies
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


def _parse_agent_token(token: str, default_depth: int) -> AgentSpec:
    """Parse strings like:

    - "v2b@3"  -> evaluator=v2b, depth=3, label=v2b@3
    - "v1"     -> evaluator=v1, depth=default_depth, label=v1@<default>

    The label is just for reporting.
    """
    token = token.strip()
    if not token:
        raise ValueError("Empty agent token")

    if "@" in token:
        name, depth_str = token.split("@", 1)
        name = name.strip()
        depth = int(depth_str.strip())
    else:
        name = token
        depth = default_depth

    label = f"{name}@{depth}"
    spec = AgentSpec(label=label, evaluator_name=name, depth=depth)
    spec.validate()
    return spec


def _compile_agent(spec: AgentSpec, search: SearchConfig) -> CompiledAgent:
    evaluator = get_evaluator(spec.evaluator_name)
    return CompiledAgent(
        label=spec.label,
        depth=spec.depth,
        evaluator_name=spec.evaluator_name,
        evaluator=evaluator,
        search=search,
        tt={},
    )


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
        left.add("DRAW", game.plies)
        right.add("DRAW", game.plies)
        return

    left.add("WIN" if game.winner == left_color else "LOSS", game.plies)
    right.add("WIN" if game.winner == right_color else "LOSS", game.plies)


def _build_search_config(args: argparse.Namespace) -> SearchConfig:
    if args.q_depth < 0:
        raise ValueError("q_depth must be >= 0")
    if args.aspiration_window < -1:
        raise ValueError("aspiration_window must be >= -1")

    aspiration_window = None if args.aspiration_window == -1 else args.aspiration_window
    return SearchConfig(
        use_tt=not args.no_tt,
        use_iterative_deepening=not args.no_iterative_deepening,
        aspiration_window=aspiration_window,
        q_depth=args.q_depth,
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
    while True:
        out = winner(state)
        if out is not None:
            w, reason = out
            w_str = "RED" if w == Player.RED else "BLUE"
            return GameRecord(
                game_id=game_id,
                pair_id=pair_id,
                seed=seed,
                swapped_colors=swapped_colors,
                red_agent=red.label,
                blue_agent=blue.label,
                winner=w_str,
                reason=reason,
                plies=plies,
                starting_player=starting_player,
            )

        if plies >= max_plies:
            return GameRecord(
                game_id=game_id,
                pair_id=pair_id,
                seed=seed,
                swapped_colors=swapped_colors,
                red_agent=red.label,
                blue_agent=blue.label,
                winner="DRAW",
                reason="Ply cap",
                plies=plies,
                starting_player=starting_player,
            )

        mover = state.to_move
        agent = red if mover == Player.RED else blue
        action = agent.select_action(state)
        state = apply_action(state, action)
        plies += 1


def _write_csv(path: Path, records: Iterable[GameRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
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
            ]
        )
        for r in records:
            w.writerow(
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
                ]
            )


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
    total_games = agg_a.games  # same for both
    assert total_games == 2 * pairs

    print("")
    print("=== Match summary (paired + color swap) ===")
    print(f"Pairs       : {pairs} (2 games per pair => {total_games} games)")
    print(f"Seed start  : {seed_start}")
    print(f"Max plies   : {max_plies}")
    print("")
    print(f"A: {a.label}  (eval={a.evaluator_name}, depth={a.depth})")
    print(f"B: {b.label}  (eval={b.evaluator_name}, depth={b.depth})")
    print("")
    print("Results (win=1, draw=0.5):")
    print(
        f"  A points  : {agg_a.points:.1f} / {total_games}  (rate={agg_a.score_rate():.3f})"
    )
    print(
        f"  B points  : {agg_b.points:.1f} / {total_games}  (rate={agg_b.score_rate():.3f})"
    )
    print("")
    print(f"  A W/D/L   : {agg_a.wins}/{agg_a.draws}/{agg_a.losses}")
    print(f"  B W/D/L   : {agg_b.wins}/{agg_b.draws}/{agg_b.losses}")
    print("")
    print(f"Avg plies   : {agg_a.avg_plies():.1f}")
    print(f"Elapsed     : {elapsed:.2f}s")
    if elapsed > 0:
        print(f"Games/sec   : {total_games/elapsed:.2f}")


def run_match(
    *,
    a: AgentSpec,
    b: AgentSpec,
    pairs: int,
    seed_start: int,
    max_plies: int,
    out_csv: Optional[Path],
    out_json: Optional[Path],
    progress_every: int,
    search: SearchConfig,
    git_commit: str | None,
) -> None:
    if pairs < 1:
        raise ValueError("pairs must be >= 1")

    agent_a = _compile_agent(a, search)
    agent_b = _compile_agent(b, search)

    records: list[GameRecord] = []
    agg_a = Aggregate()
    agg_b = Aggregate()

    t0 = time.perf_counter()
    game_id = 0
    for pair_idx in range(pairs):
        seed = seed_start + pair_idx
        pair_id = pair_idx

        # Game 1: A as RED, B as BLUE
        g1 = play_game(
            seed=seed,
            red=agent_a,
            blue=agent_b,
            max_plies=max_plies,
            game_id=game_id,
            pair_id=pair_id,
            swapped_colors=False,
        )
        game_id += 1

        # Game 2: B as RED, A as BLUE
        g2 = play_game(
            seed=seed,
            red=agent_b,
            blue=agent_a,
            max_plies=max_plies,
            game_id=game_id,
            pair_id=pair_id,
            swapped_colors=True,
        )
        game_id += 1

        records.extend([g1, g2])

        # Update aggregates (from perspective of A and B)
        for g in (g1, g2):
            _update_pair_aggregates(g, agent_a.label, agent_b.label, agg_a, agg_b)

        if progress_every > 0 and (pair_idx + 1) % progress_every == 0:
            done_pairs = pair_idx + 1
            print(
                f"Progress: {done_pairs}/{pairs} pairs "
                f"(games={2*done_pairs}) | A rate={agg_a.score_rate():.3f}"
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
            "agent_a": {"label": a.label, "eval": a.evaluator_name, "depth": a.depth},
            "agent_b": {"label": b.label, "eval": b.evaluator_name, "depth": b.depth},
            "search": {
                "use_tt": search.use_tt,
                "use_iterative_deepening": search.use_iterative_deepening,
                "aspiration_window": search.aspiration_window,
                "q_depth": search.q_depth,
            },
            "git_commit": git_commit,
            "summary": {
                "a": {
                    "games": agg_a.games,
                    "wins": agg_a.wins,
                    "draws": agg_a.draws,
                    "losses": agg_a.losses,
                    "points": agg_a.points,
                    "score_rate": agg_a.score_rate(),
                    "avg_plies": agg_a.avg_plies(),
                },
                "b": {
                    "games": agg_b.games,
                    "wins": agg_b.wins,
                    "draws": agg_b.draws,
                    "losses": agg_b.losses,
                    "points": agg_b.points,
                    "score_rate": agg_b.score_rate(),
                    "avg_plies": agg_b.avg_plies(),
                },
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
    out_csv: Optional[Path],
    out_json: Optional[Path],
    progress_every: int,
    search: SearchConfig,
    git_commit: str | None,
    offset_seeds_by_matchup: bool,
) -> None:
    if len(agents) < 2:
        raise ValueError("Need at least 2 agents for round-robin")
    labels = [a.label for a in agents]
    if len(set(labels)) != len(labels):
        raise ValueError("Round-robin agents must have unique labels (eval@depth)")

    compiled = {a.label: _compile_agent(a, search) for a in agents}
    scoreboard: dict[str, Aggregate] = {a.label: Aggregate() for a in agents}
    matchups: list[dict[str, object]] = []

    all_records: list[GameRecord] = []

    t0 = time.perf_counter()
    game_id = 0
    matchup_idx = 0
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            a = agents[i]
            b = agents[j]
            matchup_idx += 1

            # Run a paired match between a and b
            local_records: list[GameRecord] = []
            local_a = Aggregate()
            local_b = Aggregate()

            for pair_idx in range(pairs):
                if offset_seeds_by_matchup:
                    seed = seed_start + (matchup_idx - 1) * pairs + pair_idx
                else:
                    seed = seed_start + pair_idx
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

                local_records.extend([g1, g2])

                for g in (g1, g2):
                    _update_pair_aggregates(g, a.label, b.label, local_a, local_b)

            # Merge to global scoreboard
            scoreboard[a.label].games += local_a.games
            scoreboard[a.label].wins += local_a.wins
            scoreboard[a.label].losses += local_a.losses
            scoreboard[a.label].draws += local_a.draws
            scoreboard[a.label].points += local_a.points
            scoreboard[a.label].total_plies += local_a.total_plies

            scoreboard[b.label].games += local_b.games
            scoreboard[b.label].wins += local_b.wins
            scoreboard[b.label].losses += local_b.losses
            scoreboard[b.label].draws += local_b.draws
            scoreboard[b.label].points += local_b.points
            scoreboard[b.label].total_plies += local_b.total_plies

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
                }
            )

            all_records.extend(local_records)

            if progress_every > 0 and matchup_idx % progress_every == 0:
                print(
                    f"Progress: finished {matchup_idx} matchups / {len(agents)*(len(agents)-1)//2}"
                )

    elapsed = time.perf_counter() - t0

    # Print table-like summary
    print("")
    print("=== Round-robin summary (paired + color swap per matchup) ===")
    print(f"Agents     : {len(agents)}")
    print(f"Pairs/match: {pairs} (2 games per pair)")
    print(f"Seed start : {seed_start}")
    print(f"Max plies  : {max_plies}")
    print(f"Elapsed    : {elapsed:.2f}s")
    print("")

    # Rank by points
    ranking = sorted(scoreboard.items(), key=lambda kv: kv[1].points, reverse=True)
    for rank, (label, agg) in enumerate(ranking, start=1):
        print(
            f"{rank:>2}. {label:<12} "
            f"points={agg.points:>6.1f}  rate={agg.score_rate():.3f}  "
            f"W/D/L={agg.wins}/{agg.draws}/{agg.losses}  avg_plies={agg.avg_plies():.1f}"
        )

    if out_csv is not None:
        _write_csv(out_csv, all_records)
        print(f"\nCSV written: {out_csv}")

    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mode": "roundrobin",
            "pairs_per_match": pairs,
            "seed_start": seed_start,
            "offset_seeds_by_matchup": offset_seeds_by_matchup,
            "max_plies": max_plies,
            "search": {
                "use_tt": search.use_tt,
                "use_iterative_deepening": search.use_iterative_deepening,
                "aspiration_window": search.aspiration_window,
                "q_depth": search.q_depth,
            },
            "git_commit": git_commit,
            "agents": [
                {"label": a.label, "eval": a.evaluator_name, "depth": a.depth} for a in agents
            ],
            "scoreboard": {
                label: {
                    "games": agg.games,
                    "wins": agg.wins,
                    "draws": agg.draws,
                    "losses": agg.losses,
                    "points": agg.points,
                    "score_rate": agg.score_rate(),
                    "avg_plies": agg.avg_plies(),
                }
                for label, agg in scoreboard.items()
            },
            "matchups": matchups,
            "elapsed_sec": elapsed,
        }
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"JSON written: {out_json}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Onitama tournament runner for heuristic evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    def _add_search_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--no-tt",
            action="store_true",
            help="Disable transposition table during search.",
        )
        p.add_argument(
            "--no-iterative-deepening",
            action="store_true",
            help="Disable iterative deepening (single search at target depth).",
        )
        p.add_argument(
            "--aspiration-window",
            type=int,
            default=100,
            help="Aspiration window size; use -1 to disable.",
        )
        p.add_argument(
            "--q-depth",
            type=int,
            default=2,
            help="Quiescence capture extension depth.",
        )

    p_match = sub.add_parser("match", help="Run a head-to-head match between two agents.")
    p_match.add_argument(
        "--max-plies",
        type=int,
        default=300,
        help="Game is declared DRAW if this ply cap is reached.",
    )
    p_match.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="First seed. Each pair uses seed_start + i.",
    )
    p_match.add_argument(
        "--pairs",
        type=int,
        default=200,
        help="Number of paired seeds (each pair = 2 games with color swap).",
    )
    p_match.add_argument(
        "--out",
        type=str,
        default=None,
        help="Write per-game CSV to this path (optional).",
    )
    p_match.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Write summary JSON to this path (optional).",
    )
    p_match.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress every N pairs. Set 0 to disable.",
    )
    p_match.add_argument(
        "--a",
        required=True,
        help="Agent A spec: <evaluator>@<depth>  e.g. v2b@3",
    )
    p_match.add_argument(
        "--b",
        required=True,
        help="Agent B spec: <evaluator>@<depth>  e.g. v2c@3",
    )
    p_match.add_argument(
        "--default-depth",
        type=int,
        default=3,
        help="Depth used if an agent spec omits @depth.",
    )
    _add_search_args(p_match)

    p_rr = sub.add_parser("roundrobin", help="Run a round-robin across multiple agents.")
    p_rr.add_argument(
        "--max-plies",
        type=int,
        default=300,
        help="Game is declared DRAW if this ply cap is reached.",
    )
    p_rr.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="First seed. Each pair uses seed_start + i.",
    )
    p_rr.add_argument(
        "--pairs",
        type=int,
        default=200,
        help="Number of paired seeds per matchup (each pair = 2 games with color swap).",
    )
    p_rr.add_argument(
        "--out",
        type=str,
        default=None,
        help="Write per-game CSV to this path (optional).",
    )
    p_rr.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Write summary JSON to this path (optional).",
    )
    p_rr.add_argument(
        "--progress-every",
        type=int,
        default=2,
        help="Print progress every N matchups. Set 0 to disable.",
    )
    p_rr.add_argument(
        "--agents",
        nargs="+",
        required=True,
        help="Agent specs: v1@3 v2a@3 v2b@3 ...",
    )
    p_rr.add_argument(
        "--default-depth",
        type=int,
        default=3,
        help="Depth used if an agent spec omits @depth.",
    )
    p_rr.add_argument(
        "--offset-seeds-by-matchup",
        action="store_true",
        help="Use different seed ranges per matchup to reduce overfitting to one seed block.",
    )
    _add_search_args(p_rr)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    out_csv = Path(args.out) if args.out else None
    out_json = Path(args.out_json) if args.out_json else None

    # Fast sanity check: show available evaluators (useful when you typo names)
    if args.cmd in {"match", "roundrobin"}:
        # no side effects, just informative if a crash happens
        _ = sorted(EVALUATORS)
    search = _build_search_config(args)
    git_commit = _git_commit_hash()

    if args.cmd == "match":
        a = _parse_agent_token(args.a, default_depth=args.default_depth)
        b = _parse_agent_token(args.b, default_depth=args.default_depth)
        run_match(
            a=a,
            b=b,
            pairs=args.pairs,
            seed_start=args.seed_start,
            max_plies=args.max_plies,
            out_csv=out_csv,
            out_json=out_json,
            progress_every=args.progress_every,
            search=search,
            git_commit=git_commit,
        )

    elif args.cmd == "roundrobin":
        agents = [_parse_agent_token(t, default_depth=args.default_depth) for t in args.agents]
        run_roundrobin(
            agents=agents,
            pairs=args.pairs,
            seed_start=args.seed_start,
            max_plies=args.max_plies,
            out_csv=out_csv,
            out_json=out_json,
            progress_every=args.progress_every,
            search=search,
            git_commit=git_commit,
            offset_seeds_by_matchup=args.offset_seeds_by_matchup,
        )

    else:
        raise RuntimeError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
