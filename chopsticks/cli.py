"""Command-line interface.

    python -m chopsticks solve
    python -m chopsticks train --agent minimax --preset standard
    python -m chopsticks evaluate
    python -m chopsticks export
    python -m chopsticks all          # solve -> train -> evaluate -> export
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Optional, Sequence

from .game import initial_state
from .qlearn import QTable, TrainConfig, load_checkpoints, train_and_save
from .rules import PRESETS
from .solver import Solution, Value, report, solve

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
WEB_DATA = ROOT / "web" / "data"

AGENTS = ("naive", "minimax")
DEFAULT_EPISODES = 100_000
DEFAULT_GAMMA = TrainConfig.gamma
ABLATION_NAME = "minimax_gamma1"


def _presets(arg: str) -> list[str]:
    return list(PRESETS) if arg == "all" else [arg]


def cmd_solve(args: argparse.Namespace) -> dict[str, Solution]:
    sols = {name: solve(PRESETS[name]) for name in _presets(args.preset)}
    print(report(sols))
    return sols


def cmd_train(args: argparse.Namespace) -> None:
    agents = AGENTS if args.agent == "all" else (args.agent,)
    for preset in _presets(args.preset):
        for agent in agents:
            cfg = TrainConfig(episodes=args.episodes, seed=args.seed, alpha=args.alpha,
                              gamma=args.gamma)
            t = time.time()
            train_and_save(agent, preset, PRESETS[preset], cfg, args.models)
            print(f"trained {agent:<8} on {preset:<11} ({cfg.episodes:,} episodes, "
                  f"γ={cfg.gamma}, {time.time() - t:.1f}s)")


def cmd_ablation(args: argparse.Namespace) -> None:
    """Retrain Minimax B with gamma = 1.0 (no checkpoints) for the comparison in results.md."""
    for preset in _presets(args.preset):
        cfg = TrainConfig(episodes=args.episodes, seed=args.seed, alpha=args.alpha, gamma=1.0)
        t = time.time()
        train_and_save("minimax", preset, PRESETS[preset], cfg, args.models,
                       save_as=ABLATION_NAME, checkpoints=False)
        print(f"trained minimax  on {preset:<11} (γ=1.0 comparison, {time.time() - t:.1f}s)")


def _load_preset_models(preset: str, models: Path) -> tuple[dict[str, QTable], dict[str, dict[int, QTable]]]:
    tables, ckpts = {}, {}
    for agent in AGENTS:
        path = models / preset / f"{agent}.json"
        if not path.exists():
            raise SystemExit(f"missing {path}; run `python -m chopsticks train` first")
        tables[agent] = QTable.load(path)
        ckpts[agent] = load_checkpoints(agent, preset, models)
    return tables, ckpts


def cmd_evaluate(args: argparse.Namespace) -> None:
    from . import evaluate as ev

    reports = []
    for preset in _presets(args.preset):
        sol = solve(PRESETS[preset])
        tables, ckpts = _load_preset_models(preset, args.models)
        total = max(ckpts["minimax"])
        easy_ep = ev.pick_easy_checkpoint(ckpts["minimax"], sol)
        g1_path = args.models / preset / f"{ABLATION_NAME}.json"
        g1 = QTable.load(g1_path) if g1_path.exists() else None
        t = time.time()
        rep = ev.evaluate_preset(preset, sol, tables, ckpts, ckpts["minimax"][easy_ep], easy_ep,
                                 total, games=args.games, seed=args.seed, gamma1_table=g1)
        reports.append(rep)
        a, b = rep.agents["naive"], rep.agents["minimax"]
        print(f"evaluated {preset:<11} naive {a.strict:.1%}/{a.preserving:.1%}  "
              f"minimax {b.strict:.1%}/{b.preserving:.1%}  ({time.time() - t:.1f}s)")
    args.results.mkdir(parents=True, exist_ok=True)
    ev.plot_learning_curves(reports, args.results / "learning_curves.png")
    ev.plot_accuracy_bars(reports, args.results / "accuracy.png")
    for rep in reports:
        if rep.preset == "standard":
            ev.plot_tournament(rep, args.results / "tournament_standard.png")
    ev.write_results_md(reports, args.results / "results.md", commentary(reports), DEFAULT_GAMMA)
    print(f"wrote {args.results / 'results.md'} and charts")


def commentary(reports) -> str:
    """Plain-language findings, generated from the numbers."""
    by = {r.preset: r for r in reports}
    lines = []
    start = initial_state()
    if "standard" in by and "no_suicide" in by:
        s, n = by["standard"].solution, by["no_suicide"].solution
        sv, nv = s.value[start], n.value[start]
        standard = (
            f"Under standard rules the starting position is a {sv.value} for the first player"
            + (f" (the second player forces a win within {s.distance[start]} plies)"
               if sv is Value.LOSS else "")
        )
        if sv is nv:
            lines.append(f"- **Suicide splits don't change the result.** {standard}, and it is "
                         f"still a {nv.value} with suicide splits banned.")
        else:
            lines.append(f"- **Suicide splits decide the game.** {standard}. Banning suicide "
                         f"splits makes it a {nv.value}.")
    for r in reports:
        v = r.solution.value[start]
        d = r.solution.distance[start]
        if r.preset in ("standard", "no_suicide"):
            continue
        lines.append(f"- **{r.preset}**: start is {v.value} for the first player"
                     + (f" in {d} plies." if d is not None else "."))

    def avg(key, attr):
        return sum(getattr(r.agents[key], attr) for r in reports) / len(reports)

    lines.append(
        f"- **Accuracy (mean over presets).** Naive (A): {avg('naive', 'strict'):.1%} optimal, "
        f"{avg('naive', 'preserving'):.1%} value-preserving, {avg('naive', 'value_agreement'):.1%} "
        f"value agreement. Minimax (B): {avg('minimax', 'strict'):.1%} optimal, "
        f"{avg('minimax', 'preserving'):.1%} value-preserving, "
        f"{avg('minimax', 'value_agreement'):.1%} value agreement."
    )
    for r in reports:
        t = r.tournament
        b_vs_p = t[("Minimax (B)", "Perfect")]
        a_vs_p = t[("Naive (A)", "Perfect")]
        a_vs_r = t[("Naive (A)", "Random")]
        b_vs_r = t[("Minimax (B)", "Random")]
        lines.append(
            f"- **{r.preset} tournament** (W/D/L): A vs Random {a_vs_r}, B vs Random {b_vs_r}, "
            f"A vs Perfect {a_vs_p}, B vs Perfect {b_vs_p}."
        )
    return "\n".join(lines)


def cmd_export(args: argparse.Namespace) -> None:
    from .evaluate import pick_easy_checkpoint
    from .export import export_all

    sols, policies, easy = {}, {}, {}
    for preset in PRESETS:
        sols[preset] = solve(PRESETS[preset])
        tables, ckpts = _load_preset_models(preset, args.models)
        easy_ep = pick_easy_checkpoint(ckpts["minimax"], sols[preset])
        easy[preset] = easy_ep
        policies[preset] = {"easy": ckpts["minimax"][easy_ep], **tables}
    for p in export_all(sols, policies, args.web_data, easy):
        print(f"wrote {p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}")


def cmd_all(args: argparse.Namespace) -> None:
    t = time.time()
    args.preset = "all"
    print("== solve =="); cmd_solve(args)
    args.agent = "all"
    print("== train =="); cmd_train(args)
    if not args.skip_ablation:
        print("== γ=1.0 comparison =="); cmd_ablation(args)
    print("== evaluate =="); cmd_evaluate(args)
    print("== export =="); cmd_export(args)
    print(f"done in {time.time() - t:.0f}s")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="chopsticks", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", type=Path, default=MODELS, help="where Q-tables are stored")
    p.add_argument("--results", type=Path, default=RESULTS, help="charts and results.md")
    p.add_argument("--web-data", type=Path, default=WEB_DATA, help="JSON for the web game")
    sub = p.add_subparsers(dest="cmd", required=True)
    preset_choices = ["all", *PRESETS]

    s = sub.add_parser("solve", help="solve presets and print a summary")
    s.add_argument("--preset", choices=preset_choices, default="all")
    s.set_defaults(func=cmd_solve)

    def add_train_args(t: argparse.ArgumentParser) -> None:
        t.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
        t.add_argument("--alpha", type=float, default=0.2)
        t.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
        t.add_argument("--seed", type=int, default=0)

    t = sub.add_parser("train", help="train Q-learning agents")
    t.add_argument("--agent", choices=["all", *AGENTS], default="all")
    t.add_argument("--preset", choices=preset_choices, default="all")
    add_train_args(t)
    t.set_defaults(func=cmd_train)

    g = sub.add_parser("ablation", help="train Minimax B with gamma=1.0 for comparison")
    g.add_argument("--preset", choices=preset_choices, default="all")
    add_train_args(g)
    g.set_defaults(func=cmd_ablation)

    e = sub.add_parser("evaluate", help="compare agents with the solver")
    e.add_argument("--preset", choices=preset_choices, default="all")
    e.add_argument("--games", type=int, default=400, help="games per tournament pairing")
    e.add_argument("--seed", type=int, default=0)
    e.set_defaults(func=cmd_evaluate)

    x = sub.add_parser("export", help="write JSON for the web game")
    x.set_defaults(func=cmd_export)

    a = sub.add_parser("all", help="solve -> train -> evaluate -> export")
    add_train_args(a)
    a.add_argument("--games", type=int, default=400)
    a.add_argument("--skip-ablation", action="store_true",
                   help="skip retraining Minimax B with gamma=1.0")
    a.set_defaults(func=cmd_all)
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
