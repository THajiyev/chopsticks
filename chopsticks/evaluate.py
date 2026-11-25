"""Measure Q-agents against the exact solver.

Metrics (all per rule preset):

* **Move accuracy** over non-terminal reachable states. A greedy agent may have
  several moves tied for best Q; it picks among them at random, so a state's
  score is the fraction of tied moves that are correct (the *expected* accuracy).
  Two notions of "correct":
    - *strict*: one of the solver's optimal moves (fastest win / slowest loss);
    - *value-preserving*: keeps the game-theoretic value (any winning move from
      a WIN, any drawing move from a DRAW; everything counts in a LOSS).
* **Weighted move accuracy**: the same, weighted by how often each state came up
  in the agent's own games against Random and against Perfect.
* **Value agreement**: sign of ``max_a Q(s,a)`` (|Q| < DRAW_BAND counts as DRAW)
  matches the solver's WIN / LOSS / DRAW.
* **Tournament**: round-robin among Random, Easy (half-trained B), Naive (A),
  Minimax (B) and Perfect.
* **Learning curves** from the training checkpoints.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from .game import State, initial_state
from .players import PerfectPlayer, Player, QPlayer, RandomPlayer, play_game
from .qlearn import QTable
from .rules import PRESET_DESCRIPTIONS, Rules
from .solver import Solution, Value

# |Q| below this counts as "draw" when comparing an agent's values to the solver.
# With discounting, a win d plies away is worth about gamma**(d-1) (0.95**32 ~ 0.19 for
# the longest standard-rules win), while drawn positions converge to 0, so the band
# has to be narrow; 0.5 would misread long wins as draws.
DRAW_BAND = 0.05

# Reference palette (light surface): categorical slots 1-2, diverging blue<->red.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
RED = "#e34948"
NEUTRAL = "#f0efec"


# --- per-state metrics ---------------------------------------------------------


def decision_states(sol: Solution) -> list[State]:
    """Reachable states where the mover has at least one move."""
    return [s for s in sol.states if sol.optimal[s]]


def state_score(q: QTable, sol: Solution, s: State, kind: str = "strict") -> float:
    """Expected correctness of the greedy (random-tie-break) move in ``s``."""
    greedy = q.greedy_moves(s, sol.rules)
    good = set(sol.optimal[s]) if kind == "strict" else set(sol.value_preserving(s))
    return sum(m in good for m in greedy) / len(greedy)


def move_accuracy(
    q: QTable, sol: Solution, kind: str = "strict", weights: Optional[dict[State, float]] = None
) -> float:
    """Mean (optionally weighted) state score over decision states."""
    states = decision_states(sol)
    if weights is None:
        return sum(state_score(q, sol, s, kind) for s in states) / len(states)
    total = sum(weights.get(s, 0.0) for s in states)
    if total == 0:
        return float("nan")
    return sum(weights.get(s, 0.0) * state_score(q, sol, s, kind) for s in states) / total


def q_value_label(v: float) -> Value:
    """Map a Q-value to WIN / LOSS / DRAW using the draw band."""
    if v >= DRAW_BAND:
        return Value.WIN
    if v <= -DRAW_BAND:
        return Value.LOSS
    return Value.DRAW


def value_agreement(q: QTable, sol: Solution) -> float:
    """Fraction of decision states where the agent's best Q has the solver's sign."""
    states = decision_states(sol)
    return sum(q_value_label(q.best_value(s, sol.rules)) is sol.value[s] for s in states) / len(states)


def accuracy_by_value(q: QTable, sol: Solution, kind: str = "strict") -> dict[str, float]:
    """Move accuracy broken down by the solver's value of the state."""
    out = {}
    for v in Value:
        states = [s for s in decision_states(sol) if sol.value[s] is v]
        if states:
            out[v.value] = sum(state_score(q, sol, s, kind) for s in states) / len(states)
    return out


# --- games -----------------------------------------------------------------------


def visit_weights(
    agent: Player, opponents: Iterable[Player], rules: Rules, games: int, rng: random.Random
) -> dict[State, float]:
    """How often ``agent`` faced each state in ``games`` games vs each opponent."""
    counts: Counter[State] = Counter()
    for opp in opponents:
        for g in range(games):
            agent_seat = g % 2
            pair = (agent, opp) if agent_seat == 0 else (opp, agent)
            rec = play_game(pair[0], pair[1], rules, rng)
            counts.update(s for s, seat in rec.visits if seat == agent_seat)
    return dict(counts)


@dataclass
class Score:
    """Win / draw / loss counts from the row player's point of view."""

    win: int = 0
    draw: int = 0
    loss: int = 0

    @property
    def games(self) -> int:
        return self.win + self.draw + self.loss

    @property
    def net(self) -> float:
        """(wins - losses) / games, in [-1, 1]."""
        return (self.win - self.loss) / self.games if self.games else 0.0

    def __str__(self) -> str:
        return f"{self.win}/{self.draw}/{self.loss}"


def tournament(
    players: list[Player], rules: Rules, games: int, rng: random.Random
) -> dict[tuple[str, str], Score]:
    """Round-robin: every ordered pair (a, b), a != b, ``games`` games, seats alternating."""
    out: dict[tuple[str, str], Score] = {}
    for i, a in enumerate(players):
        for b in players[i + 1 :]:
            sa, sb = Score(), Score()
            for g in range(games):
                first, second = (a, b) if g % 2 == 0 else (b, a)
                r = play_game(first, second, rules, rng).result
                r_a = r if g % 2 == 0 else -r
                if r_a > 0:
                    sa.win += 1; sb.loss += 1
                elif r_a < 0:
                    sa.loss += 1; sb.win += 1
                else:
                    sa.draw += 1; sb.draw += 1
            out[(a.name, b.name)] = sa
            out[(b.name, a.name)] = sb
    return out


# --- per-preset evaluation ---------------------------------------------------------


@dataclass
class AgentReport:
    """All metrics for one agent on one preset."""

    name: str
    strict: float
    preserving: float
    weighted_strict: float
    weighted_preserving: float
    value_agreement: float
    by_value: dict[str, float]
    states_seen: int


@dataclass
class PresetReport:
    """Everything evaluated for one preset."""

    preset: str
    solution: Solution
    agents: dict[str, AgentReport]
    tournament: dict[tuple[str, str], Score]
    player_names: list[str]
    curves: dict[str, list[tuple[int, float, float]]]  # agent -> (episodes, strict, preserving)
    easy_episodes: int
    total_episodes: int
    # Minimax B retrained with gamma = 1.0, for comparison with the default gamma.
    gamma1: Optional[dict] = None
    notes: list[str] = field(default_factory=list)


def pick_easy_checkpoint(ckpts: dict[int, QTable], sol: Solution) -> int:
    """"Half-trained B": the earliest checkpoint with at least half the skill.

    Skill is solver-optimal move accuracy; "half" is the midpoint between an
    untrained (all-zero, i.e. uniformly random) table and the final checkpoint.
    Picking by episode count instead doesn't work: B learns most of the game in
    the first few percent of training, so a 50%-of-episodes checkpoint plays like
    the finished agent.
    """
    untrained = move_accuracy(QTable(), sol)
    final = move_accuracy(ckpts[max(ckpts)], sol)
    target = (untrained + final) / 2
    for ep in sorted(ckpts):
        if move_accuracy(ckpts[ep], sol) >= target:
            return ep
    return max(ckpts)


def evaluate_preset(
    preset: str,
    sol: Solution,
    tables: dict[str, QTable],
    checkpoints: dict[str, dict[int, QTable]],
    easy_table: QTable,
    easy_episodes: int,
    total_episodes: int,
    games: int = 400,
    weight_games: int = 300,
    seed: int = 0,
    gamma1_table: Optional[QTable] = None,
) -> PresetReport:
    """Run every metric for one preset. ``tables`` has keys ``naive`` and ``minimax``."""
    rules = sol.rules
    rng = random.Random(seed)
    rand = RandomPlayer(rules, name="Random")
    perfect = PerfectPlayer(sol, name="Perfect")
    named = {"naive": "Naive (A)", "minimax": "Minimax (B)"}

    agents: dict[str, AgentReport] = {}
    for key, table in tables.items():
        player = QPlayer(table, rules, name=named[key])
        w = visit_weights(player, [rand, perfect], rules, weight_games, rng)
        agents[key] = AgentReport(
            name=named[key],
            strict=move_accuracy(table, sol, "strict"),
            preserving=move_accuracy(table, sol, "preserving"),
            weighted_strict=move_accuracy(table, sol, "strict", w),
            weighted_preserving=move_accuracy(table, sol, "preserving", w),
            value_agreement=value_agreement(table, sol),
            by_value=accuracy_by_value(table, sol, "strict"),
            states_seen=sum(1 for s in decision_states(sol) if s in _seen_states(table)),
        )

    players: list[Player] = [
        rand,
        QPlayer(easy_table, rules, name="Easy (B half)"),
        QPlayer(tables["naive"], rules, name="Naive (A)"),
        QPlayer(tables["minimax"], rules, name="Minimax (B)"),
        perfect,
    ]
    tour = tournament(players, rules, games, rng)

    curves = {
        key: [
            (ep, move_accuracy(t, sol, "strict"), move_accuracy(t, sol, "preserving"))
            for ep, t in sorted(cks.items())
        ]
        for key, cks in checkpoints.items()
    }
    gamma1 = None
    if gamma1_table is not None:
        g1 = QPlayer(gamma1_table, rules, name="Minimax (B, γ=1)")
        t1 = tournament([g1, perfect, rand], rules, games, rng)
        gamma1 = {
            "strict": move_accuracy(gamma1_table, sol, "strict"),
            "preserving": move_accuracy(gamma1_table, sol, "preserving"),
            "vs_perfect": t1[(g1.name, "Perfect")],
            "vs_random": t1[(g1.name, "Random")],
        }
    return PresetReport(
        gamma1=gamma1,
        preset=preset,
        solution=sol,
        agents=agents,
        tournament=tour,
        player_names=[p.name for p in players],
        curves=curves,
        easy_episodes=easy_episodes,
        total_episodes=total_episodes,
    )


def _seen_states(table: QTable) -> set[State]:
    from .game import parse_state_key

    return {parse_state_key(k) for k in table.data}


# --- charts ----------------------------------------------------------------------------


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_learning_curves(reports: list[PresetReport], path: Path) -> None:
    """Small multiples: one panel per preset, strict (solid) and value-preserving (dashed)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(reports)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.3 * rows), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    colors = {"naive": BLUE, "minimax": ORANGE}
    labels = {"naive": "Naive (A)", "minimax": "Minimax (B)"}
    for ax, rep in zip(axes.flat, reports):
        _style_axes(ax)
        for key, pts in rep.curves.items():
            xs = [p[0] for p in pts]
            ax.plot(xs, [p[1] for p in pts], color=colors[key], lw=2, marker="o", ms=4,
                    label=f"{labels[key]}, optimal move")
            ax.plot(xs, [p[2] for p in pts], color=colors[key], lw=2, ls="--",
                    label=f"{labels[key]}, value-preserving")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.set_title(rep.preset, color=INK, fontsize=11, loc="left")
        ax.set_xlabel("training episodes", color=INK_2, fontsize=9)
        ax.set_ylabel("move accuracy", color=INK_2, fontsize=9)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    handles, labs = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower center", ncol=2, frameon=False, fontsize=9,
               labelcolor=INK_2)
    fig.suptitle("Move accuracy vs. training (solid: solver-optimal, dashed: value-preserving)",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_tournament(rep: PresetReport, path: Path) -> None:
    """Heatmap of net score (wins - losses) / games for row vs column."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    names = rep.player_names
    n = len(names)
    cmap = LinearSegmentedColormap.from_list("div", [RED, NEUTRAL, BLUE])
    grid = [[float("nan")] * n for _ in range(n)]
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if a != b:
                grid[i][j] = rep.tournament[(a, b)].net
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    im = ax.imshow(grid, cmap=cmap, vmin=-1, vmax=1)
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", color=MUTED)
                continue
            s = rep.tournament[(names[i], names[j])]
            ax.text(j, i, f"{s.net:+.2f}\n{s.win}/{s.draw}/{s.loss}", ha="center", va="center",
                    color="#ffffff" if abs(s.net) > 0.6 else INK, fontsize=8)
    ax.set_xticks(range(n), names, rotation=25, ha="right", color=INK_2, fontsize=9)
    ax.set_yticks(range(n), names, color=INK_2, fontsize=9)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([x - 0.5 for x in range(1, n)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, n)], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label("row player's (wins − losses) / games", color=INK_2, fontsize=9)
    cb.ax.tick_params(colors=INK_2, labelsize=8)
    cb.outline.set_visible(False)
    ax.set_title(f"Tournament, {rep.preset} rules (row vs column, W/D/L)", color=INK,
                 fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_accuracy_bars(reports: list[PresetReport], path: Path) -> None:
    """Grouped bars: final strict and value-preserving accuracy per preset and agent."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    xs = list(range(len(reports)))
    width = 0.38
    for ax, (attr, title) in zip(axes, [("strict", "Solver-optimal move"),
                                        ("preserving", "Value-preserving move")]):
        _style_axes(ax)
        ax.grid(axis="x", visible=False)
        for k, (key, color, label) in enumerate([("naive", BLUE, "Naive (A)"),
                                                  ("minimax", ORANGE, "Minimax (B)")]):
            vals = [getattr(r.agents[key], attr) for r in reports]
            pos = [x + (k - 0.5) * (width + 0.02) for x in xs]
            bars = ax.bar(pos, vals, width, color=color, label=label)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.0%}", ha="center",
                        va="bottom", fontsize=7.5, color=INK_2)
        ax.set_xticks(xs, [r.preset for r in reports], fontsize=9, color=INK_2)
        ax.set_ylim(0, 1.1)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
    axes[0].set_ylabel("share of reachable states", color=INK_2, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, labelcolor=INK_2, loc="lower center",
               ncol=2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


# --- markdown ---------------------------------------------------------------------------


def _pct(x: float) -> str:
    return f"{x:.1%}"


def solver_table(reports: list[PresetReport]) -> str:
    """Markdown table of solver results per preset."""
    lines = [
        "| Preset | Rules | Reachable states | WIN | LOSS | DRAW | Start position (first player) |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for r in reports:
        c = r.solution.counts()
        start = initial_state()
        v = r.solution.value[start]
        d = r.solution.distance[start]
        verdict = v.value + (f" in {d} plies" if d is not None else "")
        lines.append(
            f"| {r.preset} | {PRESET_DESCRIPTIONS[r.preset]} | {len(r.solution.value)} | "
            f"{c['WIN']} | {c['LOSS']} | {c['DRAW']} | {verdict} |"
        )
    return "\n".join(lines)


def agent_table(reports: list[PresetReport]) -> str:
    """Markdown table of per-agent metrics."""
    lines = [
        "| Preset | Agent | Optimal move | Value-preserving | Optimal (weighted) | "
        "Value-preserving (weighted) | Value agreement |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in reports:
        for a in r.agents.values():
            lines.append(
                f"| {r.preset} | {a.name} | {_pct(a.strict)} | {_pct(a.preserving)} | "
                f"{_pct(a.weighted_strict)} | {_pct(a.weighted_preserving)} | "
                f"{_pct(a.value_agreement)} |"
            )
    return "\n".join(lines)


def tournament_table(rep: PresetReport) -> str:
    """Markdown W/D/L matrix for one preset (row player's results)."""
    names = rep.player_names
    lines = ["| row vs column | " + " | ".join(names) + " |",
             "|---|" + "---:|" * len(names)]
    for a in names:
        cells = ["—" if a == b else str(rep.tournament[(a, b)]) for b in names]
        lines.append(f"| **{a}** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def gamma_table(reports: list[PresetReport], default_gamma: float) -> str:
    """Markdown comparison of Minimax B at the default gamma vs gamma = 1."""
    lines = [
        "| Preset | γ | Optimal move | Value-preserving | vs Perfect (W/D/L) | vs Random (W/D/L) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in reports:
        if r.gamma1 is None:
            continue
        b = r.agents["minimax"]
        lines.append(
            f"| {r.preset} | {default_gamma} | {_pct(b.strict)} | {_pct(b.preserving)} | "
            f"{r.tournament[('Minimax (B)', 'Perfect')]} | {r.tournament[('Minimax (B)', 'Random')]} |"
        )
        g = r.gamma1
        lines.append(
            f"| {r.preset} | 1.0 | {_pct(g['strict'])} | {_pct(g['preserving'])} | "
            f"{g['vs_perfect']} | {g['vs_random']} |"
        )
    return "\n".join(lines)


def write_results_md(
    reports: list[PresetReport], path: Path, commentary: str, default_gamma: float = 0.95
) -> None:
    """Write ``results/results.md``."""
    parts = [
        "# Chopsticks AI — results",
        "",
        "Generated by `python -m chopsticks all`. All numbers come from fixed seeds.",
        "",
        "## Solver",
        "",
        solver_table(reports),
        "",
        "Distances are in plies (single moves by either player). The solver ignores the "
        "move cap, so DRAW means neither side can force a result.",
        "",
        "## What we found",
        "",
        commentary,
        "",
        "## Agent accuracy vs. the solver",
        "",
        "*Optimal move*: the greedy move is one of the solver's optimal moves (fastest win / "
        "slowest loss). *Value-preserving*: the move keeps the game-theoretic result "
        "(any winning move from a WIN, any drawing move from a DRAW). Ties between equal "
        "Q-values count fractionally. *Weighted* versions weight each state by how often it "
        "came up in the agent's games against Random and against Perfect. *Value agreement*: "
        f"sign of max Q (|Q| < {DRAW_BAND} = draw) matches the solver.",
        "",
        agent_table(reports),
        "",
        "![accuracy](accuracy.png)",
        "",
        "## Learning curves",
        "",
        "![learning curves](learning_curves.png)",
        "",
    ]
    if any(r.gamma1 for r in reports):
        parts += [
            "## Why γ < 1: Minimax (B) at γ = 1.0",
            "",
            "With γ = 1 a win in 3 and a win in 30 are both worth +1, so the agent has no "
            "reason to finish. It keeps every winning position (value-preserving stays at 100%) "
            "but can circle among them until the move cap declares a draw. Discounting "
            f"(γ = {default_gamma}) makes faster wins worth more, which is what the solver's "
            "distance-to-result encodes.",
            "",
            gamma_table(reports, default_gamma),
            "",
        ]
    parts += [
        "## Tournaments",
        "",
        "Each pairing plays the listed number of games, alternating who moves first. Cells "
        "are the row player's wins/draws/losses. Easy = the earliest Minimax (B) checkpoint "
        "whose optimal-move accuracy is at least halfway between an untrained table and "
        "the finished agent.",
        "",
    ]
    for r in reports:
        games = r.tournament[(r.player_names[0], r.player_names[1])].games
        parts += [
            f"### {r.preset} ({games} games per pairing; Easy = {r.easy_episodes:,} of "
            f"{r.total_episodes:,} episodes)",
            "",
            tournament_table(r),
            "",
        ]
        if r.preset == "standard":
            parts += ["![tournament](tournament_standard.png)", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
