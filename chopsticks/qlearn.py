"""Tabular Q-learning agents.

Q-values are stored as ``Q[state_key][move_key]`` and are always from the point
of view of the player to move. Unseen entries default to 0.

Two trainers:

* :func:`train_naive` (Agent A): the agent plays one seat (alternating each
  episode) against a uniform-random opponent. Standard Q-learning: the target
  is ``r + gamma * max_a' Q(s', a')`` where ``s'`` is the agent's *next* turn.
* :func:`train_minimax` (Agent B): self-play sharing one table, with a negamax
  target because the next turn belongs to the opponent:
  ``target = reward`` at the end, else ``-gamma * max_a' Q(s', a')``.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .game import Move, State, apply_move, initial_state, state_key, successors, winner
from .rules import Rules

QDict = dict[str, dict[str, float]]


class QTable:
    """Sparse Q-table keyed by canonical state and move keys."""

    def __init__(self, data: Optional[QDict] = None) -> None:
        self.data: QDict = data if data is not None else {}

    def get(self, state: State, move: Move) -> float:
        """Q(s, a), defaulting to 0."""
        return self.data.get(state_key(state), {}).get(move.key, 0.0)

    def set(self, state: State, move: Move, value: float) -> None:
        """Assign Q(s, a)."""
        self.data.setdefault(state_key(state), {})[move.key] = value

    def best_value(self, state: State, rules: Rules) -> float:
        """max_a Q(s, a) over legal moves (0 if there are none)."""
        moves = [m for m, _ in successors(state, rules)]
        if not moves:
            return 0.0
        return max(self.get(state, m) for m in moves)

    def greedy_moves(self, state: State, rules: Rules, tol: float = 1e-9) -> list[Move]:
        """All legal moves tied for the highest Q-value."""
        moves = [m for m, _ in successors(state, rules)]
        if not moves:
            return []
        qs = [self.get(state, m) for m in moves]
        best = max(qs)
        return [m for m, q in zip(moves, qs) if q >= best - tol]

    def copy(self) -> "QTable":
        """Deep copy."""
        return QTable({s: dict(row) for s, row in self.data.items()})

    def save(self, path: Path, meta: Optional[dict] = None) -> None:
        """Write the table (and optional metadata) as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        rounded = {s: {m: round(q, 6) for m, q in row.items()} for s, row in self.data.items()}
        path.write_text(json.dumps({"meta": meta or {}, "q": rounded}, sort_keys=True))

    @classmethod
    def load(cls, path: Path) -> "QTable":
        """Read a table written by :meth:`save`."""
        return cls(json.loads(Path(path).read_text())["q"])


# --- update math (kept as pure functions so they can be unit-tested) ---------


def q_update(old: float, target: float, alpha: float) -> float:
    """Q(s,a) <- Q(s,a) + alpha * (target - Q(s,a))."""
    return old + alpha * (target - old)


def naive_target(reward: float, next_best: float, gamma: float, done: bool) -> float:
    """Standard Q-learning target: r if the episode ended, else r + gamma * max Q(s')."""
    return reward if done else reward + gamma * next_best


def negamax_target(reward: float, next_best: float, gamma: float, done: bool) -> float:
    """Negamax target: reward at the end, else -gamma * max_a' Q(s', a') (opponent's turn)."""
    return reward if done else -gamma * next_best


# --- training ---------------------------------------------------------------


@dataclass
class TrainConfig:
    """Hyperparameters for either trainer."""

    episodes: int = 30_000
    alpha: float = 0.2
    # gamma < 1 makes a faster win worth more than a slower one. With gamma = 1 every
    # winning move scores +1, so the agent keeps winning positions but can cycle
    # without ever finishing (see the ablation in results/results.md).
    gamma: float = 0.95
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_frac: float = 0.8  # fraction of episodes over which epsilon decays linearly
    seed: int = 0
    checkpoint_fracs: list[float] = field(
        default_factory=lambda: [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
        + [i / 10 for i in range(1, 11)]
    )

    def epsilon(self, episode: int) -> float:
        """Exploration rate for a given episode index."""
        span = max(1, int(self.episodes * self.eps_decay_frac))
        frac = min(1.0, episode / span)
        return self.eps_start + (self.eps_end - self.eps_start) * frac

    def checkpoint_episodes(self) -> list[int]:
        """Episode counts (after which) checkpoints are taken."""
        return sorted({max(1, round(f * self.episodes)) for f in self.checkpoint_fracs})

    def to_dict(self) -> dict:
        return asdict(self)


CheckpointFn = Callable[[int, QTable], None]


def _choose(q: QTable, state: State, rules: Rules, eps: float, rng: random.Random) -> Move:
    if rng.random() < eps:
        return rng.choice([m for m, _ in successors(state, rules)])
    return rng.choice(q.greedy_moves(state, rules))


def _random_move(state: State, rules: Rules, rng: random.Random) -> Move:
    return rng.choice([m for m, _ in successors(state, rules)])


def train_naive(
    rules: Rules, cfg: TrainConfig, on_checkpoint: Optional[CheckpointFn] = None
) -> QTable:
    """Train Agent A against a uniform-random opponent."""
    rng = random.Random(cfg.seed)
    q = QTable()
    checkpoints = set(cfg.checkpoint_episodes())
    for ep in range(cfg.episodes):
        eps = cfg.epsilon(ep)
        s = initial_state()
        plies = 0
        if ep % 2 == 1:  # random player moves first on odd episodes
            s = apply_move(s, _random_move(s, rules, rng), rules)
            plies += 1
        while True:
            a = _choose(q, s, rules, eps, rng)
            s1 = apply_move(s, a, rules)
            plies += 1
            if winner(s1, rules) is not None:  # opponent to move has lost
                q.set(s, a, q_update(q.get(s, a), naive_target(1.0, 0.0, cfg.gamma, True), cfg.alpha))
                break
            if plies >= rules.max_moves:
                q.set(s, a, q_update(q.get(s, a), 0.0, cfg.alpha))
                break
            s2 = apply_move(s1, _random_move(s1, rules, rng), rules)
            plies += 1
            if winner(s2, rules) is not None:  # agent to move has lost
                q.set(s, a, q_update(q.get(s, a), naive_target(-1.0, 0.0, cfg.gamma, True), cfg.alpha))
                break
            if plies >= rules.max_moves:
                q.set(s, a, q_update(q.get(s, a), 0.0, cfg.alpha))
                break
            target = naive_target(0.0, q.best_value(s2, rules), cfg.gamma, False)
            q.set(s, a, q_update(q.get(s, a), target, cfg.alpha))
            s = s2
        if on_checkpoint and (ep + 1) in checkpoints:
            on_checkpoint(ep + 1, q.copy())
    return q


def train_minimax(
    rules: Rules, cfg: TrainConfig, on_checkpoint: Optional[CheckpointFn] = None
) -> QTable:
    """Train Agent B by self-play with a shared table and negamax targets."""
    rng = random.Random(cfg.seed)
    q = QTable()
    checkpoints = set(cfg.checkpoint_episodes())
    for ep in range(cfg.episodes):
        eps = cfg.epsilon(ep)
        s = initial_state()
        plies = 0
        while True:
            a = _choose(q, s, rules, eps, rng)
            s1 = apply_move(s, a, rules)
            plies += 1
            result = winner(s1, rules)  # from the new mover's perspective
            if result is not None:
                target = negamax_target(-float(result), 0.0, cfg.gamma, True)
                q.set(s, a, q_update(q.get(s, a), target, cfg.alpha))
                break
            if plies >= rules.max_moves:
                q.set(s, a, q_update(q.get(s, a), 0.0, cfg.alpha))
                break
            target = negamax_target(0.0, q.best_value(s1, rules), cfg.gamma, False)
            q.set(s, a, q_update(q.get(s, a), target, cfg.alpha))
            s = s1
        if on_checkpoint and (ep + 1) in checkpoints:
            on_checkpoint(ep + 1, q.copy())
    return q


TRAINERS = {"naive": train_naive, "minimax": train_minimax}


def train_and_save(
    agent: str,
    preset: str,
    rules: Rules,
    cfg: TrainConfig,
    out_dir: Path,
    save_as: Optional[str] = None,
    checkpoints: bool = True,
) -> tuple[QTable, dict[int, QTable]]:
    """Train one agent, saving the final table and every checkpoint under ``out_dir``.

    Layout: ``out_dir/<preset>/<name>.json`` and
    ``out_dir/<preset>/checkpoints/<name>_<episodes>.json`` where ``name`` is
    ``save_as`` (default: the agent type).
    """
    name = save_as or agent
    ckpts: dict[int, QTable] = {}
    base = out_dir / preset
    meta = {"agent": agent, "preset": preset, "rules": rules.to_dict(), "config": cfg.to_dict()}
    if checkpoints:  # drop checkpoints from earlier runs so curves don't mix runs
        for old in (base / "checkpoints").glob(f"{name}_*.json"):
            old.unlink()

    def on_ckpt(ep: int, table: QTable) -> None:
        ckpts[ep] = table
        table.save(base / "checkpoints" / f"{name}_{ep}.json", {**meta, "episodes": ep})

    q = TRAINERS[agent](rules, cfg, on_ckpt if checkpoints else None)
    q.save(base / f"{name}.json", {**meta, "episodes": cfg.episodes})
    return q, ckpts


def load_checkpoints(agent: str, preset: str, model_dir: Path) -> dict[int, QTable]:
    """Load all saved checkpoints for an agent, keyed by episode count."""
    out = {}
    for p in (model_dir / preset / "checkpoints").glob(f"{agent}_*.json"):
        out[int(p.stem.split("_")[-1])] = QTable.load(p)
    return dict(sorted(out.items()))
