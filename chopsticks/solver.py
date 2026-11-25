"""Exact solver via retrograde analysis.

The game graph has cycles, so plain recursive minimax either loops or gives
wrong answers. Instead we label states backwards from the terminal ones:

* terminal states (mover has no live hands) are LOSS, distance 0;
* a state is WIN as soon as one move reaches a LOSS for the opponent;
* a state is LOSS once *every* move reaches a WIN for the opponent;
* anything never labelled is a DRAW (neither side can force a result).

States are processed in order of increasing distance (a FIFO queue), so a WIN
gets the shortest distance to victory and a LOSS gets the longest distance to
defeat — the perfect player wins fast and loses slowly.

Distances are in plies (single moves by either player). The solver ignores the
``max_moves`` cap: it describes the game without a move limit.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .game import Move, State, initial_state, is_terminal, reachable_states, successors
from .rules import PRESETS, Rules

logger = logging.getLogger(__name__)


class Value(str, Enum):
    """Game-theoretic value from the point of view of the player to move."""

    WIN = "WIN"
    LOSS = "LOSS"
    DRAW = "DRAW"

    @property
    def sign(self) -> int:
        """+1 for WIN, -1 for LOSS, 0 for DRAW."""
        return {"WIN": 1, "LOSS": -1, "DRAW": 0}[self.value]


@dataclass
class Solution:
    """Solver output for one rule set."""

    rules: Rules
    value: dict[State, Value]
    distance: dict[State, Optional[int]]  # None for DRAW
    optimal: dict[State, list[Move]] = field(default_factory=dict)

    @property
    def states(self) -> list[State]:
        """All reachable states."""
        return list(self.value)

    def counts(self) -> dict[str, int]:
        """Number of states with each value."""
        out = {v.value: 0 for v in Value}
        for v in self.value.values():
            out[v.value] += 1
        return out

    def value_preserving(self, state: State) -> list[Move]:
        """Moves that keep the game-theoretic value (ignoring distance).

        WIN: any move to an opponent LOSS. DRAW: any move to a DRAW.
        LOSS: every move (they all lose).
        """
        v = self.value[state]
        out = []
        for move, nxt in successors(state, self.rules):
            nv = self.value[nxt]
            if (
                (v is Value.WIN and nv is Value.LOSS)
                or (v is Value.DRAW and nv is Value.DRAW)
                or v is Value.LOSS
            ):
                out.append(move)
        return out


def solve(rules: Rules) -> Solution:
    """Solve every state reachable from the opening under ``rules``."""
    states = reachable_states(rules)
    preds: dict[State, list[State]] = {s: [] for s in states}
    remaining: dict[State, int] = {}
    for s in states:
        succ = successors(s, rules)
        remaining[s] = len(succ)
        for _, n in succ:
            preds[n].append(s)

    value: dict[State, Value] = {}
    distance: dict[State, Optional[int]] = {}
    queue: deque[State] = deque()
    for s in states:
        if is_terminal(s):
            value[s], distance[s] = Value.LOSS, 0
            queue.append(s)
        elif remaining[s] == 0:
            logger.warning("no legal moves in non-terminal state %s; scoring as a loss", s)
            value[s], distance[s] = Value.LOSS, 0
            queue.append(s)

    while queue:
        s = queue.popleft()
        d = distance[s]
        assert d is not None
        for p in preds[s]:
            if p in value:
                continue
            if value[s] is Value.LOSS:
                value[p], distance[p] = Value.WIN, d + 1
                queue.append(p)
            else:
                remaining[p] -= 1
                if remaining[p] == 0:
                    # Last WIN successor processed has the largest distance.
                    value[p], distance[p] = Value.LOSS, d + 1
                    queue.append(p)

    for s in states:
        if s not in value:
            value[s], distance[s] = Value.DRAW, None

    sol = Solution(rules=rules, value=value, distance=distance)
    for s in states:
        sol.optimal[s] = _optimal_moves(sol, s)
    return sol


def _optimal_moves(sol: Solution, s: State) -> list[Move]:
    succ = successors(s, sol.rules)
    if not succ:
        return []
    v = sol.value[s]
    if v is Value.DRAW:
        return [m for m, n in succ if sol.value[n] is Value.DRAW]
    d = sol.distance[s]
    assert d is not None
    want = Value.LOSS if v is Value.WIN else Value.WIN
    return [m for m, n in succ if sol.value[n] is want and sol.distance[n] == d - 1]


def summary_line(name: str, sol: Solution) -> str:
    """One-line human summary of a preset's solution."""
    c = sol.counts()
    start = initial_state()
    v = sol.value[start]
    d = sol.distance[start]
    dist = f" in {d} plies" if d is not None else ""
    return (
        f"{name:<11} states={len(sol.value):>4}  WIN={c['WIN']:>4}  LOSS={c['LOSS']:>4}  "
        f"DRAW={c['DRAW']:>4}  start: first player {v.value}{dist}"
    )


def solve_all() -> dict[str, Solution]:
    """Solve every preset."""
    return {name: solve(rules) for name, rules in PRESETS.items()}


def report(solutions: dict[str, Solution]) -> str:
    """Multi-line summary over presets, including the suicide-split comparison."""
    lines = [summary_line(n, s) for n, s in solutions.items()]
    if "standard" in solutions and "no_suicide" in solutions:
        start = initial_state()
        a = solutions["standard"].value[start]
        b = solutions["no_suicide"].value[start]
        if a == b:
            lines.append(
                f"Suicide splits do NOT change the starting value (both {a.value} for the first player)."
            )
        else:
            lines.append(
                f"Suicide splits DO change the starting value: standard={a.value}, "
                f"no_suicide={b.value} (for the first player)."
            )
    return "\n".join(lines)
