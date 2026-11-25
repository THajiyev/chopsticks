"""Players (Random, greedy Q-agent, Perfect) and a game runner."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional, Protocol

from .game import Move, State, apply_move, initial_state, successors, winner
from .qlearn import QTable
from .rules import Rules
from .solver import Solution


class Player(Protocol):
    """Anything that picks a move for the player to move."""

    name: str

    def choose(self, state: State, rng: random.Random) -> Move: ...


@dataclass
class RandomPlayer:
    """Uniformly random legal move."""

    rules: Rules
    name: str = "Random"

    def choose(self, state: State, rng: random.Random) -> Move:
        return rng.choice([m for m, _ in successors(state, self.rules)])


@dataclass
class QPlayer:
    """Greedy Q-agent (no exploration); ties broken at random."""

    table: QTable
    rules: Rules
    name: str = "Q"

    def choose(self, state: State, rng: random.Random) -> Move:
        return rng.choice(self.table.greedy_moves(state, self.rules))


@dataclass
class PerfectPlayer:
    """Plays a solver-optimal move (fastest win / slowest loss), ties at random."""

    solution: Solution
    name: str = "Perfect"

    def choose(self, state: State, rng: random.Random) -> Move:
        return rng.choice(self.solution.optimal[state])


@dataclass
class GameRecord:
    """Outcome of one game: +1 first player won, -1 second won, 0 draw."""

    result: int
    plies: int
    # (state, seat) for every decision; seat 0 = first player, 1 = second.
    visits: list[tuple[State, int]] = field(default_factory=list)


def play_game(
    first: Player, second: Player, rules: Rules, rng: random.Random, start: Optional[State] = None
) -> GameRecord:
    """Play one game to completion or the ``max_moves`` cap."""
    s = start or initial_state()
    players = (first, second)
    rec = GameRecord(result=0, plies=0)
    seat = 0
    while True:
        w = winner(s, rules)
        if w is not None:
            # w is from the perspective of the seat to move.
            rec.result = w if seat == 0 else -w
            return rec
        if rec.plies >= rules.max_moves:
            rec.result = 0
            return rec
        rec.visits.append((s, seat))
        s = apply_move(s, players[seat].choose(s, rng), rules)
        rec.plies += 1
        seat = 1 - seat
