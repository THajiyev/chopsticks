"""Chopsticks game engine.

A state is ``(my_hands, their_hands)`` from the point of view of the player to
move. Each pair of hands is a sorted tuple, so mirror-image positions collapse
to one canonical state. ``apply_move`` returns the next state from the *new*
mover's point of view (i.e. the two halves swap).
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Union

from .rules import Rules

logger = logging.getLogger(__name__)

Hands = tuple[int, int]
State = tuple[Hands, Hands]

MAX_FINGERS = 4


@dataclass(frozen=True)
class Attack:
    """Tap an opponent's hand with one of yours.

    Indices refer to positions in the *canonical* (sorted) state.
    """

    from_hand: int
    to_hand: int

    @property
    def key(self) -> str:
        """Compact identifier, shared with the JS engine (e.g. ``"A01"``)."""
        return f"A{self.from_hand}{self.to_hand}"

    def __str__(self) -> str:
        return f"Attack(from_hand={self.from_hand}, to_hand={self.to_hand})"


@dataclass(frozen=True)
class Split:
    """Redistribute your own fingers into ``new`` (a sorted pair)."""

    new: Hands

    @property
    def key(self) -> str:
        """Compact identifier, shared with the JS engine (e.g. ``"S13"``)."""
        return f"S{self.new[0]}{self.new[1]}"

    def __str__(self) -> str:
        return f"Split(new={self.new})"


Move = Union[Attack, Split]


def canon_hands(a: int, b: int) -> Hands:
    """Sort a pair of hands."""
    return (a, b) if a <= b else (b, a)


def canonicalize(state: tuple[tuple[int, int], tuple[int, int]]) -> State:
    """Return the canonical form of a (possibly unsorted) state."""
    (a, b), (c, d) = state
    return (canon_hands(a, b), canon_hands(c, d))


def initial_state() -> State:
    """Both players start with one finger on each hand."""
    return ((1, 1), (1, 1))


def state_key(state: State) -> str:
    """String key for JSON, e.g. ``((0, 2), (1, 3))`` -> ``"02-13"``."""
    (a, b), (c, d) = state
    return f"{a}{b}-{c}{d}"


def parse_state_key(key: str) -> State:
    """Inverse of :func:`state_key`."""
    mine, theirs = key.split("-")
    return ((int(mine[0]), int(mine[1])), (int(theirs[0]), int(theirs[1])))


def move_from_key(key: str) -> Move:
    """Inverse of ``Move.key``."""
    if key[0] == "A":
        return Attack(int(key[1]), int(key[2]))
    if key[0] == "S":
        return Split((int(key[1]), int(key[2])))
    raise ValueError(f"bad move key {key!r}")


def add_fingers(target: int, amount: int, rules: Rules) -> int:
    """Value of a hand with ``target`` fingers after receiving ``amount``."""
    total = target + amount
    if rules.overflow == "cutoff":
        return 0 if total >= 5 else total
    return total % 5


def is_dead(hands: Hands) -> bool:
    """True when both hands have 0 fingers."""
    return hands[0] == 0 and hands[1] == 0


def legal_splits(hands: Hands, rules: Rules) -> list[Hands]:
    """All sorted hand pairs reachable from ``hands`` by one legal split."""
    if not rules.splits_allowed:
        return []
    a, b = canon_hands(*hands)
    if not rules.allow_revive and a == 0:
        return []
    total = a + b
    out: list[Hands] = []
    for x in range(0, total // 2 + 1):
        y = total - x
        if y > MAX_FINGERS:
            continue
        new = (x, y)
        if new == (a, b):  # same position, including a pure swap
            continue
        if x == 0 and not rules.allow_suicide_split:
            continue
        if rules.even_splits_only and x != y:
            continue
        out.append(new)
    return out


@lru_cache(maxsize=None)
def successors(state: State, rules: Rules) -> tuple[tuple[Move, State], ...]:
    """Legal moves paired with their canonical next states, deduplicated.

    Moves that lead to the same canonical next state are merged; the first one
    in generation order (lowest hand indices) is kept.
    """
    mine, theirs = state
    if is_dead(mine) or is_dead(theirs):
        return ()
    seen: set[State] = set()
    out: list[tuple[Move, State]] = []
    for i in (0, 1):
        if mine[i] == 0:
            continue
        for j in (0, 1):
            if theirs[j] == 0:
                continue
            hit = list(theirs)
            hit[j] = add_fingers(theirs[j], mine[i], rules)
            nxt = (canon_hands(hit[0], hit[1]), mine)
            if nxt not in seen:
                seen.add(nxt)
                out.append((Attack(i, j), nxt))
    for new in legal_splits(mine, rules):
        nxt = (theirs, new)
        if nxt not in seen:
            seen.add(nxt)
            out.append((Split(new), nxt))
    return tuple(out)


def legal_moves(state: State, rules: Rules) -> list[Move]:
    """Deduplicated legal moves for the player to move."""
    return [m for m, _ in successors(state, rules)]


def apply_move(state: State, move: Move, rules: Rules) -> State:
    """Play ``move`` and return the next state from the new mover's perspective.

    Raises ``ValueError`` if the move is illegal.
    """
    mine, theirs = state
    if isinstance(move, Attack):
        if mine[move.from_hand] == 0 or theirs[move.to_hand] == 0:
            raise ValueError(f"illegal attack {move} in {state}")
        hit = list(theirs)
        hit[move.to_hand] = add_fingers(theirs[move.to_hand], mine[move.from_hand], rules)
        return (canon_hands(hit[0], hit[1]), mine)
    if isinstance(move, Split):
        new = canon_hands(*move.new)
        if new not in legal_splits(mine, rules):
            raise ValueError(f"illegal split {move} in {state}")
        return (theirs, new)
    raise TypeError(f"not a move: {move!r}")


def is_stuck(state: State, rules: Rules) -> bool:
    """Non-terminal state with no legal move (treated as a loss for the mover)."""
    return not is_terminal(state) and not successors(state, rules)


def is_terminal(state: State) -> bool:
    """True if either player has both hands dead."""
    return is_dead(state[0]) or is_dead(state[1])


def winner(state: State, rules: Optional[Rules] = None) -> Optional[int]:
    """Result from the mover's perspective: ``-1`` lost, ``+1`` won, ``None`` ongoing.

    If ``rules`` is given, a non-terminal state with no legal moves counts as a
    loss for the mover (and a warning is logged).
    """
    if is_dead(state[0]):
        return -1
    if is_dead(state[1]):
        return 1
    if rules is not None and not successors(state, rules):
        logger.warning("player to move has no legal moves in %s; counting as a loss", state)
        return -1
    return None


def reachable_states(rules: Rules, start: Optional[State] = None) -> list[State]:
    """Every state reachable from ``start`` (default: the opening), BFS order."""
    start = start or initial_state()
    seen = {start}
    order = [start]
    queue = deque([start])
    while queue:
        s = queue.popleft()
        for _, nxt in successors(s, rules):
            if nxt not in seen:
                seen.add(nxt)
                order.append(nxt)
                queue.append(nxt)
    return order


def describe_split(old: Hands, new: Hands) -> str:
    """Label a split as a revive, suicide split, or plain split."""
    old = canon_hands(*old)
    if old[0] == 0 and new[0] > 0:
        kind = "Revive"
    elif old[0] > 0 and new[0] == 0:
        kind = "Suicide split"
    else:
        kind = "Split"
    return f"{kind}: {old} → {new}"
