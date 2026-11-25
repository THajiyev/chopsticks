"""Attacks, overflow rules, dead hands, canonicalisation and reachability."""

import pytest

from chopsticks.game import (
    Attack,
    Split,
    apply_move,
    canonicalize,
    initial_state,
    is_terminal,
    legal_moves,
    reachable_states,
    successors,
    winner,
)
from chopsticks.rules import PRESETS, Rules

CUT = Rules(overflow="cutoff")
ROLL = Rules()


def test_default_overflow_is_sum_mod_5():
    assert Rules().overflow == "rollover"
    # Opponent's 4 taps our 4: we're left with (4+4) mod 5 = 3.
    assert apply_move(((0, 4), (0, 4)), Attack(1, 1), Rules()) == ((0, 3), (0, 4))
    # Exactly 5 still kills.
    assert apply_move(((0, 2), (0, 3)), Attack(1, 1), Rules()) == ((0, 0), (0, 2))


def test_basic_attack_and_perspective_swap():
    # Mover (1,2) hits opponent's 1 with their 2 -> opponent (1,3) now to move.
    assert apply_move(((1, 2), (1, 1)), Attack(1, 0), CUT) == ((1, 3), (1, 2))


@pytest.mark.parametrize(
    "attacker,target,cut,roll",
    [(1, 3, 4, 4), (1, 4, 0, 0), (2, 3, 0, 0), (3, 3, 0, 1), (4, 4, 0, 3), (2, 4, 0, 1)],
)
def test_overflow(attacker, target, cut, roll):
    state = ((0, attacker), (0, target))
    assert apply_move(state, Attack(1, 1), CUT) == ((0, cut), (0, attacker))
    assert apply_move(state, Attack(1, 1), ROLL) == ((0, roll), (0, attacker))


def test_dead_hands_cannot_attack_or_be_attacked():
    state = ((0, 2), (0, 3))
    attacks = [m for m in legal_moves(state, CUT) if isinstance(m, Attack)]
    assert attacks == [Attack(1, 1)]
    with pytest.raises(ValueError):
        apply_move(state, Attack(0, 1), CUT)
    with pytest.raises(ValueError):
        apply_move(state, Attack(1, 0), CUT)


def test_dedup_equal_hands():
    # (1,1) vs (1,1): all four attacks reach the same state.
    moves = legal_moves(((1, 1), (1, 1)), CUT)
    attacks = [m for m in moves if isinstance(m, Attack)]
    assert attacks == [Attack(0, 0)]
    assert Split((0, 2)) in moves
    nexts = [n for _, n in successors(((2, 3), (1, 3)), CUT)]
    assert len(nexts) == len(set(nexts))


def test_canonicalisation():
    assert canonicalize(((3, 1), (4, 0))) == ((1, 3), (0, 4))
    assert canonicalize(((1, 3), (0, 4))) == ((1, 3), (0, 4))
    # Mirror images produce identical successor sets.
    a = apply_move(canonicalize(((2, 1), (1, 1))), Attack(1, 0), CUT)
    assert a == ((1, 3), (1, 2))


def test_terminal_and_winner():
    assert is_terminal(((0, 0), (1, 2)))
    assert winner(((0, 0), (1, 2))) == -1
    assert winner(initial_state()) is None
    assert legal_moves(((0, 0), (1, 2)), CUT) == []
    # Killing the last hand ends the game with the new mover lost.
    nxt = apply_move(((0, 4), (0, 1)), Attack(1, 1), CUT)
    assert nxt == ((0, 0), (0, 4))
    assert winner(nxt) == -1


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_reachable_states_are_canonical_and_closed(name):
    rules = PRESETS[name]
    states = reachable_states(rules)
    assert states[0] == initial_state()
    seen = set(states)
    assert len(seen) == len(states)
    for s in states:
        assert canonicalize(s) == s
        for _, n in successors(s, rules):
            assert n in seen
