"""Split legality under the default rules and every rule variant."""

import itertools

import pytest

from chopsticks.game import Split, apply_move, legal_splits
from chopsticks.rules import Rules

DEFAULT = Rules()

# The reference table from the spec (hands unordered).
SPLIT_TABLE = {
    (0, 1): [],
    (0, 2): [(1, 1)],
    (0, 3): [(1, 2)],
    (0, 4): [(1, 3), (2, 2)],
    (1, 1): [(0, 2)],
    (1, 2): [(0, 3)],
    (1, 3): [(2, 2), (0, 4)],
    (2, 2): [(1, 3), (0, 4)],
    (1, 4): [(2, 3)],
    (2, 3): [(1, 4)],
    (2, 4): [(3, 3)],
    (3, 3): [(2, 4)],
    (3, 4): [],
    (4, 4): [],
}


@pytest.mark.parametrize("hands,expected", sorted(SPLIT_TABLE.items()))
def test_default_split_table(hands, expected):
    assert sorted(legal_splits(hands, DEFAULT)) == sorted(expected)
    # Unordered: the mirrored hand gives the same answer.
    assert sorted(legal_splits(hands[::-1], DEFAULT)) == sorted(expected)


def test_table_covers_every_live_position():
    every = {(a, b) for a in range(5) for b in range(a, 5) if (a, b) != (0, 0)}
    assert set(SPLIT_TABLE) == every


def test_no_suicide():
    rules = Rules(allow_suicide_split=False)
    assert legal_splits((1, 1), rules) == []
    assert legal_splits((1, 2), rules) == []
    assert legal_splits((1, 3), rules) == [(2, 2)]
    assert legal_splits((2, 2), rules) == [(1, 3)]
    # Revives still legal.
    assert legal_splits((0, 3), rules) == [(1, 2)]
    assert legal_splits((0, 2), rules) == [(1, 1)]
    assert sorted(legal_splits((0, 4), rules)) == [(1, 3), (2, 2)]


def test_no_revive():
    rules = Rules(allow_revive=False)
    for hands in [(0, 1), (0, 2), (0, 3), (0, 4)]:
        assert legal_splits(hands, rules) == []
    # Splits between two live hands are unaffected.
    assert legal_splits((1, 4), rules) == [(2, 3)]
    assert legal_splits((1, 1), rules) == [(0, 2)]


def test_even_splits_only():
    rules = Rules(even_splits_only=True)
    assert legal_splits((0, 4), rules) == [(2, 2)]
    assert legal_splits((1, 3), rules) == [(2, 2)]
    assert legal_splits((1, 1), rules) == []  # already even; (0,2) isn't even
    assert legal_splits((1, 4), rules) == []  # odd total
    assert legal_splits((2, 4), rules) == [(3, 3)]


def test_splits_disabled():
    rules = Rules(splits_allowed=False)
    for hands in SPLIT_TABLE:
        assert legal_splits(hands, rules) == []


ALL_VARIANTS = [
    Rules(overflow=o, splits_allowed=s, even_splits_only=e, allow_revive=r, allow_suicide_split=x)
    for o, s, e, r, x in itertools.product(
        ["cutoff", "rollover"], [True, False], [True, False], [True, False], [True, False]
    )
]


@pytest.mark.parametrize("rules", ALL_VARIANTS)
def test_swap_only_split_rejected(rules):
    for a in range(5):
        for b in range(5):
            if a + b == 0:
                continue
            results = legal_splits((a, b), rules)
            assert (min(a, b), max(a, b)) not in results
            with pytest.raises(ValueError):
                apply_move(((a, b), (1, 1)), Split((b, a)), rules)


def test_split_apply_switches_perspective():
    assert apply_move(((0, 3), (1, 4)), Split((1, 2)), DEFAULT) == ((1, 4), (1, 2))
    # Unsorted input to Split is accepted and canonicalised.
    assert apply_move(((0, 3), (1, 4)), Split((2, 1)), DEFAULT) == ((1, 4), (1, 2))
