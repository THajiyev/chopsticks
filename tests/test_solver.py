"""Solver correctness: hand-checked positions, fixed-point consistency, and an
independent depth-limited search that confirms every distance."""

from functools import lru_cache

import pytest

from chopsticks.game import Attack, Split, initial_state, is_terminal, successors
from chopsticks.rules import PRESETS, Rules
from chopsticks.solver import Value, solve

STD = Rules()  # sum mod 5
CUT = Rules(overflow="cutoff")


@pytest.fixture(scope="module")
def std():
    return solve(STD)


@pytest.fixture(scope="module")
def cut():
    return solve(CUT)


def test_terminal_is_loss(std):
    s = ((0, 0), (0, 4))
    assert std.value[s] is Value.LOSS and std.distance[s] == 0


def test_immediate_kill_is_win_in_one(std):
    # (0,1) vs (0,4): tapping the 4 with the 1 makes 5 -> dead -> opponent out.
    s = ((0, 1), (0, 4))
    assert std.value[s] is Value.WIN and std.distance[s] == 1
    assert std.optimal[s] == [Attack(1, 1)]


def test_forced_loss_in_two(std):
    # (0,1) vs (4,4): the only move kills one 4; the opponent's 4 then kills our 1.
    s = ((0, 1), (4, 4))
    assert std.value[s] is Value.LOSS and std.distance[s] == 2


def test_revive_is_the_only_winning_move_under_cutoff(cut):
    # Cutoff rules, (0,4) vs (1,1). Attacking kills one opponent hand but leaves
    # (0,1) facing our lone 4: their 1 taps it to 5 and we're out. Reviving to
    # (1,3) wins.
    s = ((0, 4), (1, 1))
    after_attack = ((0, 1), (0, 4))
    assert cut.value[after_attack] is Value.WIN and cut.distance[after_attack] == 1
    assert cut.value[s] is Value.WIN
    assert cut.optimal[s] == [Split((1, 3))]


def test_revive_avoids_the_blunder(std):
    # Same position under mod 5: attacking still loses in one, and both revives
    # hold the draw.
    s = ((0, 4), (1, 1))
    assert std.value[((0, 1), (0, 4))] is Value.WIN
    assert std.value[s] is Value.DRAW
    assert set(std.optimal[s]) == {Split((1, 3)), Split((2, 2))}


def test_four_plus_four_wraps_instead_of_killing(std, cut):
    # (0,4) vs (0,4): under cutoff 4+4 kills the last hand (win in 1); under
    # mod 5 it leaves the opponent a 3, so there's no immediate win.
    s = ((0, 4), (0, 4))
    assert cut.value[s] is Value.WIN and cut.distance[s] == 1
    assert std.distance[s] != 1
    assert dict(successors(s, STD))[Attack(1, 1)] == ((0, 3), (0, 4))


def test_win_in_three_by_hand(std):
    # (0,3) vs (1,2): tap their 2 with our 3 (exactly 5 -> dead) -> they have
    # (0,1) and no split, must tap our 3 up to 4, and our 4 then kills their 1.
    s = ((0, 3), (1, 2))
    assert std.value[s] is Value.WIN and std.distance[s] == 3
    assert std.optimal[s] == [Attack(1, 1)]


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_labels_are_a_consistent_fixed_point(name):
    rules = PRESETS[name]
    sol = solve(rules)
    for s, v in sol.value.items():
        succ = successors(s, rules)
        nvals = [(sol.value[n], sol.distance[n]) for _, n in succ]
        d = sol.distance[s]
        if is_terminal(s):
            assert v is Value.LOSS and d == 0
        elif v is Value.WIN:
            loss_d = [nd for nv, nd in nvals if nv is Value.LOSS]
            assert loss_d and d == 1 + min(loss_d)
        elif v is Value.LOSS:
            assert all(nv is Value.WIN for nv, _ in nvals)
            assert d == 1 + max(nd for _, nd in nvals)
        else:
            assert not any(nv is Value.LOSS for nv, _ in nvals)
            assert any(nv is Value.DRAW for nv, _ in nvals)
            assert d is None
        assert sol.optimal[s] or is_terminal(s)


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_distances_match_depth_limited_search(name):
    """Independent check: negamax with a ply budget, memoised on (state, depth)."""
    rules = PRESETS[name]
    sol = solve(rules)

    @lru_cache(maxsize=None)
    def forced(s, depth):
        """+1 if mover can force a win within `depth` plies, -1 if opponent can, else 0."""
        if is_terminal(s):
            return -1
        if depth == 0:
            return 0
        results = [-forced(n, depth - 1) for _, n in successors(s, rules)]
        return max(results)

    horizon = max(d for d in sol.distance.values() if d is not None) + 5
    for s, v in sol.value.items():
        d = sol.distance[s]
        if v is Value.DRAW:
            assert forced(s, horizon) == 0
        else:
            assert forced(s, d) == v.sign
            if d > 0:
                assert forced(s, d - 1) == 0


def test_start_values():
    start = initial_state()
    assert solve(PRESETS["standard"]).value[start] is Value.DRAW
    assert solve(PRESETS["no_suicide"]).value[start] is Value.DRAW
    cut = solve(PRESETS["cutoff"])
    assert cut.value[start] is Value.LOSS and cut.distance[start] == 26
