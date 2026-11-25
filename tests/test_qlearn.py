"""Q-update math on toy examples, plus table I/O and a small convergence check."""

import pytest

from chopsticks.game import Attack, Split, successors
from chopsticks.qlearn import (
    QTable,
    TrainConfig,
    naive_target,
    negamax_target,
    q_update,
    train_minimax,
    train_naive,
)
from chopsticks.rules import Rules
from chopsticks.solver import Value, solve

STD = Rules()


def test_q_update_arithmetic():
    assert q_update(0.0, 1.0, 0.5) == 0.5
    assert q_update(0.5, 1.0, 0.5) == 0.75
    assert q_update(0.2, -1.0, 0.1) == pytest.approx(0.2 + 0.1 * (-1.2))
    assert q_update(0.3, 0.3, 0.9) == pytest.approx(0.3)


def test_targets():
    # Terminal: the reward alone, regardless of next_best.
    assert naive_target(1.0, 0.7, 1.0, True) == 1.0
    assert negamax_target(-1.0, 0.7, 1.0, True) == -1.0
    # Non-terminal: naive adds gamma * max, negamax negates it.
    assert naive_target(0.0, 0.6, 0.9, False) == pytest.approx(0.54)
    assert negamax_target(0.0, 0.6, 0.9, False) == pytest.approx(-0.54)


def test_negamax_toy_backup():
    """Two-ply toy: opponent's best reply is worth +0.8 to them, so our move is worth -0.8."""
    q = QTable()
    s = ((1, 1), (1, 1))
    a = Attack(0, 0)
    reply_state = ((1, 2), (1, 1))  # what the opponent faces after we tap
    assert dict((m, n) for m, n in successors(s, STD))[a] == reply_state
    for m, _ in successors(reply_state, STD):
        q.set(reply_state, m, -0.5)
    q.set(reply_state, Attack(1, 0), 0.8)
    assert q.best_value(reply_state, STD) == 0.8
    target = negamax_target(0.0, q.best_value(reply_state, STD), 1.0, False)
    q.set(s, a, q_update(q.get(s, a), target, 0.5))
    assert q.get(s, a) == pytest.approx(-0.4)


def test_greedy_ties_and_defaults():
    q = QTable()
    s = ((1, 1), (1, 1))
    assert set(q.greedy_moves(s, STD)) == {Attack(0, 0), Split((0, 2))}
    q.set(s, Split((0, 2)), 0.1)
    assert q.greedy_moves(s, STD) == [Split((0, 2))]


def test_save_load_roundtrip(tmp_path):
    q = QTable()
    q.set(((1, 1), (1, 1)), Attack(0, 0), 0.25)
    path = tmp_path / "q.json"
    q.save(path, {"x": 1})
    assert QTable.load(path).get(((1, 1), (1, 1)), Attack(0, 0)) == 0.25


def test_training_is_reproducible():
    cfg = TrainConfig(episodes=300, seed=7)
    assert train_minimax(STD, cfg).data == train_minimax(STD, cfg).data
    assert train_naive(STD, cfg).data == train_naive(STD, cfg).data


def test_minimax_learns_no_splits_values():
    """On the small, draw-free cutoff + no-splits game, self-play should learn every value's sign."""
    rules = Rules(overflow="cutoff", splits_allowed=False)
    sol = solve(rules)
    q = train_minimax(rules, TrainConfig(episodes=15_000, seed=1))
    played = [s for s in sol.states if sol.optimal[s]]
    agree = sum(
        (q.best_value(s, rules) > 0) == (sol.value[s] is Value.WIN) for s in played
    ) / len(played)
    assert agree > 0.95
