"""Export solver results, trained policies and engine fixtures to JSON for the web game.

Files written under ``web/data/``:

* ``index.json`` — presets (rules + description) and the difficulty levels.
* ``<preset>.json`` — solver table and greedy policies for that preset:
  ``solver[stateKey] = [value, distance|null, [optimal move keys]]`` and
  ``policies[level][stateKey] = [move keys tied for best Q]``.
* ``crosscheck.json`` — for every preset and reachable state, the legal move
  keys and resulting next-state keys, used by ``web/engine.test.mjs``.
"""

from __future__ import annotations

import json
from pathlib import Path

from .game import initial_state, is_terminal, state_key, successors
from .qlearn import QTable
from .rules import PRESET_DESCRIPTIONS, PRESETS, Rules
from .solver import Solution

LEVELS = [
    {"id": "random", "label": "Random"},
    {"id": "easy", "label": "Easy (half-trained B)"},
    {"id": "naive", "label": "Medium (Naive A)"},
    {"id": "minimax", "label": "Hard (Minimax B)"},
    {"id": "perfect", "label": "Perfect (solver)"},
]


def policy_table(q: QTable, sol: Solution) -> dict[str, list[str]]:
    """Greedy move set (all moves tied for best Q) for every decision state."""
    return {
        state_key(s): [m.key for m in q.greedy_moves(s, sol.rules)]
        for s in sol.states
        if sol.optimal[s]
    }


def solver_table(sol: Solution) -> dict[str, list]:
    """Compact solver output keyed by state."""
    return {
        state_key(s): [sol.value[s].value, sol.distance[s], [m.key for m in sol.optimal[s]]]
        for s in sol.states
    }


def crosscheck_fixture(rules: Rules, sol: Solution) -> dict:
    """Legal moves and next states for every reachable state."""
    return {
        "rules": rules.to_dict(),
        "states": {
            state_key(s): {
                "terminal": is_terminal(s),
                "moves": [[m.key, state_key(n)] for m, n in successors(s, rules)],
            }
            for s in sol.states
        },
    }


def export_all(
    solutions: dict[str, Solution],
    policies: dict[str, dict[str, QTable]],
    out_dir: Path,
    easy_episodes: dict[str, int],
) -> list[Path]:
    """Write every JSON file the web game needs. ``policies[preset][level]``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    index = {
        "default": "standard",
        "levels": LEVELS,
        "presets": {
            name: {
                "rules": PRESETS[name].to_dict(),
                "description": PRESET_DESCRIPTIONS[name],
                # Value of the opening for the first player, e.g. "DRAW" or "LOSS".
                "start": {
                    "value": sol.value[initial_state()].value,
                    "distance": sol.distance[initial_state()],
                },
            }
            for name, sol in solutions.items()
        },
    }
    p = out_dir / "index.json"
    p.write_text(json.dumps(index, indent=1))
    written.append(p)

    for name, sol in solutions.items():
        data = {
            "preset": name,
            "rules": sol.rules.to_dict(),
            "easy_episodes": easy_episodes.get(name),
            "solver": solver_table(sol),
            "policies": {lvl: policy_table(q, sol) for lvl, q in policies[name].items()},
        }
        p = out_dir / f"{name}.json"
        p.write_text(json.dumps(data, separators=(",", ":")))
        written.append(p)

    p = out_dir / "crosscheck.json"
    p.write_text(
        json.dumps(
            {name: crosscheck_fixture(sol.rules, sol) for name, sol in solutions.items()},
            separators=(",", ":"),
        )
    )
    written.append(p)
    return written
