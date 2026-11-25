"""Python <-> JS engine cross-check.

Exports a fresh fixture from the *current* Python engine (every reachable state
of every preset plus a few extra rule variants, with legal moves and next
states) and has ``web/engine.test.mjs`` confirm the JS engine matches exactly.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from chopsticks.export import crosscheck_fixture
from chopsticks.rules import PRESETS, Rules
from chopsticks.solver import solve

ROOT = Path(__file__).resolve().parent.parent
NODE = shutil.which("node")

EXTRA = {
    "even_only": Rules(even_splits_only=True),
    "rollover_no_suicide": Rules(overflow="rollover", allow_suicide_split=False),
    "no_revive_no_suicide": Rules(allow_revive=False, allow_suicide_split=False),
}


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_js_engine_matches_python(tmp_path):
    rule_sets = {**PRESETS, **EXTRA}
    fixture = {name: crosscheck_fixture(r, solve(r)) for name, r in rule_sets.items()}
    path = tmp_path / "crosscheck.json"
    path.write_text(json.dumps(fixture))
    proc = subprocess.run(
        [NODE, str(ROOT / "web" / "engine.test.mjs"), str(path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.startswith("OK")
