// Cross-check the JS engine against fixtures exported from the Python engine.
//
//   node web/engine.test.mjs [path/to/crosscheck.json]
//
// For every preset and every reachable state, the JS engine must produce the
// same terminal flag, the same move keys in the same order, and the same next
// states; applyMove must agree with successors; and the JS reachable-state set
// must equal Python's. Exits non-zero on any mismatch.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import {
  applyMove, isTerminal, moveKey, moveFromKey, parseStateKey, reachableStates, stateKey, successors,
} from "./engine.js";

const here = dirname(fileURLToPath(import.meta.url));
const path = process.argv[2] ?? join(here, "data", "crosscheck.json");
const fixture = JSON.parse(readFileSync(path, "utf8"));

let failures = 0;
let checked = 0;
const fail = (msg) => {
  failures++;
  if (failures <= 20) console.error("MISMATCH", msg);
};

for (const [preset, { rules, states }] of Object.entries(fixture)) {
  for (const [key, expected] of Object.entries(states)) {
    const state = parseStateKey(key);
    if (stateKey(state) !== key) fail(`${preset} ${key}: key round-trip`);
    if (isTerminal(state) !== expected.terminal) fail(`${preset} ${key}: terminal flag`);
    const got = successors(state, rules).map(([m, n]) => [moveKey(m), stateKey(n)]);
    if (JSON.stringify(got) !== JSON.stringify(expected.moves)) {
      fail(`${preset} ${key}: successors\n  js: ${JSON.stringify(got)}\n  py: ${JSON.stringify(expected.moves)}`);
    }
    for (const [mk, nk] of expected.moves) {
      const n = stateKey(applyMove(state, moveFromKey(mk), rules));
      if (n !== nk) fail(`${preset} ${key} ${mk}: applyMove gave ${n}, python ${nk}`);
    }
    checked++;
  }
  const jsReach = new Set(reachableStates(rules).map(stateKey));
  const pyReach = new Set(Object.keys(states));
  if (jsReach.size !== pyReach.size || [...jsReach].some((k) => !pyReach.has(k))) {
    fail(`${preset}: reachable sets differ (js ${jsReach.size}, py ${pyReach.size})`);
  }
}

if (failures) {
  console.error(`FAIL: ${failures} mismatches across ${checked} states`);
  process.exit(1);
}
console.log(`OK: ${checked} states across ${Object.keys(fixture).length} presets match the Python engine`);
