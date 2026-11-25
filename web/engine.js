// Chopsticks game engine — a line-for-line port of chopsticks/game.py.
//
// A state is [myHands, theirHands] from the point of view of the player to move;
// each hands pair is sorted ascending (canonical). Moves:
//   {type: "attack", from: i, to: j}   (indices into the canonical state)
//   {type: "split", hands: [x, y]}     (x <= y)
// Move keys ("A01", "S13") and state keys ("02-13") match the Python ones.

export const MAX_FINGERS = 4;

export function canonHands(a, b) {
  return a <= b ? [a, b] : [b, a];
}

export function canonicalize([[a, b], [c, d]]) {
  return [canonHands(a, b), canonHands(c, d)];
}

export function initialState() {
  return [[1, 1], [1, 1]];
}

export function stateKey([[a, b], [c, d]]) {
  return `${a}${b}-${c}${d}`;
}

export function parseStateKey(key) {
  const [m, t] = key.split("-");
  return [[+m[0], +m[1]], [+t[0], +t[1]]];
}

export function moveKey(move) {
  return move.type === "attack" ? `A${move.from}${move.to}` : `S${move.hands[0]}${move.hands[1]}`;
}

export function moveFromKey(key) {
  if (key[0] === "A") return { type: "attack", from: +key[1], to: +key[2] };
  if (key[0] === "S") return { type: "split", hands: [+key[1], +key[2]] };
  throw new Error(`bad move key ${key}`);
}

export function addFingers(target, amount, rules) {
  const total = target + amount;
  if (rules.overflow === "cutoff") return total >= 5 ? 0 : total;
  return total % 5;
}

export function isDead([a, b]) {
  return a === 0 && b === 0;
}

export function legalSplits(hands, rules) {
  if (!rules.splits_allowed) return [];
  const [a, b] = canonHands(hands[0], hands[1]);
  if (!rules.allow_revive && a === 0) return [];
  const total = a + b;
  const out = [];
  for (let x = 0; x <= Math.floor(total / 2); x++) {
    const y = total - x;
    if (y > MAX_FINGERS) continue;
    if (x === a && y === b) continue; // same position, including a pure swap
    if (x === 0 && !rules.allow_suicide_split) continue;
    if (rules.even_splits_only && x !== y) continue;
    out.push([x, y]);
  }
  return out;
}

/** Legal moves paired with canonical next states, deduplicated by next state. */
export function successors(state, rules) {
  const [mine, theirs] = state;
  if (isDead(mine) || isDead(theirs)) return [];
  const seen = new Set();
  const out = [];
  for (const i of [0, 1]) {
    if (mine[i] === 0) continue;
    for (const j of [0, 1]) {
      if (theirs[j] === 0) continue;
      const hit = [...theirs];
      hit[j] = addFingers(theirs[j], mine[i], rules);
      const next = [canonHands(hit[0], hit[1]), [...mine]];
      const k = stateKey(next);
      if (!seen.has(k)) {
        seen.add(k);
        out.push([{ type: "attack", from: i, to: j }, next]);
      }
    }
  }
  for (const hands of legalSplits(mine, rules)) {
    const next = [[...theirs], hands];
    const k = stateKey(next);
    if (!seen.has(k)) {
      seen.add(k);
      out.push([{ type: "split", hands }, next]);
    }
  }
  return out;
}

export function legalMoves(state, rules) {
  return successors(state, rules).map(([m]) => m);
}

/** Play a move; returns the next state from the new mover's perspective. Throws if illegal. */
export function applyMove(state, move, rules) {
  const [mine, theirs] = state;
  if (move.type === "attack") {
    if (mine[move.from] === 0 || theirs[move.to] === 0) throw new Error("illegal attack");
    const hit = [...theirs];
    hit[move.to] = addFingers(theirs[move.to], mine[move.from], rules);
    return [canonHands(hit[0], hit[1]), [...mine]];
  }
  if (move.type === "split") {
    const hands = canonHands(move.hands[0], move.hands[1]);
    if (!legalSplits(mine, rules).some(([x, y]) => x === hands[0] && y === hands[1])) {
      throw new Error("illegal split");
    }
    return [[...theirs], hands];
  }
  throw new Error("not a move");
}

export function isTerminal([mine, theirs]) {
  return isDead(mine) || isDead(theirs);
}

/** -1 if the mover has lost, +1 if they have won, null while the game is on. */
export function winner(state, rules) {
  if (isDead(state[0])) return -1;
  if (isDead(state[1])) return 1;
  if (rules && successors(state, rules).length === 0) {
    console.warn("player to move has no legal moves; counting as a loss", stateKey(state));
    return -1;
  }
  return null;
}

/** Every state reachable from the opening, BFS order. */
export function reachableStates(rules) {
  const start = initialState();
  const seen = new Set([stateKey(start)]);
  const order = [start];
  for (let k = 0; k < order.length; k++) {
    for (const [, next] of successors(order[k], rules)) {
      const key = stateKey(next);
      if (!seen.has(key)) {
        seen.add(key);
        order.push(next);
      }
    }
  }
  return order;
}

/** Kind of split: "Revive", "Suicide split" or "Split". */
export function describeSplit(oldHands, newHands) {
  const [a] = canonHands(oldHands[0], oldHands[1]);
  const kind = a === 0 && newHands[0] > 0 ? "Revive" : a > 0 && newHands[0] === 0 ? "Suicide split" : "Split";
  return kind;
}
