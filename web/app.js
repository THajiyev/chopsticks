// Chopsticks web game: UI on top of engine.js, with the AI driven by the JSON
// exported from Python (solver table + greedy Q-policies).

import {
  addFingers, canonHands, describeSplit, isDead, legalSplits, moveFromKey, moveKey, stateKey,
  successors,
} from "./engine.js";

const AI_DELAY_MS = 650;
const $ = (id) => document.getElementById(id);

const app = {
  index: null,          // data/index.json
  cache: {},            // preset name -> data/<preset>.json
  data: null,           // current preset's data
  rules: null,
  level: "minimax",
  first: "you",
  hands: { you: [1, 1], ai: [1, 1] }, // display order (left, right); not sorted
  turn: "you",
  plies: 0,
  over: null,           // null | "you" | "ai" | "draw"
  history: [],          // [{who, text, before: {hands, turn, plies}}]
  selected: null,       // index of your selected hand
  hint: null,           // {move, text}
  splitOpen: false,
  gen: 0,               // bumps on new game / undo to cancel pending AI moves
};

// ---------------------------------------------------------------- state helpers

const other = (side) => (side === "you" ? "ai" : "you");

/** Canonical engine state from `side`'s point of view. */
function canonFor(side) {
  const h = app.hands;
  return [canonHands(...h[side]), canonHands(...h[other(side)])];
}

const fmt = ([a, b]) => `(${a},${b})`;

/** Apply a canonical move for `side` to the display hands; returns details for the log. */
function applyToDisplay(side, move) {
  const state = canonFor(side);
  const mine = app.hands[side];
  const theirs = app.hands[other(side)];
  if (move.type === "attack") {
    const fromVal = state[0][move.from];
    const toVal = state[1][move.to];
    const fromIdx = mine.indexOf(fromVal);
    const toIdx = theirs.indexOf(toVal);
    const after = addFingers(toVal, fromVal, app.rules);
    const next = [...theirs];
    next[toIdx] = after;
    app.hands[other(side)] = next;
    return { kind: "attack", fromIdx, toIdx, fromVal, toVal, after };
  }
  const before = canonHands(...mine);
  app.hands[side] = [...move.hands];
  return { kind: "split", before, after: move.hands };
}

function describe(side, d) {
  const who = side === "you" ? "You" : "Computer";
  if (d.kind === "attack") {
    const target = side === "you" ? "the computer's" : "your";
    const own = side === "you" ? "your" : "its";
    const result = d.after === 0 ? "It's dead." : `Now ${d.after}.`;
    return `${who} tapped ${target} ${d.toVal} with ${own} ${d.fromVal}. ${result}`;
  }
  return `${who}: ${describeSplit(d.before, d.after)} ${fmt(d.before)} → ${fmt(d.after)}`;
}

// ---------------------------------------------------------------- playing moves

function play(side, move) {
  app.history.push({
    who: side,
    before: { hands: { you: [...app.hands.you], ai: [...app.hands.ai] }, turn: app.turn, plies: app.plies },
    text: "",
  });
  const d = applyToDisplay(side, move);
  app.history[app.history.length - 1].text = describe(side, d);
  app.plies += 1;
  app.selected = null;
  app.hint = null;
  app.splitOpen = false;

  if (isDead(app.hands[other(side)])) app.over = side;
  else if (app.plies >= app.rules.max_moves) app.over = "draw";
  else app.turn = other(side);

  render();
  if (d.kind === "attack") flash(other(side), d.toIdx);
  if (!app.over && app.turn === "ai") scheduleAi();
}

function flash(side, idx) {
  const el = document.querySelector(`#${side}-hands .hand[data-i="${idx}"]`);
  if (!el) return;
  el.classList.remove("hit");
  void el.offsetWidth; // restart the animation
  el.classList.add("hit");
}

function scheduleAi() {
  const gen = app.gen;
  setTimeout(() => {
    if (gen !== app.gen || app.over || app.turn !== "ai") return;
    play("ai", chooseAiMove());
  }, AI_DELAY_MS);
}

function pick(list) {
  return list[Math.floor(Math.random() * list.length)];
}

function chooseAiMove() {
  const state = canonFor("ai");
  const key = stateKey(state);
  const legal = successors(state, app.rules).map(([m]) => m);
  let keys;
  if (app.level === "random") keys = legal.map(moveKey);
  else if (app.level === "perfect") keys = app.data.solver[key][2];
  else keys = app.data.policies[app.level]?.[key];
  if (!keys || keys.length === 0) keys = legal.map(moveKey);
  return moveFromKey(pick(keys));
}

/** Your attack from display hand i onto the computer's display hand j. */
function humanAttack(i, j) {
  const mine = app.hands.you;
  const theirs = app.hands.ai;
  if (mine[i] === 0 || theirs[j] === 0) return;
  const hit = [...theirs];
  hit[j] = addFingers(theirs[j], mine[i], app.rules);
  const wantKey = stateKey([canonHands(...hit), canonHands(...mine)]);
  // Find the engine move with the same result (the engine merges equivalent attacks).
  const match = successors(canonFor("you"), app.rules).find(([m, n]) => m.type === "attack" && stateKey(n) === wantKey);
  if (!match) return;
  // Keep the exact hands the player clicked in the display.
  app.history.push({
    who: "you",
    before: { hands: { you: [...mine], ai: [...theirs] }, turn: app.turn, plies: app.plies },
    text: `You tapped the computer's ${theirs[j]} with your ${mine[i]}. ${hit[j] === 0 ? "It's dead." : `Now ${hit[j]}.`}`,
  });
  app.hands.ai = hit;
  app.plies += 1;
  app.selected = null;
  app.hint = null;
  app.splitOpen = false;
  if (isDead(hit)) app.over = "you";
  else if (app.plies >= app.rules.max_moves) app.over = "draw";
  else app.turn = "ai";
  render();
  flash("ai", j);
  if (!app.over) scheduleAi();
}

function undo() {
  const lastYou = app.history.map((h) => h.who).lastIndexOf("you");
  if (lastYou < 0) return;
  const snap = app.history[lastYou].before;
  app.history = app.history.slice(0, lastYou);
  app.hands = { you: [...snap.hands.you], ai: [...snap.hands.ai] };
  app.turn = snap.turn;
  app.plies = snap.plies;
  app.over = null;
  app.selected = null;
  app.hint = null;
  app.splitOpen = false;
  app.gen += 1;
  render();
}

// ---------------------------------------------------------------- hint

function computeHint() {
  const state = canonFor("you");
  const [value, dist, opt] = app.data.solver[stateKey(state)];
  const move = moveFromKey(opt[0]);
  const left = app.rules.max_moves - app.plies;
  let text;
  if (value === "WIN") {
    const n = Math.ceil(dist / 2);
    text = `You can force a win in ${n} ${n === 1 ? "move" : "moves"}.`;
  } else if (value === "LOSS") {
    const n = dist / 2;
    text = `With perfect play the computer wins. The longest you can hold out is ${n} more ${n === 1 ? "move" : "moves"}; this move does that.`;
  } else {
    text = "Neither side can force a win from here. This move keeps it level.";
  }
  if (dist !== null && dist > left) text += ` The ${app.rules.max_moves}-move limit will end the game as a draw first.`;
  if (move.type === "split") {
    const before = canonHands(...app.hands.you);
    text += ` Suggested: ${describeSplit(before, move.hands)} ${fmt(before)} → ${fmt(move.hands)}.`;
  } else {
    text += ` Suggested: tap their ${state[1][move.to]} with your ${state[0][move.from]}.`;
  }
  return { move, text, state };
}

// ---------------------------------------------------------------- rendering

const FINGER_X = [18, 39, 60, 81];

function handSvg(count, dead, mirror) {
  const fingers = FINGER_X.map((x, k) => {
    const idx = mirror ? 3 - k : k;
    const up = !dead && idx < count;
    const y = up ? 8 : 58;
    const h = up ? 80 : 30;
    return `<rect class="finger${up ? "" : " down"}" x="${x}" y="${y}" width="18" height="${h}" rx="9"/>`;
  }).join("");
  return `<svg viewBox="0 0 120 150" aria-hidden="true">${fingers}<rect class="palm" x="12" y="66" width="96" height="76" rx="24"/></svg>`;
}

function renderHands(side) {
  const box = $(`${side}-hands`);
  const hands = app.hands[side];
  const yourTurn = !app.over && app.turn === "you";
  const hintMove = app.hint?.move;
  box.innerHTML = "";
  hands.forEach((count, i) => {
    const dead = count === 0;
    const b = document.createElement("button");
    b.type = "button";
    b.className = `hand ${side}${dead ? " dead" : ""}`;
    b.dataset.i = String(i);
    const where = i === 0 ? "left" : "right";
    const owner = side === "you" ? "Your" : "Computer's";
    b.setAttribute("aria-label", `${owner} ${where} hand: ${dead ? "dead" : `${count} ${count === 1 ? "finger" : "fingers"}`}`);
    let label = "";
    if (dead) {
      label = side === "you" && app.rules.allow_revive && app.rules.splits_allowed
        ? "Dead. Split to revive."
        : "Dead";
    }
    b.innerHTML = handSvg(count, dead, i === 1) + `<span class="count">${count}</span>` +
      (label ? `<span class="dead-label">${label}</span>` : "");

    if (side === "you") {
      b.disabled = !yourTurn || dead;
      b.setAttribute("aria-pressed", String(app.selected === i));
      if (app.selected === i) b.classList.add("selected");
      if (hintMove?.type === "attack" && hands.indexOf(app.hint.state[0][hintMove.from]) === i) b.classList.add("hinted");
      b.addEventListener("click", () => {
        app.selected = app.selected === i ? null : i;
        app.splitOpen = false;
        render();
      });
    } else {
      const liveMine = app.hands.you.map((v, k) => (v > 0 ? k : -1)).filter((k) => k >= 0);
      const canTarget = yourTurn && !dead && (app.selected !== null || liveMine.length === 1);
      b.disabled = !canTarget;
      if (canTarget) b.classList.add("target");
      if (hintMove?.type === "attack" && hands.indexOf(app.hint.state[1][hintMove.to]) === i) b.classList.add("hinted");
      b.addEventListener("click", () => {
        const from = app.selected ?? liveMine[0];
        humanAttack(from, i);
      });
    }
    box.appendChild(b);
  });
}

function statusText() {
  if (app.over === "you") return ["You win!", "win"];
  if (app.over === "ai") return ["The computer wins.", "loss"];
  if (app.over === "draw") return [`Draw: the game reached ${app.rules.max_moves} moves.`, ""];
  if (app.turn === "ai") return ["Computer is thinking…", ""];
  if (app.selected !== null) return ["Now pick one of the computer's hands to tap.", ""];
  return ["Your move. Pick one of your hands, or split.", ""];
}

function renderSplit() {
  const btn = $("split-btn");
  const yourTurn = !app.over && app.turn === "you";
  const mine = canonHands(...app.hands.you);
  const options = legalSplits(mine, app.rules);
  btn.disabled = !yourTurn || options.length === 0;
  btn.title = !app.rules.splits_allowed
    ? "Splits are off in these rules"
    : options.length === 0 ? `No legal split from ${fmt(mine)}` : "";
  btn.setAttribute("aria-expanded", String(app.splitOpen && !btn.disabled));
  btn.classList.toggle("hinted", app.hint?.move.type === "split");

  const panel = $("split-panel");
  panel.hidden = !(app.splitOpen && !btn.disabled);
  const list = $("split-options");
  list.innerHTML = "";
  for (const hands of options) {
    const o = document.createElement("button");
    o.type = "button";
    o.className = "split-option";
    o.innerHTML = `<strong>${describeSplit(mine, hands)}:</strong> ${fmt(mine)} → ${fmt(hands)}`;
    const h = app.hint?.move;
    if (h?.type === "split" && h.hands[0] === hands[0] && h.hands[1] === hands[1]) o.classList.add("hinted");
    o.addEventListener("click", () => play("you", { type: "split", hands }));
    list.appendChild(o);
  }
}

function render() {
  renderHands("ai");
  renderHands("you");
  const [text, cls] = statusText();
  const status = $("status");
  status.textContent = text;
  status.className = `status ${cls}`;
  $("move-count").textContent = `Move ${app.plies} of ${app.rules.max_moves}`;

  const hintEl = $("hint-text");
  hintEl.hidden = !app.hint;
  hintEl.textContent = app.hint?.text ?? "";
  $("hint-btn").disabled = !!app.over || app.turn !== "you";
  $("undo-btn").disabled = !app.history.some((h) => h.who === "you");

  renderSplit();

  const list = $("history-list");
  list.innerHTML = "";
  for (const h of app.history) {
    const li = document.createElement("li");
    const [who, ...rest] = h.text.split(" ");
    li.innerHTML = `<span class="who-${h.who}">${who}</span> ${rest.join(" ")}`;
    list.appendChild(li);
  }
  $("history-empty").hidden = app.history.length > 0;
  list.scrollTop = list.scrollHeight;
}

// ---------------------------------------------------------------- setup

async function loadPreset(name) {
  if (!app.cache[name]) {
    const res = await fetch(`data/${name}.json`);
    if (!res.ok) throw new Error(`data/${name}.json: ${res.status}`);
    app.cache[name] = await res.json();
  }
  return app.cache[name];
}

async function newGame() {
  app.gen += 1;
  const preset = $("preset").value;
  app.level = $("level").value;
  app.first = $("first").value;
  app.data = await loadPreset(preset);
  app.rules = app.data.rules;
  $("rules-desc").textContent = app.index.presets[preset].description;
  Object.assign(app, {
    hands: { you: [1, 1], ai: [1, 1] },
    turn: app.first,
    plies: 0,
    over: null,
    history: [],
    selected: null,
    hint: null,
    splitOpen: false,
  });
  render();
  if (app.turn === "ai") scheduleAi();
}

async function init() {
  try {
    const res = await fetch("data/index.json");
    if (!res.ok) throw new Error(`data/index.json: ${res.status}`);
    app.index = await res.json();
  } catch (err) {
    $("status").innerHTML = "";
    const box = document.createElement("div");
    box.className = "load-error";
    box.innerHTML = "Couldn't load the game data. Run <code>python -m chopsticks export</code>, then serve the folder with <code>python -m http.server</code> and open <code>http://localhost:8000/web/</code>.";
    $("status").appendChild(box);
    console.error(err);
    return;
  }
  const level = $("level");
  for (const l of app.index.levels) level.add(new Option(l.label, l.id));
  level.value = "minimax";
  const preset = $("preset");
  for (const name of Object.keys(app.index.presets)) preset.add(new Option(name.replace("_", " "), name));
  preset.value = app.index.default;

  for (const id of ["level", "preset", "first"]) $(id).addEventListener("change", newGame);
  $("new-game").addEventListener("click", newGame);
  $("undo-btn").addEventListener("click", undo);
  $("hint-btn").addEventListener("click", () => {
    app.hint = app.hint ? null : computeHint();
    render();
  });
  $("split-btn").addEventListener("click", () => {
    app.splitOpen = !app.splitOpen;
    app.selected = null;
    render();
    if (app.splitOpen) $("split-options").querySelector("button")?.focus();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && (app.selected !== null || app.splitOpen)) {
      app.selected = null;
      app.splitOpen = false;
      render();
    }
  });
  await newGame();
}

// Debug/test handle (used by the browser smoke test; harmless in normal play).
window.chopsticks = { app, render, play, newGame };

init();
