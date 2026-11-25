// Chopsticks web game: UI on top of engine.js, with the AI driven by the JSON
// exported from Python (solver table + greedy Q-policies).

import {
  addFingers, canonHands, describeSplit, isDead, legalSplits, moveFromKey, moveKey, stateKey,
  successors,
} from "./engine.js";

const AI_DELAY_MS = 650;
const DRAG_THRESHOLD_PX = 6;
const $ = (id) => document.getElementById(id);

// Player-facing names and explanations. Level ids match web/data/index.json.
const LEVELS = [
  { id: "random", name: "Random", desc: "Picks any legal move. Good for learning the rules." },
  { id: "easy", name: "Easy", desc: "A self-play agent stopped early in training. Knows the basics but misses a lot." },
  { id: "naive", name: "Medium", desc: "Trained only against a random opponent. Punishes careless moves, but a steady plan beats it." },
  { id: "minimax", name: "Hard", desc: "Learned the game by playing itself. In testing it matched the perfect player on every move." },
  { id: "perfect", name: "Perfect", desc: "Plays straight from the solved game. It can't lose unless it starts from a lost position." },
];
const PRESET_NAMES = {
  standard: "Standard",
  no_suicide: "No suicide splits",
  cutoff: "Cutoff at 5",
  no_splits: "No splits",
  no_revive: "No revives",
};
// How each preset reads mid-sentence in the summary line under the title.
const SUMMARY_RULES = {
  standard: "standard rules",
  no_suicide: "no suicide splits",
  cutoff: "cutoff at 5",
  no_splits: "no splits",
  no_revive: "no revives",
};

const app = {
  index: null,          // data/index.json
  cache: {},            // preset name -> data/<preset>.json
  data: null,           // current preset's data
  rules: null,
  settings: { level: "minimax", preset: "standard", first: "you" },
  hands: { you: [1, 1], ai: [1, 1] }, // display order (left, right); not sorted
  turn: "you",
  plies: 0,
  over: null,           // null | "you" | "ai" | "draw"
  history: [],          // [{who, text, before: {hands, turn, plies}}]
  selected: null,       // index of your selected hand
  hint: null,           // {move, text, state}
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
const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

function snapshot() {
  return { hands: { you: [...app.hands.you], ai: [...app.hands.ai] }, turn: app.turn, plies: app.plies };
}

/** Bookkeeping shared by every move: log it, advance the clock, check for the end. */
function finishMove(side, text, hitIdx) {
  app.history[app.history.length - 1].text = text;
  app.plies += 1;
  app.selected = null;
  app.hint = null;
  app.splitOpen = false;
  if (isDead(app.hands[other(side)])) app.over = side;
  else if (app.plies >= app.rules.max_moves) app.over = "draw";
  else app.turn = other(side);
  render();
  if (hitIdx !== undefined) flash(other(side), hitIdx);
  if (!app.over && app.turn === "ai") scheduleAi();
}

function attackText(side, fromVal, toVal, after) {
  const result = after === 0 ? "It's dead." : `Now ${after}.`;
  return side === "you"
    ? `You tapped the computer's ${toVal} with your ${fromVal}. ${result}`
    : `Computer tapped your ${toVal} with its ${fromVal}. ${result}`;
}

/** Play a canonical engine move for `side` (used for the computer and for splits). */
function play(side, move) {
  app.history.push({ who: side, before: snapshot(), text: "" });
  const state = canonFor(side);
  const mine = app.hands[side];
  const theirs = app.hands[other(side)];
  if (move.type === "attack") {
    const fromVal = state[0][move.from];
    const toVal = state[1][move.to];
    const toIdx = theirs.indexOf(toVal);
    const after = addFingers(toVal, fromVal, app.rules);
    const next = [...theirs];
    next[toIdx] = after;
    app.hands[other(side)] = next;
    finishMove(side, attackText(side, fromVal, toVal, after), toIdx);
  } else {
    const before = canonHands(...mine);
    app.hands[side] = [...move.hands];
    const who = side === "you" ? "You" : "Computer";
    finishMove(side, `${who}: ${describeSplit(before, move.hands)} ${fmt(before)} → ${fmt(move.hands)}`);
  }
}

/** Your attack from display hand i onto the computer's display hand j. */
function humanAttack(i, j) {
  const mine = app.hands.you;
  const theirs = app.hands.ai;
  if (app.over || app.turn !== "you" || mine[i] === 0 || theirs[j] === 0) return;
  const hit = [...theirs];
  hit[j] = addFingers(theirs[j], mine[i], app.rules);
  app.history.push({ who: "you", before: snapshot(), text: "" });
  app.hands.ai = hit; // keep the exact hands the player chose in the display
  finishMove("you", attackText("you", mine[i], theirs[j], hit[j]), j);
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

const pick = (list) => list[Math.floor(Math.random() * list.length)];

function chooseAiMove() {
  const state = canonFor("ai");
  const key = stateKey(state);
  const legal = successors(state, app.rules).map(([m]) => moveKey(m));
  const level = app.settings.level;
  let keys;
  if (level === "random") keys = legal;
  else if (level === "perfect") keys = app.data.solver[key][2];
  else keys = app.data.policies[level]?.[key];
  if (!keys || keys.length === 0) keys = legal;
  return moveFromKey(pick(keys));
}

function undo() {
  const lastYou = app.history.map((h) => h.who).lastIndexOf("you");
  if (lastYou < 0) return;
  const snap = app.history[lastYou].before;
  app.history = app.history.slice(0, lastYou);
  app.hands = { you: [...snap.hands.you], ai: [...snap.hands.ai] };
  app.turn = snap.turn;
  app.plies = snap.plies;
  Object.assign(app, { over: null, selected: null, hint: null, splitOpen: false });
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
    text = `You can force a win in ${plural(Math.ceil(dist / 2), "move")}.`;
  } else if (value === "LOSS") {
    text = `With perfect play the computer wins. The longest you can hold out is ${plural(dist / 2, "more move")}; this move does that.`;
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

// ---------------------------------------------------------------- drag to attack

/**
 * Pointer-drag one of your hands onto one of the computer's to attack. A press
 * that doesn't move past the threshold falls through to the normal click
 * (select, then click a target), so both styles work.
 */
function startDrag(e, el, i) {
  if (e.button !== 0 || app.over || app.turn !== "you") return;
  const start = { x: e.clientX, y: e.clientY };
  let dragging = false;
  let target = null;

  const targetAt = (x, y) => {
    const hit = document.elementFromPoint(x, y)?.closest("#ai-hands .hand");
    if (!hit) return null;
    const j = Number(hit.dataset.i);
    return app.hands.ai[j] > 0 ? hit : null;
  };

  const onMove = (ev) => {
    const dx = ev.clientX - start.x;
    const dy = ev.clientY - start.y;
    if (!dragging) {
      if (Math.hypot(dx, dy) < DRAG_THRESHOLD_PX) return;
      dragging = true;
      el.classList.remove("selected");
      el.classList.add("dragging");
      document.querySelectorAll("#ai-hands .hand").forEach((h) => {
        if (app.hands.ai[Number(h.dataset.i)] > 0) h.classList.add("target");
      });
      $("status").textContent = "Drop it on one of the computer's hands.";
    }
    el.style.transform = `translate(${dx}px, ${dy}px) scale(1.04)`;
    const t = targetAt(ev.clientX, ev.clientY);
    if (t !== target) {
      target?.classList.remove("drop-hover");
      t?.classList.add("drop-hover");
      target = t;
    }
  };

  const onUp = (ev) => {
    window.removeEventListener("pointermove", onMove);
    window.removeEventListener("pointerup", onUp);
    window.removeEventListener("pointercancel", onUp);
    if (!dragging) return; // a plain click; the click handler takes it from here
    el.dataset.dragged = "1"; // swallow the click that follows this pointerup
    const drop = ev.type === "pointerup" ? targetAt(ev.clientX, ev.clientY) : null;
    if (drop) {
      humanAttack(i, Number(drop.dataset.i));
      return;
    }
    // Missed: glide back, then redraw.
    target?.classList.remove("drop-hover");
    el.classList.remove("dragging");
    el.classList.add("returning");
    el.style.transform = "";
    setTimeout(render, 190);
  };

  window.addEventListener("pointermove", onMove);
  window.addEventListener("pointerup", onUp);
  window.addEventListener("pointercancel", onUp);
}

// ---------------------------------------------------------------- rendering

const FINGER_X = [18, 39, 60, 81];

function handSvg(count, dead, mirror) {
  const fingers = FINGER_X.map((x, k) => {
    const idx = mirror ? 3 - k : k;
    const up = !dead && idx < count;
    return `<rect class="finger${up ? "" : " down"}" x="${x}" y="${up ? 8 : 58}" width="18" height="${up ? 80 : 30}" rx="9"/>`;
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
    b.setAttribute("aria-label", `${owner} ${where} hand: ${dead ? "dead" : plural(count, "finger")}`);
    let label = "";
    if (dead) {
      label = side === "you" && app.rules.allow_revive && app.rules.splits_allowed ? "Dead. Split to revive." : "Dead";
    }
    b.innerHTML = handSvg(count, dead, i === 1) + `<span class="count">${count}</span>` +
      (label ? `<span class="dead-label">${label}</span>` : "");

    if (side === "you") {
      b.disabled = !yourTurn || dead;
      b.setAttribute("aria-pressed", String(app.selected === i));
      if (app.selected === i) b.classList.add("selected");
      if (hintMove?.type === "attack" && hands.indexOf(app.hint.state[0][hintMove.from]) === i) b.classList.add("hinted");
      b.addEventListener("pointerdown", (e) => startDrag(e, b, i));
      b.addEventListener("click", () => {
        if (b.dataset.dragged) { delete b.dataset.dragged; return; }
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
      b.addEventListener("click", () => humanAttack(app.selected ?? liveMine[0], i));
    }
    box.appendChild(b);
  });
}

function statusText() {
  if (app.over === "you") return ["You win!", "win"];
  if (app.over === "ai") return ["The computer wins.", "loss"];
  if (app.over === "draw") return [`Draw: the game reached ${app.rules.max_moves} moves.`, ""];
  if (app.turn === "ai") return ["Computer is thinking…", ""];
  if (app.selected !== null) return ["Now pick one of the computer's hands.", ""];
  return ["Your move. Drag a hand onto theirs, or split.", ""];
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

  $("split-panel").hidden = !(app.splitOpen && !btn.disabled);
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

function summaryText() {
  const { level, preset, first } = app.settings;
  const lvl = LEVELS.find((l) => l.id === level).name;
  const who = first === "you" ? "you move first" : "computer moves first";
  const rules = SUMMARY_RULES[preset] ?? preset;
  return `${lvl} opponent, ${rules}, ${who}`;
}

function render() {
  renderHands("ai");
  renderHands("you");
  const [text, cls] = statusText();
  const status = $("status");
  status.textContent = text;
  status.className = `status ${cls}`;
  $("move-count").textContent = `Move ${app.plies} of ${app.rules.max_moves}`;
  $("summary").textContent = summaryText();

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
    const span = document.createElement("span");
    span.className = `who-${h.who}`;
    span.textContent = who;
    li.append(span, ` ${rest.join(" ")}`);
    list.appendChild(li);
  }
  $("history-count").textContent = app.history.length ? `(${app.history.length})` : "";
  $("history-empty").hidden = app.history.length > 0;
  list.scrollTop = list.scrollHeight;
}

// ---------------------------------------------------------------- settings sheet

function verdictText(start) {
  if (start.value === "DRAW") return "With perfect play: a draw.";
  if (start.value === "LOSS") return "With perfect play: the second player wins.";
  return "With perfect play: the first player wins.";
}

function choice(name, value, title, desc, extra) {
  const label = document.createElement("label");
  label.className = "choice";
  const input = Object.assign(document.createElement("input"), { type: "radio", name, value });
  const t = Object.assign(document.createElement("span"), { className: "choice-name", textContent: title });
  const d = Object.assign(document.createElement("span"), { className: "choice-desc", textContent: desc });
  label.append(input, t, d);
  if (extra) label.append(Object.assign(document.createElement("span"), { className: "choice-verdict", textContent: extra }));
  return label;
}

function buildSettings() {
  const levels = $("level-choices");
  for (const l of LEVELS) levels.append(choice("level", l.id, l.name, l.desc));
  const presets = $("preset-choices");
  for (const [name, p] of Object.entries(app.index.presets)) {
    presets.append(choice("preset", name, PRESET_NAMES[name] ?? name, p.description, verdictText(p.start)));
  }
}

function openSettings() {
  const form = $("settings-form");
  for (const [key, value] of Object.entries(app.settings)) {
    const input = form.querySelector(`input[name="${key}"][value="${value}"]`);
    if (input) input.checked = true;
  }
  const dialog = $("settings");
  dialog.returnValue = ""; // otherwise Esc would reuse the last close's "apply"
  dialog.showModal();
}

function onSettingsClose() {
  const dialog = $("settings");
  if (dialog.returnValue !== "apply") return;
  const form = new FormData($("settings-form"));
  app.settings = {
    level: form.get("level") ?? app.settings.level,
    preset: form.get("preset") ?? app.settings.preset,
    first: form.get("first") ?? app.settings.first,
  };
  newGame();
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
  app.data = await loadPreset(app.settings.preset);
  app.rules = app.data.rules;
  Object.assign(app, {
    hands: { you: [1, 1], ai: [1, 1] },
    turn: app.settings.first,
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
    const box = document.createElement("div");
    box.className = "load-error";
    box.innerHTML = "Couldn't load the game data. Run <code>python -m chopsticks export</code>, then serve the folder with <code>python -m http.server</code> and open <code>http://localhost:8000/web/</code>.";
    $("status").replaceChildren(box);
    console.error(err);
    return;
  }
  app.settings.preset = app.index.default;
  buildSettings();

  $("settings-btn").addEventListener("click", openSettings);
  $("summary").addEventListener("click", openSettings);
  $("settings").addEventListener("close", onSettingsClose);
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
    if (e.key === "Escape" && !$("settings").open && (app.selected !== null || app.splitOpen)) {
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
