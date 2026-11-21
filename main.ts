// Sokoban solver: push-optimal, room by room.
//
// Idea: a box pushed out of its room through the door can never come back,
// so every box stays in its room and a level's push optimum is the sum of
// its rooms' optima. Each room (3-4 boxes in a small carved area) is solved
// on its own with an A* search over pushes, then the rooms' move strings
// are concatenated with the player's walks between them.
//
// A push state is the boxes' squares plus the player's reachable region
// (named by its lowest square). The search cost is pushes, and the
// heuristic is the cheapest assignment of boxes to goals under
// push-distances (a box moves one square per push, other boxes ignored),
// which never overestimates, so the first solved state found is
// push-optimal. Two prunings keep the search small: a box may only sit on
// a "live" square (one from which a lone box can be pulled back onto a
// goal) and a push that creates a frozen 2x2 block of walls and boxes
// (with a box off its goal) is refused. Everything is counted in search
// nodes, never wall-clock time, so two runs write identical bytes.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Per-room search budget in expanded states (work, not time).
const MAX_EXPAND = 400_000;
const DIRS: [number, number, string, string][] = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

// Squares from which a lone box can be pulled back onto some goal.
// Reverse pull: a box on x can be pulled from x-d when the player can
// stand on x-d and the square x-2d is free; walk the graph backwards.
function liveSquares(level: Level): Uint8Array {
  const { w, wall, goals } = level;
  const live = new Uint8Array(wall.length);
  const queue: number[] = [];
  for (const g of goals) {
    live[g] = 1;
    queue.push(g);
  }
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi]!;
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const y = x - d;
      if (y < 0 || y >= wall.length) continue;
      if (!wall[y] && !wall[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

// Push distance from every square to one goal, other boxes ignored: a box
// moves b -> b+d when b+d and b-d are open floor. BFS over that graph
// from the goal backwards gives a lower bound on pushes to that goal.
function pushDistTo(level: Level, goal: number): Int32Array {
  const { w, wall } = level;
  const dist = new Int32Array(wall.length).fill(-1);
  const queue: number[] = [goal];
  dist[goal] = 0;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi]!;
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const b = x - d;
      if (b < 0 || b >= wall.length || dist[b] >= 0) continue;
      if (wall[b] || wall[x] || wall[b - d]) continue;
      dist[b] = dist[x]! + 1;
      queue.push(b);
    }
  }
  return dist;
}

// Cheapest assignment of boxes to goals under push distances (exact by
// permutation for the small rooms; a box with no route scores Infinity).
function assignCost(boxes: number[], dists: Int32Array[]): number {
  const n = boxes.length;
  if (n === 0) return 0;
  const idx = boxes.map((_, i) => i);
  let best = Infinity;
  const permute = (k: number, acc: number): void => {
    if (acc >= best) return;
    if (k === n) {
      best = acc;
      return;
    }
    for (let i = k; i < n; i++) {
      [idx[k], idx[i]] = [idx[i]!, idx[k]!];
      const d = dists[idx[k]!]![boxes[k]!];
      if (d !== undefined && d >= 0) permute(k + 1, acc + d);
      [idx[k], idx[i]] = [idx[i]!, idx[k]!];
    }
  };
  permute(0, 0);
  return best;
}

// Squares the player reaches from `from` with `occupied` boxes in the
// way; returns the reached set and its lowest square.
function reachable(level: Level, occupied: Uint8Array, from: number): { seen: Uint8Array; low: number } {
  const { w, wall } = level;
  const seen = new Uint8Array(wall.length);
  const stack: number[] = [from];
  seen[from] = 1;
  let low = from;
  while (stack.length > 0) {
    const p = stack.pop()!;
    if (p < low) low = p;
    for (const [dx, dy] of DIRS) {
      const q = p + dy * w + dx;
      if (q < 0 || q >= wall.length) continue;
      if (wall[q] === 0 && occupied[q] === 0 && !seen[q]) {
        seen[q] = 1;
        stack.push(q);
      }
    }
  }
  return { seen, low };
}

// A push that lands a box in a frozen 2x2 block (walls and boxes, with a
// box off its goal) can never be undone.
function frozenBlock(level: Level, occupied: Uint8Array, goal: Uint8Array, t: number): boolean {
  const { w, wall } = level;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (a < 0 || d >= wall.length) continue;
    if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b]) || !(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
    if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b]) || (occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) return true;
  }
  return false;
}

// Shortest player walk from `from` to `to` around walls and boxes.
function walkTo(level: Level, occupied: Uint8Array, from: number, to: number): string {
  if (from === to) return '';
  const { w, wall } = level;
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array<string>(wall.length);
  const queue: number[] = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * w + dx;
      if (q < 0 || q >= wall.length) continue;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: push square out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

interface SearchResult {
  moves: string;
  pushes: number;
}

// A* over pushes for one (sub)level. Returns null when the budget runs
// out or the level has no solution under the prunings.
function solvePush(level: Level, maxExpand: number): SearchResult | null {
  const { w, wall, goals } = level;
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveSquares(level);
  if (level.boxes.some((b) => !live[b])) return null;
  const dists = goals.map((g) => pushDistTo(level, g));
  const solved = (bs: number[]): boolean => bs.every((b) => goal[b]);
  if (solved(level.boxes)) return { moves: '', pushes: 0 };

  const occupied = new Uint8Array(size);
  const boxesOf: number[][] = [level.boxes.slice()];
  const playerOf: number[] = [level.player];
  const parent: number[] = [-1];
  const pushOf: [number, number][] = [[-1, -1]];
  const gOf: number[] = [0];
  const seen = new Map<string, number>();
  const keyOf = (bs: number[], low: number): string => `${low}|${bs.join(',')}`;
  for (const b of level.boxes) occupied[b] = 1;
  const startReach = reachable(level, occupied, level.player);
  for (const b of level.boxes) occupied[b] = 0;
  seen.set(keyOf(level.boxes, startReach.low), 0);

  const heap: [number, number, number][] = [];
  const h0 = assignCost(level.boxes, dists);
  if (!Number.isFinite(h0)) return null;
  let seq = 0;
  const heapPush = (e: [number, number, number]): void => {
    heap.push(e);
    let i = heap.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      const a = heap[i]!;
      const b = heap[p]!;
      if (a[0] > b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] > b[2])))) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): [number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length) {
          const l = heap[c]!;
          const r = heap[c + 1]!;
          if (r[0] < l[0] || (r[0] === l[0] && (r[1] > l[1] || (r[1] === l[1] && r[2] < l[2])))) c++;
        }
        const a = heap[c]!;
        const b = heap[i]!;
        if (a[0] > b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] > b[2])))) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };
  heapPush([h0, 0, seq++]);

  let expanded = 0;
  let found = -1;
  while (heap.length > 0) {
    const [, g, s] = heapPop();
    if (g !== gOf[s]) continue;
    const boxes = boxesOf[s]!;
    if (solved(boxes)) {
      found = s;
      break;
    }
    if (expanded >= maxExpand) return null;
    expanded++;
    for (const b of boxes) occupied[b] = 1;
    const { seen: reach, low } = reachable(level, occupied, playerOf[s]!);
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi]!;
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx;
        const t = b + d;
        if (t < 0 || t >= size) continue;
        if (!reach[b - d] || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        if (frozenBlock(level, occupied, goal, t)) {
          occupied[t] = 0;
          occupied[b] = 1;
          continue;
        }
        const next = boxes.slice();
        next[bi] = t;
        next.sort((x, y) => x - y);
        const nextReach = reachable(level, occupied, b);
        const k = keyOf(next, nextReach.low);
        occupied[t] = 0;
        occupied[b] = 1;
        if (seen.has(k)) continue;
        const h = assignCost(next, dists);
        if (!Number.isFinite(h)) continue;
        const ns = boxesOf.length;
        seen.set(k, ns);
        boxesOf.push(next);
        playerOf.push(b);
        parent.push(s);
        pushOf.push([b, t]);
        gOf.push(g + 1);
        heapPush([g + 1 + h, g + 1, seq++]);
      }
      if (found >= 0) break;
    }
    for (const b of boxes) occupied[b] = 0;
    if (found >= 0) break;
  }
  if (found < 0) return null;

  const path: number[] = [];
  for (let s = found; s > 0; s = parent[s]!) path.push(s);
  path.reverse();
  let moves = '';
  let player = level.player;
  let boxes = level.boxes.slice();
  for (const s of path) {
    const [from, to] = pushOf[s]!;
    const d = to - from;
    const dir = DIRS.find(([dx, dy]) => dy * w + dx === d)!;
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(level, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    moves += dir[3];
    player = from;
    boxes = boxesOf[s]!.slice();
  }
  return { moves, pushes: path.length };
}

// Split a level into rooms: cut the player's row where a box reaches no
// goal (the hallway), and each remaining floor component holding boxes is
// a room. Returns null when the level is no set of rooms.
function roomsOf(level: Level): { boxes: number[]; goals: number[] }[] | null {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(level);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: { boxes: number[]; goals: number[] }[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares: number[] = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i]! + dy * w + dx;
        if (q < 0 || q >= wall.length) continue;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ boxes: [], goals: [] });
  }
  for (const b of boxes) rooms[part[b]!]!.boxes.push(b);
  for (const g of goals) {
    if (part[g]! < 0) return null;
    rooms[part[g]!]!.goals.push(g);
  }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  return p;
}

function solveLevel(level: Level): string {
  const rooms = roomsOf(level);
  if (rooms === null) {
    const whole = solvePush(level, MAX_EXPAND * 2);
    return whole === null ? '' : whole.moves;
  }
  let player = level.player;
  let moves = '';
  for (let ri = 0; ri < rooms.length; ri++) {
    const room = rooms[ri]!;
    const wall = level.wall.slice();
    for (let rj = 0; rj < rooms.length; rj++) {
      if (rj === ri) continue;
      const other = rooms[rj]!;
      const cells = rj < ri ? other.goals : other.boxes;
      for (const c of cells) wall[c] = 1;
    }
    const sub: Level = {
      w: level.w,
      h: level.h,
      wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    const solved = solvePush(sub, MAX_EXPAND);
    if (solved === null) return '';
    moves += solved.moves;
    player = endOf(level.w, player, solved.moves);
  }
  return moves;
}

for (const id of readdirSync(I).sort()) {
  const moves = solveLevel(parseLevel(readFileSync(join(I, id, 'level.txt'), 'utf8')));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
