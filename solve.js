// Fast exact drop-in for makefun-diff 1.0.0.
//
// Same edit script as makefun-diff 1.0.0 (Myers greedy search, deletion wins
// ties, no edge-diagonal pruning, snake follows every run of equal lines),
// computed over interned line ids with typed-array V snapshots instead of
// per-step path objects. The patch builder below is the same rule as the
// reference: header line, hunks with 3 lines of context, runs at most 6 kept
// lines apart sharing a hunk, deletions before insertions, "%" after a line
// with no trailing newline.

const FORMAT = 'makefun-diff/1';
const CONTEXT = 3;

const KEEP = 0;
const DELETE = 1;
const INSERT = 2;

/** A text's lines, each with its "\n" (the last one without, when the text does not end in one). */
function tokenize(text) {
  const parts = text.split('\n');
  const last = parts.length - 1;
  const lines = new Array(parts.length);
  let count = 0;
  for (let i = 0; i < last; i++) lines[count++] = parts[i] + '\n';
  if (parts[last] !== '') lines[count++] = parts[last];
  lines.length = count;
  return lines;
}

/** Intern both texts' lines to small int ids (one shared map, so equal lines share an id). */
function internLines(a, b) {
  const map = new Map();
  const ai = new Int32Array(a.length);
  const bi = new Int32Array(b.length);
  let next = 0;
  for (let i = 0; i < a.length; i++) {
    const s = a[i];
    let id = map.get(s);
    if (id === undefined) { id = next++; map.set(s, id); }
    ai[i] = id;
  }
  for (let i = 0; i < b.length; i++) {
    const s = b[i];
    let id = map.get(s);
    if (id === undefined) { id = next++; map.set(s, id); }
    bi[i] = id;
  }
  return [ai, bi];
}

/**
 * The edit-script runs of the middle slices A (length n) into B (length m),
 * as [kind, count] in order (kind 0 keep, 1 delete, 2 insert).
 *
 * Forward pass: d = 0, 1, 2, ...; k = -d..d step 2; the furthest x a path of
 * d edits reaches on k is the further of the deletion candidate (V[d-1][k-1]
 * + 1, when that diagonal exists and stays inside the old text) and the
 * insertion candidate (V[d-1][k+1], when it stays inside the new text), the
 * deletion winning ties; then the snake follows equal lines as far as it
 * goes. V snapshots (one Int32Array row per d, -2 = unreachable) are kept so
 * the exact path can be walked back. The first (d, k) to reach both ends is
 * the script, exactly as the reference's first reaching path.
 */
function myersRuns(A, n, B, m) {
  if (n === 0 && m === 0) return [];
  const rows = [];
  // d = 0: start before both texts' first lines, then snake.
  let row = new Int32Array(1);
  let x = -1;
  let y = -1;
  while (x + 1 < n && y + 1 < m && A[x + 1] === B[y + 1]) { x++; y++; }
  row[0] = x;
  rows.push(row);
  let D = 0;
  let kEnd = 0;
  let jEnd = 0;
  if (x + 1 < n || y + 1 < m) {
    outer:
    for (let d = 1; d <= n + m; d++) {
      const prev = rows[d - 1];
      const cur = new Int32Array(d + 1);
      cur.fill(-2);
      for (let j = 0; j <= d; j++) {
        const k = 2 * j - d;
        let xDel = -2;
        let xIns = -2;
        if (j - 1 >= 0) {
          const pv = prev[j - 1];
          if (pv !== -2 && pv + 1 < n) xDel = pv + 1;
        }
        if (j < d) {
          const pv = prev[j];
          if (pv !== -2 && pv - k < m) xIns = pv;
        }
        if (xDel === -2 && xIns === -2) continue;
        let xx = xDel >= xIns ? xDel : xIns;
        let yy = xx - k;
        while (xx + 1 < n && yy + 1 < m && A[xx + 1] === B[yy + 1]) { xx++; yy++; }
        cur[j] = xx;
        if (xx + 1 >= n && yy + 1 >= m) { D = d; kEnd = k; jEnd = j; rows.push(cur); break outer; }
      }
      rows.push(cur);
    }
  }
  // Backtrack from (D, kEnd): at each d the move into (d, k), then its snake.
  const rev = [];
  let d = D;
  let k = kEnd;
  let j = jEnd;
  let cx = rows[D][j];
  while (d > 0) {
    const prev = rows[d - 1];
    let xDel = -2;
    let xIns = -2;
    if (j - 1 >= 0) {
      const pv = prev[j - 1];
      if (pv !== -2 && pv + 1 < n) xDel = pv + 1;
    }
    if (j < d) {
      const pv = prev[j];
      if (pv !== -2 && pv - k < m) xIns = pv;
    }
    const isDel = xDel >= xIns;
    const xMove = isDel ? xDel : xIns;
    const keep = cx - xMove;
    if (keep > 0) rev.push([KEEP, keep]);
    rev.push([isDel ? DELETE : INSERT, 1]);
    if (isDel) { j = j - 1; k = k - 1; }
    else { k = k + 1; }
    cx = prev[j];
    d--;
  }
  const keep0 = rows[0][0] + 1;
  if (keep0 > 0) rev.push([KEEP, keep0]);
  rev.reverse();
  // Merge consecutive runs of one kind (the reference merges them as it builds).
  const runs = [];
  for (let i = 0; i < rev.length; i++) {
    const r = rev[i];
    const last = runs[runs.length - 1];
    if (last !== undefined && last[0] === r[0]) last[1] += r[1];
    else runs.push([r[0], r[1]]);
  }
  return runs;
}

/** A line as the patch writes it: its op, the line, a newline, and "%" on a line of its own when it had none. */
function written(op, line) {
  return line.endsWith('\n') ? op + line : op + line + '\n%\n';
}

/** make.fun's own patch of input.old into input.new, as one string. */
export function run(input) {
  if (input === null || typeof input !== 'object' || typeof input.old !== 'string' || typeof input.new !== 'string') {
    throw new TypeError('makefun-diff: the input is { old, new }, two strings');
  }
  const a = tokenize(input.old);
  const b = tokenize(input.new);
  const n = a.length;
  const m = b.length;
  const [ai, bi] = internLines(a, b);
  // The common prefix, then the common suffix of what is left.
  let prefix = 0;
  while (prefix < n && prefix < m && ai[prefix] === bi[prefix]) prefix++;
  let suffix = 0;
  while (suffix < n - prefix && suffix < m - prefix && ai[n - 1 - suffix] === bi[m - 1 - suffix]) suffix++;
  const runs = [];
  if (prefix > 0) runs.push([KEEP, prefix]);
  const mid = myersRuns(
    ai.subarray(prefix, n - suffix), n - prefix - suffix,
    bi.subarray(prefix, m - suffix), m - prefix - suffix,
  );
  for (let i = 0; i < mid.length; i++) runs.push(mid[i]);
  if (suffix > 0) runs.push([KEEP, suffix]);

  // The change runs: each a stretch of deletions and insertions between two kept lines, at old line o and new line p.
  const changes = [];
  let o = 0;
  let p = 0;
  let deleted = 0;
  let inserted = 0;
  for (let i = 0; i < runs.length; i++) {
    const kind = runs[i][0];
    const count = runs[i][1];
    if (kind === KEEP) {
      o += count;
      p += count;
      continue;
    }
    let c = changes[changes.length - 1];
    if (c === undefined || c.o + c.dels !== o || c.p + c.ins !== p) {
      c = { o, p, dels: 0, ins: 0 };
      changes.push(c);
    }
    if (kind === DELETE) {
      c.dels += count;
      o += count;
      deleted += count;
    } else {
      c.ins += count;
      p += count;
      inserted += count;
    }
  }

  // The hunks: change runs at most 2 * CONTEXT kept lines apart share one.
  const hunks = [];
  for (let i = 0; i < changes.length; i++) {
    const c = changes[i];
    const h = hunks[hunks.length - 1];
    const prev = h !== undefined ? h[h.length - 1] : undefined;
    if (prev !== undefined && c.o - (prev.o + prev.dels) <= 2 * CONTEXT) h.push(c);
    else hunks.push([c]);
  }
  const out = [`${FORMAT} old=${n} new=${m} hunks=${hunks.length} deleted=${deleted} inserted=${inserted}\n`];
  for (let hi = 0; hi < hunks.length; hi++) {
    const h = hunks[hi];
    const first = h[0];
    const last = h[h.length - 1];
    const oldStart = Math.max(0, first.o - CONTEXT);
    const newStart = first.p - (first.o - oldStart);
    const after = Math.min(CONTEXT, n - (last.o + last.dels));
    const oldEnd = last.o + last.dels + after;
    const newEnd = last.p + last.ins + after;
    out.push(`@ ${oldStart + 1} ${oldEnd - oldStart} ${newStart + 1} ${newEnd - newStart}\n`);
    let at = oldStart;
    for (let ci = 0; ci < h.length; ci++) {
      const c = h[ci];
      for (; at < c.o; at++) out.push(written('=', a[at]));
      for (let i = 0; i < c.dels; i++) out.push(written('-', a[c.o + i]));
      for (let i = 0; i < c.ins; i++) out.push(written('+', b[c.p + i]));
      at = c.o + c.dels;
    }
    for (; at < oldEnd; at++) out.push(written('=', a[at]));
  }
  return out.join('');
}
