/*!
 * rv-sphere.test.js — the geometry harness for rv-sphere.js.
 *
 * The layout is the one thing in that file that can be wrong in a way nobody
 * notices until a visitor sees two clips overlapping on one of the three
 * platforms. The clips are letterbox strips of three different shapes, so the
 * block is solved per platform, and it is solved in PROJECTED space: turning a
 * clip by lon narrows what the camera sees of it by cos(lon) and the cap pushes
 * it back, so a row laid out to touch on the plane opens gaps once it is posed.
 *
 * This checks the row counts, that every row is flush with every other, that
 * neighbours touch to within the gutter after projection, that each clip has a
 * pose of its own with the right sign, and that the timings are ordered.
 *
 *   node docs/assets/rv-sphere.test.js
 *   node --check docs/assets/rv-sphere.js
 *
 * Plain node. No DOM, no framework.
 */
'use strict';

var fs = require('fs'), path = require('path'), vm = require('vm');

var win = {
  matchMedia: function () { return {matches: false}; },
  addEventListener: function () {}, setTimeout: function () {},
  innerWidth: 1600, navigator: {maxTouchPoints: 0}
};
var doc = {
  readyState: 'complete', addEventListener: function () {},
  createElement: function () { return {setAttribute: function () {}, appendChild: function () {}, style: {setProperty: function () {}}}; },
  getElementById: function () { return null; }, querySelector: function () { return null; }
};
win.window = win;
var sandbox = {window: win, document: doc, navigator: win.navigator, console: console};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(__dirname, 'rv-sphere.js'), 'utf8'), sandbox);

var M = sandbox.window.LiEQSphere, C = M.constants;

var pass = 0, fail = 0;
function ok(cond, name) { if (cond) { pass++; return; } fail++; console.log('  FAIL  ' + name); }
function section(s) { console.log('\n' + s); }

/* measured with ffprobe on the shipped clips, 2026-10-23 */
var RATIOS = [
  {key: 'cs55', w: 1100, h: 406}, {key: 'avt', w: 1100, h: 332}, {key: 'courier', w: 1100, h: 340}
];
var FULL = 12;

/* a clip as the camera sees it: position and half-extents after the pose and
   the page's perspective, which is the only space the gaps are real in */
function proj(t, arn) {
  var p = M.pose(t.x, t.y), hw = C.MID_W * t.s / 2, hh = hw / arn;
  return {
    ring: t.ring, x: t.x * p.f, y: t.y * p.f,
    hw: hw * Math.cos(p.lon) * p.f, hh: hh * Math.cos(p.lat) * p.f
  };
}
function rows(slots) {
  var by = {}, i;
  for (i = 0; i < slots.length; i++) (by[slots[i].ring] = by[slots[i].ring] || []).push(slots[i]);
  return by;
}

section('the block is three clips across and four rows deep');
ok(M.ROWS.length === 4, 'four rows');
(function () {
  var total = 0, i;
  for (i = 0; i < M.ROWS.length; i++) {
    ok(M.ROWS[i].n === 3, 'row ' + i + ' holds three clips');
    ok(M.ROWS[i].s === 1, 'row ' + i + ' is the same size as the others');
    total += M.ROWS[i].n;
  }
  ok(total === FULL, 'twelve seats in all');
  ok(M.SLOTS.length === FULL, 'and buildSlots fills them');
}());

section('a platform short of a full set drops seats from the last row');
(function () {
  var s10 = M.buildSlots(M.ARN_REF, 10), r = rows(s10);
  ok(s10.length === 10, 'ten clips give ten slots');
  ok(r[0].length === 3 && r[1].length === 3 && r[2].length === 3 && r[3].length === 1,
     'and the shortfall lands in the last row, not in the middle of the block');
}());

RATIOS.forEach(function (R) {
  var arn = R.w / R.h, slots = M.buildSlots(arn, FULL), by = rows(slots), r, i, row, gaps = [], worst = 0;

  section('neighbours touch at ' + R.key + ' (' + arn.toFixed(2) + ':1)');
  for (r in by) {
    if (!by.hasOwnProperty(r)) continue;
    row = by[r].map(function (t) { return proj(t, arn); }).sort(function (a, b) { return a.x - b.x; });
    for (i = 1; i < row.length; i++) {
      var g = (row[i].x - row[i].hw) - (row[i - 1].x + row[i - 1].hw);
      gaps.push(g);
      ok(g > 0, 'row ' + r + ': clip ' + i + ' clears its neighbour');
      ok(g < C.GUTTER * 3, 'row ' + r + ': clip ' + i + ' sits ' + g.toFixed(1) + 'px from it');
      if (g > worst) worst = g;
    }
  }
  /* the rows themselves */
  /* the four rows straddle the middle, so the pitch is measured between two of
     them rather than from the axis */
  var mid = proj(by[0][0], arn), nxt = proj(by[1][0], arn);
  var vg = Math.abs(nxt.y - mid.y) - (mid.hh + nxt.hh);
  ok(vg > 0 && vg < C.GUTTER * 3, 'the rows clear each other by ' + vg.toFixed(1) + 'px');
  console.log('  widest gap in a row ' + worst.toFixed(1) + 'px, between rows ' + vg.toFixed(1) + 'px');

  /* Flush to within the cap's own foreshortening. A row further from the middle
     sits further back, so the camera sees it very slightly narrower -- about
     4px in 1515 on the tallest platform. That residual IS the curvature; a
     block whose rows matched to the pixel would be a flat poster. What this
     catches is a row off by tens of pixels, which would read as ragged. */
  section('every row is flush with every other at ' + R.key);
  var edges = [];
  for (r in by) {
    if (!by.hasOwnProperty(r)) continue;
    var ps = by[r].map(function (t) { return proj(t, arn); });
    var lo = Math.min.apply(null, ps.map(function (p) { return p.x - p.hw; }));
    var hi = Math.max.apply(null, ps.map(function (p) { return p.x + p.hw; }));
    edges.push([lo, hi]);
  }
  for (i = 1; i < edges.length; i++) {
    ok(Math.abs(edges[i][0] - edges[0][0]) < 6, 'row ' + i + ' starts where row 0 starts, to within the cap');
    ok(Math.abs(edges[i][1] - edges[0][1]) < 6, 'row ' + i + ' ends where row 0 ends, to within the cap');
  }

  section('the block fits the stage at ' + R.key);
  var W = 0, H = 0;
  slots.forEach(function (t) {
    var p = proj(t, arn);
    W = Math.max(W, Math.abs(p.x) + p.hw);
    H = Math.max(H, Math.abs(p.y) + p.hh);
  });
  console.log('  block ' + (2 * W).toFixed(0) + ' x ' + (2 * H).toFixed(0) +
              ' in a ' + C.DESIGN_W + ' x ' + C.DESIGN_H + ' stage');
  ok(2 * W <= C.DESIGN_W, 'it is inside the stage width');
  ok(2 * H <= C.DESIGN_H, 'it is inside the stage height');
  ok(C.DESIGN_W - 2 * W > 200, 'and leaves room either side for the parked platforms');
});

section('the clips sit on a cap, not on a plane');
(function () {
  var S = M.SLOTS, i, tilted = 0;
  for (i = 0; i < S.length; i++) {
    if (S[i].lon !== 0 || S[i].lat !== 0) tilted++;
    /* rotateY maps (0,0,1) to (sin, 0, cos) and rotateX to (0, -sin, cos), so a
       clip right of centre turns +lon and one below centre turns -lat. Get
       either backwards and the cap reads as a saddle. */
    if (S[i].x > 0) ok(S[i].lon > 0, 'slot ' + i + ' right of centre turns outward');
    if (S[i].x < 0) ok(S[i].lon < 0, 'slot ' + i + ' left of centre turns outward');
    if (S[i].y > 0) ok(S[i].lat < 0, 'slot ' + i + ' below centre tips its bottom away');
    if (S[i].y < 0) ok(S[i].lat > 0, 'slot ' + i + ' above centre tips its top away');
    ok(S[i].z <= 0.5, 'slot ' + i + ' never stands in front of the cap');
  }
  ok(tilted === FULL, 'all twelve carry a pose of their own');
  var edge = Math.max.apply(null, S.map(function (t) { return Math.abs(t.lon); }));
  console.log('  strongest turn ' + edge.toFixed(1) + 'deg at TILT_R ' + C.TILT_R);
  ok(edge > 10, 'the outermost clips turn far enough for the curvature to read');
  ok(edge < 28, 'and not so far that a letterbox strip becomes unwatchable');
}());

section('the constants are sane');
ok(C.FACE_STEP * 3 === 360, 'three platforms divide the sphere exactly');
ok(C.RADIUS > C.DESIGN_W / 4, 'the sphere radius keeps a parked face off edge-on');
ok(C.TILT_R > C.DESIGN_W / 4, 'the cap radius makes the tilt a curve, not a fan');
ok(C.START_IN > 0 && C.START_IN < C.SPIN_MS, 'playback starts before the turn lands');
ok(C.DROP_AFT > 0, 'sources are released after the turn, never during it');
ok(C.WORD_LAG > 0 && C.WORD_LAG < 1, 'the caption is rewritten inside the turn');
ok(C.WORD_LAG * C.SPIN_MS > 300, 'late enough in it that the words visibly lag');
ok(C.MIN_WIDTH >= C.DESIGN_W / 2, 'the fallback width is not absurdly low');
ok(C.GUTTER >= 0 && C.GUTTER < 20, 'the gutter is a hairline, not a margin');

section('buildSlots is pure');
(function () {
  var a = M.buildSlots(M.ARN_REF), b = M.buildSlots(M.ARN_REF);
  ok(a !== b, 'it returns a fresh array');
  ok(JSON.stringify(a) === JSON.stringify(b), 'and the same numbers every time');
  ok(JSON.stringify(a) === JSON.stringify(M.SLOTS), 'which are the exported SLOTS');
}());

console.log('\n' + pass + ' passed, ' + fail + ' failed');
process.exit(fail ? 1 : 0);
