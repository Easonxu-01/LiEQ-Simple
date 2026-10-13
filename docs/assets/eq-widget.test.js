/*
 * eq-widget.test.js — assertions for assets/eq-widget.js.
 *
 * Runs in plain node with no DOM, no jsdom, no test framework:
 *     node docs/assets/eq-widget.test.js
 *
 * It covers the three things that are easy to get wrong and impossible to see
 * in a screenshot: the morph algebra, the markup's shape, and the geometry
 * arithmetic (fader travel, value <-> pixel round trip, label collisions).
 */
'use strict';

var path = require('path');
var EQ = require(path.join(__dirname, 'eq-widget.js'));

var pass = 0, fail = 0, notes = [];

function ok(name, cond, detail) {
  if (cond) { pass++; console.log('  ok   ' + name + (detail ? '   (' + detail + ')' : '')); }
  else { fail++; console.log('  FAIL ' + name + (detail ? '   (' + detail + ')' : '')); }
}
function note(s) { notes.push(s); console.log('       - ' + s); }
function head(s) { console.log('\n' + s); }

function maxAbsDiff(a, b) {
  var m = 0, i, d;
  for (i = 0; i < a.length; i++) { d = Math.abs(a[i] - b[i]); if (d > m) { m = d; } }
  return m;
}
function maxAbs(a) {
  var m = 0, i;
  for (i = 0; i < a.length; i++) { if (Math.abs(a[i]) > m) { m = Math.abs(a[i]); } }
  return m;
}
function sub(a, b) {
  var o = [], i;
  for (i = 0; i < a.length; i++) { o.push(a[i] - b[i]); }
  return o;
}
function add(a, b) {
  var o = [], i;
  for (i = 0; i < a.length; i++) { o.push(a[i] + b[i]); }
  return o;
}
function sci(x) { return x === 0 ? '0' : x.toExponential(2); }

/* ------------------------------------------------------------------ */
/* Set-up: the same decomposition the widget builds at init.           */
/* ------------------------------------------------------------------ */

var R = EQ.defaults.smoothRadius, P = EQ.defaults.smoothPasses;
var teacher = EQ.curves.teacher.slice(0);
var student = EQ.curves.student.slice(0);
var N = teacher.length;
var T = EQ.math.split(teacher, R, P);
var S = EQ.math.split(student, R, P);

function morph(lf, hf) { return EQ.math.morph(S.lo, S.hi, T.lo, T.hi, lf, hf); }

head('CURVES');
ok('both curves are sampled on the same grid', teacher.length === student.length, N + ' samples');
ok('all samples finite and in [0, 1.05]', (function () {
  var i;
  for (i = 0; i < N; i++) {
    if (!isFinite(teacher[i]) || !isFinite(student[i])) { return false; }
    if (teacher[i] < -1e-9 || teacher[i] > 1.05) { return false; }
    if (student[i] < -1e-9 || student[i] > 1.05) { return false; }
  }
  return true;
})());
ok('split is exact: lo + hi === source',
   maxAbsDiff(add(T.lo, T.hi), teacher) < 1e-12 && maxAbsDiff(add(S.lo, S.hi), student) < 1e-12,
   'max ' + sci(Math.max(maxAbsDiff(add(T.lo, T.hi), teacher), maxAbsDiff(add(S.lo, S.hi), student))));

/* The figure's two claims, as numbers. */
(function () {
  var i, lowSum = 0, lowN = 0, hiT = 0, hiS = 0, hiN = 0;
  for (i = 0; i < N; i++) {
    if (i / (N - 1) < 0.35) { lowSum += S.lo[i] - T.lo[i]; lowN++; }
    if (i / (N - 1) >= 0.5) { hiT += Math.abs(T.hi[i]); hiS += Math.abs(S.hi[i]); hiN++; }
  }
  ok('student envelope exceeds teacher at low frequency', lowSum / lowN > 0.05,
     'mean excess ' + (lowSum / lowN).toFixed(3));
  ok('student carries more high-frequency energy on the right half', hiS / hiT > 1.2,
     'residual ratio ' + (hiS / hiT).toFixed(2));
})();

/* ------------------------------------------------------------------ */
head('MORPH — the four corners');

var c00 = morph(0, 0), c11 = morph(1, 1), c10 = morph(1, 0), c01 = morph(0, 1);

ok('lf=0, hf=0 reproduces the student exactly', maxAbsDiff(c00, student) < 1e-12,
   'max ' + sci(maxAbsDiff(c00, student)));
ok('lf=1, hf=1 reproduces the teacher to 1e-9', maxAbsDiff(c11, teacher) < 1e-9,
   'max ' + sci(maxAbsDiff(c11, teacher)));
ok('lf=1, hf=0 is exactly teacherLo + studentHi', maxAbsDiff(c10, add(T.lo, S.hi)) < 1e-12,
   'max ' + sci(maxAbsDiff(c10, add(T.lo, S.hi))));
ok('lf=0, hf=1 is exactly studentLo + teacherHi', maxAbsDiff(c01, add(S.lo, T.hi)) < 1e-12,
   'max ' + sci(maxAbsDiff(c01, add(S.lo, T.hi))));

/* ------------------------------------------------------------------ */
head('MORPH — at lf=1, hf=0 the envelope is the teacher\'s, the detail is still the student\'s');

(function () {
  /* The exact statement is the corner identity above: out === teacherLo + studentHi.
   * This block is an INDEPENDENT cross-check with two instruments the widget does
   * not itself use, because re-running the widget's own box blur over its own
   * output measures the blur as much as the result (a box blur is not a
   * projection: S(S(x)) != S(x)).
   *
   *   envelope  -> a WIDER moving average than the widget's own (r=7, p=4)
   *   detail    -> a discrete second difference, i.e. a plain ripple detector
   */
  function env(a) { return EQ.math.smooth(a, 7, 4); }
  function ripple(a) {
    var o = [], i, l, r;
    for (i = 0; i < a.length; i++) {
      l = a[i > 0 ? i - 1 : 0];
      r = a[i < a.length - 1 ? i + 1 : a.length - 1];
      o.push(a[i] - (l + r) / 2);
    }
    return o;
  }
  function argmax(a) { var k = 0, i; for (i = 1; i < a.length; i++) { if (a[i] > a[k]) { k = i; } } return k; }

  var eOut = env(c10), eT = env(teacher), eS = env(student);
  var envToT = maxAbsDiff(eOut, eT), envToS = maxAbsDiff(eOut, eS), envGap0 = maxAbsDiff(eS, eT);
  ok('LF alone puts the envelope on the teacher\'s',
     envToT < 0.2 * envGap0 && envToT * 4 < envToS,
     envToT.toFixed(4) + ' from blue vs ' + envToS.toFixed(4) + ' from orange; ' +
     (100 * (1 - envToT / envGap0)).toFixed(1) + '% of a ' + envGap0.toFixed(4) + ' gap closed');

  var rOut = ripple(c10), rT = ripple(teacher), rS = ripple(student);
  var ripToS = maxAbsDiff(rOut, rS), ripToT = maxAbsDiff(rOut, rT);
  ok('LF alone leaves the wiggle the student\'s own',
     ripToS * 2.2 < ripToT,
     ripToS.toFixed(4) + ' from orange vs ' + ripToT.toFixed(4) + ' from blue (' +
     (ripToT / ripToS).toFixed(1) + 'x closer to orange)');

  ok('LF alone moves the peak onto the teacher\'s',
     Math.abs(argmax(c10) - argmax(teacher)) <= 2,
     'peak x ' + (argmax(student) / (N - 1)).toFixed(2) + ' -> ' +
     (argmax(c10) / (N - 1)).toFixed(2) + ', teacher at ' + (argmax(teacher) / (N - 1)).toFixed(2));

  /* and the mirror image: HF alone swaps only the detail */
  var eOut2 = env(c01), rOut2 = ripple(c01);
  ok('HF alone leaves the envelope the student\'s',
     maxAbsDiff(eOut2, eS) * 4 < maxAbsDiff(eOut2, eT),
     maxAbsDiff(eOut2, eS).toFixed(4) + ' from orange vs ' + maxAbsDiff(eOut2, eT).toFixed(4) + ' from blue');
  ok('HF alone puts the wiggle on the teacher\'s',
     maxAbsDiff(rOut2, rT) * 2.2 < maxAbsDiff(rOut2, rS),
     maxAbsDiff(rOut2, rT).toFixed(4) + ' from blue vs ' + maxAbsDiff(rOut2, rS).toFixed(4) + ' from orange');
  ok('HF alone leaves the peak where the student put it',
     Math.abs(argmax(c01) - argmax(student)) <= 3,
     'peak x ' + (argmax(c01) / (N - 1)).toFixed(2) + ', student at ' +
     (argmax(student) / (N - 1)).toFixed(2));

  note('out - studentHi === teacherLo to ' + sci(maxAbsDiff(sub(c10, S.hi), T.lo)) +
       ' — that identity, not these ratios, is the guarantee.');
})();

/* ------------------------------------------------------------------ */
head('MORPH — continuity and monotonicity');

(function () {
  var hfFixed = [0, 0.25, 0.5, 0.75, 1], lfFixed = [0, 0.25, 0.5, 0.75, 1];
  var steps = 50, k, j, i, a, b, slope, bad = 0, worstLip = 0, expect, got;

  /* monotone in lf, for every sample, at several hf */
  for (k = 0; k < hfFixed.length; k++) {
    for (i = 0; i < N; i++) {
      slope = 0;
      for (j = 0; j < steps; j++) {
        a = morph(j / steps, hfFixed[k])[i];
        b = morph((j + 1) / steps, hfFixed[k])[i];
        if (slope === 0) { slope = Math.sign(b - a); }
        else if (Math.sign(b - a) !== 0 && Math.sign(b - a) !== slope) { bad++; }
      }
    }
  }
  ok('monotone in LF at every sample, for hf in {0,.25,.5,.75,1}', bad === 0, bad + ' sign flips');

  bad = 0;
  for (k = 0; k < lfFixed.length; k++) {
    for (i = 0; i < N; i++) {
      slope = 0;
      for (j = 0; j < steps; j++) {
        a = morph(lfFixed[k], j / steps)[i];
        b = morph(lfFixed[k], (j + 1) / steps)[i];
        if (slope === 0) { slope = Math.sign(b - a); }
        else if (Math.sign(b - a) !== 0 && Math.sign(b - a) !== slope) { bad++; }
      }
    }
  }
  ok('monotone in HF at every sample, for lf in {0,.25,.5,.75,1}', bad === 0, bad + ' sign flips');

  /* exactly linear: a step of d in lf must move sample i by d*(tLo-sLo) */
  var d = 1e-3, worst = 0;
  for (j = 0; j < 20; j++) {
    a = morph(j / 20, 0.37);
    b = morph(j / 20 + d, 0.37);
    for (i = 0; i < N; i++) {
      expect = d * (T.lo[i] - S.lo[i]);
      got = b[i] - a[i];
      if (Math.abs(got - expect) > worst) { worst = Math.abs(got - expect); }
      if (Math.abs(got) > worstLip) { worstLip = Math.abs(got); }
    }
  }
  ok('response is exactly linear in LF (no cross term, no easing)', worst < 1e-12,
     'max deviation ' + sci(worst));
  note('Lipschitz: a 0.001 LF step moves no sample by more than ' + worstLip.toFixed(6) +
       ' (bound ' + (d * maxAbs(sub(T.lo, S.lo))).toFixed(6) + ').');

  /* intermediate poses really are intermediate */
  var midOK = true;
  var half = morph(0.5, 0.5);
  for (i = 0; i < N; i++) {
    var lo = Math.min(c00[i], c11[i]) - 0.2, hi = Math.max(c00[i], c11[i]) + 0.2;
    if (half[i] < lo || half[i] > hi) { midOK = false; }
    if (Math.abs(half[i] - 0.5 * (c00[i] + c11[i])) > 1e-12) { midOK = false; }
  }
  ok('lf=hf=0.5 is the exact midpoint of the rest and locked poses', midOK);
})();

/* ------------------------------------------------------------------ */
head('MORPH — the drawn range is bounded by the four corners');

(function () {
  var corners = [c00, c10, c01, c11], hi = -Infinity, lo = Infinity, i, k, v, bad = 0;
  for (k = 0; k < 4; k++) {
    for (i = 0; i < N; i++) {
      if (corners[k][i] > hi) { hi = corners[k][i]; }
      if (corners[k][i] < lo) { lo = corners[k][i]; }
    }
  }
  for (k = 0; k <= 20; k++) {
    for (var j = 0; j <= 20; j++) {
      v = morph(k / 20, j / 20);
      for (i = 0; i < N; i++) { if (v[i] > hi + 1e-12 || v[i] < lo - 1e-12) { bad++; } }
    }
  }
  ok('no pose on a 21x21 grid leaves the corner envelope', bad === 0,
     'range [' + lo.toFixed(4) + ', ' + hi.toFixed(4) + ']');
  note('the vertical scale is derived from exactly this: span = ' +
       (EQ.geometry.PLOT.b - EQ.geometry.PLOT.t - 6) + ' / vmax.');
  note('lowest value any pose reaches is ' + lo.toFixed(4) +
       ' — below the axis, so render() clamps to 0 before drawing.');
})();

/* ------------------------------------------------------------------ */
head('PATH — Catmull-Rom to cubic Bezier');

(function () {
  var xs = [], ys = [], i, d, segs;
  for (i = 0; i < N; i++) { xs.push(48 + 296 * i / (N - 1)); ys.push(214 - 162 * student[i]); }
  d = EQ.math.catmullPath(xs, ys);
  segs = d.split('C').length - 1;
  ok('one cubic segment per interval', segs === N - 1, segs + ' segments for ' + N + ' points');
  ok('starts at the first sample', d.indexOf('M48 214') === 0, d.slice(0, 18) + '...');
  ok('ends at the last sample', /C[^C]*\s344 214$/.test(d), '...' + d.slice(-22));
  ok('only M and C commands, all coordinates finite', /^M[-\d. ]+(C[-\d. ]+)+$/.test(d));
  ok('coordinates are quantised to 0.1', !/\.\d\d/.test(d));
  ok('degenerate input does not throw',
     EQ.math.catmullPath([], []) === '' && EQ.math.catmullPath([1], [1]) === '');
  note('path length at rest: ' + d.length + ' chars (rebuilt once per animation frame while dragging).');
})();

/* ------------------------------------------------------------------ */
head('GEOMETRY — fader travel and value round trip');

(function () {
  var TR = EQ.geometry.TRACK, PL = EQ.geometry.PLOT, i, v, y, back, worst = 0, RK = EQ.geometry.RACK;
  var knobAt = function (val) { return TR.mid + (TR.half - val * TR.half * 2); };
  var valueAt = function (py) { return (TR.bottom - py) / (TR.bottom - TR.top); };

  ok('v=0 puts the cap at the bottom of the track', Math.abs(knobAt(0) - TR.bottom) < 1e-9,
     'y=' + knobAt(0));
  ok('v=1 puts the cap at the top of the track', Math.abs(knobAt(1) - TR.top) < 1e-9,
     'y=' + knobAt(1));
  ok('track length is 134 user units', TR.bottom - TR.top === 134);

  for (i = 0; i <= 100; i++) {
    v = i / 100; y = knobAt(v); back = valueAt(y);
    if (Math.abs(back - v) > worst) { worst = Math.abs(back - v); }
  }
  ok('pixel <-> value round trip is exact', worst < 1e-12, 'max ' + sci(worst));

  var IN = EQ.geometry.INSET;
  ok('plot frame sits inside the viewBox inset',
     PL.l >= IN && PL.r <= EQ.geometry.VB_W - IN && PL.t >= IN && PL.b <= EQ.geometry.VB_H - IN,
     IN + ' units');
  ok('rack sits inside the viewBox inset',
     RK.x + RK.w <= EQ.geometry.VB_W - IN && RK.y >= IN && RK.y + RK.h <= EQ.geometry.VB_H - IN);
  ok('plot and rack do not overlap', PL.r < RK.x, (RK.x - PL.r) + ' units of gutter');
  ok('the whole fader travel sits inside the rack',
     TR.top > RK.y && TR.bottom < RK.y + RK.h);
})();

/* ------------------------------------------------------------------ */
/* Pull font-size and letter-spacing straight out of the exported stylesheet so
 * these numbers cannot drift away from what the browser will actually paint.
 * JetBrains Mono advance = 0.6em. The narrow-screen block is parsed separately
 * and layered over the base, exactly as the cascade would. */
var NARROW_MARK = '@media (max-width:760px){';
function sheet(narrow) {
  var i = EQ.css.indexOf(NARROW_MARK);
  return narrow ? (i < 0 ? '' : EQ.css.slice(i + NARROW_MARK.length)) : EQ.css.slice(0, i < 0 ? EQ.css.length : i);
}
function styleOf(cls, narrow) {
  function read(src) {
    var m = new RegExp('\\.' + cls + '\\{([^}]*)\\}').exec(src), body = m ? m[1] : '';
    var fs = /font-size:([\d.]+)px/.exec(body), ls = /letter-spacing:([\d.]*)em/.exec(body);
    return { px: fs ? parseFloat(fs[1]) : 0, ls: ls ? parseFloat('0' + ls[1]) : null };
  }
  var base = read(sheet(false));
  if (!narrow) { return { px: base.px, ls: base.ls || 0 }; }
  var over = read(sheet(true));
  return { px: over.px || base.px, ls: over.ls === null ? (base.ls || 0) : over.ls };
}

function labelGeometry(narrow) {
  var tag = narrow ? ' [narrow]' : ' [base]';
  function w(text, cls) {
    var s = styleOf(cls, narrow);
    if (!s.px) { throw new Error('no font-size for .' + cls); }
    return text.length * s.px * 0.6 + (text.length - 1) * s.ls * s.px;
  }
  var D = EQ.defaults, G = EQ.geometry, PL = G.PLOT, RK = G.RACK, AN = G.ANN, LG = G.LEGEND;
  var barX = PL.l + (PL.r - PL.l) * AN.lfFrac;

  ok('every measured class has a size' + tag, (function () {
    var want = ['lieq-leg', 'lieq-ann-t', 'lieq-cap-sub', 'lieq-axis-t', 'lieq-ro-k',
                'lieq-tick-t', 'lieq-rack-eyebrow', 'lieq-val', 'lieq-cap-label'], i;
    for (i = 0; i < want.length; i++) { if (!styleOf(want[i], narrow).px) { return false; } }
    return true;
  })());

  var legendEnd = LG.t2 + w(D.studentLabel, 'lieq-leg');
  ok('legend fits inside the plot' + tag, legendEnd < PL.r,
     'ends at ' + legendEnd.toFixed(1) + ' of ' + PL.r);
  ok('legend entries do not collide' + tag, LG.t1 + w(D.teacherLabel, 'lieq-leg') < LG.sw2,
     'first ends at ' + (LG.t1 + w(D.teacherLabel, 'lieq-leg')).toFixed(1) +
     ', second swatch at ' + LG.sw2);

  var a1 = w(D.lfAnnotation[0], 'lieq-ann-t');
  ok('LF annotation fits between the plot edge and its dimension bar' + tag,
     AN.lfTextX - a1 > PL.l && AN.lfTextX < barX - AN.gapCap,
     'spans ' + (AN.lfTextX - a1).toFixed(1) + '..' + AN.lfTextX + ', plot starts ' + PL.l +
     ', bar cap at ' + (barX - AN.gapCap).toFixed(1));
  var a2 = w(D.hfAnnotation[0], 'lieq-ann-t');
  ok('HF annotation fits inside the plot' + tag, AN.hfTextX - a2 > PL.l,
     'starts at ' + (AN.hfTextX - a2).toFixed(1));

  var capw = w('EQUALIZER', 'lieq-cap-sub');
  ok('fader captions do not collide' + tag,
     G.FADER_X.lf + capw / 2 + 5 < G.FADER_X.hf - capw / 2,
     'gap ' + ((G.FADER_X.hf - capw / 2) - (G.FADER_X.lf + capw / 2)).toFixed(1) + ' units');
  ok('fader captions stay inside the rack' + tag,
     G.FADER_X.lf - capw / 2 > RK.x && G.FADER_X.hf + capw / 2 < RK.x + RK.w,
     (G.FADER_X.lf - capw / 2).toFixed(1) + '..' + (G.FADER_X.hf + capw / 2).toFixed(1) +
     ' inside ' + RK.x + '..' + (RK.x + RK.w));

  var mid = (PL.l + PL.r) / 2;
  var xt = w(D.xLabel, 'lieq-axis-t'), ro = w(D.readoutLabel + ' 0.00', 'lieq-ro-k');
  ok('x-axis title and the readout do not collide' + tag, mid + xt / 2 < PL.r - ro,
     'title ends ' + (mid + xt / 2).toFixed(1) + ', readout starts ' + (PL.r - ro).toFixed(1));
  var yt = w(D.yLabel, 'lieq-axis-t');
  ok('rotated y-axis title fits the plot height' + tag, yt < PL.b - PL.t,
     yt.toFixed(1) + ' of ' + (PL.b - PL.t));
  ok('LOW / HIGH ticks do not collide' + tag,
     PL.l + w(D.lowTick, 'lieq-tick-t') < PL.r - w(D.highTick, 'lieq-tick-t'));
  ok('rack eyebrow fits the rack' + tag, w(D.rackLabel, 'lieq-rack-eyebrow') < RK.w - 8,
     w(D.rackLabel, 'lieq-rack-eyebrow').toFixed(1) + ' of ' + (RK.w - 8));
  ok('value readout fits between the faders' + tag,
     w('0.00', 'lieq-val') < G.FADER_X.hf - G.FADER_X.lf);
}

head('GEOMETRY — label boxes at desktop size');
labelGeometry(false);
head('GEOMETRY — label boxes at phone size (the @media block)');
labelGeometry(true);

head('GEOMETRY — controls');
(function () {
  var G = EQ.geometry, RK = G.RACK;
  ok('focus rings stay inside the rack and clear of each other',
     G.FADER_X.lf - 21 > RK.x && G.FADER_X.hf + 21 < RK.x + RK.w &&
     G.FADER_X.lf + 21 < G.FADER_X.hf - 21);
  /* 50 user units of hit width -> 50 * (320/470) = 34 CSS px on a 390px phone */
  var hit = 50, phoneCss = hit * (320 / G.VB_W);
  ok('fader hit areas are wide enough to touch and do not overlap',
     phoneCss > 30 && G.FADER_X.lf + hit / 2 < G.FADER_X.hf - hit / 2,
     phoneCss.toFixed(1) + ' CSS px wide on a 390px phone');
  ok('hit areas stay inside the rack',
     G.FADER_X.lf - hit / 2 >= RK.x && G.FADER_X.hf + hit / 2 <= RK.x + RK.w);
})();

/* ------------------------------------------------------------------ */
head('MARKUP — shape, hooks and accessibility');

(function () {
  var src = EQ.renderMarkup(), stack = [], depth = [], i, m, tag, self, hidden = 0,
      faderHidden = 0, re = /<(\/?)([A-Za-z][\w:-]*)((?:[^>"']|"[^"]*"|'[^']*')*?)(\/?)>/g,
      balanced = true, maxDepth = 0, attrs;

  while ((m = re.exec(src)) !== null) {
    tag = m[2]; attrs = m[3] || ''; self = m[4] === '/';
    if (m[1] === '/') {
      if (stack.pop() !== tag) { balanced = false; }
      if (depth.pop()) { hidden--; }
    } else if (!self) {
      stack.push(tag);
      if (/aria-hidden="true"/.test(attrs)) { depth.push(true); hidden++; } else { depth.push(false); }
      if (stack.length > maxDepth) { maxDepth = stack.length; }
    }
    if (/data-lieq-fader=/.test(attrs) && hidden > 0) { faderHidden++; }
  }
  ok('every tag is balanced', balanced && stack.length === 0,
     stack.length ? 'left open: ' + stack.join(',') : 'nesting depth ' + maxDepth);
  ok('the faders are not inside an aria-hidden group', faderHidden === 0);

  var hooks = ['sArea', 'sCore', 'sFlow', 'tHalo', 'tCore', 'tFlow', 'annLF', 'annHF',
               'gapStem', 'gapTop', 'gapBot', 'hfDrop', 'hfHalo', 'hfDot', 'gapValue'];
  var missing = [];
  for (i = 0; i < hooks.length; i++) {
    if (src.indexOf('data-lieq="' + hooks[i] + '"') < 0) { missing.push(hooks[i]); }
  }
  ok('every element render() writes to exists in the markup', missing.length === 0,
     missing.length ? 'missing ' + missing.join(',') : hooks.length + ' hooks');

  ok('two faders', (src.match(/data-lieq-fader="/g) || []).length === 2);
  ok('both faders are role=slider', (src.match(/role="slider"/g) || []).length === 2);
  ok('both faders are focusable', (src.match(/tabindex="0"/g) || []).length === 2);
  ok('both faders carry aria-valuemin/max/now',
     (src.match(/aria-valuemin="0"/g) || []).length === 2 &&
     (src.match(/aria-valuemax="100"/g) || []).length === 2 &&
     (src.match(/aria-valuenow="0"/g) || []).length === 2);
  ok('both faders carry aria-valuetext and aria-orientation',
     (src.match(/aria-valuetext="0%"/g) || []).length === 2 &&
     (src.match(/aria-orientation="vertical"/g) || []).length === 2);
  ok('both faders are labelled', src.indexOf('aria-label="LF equalizer strength') > 0 &&
     src.indexOf('aria-label="HF equalizer strength') > 0);
  ok('both faders carry a focus ring element', (src.match(/lieq-focusring/g) || []).length === 2);
  ok('svg is a group with a label and a long description',
     /role="group"/.test(src) && /aria-label="/.test(src) && /<desc id="lieqDesc/.test(src));
  ok('no role="img" (which would hide the sliders)', src.indexOf('role="img"') < 0);
  ok('the figure\'s own labels are on the axes',
     src.indexOf('Graph frequency') > 0 && src.indexOf('Normalized spectral energy') > 0);
  ok('both annotations from the paper figure are present',
     src.indexOf('LOW-FREQUENCY') > 0 && src.indexOf('SPURIOUS HIGH-FREQUENCY') > 0);

  /* id collisions between two instances on one page */
  var a = EQ.renderMarkup(null, '1'), b = EQ.renderMarkup(null, '2');
  ok('def ids are namespaced per instance',
     a.indexOf('lieqGlow1') > 0 && b.indexOf('lieqGlow2') > 0 && b.indexOf('lieqGlow1') < 0);

  /* escaping */
  var evil = EQ.renderMarkup({ teacherLabel: '<script>x</script>' });
  ok('option strings are escaped', evil.indexOf('<script>') < 0 && evil.indexOf('&lt;script&gt;') > 0);
})();

/* ------------------------------------------------------------------ */
head('ROBUSTNESS');

ok('loads with no DOM at all', typeof EQ.init === 'function');
ok('init on a missing element returns null instead of throwing', EQ.init('#not-here') === null);
ok('init on null returns null', EQ.init(null) === null);
ok('initAll on a missing selector returns an empty list', EQ.initAll('#not-here').length === 0);
ok('custom curves of a different length are accepted by the maths', (function () {
  var t = [], s = [], i;
  for (i = 0; i < 24; i++) { t.push(Math.sin(i / 4)); s.push(Math.cos(i / 3)); }
  var tt = EQ.math.split(t, 2, 2), sstt = EQ.math.split(s, 2, 2);
  var out = EQ.math.morph(sstt.lo, sstt.hi, tt.lo, tt.hi, 1, 1);
  return maxAbsDiff(out, t) < 1e-12;
})());
ok('smooth() with radius 0 or 0 passes is the identity',
   maxAbsDiff(EQ.math.smooth(student, 0, 3), student) === 0 &&
   maxAbsDiff(EQ.math.smooth(student, 3, 0), student) === 0);
ok('parseCurve tolerates junk', (function () {
  var p = EQ.math.parseCurve('1,,x,0.5');
  return p.length === 4 && p[0] === 1 && p[1] === 0 && p[2] === 0 && p[3] === 0.5;
})());

/* ------------------------------------------------------------------ */
console.log('\n' + (fail === 0 ? 'ALL PASS' : 'FAILURES') + ': ' + pass + ' passed, ' + fail + ' failed.');
process.exit(fail === 0 ? 0 : 1);
