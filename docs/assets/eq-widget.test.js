/*
 * eq-widget.test.js — assertions for assets/eq-widget.js.
 *
 * Runs in plain node with no DOM, no jsdom, no test framework:
 *     node docs/assets/eq-widget.test.js
 *
 * It covers the things that are easy to get wrong and impossible to see in a
 * screenshot: the morph algebra, the full-travel residual, the pluck spring
 * (deform, settle, and the guarantee that a pluck cannot move a fader), the
 * markup's shape, and the geometry arithmetic.
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

var RESIDUAL = EQ.constants.FULL_SCALE_RESIDUAL;
var REACH = EQ.math.reach(student, teacher, RESIDUAL);

/* the raw family (no residual) and the family the widget actually draws */
function raw(lf, hf) { return EQ.math.morph(S.lo, S.hi, T.lo, T.hi, lf, hf); }
function morph(lf, hf) { return raw(lf * REACH, hf * REACH); }

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
head('RESIDUAL — full travel lands NEAR the teacher, not ON it');

var c00 = morph(0, 0), c11 = morph(1, 1), c10 = morph(1, 0), c01 = morph(0, 1);
var gap0 = maxAbsDiff(student, teacher);
var gap11 = maxAbsDiff(c11, teacher);

ok('lf=0, hf=0 reproduces the student exactly', maxAbsDiff(c00, student) < 1e-12,
   'max ' + sci(maxAbsDiff(c00, student)));
ok('lf=1, hf=1 is NOT the teacher', gap11 > 1e-6, 'peak gap ' + gap11.toFixed(6));
ok('lf=1, hf=1 sits exactly FULL_SCALE_RESIDUAL from the teacher',
   Math.abs(gap11 - RESIDUAL) < 1e-12,
   'residual ' + gap11.toFixed(6) + ' vs constant ' + RESIDUAL + ', |diff| ' +
   sci(Math.abs(gap11 - RESIDUAL)));
ok('that residual is a few per cent of the signal range',
   gap11 > 0.01 && gap11 < 0.08, (gap11 * 100).toFixed(1) + '% of the 0..1 range');
ok('full travel still closes most of the gap', gap11 < 0.12 * gap0,
   (100 * (1 - gap11 / gap0)).toFixed(1) + '% of a ' + gap0.toFixed(4) + ' peak gap closed');
ok('reach is the stated fraction of the student-to-teacher path',
   Math.abs(REACH - (1 - RESIDUAL / gap0)) < 1e-12, 'reach ' + REACH.toFixed(5));
ok('the limit curve is the teacher with a trace of the student still in it',
   (function () {
     var k = 1 - REACH, i, want, worst = 0;
     for (i = 0; i < N; i++) {
       want = teacher[i] + k * (student[i] - teacher[i]);
       if (Math.abs(c11[i] - want) > worst) { worst = Math.abs(c11[i] - want); }
     }
     return worst < 1e-12;
   })(), 'teacher + ' + (100 * (1 - REACH)).toFixed(1) + '% of the original error, max dev ' +
   sci((function () {
     var k = 1 - REACH, i, worst = 0, want;
     for (i = 0; i < N; i++) {
       want = teacher[i] + k * (student[i] - teacher[i]);
       if (Math.abs(c11[i] - want) > worst) { worst = Math.abs(c11[i] - want); }
     }
     return worst;
   })()));
note('the PEAK GAP readout therefore reads ' + gap11.toFixed(2) + ' at full travel, never 0.00.');

/* ------------------------------------------------------------------ */
head('MORPH — the four corners (in the raw, residual-free family)');

ok('raw lf=1, hf=1 reproduces the teacher to 1e-9', maxAbsDiff(raw(1, 1), teacher) < 1e-9,
   'max ' + sci(maxAbsDiff(raw(1, 1), teacher)));
ok('raw lf=1, hf=0 is exactly teacherLo + studentHi', maxAbsDiff(raw(1, 0), add(T.lo, S.hi)) < 1e-12,
   'max ' + sci(maxAbsDiff(raw(1, 0), add(T.lo, S.hi))));
ok('raw lf=0, hf=1 is exactly studentLo + teacherHi', maxAbsDiff(raw(0, 1), add(S.lo, T.hi)) < 1e-12,
   'max ' + sci(maxAbsDiff(raw(0, 1), add(S.lo, T.hi))));
ok('the drawn family is the raw family with the travel shortened',
   maxAbsDiff(c10, raw(REACH, 0)) < 1e-12 && maxAbsDiff(c01, raw(0, REACH)) < 1e-12);

/* ------------------------------------------------------------------ */
head('MORPH — at lf=1, hf=0 the envelope is the teacher\'s, the detail is still the student\'s');

(function () {
  /* An INDEPENDENT cross-check with two instruments the widget does not itself
   * use, because re-running the widget's own box blur over its own output
   * measures the blur as much as the result (a box blur is not a projection).
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
     envToT < 0.25 * envGap0 && envToT * 3 < envToS,
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
})();

/* ------------------------------------------------------------------ */
head('MORPH — continuity and monotonicity');

(function () {
  var hfFixed = [0, 0.25, 0.5, 0.75, 1], lfFixed = [0, 0.25, 0.5, 0.75, 1];
  var steps = 50, k, j, i, a, b, slope, bad = 0, worstLip = 0, expect, got;

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

  var d = 1e-3, worst = 0;
  for (j = 0; j < 20; j++) {
    a = morph(j / 20, 0.37);
    b = morph(j / 20 + d, 0.37);
    for (i = 0; i < N; i++) {
      expect = d * REACH * (T.lo[i] - S.lo[i]);
      got = b[i] - a[i];
      if (Math.abs(got - expect) > worst) { worst = Math.abs(got - expect); }
      if (Math.abs(got) > worstLip) { worstLip = Math.abs(got); }
    }
  }
  ok('response is exactly linear in LF (no cross term, no easing)', worst < 1e-12,
     'max deviation ' + sci(worst));
  note('Lipschitz: a 0.001 LF step moves no sample by more than ' + worstLip.toFixed(6) + '.');

  var midOK = true;
  var half = morph(0.5, 0.5);
  for (i = 0; i < N; i++) {
    if (Math.abs(half[i] - 0.5 * (c00[i] + c11[i])) > 1e-12) { midOK = false; }
  }
  ok('lf=hf=0.5 is the exact midpoint of the rest and full-travel poses', midOK);
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
       (EQ.geometry.PLOT.b - EQ.geometry.PLOT.t - EQ.geometry.PLOT_HEAD) + ' / vmax.');
})();

/* ------------------------------------------------------------------ */
head('PLUCK — the deformation kernel');

var SPAN = Math.max(2, EQ.constants.PLUCK_SPAN_FRAC * (N - 1));

(function () {
  var i, prev, monoUp = true, monoDown = true;
  ok('bump is 1 at the grab point', EQ.math.bump(0, SPAN) === 1);
  ok('bump is 0 at and beyond the support', EQ.math.bump(SPAN, SPAN) === 0 &&
     EQ.math.bump(-SPAN, SPAN) === 0 && EQ.math.bump(SPAN * 2, SPAN) === 0);
  ok('bump is symmetric', Math.abs(EQ.math.bump(3.7, SPAN) - EQ.math.bump(-3.7, SPAN)) < 1e-15);
  ok('bump is smooth, not a spike: it is still above 0.5 half a span out',
     EQ.math.bump(SPAN / 2, SPAN) > 0.49 && EQ.math.bump(SPAN / 2, SPAN) < 0.51,
     'w(span/2) = ' + EQ.math.bump(SPAN / 2, SPAN).toFixed(4));
  prev = 1;
  for (i = 0; i <= 200; i++) {
    var w = EQ.math.bump(i * SPAN / 200, SPAN);
    if (w > prev + 1e-12) { monoDown = false; }
    prev = w;
  }
  ok('bump falls off monotonically', monoDown);
  note('support is +-' + SPAN.toFixed(1) + ' of ' + N + ' samples (' +
       (200 * SPAN / (N - 1)).toFixed(0) + '% of the plot width).');
  monoUp = monoUp; /* keep the linter quiet */
})();

/* ------------------------------------------------------------------ */
head('PLUCK — drag deforms, release springs back exactly');

(function () {
  var base = morph(0.4, 0.6);
  var pluck = new EQ.PluckState({ span: SPAN });
  var i0 = 48, AMP = 0.3, now = 1000;
  var held, maxDev, edge;

  pluck.begin('s', i0, base[i0]);
  pluck.to(base[i0] + AMP);
  held = pluck.apply(base.slice(0), 's', pluck.amp(now));

  ok('a drag moves the grabbed sample by exactly the drag amount',
     Math.abs(held[i0] - (base[i0] + AMP)) < 1e-12,
     'moved ' + (held[i0] - base[i0]).toFixed(6));
  maxDev = maxAbsDiff(held, base);
  ok('the deformation never exceeds the drag amount', Math.abs(maxDev - AMP) < 1e-12,
     'max deviation ' + maxDev.toFixed(6));
  edge = 0;
  for (var i = 0; i < N; i++) {
    if (Math.abs(i - i0) >= SPAN && Math.abs(held[i] - base[i]) > edge) { edge = Math.abs(held[i] - base[i]); }
  }
  ok('the deformation is local: nothing outside the support moves', edge === 0,
     'max move outside +-' + SPAN.toFixed(1) + ' samples: ' + edge);
  /* measure the DEFORMATION's own step, not the curve's slope underneath it */
  var step1 = Math.abs((held[i0] - base[i0]) - (held[i0 - 1] - base[i0 - 1]));
  ok('the deformation is smooth, not a spike',
     step1 < AMP * 0.03,
     'the bump rises ' + step1.toFixed(5) + ' per sample at its apex, ' +
     (100 * step1 / AMP).toFixed(1) + '% of a ' + AMP + ' pull');

  /* the other curve is untouched */
  ok('plucking the student leaves the teacher alone',
     maxAbsDiff(pluck.apply(teacher.slice(0), 't', pluck.amp(now)), teacher) === 0);

  /* ---- release ---- */
  pluck.release(now);
  var t, a, settleAt = -1, overshoot = 0, crossings = 0, prevSign = 1;
  for (t = 0; t <= 1200; t += 4) {
    a = EQ.math.spring(AMP, t / 1000, EQ.constants.PLUCK_ZETA, EQ.constants.PLUCK_FREQ);
    if (settleAt >= 0) { continue; }           /* only what the eye actually sees */
    if (a < -overshoot) { overshoot = -a; }
    if (Math.sign(a) !== 0 && Math.sign(a) !== prevSign) { crossings++; prevSign = Math.sign(a); }
    if (t / 1000 > EQ.constants.PLUCK_MIN_TIME &&
        Math.abs(a) < EQ.constants.PLUCK_SETTLE_EPS) { settleAt = t; }
  }
  ok('the spring is underdamped: it swings back past rest', overshoot > 0.02 * AMP,
     'overshoot ' + (100 * overshoot / AMP).toFixed(1) + '% of the pull');
  ok('it is not a slow glide: it settles in a few hundred ms',
     settleAt > 120 && settleAt < 600, 'settles at ' + settleAt + ' ms');
  ok('it reads as a plucked cord, not a single bounce: 2-5 visible swings',
     crossings >= 2 && crossings <= 5,
     crossings + ' zero crossings before it settles, period ' +
     (1000 / EQ.constants.PLUCK_FREQ).toFixed(0) + ' ms');

  /* drive the state machine exactly as the raf loop would */
  var frames = 0, last = null;
  for (t = 0; t <= 1500 && !pluck.settled(); t += 16) {
    last = pluck.apply(base.slice(0), 's', pluck.amp(now + t));
    frames++;
  }
  ok('the pluck state clears itself once settled', pluck.settled(), frames + ' frames');
  ok('after settling the curve is the slider-determined shape to 1e-9',
     maxAbsDiff(last, base) < 1e-9, 'max |after - before| = ' + sci(maxAbsDiff(last, base)));
  ok('and in fact it is bit-for-bit identical', maxAbsDiff(last, base) === 0);
  ok('a settled pluck is a no-op on any curve',
     pluck.apply(base.slice(0), 's', pluck.amp(now + 9999)) !== null &&
     maxAbsDiff(pluck.apply(base.slice(0), 's', pluck.amp(now + 9999)), base) === 0);
  note('spring: zeta ' + EQ.constants.PLUCK_ZETA + ', ' + EQ.constants.PLUCK_FREQ +
       ' Hz, settle threshold ' + EQ.constants.PLUCK_SETTLE_EPS + ' signal units.');
})();

/* ------------------------------------------------------------------ */
head('PLUCK — a curve drag cannot move a fader');

(function () {
  /* Drive the instance's own pluck methods with a DOM-free `this`. These are
   * the exact three functions the pointer handlers call. */
  var fake = {
    destroyed: false,
    state: { lf: 0.4, hf: 0.6 },
    pluck: new EQ.PluckState({ span: SPAN }),
    plucked: false,
    scheduled: 0,
    sprung: 0,
    schedule: function () { this.scheduled++; },
    startSpring: function () { this.sprung++; },
    reducedMotion: function () { return false; }
  };
  var base = morph(fake.state.lf, fake.state.hf);
  var i0 = 61;

  EQ.Instance.prototype.pluckBegin.call(fake, 's', i0, base[i0]);
  EQ.Instance.prototype.pluckTo.call(fake, base[i0] - 0.22);
  EQ.Instance.prototype.pluckTo.call(fake, base[i0] + 0.31);

  var during = fake.pluck.apply(base.slice(0), 's', fake.pluck.amp(0));
  ok('the drag really did deform the curve', maxAbsDiff(during, base) > 0.3,
     'max deviation ' + maxAbsDiff(during, base).toFixed(4));
  ok('lf is untouched by a curve drag', fake.state.lf === 0.4, 'lf = ' + fake.state.lf);
  ok('hf is untouched by a curve drag', fake.state.hf === 0.6, 'hf = ' + fake.state.hf);

  EQ.Instance.prototype.pluckRelease.call(fake, 0);
  ok('releasing a curve starts the spring, it does not snap', fake.sprung === 1);
  ok('lf and hf are still untouched after release',
     fake.state.lf === 0.4 && fake.state.hf === 0.6);

  var after = fake.pluck.apply(base.slice(0), 's', fake.pluck.amp(2000));
  ok('and the settled shape is exactly what lf=0.4, hf=0.6 imply',
     maxAbsDiff(after, morph(0.4, 0.6)) === 0,
     'max |after - morph(0.4,0.6)| = ' + maxAbsDiff(after, morph(0.4, 0.6)));

  /* reduced motion takes the spring away but not the correctness */
  var quiet = {
    destroyed: false, state: { lf: 0.2, hf: 0.9 }, pluck: new EQ.PluckState({ span: SPAN }),
    scheduled: 0, sprung: 0,
    schedule: function () { this.scheduled++; },
    startSpring: function () { this.sprung++; },
    reducedMotion: function () { return true; }
  };
  var qbase = morph(0.2, 0.9);
  EQ.Instance.prototype.pluckBegin.call(quiet, 't', 30, teacher[30]);
  EQ.Instance.prototype.pluckTo.call(quiet, teacher[30] + 0.2);
  EQ.Instance.prototype.pluckRelease.call(quiet, 0);
  ok('with prefers-reduced-motion the pluck is dropped, not animated',
     quiet.sprung === 0 && quiet.pluck.settled());
  ok('and the shape is immediately back to the slider pose',
     maxAbsDiff(quiet.pluck.apply(qbase.slice(0), 's', quiet.pluck.amp(0)), qbase) === 0);
  ok('reduced motion leaves the faders alone too',
     quiet.state.lf === 0.2 && quiet.state.hf === 0.9);
})();

/* ------------------------------------------------------------------ */
head('PATH — Catmull-Rom to cubic Bezier');

(function () {
  var G = EQ.geometry, PL = G.PLOT, xs = [], ys = [], i, d, segs;
  for (i = 0; i < N; i++) {
    xs.push(PL.l + (PL.r - PL.l) * i / (N - 1));
    ys.push(PL.b - (PL.b - PL.t - G.PLOT_HEAD) * student[i]);
  }
  d = EQ.math.catmullPath(xs, ys);
  segs = d.split('C').length - 1;
  ok('one cubic segment per interval', segs === N - 1, segs + ' segments for ' + N + ' points');
  ok('starts at the first sample', d.indexOf('M' + PL.l + ' ' + PL.b) === 0, d.slice(0, 18) + '...');
  ok('ends at the last sample', new RegExp('C[^C]*\\s' + PL.r + ' ' + PL.b + '$').test(d),
     '...' + d.slice(-22));
  ok('only M and C commands, all coordinates finite', /^M[-\d. ]+(C[-\d. ]+)+$/.test(d));
  ok('coordinates are quantised to 0.1', !/\.\d\d/.test(d));
  ok('degenerate input does not throw',
     EQ.math.catmullPath([], []) === '' && EQ.math.catmullPath([1], [1]) === '');
  note('path length at rest: ' + d.length + ' chars; two of these are rebuilt per frame.');
})();

/* ------------------------------------------------------------------ */
head('GEOMETRY — footprint');

(function () {
  var G = EQ.geometry, aspect = G.VB_W / G.VB_H;
  /* the hero column at the page's own numbers: 1180px container, 40px gutters,
   * 52px grid gap, 1.22fr of the pair, 25px card padding, 1px border */
  var col = (1180 - 80 - 52) * 1.22 / 2, inner = col - 50 - 2;
  ok('the card is wider than it is tall by more than the old 1.679',
     aspect > 1.9, 'viewBox ' + G.VB_W + ' x ' + G.VB_H + ', aspect ' + aspect.toFixed(3));
  note('at the hero width the SVG paints ' + inner.toFixed(0) + ' x ' +
       (inner / aspect).toFixed(0) + ' CSS px (was ' + inner.toFixed(0) + ' x ' +
       (inner / (470 / 280)).toFixed(0) + ' at 470x280): ' +
       (inner / (470 / 280) - inner / aspect).toFixed(0) + ' px shorter.');
  ok('everything computed stays inside the 8-unit inset',
     G.PLOT.l >= G.INSET && G.PLOT.r <= G.VB_W - G.INSET &&
     G.PLOT.t >= G.INSET && G.PLOT.b <= G.VB_H - G.INSET &&
     G.RACK.x + G.RACK.w <= G.VB_W - G.INSET && G.RACK.y >= G.INSET &&
     G.RACK.y + G.RACK.h <= G.VB_H - G.INSET);
  ok('the bottom row of type clears the inset', G.ROW.chip + G.CHIP.h / 2 <= G.VB_H - G.INSET,
     'lowest ink at ' + (G.ROW.chip + G.CHIP.h / 2) + ' of ' + (G.VB_H - G.INSET));
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
                'lieq-tick-t', 'lieq-rack-eyebrow', 'lieq-val', 'lieq-cap-label',
                'lieq-chip-t', 'lieq-hint'], i;
    for (i = 0; i < want.length; i++) { if (!styleOf(want[i], narrow).px) { return false; } }
    return true;
  })());

  var legendEnd = LG.t2 + w(D.studentLabel, 'lieq-leg');
  ok('legend fits inside the plot' + tag, legendEnd < PL.r,
     'ends at ' + legendEnd.toFixed(1) + ' of ' + PL.r);
  ok('legend entries do not collide' + tag, LG.t1 + w(D.teacherLabel, 'lieq-leg') < LG.sw2,
     'first ends at ' + (LG.t1 + w(D.teacherLabel, 'lieq-leg')).toFixed(1) +
     ', second swatch at ' + LG.sw2);

  var hintW = Math.max(w(D.hint[0], 'lieq-hint'), w(D.hint[1], 'lieq-hint'));
  ok('the pluck hint clears the legend' + tag, G.HINT.x - hintW > legendEnd + 4,
     'hint starts at ' + (G.HINT.x - hintW).toFixed(1) + ', legend ends at ' + legendEnd.toFixed(1));

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
  var xt = w(D.xLabel, 'lieq-axis-t');
  ok('x-axis title clears both end ticks' + tag,
     mid - xt / 2 > PL.l + w(D.lowTick, 'lieq-tick-t') + 4 &&
     mid + xt / 2 < PL.r - w(D.highTick, 'lieq-tick-t') - 4,
     'title spans ' + (mid - xt / 2).toFixed(1) + '..' + (mid + xt / 2).toFixed(1));
  var yt = w(D.yLabel, 'lieq-axis-t');
  ok('rotated y-axis title fits the plot height' + tag, yt < PL.b - PL.t,
     yt.toFixed(1) + ' of ' + (PL.b - PL.t));
  ok('end ticks do not collide' + tag,
     PL.l + w(D.lowTick, 'lieq-tick-t') < PL.r - w(D.highTick, 'lieq-tick-t'));
  ok('rack eyebrow fits the rack' + tag, w(D.rackLabel, 'lieq-rack-eyebrow') < RK.w - 8,
     w(D.rackLabel, 'lieq-rack-eyebrow').toFixed(1) + ' of ' + (RK.w - 8));
  ok('value readout fits between the faders' + tag,
     w('0.00', 'lieq-val') < G.FADER_X.hf - G.FADER_X.lf);

  /* the preset row and the readout share the bottom row */
  var chipEnd = PL.l, i2;
  for (i2 = 0; i2 < D.presets.length; i2++) {
    chipEnd += G.chipWidth(D.presets[i2].label) + (i2 < D.presets.length - 1 ? G.CHIP.gap : 0);
  }
  var roW = Math.max(w(D.readoutLabel + ' 0.00', 'lieq-ro-k'), w(D.scrubLabel + ' 0.00', 'lieq-ro-k'));
  ok('the preset pills and the readout do not collide' + tag, chipEnd < PL.r - roW - 6,
     'pills end at ' + chipEnd.toFixed(1) + ', readout starts at ' + (PL.r - roW).toFixed(1));
  ok('every preset label fits its pill' + tag, (function () {
    var i3, lab, box;
    for (i3 = 0; i3 < D.presets.length; i3++) {
      lab = w(D.presets[i3].label, 'lieq-chip-t');
      box = G.chipWidth(D.presets[i3].label);
      if (lab > box - 4) { return false; }
    }
    return true;
  })(), 'widest label ' + w('LF+HF', 'lieq-chip-t').toFixed(1) + ' in a ' +
     G.chipWidth('LF+HF') + ' pill');
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
  var hit = 50, phoneCss = hit * (320 / G.VB_W);
  ok('fader hit areas are wide enough to touch and do not overlap',
     phoneCss > 30 && G.FADER_X.lf + hit / 2 < G.FADER_X.hf - hit / 2,
     phoneCss.toFixed(1) + ' CSS px wide on a 390px phone');
  ok('hit areas stay inside the rack',
     G.FADER_X.lf - hit / 2 >= RK.x && G.FADER_X.hf + hit / 2 <= RK.x + RK.w);
  note('preset pills are ' + G.CHIP.hit + ' user units tall = ' +
       (G.CHIP.hit * 320 / G.VB_W).toFixed(0) + ' CSS px on a 390px phone, ' +
       (G.CHIP.hit * 587 / G.VB_W).toFixed(0) + ' on the desktop hero.');
  note('the pluck band is 22 user units wide = ' + (22 * 320 / G.VB_W).toFixed(0) +
       ' CSS px on a phone, ' + (22 * 587 / G.VB_W).toFixed(0) + ' on the desktop hero.');
})();

/* ------------------------------------------------------------------ */
head('MARKUP — shape, hooks and accessibility');

(function () {
  var src = EQ.renderMarkup(), stack = [], depth = [], i, m, tag, self, hidden = 0,
      faderHidden = 0, chipHidden = 0,
      re = /<(\/?)([A-Za-z][\w:-]*)((?:[^>"']|"[^"]*"|'[^']*')*?)(\/?)>/g,
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
    if (/data-lieq-preset=/.test(attrs) && hidden > 0) { chipHidden++; }
  }
  ok('every tag is balanced', balanced && stack.length === 0,
     stack.length ? 'left open: ' + stack.join(',') : 'nesting depth ' + maxDepth);
  ok('no control is inside an aria-hidden group', faderHidden === 0 && chipHidden === 0);

  var hooks = ['sArea', 'sCore', 'sFlow', 'tHalo', 'tCore', 'tFlow', 'sGrab', 'tGrab',
               'annLF', 'annHF', 'gapStem', 'gapTop', 'gapBot', 'hfDrop', 'hfHalo', 'hfDot',
               'gapValue', 'roLabel', 'hint', 'scrub', 'scrubLine', 'scrubDotS', 'scrubDotT'];
  var missing = [];
  for (i = 0; i < hooks.length; i++) {
    if (src.indexOf('data-lieq="' + hooks[i] + '"') < 0) { missing.push(hooks[i]); }
  }
  ok('every element render() writes to exists in the markup', missing.length === 0,
     missing.length ? 'missing ' + missing.join(',') : hooks.length + ' hooks');

  ok('two faders', (src.match(/data-lieq-fader="/g) || []).length === 2);
  ok('both faders are role=slider', (src.match(/role="slider"/g) || []).length === 2);
  ok('three preset pills, each a button', (src.match(/data-lieq-preset="/g) || []).length === 3 &&
     (src.match(/role="button"/g) || []).length === 3);
  ok('every control is focusable', (src.match(/tabindex="0"/g) || []).length === 5);
  ok('both faders carry aria-valuemin/max/now',
     (src.match(/aria-valuemin="0"/g) || []).length === 2 &&
     (src.match(/aria-valuemax="100"/g) || []).length === 2 &&
     (src.match(/aria-valuenow="0"/g) || []).length === 2);
  ok('both faders carry aria-valuetext and aria-orientation',
     (src.match(/aria-valuetext="0%"/g) || []).length === 2 &&
     (src.match(/aria-orientation="vertical"/g) || []).length === 2);
  ok('both faders are labelled', src.indexOf('aria-label="LF equalizer strength') > 0 &&
     src.indexOf('aria-label="HF equalizer strength') > 0);
  ok('every preset is labelled and carries aria-pressed',
     (src.match(/aria-pressed="/g) || []).length === 3 &&
     src.indexOf('aria-label="Preset: bypass') > 0);
  ok('every control carries a focus ring element', (src.match(/lieq-focusring/g) || []).length === 5);
  ok('svg is a group with a label and a long description',
     /role="group"/.test(src) && /aria-label="/.test(src) && /<desc id="lieqDesc/.test(src));
  ok('no role="img" (which would hide the sliders)', src.indexOf('role="img"') < 0);

  ok('the axes are labelled in the SPATIAL domain, as the source figure is',
     src.indexOf('Spatial position') > 0 && src.indexOf('Signal / feature response') > 0 &&
     src.indexOf('Graph frequency') < 0 && src.indexOf('spectral energy') < 0);
  ok('both annotations from the paper figure are present',
     src.indexOf('LOW-FREQUENCY') > 0 && src.indexOf('SPURIOUS HIGH-FREQUENCY') > 0);
  ok('the description states that full travel does NOT reach the teacher',
     /does not reach it/.test(src) && /residual of four per cent/.test(src));
  ok('the description states that a pluck cannot move a fader',
     /never changes the faders/.test(src));
  ok('the illustrative-not-measured disclosure reaches the accessible description',
     src.indexOf('Illustrative, not measured') > 0 && src.indexOf('synthesised') > 0);

  var a = EQ.renderMarkup(null, '1'), b = EQ.renderMarkup(null, '2');
  ok('def ids are namespaced per instance',
     a.indexOf('lieqGlow1') > 0 && b.indexOf('lieqGlow2') > 0 && b.indexOf('lieqGlow1') < 0);

  var evil = EQ.renderMarkup({ teacherLabel: '<script>x</script>' });
  ok('option strings are escaped', evil.indexOf('<script>') < 0 && evil.indexOf('&lt;script&gt;') > 0);
})();

/* ------------------------------------------------------------------ */
head('CAPTION — the illustrative disclosure');

ok('the module exposes a one-line caption', typeof EQ.caption === 'string' && EQ.caption.length > 40);
ok('it is one line', EQ.caption.indexOf('\n') < 0);
ok('it says the curves are not measured', /[Ii]llustrative, not measured/.test(EQ.caption));
ok('it names what was synthesised', /ripple is synthesised/.test(EQ.caption));
ok('the default option carries the same string', EQ.defaults.caption === EQ.caption);
note('caption: ' + EQ.caption);

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
ok('reach() on identical curves is 1 (nothing to close)',
   EQ.math.reach(teacher, teacher, RESIDUAL) === 1);
ok('reach() never goes negative, however big the residual',
   EQ.math.reach(student, teacher, 99) === 0);
ok('smooth() with radius 0 or 0 passes is the identity',
   maxAbsDiff(EQ.math.smooth(student, 0, 3), student) === 0 &&
   maxAbsDiff(EQ.math.smooth(student, 3, 0), student) === 0);
ok('parseCurve tolerates junk', (function () {
  var p = EQ.math.parseCurve('1,,x,0.5');
  return p.length === 4 && p[0] === 1 && p[1] === 0 && p[2] === 0 && p[3] === 0.5;
})());
ok('a PluckState with nothing grabbed is a no-op', (function () {
  var p = new EQ.PluckState({ span: SPAN }), v = student.slice(0);
  p.to(0.5); p.release(0);
  return p.amp(0) === 0 && maxAbsDiff(p.apply(v, 's', p.amp(0)), student) === 0;
})());
ok('a pluck amplitude is clamped to PLUCK_MAX_AMP', (function () {
  var p = new EQ.PluckState({ span: SPAN });
  p.begin('s', 10, 0); p.to(999);
  return Math.abs(p.amp(0) - EQ.constants.PLUCK_MAX_AMP) < 1e-12;
})(), 'max ' + EQ.constants.PLUCK_MAX_AMP);

/* ------------------------------------------------------------------ */
console.log('\n' + (fail === 0 ? 'ALL PASS' : 'FAILURES') + ': ' + pass + ' passed, ' + fail + ' failed.');
process.exit(fail === 0 ? 0 : 1);
