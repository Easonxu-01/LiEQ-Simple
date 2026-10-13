/*
 * eq-widget.test.js — assertions for assets/eq-widget.js (VERSION 3).
 *
 * Runs in plain node with no DOM, no jsdom, no test framework:
 *     node docs/assets/eq-widget.test.js
 *
 * It covers the things that are easy to get wrong and impossible to see in a
 * screenshot: the SPATIAL-domain semantics of the two curves (shape vs fine
 * detail, and which curve carries which), the morph algebra, the full-travel
 * residual, the nearest-curve hit test that replaced v2's two stacked grab
 * bands, the pluck spring, the markup's shape, and the geometry arithmetic.
 *
 * Where version 2 asserted the INVERTED semantics — a student with a spurious
 * high-frequency ripple and a comparatively smooth teacher, plus the two
 * frequency-domain callouts — the assertion is replaced, and the replacement
 * says so in its name.
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

/* The two instruments every spatial-domain claim in this file is measured
 * with. Neither is the widget's own machinery, so they cannot flatter it.
 *
 *   curvature(a)  max |a[i] - (a[i-1] + a[i+1]) / 2|, a discrete second
 *                 difference: how KINKED a curve is sample to sample. In the
 *                 spatial domain this is what "carries fine detail" means.
 *   texture(a)    RMS of that same second difference: ripple energy.
 * Both are blind to the envelope: a smooth hump of any height scores ~0. */
function d2(a) {
  var o = [], i, l, r;
  for (i = 0; i < a.length; i++) {
    l = a[i > 0 ? i - 1 : 0];
    r = a[i < a.length - 1 ? i + 1 : a.length - 1];
    o.push(a[i] - (l + r) / 2);
  }
  return o;
}
function curvature(a) { return maxAbs(d2(a)); }
function texture(a) {
  var k = d2(a), s = 0, i;
  for (i = 0; i < k.length; i++) { s += k[i] * k[i]; }
  return Math.sqrt(s / k.length);
}
/* A wide moving average: an envelope estimator the widget does not itself use
 * (its own four arrays are generated, never blurred out of a curve). */
function env(a) { return EQ.math.smooth(a, 7, 4); }
function argmax(a) { var k = 0, i; for (i = 1; i < a.length; i++) { if (a[i] > a[k]) { k = i; } } return k; }

/* Distinct humps of `a` between sample lo and hi that reach `frac` of the
 * array's peak. Local maxima are found over a +-3 window and then merged, so
 * the two samples either side of one summit do not count as three humps. */
function humps(a, lo, hi, frac) {
  var top = Math.max.apply(null, a), out = [], i, j, isMax;
  for (i = Math.max(3, lo); i <= Math.min(a.length - 4, hi); i++) {
    if (a[i] < frac * top) { continue; }
    isMax = true;
    for (j = i - 3; j <= i + 3; j++) { if (j !== i && a[j] > a[i]) { isMax = false; } }
    if (isMax && (!out.length || i - out[out.length - 1] > 10)) { out.push(i); }
  }
  return out;
}

/* ------------------------------------------------------------------ */
/* Set-up: the four arrays the widget is built from.                   */
/* ------------------------------------------------------------------ */

var C = EQ.curves;
var teacher = C.teacher.slice(0);
var student = C.student.slice(0);
var N = C.n;
/* T.lo / S.lo are ENVELOPES (the gross shape) and T.hi / S.hi are RIPPLES
 * (the fine spiky detail). Same shape as v2's split() output, so the morph
 * algebra below is unchanged — but these are generated, not blurred. */
var T = { lo: C.teacherEnvelope, hi: C.teacherRipple };
var S = { lo: C.studentEnvelope, hi: C.studentRipple };

var RESIDUAL = EQ.constants.FULL_SCALE_RESIDUAL;
var REACH = EQ.math.reach(student, teacher, RESIDUAL);
/* The DETAIL channel travels less far than the SHAPE channel, on purpose: the
 * paper's measurements leave the distilled student's high-frequency retention
 * short of the dense teacher's even where the coarse shape has been brought
 * into line. Every "drawn" helper below uses it, because the widget does —
 * `morph(lf, hf)` used to send hf over REACH as well, which made the whole
 * section below describe a curve nothing on the page draws. */
var DREACH = EQ.constants.DETAIL_REACH;
/* the share of the teacher's ripple the student is left carrying at full HF */
var DKEEP = (1 - DREACH) * EQ.constants.STUDENT_RIPPLE_KEEP + DREACH;

/* the raw family (no residual) and the family the widget actually draws */
function raw(lf, hf) { return EQ.math.morph(S.lo, S.hi, T.lo, T.hi, lf, hf); }
function morph(lf, hf) { return raw(lf * REACH, hf * DREACH); }
/* the drawn student's ENVELOPE alone, which is what LF moves */
function drawnEnv(lf) {
  var o = [], i, t = lf * REACH;
  for (i = 0; i < N; i++) { o.push(S.lo[i] + (T.lo[i] - S.lo[i]) * t); }
  return o;
}
/* the drawn student's RIPPLE alone, which is what HF grows */
function drawnRipple(hf) {
  var o = [], i, t = hf * DREACH;
  for (i = 0; i < N; i++) { o.push(S.hi[i] + (T.hi[i] - S.hi[i]) * t); }
  return o;
}
/* RMS of a slice, for the per-sixth texture measurements below */
function rmsOf(a, lo, hi) {
  var s = 0, i;
  for (i = lo; i < hi; i++) { s += a[i] * a[i]; }
  return Math.sqrt(s / (hi - lo));
}
/* The samples where buildRipple's baseline clamp fired: the spike the mask
 * floor allowed was deeper than the envelope could carry, so it was trimmed to
 * exactly -envelope and the drawn curve sits ON its own baseline there. Those
 * samples are a baseline touch, not a spike. */
function pinned() {
  var out = [], i;
  for (i = 0; i < N; i++) {
    if (teacher[i] <= 1e-12 && Math.abs(T.hi[i]) > 1e-12) { out.push(i); }
  }
  return out;
}

head('CURVES — four arrays, in the SPATIAL domain');
ok('the model is four arrays plus their two sums',
   T.lo.length === N && T.hi.length === N && S.lo.length === N && S.hi.length === N &&
   teacher.length === N && student.length === N, N + ' samples each');
ok('teacher = teacherEnvelope + teacherRipple, exactly',
   maxAbsDiff(add(T.lo, T.hi), teacher) < 1e-12, 'max ' + sci(maxAbsDiff(add(T.lo, T.hi), teacher)));
ok('student = studentEnvelope + studentRipple, exactly',
   maxAbsDiff(add(S.lo, S.hi), student) < 1e-12, 'max ' + sci(maxAbsDiff(add(S.lo, S.hi), student)));
ok('all samples finite, non-negative, and within the headroom', (function () {
  var i;
  for (i = 0; i < N; i++) {
    if (!isFinite(teacher[i]) || !isFinite(student[i])) { return false; }
    if (teacher[i] < -1e-9 || teacher[i] > 1.12) { return false; }
    if (student[i] < -1e-9 || student[i] > 1.12) { return false; }
  }
  return true;
})(), 'teacher peaks at ' + Math.max.apply(null, teacher).toFixed(3));
ok('both profiles start and end at exactly zero, so the filled areas close',
   teacher[0] === 0 && student[0] === 0 && teacher[N - 1] === 0 && student[N - 1] === 0);
ok('the curves are generated, not a table of typed coordinates',
   typeof EQ.math.buildModel === 'function' && typeof EQ.math.lobeSum === 'function' &&
   EQ.constants.TEACHER_LOBES.length >= 3 && EQ.constants.STUDENT_LOBES.length >= 2,
   EQ.constants.TEACHER_LOBES.length + ' teacher lobes, ' +
   EQ.constants.STUDENT_LOBES.length + ' student lobes');
ok('the model is deterministic: building it twice gives identical arrays',
   maxAbsDiff(EQ.math.buildModel({}).teacherRipple, T.hi) === 0 &&
   maxAbsDiff(EQ.math.buildModel({}).studentEnvelope, S.lo) === 0);

/* ------------------------------------------------------------------ */
head('CURVES — DEFECT 1: the sparse student is SMOOTH, the dense teacher is SPIKY');
note('v2 had this exactly inverted: it synthesised a ripple ON THE STUDENT and left the');
note('teacher comparatively smooth. In the paper figure the dense scan carries the fine');
note('detail and the sparse one lost it, so these assertions are the inverse of v2\'s.');

(function () {
  var kT = curvature(teacher), kS = curvature(student);
  ok('the student at lf=0, hf=0 is smooth: its second difference is tiny',
     kS < 0.01, 'max |2nd difference| ' + kS.toFixed(5) + ' of a ~1.0 range');
  ok('the teacher is NOT smooth: it carries fine detail everywhere',
     kT > 0.03, 'max |2nd difference| ' + kT.toFixed(5));
  ok('the teacher is an order of magnitude kinkier than the student',
     kT > 8 * kS, (kT / kS).toFixed(1) + 'x');
  ok('and in ripple ENERGY, not just one spike',
     texture(teacher) > 8 * texture(student),
     texture(teacher).toFixed(5) + ' vs ' + texture(student).toFixed(5) + ' RMS, ' +
     (texture(teacher) / texture(student)).toFixed(1) + 'x');
  ok('the student keeps only a trace of the texture, not none of it',
     texture(student) > 0 && EQ.constants.STUDENT_RIPPLE_KEEP > 0 &&
     EQ.constants.STUDENT_RIPPLE_KEEP < 0.15,
     'studentRipple = ' + (100 * EQ.constants.STUDENT_RIPPLE_KEEP).toFixed(0) +
     '% of teacherRipple');
  note('on screen that trace is ' +
       (maxAbs(S.hi) * (EQ.geometry.PLOT.b - EQ.geometry.PLOT.t - EQ.geometry.PLOT_HEAD) /
        EQ.curves.vmax).toFixed(2) + ' user units of amplitude: invisible, which is the point.');
})();

/* ------------------------------------------------------------------ */
head('CURVES — the teacher\'s detail is SPIKY, not a soft sine');

(function () {
  var r = T.hi, i, flips = 0, peaks = [], gaps = [], hiLo;
  for (i = 1; i < N; i++) { if (r[i] * r[i - 1] < 0) { flips++; } }
  ok('the detail changes sign dozens of times across the plot', flips > 24,
     flips + ' sign changes over ' + N + ' samples');
  for (i = 1; i < N - 1; i++) {
    if (Math.abs(r[i]) > Math.abs(r[i - 1]) && Math.abs(r[i]) > Math.abs(r[i + 1]) &&
        Math.abs(r[i]) > 0.2 * maxAbs(r)) { peaks.push(i); }
  }
  ok('it is a dense train of narrow excursions', peaks.length > 20,
     peaks.length + ' spikes, one every ' + (N / peaks.length).toFixed(1) + ' samples');
  for (i = 1; i < peaks.length; i++) { gaps.push(peaks[i] - peaks[i - 1]); }
  ok('the spikes are narrow: a spike and its neighbour are at most a few samples apart',
     Math.max.apply(null, gaps) <= 8, 'widest spacing ' + Math.max.apply(null, gaps) + ' samples');
  hiLo = (function () {
    var hs = [], j;
    for (j = 0; j < peaks.length; j++) { hs.push(Math.abs(r[peaks[j]])); }
    return Math.max.apply(null, hs) / Math.min.apply(null, hs);
  })();
  ok('spike heights VARY, so it reads as structure and not as a comb', hiLo > 2,
     'tallest / shortest spike = ' + hiLo.toFixed(1) + 'x');
  /* The sharpness claim is about SPIKES, and three samples are not spikes: the
   * baseline clamp in buildRipple trims a spike the envelope cannot carry to
   * exactly -envelope, which puts the drawn curve on its own baseline. Measured,
   * that is samples 6, 11 and 174, every one of them where the envelope is under
   * 0.04, and the one at 11 is the single blunt "spike" in the whole array
   * (0.38 of its height). Excluded by name, with the clamp's reach bounded here
   * so it can never quietly spread, and the trimmed samples themselves guarded
   * by the non-negativity assertion above. */
  (function () {
    var pin = pinned(), j, k, worst = 1, counted = 0, flankish = true;
    for (j = 0; j < pin.length; j++) { if (!(T.lo[pin[j]] < 0.04)) { flankish = false; } }
    for (j = 0; j < peaks.length; j++) {
      k = peaks[j];
      if (k < 1 || k > N - 2) { continue; }
      if (pin.indexOf(k) >= 0 || pin.indexOf(k + 1) >= 0) { continue; }
      counted++;
      worst = Math.min(worst, Math.abs(r[k] - r[k + 1]) / Math.abs(r[k]));
    }
    ok('a spike is sharp: the drop from a spike to the next sample is most of its height',
       worst > 0.5 && counted > 0.9 * peaks.length,
       'worst of ' + counted + ' spikes drops ' + (100 * worst).toFixed(0) + '% of its height');
    ok('and the only blunt samples are the few the clamp pinned to the baseline',
       pin.length <= 5 && flankish,
       pin.length + ' pinned samples (' + pin.join(', ') + '), all where the envelope is under 0.04');
  })();
  /* v2's wording was "no detail where there is no signal", and RIPPLE.maskFloor
   * retired it: multiplying by (envelope/max)^maskPow alone drove the outer
   * sixths to an RMS of 0.0093 and 0.0079 against 0.0345 mid-curve, i.e. flanks
   * a reader sees as smooth, which is the "too little high-frequency detail"
   * the floor was added for. The invariant that survives is weaker but still
   * has teeth, and it is the one the paper's insight figure draws: the detail is
   * STRONGEST where the signal is strongest, falls off from there, and never
   * vanishes. The old no-floor numbers fail every clause of it. */
  (function () {
    var sixths = [], esix = [], k, lo, hi, top = 0, etop = 0, decays = true;
    for (k = 0; k < 6; k++) {
      lo = Math.floor(k * N / 6); hi = Math.floor((k + 1) * N / 6);
      sixths.push(rmsOf(r, lo, hi));
      esix.push(rmsOf(T.lo, lo, hi));
      if (sixths[k] > sixths[top]) { top = k; }
      if (esix[k] > esix[etop]) { etop = k; }
    }
    for (k = top; k > 0; k--) { if (sixths[k - 1] > sixths[k]) { decays = false; } }
    for (k = top; k < 5; k++) { if (sixths[k + 1] > sixths[k]) { decays = false; } }
    ok('the detail rides ON the signal: strongest where the signal is strongest',
       top === etop && decays,
       'ripple RMS by sixths ' + sixths.map(function (v) { return v.toFixed(4); }).join(' ') +
       ', peak sixth ' + top + ' is where the envelope peaks too');
    ok('and it never vanishes: the flanks keep most of that texture',
       Math.min.apply(null, sixths) > 0.015 &&
       Math.min.apply(null, sixths) > 0.5 * Math.max.apply(null, sixths),
       'quietest sixth ' + Math.min.apply(null, sixths).toFixed(4) + ' is ' +
       (100 * Math.min.apply(null, sixths) / Math.max.apply(null, sixths)).toFixed(0) +
       '% of the loudest, floor ' + EQ.constants.RIPPLE.maskFloor +
       ' over envelope^' + EQ.constants.RIPPLE.maskPow);
    ok('and the two ends are still exactly zero, so the filled areas close',
       r[0] === 0 && r[N - 1] === 0);
  })();
  ok('the detail has no net offset: it is a detail term, not a shape term',
     Math.abs((function () { var s = 0, j; for (j = 0; j < N; j++) { s += r[j]; } return s / N; })()) < 0.01,
     'mean ' + ((function () { var s = 0, j; for (j = 0; j < N; j++) { s += r[j]; } return s / N; })()).toFixed(5));
})();

/* ------------------------------------------------------------------ */
head('CURVES — the sparse student\'s SHAPE is wrong: two humps, not one');

(function () {
  var tE = T.lo, sE = S.lo, i, tp = argmax(tE), dip, lo,
      /* the stretch the two shapes disagree over: everything left of the
       * shared right-hand shoulders, which both profiles get right */
      mid = Math.round(0.66 * (N - 1)),
      tHumps = humps(tE, 0, mid, 0.5),
      merged = humps(sE, 0, mid, 0.4);
  ok('the teacher envelope has ONE dominant hump', tHumps.length === 1,
     'peak at u = ' + (tp / (N - 1)).toFixed(2) + ', height ' + tE[tp].toFixed(2) +
     '; ' + tHumps.length + ' hump(s) over u = 0..0.66');
  ok('the student envelope has TWO comparable humps there instead', merged.length === 2,
     merged.map(function (k) { return 'u=' + (k / (N - 1)).toFixed(2) + ' h=' + sE[k].toFixed(2); }).join(', '));
  lo = Infinity;
  for (i = merged[0]; i <= merged[merged.length - 1]; i++) { if (sE[i] < lo) { lo = sE[i]; } }
  dip = lo / Math.min(sE[merged[0]], sE[merged[1]]);
  ok('with a real valley between them, so they read as two humps', dip < 0.5,
     'valley is ' + (100 * dip).toFixed(0) + '% of the shorter hump');
  ok('the student OVERSHOOTS the dense shape in one place',
     Math.max.apply(null, sub(sE, tE)) > 0.15,
     'by ' + Math.max.apply(null, sub(sE, tE)).toFixed(3) + ' at u = ' +
     (argmax(sub(sE, tE)) / (N - 1)).toFixed(2));
  ok('and falls far SHORT of it where the dense scan peaks',
     Math.max.apply(null, sub(tE, sE)) > 0.4,
     'by ' + Math.max.apply(null, sub(tE, sE)).toFixed(3) + ' at u = ' +
     (argmax(sub(tE, sE)) / (N - 1)).toFixed(2));
  ok('the student\'s peak is in the WRONG place',
     Math.abs(argmax(sE) - tp) > 0.1 * N,
     'student peaks at u = ' + (argmax(sE) / (N - 1)).toFixed(2) +
     ', teacher at u = ' + (tp / (N - 1)).toFixed(2));
  ok('the two shapes do agree somewhere, so the DETAIL callout has an honest home',
     (function () {
       var a = Math.round(EQ.constants.ANN_DETAIL_FROM * (N - 1)),
           b = Math.round(EQ.constants.ANN_DETAIL_TO * (N - 1)), j, worst = 0;
       for (j = a; j <= b; j++) { worst = Math.max(worst, Math.abs(sE[j] - tE[j])); }
       return worst < 0.06;
     })(), 'over u = ' + EQ.constants.ANN_DETAIL_FROM + '..' + EQ.constants.ANN_DETAIL_TO +
     ' the two shapes differ by less than 0.06');
  ok('and the dense scan\'s detail is clearly visible in that stretch',
     (function () {
       var a = Math.round(EQ.constants.ANN_DETAIL_FROM * (N - 1)),
           b = Math.round(EQ.constants.ANN_DETAIL_TO * (N - 1)), j, m = 0;
       for (j = a; j <= b; j++) { m = Math.max(m, Math.abs(T.hi[j])); }
       return m * (EQ.geometry.PLOT.b - EQ.geometry.PLOT.t - EQ.geometry.PLOT_HEAD) / C.vmax > 4;
     })(), 'spikes there are ' + (function () {
       var a = Math.round(EQ.constants.ANN_DETAIL_FROM * (N - 1)),
           b = Math.round(EQ.constants.ANN_DETAIL_TO * (N - 1)), j, m = 0;
       for (j = a; j <= b; j++) { m = Math.max(m, Math.abs(T.hi[j])); }
       return (m * (EQ.geometry.PLOT.b - EQ.geometry.PLOT.t - EQ.geometry.PLOT_HEAD) / C.vmax).toFixed(1);
     })() + ' user units tall');
})();

/* ------------------------------------------------------------------ */
head('RESIDUAL — full travel lands NEAR the teacher, not ON it');

var c00 = morph(0, 0), c11 = morph(1, 1), c10 = morph(1, 0), c01 = morph(0, 1);
var gap0 = maxAbsDiff(student, teacher);
var gap11 = maxAbsDiff(c11, teacher);

ok('lf=0, hf=0 reproduces the student exactly', maxAbsDiff(c00, student) < 1e-12,
   'max ' + sci(maxAbsDiff(c00, student)));
ok('lf=1, hf=1 is NOT the teacher', gap11 > 1e-6, 'peak gap ' + gap11.toFixed(6));
/* The two channels are left short by DIFFERENT amounts, so one number can no
 * longer describe the full-travel pose. What is still exact, at every one of
 * the 181 samples, is the closed form: the teacher plus (1 - REACH) of the
 * original SHAPE error plus (1 - DETAIL_REACH) of the original DETAIL error.
 * This pins all 181 samples where the retired assertion pinned one. */
ok('lf=1, hf=1 is the teacher plus exactly the two residuals the design leaves',
   (function () {
     var i, want, worst = 0;
     for (i = 0; i < N; i++) {
       want = teacher[i] + (1 - REACH) * (S.lo[i] - T.lo[i]) + (1 - DREACH) * (S.hi[i] - T.hi[i]);
       if (Math.abs(c11[i] - want) > worst) { worst = Math.abs(c11[i] - want); }
     }
     return worst < 1e-12;
   })(), 'teacher + ' + (100 * (1 - REACH)).toFixed(1) + '% of the shape error + ' +
   (100 * (1 - DREACH)).toFixed(1) + '% of the detail error');
ok('the SHAPE channel\'s own travel is still set by FULL_SCALE_RESIDUAL exactly',
   Math.abs(maxAbsDiff(raw(REACH, REACH), teacher) - RESIDUAL) < 1e-12,
   'both channels at REACH would sit ' + maxAbsDiff(raw(REACH, REACH), teacher).toFixed(6) +
   ' from the teacher, the constant is ' + RESIDUAL);
/* The peak gap is therefore BIGGER than FULL_SCALE_RESIDUAL, and bounded above
 * by the two channels' residuals added: the triangle inequality, in the inputs,
 * with no hand-typed number to drift. */
ok('so the drawn peak gap is more than the shape residual, and no more than both added',
   gap11 > RESIDUAL &&
   gap11 <= (1 - REACH) * maxAbsDiff(S.lo, T.lo) + (1 - DREACH) * maxAbsDiff(S.hi, T.hi) + 1e-12,
   'peak gap ' + gap11.toFixed(5) + ' in (' + RESIDUAL + ', ' +
   ((1 - REACH) * maxAbsDiff(S.lo, T.lo) + (1 - DREACH) * maxAbsDiff(S.hi, T.hi)).toFixed(5) + ']');
ok('that residual is a few per cent of the signal range',
   gap11 > 0.01 && gap11 < 0.12, (gap11 * 100).toFixed(1) + '% of the 0..1 range');
ok('full travel still closes most of the gap', gap11 < 0.12 * gap0,
   (100 * (1 - gap11 / gap0)).toFixed(1) + '% of a ' + gap0.toFixed(4) + ' peak gap closed');
ok('reach is the stated fraction of the student-to-teacher path',
   Math.abs(REACH - (1 - RESIDUAL / gap0)) < 1e-12, 'reach ' + REACH.toFixed(5));
/* and it has to be VISIBLE, which is the whole reason FULL_SCALE_RESIDUAL went
 * from 0.04 to 0.075: at 0.04 the mean separation over the plot was 1.25 user
 * units, under a pixel and a half, and 135 of 181 samples came closer than 0.01. */
ok('and the two curves stay legibly apart on screen at full travel', (function () {
  var span = (EQ.geometry.PLOT.b - EQ.geometry.PLOT.t - EQ.geometry.PLOT_HEAD) / C.vmax,
      i, s = 0, worst = 0, d;
  for (i = 0; i < N; i++) {
    d = Math.abs(c11[i] - teacher[i]) * span;
    s += d;
    if (d > worst) { worst = d; }
  }
  return s / N > 2 && worst > 8;
})(), (function () {
  var span = (EQ.geometry.PLOT.b - EQ.geometry.PLOT.t - EQ.geometry.PLOT_HEAD) / C.vmax,
      i, s = 0, worst = 0, d;
  for (i = 0; i < N; i++) {
    d = Math.abs(c11[i] - teacher[i]) * span;
    s += d;
    if (d > worst) { worst = d; }
  }
  return 'mean ' + (s / N).toFixed(2) + ', max ' + worst.toFixed(2) + ' user units apart';
})());
ok('a residual remains in the SHAPE too, not only in one spike',
   maxAbsDiff(drawnEnv(1), T.lo) > 1e-6,
   'envelope still ' + maxAbsDiff(drawnEnv(1), T.lo).toFixed(4) + ' from the dense shape');
/* The DETAIL residual is the larger of the two, deliberately: it is the one
 * difference visible along the WHOLE curve instead of only at the humps. The
 * retired assertion asked for the texture to land within 15% of the teacher's,
 * which DETAIL_REACH makes false by design, so it is replaced by the exact
 * retention the two constants imply — a scaled copy of the dense scan's own
 * ripple, to machine precision — plus the bound that makes it legible. */
ok('and in the DETAIL: the student keeps exactly DETAIL_REACH\'s share of the texture',
   (function () {
     var d1 = drawnRipple(1), i, worst = 0;
     for (i = 0; i < N; i++) { worst = Math.max(worst, Math.abs(d1[i] - DKEEP * T.hi[i])); }
     return worst < 1e-12;
   })(), 'ripple is exactly ' + DKEEP.toFixed(3) + ' x the dense scan\'s, max |diff| ' +
   sci((function () {
     var d1 = drawnRipple(1), i, worst = 0;
     for (i = 0; i < N; i++) { worst = Math.max(worst, Math.abs(d1[i] - DKEEP * T.hi[i])); }
     return worst;
   })()));
ok('and that share is most of the way there but legibly short of it',
   DKEEP > 0.6 && DKEEP < 0.85 &&
   Math.abs(texture(c11) / texture(teacher) - DKEEP) < 0.02,
   'ripple energy ratio ' + (texture(c11) / texture(teacher)).toFixed(4) +
   ' against DETAIL_REACH ' + DREACH);
note('the PEAK GAP readout therefore reads ' + gap11.toFixed(2) + ' at full travel, never 0.00.');

/* ------------------------------------------------------------------ */
head('FADERS — DEFECT 1: LF moves the SHAPE, HF grows the fine DETAIL');
note('In v2 the HF fader REMOVED a ripple the student should never have had.');
note('Here HF fills in detail the sparse scan never recorded, which is what panel 3');
note('of the paper figure shows, and LF alone leaves the student smooth.');

(function () {
  var steps = [0, 0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1], i, k,
      prevD = Infinity, prevE = -1, monoShape = true, monoDetail = true, dists = [], energies = [];

  /* --- LF closes the SHAPE distance, strictly, at every step --- */
  for (k = 0; k < steps.length; k++) {
    i = maxAbsDiff(drawnEnv(steps[k]), T.lo);
    dists.push(i);
    if (i >= prevD) { monoShape = false; }
    prevD = i;
  }
  ok('raising LF strictly decreases the envelope distance to the dense shape', monoShape,
     dists.map(function (d) { return d.toFixed(3); }).join(' > '));
  /* What is left is not "essentially nothing" any more and must not be: raising
   * FULL_SCALE_RESIDUAL from 0.04 to 0.075 is what makes the two curves
   * distinguishable at full travel, and it buys that by leaving MORE shape error
   * behind. So the threshold is derived from the constant instead of being a
   * hand-typed 0.06 that silently decides the residual: what LF leaves is
   * exactly (1 - REACH) of where it started, which can never exceed the
   * residual itself, and the residual is what sets REACH. A hard-coded number
   * here would have to be edited every time section 1 moves; this cannot drift. */
  ok('LF closes all of that shape error except the deliberate residual',
     Math.abs(dists[dists.length - 1] - (1 - REACH) * dists[0]) < 1e-12 &&
     dists[dists.length - 1] <= RESIDUAL + 1e-12 &&
     dists[dists.length - 1] < 0.12 * dists[0],
     (100 * (1 - dists[dists.length - 1] / dists[0])).toFixed(1) + '% of ' + dists[0].toFixed(3) +
     ' closed, ' + dists[dists.length - 1].toFixed(4) + ' left of a ' + RESIDUAL + ' residual');

  /* --- HF grows the DETAIL energy, strictly, at every step --- */
  for (k = 0; k < steps.length; k++) {
    i = texture(drawnRipple(steps[k]));
    energies.push(i);
    if (i <= prevE) { monoDetail = false; }
    prevE = i;
  }
  ok('raising HF strictly increases the student\'s ripple energy', monoDetail,
     energies.map(function (d) { return d.toFixed(4); }).join(' < '));
  ok('HF multiplies that energy by more than ten', energies[energies.length - 1] > 10 * energies[0],
     (energies[energies.length - 1] / energies[0]).toFixed(1) + 'x');

  /* --- the two faders do not interfere --- */
  ok('the envelope distance does not depend on HF at all',
     maxAbsDiff(sub(morph(0.3, 0), morph(0.3, 1)), sub(drawnRipple(0), drawnRipple(1))) < 1e-12);
  ok('the ripple energy does not depend on LF at all',
     Math.abs(texture(sub(morph(0, 0.6), drawnEnv(0))) - texture(sub(morph(1, 0.6), drawnEnv(1)))) < 1e-12,
     'identical to ' + sci(Math.abs(texture(sub(morph(0, 0.6), drawnEnv(0))) -
                                    texture(sub(morph(1, 0.6), drawnEnv(1))))));

  /* --- lf=1, hf=0: right shape, still smooth (panel 2 of the figure) --- */
  var kS = curvature(student), k10 = curvature(c10), kT = curvature(teacher);
  ok('LF alone leaves the student SMOOTH: it does not grow detail', k10 < 2.2 * kS,
     'curvature ' + k10.toFixed(5) + ' vs ' + kS.toFixed(5) + ' at rest, teacher ' + kT.toFixed(5));
  ok('LF alone puts the SHAPE on the dense scan\'s',
     maxAbsDiff(env(c10), env(teacher)) < 0.25 * maxAbsDiff(env(student), env(teacher)),
     maxAbsDiff(env(c10), env(teacher)).toFixed(4) + ' from blue vs ' +
     maxAbsDiff(env(student), env(teacher)).toFixed(4) + ' at rest');
  ok('LF alone moves the peak onto the teacher\'s', Math.abs(argmax(c10) - argmax(teacher)) <= 3,
     'peak u ' + (argmax(student) / (N - 1)).toFixed(2) + ' -> ' + (argmax(c10) / (N - 1)).toFixed(2) +
     ', teacher at ' + (argmax(teacher) / (N - 1)).toFixed(2));

  /* --- lf=0, hf=1: the texture arrives, the wrong shape stays (why both exist) --- */
  ok('HF alone leaves the SHAPE the sparse scan\'s',
     maxAbsDiff(env(c01), env(student)) * 4 < maxAbsDiff(env(c01), env(teacher)),
     maxAbsDiff(env(c01), env(student)).toFixed(4) + ' from orange vs ' +
     maxAbsDiff(env(c01), env(teacher)).toFixed(4) + ' from blue');
  ok('HF alone DOES grow the fine detail', curvature(c01) > 8 * kS,
     'curvature ' + curvature(c01).toFixed(5) + ', ' + (curvature(c01) / kS).toFixed(1) + 'x the rest pose');
  ok('HF alone leaves the peak where the sparse scan put it',
     Math.abs(argmax(env(c01)) - argmax(env(student))) <= 3);

  /* --- lf=1, hf=1: the texture comes back, short of parity (panel 3) ---
   * "Matches the teacher's" was the claim while both channels travelled REACH.
   * DETAIL_REACH retired it on purpose: the paper's measurements put the
   * distilled student's high-frequency retention well short of the dense scan's
   * even where the coarse shape is in line. So the claim is now two-sided —
   * most of the way back, and still clearly short — which is a narrower band
   * than the one-sided 12% it replaces. */
  ok('LF+HF brings the texture most of the way back, and no further',
     texture(c11) > 0.6 * texture(teacher) && texture(c11) < 0.85 * texture(teacher),
     texture(c11).toFixed(5) + ' vs ' + texture(teacher).toFixed(5) + ', ratio ' +
     (texture(c11) / texture(teacher)).toFixed(3));
})();

/* ------------------------------------------------------------------ */
head('MORPH — the four corners (in the raw, residual-free family)');

ok('raw lf=1, hf=1 reproduces the teacher to 1e-9', maxAbsDiff(raw(1, 1), teacher) < 1e-9,
   'max ' + sci(maxAbsDiff(raw(1, 1), teacher)));
ok('raw lf=1, hf=0 is exactly teacherEnvelope + studentRipple',
   maxAbsDiff(raw(1, 0), add(T.lo, S.hi)) < 1e-12, 'max ' + sci(maxAbsDiff(raw(1, 0), add(T.lo, S.hi))));
ok('raw lf=0, hf=1 is exactly studentEnvelope + teacherRipple',
   maxAbsDiff(raw(0, 1), add(S.lo, T.hi)) < 1e-12, 'max ' + sci(maxAbsDiff(raw(0, 1), add(S.lo, T.hi))));
/* two travels, not one: the SHAPE channel stops at REACH, the DETAIL channel at
 * the shorter DETAIL_REACH. Asserting c01 against raw(0, REACH) was asserting a
 * curve the widget stopped drawing. */
ok('the drawn family is the raw family with each channel\'s travel shortened',
   maxAbsDiff(c10, raw(REACH, 0)) < 1e-12 && maxAbsDiff(c01, raw(0, DREACH)) < 1e-12 &&
   maxAbsDiff(c11, raw(REACH, DREACH)) < 1e-12,
   'lf over ' + REACH.toFixed(4) + ', hf over ' + DREACH);

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
  /* The student family is not the only thing drawn: the TEACHER is on the same
   * axes, and because the full-travel student sits the residual below it at the
   * dense peak, max(teacher) is the taller of the two by exactly that residual.
   * Scaling to the corners alone put the blue curve's peak at y = 21.97 against
   * PLOT.t = 30 — through the top frame line, with only Instance.y()'s clamp
   * (PLOT.t - 4) stopping it from leaving the box altogether, which is why the
   * old assertion could pass while the picture was wrong. So: vmax covers the
   * teacher too, and the teacher's peak is checked UNCLAMPED against the
   * headroom the geometry reserves for it. */
  var tHi = Math.max.apply(null, teacher),
      bound = Math.max(hi, tHi),
      sp = (EQ.geometry.PLOT.b - EQ.geometry.PLOT.t - EQ.geometry.PLOT_HEAD) / C.vmax,
      tTop = EQ.geometry.PLOT.b - tHi * sp;
  ok('the exported vmax is exactly that upper bound, the teacher included',
     Math.abs(C.vmax - bound) < 1e-12 && C.vmax >= tHi,
     'vmax ' + C.vmax.toFixed(4) + ' = max(corners ' + hi.toFixed(4) +
     ', teacher ' + tHi.toFixed(4) + ')');
  ok('so the dense curve\'s peak lands inside the reserved headroom, not on the frame',
     tTop >= EQ.geometry.PLOT.t + EQ.geometry.PLOT_HEAD - 1e-9,
     'peak renders at y ' + tTop.toFixed(2) + ', top frame at ' + EQ.geometry.PLOT.t +
     ', headroom floor at ' + (EQ.geometry.PLOT.t + EQ.geometry.PLOT_HEAD) +
     ' (corners-only scaling put it at ' +
     (EQ.geometry.PLOT.b - tHi * ((EQ.geometry.PLOT.b - EQ.geometry.PLOT.t - EQ.geometry.PLOT_HEAD) / hi)).toFixed(2) + ')');
  ok('no pose ever dips below the axis, so the filled areas stay sane', lo >= -1e-9,
     'minimum ' + lo.toFixed(6));
  note('the vertical scale is derived from exactly this: span = ' +
       (EQ.geometry.PLOT.b - EQ.geometry.PLOT.t - EQ.geometry.PLOT_HEAD) + ' / vmax = ' +
       ((EQ.geometry.PLOT.b - EQ.geometry.PLOT.t - EQ.geometry.PLOT_HEAD) / C.vmax).toFixed(2) + '.');
})();

/* ------------------------------------------------------------------ */
head('PLUCK — the deformation kernel');

var SPAN = Math.max(2, EQ.constants.PLUCK_SPAN_FRAC * (N - 1));

(function () {
  var i, prev, monoDown = true;
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
})();

/* ------------------------------------------------------------------ */
head('PLUCK — drag deforms, release springs back exactly');

(function () {
  var base = morph(0.4, 0.6);
  var pluck = new EQ.PluckState({ span: SPAN });
  var i0 = 90, AMP = 0.3, now = 1000;
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
  var step1 = Math.abs((held[i0] - base[i0]) - (held[i0 - 1] - base[i0 - 1]));
  ok('the deformation is smooth, not a spike', step1 < AMP * 0.03,
     'the bump rises ' + step1.toFixed(5) + ' per sample at its apex, ' +
     (100 * step1 / AMP).toFixed(1) + '% of a ' + AMP + ' pull');

  ok('plucking the student leaves the teacher alone',
     maxAbsDiff(pluck.apply(teacher.slice(0), 't', pluck.amp(now)), teacher) === 0);

  /* ---- release ---- */
  pluck.release(now);
  var t, a, settleAt = -1, overshoot = 0, crossings = 0, prevSign = 1;
  for (t = 0; t <= 1200; t += 4) {
    a = EQ.math.spring(AMP, t / 1000, EQ.constants.PLUCK_ZETA, EQ.constants.PLUCK_FREQ);
    if (settleAt >= 0) { continue; }
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
     maxAbsDiff(pluck.apply(base.slice(0), 's', pluck.amp(now + 9999)), base) === 0);
  note('spring: zeta ' + EQ.constants.PLUCK_ZETA + ', ' + EQ.constants.PLUCK_FREQ +
       ' Hz, settle threshold ' + EQ.constants.PLUCK_SETTLE_EPS + ' signal units.');
})();

/* ------------------------------------------------------------------ */
head('PLUCK — a curve drag cannot move a fader');

(function () {
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
  var i0 = 112;

  EQ.Instance.prototype.pluckBegin.call(fake, 's', i0, base[i0]);
  EQ.Instance.prototype.pluckTo.call(fake, base[i0] - 0.22);
  EQ.Instance.prototype.pluckTo.call(fake, base[i0] + 0.31);

  var during = fake.pluck.apply(base.slice(0), 's', fake.pluck.amp(0));
  ok('the drag really did deform the curve', maxAbsDiff(during, base) > 0.3,
     'max deviation ' + maxAbsDiff(during, base).toFixed(4));
  ok('a pluck sets the "someone has plucked" flag, which retires the hint', fake.plucked === true);
  ok('lf is untouched by a curve drag', fake.state.lf === 0.4, 'lf = ' + fake.state.lf);
  ok('hf is untouched by a curve drag', fake.state.hf === 0.6, 'hf = ' + fake.state.hf);

  EQ.Instance.prototype.pluckRelease.call(fake, 0);
  ok('releasing a curve starts the spring, it does not snap', fake.sprung === 1);
  ok('lf and hf are still untouched after release',
     fake.state.lf === 0.4 && fake.state.hf === 0.6);

  var after = fake.pluck.apply(base.slice(0), 's', fake.pluck.amp(2000));
  ok('and the settled shape is exactly what lf=0.4, hf=0.6 imply',
     maxAbsDiff(after, morph(0.4, 0.6)) === 0);

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
head('HIT TEST — DEFECT 3: one surface, nearest curve wins');
note('v2 stacked two invisible 22-unit strokes and painted the student\'s last, so where');
note('the curves ran within ~11 units the student swallowed the press: measured with real');
note('pointer events, 3 of 9 drags along the teacher moved it. These probes are that test.');

var PL = EQ.geometry.PLOT, SPANY = (PL.b - PL.t - EQ.geometry.PLOT_HEAD) / C.vmax;
function yOf(v) { return Math.min(Math.max(PL.b - v * SPANY, PL.t - 4), PL.b); }
function screenYs(vals) { var o = [], i; for (i = 0; i < vals.length; i++) { o.push(yOf(vals[i])); } return o; }
function pick(i, py, tys, sys, r) {
  return EQ.math.pickCurve(i, py, tys, sys, r === undefined ? EQ.constants.GRAB_RADIUS : r,
                           EQ.constants.GRAB_WINDOW);
}

(function () {
  var tys = screenYs(teacher), sys = screenYs(student), i, k, p, hitT = 0, hitS = 0,
      close = [], probes = 9, R = EQ.constants.GRAB_RADIUS;

  /* the v2 measurement, repeated: nine evenly spaced grabs along each curve */
  for (k = 0; k < probes; k++) {
    i = Math.round((k + 0.5) * (N - 1) / probes);
    p = pick(i, tys[i], tys, sys);
    if (p && p.which === 't') { hitT++; }
    p = pick(i, sys[i], tys, sys);
    if (p && p.which === 's') { hitS++; }
  }
  ok('9 of 9 grabs ON the teacher pick the teacher', hitT === probes, hitT + '/' + probes +
     ' (v2 measured 3/9 with real pointer events)');
  ok('9 of 9 grabs ON the student pick the student', hitS === probes, hitS + '/' + probes);

  /* the exact condition v2 failed under: the two curves within half a grab band */
  for (i = 0; i < N; i++) { if (Math.abs(tys[i] - sys[i]) < 11) { close.push(i); } }
  ok('there really are stretches where the two curves run within 11 units', close.length > 20,
     close.length + ' of ' + N + ' samples');
  hitT = 0;
  for (k = 0; k < close.length; k++) {
    i = close[k];
    p = pick(i, tys[i], tys, sys);
    if (p && p.which === 't') { hitT++; }
  }
  ok('the teacher is grabbable at EVERY one of them', hitT === close.length,
     hitT + '/' + close.length + ' — this is the defect, fixed');

  /* and at full travel, where the curves are only the residual apart */
  var c = c11, cys = screenYs(c), peak = argmax(teacher), gapAtPeak = Math.abs(cys[peak] - tys[peak]);
  ok('at lf=hf=1 the two curves are still a few units apart on screen', gapAtPeak > 3,
     gapAtPeak.toFixed(1) + ' units at the dense peak');
  ok('and each is still grabbable there',
     (function () {
       var a = pick(peak, tys[peak], tys, cys), b = pick(peak, cys[peak], tys, cys);
       return a && a.which === 't' && b && b.which === 's';
     })());

  /* the invariant, over a grid of probes: the curve returned is the nearer one */
  var bad = 0, nulls = 0, tested = 0, j;
  for (i = 0; i < N; i += 3) {
    for (j = 0; j <= 20; j++) {
      var py = PL.t + (PL.b - PL.t) * j / 20;
      p = pick(i, py, tys, sys);
      tested++;
      if (!p) {
        nulls++;
        if (Math.abs(py - tys[i]) <= R || Math.abs(py - sys[i]) <= R) { bad++; }
      } else if (p.dist > p.other + 1e-12) { bad++; }
    }
  }
  ok('over ' + tested + ' probes the pick is always the nearer curve, or null when both are far',
     bad === 0, nulls + ' of ' + tested + ' probes were out of reach');

  /* ties, radius, and degenerate input */
  ok('a tie goes to the teacher, the curve that used to be unreachable',
     pick(40, 100, [100, 100, 100], [100, 100, 100], 20).which === 't');
  ok('a press far from both curves is not a pluck at all',
     pick(Math.round(N / 2), PL.t, tys, sys) === null ||
     pick(Math.round(N / 2), PL.t, tys, sys).dist <= R);
  ok('a press 20 units off a curve misses with a mouse but lands with a finger',
     pick(90, tys[90] - 20, tys, sys, EQ.constants.GRAB_RADIUS) === null &&
     pick(90, tys[90] - 20, tys, sys, EQ.constants.GRAB_RADIUS_TOUCH) !== null,
     'mouse radius ' + EQ.constants.GRAB_RADIUS + ', touch radius ' + EQ.constants.GRAB_RADIUS_TOUCH);
  ok('the window search forgives a steep flank',
     (function () {
       var steep = [200, 150, 100, 50, 0], flat = [10, 10, 10, 10, 10], hit, miss;
       /* y=138 is 38 units off sample 2 itself, but only 12 off its neighbour:
        * on a flank this steep, a one-sample test would drop the grab */
       hit = EQ.math.pickCurve(2, 138, steep, flat, 14, EQ.constants.GRAB_WINDOW);
       miss = EQ.math.pickCurve(2, 138, steep, flat, 14, 0);
       return hit && hit.which === 't' && miss === null;
     })(), 'searches +-' + EQ.constants.GRAB_WINDOW + ' samples');
  ok('pickCurve tolerates junk instead of throwing',
     pick(0, NaN, tys, sys) === null && EQ.math.pickCurve(0, 100, [], [], 14, 2) === null &&
     EQ.math.pickCurve(999, tys[0], tys, sys, 1e6, 0) !== null);
  ok('the index is clamped into the grid, never read off the end',
     EQ.math.pickCurve(-50, tys[0], tys, sys, 1e6, 0).i === 0 &&
     EQ.math.pickCurve(1e6, tys[N - 1], tys, sys, 1e6, 0).i === N - 1);
  note('one hit surface spans the whole plot: ' + (PL.r - PL.l + 4) + ' x ' +
       (PL.b - PL.t + 12) + ' user units, with pointer-events:all and touch-action:none.');
})();

/* ------------------------------------------------------------------ */
head('PATH — the two drawn curves are polylines, the reference is a spline');

(function () {
  var G = EQ.geometry, xs = [], ys = [], i, d, segs;
  for (i = 0; i < N; i++) {
    xs.push(PL.l + (PL.r - PL.l) * i / (N - 1));
    ys.push(yOf(student[i]));
  }
  d = EQ.math.linePath(xs, ys);
  segs = d.split('L').length - 1;
  ok('linePath emits one straight segment per interval', segs === N - 1,
     segs + ' segments for ' + N + ' points');
  ok('it starts at the first sample and ends at the last',
     d.indexOf('M' + PL.l + ' ' + PL.b) === 0 && new RegExp('L' + PL.r + ' ' + PL.b + '$').test(d));
  ok('only M and L commands, all coordinates finite', /^M[-\d. ]+(L[-\d. ]+)+$/.test(d));
  ok('coordinates are quantised to 0.1', !/\.\d\d/.test(d));
  ok('a polyline keeps a spike SHARP: no spline overshoot to round the tip',
     (function () {
       var k = argmax(d2(teacher)), tipLine, tipSpline, tys2 = [], j;
       for (j = 0; j < N; j++) { tys2.push(yOf(teacher[j])); }
       tipLine = EQ.math.linePath(xs, tys2);
       tipSpline = EQ.math.catmullPath(xs, tys2);
       return tipSpline.length > tipLine.length && k > 0;
     })(), 'the spline is ' +
     (EQ.math.catmullPath(xs, ys).length / EQ.math.linePath(xs, ys).length).toFixed(1) +
     'x the string length too');
  ok('degenerate input does not throw',
     EQ.math.linePath([], []) === '' && EQ.math.linePath([1], [1]) === '' &&
     EQ.math.catmullPath([], []) === '' && EQ.math.catmullPath([1], [1]) === '');

  var cm = EQ.math.catmullPath(xs, EQ.curves.teacherEnvelope.map(yOf));
  ok('the dashed reference envelope is a Catmull-Rom spline, one cubic per interval',
     cm.split('C').length - 1 === N - 1 && /^M[-\d. ]+(C[-\d. ]+)+$/.test(cm));

  var area = EQ.math.areaPath(d, xs[0], xs[N - 1], PL.b);
  ok('areaPath closes the curve down to the baseline and back',
     area.indexOf(d) === 0 && /Z$/.test(area) &&
     area.indexOf('L' + PL.r + ' ' + PL.b + 'L' + PL.l + ' ' + PL.b + 'Z') > 0);
  ok('areaPath on an empty curve is empty, not a stray Z', EQ.math.areaPath('', 0, 1, 2) === '');
  ok('bracketPath is a bracket: two legs down and a bar across',
     EQ.math.bracketPath(10, 50, 100, 5) === 'M10 105L10 100L50 100L50 105');
  note('path length at rest: ' + d.length + ' chars (a spline of the same curve would be ' +
       EQ.math.catmullPath(xs, ys).length + ').');
})();

/* ------------------------------------------------------------------ */
head('GEOMETRY — footprint');

(function () {
  var G = EQ.geometry, aspect = G.VB_W / G.VB_H;
  var col = (1180 - 80 - 52) * 1.22 / 2, inner = col - 50 - 2;
  ok('the card is wider than it is tall by more than the old 1.679',
     aspect > 1.9, 'viewBox ' + G.VB_W + ' x ' + G.VB_H + ', aspect ' + aspect.toFixed(3));
  note('at the hero width the SVG paints ' + inner.toFixed(0) + ' x ' +
       (inner / aspect).toFixed(0) + ' CSS px.');
  ok('everything computed stays inside the 8-unit inset',
     G.PLOT.l >= G.INSET && G.PLOT.r <= G.VB_W - G.INSET &&
     G.PLOT.t >= G.INSET && G.PLOT.b <= G.VB_H - G.INSET &&
     G.RACK.x + G.RACK.w <= G.VB_W - G.INSET && G.RACK.y >= G.INSET &&
     G.RACK.y + G.RACK.h <= G.VB_H - G.INSET);
  ok('the bottom row of type clears the inset', G.ROW.chip + G.CHIP.h / 2 <= G.VB_H - G.INSET,
     'lowest ink at ' + (G.ROW.chip + G.CHIP.h / 2) + ' of ' + (G.VB_H - G.INSET));
  ok('the hint pill sits inside the plot box',
     G.HINT.x <= G.PLOT.r && G.HINT.x - G.hintWidth(EQ.defaults.hint) > G.PLOT.l &&
     G.HINT.y > G.PLOT.t && G.HINT.y + G.HINT.h < G.PLOT.b,
     'pill spans x ' + (G.HINT.x - G.hintWidth(EQ.defaults.hint)).toFixed(0) + '..' + G.HINT.x +
     ', y ' + G.HINT.y + '..' + (G.HINT.y + G.HINT.h));
})();

/* ------------------------------------------------------------------ */
head('GEOMETRY — fader travel and value round trip');

(function () {
  var TR = EQ.geometry.TRACK, PLG = EQ.geometry.PLOT, i, v, y, back, worst = 0, RK = EQ.geometry.RACK;
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
  ok('plot and rack do not overlap', PLG.r < RK.x, (RK.x - PLG.r) + ' units of gutter');
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

/* The topmost ink either curve puts between two x positions, over every fader
 * pose — so a label can be proved to sit in empty space rather than on top of
 * the drawing. Smaller y is higher on screen. */
function inkTop(xa, xb) {
  var G = EQ.geometry, poses = [[0, 0], [1, 0], [0, 1], [1, 1]], top = Infinity, k, i, v, x;
  for (k = 0; k < poses.length; k++) {
    v = morph(poses[k][0], poses[k][1]);
    for (i = 0; i < N; i++) {
      x = G.PLOT.l + (G.PLOT.r - G.PLOT.l) * i / (N - 1);
      if (x < xa || x > xb) { continue; }
      top = Math.min(top, yOf(v[i]), yOf(teacher[i]));
    }
  }
  return top;
}

function labelGeometry(narrow) {
  var tag = narrow ? ' [narrow]' : ' [base]';
  function w(text, cls) {
    var s = styleOf(cls, narrow);
    if (!s.px) { throw new Error('no font-size for .' + cls); }
    return EQ.geometry.advance(text, s.px, s.ls);
  }
  function px(cls) { return styleOf(cls, narrow).px; }
  var D = EQ.defaults, G = EQ.geometry, PLG = G.PLOT, RK = G.RACK, AN = G.ANN, LG = G.LEGEND;

  ok('every measured class has a size' + tag, (function () {
    var want = ['lieq-leg', 'lieq-ann-t', 'lieq-cap-sub', 'lieq-axis-t', 'lieq-ro-k',
                'lieq-tick-t', 'lieq-rack-eyebrow', 'lieq-val', 'lieq-cap-label',
                'lieq-chip-t', 'lieq-hint'], i;
    for (i = 0; i < want.length; i++) { if (!styleOf(want[i], narrow).px) { return false; } }
    return true;
  })());

  var legendEnd = LG.t2 + w(D.studentLabel, 'lieq-leg');
  ok('legend fits inside the plot' + tag, legendEnd < PLG.r,
     'ends at ' + legendEnd.toFixed(1) + ' of ' + PLG.r);
  ok('legend entries do not collide' + tag, LG.t1 + w(D.teacherLabel, 'lieq-leg') < LG.sw2,
     'first ends at ' + (LG.t1 + w(D.teacherLabel, 'lieq-leg')).toFixed(1) +
     ', second swatch at ' + LG.sw2);
  ok('the legend sits above the plot, clear of every label inside it' + tag,
     LG.baseline < PLG.t, 'baseline ' + LG.baseline + ', plot starts at ' + PLG.t);

  /* ---- the pluck hint pill ---- */
  var hintW = G.hintWidth(D.hint), hintX = G.HINT.x - hintW;
  ok('the hint text fits its pill' + tag, w(D.hint, 'lieq-hint') + G.HINT.glyph + G.HINT.gap <
     hintW - G.HINT.pad, 'text ' + w(D.hint, 'lieq-hint').toFixed(1) + ' in a ' + hintW + ' pill');
  ok('the hint pill sits in empty plot space' + tag,
     G.HINT.y + G.HINT.h < inkTop(hintX, G.HINT.x) - 2,
     'pill bottom ' + (G.HINT.y + G.HINT.h) + ', topmost ink there ' +
     inkTop(hintX, G.HINT.x).toFixed(1));

  /* ---- callout 1: SHAPE, in the plot's upper-left bay ---- */
  var sw1 = Math.max(w(D.shapeAnnotation[0], 'lieq-ann-t'), w(D.shapeAnnotation[1], 'lieq-ann-t'),
                     w(D.shapeAnnotationDone[0], 'lieq-ann-t'), w(D.shapeAnnotationDone[1], 'lieq-ann-t'));
  var shapeRight = AN.shape.textX + sw1;
  var barX = PLG.l + (PLG.r - PLG.l) * C.shapeDepartureIndex / (N - 1);
  ok('the SHAPE callout fits inside the plot' + tag, shapeRight < PLG.r,
     'spans ' + AN.shape.textX + '..' + shapeRight.toFixed(1) + ' of ' + PLG.l + '..' + PLG.r);
  ok('the SHAPE callout sits in empty plot space, in BOTH wordings' + tag,
     AN.shape.y2 + 0.1 * px('lieq-ann-t') < inkTop(AN.shape.textX, shapeRight) - 2,
     'lowest text ink ' + AN.shape.y2.toFixed(1) + ', topmost curve ink ' +
     inkTop(AN.shape.textX, shapeRight).toFixed(1));
  ok('its leader starts below the text and above the curves' + tag,
     AN.shape.leadTop > AN.shape.y2 && AN.shape.leadTop < inkTop(barX - 2, barX + 2),
     'leader from y ' + AN.shape.leadTop + ' down to the ' +
     inkTop(barX - 2, barX + 2).toFixed(0) + ' ink at x ' + barX.toFixed(0));
  ok('the dimension bar is under the text block, so the leader is vertical' + tag,
     barX > AN.shape.textX && barX < shapeRight, 'bar at x ' + barX.toFixed(1));

  /* ---- callout 2: DETAIL, in the plot's upper-right bay ---- */
  var dw1 = Math.max(w(D.detailAnnotation[0], 'lieq-ann-t'), w(D.detailAnnotation[1], 'lieq-ann-t'),
                     w(D.detailAnnotationDone[0], 'lieq-ann-t'), w(D.detailAnnotationDone[1], 'lieq-ann-t'));
  var detailLeft = AN.detail.textX - dw1;
  ok('the DETAIL callout fits inside the plot' + tag, detailLeft > PLG.l,
     'spans ' + detailLeft.toFixed(1) + '..' + AN.detail.textX);
  ok('the DETAIL callout sits in empty plot space, in BOTH wordings' + tag,
     AN.detail.y2 + 0.1 * px('lieq-ann-t') < inkTop(detailLeft, AN.detail.textX) - 2,
     'lowest text ink ' + AN.detail.y2.toFixed(1) + ', topmost curve ink ' +
     inkTop(detailLeft, AN.detail.textX).toFixed(1));
  ok('the DETAIL callout clears the hint pill above it' + tag,
     AN.detail.y1 - px('lieq-ann-t') > G.HINT.y + G.HINT.h,
     'text ink starts at ' + (AN.detail.y1 - px('lieq-ann-t')).toFixed(1) +
     ', pill ends at ' + (G.HINT.y + G.HINT.h));
  ok('its bracket is clamped below the text, whatever the faders do' + tag,
     AN.detail.minY > AN.detail.y2 + 6 && AN.detail.leadTop > AN.detail.y2,
     'bracket floor y ' + AN.detail.minY);
  ok('the bracketed stretch is under the DETAIL text' + tag, (function () {
    var a = PLG.l + (PLG.r - PLG.l) * AN.detailFrom, b = PLG.l + (PLG.r - PLG.l) * AN.detailTo;
    return (a + b) / 2 > detailLeft && (a + b) / 2 < AN.detail.textX;
  })());
  ok('the two callouts do not collide with each other' + tag, shapeRight < detailLeft - 4,
     'SHAPE ends at ' + shapeRight.toFixed(1) + ', DETAIL starts at ' + detailLeft.toFixed(1));

  /* ---- rack and bottom row ---- */
  var capw = Math.max(w(D.lfSub, 'lieq-cap-sub'), w(D.hfSub, 'lieq-cap-sub'));
  ok('fader captions do not collide' + tag,
     G.FADER_X.lf + capw / 2 + 5 < G.FADER_X.hf - capw / 2,
     'gap ' + ((G.FADER_X.hf - capw / 2) - (G.FADER_X.lf + capw / 2)).toFixed(1) + ' units');
  ok('fader captions stay inside the rack' + tag,
     G.FADER_X.lf - capw / 2 > RK.x && G.FADER_X.hf + capw / 2 < RK.x + RK.w,
     (G.FADER_X.lf - capw / 2).toFixed(1) + '..' + (G.FADER_X.hf + capw / 2).toFixed(1) +
     ' inside ' + RK.x + '..' + (RK.x + RK.w));

  var mid = (PLG.l + PLG.r) / 2;
  var xt = w(D.xLabel, 'lieq-axis-t');
  ok('x-axis title is centred and inside the plot' + tag,
     mid - xt / 2 > PLG.l && mid + xt / 2 < PLG.r,
     'title spans ' + (mid - xt / 2).toFixed(1) + '..' + (mid + xt / 2).toFixed(1));
  var yt = w(D.yLabel, 'lieq-axis-t');
  ok('rotated y-axis title fits the plot height' + tag, yt < PLG.b - PLG.t,
     yt.toFixed(1) + ' of ' + (PLG.b - PLG.t));
  ok('rack eyebrow fits the rack' + tag, w(D.rackLabel, 'lieq-rack-eyebrow') < RK.w - 8,
     w(D.rackLabel, 'lieq-rack-eyebrow').toFixed(1) + ' of ' + (RK.w - 8));
  ok('value readout fits between the faders' + tag,
     w('0.00', 'lieq-val') < G.FADER_X.hf - G.FADER_X.lf);

  var chipEnd = PLG.l, i2;
  for (i2 = 0; i2 < D.presets.length; i2++) {
    chipEnd += G.chipWidth(D.presets[i2].label) + (i2 < D.presets.length - 1 ? G.CHIP.gap : 0);
  }
  var roW = Math.max(w(D.readoutLabel + ' 0.00', 'lieq-ro-k'), w(D.scrubLabel + ' 0.00', 'lieq-ro-k'));
  ok('the preset pills and the readout do not collide' + tag, chipEnd < PLG.r - roW - 6,
     'pills end at ' + chipEnd.toFixed(1) + ', readout starts at ' + (PLG.r - roW).toFixed(1));
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
  ok('the plot hit surface does not reach the rack or the preset row',
     G.PLOT.r + 2 < RK.x && G.PLOT.b + 12 < G.ROW.chip - G.CHIP.hit / 2,
     'surface ends at x ' + (G.PLOT.r + 2) + ', y ' + (G.PLOT.b + 12));
  note('preset pills are ' + G.CHIP.hit + ' user units tall = ' +
       (G.CHIP.hit * 320 / G.VB_W).toFixed(0) + ' CSS px on a 390px phone.');
  note('the grab radius is ' + EQ.constants.GRAB_RADIUS + ' user units = ' +
       (EQ.constants.GRAB_RADIUS * 320 / G.VB_W).toFixed(0) + ' CSS px on a phone, ' +
       (EQ.constants.GRAB_RADIUS * 587 / G.VB_W).toFixed(0) + ' on the desktop hero; a touch gets ' +
       (EQ.constants.GRAB_RADIUS_TOUCH * 320 / G.VB_W).toFixed(0) + ' px.');
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

  var hooks = ['tArea', 'tCore', 'ghost', 'sArea', 'sCore', 'armDot', 'hit', 'hintg',
               'annShape', 'shapeT1', 'shapeT2', 'shapeLead', 'gapStem', 'gapTop', 'gapBot',
               'annDetail', 'detailT1', 'detailT2', 'detailLead', 'detailBr',
               'gapValue', 'roLabel', 'scrub', 'scrubLine', 'scrubDotS', 'scrubDotT'];
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
  ok('the fader sub-captions name what each one does',
     src.indexOf('>SHAPE<') > 0 && src.indexOf('>DETAIL<') > 0);
  ok('every preset is labelled and carries aria-pressed',
     (src.match(/aria-pressed="/g) || []).length === 3 &&
     src.indexOf('aria-label="Preset: bypass') > 0);
  ok('every control carries a focus ring element', (src.match(/lieq-focusring/g) || []).length === 5);
  ok('svg is a group with a label and a long description',
     /role="group"/.test(src) && /aria-label="/.test(src) && /<desc id="lieqDesc/.test(src));
  ok('no role="img" (which would hide the sliders)', src.indexOf('role="img"') < 0);

  /* ---- DEFECT 3: one hit surface, and the v2 bands are gone ---- */
  ok('there is exactly ONE hit surface, over the plot',
     (src.match(/lieq-hitfield/g) || []).length === 1, 'was two stacked 22-unit strokes in v2');
  ok('the v2 grab bands are gone from the markup and the stylesheet',
     src.indexOf('lieq-grab') < 0 && EQ.css.indexOf('lieq-grab{') < 0 &&
     EQ.css.indexOf('stroke-width:22') < 0);
  ok('the hit surface declares pointer-events and touch-action in the sheet',
     /\.lieq-hitfield\{[^}]*pointer-events:all/.test(EQ.css) &&
     /\.lieq-hitfield\{[^}]*touch-action:none/.test(EQ.css));
  ok('hovering shows a grab cursor and a hover highlight, so the affordance is honest',
     EQ.css.indexOf('[data-lieq-arm] .lieq-hitfield{cursor:grab}') > 0 &&
     EQ.css.indexOf('[data-lieq-hot="t"] .lieq-t-core') > 0 &&
     EQ.css.indexOf('[data-lieq-hot="s"] .lieq-s-core') > 0);
  ok('and a grabbing cursor while a pluck is live',
     EQ.css.indexOf('[data-lieq-pluck="1"] .lieq-hitfield{cursor:grabbing}') > 0);
  ok('the pluck hint is a pill with a glyph, not two words in a corner',
     src.indexOf('lieq-hintbox') > 0 && src.indexOf('lieq-hintglyph') > 0 &&
     src.indexOf('>' + EQ.defaults.hint + '<') > 0, 'hint: "' + EQ.defaults.hint + '"');
  ok('reduced motion stops the hint nudging', /prefers-reduced-motion[^}]*}[\s\S]*lieq-hintg\{animation:none/.test(EQ.css) ||
     EQ.css.indexOf('.lieq-hintg{animation:none;transition:none}') > 0);

  /* ---- DEFECT 2: the frequency-domain callouts are gone ---- */
  ok('the axes are labelled in the SPATIAL domain, as the source figure is',
     src.indexOf('Spatial position') > 0 && src.indexOf('Signal / feature response') > 0);
  ok('the v2 callout "LOW-FREQUENCY EXCESS" is gone outright',
     src.indexOf('LOW-FREQUENCY') < 0 && src.indexOf('EXCESS') < 0 &&
     EQ.defaults.lfAnnotation === undefined);
  ok('the v2 callout "SPURIOUS HIGH-FREQUENCY RESPONSE" is gone outright',
     src.indexOf('SPURIOUS') < 0 && src.indexOf('HIGH-FREQUENCY') < 0 &&
     EQ.defaults.hfAnnotation === undefined);
  ok('the meaningless "X MAX" tick is gone, and no tick pretends to be a coordinate',
     src.indexOf('X MAX') < 0 && EQ.defaults.lowTick === '' && EQ.defaults.highTick === '' &&
     src.indexOf('lieq-tick-t') < 0, 'the x axis carries tick MARKS and a title, no numbers');
  ok('no REGION of either curve is labelled with a frequency, a spectrum or a band',
     (function () {
       var D = EQ.defaults, bad = /FREQ|SPECTR|BAND|HERTZ|HZ\b/i, lists = [
         D.shapeAnnotation, D.shapeAnnotationDone, D.detailAnnotation, D.detailAnnotationDone,
         [D.xLabel, D.yLabel], [D.lowTick, D.highTick], [D.hint], [D.readoutLabel, D.scrubLabel],
         [D.teacherLabel, D.studentLabel]
       ], i2, j;
       for (i2 = 0; i2 < lists.length; i2++) {
         for (j = 0; j < lists[i2].length; j++) { if (bad.test(String(lists[i2][j]))) { return false; } }
       }
       return true;
     })(), 'checked both callouts in both wordings, the axis titles, the ticks, the legend and the readout');
  ok('the LF / HF fader labels themselves are still there, as they should be',
     src.indexOf('>LF<') > 0 && src.indexOf('>HF<') > 0 &&
     src.indexOf('aria-label="LF equalizer strength, the overall shape') > 0 &&
     src.indexOf('aria-label="HF equalizer strength, the fine detail') > 0);
  ok('the two callouts say something true about a SPATIAL curve',
     src.indexOf('SHAPE THE SPARSE') > 0 && src.indexOf('SCAN GOT WRONG') > 0 &&
     src.indexOf('DETAIL THE SPARSE SCAN') > 0 && src.indexOf('NEVER RECORDED') > 0);
  ok('each callout has a second wording for when its sentence stops being true',
     EQ.defaults.shapeAnnotationDone[0].length > 0 && EQ.defaults.detailAnnotationDone[0].length > 0 &&
     EQ.constants.ANN_DONE_AT > 0 && EQ.constants.ANN_DONE_AT < 1,
     'they swap at ' + EQ.constants.ANN_DONE_AT + ' travel');

  /* ---- the description ---- */
  /* the number is read off the constant, because the description went on saying
   * "four per cent" after FULL_SCALE_RESIDUAL became 0.075 — a screen reader
   * being given a figure the page had stopped drawing */
  ok('the description states that full travel does NOT reach the teacher',
     /does not reach it/.test(src) &&
     src.indexOf('residual of ' + Math.round(100 * RESIDUAL) + ' per cent') > 0 &&
     /detail comes back shallower/.test(src));
  ok('the description states that a pluck cannot move a fader',
     /never changes the faders/.test(src));
  ok('the description says the nearer curve is the one that moves',
     /nearer curve to the pointer is the one that moves/.test(src));
  ok('the description says height is a signal value, not a frequency',
     /Height on this plot is a signal value, not a frequency/.test(src));
  ok('the description says which curve carries the fine detail, and which lost it',
     /dominant hump carrying dense fine detail/.test(src) &&
     /never recorded the fine/.test(src));
  ok('the illustrative-not-measured disclosure reaches the accessible description',
     src.indexOf('Illustrative, not measured') > 0);

  var a = EQ.renderMarkup(null, '1'), b = EQ.renderMarkup(null, '2');
  ok('def ids are namespaced per instance',
     a.indexOf('lieqGlow1') > 0 && b.indexOf('lieqGlow2') > 0 && b.indexOf('lieqGlow1') < 0);
  ok('both area fills are per-instance gradients',
     a.indexOf('lieqTArea1') > 0 && a.indexOf('lieqSArea1') > 0);

  var evil = EQ.renderMarkup({ teacherLabel: '<script>x</script>' });
  ok('option strings are escaped', evil.indexOf('<script>') < 0 && evil.indexOf('&lt;script&gt;') > 0);
  var evil2 = EQ.renderMarkup({ hint: '<img src=x onerror=1>', shapeAnnotation: ['<b>', '<i>'] });
  ok('the hint and the callouts are escaped too',
     evil2.indexOf('<img') < 0 && evil2.indexOf('<b>') < 0 && evil2.indexOf('&lt;b&gt;') > 0);
  ok('a plain string is accepted where a two-line callout is expected',
     EQ.renderMarkup({ shapeAnnotation: 'ONE LINE' }).indexOf('ONE LINE') > 0);
})();

/* ------------------------------------------------------------------ */
head('CAPTION — the illustrative disclosure');

ok('the module exposes a one-line caption', typeof EQ.caption === 'string' && EQ.caption.length > 40);
ok('it is one line', EQ.caption.indexOf('\n') < 0);
ok('it says the curves are not measured', /[Ii]llustrative, not measured/.test(EQ.caption));
ok('it names what is illustrative: both profiles are generated',
   /drawn procedurally/.test(EQ.caption) && /not plotted from logged features/.test(EQ.caption));
ok('it no longer claims the STUDENT\'s ripple was synthesised, which was v2\'s inverted story',
   EQ.caption.indexOf('student’s high-frequency ripple is synthesised') < 0 &&
   EQ.caption.indexOf("student's high-frequency ripple is synthesised") < 0);
ok('it describes the dense scan as the one carrying the detail',
   /dominant shape carrying fine detail for the dense scan/.test(EQ.caption));
ok('the default option carries the same string', EQ.defaults.caption === EQ.caption);
note('caption: ' + EQ.caption);

/* ------------------------------------------------------------------ */
head('ROBUSTNESS');

ok('loads with no DOM at all', typeof EQ.init === 'function');
ok('the module reports version 3', /^3\./.test(EQ.version), EQ.version);
ok('init on a missing element returns null instead of throwing', EQ.init('#not-here') === null);
ok('init on null returns null', EQ.init(null) === null);
ok('initAll on a missing selector returns an empty list', EQ.initAll('#not-here').length === 0);
ok('the public surface index.html depends on is intact',
   typeof EQ.caption === 'string' && typeof EQ.init === 'function' &&
   typeof EQ.initAll === 'function' && typeof EQ.autoInit === 'function' &&
   typeof EQ.renderMarkup === 'function' && typeof EQ.css === 'string' &&
   EQ.defaults && EQ.presets && EQ.Instance && EQ.PluckState &&
   EQ.curves && EQ.constants && EQ.math && EQ.geometry);
ok('custom curves of a different length are still accepted by the maths', (function () {
  var t = [], s = [], i;
  for (i = 0; i < 24; i++) { t.push(Math.sin(i / 4)); s.push(Math.cos(i / 3)); }
  var tt = EQ.math.split(t, 2, 2), ss = EQ.math.split(s, 2, 2);
  var out = EQ.math.morph(ss.lo, ss.hi, tt.lo, tt.hi, 1, 1);
  return maxAbsDiff(out, t) < 1e-12;
})());
ok('buildModel honours a different sample count', (function () {
  var m = EQ.math.buildModel({ n: 61 });
  return m.n === 61 && m.teacher.length === 61 && m.teacherRipple.length === 61 &&
         Math.abs(Math.max.apply(null, m.teacherEnvelope) - 1) < 1e-9;
})());
ok('buildModel honours different lobes', (function () {
  var m = EQ.math.buildModel({ studentLobes: [{ c: 0.5, w: 0.2, a: 1 }] });
  return Math.abs(argmax(m.studentEnvelope) - 90) <= 2;
})());
ok('a zero studentKeep gives a perfectly smooth student', (function () {
  var m = EQ.math.buildModel({ studentKeep: 0 });
  return maxAbs(m.studentRipple) === 0 && curvature(m.student) < 0.01;
})());
ok('the deterministic generator is deterministic', (function () {
  var a = EQ.math.lcg(7), b = EQ.math.lcg(7), i;
  for (i = 0; i < 50; i++) { if (a() !== b()) { return false; } }
  return EQ.math.lcg(7)() !== EQ.math.lcg(8)();
})());
ok('lobeSum of nothing is zero, not NaN', EQ.math.lobeSum(null, 0.5) === 0);
ok('the edge window is 0 at both ends and 1 in the middle', (function () {
  var w = EQ.math.edgeWindow(101, 0.1);
  return w[0] === 0 && w[100] === 0 && Math.abs(w[50] - 1) < 1e-12;
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
head('RENDER — the draw path, driven with stub nodes');
note('There is no DOM here, so render() is run against stub elements that record');
note('every setAttribute. This is what proves the callouts, the dashed reference, the');
note('grab highlight and the readout actually move when the faders do.');

function stub(attrs) {
  var n = {
    attrs: attrs || {}, textContent: '',
    setAttribute: function (k, v) { this.attrs[k] = String(v); },
    getAttribute: function (k) { return this.attrs.hasOwnProperty(k) ? this.attrs[k] : null; },
    removeAttribute: function (k) { delete this.attrs[k]; }
  };
  return n;
}

var RENDER_HOOKS = ['tArea', 'tCore', 'ghost', 'sArea', 'sCore', 'armDot',
                    'annShape', 'shapeT1', 'shapeT2', 'shapeLead', 'gapStem', 'gapTop', 'gapBot',
                    'annDetail', 'detailT1', 'detailT2', 'detailLead', 'detailBr',
                    'gapValue', 'roLabel', 'scrub', 'scrubLine', 'scrubDotS', 'scrubDotT'];

function fakeInstance() {
  var inst = Object.create(EQ.Instance.prototype), i, o = {}, k;
  for (k in EQ.defaults) { if (Object.prototype.hasOwnProperty.call(EQ.defaults, k)) { o[k] = EQ.defaults[k]; } }
  o.onChange = null;
  inst.opts = o;
  inst.destroyed = false;
  inst.n = N;
  inst.model = { teacher: teacher, student: student };
  inst.teacher = teacher; inst.student = student;
  inst.tEnv = T.lo; inst.tRip = T.hi; inst.sEnv = S.lo; inst.sRip = S.hi;
  inst.reach = REACH;
  /* The stub has to carry EVERY field render() reads off an instance, and this
   * one was missing: Instance.prototype.morph multiplies hf by this.detailReach,
   * so an absent field made `hf * undefined` = NaN at every pose including
   * hf = 0, and all 181 student samples, the dimension bar, the bracket, the
   * scrub dots and the readout followed it down. The widget now also declares
   * both travel limits on the prototype, so this line states the shipped pose
   * rather than papering over a missing one. */
  inst.detailReach = EQ.constants.DETAIL_REACH;
  inst.state = { lf: 0, hf: 0 };
  inst.pluck = new EQ.PluckState({ span: SPAN });
  inst.arm = null; inst.armAt = -1; inst.scrubAt = -1;
  inst.lastShapeDone = null; inst.lastDetailDone = null;
  inst.vmax = C.vmax;
  inst.span = SPANY;
  inst.xs = []; inst.ys = []; inst.tys = [];
  for (i = 0; i < N; i++) { inst.xs.push(PL.l + (PL.r - PL.l) * i / (N - 1)); inst.ys.push(0); inst.tys.push(0); }
  inst.annShapeI = C.shapeDepartureIndex;
  inst.detailFrom = Math.round(EQ.constants.ANN_DETAIL_FROM * (N - 1));
  inst.detailTo = Math.round(EQ.constants.ANN_DETAIL_TO * (N - 1));
  inst.el = {};
  for (i = 0; i < RENDER_HOOKS.length; i++) { inst.el[RENDER_HOOKS[i]] = stub(); }
  inst.faders = {
    lf: { group: stub(), cap: stub(), fill: stub(), val: stub() },
    hf: { group: stub(), cap: stub(), fill: stub(), val: stub() }
  };
  inst.chips = [
    stub({ 'data-lieq-lf': '0', 'data-lieq-hf': '0' }),
    stub({ 'data-lieq-lf': '1', 'data-lieq-hf': '0' }),
    stub({ 'data-lieq-lf': '1', 'data-lieq-hf': '1' })
  ];
  inst.schedule = function () { /* no raf here: render is called directly */ };
  inst.stage = stub();
  return inst;
}

(function () {
  var I = fakeInstance(), el = I.el, i, d, stemLen, brY0, brY1, thrown = null;

  try { I.render(); } catch (err) { thrown = err; }
  ok('render() completes against stub nodes without throwing', thrown === null,
     thrown ? String(thrown && thrown.message) : 'all ' + RENDER_HOOKS.length + ' hooks written');

  /* ---- the four painted paths ---- */
  ok('both curves and both area fills are drawn',
     /^M[-\d. ]+L/.test(el.tCore.getAttribute('d')) && /^M[-\d. ]+L/.test(el.sCore.getAttribute('d')) &&
     /Z$/.test(el.tArea.getAttribute('d')) && /Z$/.test(el.sArea.getAttribute('d')));
  ok('the student path has one vertex per sample',
     el.sCore.getAttribute('d').split('L').length - 1 === N - 1);
  ok('the drawn curves land inside the plot box', (function () {
     var j, worst = 0;
     for (j = 0; j < N; j++) {
       if (I.ys[j] < PL.t - 4 || I.ys[j] > PL.b) { worst++; }
       if (I.tys[j] < PL.t - 4 || I.tys[j] > PL.b) { worst++; }
     }
     return worst === 0;
   })(), 'y from ' + Math.min.apply(null, I.tys).toFixed(1) + ' to ' + PL.b);

  /* ---- the dashed dense SHAPE: visible at rest, gone at full LF ---- */
  ok('the dashed reference shape is visible when LF is at rest',
     parseFloat(el.ghost.getAttribute('opacity')) > 0.5,
     'opacity ' + el.ghost.getAttribute('opacity'));
  I.state.lf = 1; I.render();
  ok('and it retires completely once LF has arrived',
     parseFloat(el.ghost.getAttribute('opacity')) === 0);
  I.state.lf = 0.5; I.render();
  ok('fading in between', (function () {
    var v = parseFloat(el.ghost.getAttribute('opacity'));
    return v > 0 && v < 0.5;
  })(), 'opacity ' + el.ghost.getAttribute('opacity') + ' at lf=0.5');

  /* ---- callout 1: the SHAPE bar tracks the two curves and closes with LF ---- */
  I.state.lf = 0; I.state.hf = 0; I.render();
  i = I.annShapeI;
  ok('the SHAPE dimension bar spans exactly the two drawn curves',
     Math.abs(parseFloat(el.gapStem.getAttribute('y1')) - Math.round(I.ys[i] * 10) / 10) < 0.06 &&
     Math.abs(parseFloat(el.gapStem.getAttribute('y2')) - Math.round(I.tys[i] * 10) / 10) < 0.06,
     'from y ' + el.gapStem.getAttribute('y1') + ' to ' + el.gapStem.getAttribute('y2'));
  ok('its caps sit on the two curves and are the stated width',
     Math.abs(parseFloat(el.gapTop.getAttribute('x2')) - parseFloat(el.gapTop.getAttribute('x1'))) ===
       2 * EQ.geometry.ANN.shape.cap &&
     el.gapTop.getAttribute('y1') === el.gapTop.getAttribute('y2'));
  stemLen = Math.abs(parseFloat(el.gapStem.getAttribute('y1')) - parseFloat(el.gapStem.getAttribute('y2')));
  ok('the bar is tall enough to read at rest', stemLen > 25, stemLen.toFixed(1) + ' user units');
  ok('the SHAPE callout starts with the sentence that is true at rest',
     el.shapeT1.textContent === EQ.defaults.shapeAnnotation[0], '"' + el.shapeT1.textContent + '"');
  I.state.lf = 1; I.render();
  ok('the bar all but closes once LF is at full travel',
     Math.abs(parseFloat(el.gapStem.getAttribute('y1')) -
              parseFloat(el.gapStem.getAttribute('y2'))) < 0.25 * stemLen,
     Math.abs(parseFloat(el.gapStem.getAttribute('y1')) -
              parseFloat(el.gapStem.getAttribute('y2'))).toFixed(1) + ' units left');
  ok('and the callout says so instead of still claiming a wrong shape',
     el.shapeT1.textContent === EQ.defaults.shapeAnnotationDone[0] &&
     el.shapeT2.textContent === EQ.defaults.shapeAnnotationDone[1],
     '"' + el.shapeT1.textContent + ' ' + el.shapeT2.textContent + '"');
  ok('the wording swaps exactly at ANN_DONE_AT', (function () {
    I.state.lf = EQ.constants.ANN_DONE_AT - 0.01; I.render();
    var before = el.shapeT1.textContent;
    I.state.lf = EQ.constants.ANN_DONE_AT; I.render();
    return before === EQ.defaults.shapeAnnotation[0] &&
           el.shapeT1.textContent === EQ.defaults.shapeAnnotationDone[0];
  })());
  ok('the callout fades as it becomes less urgent, but stays readable',
     (function () {
       I.state.lf = 0; I.render();
       var a = parseFloat(el.annShape.getAttribute('opacity'));
       I.state.lf = 1; I.render();
       var b = parseFloat(el.annShape.getAttribute('opacity'));
       return a === 1 && b >= 0.4 && b < a;
     })());
  ok('its leader is vertical and sits between the text and the ink',
     el.shapeLead.getAttribute('x1') === el.shapeLead.getAttribute('x2') &&
     parseFloat(el.shapeLead.getAttribute('y1')) === EQ.geometry.ANN.shape.leadTop &&
     parseFloat(el.shapeLead.getAttribute('y2')) >= EQ.geometry.ANN.shape.leadTop);

  /* ---- callout 2: the DETAIL bracket ---- */
  I.state.lf = 0; I.state.hf = 0; I.render();
  d = el.detailBr.getAttribute('d');
  ok('the DETAIL bracket is drawn over the bracketed stretch',
     d.indexOf('M' + (Math.round(I.xs[I.detailFrom] * 10) / 10)) === 0 &&
     d.indexOf('' + (Math.round(I.xs[I.detailTo] * 10) / 10)) > 0, d);
  brY0 = parseFloat(d.split('L')[1].split(' ')[1]);
  ok('it sits above both curves in that stretch', (function () {
    var j, lowest = Infinity;
    for (j = I.detailFrom; j <= I.detailTo; j++) { lowest = Math.min(lowest, I.ys[j], I.tys[j]); }
    return brY0 < lowest - 4;
  })(), 'bracket at y ' + brY0);
  ok('it is never allowed to climb into the text', brY0 >= EQ.geometry.ANN.detail.minY,
     'floor is y ' + EQ.geometry.ANN.detail.minY);
  /* In the bracketed stretch the two SHAPES agree, so moving LF does not move
   * the bracket: it is pinned by whichever curve reaches highest there, which
   * is the dense scan's spikes. What must hold is that it always clears both
   * curves, at every pose and under a pluck. */
  ok('it clears both curves at every pose on an 11x11 fader grid', (function () {
    var a, b, j, low, bad = 0, yb;
    for (a = 0; a <= 10; a++) {
      for (b = 0; b <= 10; b++) {
        I.state.lf = a / 10; I.state.hf = b / 10; I.render();
        yb = parseFloat(el.detailBr.getAttribute('d').split('L')[1].split(' ')[1]);
        low = Infinity;
        for (j = I.detailFrom; j <= I.detailTo; j++) { low = Math.min(low, I.ys[j], I.tys[j]); }
        if (!(yb < low - 4) || yb < EQ.geometry.ANN.detail.minY) { bad++; }
      }
    }
    return bad === 0;
  })(), '121 poses checked');
  ok('and it follows a curve that is pulled up into the bracketed stretch', (function () {
    I.state.lf = 1; I.state.hf = 1; I.render();
    var y0 = parseFloat(el.detailBr.getAttribute('d').split('L')[1].split(' ')[1]), y1;
    I.pluck.begin('s', Math.round((I.detailFrom + I.detailTo) / 2), I.svals[Math.round((I.detailFrom + I.detailTo) / 2)]);
    I.pluck.to(I.svals[Math.round((I.detailFrom + I.detailTo) / 2)] + 0.3);
    I.render();
    y1 = parseFloat(el.detailBr.getAttribute('d').split('L')[1].split(' ')[1]);
    I.pluck.cancel(); I.render();
    brY1 = y1;
    return y1 < y0 - 10;
  })(), 'bracket rose to y ' + brY1 + ' under a pluck');
  ok('the DETAIL callout starts with the sentence that is true at rest',
     (function () { I.state.hf = 0; I.render(); return el.detailT1.textContent === EQ.defaults.detailAnnotation[0]; })());
  ok('and swaps once HF has filled the detail in',
     (function () { I.state.hf = 1; I.render(); return el.detailT1.textContent === EQ.defaults.detailAnnotationDone[0]; })(),
     '"' + el.detailT1.textContent + '"');

  /* ---- the grab highlight ---- */
  I.arm = null; I.armAt = -1; I.render();
  ok('the grab dot is invisible when the pointer is not near a curve',
     parseFloat(el.armDot.getAttribute('opacity')) === 0);
  I.arm = 't'; I.armAt = 60; I.render();
  ok('it lands on the TEACHER when the teacher would be grabbed',
     Math.abs(parseFloat(el.armDot.getAttribute('cy')) - Math.round(I.tys[60] * 10) / 10) < 0.06 &&
     el.armDot.getAttribute('stroke') === '#bfe0ff' &&
     parseFloat(el.armDot.getAttribute('opacity')) > 0.9);
  I.arm = 's'; I.render();
  ok('and on the STUDENT when the student would be grabbed',
     Math.abs(parseFloat(el.armDot.getAttribute('cy')) - Math.round(I.ys[60] * 10) / 10) < 0.06 &&
     el.armDot.getAttribute('stroke') === '#f6bd8c');
  I.arm = null; I.armAt = -1;

  /* ---- readout and scrub ---- */
  I.state.lf = 1; I.state.hf = 1; I.scrubAt = -1; I.render();
  /* Compared against the gap between the two curves the instance actually drew,
   * not against FULL_SCALE_RESIDUAL: the shape residual is only part of the
   * full-travel gap now that the detail channel keeps its own, and 0.075 and
   * the drawn 0.085 happen to print the same two decimals, so matching the
   * constant would pass for the wrong reason. The "never 0.00" in the name is
   * the one the NaN regression broke: with hf * undefined the loop compared
   * NaN > gap, which is false at every sample, and the readout sat at 0.00. */
  var drawnPeak = (function () {
    var j, g = 0, dd;
    for (j = 0; j < N; j++) {
      dd = Math.abs(I.svals[j] - I.tvals[j]);
      if (dd > g) { g = dd; }
    }
    return g;
  })();
  ok('the PEAK GAP readout shows the residual at full travel, never 0.00',
     el.gapValue.textContent === drawnPeak.toFixed(2) && el.gapValue.textContent !== '0.00' &&
     drawnPeak >= RESIDUAL - 1e-12 && el.roLabel.textContent === 'PEAK GAP ',
     'reads "' + el.roLabel.textContent + el.gapValue.textContent + '" for a drawn gap of ' +
     drawnPeak.toFixed(5) + ', shape residual ' + RESIDUAL);
  I.scrubAt = 100; I.render();
  ok('the scrub cursor switches the readout to the LOCAL gap',
     el.roLabel.textContent === 'GAP HERE ' &&
     el.gapValue.textContent === Math.abs(I.svals[100] - I.tvals[100]).toFixed(2).replace('0.00', '0.00'),
     'reads "' + el.roLabel.textContent + el.gapValue.textContent + '"');
  ok('and parks its two dots on the two curves',
     parseFloat(el.scrub.getAttribute('opacity')) === 1 &&
     Math.abs(parseFloat(el.scrubDotS.getAttribute('cy')) - Math.round(I.ys[100] * 10) / 10) < 0.06 &&
     Math.abs(parseFloat(el.scrubDotT.getAttribute('cy')) - Math.round(I.tys[100] * 10) / 10) < 0.06);
  I.scrubAt = -1; I.render();
  ok('the scrub cursor hides itself again', parseFloat(el.scrub.getAttribute('opacity')) === 0);

  /* ---- faders and presets ---- */
  I.state.lf = 0.25; I.state.hf = 0.75; I.render();
  ok('each fader cap is translated to its own value',
     I.faders.lf.cap.getAttribute('transform') === 'translate(0,' +
       (EQ.geometry.TRACK.half - 0.25 * EQ.geometry.TRACK.half * 2) + ')' &&
     I.faders.hf.cap.getAttribute('transform') === 'translate(0,' +
       (EQ.geometry.TRACK.half - 0.75 * EQ.geometry.TRACK.half * 2) + ')');
  ok('each fader prints its value and reports it to assistive tech',
     I.faders.lf.val.textContent === '0.25' && I.faders.hf.val.textContent === '0.75' &&
     I.faders.lf.group.getAttribute('aria-valuenow') === '25' &&
     I.faders.hf.group.getAttribute('aria-valuetext') === '75%');
  ok('no preset is lit when the faders are between presets',
     I.chips[0].getAttribute('aria-pressed') === 'false' &&
     I.chips[1].getAttribute('aria-pressed') === 'false' &&
     I.chips[2].getAttribute('aria-pressed') === 'false');
  I.state.lf = 1; I.state.hf = 1; I.render();
  ok('and the LF+HF preset lights up when the faders sit exactly on it',
     I.chips[2].getAttribute('aria-pressed') === 'true' &&
     I.chips[0].getAttribute('aria-pressed') === 'false');

  /* ---- a pluck is drawn, and only on the curve that was grabbed ---- */
  I.state.lf = 0.4; I.state.hf = 0.4; I.render();
  var beforeT = I.tys.slice(0), beforeS = I.ys.slice(0);
  /* sample 150 is out on the right-hand shoulder, well clear of the plot's top
   * edge, so the full pull is visible instead of saturating against the frame */
  I.pluck.begin('t', 150, I.tvals[150]);
  I.pluck.to(I.tvals[150] + 0.25);
  I.render();
  ok('plucking the teacher moves the teacher on screen',
     Math.abs(I.tys[150] - beforeT[150]) > 20, (beforeT[150] - I.tys[150]).toFixed(1) + ' units');
  ok('and leaves the student exactly where it was',
     maxAbsDiff(I.ys, beforeS) === 0);
  I.pluck.cancel(); I.render();
  ok('cancelling the pluck restores the fader-determined shape bit for bit',
     maxAbsDiff(I.tys, beforeT) === 0 && maxAbsDiff(I.ys, beforeS) === 0);

  /* ---- onChange ---- */
  var seen = [];
  I.opts.onChange = function (lf, hf) { seen.push(lf + '/' + hf); };
  I.state.lf = 0.6; I.render();
  ok('onChange is called with the live fader values', seen.length === 1 && seen[0] === '0.6/0.4');
  I.opts.onChange = function () { throw new Error('host blew up'); };
  thrown = null;
  try { I.render(); } catch (err2) { thrown = err2; }
  ok('a host callback that throws does not take the widget down', thrown === null);
  I.opts.onChange = null;
})();

/* ------------------------------------------------------------------ */
console.log('\n' + (fail === 0 ? 'ALL PASS' : 'FAILURES') + ': ' + pass + ' passed, ' + fail + ' failed.');
process.exit(fail === 0 ? 0 : 1);
