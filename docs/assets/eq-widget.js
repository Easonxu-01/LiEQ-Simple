/*!
 * eq-widget.js — LiEqualizer project page, the spatial-profile equaliser.
 *
 * A self-contained hero instrument. Two vertical faders ("LF" and "HF")
 * continuously morph the sparse student's profile toward the dense teacher's.
 * Both rest at 0. Releasing a fader leaves the curve exactly where it was put —
 * the faders never spring back.
 *
 * THE PLOT IS IN THE SPATIAL DOMAIN. x is a spatial position, y is a signal /
 * feature response. The HEIGHT of a curve therefore says NOTHING about
 * frequency, and no region of either curve is ever labelled with a frequency.
 * What the two words mean here is what they mean in the paper's insight figure,
 * whose own x axis is labelled "Spatial position":
 *
 *   LOW  FREQUENCY  ==  the ENVELOPE, the gross shape of the curve.
 *   HIGH FREQUENCY  ==  the fine spiky ripple riding on top of that envelope.
 *
 * So the instrument is built from four arrays, not two curves:
 *
 *   teacherEnvelope   one dominant hump with subordinate shoulders
 *   teacherRipple     dense, sharp, varied spikes — detail a DENSE scan keeps
 *   studentEnvelope   a different, WRONG shape: two humps where the dense scan
 *                     has one dominant hump
 *   studentRipple     a 5% trace of the teacher's texture — what survived a
 *                     sparse sampling of it, i.e. visually nothing
 *
 * and the drawn student at faders (lf, hf) is exactly
 *
 *   env[i]    = lerp(studentEnvelope[i], teacherEnvelope[i], lf)
 *   ripple[i] = lerp(studentRipple[i],   teacherRipple[i],   hf)
 *   y[i]      = env[i] + ripple[i]
 *
 * LF moves the SHAPE onto the dense scan's and leaves the student smooth;
 * HF grows the fine detail until the student's texture approaches the
 * teacher's. At full travel the student lands CLOSE TO but not ON the teacher:
 * FULL_SCALE_RESIDUAL is left behind on the shape channel and a deliberately
 * larger residual on the detail channel (DETAIL_REACH), because the paper's own
 * measurements show a distilled student near the teacher, not equal to it. Both
 * numbers live in section 1; no prose in this file restates them, because the
 * last copy of "4% of the signal range" outlived the constant it described.
 *
 * Four more things move, so the instrument is not just two sliders:
 *   · either curve can be PLUCKED — grab it anywhere, pull, and on release it
 *     springs back to the shape the faders imply, with a short underdamped
 *     settle. A pluck never touches the fader values. ONE transparent hit
 *     surface covers the plot and plucks whichever curve is nearer in y, so
 *     the teacher stays grabbable where the two curves are close (v2 stacked
 *     two fat invisible strokes and the student's swallowed the teacher's).
 *   · a scrub cursor follows the pointer across the plot and reads out the
 *     local teacher-student gap.
 *   · two in-plot callouts state, in plain spatial language, what the sparse
 *     scan got wrong: the SHAPE (which LF closes) and the DETAIL it never
 *     recorded (which HF fills in). Their wording changes as the faders move,
 *     because at full travel neither sentence is true any more.
 *   · three preset pills (BYPASS / LF / LF+HF) jump the faders to the three
 *     states of the paper's insight figure. They jump; they do not rebound.
 *
 * Native inline SVG + one injected <style>. No framework, no bundler, no CDN,
 * no chart library, no build step. ES5-compatible. Safe to load with a plain
 * <script src="assets/eq-widget.js" defer></script>.
 *
 * Public API (window.LiEQEqualizer):
 *   init(target, opts)   -> instance | null     mount into one element
 *   initAll(sel, opts)   -> [instance]          mount into every match
 *   autoInit()                                  runs itself on #eq-mount
 *   renderMarkup(o, uid) -> string              DOM-free markup, for the tests
 *   css / caption / defaults / presets / constants / math / geometry
 *   Instance / PluckState
 *   curves               -> { teacher, student, teacherEnvelope, teacherRipple,
 *                             studentEnvelope, studentRipple, vmax, reach }
 *
 * Instance:
 *   .element .svg .state .caption
 *   .getValues() -> {lf, hf}
 *   .setValues(lf, hf)      .setValue('lf', v)
 *   .morph(lf, hf) -> []    the curve the given pose would draw (pure)
 *   .destroy()
 *
 * Both profiles are drawn procedurally from the named constants in section 1,
 * after the paper's insight figure. They are illustrative, not measured; see
 * CAPTION below.
 */
(function (root, factory) {
  'use strict';
  var api = factory();
  if (typeof module === 'object' && module && module.exports) { module.exports = api; }
  if (root) { root.LiEQEqualizer = api; }
})(
  typeof window !== 'undefined' ? window
    : (typeof globalThis !== 'undefined' ? globalThis : null),
  function () {
'use strict';

var VERSION = '3.0.0';

/* The one-line disclosure the page should print under the instrument. Nothing
 * here is traced off a measurement: the two profiles are built out of the
 * constants below so that the one thing the figure is about — a shape plus its
 * fine detail — can be moved by a fader. That is an illustrative choice and it
 * is stated out loud. */
var CAPTION =
  'Illustrative, not measured: both profiles are drawn procedurally after the paper’s insight figure — ' +
  'one dominant shape carrying fine detail for the dense scan, a smoothed two-hump shape for the sparse ' +
  'one — and not plotted from logged features.';

/* ==============================================================
 * 1.  The curve model — every number that shapes the two profiles.
 *
 * Tune the instrument here and nowhere else. There is no table of
 * hand-typed coordinates in this file on purpose: the four arrays are
 * generated from these constants, so a change of mind is one number.
 * ============================================================== */

/* How many samples the shared x grid carries. High enough that a spike can be
 * two samples wide and still read as a spike (181 samples over a 300-unit plot
 * is 1.67 units per sample), low enough that the path strings stay cheap. */
var N_SAMPLES = 181;

/* Both profiles are a sum of Gaussian lobes over u in [0, 1]:
 *     y(u) = sum a * exp(-((u - c) / w)^2)
 * c is where the lobe sits, w its half-width, a its height. The teacher gets
 * ONE dominant lobe and a few subordinate shoulders — "a clear single dominant
 * envelope". The student gets TWO comparable humps, neither of them where the
 * dominant lobe is: the shape a sparse scan reports instead. */
var TEACHER_LOBES = [
  { c: 0.160, w: 0.065, a: 0.34 },
  { c: 0.300, w: 0.085, a: 0.60 },
  { c: 0.430, w: 0.105, a: 1.00 },   /* the one dominant hump */
  { c: 0.560, w: 0.070, a: 0.34 },
  { c: 0.735, w: 0.085, a: 0.42 },
  { c: 0.875, w: 0.055, a: 0.17 }
];

var STUDENT_LOBES = [
  { c: 0.260, w: 0.112, a: 0.82 },   /* hump one: too early, too tall */
  { c: 0.550, w: 0.098, a: 0.60 },   /* hump two: a shape the dense scan lacks */
  /* the right-hand shoulders ARE roughly right: a sparse scan does not get
   * everything wrong, and a stretch where only the DETAIL differs is what the
   * second callout needs in order to be true */
  { c: 0.740, w: 0.082, a: 0.38 },
  { c: 0.875, w: 0.055, a: 0.16 }
];

/* Both curves are windowed to exactly 0 at both ends over this fraction of the
 * width, so the filled areas close on the axis as the paper's do. */
var EDGE_TAPER = 0.035;

/* The teacher's fine detail. Spikes, not a sine: a tall sample with two short
 * wings, repeated every stride samples, heights drawn from a deterministic
 * generator so every visitor and the test harness see the same curve.
 *   amp          the tallest ordinary spike, in signal units (plot spans ~0..1.1)
 *   minH         the shortest spike as a fraction of the tallest
 *   accentEvery  every nth spike is taller still: structure, not noise
 *   wing         what the samples either side of a spike get, as a fraction
 *   strideMin/Max  spacing in samples, varied so it does not read as a comb
 *   flip         chance of alternating sign at each spike (mostly, not always)
 *   maskPow      the detail rides ON the signal: amplitude follows the
 *                envelope to this power, so the texture is strongest where the
 *                signal is
 *   maskFloor    but never vanishes. A dense scan resolves fine structure
 *                along the whole sweep, not only under the main lobe, and the
 *                paper's insight figure draws it that way. Without a floor the
 *                mask drove the outer sixths to a measured RMS of 0.009 and
 *                0.008 against 0.035 mid-curve, i.e. visibly smooth flanks --
 *                exactly the "too little high-frequency detail" this is for.
 *   edge         samples at each end left completely alone
 */
var RIPPLE = {
  amp: 0.082,
  minH: 0.38,
  accentEvery: 7,
  accentGain: 1.55,
  wing: 0.26,
  strideMin: 2,
  strideMax: 4,
  flip: 0.82,
  maskPow: 0.8,
  maskFloor: 0.5,
  edge: 4,
  seed: 20260413
};

/* What the sparse scan kept of that texture. Not zero — a sparse scan is not
 * perfectly smooth — but 5% of a 0.082 spike is 0.004 of a unit range, which
 * is a third of a screen pixel: visually nothing, which is the point.
 *
 * It is a SCALED COPY of the teacher's ripple rather than an independent
 * wiggle, and that is deliberate: it makes the HF fader exactly a gain on the
 * dense scan's own texture, so the drawn ripple energy is strictly monotone in
 * hf (it is |(1-h)k + h| * ||r|| with k < 1), which the harness asserts. */
var STUDENT_RIPPLE_KEEP = 0.05;

/* What the equaliser CANNOT recover. At lf = hf = 1 the student lands near the
 * teacher but not on it. Honest, too: the paper reports a distilled student
 * close to the dense teacher, never equal to it.
 *
 * Measured at 0.04 the two curves read as merged: 135 of 181 samples came
 * closer than 0.01 and the mean separation was 1.25 user units, under a pixel
 * and a half. Raised so the shape residual is actually legible; the difference
 * that persists along the WHOLE curve is DETAIL_REACH's job, below.
 *
 * The faders travel over `reach = 1 - residual / peakGap` of the path. */
var FULL_SCALE_RESIDUAL = 0.075;

/* The detail channel keeps a BIGGER residual than the shape channel, and that
 * is the physics, not a cosmetic choice: the paper's own measurements put the
 * distilled student's high-frequency retention well short of the dense
 * teacher's even where the coarse shape has been brought into line. Drawn, it
 * is the one residual that is visible along the WHOLE curve rather than only
 * at the humps -- the student's spikes come back shallower than the teacher's
 * everywhere, so at full travel the two profiles stay legibly distinct instead
 * of merging over two thirds of the span. */
var DETAIL_REACH = 0.78;

/* Grabbing. ONE transparent surface covers the plot; a press picks the nearer
 * curve in y within this radius, in viewBox units (14 units is ~17 CSS px on
 * the desktop hero, ~10 px on a phone). A touch gets a more generous radius
 * because a fingertip has no pixel. GRAB_WINDOW samples either side of the
 * pointer are searched, so a steep or spiky stretch is not penalised. */
var GRAB_RADIUS = 14;
var GRAB_RADIUS_TOUCH = 26;
var GRAB_WINDOW = 2;

/* Pluck: how far either side of the grab point the deformation reaches, as a
 * fraction of the sample count. A raised-cosine bump, so the pull is local but
 * perfectly smooth — never a single-sample spike. */
var PLUCK_SPAN_FRAC = 0.14;

/* Pluck spring: slightly underdamped, like a cord. zeta 0.34 at 7.2 Hz settles
 * to invisibility in ~0.32 s after about two visible swings. */
var PLUCK_ZETA = 0.34;
var PLUCK_FREQ = 7.2;
var PLUCK_SETTLE_EPS = 0.0015;   /* signal units; < 0.25 user units on screen */
var PLUCK_MIN_TIME = 0.05;       /* seconds; never settle before this */
var PLUCK_MAX_AMP = 0.45;        /* signal units; the cord has a limit */

/* Where the DETAIL callout's bracket spans, as a fraction of the x grid: a
 * stretch where the dense scan's texture is strong and the sparse scan's
 * SHAPE is already roughly right, so the only thing missing there is detail. */
var ANN_DETAIL_FROM = 0.68;
var ANN_DETAIL_TO = 0.81;

/* Above this much travel each callout says the opposite thing, because the
 * sentence it started with has stopped being true. */
var ANN_DONE_AT = 0.55;

/* ==============================================================
 * 2.  Pure maths — no DOM, safe to require() in node
 * ============================================================== */

function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

function lerp(a, b, t) { return a + (b - a) * t; }

function parseCurve(src) {
  var raw, out, i, v;
  if (typeof src !== 'string') {
    out = [];
    if (src && src.length) { for (i = 0; i < src.length; i++) { out.push(+src[i]); } }
    return out;
  }
  raw = src.split(',');
  out = [];
  for (i = 0; i < raw.length; i++) {
    v = parseFloat(raw[i]);
    out.push(isFinite(v) ? v : 0);
  }
  return out;
}

/* Box moving average, `passes` times, edges clamped. A few passes of a short
 * window is the cheap Gaussian the recipe calls for. Kept because the test
 * harness measures envelopes with it and a host may pass its own curves. */
function smooth(src, radius, passes) {
  var a = src.slice(0), n = a.length, w = 2 * radius + 1, p, i, j, s, b;
  if (!(radius > 0) || !(passes > 0) || n === 0) { return a; }
  for (p = 0; p < passes; p++) {
    b = new Array(n);
    for (i = 0; i < n; i++) {
      s = 0;
      for (j = i - radius; j <= i + radius; j++) {
        s += a[j < 0 ? 0 : (j > n - 1 ? n - 1 : j)];
      }
      b[i] = s / w;
    }
    a = b;
  }
  return a;
}

/* y = lo + hi, exactly. */
function split(src, radius, passes) {
  var lo = smooth(src, radius, passes), hi = new Array(src.length), i;
  for (i = 0; i < src.length; i++) { hi[i] = src[i] - lo[i]; }
  return { lo: lo, hi: hi };
}

/* The whole instrument, in one line:
 *     y[i] = lerp(studentEnvelope, teacherEnvelope, lf)
 *          + lerp(studentRipple,   teacherRipple,   hf)
 * Separately linear in lf and in hf, so the response is continuous, every
 * intermediate fader position is a proportional intermediate curve, and the
 * extremes of the family are reached at the four corners of the unit square.
 * The widget feeds it lf * reach and hf * reach (see `reach` below), which is
 * still exactly linear — it only shortens the path.
 *
 * The parameter names are historical (sLo/sHi/tLo/tHi): read them as
 * studentEnvelope / studentRipple / teacherEnvelope / teacherRipple. */
function morph(sLo, sHi, tLo, tHi, lf, hf) {
  var n = sLo.length, out = new Array(n), i;
  for (i = 0; i < n; i++) {
    out[i] = lerp(sLo[i], tLo[i], lf) + lerp(sHi[i], tHi[i], hf);
  }
  return out;
}

/* How much of the student-to-teacher path a full fader travels, so that at
 * full travel the largest remaining difference is exactly `residual`.
 * Returns 1 when the two curves already coincide. */
function reach(student, teacher, residual) {
  var n = Math.min(student.length, teacher.length), g = 0, i, d, k;
  for (i = 0; i < n; i++) {
    d = student[i] - teacher[i];
    if (d < 0) { d = -d; }
    if (d > g) { g = d; }
  }
  if (!(g > 0)) { return 1; }
  k = residual / g;
  if (!(k > 0)) { return 1; }
  if (k >= 1) { return 0; }
  return 1 - k;
}

/* Raised cosine bump: 1 at d = 0, 0 at |d| >= w, smooth (C1) at the edges.
 * This is the falloff a pluck deforms the curve with. */
function bump(d, w) {
  var t;
  if (!(w > 0)) { return d === 0 ? 1 : 0; }
  t = d / w;
  if (t <= -1 || t >= 1) { return 0; }
  return 0.5 * (1 + Math.cos(Math.PI * t));
}

/* Underdamped free response, released from rest at amplitude a0:
 *     a(t) = a0 e^(-z w0 t) [ cos(wd t) + (z w0 / wd) sin(wd t) ]
 * Time-based, so it is frame-rate independent and a dropped frame does not
 * change where the cord ends up. */
function spring(a0, t, zeta, freq) {
  var z = zeta === undefined ? PLUCK_ZETA : zeta,
      f = freq === undefined ? PLUCK_FREQ : freq,
      w0 = 2 * Math.PI * f, wd;
  if (!(t > 0)) { return a0; }
  if (z >= 1) { return a0 * Math.exp(-w0 * t) * (1 + w0 * t); }
  wd = w0 * Math.sqrt(1 - z * z);
  return a0 * Math.exp(-z * w0 * t) *
         (Math.cos(wd * t) + (z * w0 / wd) * Math.sin(wd * t));
}

function n1(v) { return Math.round(v * 10) / 10; }

/* Straight segments, one per sample. This is what the two live curves are
 * drawn with: at 1.67 units per sample the envelope still reads as a smooth
 * curve, and a spike keeps a SHARP tip instead of the rounded overshoot a
 * spline puts on it. Shorter strings too, which matters several times a frame. */
function linePath(xs, ys) {
  var n = Math.min(xs.length, ys.length), d, i;
  if (n < 2) { return ''; }
  d = 'M' + n1(xs[0]) + ' ' + n1(ys[0]);
  for (i = 1; i < n; i++) { d += 'L' + n1(xs[i]) + ' ' + n1(ys[i]); }
  return d;
}

/* Uniform Catmull-Rom through every sample, emitted as cubic Béziers. Used for
 * the dashed reference envelope, which has no spikes to blunt.
 * End tangents are clamped (P-1 = P0, Pn+1 = Pn). */
function catmullPath(xs, ys) {
  var n = xs.length, d, i, x0, y0, x1, y1, x2, y2, x3, y3, c1x, c1y, c2x, c2y;
  if (n < 2) { return ''; }
  d = 'M' + n1(xs[0]) + ' ' + n1(ys[0]);
  for (i = 0; i < n - 1; i++) {
    x0 = xs[i > 0 ? i - 1 : 0];           y0 = ys[i > 0 ? i - 1 : 0];
    x1 = xs[i];                           y1 = ys[i];
    x2 = xs[i + 1];                       y2 = ys[i + 1];
    x3 = xs[i + 2 < n ? i + 2 : n - 1];   y3 = ys[i + 2 < n ? i + 2 : n - 1];
    c1x = x1 + (x2 - x0) / 6;  c1y = y1 + (y2 - y0) / 6;
    c2x = x2 - (x3 - x1) / 6;  c2y = y2 - (y3 - y1) / 6;
    d += 'C' + n1(c1x) + ' ' + n1(c1y) + ' ' + n1(c2x) + ' ' + n1(c2y) +
         ' ' + n1(x2) + ' ' + n1(y2);
  }
  return d;
}

/* Close a curve path down to the baseline and back: the filled area the paper
 * figure draws under both profiles. */
function areaPath(d, x0, x1, baseline) {
  if (!d) { return ''; }
  return d + 'L' + n1(x1) + ' ' + n1(baseline) + 'L' + n1(x0) + ' ' + n1(baseline) + 'Z';
}

/* A square bracket over [x1, x2] with legs pointing down at the curve. */
function bracketPath(x1, x2, y, drop) {
  return 'M' + n1(x1) + ' ' + n1(y + drop) + 'L' + n1(x1) + ' ' + n1(y) +
         'L' + n1(x2) + ' ' + n1(y) + 'L' + n1(x2) + ' ' + n1(y + drop);
}

/* ---- the four arrays -------------------------------------------------- */

function lobeSum(lobes, u) {
  var s = 0, i, z;
  if (!lobes) { return 0; }
  for (i = 0; i < lobes.length; i++) {
    z = (u - lobes[i].c) / lobes[i].w;
    s += lobes[i].a * Math.exp(-z * z);
  }
  return s;
}

/* Raised-cosine edge window: 1 in the middle, exactly 0 at both ends. */
function edgeWindow(n, frac) {
  var out = new Array(n), i, u;
  for (i = 0; i < n; i++) {
    u = i / (n - 1);
    if (!(frac > 0)) { out[i] = 1; }
    else if (u < frac) { out[i] = 0.5 * (1 - Math.cos(Math.PI * u / frac)); }
    else if (u > 1 - frac) { out[i] = 0.5 * (1 - Math.cos(Math.PI * (1 - u) / frac)); }
    else { out[i] = 1; }
  }
  return out;
}

/* A 32-bit linear congruential generator. Deliberately NOT Math.random: the
 * spike heights must be identical for every visitor, in every browser, and in
 * the test harness. */
function lcg(seed) {
  var s = (seed >>> 0) || 1;
  return function () {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

/* Dense, sharp, varied spikes, masked by `mask` so the detail rides on the
 * signal. See the RIPPLE constants for what each knob does. */
/* Per-key fallback for a config object. `rip = o.ripple || RIPPLE` was a
 * wholesale swap while every sibling option fell back per key, so passing
 * {ripple:{amp:0.11}} -- the one knob most likely to be turned -- left
 * cfg.edge undefined, the spike loop never ran, and the mask multiplied by
 * Math.pow(m, undefined) = NaN. Nothing threw; the curve just disappeared. */
function mergeCfg(base, over) {
  var out = {}, k;
  for (k in base) { if (Object.prototype.hasOwnProperty.call(base, k)) { out[k] = base[k]; } }
  if (over) {
    for (k in over) {
      if (Object.prototype.hasOwnProperty.call(over, k) && over[k] !== undefined) { out[k] = over[k]; }
    }
  }
  return out;
}

function buildRipple(n, mask, cfg) {
  var out = new Array(n), rnd = lcg(cfg.seed), i, k, h, w, sign = 1, hits = 0,
      mmax = 0, m, stride, strideSpan, floor;

  for (i = 0; i < n; i++) { out[i] = 0; }
  strideSpan = Math.max(0, (cfg.strideMax | 0) - (cfg.strideMin | 0));
  k = cfg.edge;
  while (k < n - cfg.edge) {
    hits++;
    h = cfg.amp * (cfg.minH + (1 - cfg.minH) * rnd());
    if (cfg.accentEvery > 0 && hits % cfg.accentEvery === 0) { h *= cfg.accentGain; }
    h *= sign;
    out[k] += h;
    w = cfg.wing * h;
    if (k > 0) { out[k - 1] += w; }
    if (k + 1 < n) { out[k + 1] += w; }
    /* mostly alternate: a perfect comb reads as a machine, never as texture */
    if (rnd() < cfg.flip) { sign = -sign; }
    stride = (cfg.strideMin | 0) + Math.floor(rnd() * (strideSpan + 1));
    if (stride < 1) { stride = 1; }
    k += stride;
  }

  for (i = 0; i < n; i++) { if (mask[i] > mmax) { mmax = mask[i]; } }
  if (!(mmax > 0)) { mmax = 1; }
  floor = cfg.maskFloor > 0 ? cfg.maskFloor : 0;
  if (floor > 1) { floor = 1; }
  for (i = 0; i < n; i++) {
    m = mask[i] / mmax;
    if (m < 0) { m = 0; }
    out[i] *= floor + (1 - floor) * Math.pow(m, cfg.maskPow);
    /* the floor keeps texture on the flanks, but a filled area may never
     * cross its own baseline: where the envelope is smaller than the spike
     * the floor would allow, the spike is trimmed to what the signal can
     * carry. Tails therefore stay textured AND non-negative — without this the
     * tails reached -0.0197.
     *
     * The trim is deliberately a hard pin to -mask[i] rather than a rescale:
     * measured, it fires at 3 of 181 samples (6, 11, 174), all of them out
     * where the envelope is under 0.04, and it costs the sharpness of 1 of the
     * 60 detected spikes (the one at sample 11, whose drop to its neighbour
     * falls to 0.38 of its height against 0.73 for the worst untrimmed spike).
     * Three samples sitting exactly on the baseline is cheaper than a rescale
     * that would flatten the flank texture the floor exists to keep. */
    if (mask[i] + out[i] < 0) { out[i] = -mask[i]; }
  }
  return out;
}

function addArrays(a, b) {
  var n = Math.min(a.length, b.length), out = new Array(n), i;
  for (i = 0; i < n; i++) { out[i] = a[i] + b[i]; }
  return out;
}

function maxOf(a) {
  var m = -Infinity, i;
  for (i = 0; i < a.length; i++) { if (a[i] > m) { m = a[i]; } }
  return m;
}

/* The whole model, built from section 1. Both envelopes are scaled by the SAME
 * factor — the teacher's own peak — so their relative heights mean something:
 * the student overshoots the dense scan in one place and falls far short of it
 * in another, which is the entire job of the LF branch. */
function buildModel(cfg) {
  var o = cfg || {},
      n = o.n > 3 ? (o.n | 0) : N_SAMPLES,
      tl = o.teacherLobes || TEACHER_LOBES,
      sl = o.studentLobes || STUDENT_LOBES,
      rip = mergeCfg(RIPPLE, o.ripple),
      keep = o.studentKeep === undefined ? STUDENT_RIPPLE_KEEP : o.studentKeep,
      win = edgeWindow(n, o.taper === undefined ? EDGE_TAPER : o.taper),
      tEnv = new Array(n), sEnv = new Array(n), sRip = new Array(n),
      tRip, i, u, k;

  for (i = 0; i < n; i++) {
    u = i / (n - 1);
    tEnv[i] = lobeSum(tl, u);
    sEnv[i] = lobeSum(sl, u);
  }
  k = maxOf(tEnv);
  k = k > 0 ? 1 / k : 1;
  for (i = 0; i < n; i++) {
    tEnv[i] = tEnv[i] * k * win[i];
    sEnv[i] = sEnv[i] * k * win[i];
  }

  tRip = buildRipple(n, tEnv, rip);
  for (i = 0; i < n; i++) { sRip[i] = keep * tRip[i]; }

  return {
    n: n,
    teacherEnvelope: tEnv,
    teacherRipple: tRip,
    studentEnvelope: sEnv,
    studentRipple: sRip,
    teacher: addArrays(tEnv, tRip),
    student: addArrays(sEnv, sRip)
  };
}

/* The vertical scale. The family is separately linear in lf and in hf, so its
 * extremes over the unit square are attained at the four corners. */
function modelVmax(m, rch, drch) {
  var hi = 0, c, i, v, poses = [[0, 0], [1, 0], [0, 1], [1, 1]],
      dr = drch === undefined ? rch : drch;
  for (c = 0; c < poses.length; c++) {
    v = morph(m.studentEnvelope, m.studentRipple, m.teacherEnvelope, m.teacherRipple,
              poses[c][0] * rch, poses[c][1] * dr);
    for (i = 0; i < v.length; i++) { if (v[i] > hi) { hi = v[i]; } }
  }
  /* the teacher is drawn too. Taking the max over the student family alone put
   * the teacher's peak at y = 29.68 against PLOT.t = 30, so the blue curve
   * crossed the top frame line and ate the headroom. */
  for (i = 0; i < m.teacher.length; i++) { if (m.teacher[i] > hi) { hi = m.teacher[i]; } }
  return hi > 0 ? hi : 1;
}

/* Where the sparse scan's SHAPE departs upward from the dense scan's by the
 * most: the sample the SHAPE callout points at. Signed on purpose — the
 * overshooting hump has clear space above it for a label, and it is the
 * departure the LF fader visibly erases. */
function shapeDepartureIndex(sEnv, tEnv) {
  var n = Math.min(sEnv.length, tEnv.length), best = 0, bestD = -Infinity, i, d;
  for (i = 0; i < n; i++) {
    d = sEnv[i] - tEnv[i];
    if (d > bestD) { bestD = d; best = i; }
  }
  return best;
}

/* ---- the hit test ----------------------------------------------------- */

/* Which curve a press at (sample i0, screen y py) should pluck.
 *
 * Version 2 painted two invisible 22-unit-wide strokes, one per curve, with
 * the student's LAST. Wherever the two curves came within ~11 units of each
 * other the student's band covered the teacher's and the teacher could not be
 * grabbed at all: measured with real pointer events, 3 of 9 presses along the
 * teacher moved it, and at the 6 failures the element under the point was the
 * student's band. This is the replacement, and it is a pure function so the
 * harness can probe it directly: ONE surface, nearest curve wins.
 *
 * tys/sys are the drawn screen y of the two curves. `window` samples either
 * side of i0 are searched so a steep or spiky stretch is not penalised.
 * Returns null when neither curve is within `radius`. Ties go to the teacher,
 * which is both the curve drawn underneath and the one that used to be
 * impossible to reach. */
function segDist(px, py, ax, ay, bx, by) {
  var vx = bx - ax, vy = by - ay, L = vx * vx + vy * vy, t, qx, qy;
  if (!(L > 0)) { qx = ax; qy = ay; }
  else {
    t = ((px - ax) * vx + (py - ay) * vy) / L;
    t = t < 0 ? 0 : (t > 1 ? 1 : t);
    qx = ax + t * vx; qy = ay + t * vy;
  }
  return Math.sqrt((px - qx) * (px - qx) + (py - qy) * (py - qy));
}

/* Distance from the pointer to a drawn polyline, searched over the samples
 * around i0. Falls back to the vertical distance at the samples when no xs
 * array is supplied, which is what the pure-function tests exercise. */
function curveDist(i0, px, py, xs, ys, w, n) {
  var d = Infinity, lo = i0 - w < 0 ? 0 : i0 - w,
      hi = i0 + w > n - 1 ? n - 1 : i0 + w, i, v;
  if (!xs) {
    for (i = lo; i <= hi; i++) { v = Math.abs(py - ys[i]); if (v < d) { d = v; } }
    return d;
  }
  for (i = lo; i < hi; i++) {
    v = segDist(px, py, xs[i], ys[i], xs[i + 1], ys[i + 1]);
    if (v < d) { d = v; }
  }
  if (lo === hi) { d = Math.abs(py - ys[lo]); }
  return d;
}

/* ONE surface, nearest curve wins.
 *
 * `xs` and `px` are optional. When they are given the comparison is the true
 * point-to-segment distance to the DRAWN polyline, which is what the user is
 * aiming at; without them it degrades to the vertical distance at the nearest
 * samples. The difference matters because the teacher carries dense fine
 * ripple, so between two samples the drawn curve can sit several units away
 * from either sample's value -- measured with real pointer events, the
 * sample-only form picked the wrong curve at 2 of 12 presses on each curve.
 *
 * Returns null when neither curve is within `radius`. Ties go to the teacher,
 * which is both the curve drawn underneath and the one that used to be
 * impossible to reach. */
function pickCurve(i0, py, tys, sys, radius, window, xs, px) {
  var n = Math.min(tys ? tys.length : 0, sys ? sys.length : 0),
      w = window > 0 ? Math.round(window) : 0,
      useX = !!(xs && xs.length >= n && isFinite(px)),
      dt, ds;
  if (!(n > 0) || !isFinite(py)) { return null; }
  i0 = clamp(Math.round(i0), 0, n - 1);
  if (useX && w < 3) { w = 3; }     /* segments need a little more reach */
  dt = curveDist(i0, px, py, useX ? xs : null, tys, w, n);
  ds = curveDist(i0, px, py, useX ? xs : null, sys, w, n);
  if (dt > radius && ds > radius) { return null; }
  if (dt <= ds) { return { which: 't', i: i0, dist: dt, other: ds }; }
  return { which: 's', i: i0, dist: ds, other: dt };
}

/* ==============================================================
 * 3.  PluckState — the direct-manipulation state machine.
 *
 * Deliberately DOM-free and deliberately ignorant of lf/hf: it cannot
 * change a fader because it cannot see one. The instance's pointer
 * handlers are thin wrappers over begin/to/release, and render() asks
 * it for the current amplitude once a frame.
 * ============================================================== */

function PluckState(opts) {
  var o = opts || {};
  this.span = o.span > 0 ? o.span : 1;
  this.zeta = o.zeta === undefined ? PLUCK_ZETA : o.zeta;
  this.freq = o.freq === undefined ? PLUCK_FREQ : o.freq;
  this.maxAmp = o.maxAmp === undefined ? PLUCK_MAX_AMP : o.maxAmp;
  this.active = null;      /* { which: 's'|'t', i0: int, base: number } */
  this.amp0 = 0;
  this.held = false;
  this.releasedAt = 0;
}

/* Grab curve `which` at sample `i0`, whose undeformed value is `base`. */
PluckState.prototype.begin = function (which, i0, base) {
  this.active = { which: which, i0: i0, base: +base || 0 };
  this.amp0 = 0;
  this.held = true;
  this.releasedAt = 0;
  return this;
};

/* Drag: the grabbed sample is pulled to `value`. */
PluckState.prototype.to = function (value) {
  if (!this.active) { return this; }
  this.amp0 = clamp((+value || 0) - this.active.base, -this.maxAmp, this.maxAmp);
  this.held = true;
  return this;
};

/* Let go at time `now` (ms). */
PluckState.prototype.release = function (now) {
  if (!this.active) { return this; }
  this.held = false;
  this.releasedAt = +now || 0;
  return this;
};

/* Drop the deformation immediately — used for reduced motion and destroy(). */
PluckState.prototype.cancel = function () {
  this.active = null; this.amp0 = 0; this.held = false;
  return this;
};

/* Current amplitude at time `now` (ms). Clears itself once the cord has
 * settled, so the drawn shape returns to exactly what the faders imply. */
PluckState.prototype.amp = function (now) {
  var t, a;
  if (!this.active) { return 0; }
  if (this.held) { return this.amp0; }
  t = ((+now || 0) - this.releasedAt) / 1000;
  if (!(t > 0)) { t = 0; }
  a = spring(this.amp0, t, this.zeta, this.freq);
  if (t > PLUCK_MIN_TIME && Math.abs(a) < PLUCK_SETTLE_EPS) { this.cancel(); return 0; }
  return a;
};

PluckState.prototype.settled = function () { return !this.active; };

/* Add the deformation to `vals` IN PLACE and return it. A no-op when nothing
 * is grabbed, when the grab is on the other curve, or when amp is 0 — so the
 * settled shape is bit-for-bit the shape the faders imply. */
PluckState.prototype.apply = function (vals, which, amp) {
  var a = this.active, w = this.span, i, d, lo, hi;
  if (!a || a.which !== which || !amp) { return vals; }
  lo = Math.ceil(a.i0 - w); hi = Math.floor(a.i0 + w);
  if (lo < 0) { lo = 0; }
  if (hi > vals.length - 1) { hi = vals.length - 1; }
  for (i = lo; i <= hi; i++) {
    d = i - a.i0;
    vals[i] += amp * bump(d, w);
  }
  return vals;
};

/* ==============================================================
 * 4.  Geometry — viewBox 480 x 248, aspect 1.935.
 *
 * Deliberately SHORTER than the 470 x 280 (1.679) of version 1: at the
 * hero's 1.22fr column the card was taller than the sibling page's and
 * crowded the title column. See the README for the arithmetic.
 * ============================================================== */

var VB_W = 480, VB_H = 248;

var PLOT = { l: 46, r: 346, t: 30, b: 200 };         /* drawing frame           */
var PLOT_HEAD = 6;                                    /* headroom above the peak */
var RACK = { x: 362, y: 20, w: 110, h: 216 };         /* the fader rack          */
var FADER_X = { lf: 389, hf: 445 };
var TRACK = { top: 66, bottom: 200, mid: 133, half: 67 };  /* len = 134 */

/* legend run: swatch, text, swatch, text */
var LEGEND = { y: 17, baseline: 20.5, sw1: 46, t1: 70, sw2: 164, t2: 188, swLen: 18 };

/* The SHAPE callout lives in the plot's empty upper LEFT bay, above the
 * sparse scan's first hump; its dimension bar drops straight down from the
 * text onto the two curves, so the leader never crosses any ink. */
var ANN_SHAPE = { textX: PLOT.l + 6, y1: 44, y2: 55, leadTop: 60, cap: 4 };

/* The DETAIL callout lives in the upper RIGHT bay, under the pluck hint, and
 * brackets the stretch where the dense scan's texture is strongest. */
var ANN_DETAIL = { textX: PLOT.r - 2, y1: 60, y2: 71, leadTop: 75, minY: 82, drop: 5 };

/* The pluck hint: a pill, top right INSIDE the plot, with a vertical
 * double-arrow glyph. It retires for good the first time anyone plucks. */
var HINT = { x: PLOT.r - 2, y: 34, h: 13, pad: 7, fs: 8, ls: 0.12, glyph: 7, gap: 5 };

/* the bottom rows: axis ticks + title, then the preset pills and the readout */
var ROW_TICK = 213, ROW_TITLE = 215, ROW_CHIP = 229, ROW_READOUT = 232;
var CHIP = { h: 13, hit: 24, gap: 8, pad: 8, fs: 8, ls: 0.1 };

/* Dual clamp, exactly as the sibling widget does it: a tight box for anything
 * the user drags, an 8px-inset box for anything computed. */
var VB_INSET = 8;

/* Chip boxes are sized from the LARGEST font the chip label will ever be
 * painted at (the narrow-screen size), so the pill never has to resize and the
 * text never escapes it. Same advance model the test harness uses. */
function advance(label, fs, ls) {
  var s = String(label == null ? '' : label);
  if (!s.length) { return 0; }
  return s.length * fs * 0.6 + (s.length - 1) * ls * fs;
}

function chipWidth(label) {
  return n1(advance(label, CHIP.fs, CHIP.ls) + CHIP.pad * 2);
}

/* Same idea for the hint pill, plus room for the glyph. */
function hintWidth(label) {
  return n1(advance(label, HINT.fs, HINT.ls) + HINT.pad * 2 + HINT.glyph + HINT.gap);
}

/* ==============================================================
 * 5.  Styles — injected once per document
 * ============================================================== */

var MONO = "var(--mono,'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,monospace)";

var CSS = [
  /* ---- stage ---- */
  '.lieq-stage{position:relative;overflow:hidden;border-radius:14px;font-family:' + MONO + '}',
  '.lieq-svg{display:block;width:100%;height:auto;position:relative;z-index:2}',
  '.lieq-svg text{font-family:' + MONO + ';pointer-events:none}',

  /* ---- the "powered on" scan line ---- */
  '.lieq-scan{position:absolute;left:0;right:0;top:0;height:1px;z-index:4;pointer-events:none;opacity:0;',
  'background:linear-gradient(90deg,transparent 7%,#60c4f5 50%,transparent 93%);',
  'box-shadow:0 0 16px rgba(96,196,245,.7);animation:lieq-scan 5.4s ease-in-out infinite}',
  '@keyframes lieq-scan{0%,8%{top:2%;opacity:0}16%{opacity:.72}78%{opacity:.26}90%,100%{top:98%;opacity:0}}',

  /* ---- optional standalone card chrome (opts.chrome) ---- */
  '.lieq-card{position:relative;width:100%;margin:0;overflow:hidden;padding:clamp(18px,2vw,25px);',
  'border:1px solid rgba(255,255,255,.12);border-radius:25px;',
  'background:radial-gradient(circle at 48% 42%,rgba(58,126,188,.16),transparent 42%),',
  'linear-gradient(155deg,#16223a 0%,#0c1322 100%);',
  'box-shadow:inset 0 1px rgba(255,255,255,.1),inset 0 0 52px rgba(71,135,190,.045),',
  '0 30px 72px -42px rgba(21,90,145,.9),0 0 0 1px rgba(86,151,203,.1)}',
  '.lieq-card::before{content:"";position:absolute;inset:0;pointer-events:none;',
  'background:linear-gradient(rgba(121,169,211,.035) 1px,transparent 1px),',
  'linear-gradient(90deg,rgba(121,169,211,.035) 1px,transparent 1px);background-size:34px 34px;',
  '-webkit-mask-image:linear-gradient(180deg,#000,transparent);mask-image:linear-gradient(180deg,#000,transparent)}',

  /* ---- axes, grid, legend ---- */
  '.lieq-grid{stroke:rgba(121,169,211,.11);stroke-width:1;stroke-dasharray:2 5}',
  '.lieq-axis{stroke:rgba(134,182,239,.34);stroke-width:1.2;stroke-linecap:square}',
  '.lieq-tick{stroke:rgba(134,182,239,.3);stroke-width:1.2}',
  '.lieq-tick-t{fill:#7f8ea4;font-size:7.5px;font-weight:700;letter-spacing:.14em}',
  '.lieq-axis-t{fill:#8d99ab;font-size:8.5px;letter-spacing:.06em}',
  '.lieq-leg{fill:#8d99ab;font-size:8.5px;font-weight:700;letter-spacing:.09em}',
  '.lieq-sw-t{stroke:#56adf0;stroke-width:3.2;stroke-linecap:round}',
  '.lieq-sw-s{stroke:#e58236;stroke-width:3.2;stroke-linecap:round}',

  /* ---- the two profiles: a filled area with a solid outline, as the paper
   *      figure draws them. Thin outlines, because a fat stroke swallows the
   *      fine detail this whole widget is about. ---- */
  '.lieq-curve{fill:none;stroke-linecap:round;stroke-linejoin:round;pointer-events:none}',
  '.lieq-area{stroke:none;pointer-events:none}',
  '.lieq-t-core{stroke:#56adf0;stroke-width:1.5;transition:stroke-width .14s ease,stroke .14s ease}',
  '.lieq-s-core{stroke:#e58236;stroke-width:1.6;transition:stroke-width .14s ease,stroke .14s ease}',
  /* the dense scan's SHAPE, dashed: what the LF fader is aiming the student at */
  '.lieq-ghost{fill:none;stroke:#cfe6ff;stroke-width:1.2;stroke-dasharray:6 5;',
  'stroke-linecap:round;pointer-events:none;transition:opacity .2s ease}',

  /* ---- grabbing: ONE transparent surface over the plot. The nearest curve in
   *      y wins the press, so the teacher stays reachable where the two curves
   *      run close together. ---- */
  '.lieq-hitfield{fill:rgba(0,0,0,0);pointer-events:all;touch-action:none;cursor:default}',
  '.lieq-stage[data-lieq-arm] .lieq-hitfield{cursor:grab}',
  '.lieq-stage[data-lieq-pluck="1"] .lieq-hitfield{cursor:grabbing}',
  /* whichever curve WOULD be grabbed lights up, so the affordance is honest */
  '.lieq-stage[data-lieq-hot="t"] .lieq-t-core{stroke-width:2.6;stroke:#bfe0ff}',
  '.lieq-stage[data-lieq-hot="s"] .lieq-s-core{stroke-width:2.7;stroke:#f6bd8c}',
  '.lieq-armdot{fill:none;stroke-width:1.6;pointer-events:none;transition:opacity .12s ease}',

  /* ---- the pluck hint, retired the first time someone plucks ---- */
  '.lieq-hintg{pointer-events:none;transition:opacity .45s ease;animation:lieq-nudge 3.6s ease-in-out infinite}',
  '.lieq-hintbox{fill:rgba(18,30,52,.86);stroke:rgba(159,208,255,.52);stroke-width:1}',
  '.lieq-hint{fill:#cfe3f7;font-size:7px;font-weight:700;letter-spacing:.12em}',
  '.lieq-hintglyph{stroke:#9fd0ff;stroke-width:1.2;stroke-linecap:round;fill:none}',
  '@keyframes lieq-nudge{0%,100%{transform:translateY(0)}50%{transform:translateY(2.2px)}}',
  '.lieq-stage[data-lieq-plucked="1"] .lieq-hintg{animation:none}',

  /* ---- scrub cursor ---- */
  '.lieq-scrub{pointer-events:none;opacity:0;transition:opacity .16s ease}',
  '.lieq-scrub-line{stroke:rgba(134,182,239,.45);stroke-width:1;stroke-dasharray:1 4}',
  '.lieq-scrub-s{fill:#f6bd8c}',
  '.lieq-scrub-t{fill:#bfe0ff}',

  /* ---- the two callouts. Plain spatial statements: a shape, and a texture.
   *      Nothing here labels a REGION of the curve with a frequency, because
   *      in the spatial domain the height of a curve carries no frequency. ---- */
  '.lieq-ann{pointer-events:none;transition:opacity .2s ease}',
  '.lieq-ann-t{fill:#eab089;font-size:7.5px;font-weight:700;letter-spacing:.06em}',
  '.lieq-ann-line{stroke:#eab089;stroke-width:1.1;stroke-linecap:round}',
  '.lieq-ann-dash{stroke:#eab089;stroke-width:1.1;stroke-dasharray:2 3;stroke-linecap:round}',
  '.lieq-ann-br{fill:none;stroke:#eab089;stroke-width:1.1;stroke-linecap:round;stroke-linejoin:round}',

  /* ---- readout ---- */
  '.lieq-ro-k{fill:#6d7b8e;font-size:8px;font-weight:700;letter-spacing:.13em}',
  '.lieq-ro-v{fill:#dbe7f5;font-size:8px;font-weight:700;letter-spacing:.08em}',

  /* ---- preset pills ---- */
  '.lieq-chip{cursor:pointer;outline:none;touch-action:manipulation}',
  '.lieq-chip-hit{fill:rgba(0,0,0,0)}',
  '.lieq-chip-box{fill:rgba(22,34,58,.72);stroke:rgba(134,182,239,.26);stroke-width:1;',
  'transition:fill .16s ease,stroke .16s ease}',
  '.lieq-chip-t{fill:#8d99ab;font-size:7px;font-weight:700;letter-spacing:.1em;text-anchor:middle;',
  'transition:fill .16s ease}',
  '.lieq-chip:hover .lieq-chip-box{fill:#16314f;stroke:rgba(134,182,239,.5)}',
  '.lieq-chip:hover .lieq-chip-t{fill:#dbe7f5}',
  '.lieq-chip[aria-pressed="true"] .lieq-chip-box{fill:rgba(42,120,214,.3);stroke:#56adf0}',
  '.lieq-chip[aria-pressed="true"] .lieq-chip-t{fill:#dbe7f5}',

  /* ---- rack ---- */
  '.lieq-rack-eyebrow{fill:#8d99ab;font-size:8px;font-weight:700;letter-spacing:.17em}',
  '.lieq-fader{cursor:grab;touch-action:none;outline:none}',
  '.lieq-fader.lieq-dragging{cursor:grabbing}',
  '.lieq-hit{fill:rgba(0,0,0,0)}',
  '.lieq-detent{stroke:rgba(134,182,239,.3);stroke-width:1.1;stroke-linecap:round}',
  '.lieq-cap-body{fill:#0c1322;stroke:#9fd0ff;stroke-width:1.8;transition:fill .18s ease}',
  '.lieq-cap-line{stroke:#9fd0ff;stroke-width:1.4;stroke-linecap:round;opacity:.85}',
  '.lieq-fader:hover .lieq-cap-body,.lieq-fader.lieq-dragging .lieq-cap-body{fill:#16314f}',
  '.lieq-val{fill:#dbe7f5;font-size:9.5px;font-weight:700;letter-spacing:.05em;text-anchor:middle}',
  '.lieq-cap-label{fill:#dbe7f5;font-size:12px;font-weight:700;letter-spacing:.1em;text-anchor:middle}',
  '.lieq-cap-sub{fill:#8d99ab;font-size:7px;font-weight:700;letter-spacing:.1em;text-anchor:middle}',

  /* ---- focus ---- */
  '.lieq-focusring{fill:none;stroke:#58b9eb;stroke-width:2;opacity:0;pointer-events:none}',
  '.lieq-fader:focus .lieq-focusring,.lieq-chip:focus .lieq-focusring{opacity:1}',
  '.lieq-fader:focus:not(:focus-visible) .lieq-focusring,',
  '.lieq-chip:focus:not(:focus-visible) .lieq-focusring{opacity:0}',
  '.lieq-fader:focus-visible .lieq-focusring,.lieq-chip:focus-visible .lieq-focusring{opacity:1}',

  /* ---- respect the setting ---- */
  '@media (prefers-reduced-motion:reduce){',
  '.lieq-scan{animation:none;opacity:0}',
  '.lieq-hintg{animation:none;transition:none}',
  '.lieq-scrub,.lieq-ann,.lieq-ghost,.lieq-armdot{transition:none}',
  '.lieq-t-core,.lieq-s-core,.lieq-cap-body,.lieq-chip-box,.lieq-chip-t{transition:none}}',

  /* ---- narrow screens: the card shrinks, so the type grows in user units.
   *      Every size here is collision-checked by eq-widget.test.js.
   *      This block must stay LAST in the sheet; the harness parses its tail. ---- */
  '@media (max-width:760px){',
  '.lieq-leg{font-size:9.5px}',
  '.lieq-axis-t{font-size:9.5px}',
  '.lieq-tick-t{font-size:9px}',
  '.lieq-ro-k{font-size:9px}',
  '.lieq-ro-v{font-size:9px}',
  '.lieq-ann-t{font-size:8px}',
  '.lieq-hint{font-size:8px}',
  '.lieq-chip-t{font-size:8px}',
  '.lieq-rack-eyebrow{font-size:9.5px}',
  '.lieq-val{font-size:11.5px}',
  '.lieq-cap-label{font-size:14px}',
  '.lieq-cap-sub{font-size:8px}}'
].join('');

function injectCSS(doc) {
  var style;
  if (!doc || doc.getElementById('lieq-equalizer-css')) { return; }
  style = doc.createElement('style');
  style.id = 'lieq-equalizer-css';
  style.appendChild(doc.createTextNode(CSS));
  (doc.head || doc.documentElement).appendChild(style);
}

/* ==============================================================
 * 6.  Markup
 * ============================================================== */

function esc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function faderMarkup(uid, key, cx, label, sub, aria) {
  var g = [], k, ty;
  g.push('<g class="lieq-fader" data-lieq-fader="' + key + '"');
  g.push(' transform="translate(' + cx + ',' + TRACK.mid + ')"');
  g.push(' role="slider" tabindex="0" aria-orientation="vertical"');
  g.push(' aria-label="' + esc(aria) + '"');
  g.push(' aria-valuemin="0" aria-valuemax="100" aria-valuenow="0" aria-valuetext="0%">');
  g.push('<rect class="lieq-focusring" x="-21" y="-98" width="42" height="200" rx="12"/>');
  /* generous touch target: 50 x 156 user units, ~33 x 104 CSS px at phone width */
  g.push('<rect class="lieq-hit" x="-25" y="-78" width="50" height="156"/>');
  g.push('<rect class="lieq-track" x="-5" y="-' + TRACK.half + '" width="10" height="' +
         (TRACK.half * 2) + '" rx="5" fill="url(#lieqTrack' + uid + ')" ' +
         'stroke="rgba(134,182,239,.22)" stroke-width="1"/>');
  for (k = 0; k <= 4; k++) {
    ty = n1(TRACK.half - k * (TRACK.half * 2 / 4));
    g.push('<line class="lieq-detent" x1="-12" y1="' + ty + '" x2="-8" y2="' + ty + '"/>');
    g.push('<line class="lieq-detent" x1="8" y1="' + ty + '" x2="12" y2="' + ty + '"/>');
  }
  g.push('<rect class="lieq-fill" x="-5" y="' + TRACK.half + '" width="10" height="0" rx="5" ' +
         'fill="url(#lieqFill' + uid + ')"/>');
  g.push('<g class="lieq-capg" data-lieq-knob="1" transform="translate(0,' + TRACK.half + ')" ' +
         'filter="url(#lieqGlow' + uid + ')">');
  g.push('<rect class="lieq-cap-body" x="-13" y="-7.5" width="26" height="15" rx="4.5"/>');
  g.push('<line class="lieq-cap-line" x1="-8" y1="0" x2="8" y2="0"/>');
  g.push('</g>');
  g.push('<text class="lieq-val" y="-79">0.00</text>');
  g.push('<text class="lieq-cap-label" y="88">' + esc(label) + '</text>');
  g.push('<text class="lieq-cap-sub" y="100">' + esc(sub) + '</text>');
  g.push('</g>');
  return g.join('');
}

function presetMarkup(presets) {
  var s = [], x = PLOT.l, i, p, w;
  if (!presets || !presets.length) { return ''; }
  for (i = 0; i < presets.length; i++) {
    p = presets[i];
    w = chipWidth(p.label);
    s.push('<g class="lieq-chip" data-lieq-preset="' + i + '" data-lieq-lf="' + p.lf +
           '" data-lieq-hf="' + p.hf + '" role="button" tabindex="0" aria-pressed="false" ' +
           'aria-label="' + esc(p.aria) + '" transform="translate(' + n1(x) + ',' + ROW_CHIP + ')">');
    s.push('<rect class="lieq-chip-hit" x="0" y="' + n1(-CHIP.hit / 2) + '" width="' + w +
           '" height="' + CHIP.hit + '"/>');
    s.push('<rect class="lieq-focusring" x="-2.5" y="' + n1(-CHIP.h / 2 - 2.5) + '" width="' +
           n1(w + 5) + '" height="' + n1(CHIP.h + 5) + '" rx="' + n1(CHIP.h / 2 + 2.5) + '"/>');
    s.push('<rect class="lieq-chip-box" x="0" y="' + n1(-CHIP.h / 2) + '" width="' + w +
           '" height="' + CHIP.h + '" rx="' + n1(CHIP.h / 2) + '"/>');
    s.push('<text class="lieq-chip-t" x="' + n1(w / 2) + '" y="2.5">' + esc(p.label) + '</text>');
    s.push('</g>');
    x += w + CHIP.gap;
  }
  return s.join('');
}

/* The pluck affordance, spelled out. Version 2 put two small words in the
 * corner and the first thing a reader reported was that the curves could not
 * be dragged at all — so it is a pill with a border, a vertical double-arrow
 * glyph, and a slow nudge, and it retires for good after the first pluck. */
function hintMarkup(label) {
  var w = hintWidth(label), bx = n1(HINT.x - w), cy = n1(HINT.y + HINT.h / 2),
      gx = n1(HINT.x - w + HINT.pad + HINT.glyph / 2), s = [];
  s.push('<g class="lieq-hintg" data-lieq="hintg">');
  s.push('<rect class="lieq-hintbox" x="' + bx + '" y="' + HINT.y + '" width="' + w +
         '" height="' + HINT.h + '" rx="' + n1(HINT.h / 2) + '"/>');
  s.push('<path class="lieq-hintglyph" d="M' + gx + ' ' + n1(cy - 4.2) + 'L' + gx + ' ' + n1(cy + 4.2) +
         'M' + n1(gx - 2.2) + ' ' + n1(cy - 2) + 'L' + gx + ' ' + n1(cy - 4.4) +
         'L' + n1(gx + 2.2) + ' ' + n1(cy - 2) +
         'M' + n1(gx - 2.2) + ' ' + n1(cy + 2) + 'L' + gx + ' ' + n1(cy + 4.4) +
         'L' + n1(gx + 2.2) + ' ' + n1(cy + 2) + '"/>');
  s.push('<text class="lieq-hint" x="' + n1(bx + HINT.pad + HINT.glyph + HINT.gap) +
         '" y="' + n1(cy + 2.6) + '">' + esc(label) + '</text>');
  s.push('</g>');
  return s.join('');
}

function svgMarkup(uid, o) {
  var s = [], i, gx, gy;

  s.push('<svg class="lieq-svg" viewBox="0 0 ' + VB_W + ' ' + VB_H + '" width="100%" ' +
         'role="group" aria-label="' + esc(o.ariaLabel) + '" ' +
         'aria-describedby="lieqDesc' + uid + '">');
  s.push('<desc id="lieqDesc' + uid + '">' + esc(o.desc + ' ' + o.caption) + '</desc>');

  /* ---- defs ---- */
  s.push('<defs>');
  s.push('<linearGradient id="lieqTrack' + uid + '" x1="0" y1="0" x2="1" y2="0">' +
         '<stop offset="0" stop-color="#0a111b"/><stop offset=".55" stop-color="#16223a"/>' +
         '<stop offset="1" stop-color="#0a111b"/></linearGradient>');
  s.push('<linearGradient id="lieqFill' + uid + '" x1="0" y1="0" x2="0" y2="1">' +
         '<stop offset="0" stop-color="#f6bd8c"/><stop offset="1" stop-color="#e58236"/></linearGradient>');
  s.push('<linearGradient id="lieqTArea' + uid + '" x1="0" y1="0" x2="0" y2="1">' +
         '<stop offset="0" stop-color="#56adf0" stop-opacity=".26"/>' +
         '<stop offset=".7" stop-color="#56adf0" stop-opacity=".07"/>' +
         '<stop offset="1" stop-color="#16223a" stop-opacity="0"/></linearGradient>');
  s.push('<linearGradient id="lieqSArea' + uid + '" x1="0" y1="0" x2="0" y2="1">' +
         '<stop offset="0" stop-color="#e58236" stop-opacity=".22"/>' +
         '<stop offset=".7" stop-color="#e58236" stop-opacity=".06"/>' +
         '<stop offset="1" stop-color="#16223a" stop-opacity="0"/></linearGradient>');
  s.push('<linearGradient id="lieqRack' + uid + '" x1="0" y1="0" x2=".55" y2="1">' +
         '<stop offset="0" stop-color="#16223a" stop-opacity=".82"/>' +
         '<stop offset="1" stop-color="#0a111b" stop-opacity=".92"/></linearGradient>');
  s.push('<filter id="lieqGlow' + uid + '" x="-70%" y="-70%" width="240%" height="240%">' +
         '<feGaussianBlur stdDeviation="2.4" result="b"/>' +
         '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>');
  s.push('<clipPath id="lieqClip' + uid + '"><rect x="' + (PLOT.l - 2) + '" y="' + (PLOT.t - 8) +
         '" width="' + (PLOT.r - PLOT.l + 4) + '" height="' + (PLOT.b - PLOT.t + 10) + '"/></clipPath>');
  s.push('</defs>');

  /* ---- grid ---- */
  s.push('<g aria-hidden="true" class="lieq-deco">');
  for (i = 1; i <= 4; i++) {
    gy = n1(PLOT.b - (PLOT.b - PLOT.t - PLOT_HEAD) * (i / 4));
    s.push('<line class="lieq-grid" x1="' + PLOT.l + '" y1="' + gy + '" x2="' + PLOT.r + '" y2="' + gy + '"/>');
  }

  /* ---- legend ---- */
  s.push('<line class="lieq-sw-t" x1="' + LEGEND.sw1 + '" y1="' + LEGEND.y + '" x2="' +
         (LEGEND.sw1 + LEGEND.swLen) + '" y2="' + LEGEND.y + '"/>');
  s.push('<text class="lieq-leg" x="' + LEGEND.t1 + '" y="' + LEGEND.baseline + '">' +
         esc(o.teacherLabel) + '</text>');
  s.push('<line class="lieq-sw-s" x1="' + LEGEND.sw2 + '" y1="' + LEGEND.y + '" x2="' +
         (LEGEND.sw2 + LEGEND.swLen) + '" y2="' + LEGEND.y + '"/>');
  s.push('<text class="lieq-leg" x="' + LEGEND.t2 + '" y="' + LEGEND.baseline + '">' +
         esc(o.studentLabel) + '</text>');

  /* ---- the two profiles, back to front: dense area + outline, then the
   *      dashed dense SHAPE the LF fader aims at, then the sparse scan on top
   *      with a translucent fill so the curve underneath still reads ---- */
  s.push('<g clip-path="url(#lieqClip' + uid + ')">');
  s.push('<path class="lieq-area" data-lieq="tArea" d="" fill="url(#lieqTArea' + uid + ')"/>');
  s.push('<path class="lieq-curve lieq-t-core" data-lieq="tCore" d=""/>');
  s.push('<path class="lieq-area" data-lieq="sArea" d="" fill="url(#lieqSArea' + uid + ')"/>');
  s.push('<path class="lieq-curve lieq-s-core" data-lieq="sCore" d=""/>');
  /* The dense scan's SHAPE, dashed: the reference overlay the paper figure puts
   * on the student panel. Both profiles share one pair of axes here, so this
   * sits ON TOP of them — underneath, it would vanish inside the dense scan's
   * own spiky curve, which is the line it runs through the middle of. */
  s.push('<path class="lieq-ghost" data-lieq="ghost" d="" opacity="0"/>');
  s.push('<circle class="lieq-armdot" data-lieq="armDot" cx="0" cy="0" r="4.2" opacity="0"/>');
  s.push('</g>');

  /* ---- callout 1: the SHAPE the sparse scan got wrong. A dimension bar
   *      between the two curves where their shapes differ most, and a label
   *      that stops claiming it once LF has closed the gap. ---- */
  s.push('<g class="lieq-ann" data-lieq="annShape">');
  s.push('<text class="lieq-ann-t" data-lieq="shapeT1" x="' + ANN_SHAPE.textX + '" y="' +
         ANN_SHAPE.y1 + '">' + esc(o.shapeAnnotation[0]) + '</text>');
  s.push('<text class="lieq-ann-t" data-lieq="shapeT2" x="' + ANN_SHAPE.textX + '" y="' +
         ANN_SHAPE.y2 + '">' + esc(o.shapeAnnotation[1]) + '</text>');
  s.push('<line class="lieq-ann-dash" data-lieq="shapeLead" x1="0" y1="0" x2="0" y2="0"/>');
  s.push('<line class="lieq-ann-dash" data-lieq="gapStem" x1="0" y1="0" x2="0" y2="0"/>');
  s.push('<line class="lieq-ann-line" data-lieq="gapTop" x1="0" y1="0" x2="0" y2="0"/>');
  s.push('<line class="lieq-ann-line" data-lieq="gapBot" x1="0" y1="0" x2="0" y2="0"/>');
  s.push('</g>');

  /* ---- callout 2: the DETAIL the sparse scan never recorded. A bracket over
   *      the stretch where the dense scan's texture is strongest. ---- */
  s.push('<g class="lieq-ann" data-lieq="annDetail">');
  s.push('<text class="lieq-ann-t" data-lieq="detailT1" x="' + ANN_DETAIL.textX + '" y="' +
         ANN_DETAIL.y1 + '" text-anchor="end">' + esc(o.detailAnnotation[0]) + '</text>');
  s.push('<text class="lieq-ann-t" data-lieq="detailT2" x="' + ANN_DETAIL.textX + '" y="' +
         ANN_DETAIL.y2 + '" text-anchor="end">' + esc(o.detailAnnotation[1]) + '</text>');
  s.push('<line class="lieq-ann-dash" data-lieq="detailLead" x1="0" y1="0" x2="0" y2="0"/>');
  s.push('<path class="lieq-ann-br" data-lieq="detailBr" d=""/>');
  s.push('</g>');

  /* ---- scrub cursor ---- */
  s.push('<g class="lieq-scrub" data-lieq="scrub">');
  s.push('<line class="lieq-scrub-line" data-lieq="scrubLine" x1="0" y1="' + PLOT.t +
         '" x2="0" y2="' + PLOT.b + '"/>');
  s.push('<circle class="lieq-scrub-t" data-lieq="scrubDotT" cx="0" cy="0" r="2.4"/>');
  s.push('<circle class="lieq-scrub-s" data-lieq="scrubDotS" cx="0" cy="0" r="2.4"/>');
  s.push('</g>');

  /* ---- axes. The x ticks carry NO numbers: this axis is a spatial
   *      coordinate with no units in the figure it comes from, and the old
   *      "X MAX" label pretended to be a coordinate it was not. ---- */
  s.push('<line class="lieq-axis" x1="' + PLOT.l + '" y1="' + PLOT.t + '" x2="' + PLOT.l + '" y2="' + PLOT.b + '"/>');
  s.push('<line class="lieq-axis" x1="' + PLOT.l + '" y1="' + PLOT.b + '" x2="' + PLOT.r + '" y2="' + PLOT.b + '"/>');
  for (i = 0; i <= 4; i++) {
    gx = n1(PLOT.l + (PLOT.r - PLOT.l) * (i / 4));
    s.push('<line class="lieq-tick" x1="' + gx + '" y1="' + PLOT.b + '" x2="' + gx + '" y2="' + (PLOT.b + 4) + '"/>');
  }
  if (o.lowTick) {
    s.push('<text class="lieq-tick-t" x="' + PLOT.l + '" y="' + ROW_TICK + '">' + esc(o.lowTick) + '</text>');
  }
  if (o.highTick) {
    s.push('<text class="lieq-tick-t" x="' + PLOT.r + '" y="' + ROW_TICK + '" text-anchor="end">' +
           esc(o.highTick) + '</text>');
  }
  s.push('<text class="lieq-axis-t" x="' + n1((PLOT.l + PLOT.r) / 2) + '" y="' + ROW_TITLE +
         '" text-anchor="middle">' + esc(o.xLabel) + '</text>');
  s.push('<text class="lieq-axis-t" transform="translate(16,' + n1((PLOT.t + PLOT.b) / 2) +
         ') rotate(-90)" text-anchor="middle">' + esc(o.yLabel) + '</text>');

  /* ---- live readout ---- */
  s.push('<text x="' + PLOT.r + '" y="' + ROW_READOUT + '" text-anchor="end">' +
         '<tspan class="lieq-ro-k" data-lieq="roLabel">' + esc(o.readoutLabel) + ' </tspan>' +
         '<tspan class="lieq-ro-v" data-lieq="gapValue">0.00</tspan></text>');

  /* ---- end of the static decoration; everything below must stay reachable ---- */
  s.push('</g>');

  /* ---- the hint sits above the ink but below the hit surface ---- */
  s.push(hintMarkup(o.hint));

  /* ---- ONE hit surface over the whole plot. It is the only thing in the plot
   *      that takes a pointer event, and it plucks the nearest curve. ---- */
  s.push('<rect class="lieq-hitfield" data-lieq="hit" x="' + (PLOT.l - 2) + '" y="' + (PLOT.t - 8) +
         '" width="' + (PLOT.r - PLOT.l + 4) + '" height="' + (PLOT.b - PLOT.t + 12) + '"/>');

  /* ---- preset pills ---- */
  s.push(presetMarkup(o.presets));

  /* ---- rack ---- */
  s.push('<rect class="lieq-rack-bg" x="' + RACK.x + '" y="' + RACK.y + '" width="' + RACK.w +
         '" height="' + RACK.h + '" rx="14" fill="url(#lieqRack' + uid +
         ')" stroke="rgba(134,182,239,.17)" stroke-width="1"/>');
  s.push('<text class="lieq-rack-eyebrow" x="' + n1(RACK.x + RACK.w / 2) + '" y="38" text-anchor="middle">' +
         esc(o.rackLabel) + '</text>');
  s.push(faderMarkup(uid, 'lf', FADER_X.lf, 'LF', o.lfSub, o.lfAria));
  s.push(faderMarkup(uid, 'hf', FADER_X.hf, 'HF', o.hfSub, o.hfAria));

  s.push('</svg>');
  return s.join('');
}

/* ==============================================================
 * 7.  The widget
 * ============================================================== */

var uidSeq = 0;

var PRESETS = [
  { label: 'BYPASS', lf: 0, hf: 0,
    aria: 'Preset: bypass — both equalizers at zero, the undistilled sparse student' },
  { label: 'LF',     lf: 1, hf: 0,
    aria: 'Preset: LF equalizer only — pull the sparse scan’s shape onto the dense scan’s, still smooth' },
  { label: 'LF+HF',  lf: 1, hf: 1,
    aria: 'Preset: LF and HF equalizers — the shape, and the fine detail that goes on it' }
];

var DEFAULTS = {
  lf: 0,
  hf: 0,
  chrome: false,                       /* true = also paint the dark card shell */
  residual: FULL_SCALE_RESIDUAL,
  samples: N_SAMPLES,
  /* only used when a host passes its own `teacher` / `student` curves: they are
   * split into envelope + ripple with this box blur */
  smoothRadius: 3,
  smoothPasses: 3,
  /* The source figure's own axis is SPATIAL — its x is literally labelled
   * "Spatial position". LF and HF name what the two faders act on, the shape
   * and the fine detail; they are not axes, and no REGION of either curve is
   * ever labelled with a frequency. */
  xLabel: 'Spatial position',
  yLabel: 'Signal / feature response',
  lowTick: '',                         /* no fake coordinates on this axis */
  highTick: '',
  teacherLabel: 'DENSE TEACHER',
  studentLabel: 'SPARSE STUDENT',
  rackLabel: 'SPECTRAL EQ',
  readoutLabel: 'PEAK GAP',
  scrubLabel: 'GAP HERE',
  hint: 'DRAG EITHER CURVE',
  /* Plain spatial statements, and each one stops being true at full travel, so
   * each has a second wording the widget swaps in. */
  shapeAnnotation: ['SHAPE THE SPARSE', 'SCAN GOT WRONG'],
  shapeAnnotationDone: ['SHAPE BROUGHT', 'BACK IN LINE'],
  detailAnnotation: ['DETAIL THE SPARSE SCAN', 'NEVER RECORDED'],
  detailAnnotationDone: ['DETAIL BACK FROM', 'THE DENSE SCAN'],
  lfSub: 'SHAPE',
  hfSub: 'DETAIL',
  lfAria: 'LF equalizer strength, the overall shape of the profile',
  hfAria: 'HF equalizer strength, the fine detail riding on that shape',
  ariaLabel: 'Spectral equalizer: two faders morph the sparse student’s spatial profile toward the dense teacher’s.',
  desc: 'A plot of two signal profiles against spatial position. The blue curve is the dense ' +
        'teacher: one dominant hump carrying dense fine detail. The orange curve is the sparse ' +
        'student: at rest its overall shape is wrong — two humps where the dense scan has one — ' +
        'and it is almost perfectly smooth, because sparse sampling never recorded the fine ' +
        'detail. The LF fader pulls the student’s shape onto the teacher’s and leaves it smooth; ' +
        'the HF fader grows the fine detail until the student’s texture approaches the teacher’s. ' +
        'Height on this plot is a signal value, not a frequency. With both faders at full the ' +
        /* interpolated, not spelled out: this sentence said "four per cent" for as long as
         * FULL_SCALE_RESIDUAL was 0.04 and then went on saying it after the constant moved to
         * 0.075, which is a screen reader being told the wrong number. */
        'student comes close to the teacher but does not reach it — a residual of ' +
        Math.round(100 * FULL_SCALE_RESIDUAL) + ' per cent ' +
        'of the signal range is left in its shape, and its fine detail comes back shallower than ' +
        'the dense scan’s everywhere, as in the paper’s own measurements. Either curve can also ' +
        'be dragged to deform it; the nearer curve to the pointer is the one that moves, it ' +
        'springs back when released, and dragging a curve never changes the faders.',
  caption: CAPTION,
  presets: PRESETS,
  teacher: null,
  student: null,
  onChange: null
};

function extend(dst, src) {
  var k;
  if (!src) { return dst; }
  for (k in src) {
    if (Object.prototype.hasOwnProperty.call(src, k) && src[k] !== undefined) { dst[k] = src[k]; }
  }
  return dst;
}

function normLines(v, fallback) {
  if (typeof v === 'string') { return [v, '']; }
  if (v && v.length) { return [v[0] == null ? '' : v[0], v[1] == null ? '' : v[1]]; }
  return fallback;
}

function normOpts(o) {
  o.shapeAnnotation = normLines(o.shapeAnnotation, DEFAULTS.shapeAnnotation);
  o.shapeAnnotationDone = normLines(o.shapeAnnotationDone, DEFAULTS.shapeAnnotationDone);
  o.detailAnnotation = normLines(o.detailAnnotation, DEFAULTS.detailAnnotation);
  o.detailAnnotationDone = normLines(o.detailAnnotationDone, DEFAULTS.detailAnnotationDone);
  o.hint = typeof o.hint === 'string' ? o.hint : (o.hint && o.hint.length ? String(o.hint[0]) : '');
  return o;
}

function resolve(target, doc) {
  if (!target) { return null; }
  if (typeof target === 'string') { return doc ? doc.querySelector(target) : null; }
  if (target.nodeType === 1) { return target; }
  return null;
}

function Instance(el, opts) {
  var o = normOpts(extend(extend({}, DEFAULTS), opts)),
      doc = el.ownerDocument,
      uid = String(++uidSeq),
      model, t, s, ts, ss, n, i;

  if (o.teacher && o.student) {
    /* a host brought its own curves: split each into shape + detail */
    t = parseCurve(o.teacher);
    s = parseCurve(o.student);
    n = Math.min(t.length, s.length);
    if (n < 4) { throw new Error('LiEQEqualizer: need at least 4 samples'); }
    t = t.slice(0, n); s = s.slice(0, n);
    ts = split(t, o.smoothRadius, o.smoothPasses);
    ss = split(s, o.smoothRadius, o.smoothPasses);
    model = {
      n: n,
      teacherEnvelope: ts.lo, teacherRipple: ts.hi,
      studentEnvelope: ss.lo, studentRipple: ss.hi,
      teacher: t, student: s
    };
  } else {
    model = buildModel({ n: o.samples });
    n = model.n;
    if (n < 4) { throw new Error('LiEQEqualizer: need at least 4 samples'); }
  }

  this.opts = o;
  this.caption = o.caption;
  this.element = el;
  this.doc = doc;
  this.uid = uid;
  this.n = n;
  this.model = model;
  this.teacher = model.teacher;
  this.student = model.student;
  this.tEnv = model.teacherEnvelope; this.tRip = model.teacherRipple;
  this.sEnv = model.studentEnvelope; this.sRip = model.studentRipple;
  /* how much of the gap a full fader closes — the rest is FULL_SCALE_RESIDUAL */
  this.reach = reach(this.student, this.teacher, o.residual);
  /* the detail channel stops further short — see DETAIL_REACH */
  this.detailReach = clamp(o.detailReach === undefined ? DETAIL_REACH : +o.detailReach, 0, 1);
  this.state = { lf: clamp(+o.lf || 0, 0, 1), hf: clamp(+o.hf || 0, 0, 1) };
  this.drag = null;
  this.cdrag = null;
  this.pluck = new PluckState({ span: Math.max(2, PLUCK_SPAN_FRAC * (n - 1)) });
  this.plucked = false;
  this.hintRetired = false;
  this.arm = null;
  this.armAt = -1;
  this.scrubAt = -1;
  this.frame = 0;
  this.springFrame = 0;
  this.destroyed = false;
  this.lastShapeDone = null;
  this.lastDetailDone = null;

  /* --- vertical scale, from the four corners of the fader square --- */
  this.vmax = modelVmax(model, this.reach, this.detailReach);
  this.vmin = 0;
  this.span = (PLOT.b - PLOT.t - PLOT_HEAD) / this.vmax;

  /* --- fixed x grid --- */
  this.xs = new Array(n);
  for (i = 0; i < n; i++) { this.xs[i] = PLOT.l + (PLOT.r - PLOT.l) * (i / (n - 1)); }
  this.ys = new Array(n);
  this.tys = new Array(n);

  /* --- where the two callouts point --- */
  this.annShapeI = shapeDepartureIndex(this.sEnv, this.tEnv);
  this.detailFrom = clamp(Math.round(ANN_DETAIL_FROM * (n - 1)), 0, n - 1);
  this.detailTo = clamp(Math.round(ANN_DETAIL_TO * (n - 1)), 0, n - 1);

  /* --- the dashed reference: the dense scan's SHAPE, which never moves --- */
  this.ghostD = catmullPath(this.xs, (function (self) {
    var out = new Array(n), j;
    for (j = 0; j < n; j++) { out[j] = self.y(self.tEnv[j]); }
    return out;
  })(this));

  this.build();
  this.bind();
  this.render();
}

/* The two travel limits are declared HERE as well as assigned in the
 * constructor, because morph() reads them off `this` and the constructor is not
 * the only thing that builds an object on this prototype. detailReach arrived
 * with DETAIL_REACH while the other builder — the harness's stub instance —
 * still set `reach` alone, and `hf * undefined` is NaN even at hf = 0: all 181
 * student samples, every screen y derived from them, the SHAPE dimension bar,
 * the DETAIL bracket, both scrub dots and the PEAK GAP readout went NaN (the
 * readout printed 0.00, because `NaN > gap` is false) while the model itself
 * was clean. 14 assertions failed on one missing field. A declared default
 * means reading either limit can never yield undefined, whoever built the
 * object; the constructor still overwrites both. */
Instance.prototype.reach = 1;
Instance.prototype.detailReach = DETAIL_REACH;

/* The pose the faders imply. lf travels over `reach` of the path and hf over
 * the shorter `detailReach`, which leaves `residual` between the full-travel
 * student's SHAPE and the teacher's and more than that in its DETAIL. */
Instance.prototype.morph = function (lf, hf) {
  return morph(this.sEnv, this.sRip, this.tEnv, this.tRip,
               lf * this.reach, hf * this.detailReach);
};

/* The student's ENVELOPE alone at a given lf — what the dashed reference is
 * compared against, and what the SHAPE callout measures. */
Instance.prototype.envAt = function (lf) {
  var n = this.n, out = new Array(n), t = lf * this.reach, i;
  for (i = 0; i < n; i++) { out[i] = lerp(this.sEnv[i], this.tEnv[i], t); }
  return out;
};

/* value -> y, clamped to the plot box so a hard pluck saturates against the
 * frame instead of escaping it. */
Instance.prototype.y = function (v) {
  return clamp(PLOT.b - v * this.span, PLOT.t - 4, PLOT.b);
};

/* y -> value, the inverse without the clamp (used while plucking) */
Instance.prototype.vy = function (y) {
  return (PLOT.b - y) / this.span;
};

Instance.prototype.q = function (name) {
  return this.svg.querySelector('[data-lieq="' + name + '"]');
};

Instance.prototype.now = function () {
  var w = this.win;
  if (w && w.performance && w.performance.now) { return w.performance.now(); }
  return Date.now();
};

Instance.prototype.reducedMotion = function () {
  var w = this.win;
  try {
    return !!(w && w.matchMedia && w.matchMedia('(prefers-reduced-motion: reduce)').matches);
  } catch (err) { return false; }
};

Instance.prototype.build = function () {
  var doc = this.doc, o = this.opts, wrap, i, faders, chips;

  injectCSS(doc);

  wrap = doc.createElement('div');
  wrap.className = o.chrome ? 'lieq-card' : 'lieq-shell';
  wrap.innerHTML =
    '<div class="lieq-stage"><i class="lieq-scan" aria-hidden="true"></i>' +
    svgMarkup(this.uid, o) + '</div>';

  this.element.innerHTML = '';
  this.element.appendChild(wrap);
  this.element.setAttribute('data-lieq-ready', '1');

  this.wrap = wrap;
  this.stage = wrap.querySelector('.lieq-stage');
  this.svg = wrap.querySelector('svg');

  this.el = {
    tArea: this.q('tArea'), tCore: this.q('tCore'), ghost: this.q('ghost'),
    sArea: this.q('sArea'), sCore: this.q('sCore'), armDot: this.q('armDot'),
    hit: this.q('hit'), hintg: this.q('hintg'),
    annShape: this.q('annShape'), annDetail: this.q('annDetail'),
    shapeT1: this.q('shapeT1'), shapeT2: this.q('shapeT2'), shapeLead: this.q('shapeLead'),
    gapStem: this.q('gapStem'), gapTop: this.q('gapTop'), gapBot: this.q('gapBot'),
    detailT1: this.q('detailT1'), detailT2: this.q('detailT2'),
    detailLead: this.q('detailLead'), detailBr: this.q('detailBr'),
    gapValue: this.q('gapValue'), roLabel: this.q('roLabel'),
    scrub: this.q('scrub'), scrubLine: this.q('scrubLine'),
    scrubDotS: this.q('scrubDotS'), scrubDotT: this.q('scrubDotT')
  };

  if (this.el.ghost) { this.el.ghost.setAttribute('d', this.ghostD); }

  this.faders = {};
  faders = this.svg.querySelectorAll('[data-lieq-fader]');
  for (i = 0; i < faders.length; i++) {
    this.faders[faders[i].getAttribute('data-lieq-fader')] = {
      group: faders[i],
      cap: faders[i].querySelector('.lieq-capg'),
      fill: faders[i].querySelector('.lieq-fill'),
      val: faders[i].querySelector('.lieq-val')
    };
  }

  this.chips = [];
  chips = this.svg.querySelectorAll('[data-lieq-preset]');
  for (i = 0; i < chips.length; i++) { this.chips.push(chips[i]); }
};

/* ---- pointer / keyboard ---------------------------------------------- */

Instance.prototype.toSvg = function (e) {
  var svg = this.svg, m, p, r;
  if (svg.getScreenCTM && svg.createSVGPoint) {
    m = svg.getScreenCTM();
    if (m) {
      p = svg.createSVGPoint();
      p.x = e.clientX; p.y = e.clientY;
      p = p.matrixTransform(m.inverse());
      return { x: p.x, y: p.y };
    }
  }
  r = svg.getBoundingClientRect();
  if (!r.width || !r.height) { return { x: 0, y: 0 }; }
  return {
    x: (e.clientX - r.left) / r.width * VB_W,
    y: (e.clientY - r.top) / r.height * VB_H
  };
};

function isKnob(node, stop) {
  while (node && node !== stop) {
    if (node.getAttribute && node.getAttribute('data-lieq-knob')) { return true; }
    node = node.parentNode;
  }
  return false;
}

Instance.prototype.valueFromY = function (y) {
  return clamp((TRACK.bottom - y) / (TRACK.bottom - TRACK.top), 0, 1);
};

/* nearest sample index to an x in viewBox units */
Instance.prototype.indexAt = function (x) {
  var n = this.n, f = (x - PLOT.l) / (PLOT.r - PLOT.l);
  return clamp(Math.round(f * (n - 1)), 0, n - 1);
};

/* A fingertip has no pixel, so it gets a bigger radius than a mouse. */
Instance.prototype.grabRadius = function (e) {
  return e && (e.pointerType === 'touch' || e.pointerType === 'pen')
    ? GRAB_RADIUS_TOUCH : GRAB_RADIUS;
};

Instance.prototype.bind = function () {
  var self = this, key, f, i, win;

  this.onDown = function (e) { self.pointerDown(e); };
  this.onMove = function (e) { self.pointerMove(e); };
  this.onUp = function (e) { self.pointerUp(e); };
  this.onKey = function (e) { self.keyDown(e); };
  this.onHitDown = function (e) { self.hitDown(e); };
  this.onStageMove = function (e) { self.stageMove(e); };
  this.onStageLeave = function () { self.stageLeave(); };
  this.onChipDown = function (e) { self.chipActivate(e); };
  this.onChipKey = function (e) { self.chipKey(e); };

  for (key in this.faders) {
    if (!Object.prototype.hasOwnProperty.call(this.faders, key)) { continue; }
    f = this.faders[key];
    f.group.addEventListener('pointerdown', this.onDown, false);
    f.group.addEventListener('keydown', this.onKey, false);
  }

  if (this.el.hit) {
    this.el.hit.addEventListener('pointerdown', this.onHitDown, false);
  }

  for (i = 0; i < this.chips.length; i++) {
    this.chips[i].addEventListener('click', this.onChipDown, false);
    this.chips[i].addEventListener('keydown', this.onChipKey, false);
  }

  if (this.svg) {
    this.svg.addEventListener('pointermove', this.onStageMove, false);
    this.svg.addEventListener('pointerleave', this.onStageLeave, false);
  }

  /* Move / up / cancel live on the window so a drag survives the pointer
   * leaving the SVG even if setPointerCapture is unavailable. */
  win = this.doc.defaultView || (typeof window !== 'undefined' ? window : null);
  this.win = win;
  if (win) {
    win.addEventListener('pointermove', this.onMove, false);
    win.addEventListener('pointerup', this.onUp, false);
    win.addEventListener('pointercancel', this.onUp, false);
  }
};

/* ---- faders: continuous, and they STAY where you leave them ---------- */

Instance.prototype.pointerDown = function (e) {
  var group = e.currentTarget,
      key = group.getAttribute('data-lieq-fader'),
      p, knobY;
  if (this.drag || this.cdrag) { return; }
  if (e.button !== undefined && e.button !== 0 && e.pointerType === 'mouse') { return; }
  e.preventDefault();
  e.stopPropagation();

  p = this.toSvg(e);
  knobY = TRACK.bottom - this.state[key] * (TRACK.bottom - TRACK.top);

  this.drag = { key: key, id: e.pointerId, group: group, dy: 0 };
  if (isKnob(e.target, group)) {
    /* grab offset, so the cap does not jump under the finger */
    this.drag.dy = knobY - p.y;
  } else {
    /* a press on the track jumps the cap there */
    this.setValue(key, this.valueFromY(p.y));
  }

  try { group.setPointerCapture(e.pointerId); } catch (err) { /* not supported */ }
  group.classList.add('lieq-dragging');
  try { group.focus({ preventScroll: true }); } catch (err2) {
    try { group.focus(); } catch (err3) { /* ignore */ }
  }
};

Instance.prototype.pointerMove = function (e) {
  var d = this.drag, c = this.cdrag;
  if (c) {
    if (c.id !== undefined && e.pointerId !== undefined && e.pointerId !== c.id) { return; }
    if (e.cancelable) { e.preventDefault(); }
    this.pluckTo(this.vy(this.toSvg(e).y));
    return;
  }
  if (!d) { return; }
  if (d.id !== undefined && e.pointerId !== undefined && e.pointerId !== d.id) { return; }
  if (e.cancelable) { e.preventDefault(); }
  this.setValue(d.key, this.valueFromY(this.toSvg(e).y + d.dy));
};

Instance.prototype.pointerUp = function (e) {
  var d = this.drag, c = this.cdrag;
  if (c) {
    if (c.id !== undefined && e.pointerId !== undefined && e.pointerId !== c.id) { return; }
    try { c.node.releasePointerCapture(c.id); } catch (err) { /* ignore */ }
    this.cdrag = null;
    if (this.stage) { this.stage.removeAttribute('data-lieq-pluck'); }
    this.pluckRelease(this.now());
    return;
  }
  if (!d) { return; }
  if (d.id !== undefined && e.pointerId !== undefined && e.pointerId !== d.id) { return; }
  try { d.group.releasePointerCapture(d.id); } catch (err2) { /* ignore */ }
  d.group.classList.remove('lieq-dragging');
  this.drag = null;
  /* No spring-back, no reset, no settle animation. The value is the state. */
};

Instance.prototype.keyDown = function (e) {
  var group = e.currentTarget,
      key = group.getAttribute('data-lieq-fader'),
      k = e.key, v = this.state[key], next = v,
      step = e.shiftKey ? 0.2 : 0.05;
  if (k === 'ArrowUp' || k === 'Up' || k === 'ArrowRight' || k === 'Right') { next = v + step; }
  else if (k === 'ArrowDown' || k === 'Down' || k === 'ArrowLeft' || k === 'Left') { next = v - step; }
  else if (k === 'PageUp') { next = v + 0.2; }
  else if (k === 'PageDown') { next = v - 0.2; }
  else if (k === 'Home') { next = 0; }
  else if (k === 'End') { next = 1; }
  else { return; }
  e.preventDefault();
  this.setValue(key, Math.round(next * 1000) / 1000);
};

/* ---- the curves: pluck them, and they spring back ------------------- */

/* All three of these are DOM-free on purpose: they are the whole contract
 * between a pointer and the cord, and NONE of them reads or writes
 * this.state, so a pluck can never move a fader. */
Instance.prototype.pluckBegin = function (which, i0, base) {
  if (this.destroyed) { return this; }
  this.pluck.begin(which, i0, base);
  this.plucked = true;
  return this;
};

Instance.prototype.pluckTo = function (value) {
  if (this.destroyed || this.pluck.settled()) { return this; }
  this.pluck.to(value);
  this.schedule();
  return this;
};

Instance.prototype.pluckRelease = function (now) {
  if (this.destroyed || this.pluck.settled()) { return this; }
  if (this.reducedMotion()) { this.pluck.cancel(); this.schedule(); return this; }
  this.pluck.release(now);
  this.startSpring();
  return this;
};

/* The hint has done its job the first time anyone plucks, and it does not come
 * back. */
Instance.prototype.retireHint = function () {
  if (this.hintRetired) { return; }
  this.hintRetired = true;
  if (this.stage) { this.stage.setAttribute('data-lieq-plucked', '1'); }
  if (this.el.hintg) { this.el.hintg.setAttribute('opacity', '0'); }
};

/* Which curve the pointer is over, i.e. which one a press would grab. The
 * highlight exists so that answer is never a surprise. */
Instance.prototype.setArm = function (which, i) {
  var changed = this.arm !== which || this.armAt !== i;
  this.arm = which || null;
  this.armAt = which ? i : -1;
  if (this.stage) {
    if (which) {
      this.stage.setAttribute('data-lieq-hot', which);
      this.stage.setAttribute('data-lieq-arm', which);
    } else {
      this.stage.removeAttribute('data-lieq-hot');
      this.stage.removeAttribute('data-lieq-arm');
    }
  }
  if (changed) { this.schedule(); }
  return this;
};

/* ONE hit surface, nearest curve wins. No stacked bands, so no curve can hide
 * another's hit area — the bug this replaces. */
Instance.prototype.hitDown = function (e) {
  var node = e.currentTarget, p, i, pick, base;
  if (this.drag || this.cdrag) { return; }
  if (e.button !== undefined && e.button !== 0 && e.pointerType === 'mouse') { return; }

  p = this.toSvg(e);
  i = this.indexAt(p.x);
  pick = pickCurve(i, p.y, this.tys, this.ys, this.grabRadius(e), GRAB_WINDOW, this.xs, p.x);
  if (!pick) { return; }            /* a press on empty plot is not a pluck */

  e.preventDefault();
  e.stopPropagation();

  base = pick.which === 't' ? this.tvals[i] : this.svals[i];
  this.cdrag = { id: e.pointerId, node: node, which: pick.which };
  this.setArm(pick.which, i);
  this.pluckBegin(pick.which, i, base);
  try { node.setPointerCapture(e.pointerId); } catch (err) { /* not supported */ }
  if (this.stage) { this.stage.setAttribute('data-lieq-pluck', '1'); }
  this.retireHint();
  this.schedule();
};

/* the spring loop: time-based, so a dropped frame changes nothing */
Instance.prototype.startSpring = function () {
  var self = this, win = this.win, raf = win && win.requestAnimationFrame, step;
  if (!raf) { this.pluck.cancel(); this.schedule(); return; }
  if (this.springFrame) { return; }
  step = function () {
    self.springFrame = 0;
    if (self.destroyed) { return; }
    self.render();
    if (!self.pluck.settled()) { self.springFrame = raf.call(win, step); }
  };
  this.springFrame = raf.call(win, step);
};

/* ---- scrub cursor + the grab highlight ------------------------------ */

Instance.prototype.stageMove = function (e) {
  var p = this.toSvg(e), i, pick;
  if (this.drag) { return; }
  if (p.x < PLOT.l - 2 || p.x > PLOT.r + 2 || p.y < PLOT.t - 10 || p.y > PLOT.b + 6) {
    this.setArm(null, -1);
    if (this.scrubAt >= 0) { this.scrubAt = -1; this.schedule(); }
    return;
  }
  i = this.indexAt(p.x);
  if (i !== this.scrubAt) { this.scrubAt = i; this.schedule(); }
  if (this.cdrag) { return; }        /* a live drag owns the highlight */
  pick = pickCurve(i, p.y, this.tys, this.ys, this.grabRadius(e), GRAB_WINDOW, this.xs, p.x);
  this.setArm(pick ? pick.which : null, i);
};

Instance.prototype.stageLeave = function () {
  if (this.scrubAt >= 0) { this.scrubAt = -1; this.schedule(); }
  if (!this.cdrag) { this.setArm(null, -1); }
};

/* ---- presets: they jump, and like the faders they never rebound ----- */

Instance.prototype.chipActivate = function (e) {
  var g = e.currentTarget;
  e.preventDefault();
  this.setValues(parseFloat(g.getAttribute('data-lieq-lf')),
                 parseFloat(g.getAttribute('data-lieq-hf')));
  try { g.focus({ preventScroll: true }); } catch (err) {
    try { g.focus(); } catch (err2) { /* ignore */ }
  }
};

Instance.prototype.chipKey = function (e) {
  if (e.key === 'Enter' || e.key === ' ' || e.key === 'Spacebar') { this.chipActivate(e); }
};

/* ---- state ------------------------------------------------------------ */

Instance.prototype.setValue = function (key, v) {
  if (this.destroyed || (key !== 'lf' && key !== 'hf')) { return this; }
  v = clamp(+v || 0, 0, 1);
  if (v === this.state[key]) { return this; }
  this.state[key] = v;
  this.schedule();
  return this;
};

Instance.prototype.setValues = function (lf, hf) {
  var changed = false, v;
  if (lf !== undefined && lf !== null && lf === lf) {
    v = clamp(+lf || 0, 0, 1);
    if (v !== this.state.lf) { this.state.lf = v; changed = true; }
  }
  if (hf !== undefined && hf !== null && hf === hf) {
    v = clamp(+hf || 0, 0, 1);
    if (v !== this.state.hf) { this.state.hf = v; changed = true; }
  }
  if (changed) { this.schedule(); }
  return this;
};

Instance.prototype.getValues = function () {
  return { lf: this.state.lf, hf: this.state.hf };
};

Instance.prototype.schedule = function () {
  var self = this, raf = this.win && this.win.requestAnimationFrame;
  if (this.destroyed) { return; }
  if (!raf) { this.render(); return; }
  if (this.frame) { return; }
  this.frame = raf.call(this.win, function () {
    self.frame = 0;
    if (!self.destroyed) { self.render(); }
  });
};

/* ---- render ----------------------------------------------------------- */

function setLine(el, x1, y1, x2, y2) {
  if (!el) { return; }
  el.setAttribute('x1', n1(x1)); el.setAttribute('y1', n1(y1));
  el.setAttribute('x2', n1(x2)); el.setAttribute('y2', n1(y2));
}

Instance.prototype.render = function () {
  var st = this.state, n = this.n, el = this.el,
      amp = this.pluck.amp(this.now()),
      vals, tvals, i, d, td, gap, dv, ax, tY, sY, key, f, knobY, want,
      done, lines, top, brY, cx, x1, x2;

  /* the two shapes: the faders' pose, plus whatever the cord is doing */
  vals = this.pluck.apply(this.morph(st.lf, st.hf), 's', amp);
  tvals = this.pluck.apply(this.teacher.slice(0), 't', amp);
  this.svals = vals;
  this.tvals = tvals;

  for (i = 0; i < n; i++) { this.ys[i] = this.y(vals[i]); this.tys[i] = this.y(tvals[i]); }
  d = linePath(this.xs, this.ys);
  td = linePath(this.xs, this.tys);

  if (el.tCore) { el.tCore.setAttribute('d', td); }
  if (el.tArea) { el.tArea.setAttribute('d', areaPath(td, this.xs[0], this.xs[n - 1], PLOT.b)); }
  if (el.sCore) { el.sCore.setAttribute('d', d); }
  if (el.sArea) { el.sArea.setAttribute('d', areaPath(d, this.xs[0], this.xs[n - 1], PLOT.b)); }

  /* the dashed dense SHAPE is the LF fader's target, so it retires as LF
   * arrives — exactly the reference overlay the paper figure draws */
  if (el.ghost) { el.ghost.setAttribute('opacity', n1(0.72 * clamp(1 - st.lf * 1.18, 0, 1)) + ''); }

  /* the grab highlight's dot, on whichever curve a press would take */
  if (el.armDot) {
    if (this.arm && this.armAt >= 0) {
      el.armDot.setAttribute('cx', n1(this.xs[this.armAt]));
      el.armDot.setAttribute('cy', n1(this.arm === 't' ? this.tys[this.armAt] : this.ys[this.armAt]));
      el.armDot.setAttribute('stroke', this.arm === 't' ? '#bfe0ff' : '#f6bd8c');
      el.armDot.setAttribute('opacity', '0.95');
    } else {
      el.armDot.setAttribute('opacity', '0');
    }
  }

  /* readout — either the peak gap over the whole plot, or the local gap under
   * the scrub cursor. Both are properties of the two drawn curves, nothing more. */
  if (this.scrubAt >= 0) {
    i = this.scrubAt;
    gap = Math.abs(vals[i] - tvals[i]);
    want = this.opts.scrubLabel;
  } else {
    gap = 0;
    for (i = 0; i < n; i++) {
      dv = vals[i] - tvals[i];
      if (dv < 0) { dv = -dv; }
      if (dv > gap) { gap = dv; }
    }
    want = this.opts.readoutLabel;
  }
  if (el.gapValue) { el.gapValue.textContent = (Math.round(gap * 100) / 100).toFixed(2); }
  if (el.roLabel && el.roLabel.textContent !== want + ' ') { el.roLabel.textContent = want + ' '; }

  /* scrub cursor */
  if (el.scrub) {
    if (this.scrubAt >= 0) {
      i = this.scrubAt; ax = this.xs[i];
      el.scrub.setAttribute('opacity', '1');
      if (el.scrubLine) {
        el.scrubLine.setAttribute('x1', n1(ax)); el.scrubLine.setAttribute('x2', n1(ax));
      }
      if (el.scrubDotS) {
        el.scrubDotS.setAttribute('cx', n1(ax)); el.scrubDotS.setAttribute('cy', n1(this.ys[i]));
      }
      if (el.scrubDotT) {
        el.scrubDotT.setAttribute('cx', n1(ax)); el.scrubDotT.setAttribute('cy', n1(this.tys[i]));
      }
    } else {
      el.scrub.setAttribute('opacity', '0');
    }
  }

  /* ---- callout 1: the SHAPE. A dimension bar between the two curves where
   *      their shapes differ most. As LF closes the gap the bar shrinks to
   *      nothing, so the label stops claiming a shape error and says so. ---- */
  i = this.annShapeI;
  ax = this.xs[i]; sY = this.ys[i]; tY = this.tys[i];
  done = st.lf >= ANN_DONE_AT;
  if (done !== this.lastShapeDone) {
    this.lastShapeDone = done;
    lines = done ? this.opts.shapeAnnotationDone : this.opts.shapeAnnotation;
    if (el.shapeT1) { el.shapeT1.textContent = lines[0]; }
    if (el.shapeT2) { el.shapeT2.textContent = lines[1]; }
  }
  if (el.annShape) { el.annShape.setAttribute('opacity', n1(0.45 + 0.55 * (1 - st.lf)) + ''); }
  setLine(el.gapStem, ax, sY, ax, tY);
  setLine(el.gapTop, ax - ANN_SHAPE.cap, sY, ax + ANN_SHAPE.cap, sY);
  setLine(el.gapBot, ax - ANN_SHAPE.cap, tY, ax + ANN_SHAPE.cap, tY);
  top = Math.min(sY, tY) - 4;
  setLine(el.shapeLead, ax, ANN_SHAPE.leadTop, ax,
          top > ANN_SHAPE.leadTop ? top : ANN_SHAPE.leadTop);

  /* ---- callout 2: the DETAIL. A bracket over the stretch where the dense
   *      scan's texture is strongest; HF is what fills it in. ---- */
  x1 = this.xs[this.detailFrom]; x2 = this.xs[this.detailTo];
  brY = Infinity;
  for (i = this.detailFrom; i <= this.detailTo; i++) {
    if (this.ys[i] < brY) { brY = this.ys[i]; }
    if (this.tys[i] < brY) { brY = this.tys[i]; }
  }
  brY -= 9;
  if (!(brY > ANN_DETAIL.minY)) { brY = ANN_DETAIL.minY; }
  done = st.hf >= ANN_DONE_AT;
  if (done !== this.lastDetailDone) {
    this.lastDetailDone = done;
    lines = done ? this.opts.detailAnnotationDone : this.opts.detailAnnotation;
    if (el.detailT1) { el.detailT1.textContent = lines[0]; }
    if (el.detailT2) { el.detailT2.textContent = lines[1]; }
  }
  if (el.annDetail) { el.annDetail.setAttribute('opacity', n1(0.45 + 0.55 * (1 - st.hf)) + ''); }
  if (el.detailBr) { el.detailBr.setAttribute('d', bracketPath(x1, x2, brY, ANN_DETAIL.drop)); }
  cx = (x1 + x2) / 2;
  setLine(el.detailLead, cx, ANN_DETAIL.leadTop, cx,
          brY - 3 > ANN_DETAIL.leadTop ? brY - 3 : ANN_DETAIL.leadTop);

  /* faders */
  for (key in this.faders) {
    if (!Object.prototype.hasOwnProperty.call(this.faders, key)) { continue; }
    f = this.faders[key];
    knobY = TRACK.half - st[key] * (TRACK.half * 2);
    if (f.cap) { f.cap.setAttribute('transform', 'translate(0,' + n1(knobY) + ')'); }
    if (f.fill) {
      f.fill.setAttribute('y', n1(knobY));
      f.fill.setAttribute('height', n1(TRACK.half - knobY));
    }
    if (f.val) { f.val.textContent = st[key].toFixed(2); }
    f.group.setAttribute('aria-valuenow', String(Math.round(st[key] * 100)));
    f.group.setAttribute('aria-valuetext', Math.round(st[key] * 100) + '%');
  }

  /* presets light up when the faders happen to be sitting on them */
  for (i = 0; i < this.chips.length; i++) {
    f = this.chips[i];
    f.setAttribute('aria-pressed',
      (Math.abs(st.lf - parseFloat(f.getAttribute('data-lieq-lf'))) < 1e-6 &&
       Math.abs(st.hf - parseFloat(f.getAttribute('data-lieq-hf'))) < 1e-6) ? 'true' : 'false');
  }

  if (typeof this.opts.onChange === 'function') {
    try { this.opts.onChange(st.lf, st.hf, this); } catch (err) { /* host's problem */ }
  }
  return this;
};

Instance.prototype.destroy = function () {
  var key, f, i;
  if (this.destroyed) { return; }
  this.destroyed = true;
  if (this.win && this.win.cancelAnimationFrame) {
    if (this.frame) { this.win.cancelAnimationFrame(this.frame); }
    if (this.springFrame) { this.win.cancelAnimationFrame(this.springFrame); }
  }
  this.frame = 0;
  this.springFrame = 0;
  this.pluck.cancel();
  for (key in this.faders) {
    if (!Object.prototype.hasOwnProperty.call(this.faders, key)) { continue; }
    f = this.faders[key];
    f.group.removeEventListener('pointerdown', this.onDown, false);
    f.group.removeEventListener('keydown', this.onKey, false);
  }
  if (this.el && this.el.hit) {
    this.el.hit.removeEventListener('pointerdown', this.onHitDown, false);
  }
  for (i = 0; i < this.chips.length; i++) {
    this.chips[i].removeEventListener('click', this.onChipDown, false);
    this.chips[i].removeEventListener('keydown', this.onChipKey, false);
  }
  if (this.svg) {
    this.svg.removeEventListener('pointermove', this.onStageMove, false);
    this.svg.removeEventListener('pointerleave', this.onStageLeave, false);
  }
  if (this.win) {
    this.win.removeEventListener('pointermove', this.onMove, false);
    this.win.removeEventListener('pointerup', this.onUp, false);
    this.win.removeEventListener('pointercancel', this.onUp, false);
  }
  if (this.element) {
    this.element.innerHTML = '';
    this.element.removeAttribute('data-lieq-ready');
  }
  this.drag = null;
  this.cdrag = null;
};

/* ==============================================================
 * 8.  Entry points
 * ============================================================== */

function currentDoc() {
  return typeof document !== 'undefined' ? document : null;
}

/* Build the SVG source for a given option set without touching the DOM.
 * Paths are left empty; render() fills them. Used by the test harness. */
function renderMarkup(opts, uid) {
  return svgMarkup(uid || 'T', normOpts(extend(extend({}, DEFAULTS), opts)));
}

function init(target, opts) {
  var doc = currentDoc(), el;
  if (!doc) { return null; }
  el = resolve(target, doc);
  if (!el) { return null; }
  try {
    return new Instance(el, opts);
  } catch (err) {
    if (typeof console !== 'undefined' && console.warn) {
      console.warn('LiEQEqualizer: ' + (err && err.message ? err.message : err));
    }
    return null;
  }
}

function initAll(selector, opts) {
  var doc = currentDoc(), list, out = [], i, inst;
  if (!doc) { return out; }
  list = doc.querySelectorAll(selector || '[data-lieq-equalizer]');
  for (i = 0; i < list.length; i++) {
    if (list[i].getAttribute('data-lieq-ready')) { continue; }
    inst = init(list[i], opts);
    if (inst) { out.push(inst); }
  }
  return out;
}

function autoInit() {
  var doc = currentDoc(), made, fallback;
  if (!doc) { return; }
  made = initAll('[data-lieq-equalizer]');
  if (made.length) { return made; }
  fallback = doc.getElementById('eq-mount');
  if (fallback && !fallback.getAttribute('data-lieq-ready')) {
    made = init(fallback);
    return made ? [made] : [];
  }
  return [];
}

(function () {
  var doc = currentDoc();
  if (!doc) { return; }
  if (doc.readyState === 'loading') {
    doc.addEventListener('DOMContentLoaded', autoInit, false);
  } else {
    autoInit();
  }
})();

/* The model every instance starts from, built once so the page and the test
 * harness are looking at the same numbers. */
var MODEL = buildModel({ n: N_SAMPLES });
var MODEL_REACH = reach(MODEL.student, MODEL.teacher, FULL_SCALE_RESIDUAL);

return {
  version: VERSION,
  init: init,
  initAll: initAll,
  autoInit: autoInit,
  renderMarkup: renderMarkup,
  css: CSS,
  caption: CAPTION,
  defaults: DEFAULTS,
  presets: PRESETS,
  Instance: Instance,
  PluckState: PluckState,
  curves: {
    teacher: MODEL.teacher,
    student: MODEL.student,
    teacherEnvelope: MODEL.teacherEnvelope,
    teacherRipple: MODEL.teacherRipple,
    studentEnvelope: MODEL.studentEnvelope,
    studentRipple: MODEL.studentRipple,
    n: MODEL.n,
    vmax: modelVmax(MODEL, MODEL_REACH),
    reach: MODEL_REACH,
    shapeDepartureIndex: shapeDepartureIndex(MODEL.studentEnvelope, MODEL.teacherEnvelope)
  },
  constants: {
    N_SAMPLES: N_SAMPLES,
    EDGE_TAPER: EDGE_TAPER,
    TEACHER_LOBES: TEACHER_LOBES,
    STUDENT_LOBES: STUDENT_LOBES,
    RIPPLE: RIPPLE,
    STUDENT_RIPPLE_KEEP: STUDENT_RIPPLE_KEEP,
    FULL_SCALE_RESIDUAL: FULL_SCALE_RESIDUAL,
    DETAIL_REACH: DETAIL_REACH,
    GRAB_RADIUS: GRAB_RADIUS,
    GRAB_RADIUS_TOUCH: GRAB_RADIUS_TOUCH,
    GRAB_WINDOW: GRAB_WINDOW,
    PLUCK_SPAN_FRAC: PLUCK_SPAN_FRAC,
    PLUCK_ZETA: PLUCK_ZETA,
    PLUCK_FREQ: PLUCK_FREQ,
    PLUCK_SETTLE_EPS: PLUCK_SETTLE_EPS,
    PLUCK_MIN_TIME: PLUCK_MIN_TIME,
    PLUCK_MAX_AMP: PLUCK_MAX_AMP,
    ANN_DETAIL_FROM: ANN_DETAIL_FROM,
    ANN_DETAIL_TO: ANN_DETAIL_TO,
    ANN_DONE_AT: ANN_DONE_AT
  },
  math: {
    clamp: clamp,
    lerp: lerp,
    smooth: smooth,
    split: split,
    morph: morph,
    reach: reach,
    bump: bump,
    spring: spring,
    linePath: linePath,
    catmullPath: catmullPath,
    areaPath: areaPath,
    bracketPath: bracketPath,
    parseCurve: parseCurve,
    lobeSum: lobeSum,
    edgeWindow: edgeWindow,
    lcg: lcg,
    buildRipple: buildRipple,
    buildModel: buildModel,
    modelVmax: modelVmax,
    shapeDepartureIndex: shapeDepartureIndex,
    pickCurve: pickCurve,
    curveDist: curveDist,
    segDist: segDist
  },
  geometry: {
    VB_W: VB_W, VB_H: VB_H, PLOT: PLOT, PLOT_HEAD: PLOT_HEAD,
    RACK: RACK, TRACK: TRACK, FADER_X: FADER_X, INSET: VB_INSET, LEGEND: LEGEND,
    CHIP: CHIP, chipWidth: chipWidth, hintWidth: hintWidth, advance: advance,
    ROW: { tick: ROW_TICK, title: ROW_TITLE, chip: ROW_CHIP, readout: ROW_READOUT },
    HINT: HINT,
    ANN: { shape: ANN_SHAPE, detail: ANN_DETAIL,
           detailFrom: ANN_DETAIL_FROM, detailTo: ANN_DETAIL_TO, doneAt: ANN_DONE_AT }
  }
};

});
