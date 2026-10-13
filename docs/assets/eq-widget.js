/*!
 * eq-widget.js — LiEqualizer project page, interactive spectral equaliser.
 *
 * A self-contained hero instrument. Two vertical faders ("LF EQUALIZER" and
 * "HF EQUALIZER") continuously morph the sparse student's profile toward the
 * dense teacher's: LF moves the low-frequency ENVELOPE, HF moves the
 * high-frequency DETAIL. Both rest at 0. Releasing a fader leaves the curve
 * exactly where it was put — the faders never spring back.
 *
 * Six more things move, so the instrument is not just two sliders:
 *   · either curve can be PLUCKED — grab it anywhere, pull, and on release it
 *     springs back to the shape the faders imply, with a short underdamped
 *     settle. A pluck never touches the fader values.
 *   · a scrub cursor follows the pointer across the plot and reads out the
 *     local teacher-student gap.
 *   · three preset pills (BYPASS / LF / LF+HF) jump the faders to the three
 *     states of the paper's insight figure. They jump; they do not rebound.
 *
 * At full travel the student lands CLOSE TO but not ON the teacher: a residual
 * of FULL_SCALE_RESIDUAL (4% of the signal range) is left behind, because the
 * paper's own measurements show a distilled student near the teacher, not
 * equal to it.
 *
 * Native inline SVG + one injected <style>. No framework, no bundler, no CDN,
 * no chart library, no build step. ES5-compatible. Safe to load with a plain
 * <script src="assets/eq-widget.js" defer></script>.
 *
 * Public API (window.LiEQEqualizer):
 *   init(target, opts)   -> instance | null     mount into one element
 *   initAll(sel, opts)   -> [instance]          mount into every match
 *   caption              -> the illustrative-not-measured disclosure line
 *   math                 -> pure helpers (smooth / split / morph / reach /
 *                           bump / spring / path)
 *   PluckState           -> the DOM-free pluck + spring state machine
 *   curves               -> { teacher: [...], student: [...] }
 *
 * Instance:
 *   .element .svg .state .caption
 *   .getValues() -> {lf, hf}
 *   .setValues(lf, hf)      .setValue('lf', v)
 *   .morph(lf, hf) -> []    the curve the given pose would draw (pure)
 *   .destroy()
 *
 * The two curves are traced from the left half of the paper's insight figure
 * (the "Bypass / LF Equalizer / HF Equalizer" triptych), whose own x axis is
 * labelled "Spatial position" — so this plot is in the SPATIAL domain, not the
 * frequency domain: x is a spatial coordinate, y is a signal/feature value.
 * The curves are illustrative, not measured; see CAPTION below.
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

var VERSION = '2.0.0';

/* The one-line disclosure the page should print under the instrument. The
 * student's high-frequency ripple is synthesised: the curve traced off the
 * figure is smooth, and without a ripple the HF fader would have nothing to
 * take away. That is an illustrative choice and it is stated out loud. */
var CAPTION =
  'Illustrative, not measured: both curves are traced from the paper’s insight figure, ' +
  'and the student’s high-frequency ripple is synthesised so the HF fader has something to remove.';

/* ==============================================================
 * 1.  Source curves
 *
 * 97 samples on a shared, uniform x grid, y normalised so the taller
 * of the two peaks is 1.0. Traced column by column off the paper's
 * insight figure: the teacher from the top-left panel (rescaled by the
 * 1.046 factor that least-squares-fits the dashed teacher drawn in the
 * student's own axes), the student from the bottom-left panel. The
 * student additionally carries a small decaying ripple over its right
 * half — the paper's "unsupported / spurious high-frequency response",
 * which is what the HF fader exists to remove, and which is the one
 * part of these curves that is synthesised rather than traced.
 * ============================================================== */

var TEACHER_SRC =
  '0,0.0148,0.0363,0.0502,0.0705,0.0967,0.127,0.1615,0.1991,0.2397,0.282,0.3254,' +
  '0.3678,0.4079,0.4442,0.4751,0.4992,0.5149,0.5237,0.5262,0.5243,0.5223,0.5271,' +
  '0.541,0.5597,0.5829,0.6143,0.6549,0.6914,0.7135,0.7198,0.7192,0.7217,0.7342,' +
  '0.7599,0.7961,0.8387,0.882,0.9215,0.9526,0.974,0.9862,0.9916,0.9913,0.9842,' +
  '0.9671,0.9386,0.8991,0.8594,0.83,0.8198,0.8164,0.8052,0.7811,0.7549,0.7366,' +
  '0.7188,0.6912,0.6487,0.5965,0.5409,0.4863,0.4363,0.3939,0.3598,0.3351,0.3201,' +
  '0.3164,0.3227,0.3378,0.3574,0.378,0.3943,0.4022,0.3964,0.3787,0.3565,0.34,' +
  '0.3268,0.3113,0.2904,0.2709,0.2545,0.2375,0.2154,0.1887,0.1604,0.133,0.1083,' +
  '0.0871,0.069,0.0539,0.0415,0.0331,0.0288,0.0137,0';

var STUDENT_SRC =
  '0,0.0331,0.0775,0.0969,0.1251,0.158,0.1953,0.2336,0.2728,0.3088,0.341,0.3669,' +
  '0.3873,0.4032,0.4183,0.4376,0.4639,0.4992,0.541,0.5899,0.6439,0.7017,0.7588,' +
  '0.8112,0.8568,0.8953,0.9277,0.9541,0.9744,0.9886,0.9969,1,0.999,0.9935,0.9819,' +
  '0.9634,0.9377,0.9055,0.8661,0.8216,0.7741,0.7254,0.674,0.6201,0.5652,0.5152,' +
  '0.4741,0.4474,0.4335,0.43,0.4309,0.4328,0.4357,0.4455,0.4718,0.5198,0.585,' +
  '0.6502,0.6951,0.7073,0.6925,0.6716,0.6669,0.6861,0.7183,0.7396,0.7287,0.6827,' +
  '0.618,0.5603,0.5249,0.5097,0.4969,0.47,0.4235,0.369,0.3245,0.3048,0.3096,' +
  '0.3253,0.3329,0.3207,0.2921,0.2634,0.2491,0.2491,0.2468,0.2251,0.1845,0.1417,' +
  '0.1112,0.0939,0.0827,0.0722,0.0645,0.0306,0';

/* Where each annotation points, as a fraction of the x grid. 26/96 is where
 * the student's envelope most exceeds the teacher's (the low-frequency
 * excess); 74/96 sits in the middle of the student's spurious ripple. */
var ANN_LF_FRAC = 26 / 96;
var ANN_HF_FRAC = 74 / 96;

/* ==============================================================
 * 2.  Named constants
 * ============================================================== */

/* What the equaliser CANNOT recover. At lf = hf = 1 the student lands this far
 * from the teacher, measured as the largest absolute difference in normalised
 * signal units — 4% of the plotted range, which reads as "nearly recovered,
 * not identical". Honest, too: the paper reports a distilled student close to
 * the dense teacher, never equal to it. The faders therefore travel over
 * `reach = 1 - residual / peakGap` of the student-to-teacher path. */
var FULL_SCALE_RESIDUAL = 0.04;

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

/* ==============================================================
 * 3.  Pure maths — no DOM, safe to require() in node
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
 * window is the cheap Gaussian the recipe calls for. */
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
 *     y[i] = lerp(studentLo, teacherLo, lf) + lerp(studentHi, teacherHi, hf)
 * Separately linear in lf and in hf, so the response is continuous, every
 * intermediate fader position is a proportional intermediate curve, and the
 * extremes of the family are reached at the four corners of the unit square.
 * The widget feeds it lf * reach and hf * reach (see `reach` below), which is
 * still exactly linear — it only shortens the path. */
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

/* Uniform Catmull-Rom through every sample, emitted as cubic Béziers.
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

/* ==============================================================
 * 4.  PluckState — the direct-manipulation state machine.
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
 * 5.  Geometry — viewBox 480 x 248, aspect 1.935.
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

/* where the two annotations anchor their text, and the dimension bar's cap width */
var ANN_LF_TEXT_X = PLOT.l + 70;
var ANN_HF_TEXT_X = PLOT.r - 2;
var GAP_CAP = 4;

/* legend run: swatch, text, swatch, text */
var LEGEND = { y: 17, baseline: 20.5, sw1: 46, t1: 70, sw2: 164, t2: 188, swLen: 18 };

/* the bottom rows: axis ticks + title, then the preset pills and the readout */
var ROW_TICK = 213, ROW_TITLE = 215, ROW_CHIP = 229, ROW_READOUT = 232;
var CHIP = { h: 13, hit: 24, gap: 8, pad: 8, fs: 8, ls: 0.1 };

/* the pluck hint, top right of the plot */
var HINT = { x: PLOT.r, y1: 16, y2: 24.5 };

/* Dual clamp, exactly as the sibling widget does it: a tight box for anything
 * the user drags, an 8px-inset box for anything computed. */
var VB_INSET = 8;

/* Chip boxes are sized from the LARGEST font the chip label will ever be
 * painted at (the narrow-screen size), so the pill never has to resize and the
 * text never escapes it. Same advance model the test harness uses. */
function chipWidth(label) {
  var s = String(label == null ? '' : label);
  return n1(s.length * CHIP.fs * 0.6 + (s.length - 1) * CHIP.ls * CHIP.fs + CHIP.pad * 2);
}

/* ==============================================================
 * 6.  Styles — injected once per document
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

  /* ---- curves: fat core + thin bright dashed flow, the sibling's beam idiom ---- */
  '.lieq-curve{fill:none;stroke-linecap:round;stroke-linejoin:round;pointer-events:none}',
  '.lieq-t-halo{stroke:#56adf0;stroke-width:8;opacity:.22}',
  '.lieq-t-core{stroke:#56adf0;stroke-width:2.9;transition:stroke-width .14s ease}',
  '.lieq-t-flow{stroke:#bfe0ff;stroke-width:1.2}',
  '.lieq-s-core{stroke:#e58236;stroke-width:2.9;transition:stroke-width .14s ease}',
  '.lieq-s-flow{stroke:#f6bd8c;stroke-width:1.2}',
  '.lieq-flow{stroke-dasharray:6 12;animation:lieq-flow 2.4s steps(18,end) infinite}',
  '@keyframes lieq-flow{to{stroke-dashoffset:-36}}',

  /* ---- the pluck bands: invisible, wide, and the only thing you can grab ---- */
  '.lieq-grab{fill:none;stroke:rgba(0,0,0,0);stroke-width:22;stroke-linecap:round;',
  'stroke-linejoin:round;pointer-events:stroke;cursor:grab;touch-action:none}',
  '.lieq-stage[data-lieq-hot="s"] .lieq-s-core{stroke-width:3.9}',
  '.lieq-stage[data-lieq-hot="t"] .lieq-t-core{stroke-width:3.9}',
  '.lieq-stage[data-lieq-pluck="1"] .lieq-grab{cursor:grabbing}',

  /* ---- the pluck hint, retired the first time someone plucks ---- */
  '.lieq-hint{fill:#6f8bab;font-size:6.5px;font-weight:700;letter-spacing:.14em;',
  'transition:opacity .4s ease;animation:lieq-hint 3.4s ease-in-out infinite}',
  '@keyframes lieq-hint{0%,100%{opacity:.5}50%{opacity:.95}}',

  /* ---- scrub cursor ---- */
  '.lieq-scrub{pointer-events:none;opacity:0;transition:opacity .16s ease}',
  '.lieq-scrub-line{stroke:rgba(134,182,239,.45);stroke-width:1;stroke-dasharray:1 4}',
  '.lieq-scrub-s{fill:#f6bd8c}',
  '.lieq-scrub-t{fill:#bfe0ff}',

  /* ---- annotations ---- */
  '.lieq-ann{pointer-events:none}',
  '.lieq-ann-t{fill:#eab089;font-size:7.5px;font-weight:700;letter-spacing:.06em}',
  '.lieq-ann-line{stroke:#eab089;stroke-width:1.1;stroke-linecap:round}',
  '.lieq-ann-dash{stroke:#eab089;stroke-width:1.1;stroke-dasharray:2 3;stroke-linecap:round}',
  '.lieq-ann-halo{fill:#eab089;opacity:.2}',
  '.lieq-ann-dot{fill:#f6bd8c}',

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
  '.lieq-flow{animation:none}',
  '.lieq-scan{animation:none;opacity:0}',
  '.lieq-hint{animation:none;opacity:.8}',
  '.lieq-scrub{transition:none}',
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
  '.lieq-hint{font-size:7.5px}',
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
 * 7.  Markup
 * ============================================================== */

function esc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function faderMarkup(uid, key, cx, label, aria) {
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
  g.push('<text class="lieq-cap-sub" y="100">EQUALIZER</text>');
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
  s.push('<linearGradient id="lieqArea' + uid + '" x1="0" y1="0" x2="0" y2="1">' +
         '<stop offset="0" stop-color="#e58236" stop-opacity=".34"/>' +
         '<stop offset=".62" stop-color="#e58236" stop-opacity=".1"/>' +
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

  /* ---- the pluck hint ---- */
  s.push('<g data-lieq="hint">');
  s.push('<text class="lieq-hint" x="' + HINT.x + '" y="' + HINT.y1 + '" text-anchor="end">' +
         esc(o.hint[0]) + '</text>');
  s.push('<text class="lieq-hint" x="' + HINT.x + '" y="' + HINT.y2 + '" text-anchor="end">' +
         esc(o.hint[1]) + '</text>');
  s.push('</g>');

  /* ---- curves (declared before everything solid, as the sibling does) ---- */
  s.push('<g clip-path="url(#lieqClip' + uid + ')">');
  s.push('<path class="lieq-s-area" data-lieq="sArea" d="" fill="url(#lieqArea' + uid + ')" stroke="none"/>');
  s.push('<path class="lieq-curve lieq-t-halo" data-lieq="tHalo" d=""/>');
  s.push('<path class="lieq-curve lieq-t-core" data-lieq="tCore" d=""/>');
  s.push('<path class="lieq-curve lieq-t-flow lieq-flow" data-lieq="tFlow" d=""/>');
  s.push('<path class="lieq-curve lieq-s-core" data-lieq="sCore" d="" opacity=".62"/>');
  s.push('<path class="lieq-curve lieq-s-flow lieq-flow" data-lieq="sFlow" d="" opacity=".75"/>');
  s.push('</g>');

  /* ---- annotation 1: low-frequency excess (fades out with LF) ---- */
  s.push('<g class="lieq-ann" data-lieq="annLF">');
  s.push('<text class="lieq-ann-t" x="' + ANN_LF_TEXT_X + '" y="56" text-anchor="end">' +
         esc(o.lfAnnotation[0]) + '</text>');
  s.push('<text class="lieq-ann-t" x="' + ANN_LF_TEXT_X + '" y="68" text-anchor="end">' +
         esc(o.lfAnnotation[1]) + '</text>');
  s.push('<line class="lieq-ann-dash" data-lieq="gapStem" x1="0" y1="0" x2="0" y2="0"/>');
  s.push('<line class="lieq-ann-line" data-lieq="gapTop" x1="0" y1="0" x2="0" y2="0"/>');
  s.push('<line class="lieq-ann-line" data-lieq="gapBot" x1="0" y1="0" x2="0" y2="0"/>');
  s.push('</g>');

  /* ---- annotation 2: spurious high-frequency response (fades out with HF) ---- */
  s.push('<g class="lieq-ann" data-lieq="annHF">');
  s.push('<text class="lieq-ann-t" x="' + ANN_HF_TEXT_X + '" y="42" text-anchor="end">' +
         esc(o.hfAnnotation[0]) + '</text>');
  s.push('<text class="lieq-ann-t" x="' + ANN_HF_TEXT_X + '" y="54" text-anchor="end">' +
         esc(o.hfAnnotation[1]) + '</text>');
  s.push('<line class="lieq-ann-dash" data-lieq="hfDrop" x1="0" y1="60" x2="0" y2="0"/>');
  s.push('<circle class="lieq-ann-halo" data-lieq="hfHalo" cx="0" cy="0" r="7"/>');
  s.push('<circle class="lieq-ann-dot" data-lieq="hfDot" cx="0" cy="0" r="2.6"/>');
  s.push('</g>');

  /* ---- scrub cursor ---- */
  s.push('<g class="lieq-scrub" data-lieq="scrub">');
  s.push('<line class="lieq-scrub-line" data-lieq="scrubLine" x1="0" y1="' + PLOT.t +
         '" x2="0" y2="' + PLOT.b + '"/>');
  s.push('<circle class="lieq-scrub-t" data-lieq="scrubDotT" cx="0" cy="0" r="2.4"/>');
  s.push('<circle class="lieq-scrub-s" data-lieq="scrubDotS" cx="0" cy="0" r="2.4"/>');
  s.push('</g>');

  /* ---- axes ---- */
  s.push('<line class="lieq-axis" x1="' + PLOT.l + '" y1="' + PLOT.t + '" x2="' + PLOT.l + '" y2="' + PLOT.b + '"/>');
  s.push('<line class="lieq-axis" x1="' + PLOT.l + '" y1="' + PLOT.b + '" x2="' + PLOT.r + '" y2="' + PLOT.b + '"/>');
  for (i = 0; i <= 4; i++) {
    gx = n1(PLOT.l + (PLOT.r - PLOT.l) * (i / 4));
    s.push('<line class="lieq-tick" x1="' + gx + '" y1="' + PLOT.b + '" x2="' + gx + '" y2="' + (PLOT.b + 4) + '"/>');
  }
  s.push('<text class="lieq-tick-t" x="' + PLOT.l + '" y="' + ROW_TICK + '">' + esc(o.lowTick) + '</text>');
  s.push('<text class="lieq-tick-t" x="' + PLOT.r + '" y="' + ROW_TICK + '" text-anchor="end">' +
         esc(o.highTick) + '</text>');
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

  /* ---- the pluck bands, on top of the ink so they can be grabbed ---- */
  s.push('<g clip-path="url(#lieqClip' + uid + ')">');
  s.push('<path class="lieq-grab" data-lieq="tGrab" data-lieq-curve="t" d=""/>');
  s.push('<path class="lieq-grab" data-lieq="sGrab" data-lieq-curve="s" d=""/>');
  s.push('</g>');

  /* ---- preset pills ---- */
  s.push(presetMarkup(o.presets));

  /* ---- rack ---- */
  s.push('<rect class="lieq-rack-bg" x="' + RACK.x + '" y="' + RACK.y + '" width="' + RACK.w +
         '" height="' + RACK.h + '" rx="14" fill="url(#lieqRack' + uid +
         ')" stroke="rgba(134,182,239,.17)" stroke-width="1"/>');
  s.push('<text class="lieq-rack-eyebrow" x="' + n1(RACK.x + RACK.w / 2) + '" y="38" text-anchor="middle">' +
         esc(o.rackLabel) + '</text>');
  s.push(faderMarkup(uid, 'lf', FADER_X.lf, 'LF', o.lfAria));
  s.push(faderMarkup(uid, 'hf', FADER_X.hf, 'HF', o.hfAria));

  s.push('</svg>');
  return s.join('');
}

/* ==============================================================
 * 8.  The widget
 * ============================================================== */

var uidSeq = 0;

var PRESETS = [
  { label: 'BYPASS', lf: 0, hf: 0, aria: 'Preset: bypass — both equalizers at zero, the undistilled student' },
  { label: 'LF',     lf: 1, hf: 0, aria: 'Preset: LF equalizer only — restore the low-frequency envelope' },
  { label: 'LF+HF',  lf: 1, hf: 1, aria: 'Preset: LF and HF equalizers — restore the envelope and remove the spurious detail' }
];

var DEFAULTS = {
  lf: 0,
  hf: 0,
  chrome: false,                       /* true = also paint the dark card shell */
  smoothRadius: 3,
  smoothPasses: 3,
  residual: FULL_SCALE_RESIDUAL,
  /* The source figure's own axis is SPATIAL — its x is literally labelled
   * "Spatial position". LF and HF name bands of that spatial signal, which is
   * what the two faders act on; they are not axes. */
  xLabel: 'Spatial position',
  yLabel: 'Signal / feature response',
  lowTick: '0',
  highTick: 'X MAX',
  teacherLabel: 'DENSE TEACHER',
  studentLabel: 'SPARSE STUDENT',
  rackLabel: 'SPECTRAL EQ',
  readoutLabel: 'PEAK GAP',
  scrubLabel: 'GAP HERE',
  hint: ['PLUCK', 'A CURVE'],
  lfAnnotation: ['LOW-FREQUENCY', 'EXCESS'],
  hfAnnotation: ['SPURIOUS HIGH-FREQUENCY', 'RESPONSE'],
  lfAria: 'LF equalizer strength, low-frequency envelope',
  hfAria: 'HF equalizer strength, high-frequency detail',
  ariaLabel: 'Spectral equalizer: two faders morph the sparse student’s spatial profile toward the dense teacher’s.',
  desc: 'A plot of two signal profiles against spatial position. The blue curve is the dense ' +
        'teacher. The orange curve is the sparse student: at rest it carries too much energy in ' +
        'the low-frequency envelope and a spurious high-frequency ripple. The LF fader moves the ' +
        'student’s low-frequency envelope onto the teacher’s; the HF fader moves its ' +
        'high-frequency detail onto the teacher’s. With both at full the student comes close ' +
        'to the teacher but does not reach it — a residual of four per cent of the signal ' +
        'range is left, as in the paper’s own measurements. Either curve can also be dragged ' +
        'to deform it; it springs back when released, and dragging a curve never changes the faders.',
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

function resolve(target, doc) {
  if (!target) { return null; }
  if (typeof target === 'string') { return doc ? doc.querySelector(target) : null; }
  if (target.nodeType === 1) { return target; }
  return null;
}

function Instance(el, opts) {
  var o = extend(extend({}, DEFAULTS), opts),
      doc = el.ownerDocument,
      uid = String(++uidSeq),
      teacher = o.teacher ? parseCurve(o.teacher) : parseCurve(TEACHER_SRC),
      student = o.student ? parseCurve(o.student) : parseCurve(STUDENT_SRC),
      n = Math.min(teacher.length, student.length),
      ts, ss, i, corner, hi, lo, v;

  if (n < 4) { throw new Error('LiEQEqualizer: need at least 4 samples'); }
  teacher = teacher.slice(0, n);
  student = student.slice(0, n);

  /* tolerate a plain string where a two-line label is expected */
  o.lfAnnotation = normLines(o.lfAnnotation, DEFAULTS.lfAnnotation);
  o.hfAnnotation = normLines(o.hfAnnotation, DEFAULTS.hfAnnotation);
  o.hint = normLines(o.hint, DEFAULTS.hint);

  /* --- sample both curves onto the shared grid, then split each into a
   *     low-frequency envelope and a high-frequency residual --- */
  ts = split(teacher, o.smoothRadius, o.smoothPasses);
  ss = split(student, o.smoothRadius, o.smoothPasses);

  this.opts = o;
  this.caption = o.caption;
  this.element = el;
  this.doc = doc;
  this.uid = uid;
  this.n = n;
  this.teacher = teacher;
  this.student = student;
  this.tLo = ts.lo; this.tHi = ts.hi;
  this.sLo = ss.lo; this.sHi = ss.hi;
  /* how much of the gap a full fader closes — the rest is FULL_SCALE_RESIDUAL */
  this.reach = reach(student, teacher, o.residual);
  this.state = { lf: clamp(+o.lf || 0, 0, 1), hf: clamp(+o.hf || 0, 0, 1) };
  this.drag = null;
  this.cdrag = null;
  this.pluck = new PluckState({ span: Math.max(2, PLUCK_SPAN_FRAC * (n - 1)) });
  this.plucked = false;
  this.scrubAt = -1;
  this.frame = 0;
  this.springFrame = 0;
  this.destroyed = false;

  /* --- vertical scale. The family is separately linear in lf and in hf, so
   *     its extremes over the unit square are attained at the four corners. --- */
  hi = 0; lo = 0;
  for (corner = 0; corner < 4; corner++) {
    v = this.morph(corner === 1 || corner === 3 ? 1 : 0, corner >= 2 ? 1 : 0);
    for (i = 0; i < n; i++) {
      if (v[i] > hi) { hi = v[i]; }
      if (v[i] < lo) { lo = v[i]; }
    }
  }
  this.vmax = hi > 0 ? hi : 1;
  this.vmin = 0;
  this.span = (PLOT.b - PLOT.t - PLOT_HEAD) / this.vmax;

  /* --- fixed x grid --- */
  this.xs = new Array(n);
  for (i = 0; i < n; i++) { this.xs[i] = PLOT.l + (PLOT.r - PLOT.l) * (i / (n - 1)); }
  this.ys = new Array(n);
  this.tys = new Array(n);
  this.annLFi = clamp(Math.round(ANN_LF_FRAC * (n - 1)), 0, n - 1);
  this.annHFi = clamp(Math.round(ANN_HF_FRAC * (n - 1)), 0, n - 1);

  this.build();
  this.bind();
  this.render();
}

/* The pose the faders imply. lf and hf travel over `reach` of the path, which
 * leaves exactly `residual` between the full-travel student and the teacher. */
Instance.prototype.morph = function (lf, hf) {
  return morph(this.sLo, this.sHi, this.tLo, this.tHi, lf * this.reach, hf * this.reach);
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
    sArea: this.q('sArea'), sCore: this.q('sCore'), sFlow: this.q('sFlow'),
    tHalo: this.q('tHalo'), tCore: this.q('tCore'), tFlow: this.q('tFlow'),
    sGrab: this.q('sGrab'), tGrab: this.q('tGrab'),
    annLF: this.q('annLF'), annHF: this.q('annHF'),
    gapStem: this.q('gapStem'), gapTop: this.q('gapTop'), gapBot: this.q('gapBot'),
    hfDrop: this.q('hfDrop'), hfHalo: this.q('hfHalo'), hfDot: this.q('hfDot'),
    gapValue: this.q('gapValue'), roLabel: this.q('roLabel'), hint: this.q('hint'),
    scrub: this.q('scrub'), scrubLine: this.q('scrubLine'),
    scrubDotS: this.q('scrubDotS'), scrubDotT: this.q('scrubDotT')
  };

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

Instance.prototype.bind = function () {
  var self = this, key, f, i, win;

  this.onDown = function (e) { self.pointerDown(e); };
  this.onMove = function (e) { self.pointerMove(e); };
  this.onUp = function (e) { self.pointerUp(e); };
  this.onKey = function (e) { self.keyDown(e); };
  this.onCurveDown = function (e) { self.curveDown(e); };
  this.onCurveOver = function (e) { self.curveOver(e); };
  this.onCurveOut = function () { self.curveOut(); };
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

  this.grabs = [];
  if (this.el.sGrab) { this.grabs.push(this.el.sGrab); }
  if (this.el.tGrab) { this.grabs.push(this.el.tGrab); }
  for (i = 0; i < this.grabs.length; i++) {
    this.grabs[i].addEventListener('pointerdown', this.onCurveDown, false);
    this.grabs[i].addEventListener('pointerover', this.onCurveOver, false);
    this.grabs[i].addEventListener('pointerout', this.onCurveOut, false);
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

Instance.prototype.curveDown = function (e) {
  var node = e.currentTarget, which, i0, base;
  if (this.drag || this.cdrag) { return; }
  if (e.button !== undefined && e.button !== 0 && e.pointerType === 'mouse') { return; }
  which = node.getAttribute('data-lieq-curve');
  e.preventDefault();
  e.stopPropagation();

  i0 = this.indexAt(this.toSvg(e).x);
  base = which === 't' ? this.tvals[i0] : this.svals[i0];

  this.cdrag = { id: e.pointerId, node: node, which: which };
  this.pluckBegin(which, i0, base);
  try { node.setPointerCapture(e.pointerId); } catch (err) { /* not supported */ }
  if (this.stage) { this.stage.setAttribute('data-lieq-pluck', '1'); }
  if (this.el.hint) { this.el.hint.setAttribute('opacity', '0'); }
  this.schedule();
};

Instance.prototype.curveOver = function (e) {
  var which = e.currentTarget.getAttribute('data-lieq-curve');
  if (this.stage) { this.stage.setAttribute('data-lieq-hot', which); }
};

Instance.prototype.curveOut = function () {
  if (this.stage && !this.cdrag) { this.stage.removeAttribute('data-lieq-hot'); }
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

/* ---- scrub cursor --------------------------------------------------- */

Instance.prototype.stageMove = function (e) {
  var p = this.toSvg(e), i;
  if (this.drag) { return; }
  if (p.x < PLOT.l || p.x > PLOT.r || p.y < PLOT.t - 10 || p.y > PLOT.b + 6) {
    if (this.scrubAt >= 0) { this.scrubAt = -1; this.schedule(); }
    return;
  }
  i = this.indexAt(p.x);
  if (i !== this.scrubAt) { this.scrubAt = i; this.schedule(); }
};

Instance.prototype.stageLeave = function () {
  if (this.scrubAt >= 0) { this.scrubAt = -1; this.schedule(); }
  this.curveOut();
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

Instance.prototype.render = function () {
  var st = this.state, n = this.n, el = this.el,
      amp = this.pluck.amp(this.now()),
      vals, tvals, i, d, td, area, conv, gap, dv,
      ax, tY, sY, fadeLF, fadeHF, key, f, knobY, want;

  /* the two shapes: the faders' pose, plus whatever the cord is doing */
  vals = this.pluck.apply(this.morph(st.lf, st.hf), 's', amp);
  tvals = this.pluck.apply(this.teacher.slice(0), 't', amp);
  this.svals = vals;
  this.tvals = tvals;

  for (i = 0; i < n; i++) { this.ys[i] = this.y(vals[i]); this.tys[i] = this.y(tvals[i]); }
  d = catmullPath(this.xs, this.ys);
  td = catmullPath(this.xs, this.tys);
  area = d + 'L' + n1(this.xs[n - 1]) + ' ' + PLOT.b + 'L' + n1(this.xs[0]) + ' ' + PLOT.b + 'Z';
  this.teacherYs = this.tys;

  if (el.sCore) { el.sCore.setAttribute('d', d); }
  if (el.sFlow) { el.sFlow.setAttribute('d', d); }
  if (el.sArea) { el.sArea.setAttribute('d', area); }
  if (el.sGrab) { el.sGrab.setAttribute('d', d); }
  if (el.tHalo) { el.tHalo.setAttribute('d', td); }
  if (el.tCore) { el.tCore.setAttribute('d', td); }
  if (el.tFlow) { el.tFlow.setAttribute('d', td); }
  if (el.tGrab) { el.tGrab.setAttribute('d', td); }

  /* the suppressed-vs-clean opacity language: the student brightens as it converges */
  conv = (st.lf + st.hf) / 2;
  if (el.sCore) { el.sCore.setAttribute('opacity', n1(0.62 + 0.38 * conv) + ''); }
  if (el.sFlow) { el.sFlow.setAttribute('opacity', n1(0.75 + 0.25 * conv) + ''); }

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

  /* annotation 1 — the low-frequency excess, as a dimension bar */
  fadeLF = clamp(1 - st.lf * 1.15, 0, 1);
  if (el.annLF) { el.annLF.setAttribute('opacity', n1(fadeLF) + ''); }
  i = this.annLFi;
  ax = this.xs[i]; sY = this.ys[i]; tY = this.tys[i];
  if (el.gapStem) {
    el.gapStem.setAttribute('x1', n1(ax)); el.gapStem.setAttribute('x2', n1(ax));
    el.gapStem.setAttribute('y1', n1(sY)); el.gapStem.setAttribute('y2', n1(tY));
  }
  if (el.gapTop) {
    el.gapTop.setAttribute('x1', n1(ax - GAP_CAP)); el.gapTop.setAttribute('x2', n1(ax + GAP_CAP));
    el.gapTop.setAttribute('y1', n1(sY)); el.gapTop.setAttribute('y2', n1(sY));
  }
  if (el.gapBot) {
    el.gapBot.setAttribute('x1', n1(ax - GAP_CAP)); el.gapBot.setAttribute('x2', n1(ax + GAP_CAP));
    el.gapBot.setAttribute('y1', n1(tY)); el.gapBot.setAttribute('y2', n1(tY));
  }

  /* annotation 2 — the spurious ripple, as a dropline onto the live curve */
  fadeHF = clamp(1 - st.hf * 1.15, 0, 1);
  if (el.annHF) { el.annHF.setAttribute('opacity', n1(fadeHF) + ''); }
  i = this.annHFi;
  ax = this.xs[i]; sY = this.ys[i];
  if (el.hfDrop) {
    el.hfDrop.setAttribute('x1', n1(ax)); el.hfDrop.setAttribute('x2', n1(ax));
    el.hfDrop.setAttribute('y2', n1(sY - 7));
  }
  if (el.hfHalo) { el.hfHalo.setAttribute('cx', n1(ax)); el.hfHalo.setAttribute('cy', n1(sY)); }
  if (el.hfDot) { el.hfDot.setAttribute('cx', n1(ax)); el.hfDot.setAttribute('cy', n1(sY)); }

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
  for (i = 0; this.grabs && i < this.grabs.length; i++) {
    this.grabs[i].removeEventListener('pointerdown', this.onCurveDown, false);
    this.grabs[i].removeEventListener('pointerover', this.onCurveOver, false);
    this.grabs[i].removeEventListener('pointerout', this.onCurveOut, false);
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
 * 9.  Entry points
 * ============================================================== */

function currentDoc() {
  return typeof document !== 'undefined' ? document : null;
}

/* Build the SVG source for a given option set without touching the DOM.
 * Paths are left empty; render() fills them. Used by the test harness. */
function renderMarkup(opts, uid) {
  var o = extend(extend({}, DEFAULTS), opts);
  o.lfAnnotation = normLines(o.lfAnnotation, DEFAULTS.lfAnnotation);
  o.hfAnnotation = normLines(o.hfAnnotation, DEFAULTS.hfAnnotation);
  o.hint = normLines(o.hint, DEFAULTS.hint);
  return svgMarkup(uid || 'T', o);
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
  curves: { teacher: parseCurve(TEACHER_SRC), student: parseCurve(STUDENT_SRC) },
  constants: {
    FULL_SCALE_RESIDUAL: FULL_SCALE_RESIDUAL,
    PLUCK_SPAN_FRAC: PLUCK_SPAN_FRAC,
    PLUCK_ZETA: PLUCK_ZETA,
    PLUCK_FREQ: PLUCK_FREQ,
    PLUCK_SETTLE_EPS: PLUCK_SETTLE_EPS,
    PLUCK_MIN_TIME: PLUCK_MIN_TIME,
    PLUCK_MAX_AMP: PLUCK_MAX_AMP
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
    catmullPath: catmullPath,
    parseCurve: parseCurve
  },
  geometry: {
    VB_W: VB_W, VB_H: VB_H, PLOT: PLOT, PLOT_HEAD: PLOT_HEAD,
    RACK: RACK, TRACK: TRACK, FADER_X: FADER_X, INSET: VB_INSET, LEGEND: LEGEND,
    CHIP: CHIP, chipWidth: chipWidth,
    ROW: { tick: ROW_TICK, title: ROW_TITLE, chip: ROW_CHIP, readout: ROW_READOUT },
    HINT: HINT,
    ANN: { lfTextX: ANN_LF_TEXT_X, hfTextX: ANN_HF_TEXT_X, gapCap: GAP_CAP,
           lfFrac: ANN_LF_FRAC, hfFrac: ANN_HF_FRAC }
  }
};

});
