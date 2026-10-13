/*!
 * eq-widget.js — LiEqualizer project page, interactive spectral equaliser.
 *
 * A self-contained hero instrument: two vertical faders ("LF EQUALIZER" and
 * "HF EQUALIZER") that continuously morph the sparse student's spectrum toward
 * the dense teacher's. The LF fader moves the low-frequency ENVELOPE; the HF
 * fader moves the high-frequency DETAIL. Both rest at 0. Releasing a fader
 * leaves the curve exactly where it was put — there is no spring-back.
 *
 * Native inline SVG + one injected <style>. No framework, no bundler, no CDN,
 * no chart library, no build step. ES5-compatible. Safe to load with a plain
 * <script src="assets/eq-widget.js" defer></script>.
 *
 * Public API (window.LiEQEqualizer):
 *   init(target, opts)   -> instance | null     mount into one element
 *   initAll(sel, opts)   -> [instance]          mount into every match
 *   math                 -> pure helpers (smooth / split / morph / path)
 *   curves               -> { teacher: [...], student: [...] }  the sampled source curves
 *
 * Instance:
 *   .element .svg .state
 *   .getValues() -> {lf, hf}
 *   .setValues(lf, hf)      .setValue('lf', v)
 *   .morph(lf, hf) -> []    the curve the given pose would draw (pure)
 *   .destroy()
 *
 * The two curves are traced from the left half of the paper's insight figure
 * (the "Bypass / LF Equalizer / HF Equalizer" triptych) and are illustrative,
 * not measured data.
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

var VERSION = '1.0.0';

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
 * which is what the HF fader exists to remove.
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
 * extremes of the family are reached at the four corners of the unit square. */
function morph(sLo, sHi, tLo, tHi, lf, hf) {
  var n = sLo.length, out = new Array(n), i;
  for (i = 0; i < n; i++) {
    out[i] = lerp(sLo[i], tLo[i], lf) + lerp(sHi[i], tHi[i], hf);
  }
  return out;
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
 * 3.  Geometry — viewBox 470 x 280, aspect 1.679, same family as the
 *     sibling page's optics bench.
 * ============================================================== */

var VB_W = 470, VB_H = 280;

var PLOT = { l: 48, r: 334, t: 46, b: 214 };        /* drawing frame          */
var PLOT_HEAD = 6;                                   /* headroom above the peak */
var RACK = { x: 352, y: 26, w: 110, h: 226 };        /* the fader rack         */
var FADER_X = { lf: 379, hf: 435 };
var TRACK = { top: 76, bottom: 210, mid: 143, half: 67 };  /* len = 134 */

/* where the two annotations anchor their text, and the dimension bar's cap width */
var ANN_LF_TEXT_X = PLOT.l + 70;
var ANN_HF_TEXT_X = PLOT.r - 2;
var GAP_CAP = 4;

/* legend run: swatch, text, swatch, text */
var LEGEND = { y: 21, baseline: 24.5, sw1: 48, t1: 72, sw2: 166, t2: 190, swLen: 18 };

/* Dual clamp, exactly as the sibling widget does it: a tight box for anything
 * the user drags, an 8px-inset box for anything computed. */
var VB_INSET = 8;

/* ==============================================================
 * 4.  Styles — injected once per document
 * ============================================================== */

var MONO = "var(--mono,'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,monospace)";

var CSS = [
  /* ---- stage ---- */
  '.lieq-stage{position:relative;overflow:hidden;border-radius:14px;font-family:' + MONO + '}',
  '.lieq-svg{display:block;width:100%;height:auto;position:relative;z-index:2}',
  '.lieq-svg text{font-family:' + MONO + '}',

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
  '.lieq-curve{fill:none;stroke-linecap:round;stroke-linejoin:round}',
  '.lieq-t-halo{stroke:#56adf0;stroke-width:9;opacity:.24}',
  '.lieq-t-core{stroke:#56adf0;stroke-width:3.2}',
  '.lieq-t-flow{stroke:#bfe0ff;stroke-width:1.3}',
  '.lieq-s-core{stroke:#e58236;stroke-width:3.2}',
  '.lieq-s-flow{stroke:#f6bd8c;stroke-width:1.3}',
  '.lieq-flow{stroke-dasharray:6 12;animation:lieq-flow 2.4s steps(18,end) infinite}',
  '@keyframes lieq-flow{to{stroke-dashoffset:-36}}',

  /* ---- annotations ---- */
  '.lieq-ann-t{fill:#eab089;font-size:7.5px;font-weight:700;letter-spacing:.06em}',
  '.lieq-ann-line{stroke:#eab089;stroke-width:1.1;stroke-linecap:round}',
  '.lieq-ann-dash{stroke:#eab089;stroke-width:1.1;stroke-dasharray:2 3;stroke-linecap:round}',
  '.lieq-ann-halo{fill:#eab089;opacity:.2}',
  '.lieq-ann-dot{fill:#f6bd8c}',

  /* ---- readout ---- */
  '.lieq-ro-k{fill:#6d7b8e;font-size:8px;font-weight:700;letter-spacing:.13em}',
  '.lieq-ro-v{fill:#dbe7f5;font-size:8px;font-weight:700;letter-spacing:.08em}',

  /* ---- rack ---- */
  '.lieq-rack-eyebrow{fill:#8d99ab;font-size:8px;font-weight:700;letter-spacing:.17em}',
  '.lieq-fader{cursor:grab;touch-action:none;outline:none}',
  '.lieq-fader.lieq-dragging{cursor:grabbing}',
  '.lieq-hit{fill:transparent}',
  '.lieq-detent{stroke:rgba(134,182,239,.3);stroke-width:1.1;stroke-linecap:round}',
  '.lieq-cap-body{fill:#0c1322;stroke:#9fd0ff;stroke-width:1.8;transition:fill .18s ease}',
  '.lieq-cap-line{stroke:#9fd0ff;stroke-width:1.4;stroke-linecap:round;opacity:.85}',
  '.lieq-fader:hover .lieq-cap-body,.lieq-fader.lieq-dragging .lieq-cap-body{fill:#16314f}',
  '.lieq-val{fill:#dbe7f5;font-size:9.5px;font-weight:700;letter-spacing:.05em;text-anchor:middle}',
  '.lieq-cap-label{fill:#dbe7f5;font-size:12px;font-weight:700;letter-spacing:.1em;text-anchor:middle}',
  '.lieq-cap-sub{fill:#8d99ab;font-size:7px;font-weight:700;letter-spacing:.1em;text-anchor:middle}',

  /* ---- focus ---- */
  '.lieq-focusring{fill:none;stroke:#58b9eb;stroke-width:2;opacity:0;pointer-events:none}',
  '.lieq-fader:focus .lieq-focusring{opacity:1}',
  '.lieq-fader:focus:not(:focus-visible) .lieq-focusring{opacity:0}',
  '.lieq-fader:focus-visible .lieq-focusring{opacity:1}',

  /* ---- respect the setting ---- */
  '@media (prefers-reduced-motion:reduce){',
  '.lieq-flow{animation:none}',
  '.lieq-scan{animation:none;opacity:0}',
  '.lieq-cap-body{transition:none}}',

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
 * 5.  Markup
 * ============================================================== */

function esc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function faderMarkup(uid, key, cx, label, aria) {
  var g = [];
  g.push('<g class="lieq-fader" data-lieq-fader="' + key + '"');
  g.push(' transform="translate(' + cx + ',' + TRACK.mid + ')"');
  g.push(' role="slider" tabindex="0" aria-orientation="vertical"');
  g.push(' aria-label="' + esc(aria) + '"');
  g.push(' aria-valuemin="0" aria-valuemax="100" aria-valuenow="0" aria-valuetext="0%">');
  g.push('<rect class="lieq-focusring" x="-21" y="-98" width="42" height="200" rx="12"/>');
  /* generous touch target: 50 x 156 user units, ~34 x 106 CSS px at phone width */
  g.push('<rect class="lieq-hit" x="-25" y="-78" width="50" height="156"/>');
  g.push('<rect class="lieq-track" x="-5" y="-' + TRACK.half + '" width="10" height="' +
         (TRACK.half * 2) + '" rx="5" fill="url(#lieqTrack' + uid + ')" ' +
         'stroke="rgba(134,182,239,.22)" stroke-width="1"/>');
  var k, ty;
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
  g.push('<text class="lieq-val" y="-81">0.00</text>');
  g.push('<text class="lieq-cap-label" y="86">' + esc(label) + '</text>');
  g.push('<text class="lieq-cap-sub" y="98">EQUALIZER</text>');
  g.push('</g>');
  return g.join('');
}

function svgMarkup(uid, o) {
  var s = [], i, gx, gy;

  s.push('<svg class="lieq-svg" viewBox="0 0 ' + VB_W + ' ' + VB_H + '" width="100%" ' +
         'role="group" aria-label="' + esc(o.ariaLabel) + '" ' +
         'aria-describedby="lieqDesc' + uid + '">');
  s.push('<desc id="lieqDesc' + uid + '">' + esc(o.desc) + '</desc>');

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
  s.push('<clipPath id="lieqClip' + uid + '"><rect x="' + (PLOT.l - 2) + '" y="' + (PLOT.t - 10) +
         '" width="' + (PLOT.r - PLOT.l + 4) + '" height="' + (PLOT.b - PLOT.t + 12) + '"/></clipPath>');
  s.push('</defs>');

  /* ---- grid ---- */
  s.push('<g aria-hidden="true">');
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
  s.push('<g data-lieq="annLF">');
  s.push('<text class="lieq-ann-t" x="' + ANN_LF_TEXT_X + '" y="70" text-anchor="end">' +
         esc(o.lfAnnotation[0]) + '</text>');
  s.push('<text class="lieq-ann-t" x="' + ANN_LF_TEXT_X + '" y="82" text-anchor="end">' +
         esc(o.lfAnnotation[1]) + '</text>');
  s.push('<line class="lieq-ann-dash" data-lieq="gapStem" x1="0" y1="0" x2="0" y2="0"/>');
  s.push('<line class="lieq-ann-line" data-lieq="gapTop" x1="0" y1="0" x2="0" y2="0"/>');
  s.push('<line class="lieq-ann-line" data-lieq="gapBot" x1="0" y1="0" x2="0" y2="0"/>');
  s.push('</g>');

  /* ---- annotation 2: spurious high-frequency response (fades out with HF) ---- */
  s.push('<g data-lieq="annHF">');
  s.push('<text class="lieq-ann-t" x="' + ANN_HF_TEXT_X + '" y="58" text-anchor="end">' +
         esc(o.hfAnnotation[0]) + '</text>');
  s.push('<text class="lieq-ann-t" x="' + ANN_HF_TEXT_X + '" y="70" text-anchor="end">' +
         esc(o.hfAnnotation[1]) + '</text>');
  s.push('<line class="lieq-ann-dash" data-lieq="hfDrop" x1="0" y1="78" x2="0" y2="0"/>');
  s.push('<circle class="lieq-ann-halo" data-lieq="hfHalo" cx="0" cy="0" r="7"/>');
  s.push('<circle class="lieq-ann-dot" data-lieq="hfDot" cx="0" cy="0" r="2.6"/>');
  s.push('</g>');

  /* ---- axes ---- */
  s.push('<line class="lieq-axis" x1="' + PLOT.l + '" y1="' + PLOT.t + '" x2="' + PLOT.l + '" y2="' + PLOT.b + '"/>');
  s.push('<line class="lieq-axis" x1="' + PLOT.l + '" y1="' + PLOT.b + '" x2="' + PLOT.r + '" y2="' + PLOT.b + '"/>');
  for (i = 0; i <= 4; i++) {
    gx = n1(PLOT.l + (PLOT.r - PLOT.l) * (i / 4));
    s.push('<line class="lieq-tick" x1="' + gx + '" y1="' + PLOT.b + '" x2="' + gx + '" y2="' + (PLOT.b + 4) + '"/>');
  }
  s.push('<text class="lieq-tick-t" x="' + PLOT.l + '" y="228">' + esc(o.lowTick) + '</text>');
  s.push('<text class="lieq-tick-t" x="' + PLOT.r + '" y="228" text-anchor="end">' + esc(o.highTick) + '</text>');
  s.push('<text class="lieq-axis-t" x="' + n1((PLOT.l + PLOT.r) / 2) + '" y="246" text-anchor="middle">' +
         esc(o.xLabel) + '</text>');
  s.push('<text class="lieq-axis-t" transform="translate(16,' + n1((PLOT.t + PLOT.b) / 2) +
         ') rotate(-90)" text-anchor="middle">' + esc(o.yLabel) + '</text>');

  /* ---- live readout ---- */
  s.push('<text x="' + PLOT.r + '" y="246" text-anchor="end">' +
         '<tspan class="lieq-ro-k">' + esc(o.readoutLabel) + ' </tspan>' +
         '<tspan class="lieq-ro-v" data-lieq="gapValue">0.00</tspan></text>');

  /* ---- end of the static decoration; the faders below must stay reachable ---- */
  s.push('</g>');

  /* ---- rack ---- */
  s.push('<rect x="' + RACK.x + '" y="' + RACK.y + '" width="' + RACK.w + '" height="' + RACK.h +
         '" rx="14" fill="url(#lieqRack' + uid + ')" stroke="rgba(134,182,239,.17)" stroke-width="1"/>');
  s.push('<text class="lieq-rack-eyebrow" x="' + n1(RACK.x + RACK.w / 2) + '" y="44" text-anchor="middle">' +
         esc(o.rackLabel) + '</text>');
  s.push(faderMarkup(uid, 'lf', FADER_X.lf, 'LF', o.lfAria));
  s.push(faderMarkup(uid, 'hf', FADER_X.hf, 'HF', o.hfAria));

  s.push('</svg>');
  return s.join('');
}

/* ==============================================================
 * 6.  The widget
 * ============================================================== */

var uidSeq = 0;

var DEFAULTS = {
  lf: 0,
  hf: 0,
  chrome: false,                       /* true = also paint the dark card shell */
  smoothRadius: 3,
  smoothPasses: 3,
  xLabel: 'Graph frequency',
  yLabel: 'Normalized spectral energy',
  lowTick: 'LOW',
  highTick: 'HIGH',
  teacherLabel: 'DENSE TEACHER',
  studentLabel: 'SPARSE STUDENT',
  rackLabel: 'SPECTRAL EQ',
  readoutLabel: 'PEAK GAP',
  lfAnnotation: ['LOW-FREQUENCY', 'EXCESS'],
  hfAnnotation: ['SPURIOUS HIGH-FREQUENCY', 'RESPONSE'],
  lfAria: 'LF equalizer strength, low-frequency envelope',
  hfAria: 'HF equalizer strength, high-frequency detail',
  ariaLabel: 'Spectral equalizer: two faders morph the sparse student’s spectrum toward the dense teacher’s.',
  desc: 'A spectrum plot with two curves against graph frequency. The blue curve is the dense ' +
        'teacher. The orange curve is the sparse student: at rest it carries too much energy at ' +
        'low frequency and a spurious response at high frequency. The LF fader moves the ' +
        'student’s low-frequency envelope onto the teacher’s; the HF fader moves its ' +
        'high-frequency detail onto the teacher’s. With both at full the two curves coincide.',
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

  /* tolerate a plain string where a two-line annotation is expected */
  o.lfAnnotation = normLines(o.lfAnnotation, DEFAULTS.lfAnnotation);
  o.hfAnnotation = normLines(o.hfAnnotation, DEFAULTS.hfAnnotation);

  /* --- sample both curves onto the shared grid, then split each into a
   *     low-frequency envelope and a high-frequency residual --- */
  ts = split(teacher, o.smoothRadius, o.smoothPasses);
  ss = split(student, o.smoothRadius, o.smoothPasses);

  this.opts = o;
  this.element = el;
  this.doc = doc;
  this.uid = uid;
  this.n = n;
  this.teacher = teacher;
  this.student = student;
  this.tLo = ts.lo; this.tHi = ts.hi;
  this.sLo = ss.lo; this.sHi = ss.hi;
  this.state = { lf: clamp(+o.lf || 0, 0, 1), hf: clamp(+o.hf || 0, 0, 1) };
  this.drag = null;
  this.frame = 0;
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
  this.annLFi = clamp(Math.round(ANN_LF_FRAC * (n - 1)), 0, n - 1);
  this.annHFi = clamp(Math.round(ANN_HF_FRAC * (n - 1)), 0, n - 1);

  this.build();
  this.bind();
  this.render();
}

Instance.prototype.morph = function (lf, hf) {
  return morph(this.sLo, this.sHi, this.tLo, this.tHi, lf, hf);
};

Instance.prototype.y = function (v) {
  return PLOT.b - clamp(v, this.vmin, this.vmax) * this.span;
};

Instance.prototype.q = function (name) {
  return this.svg.querySelector('[data-lieq="' + name + '"]');
};

Instance.prototype.build = function () {
  var doc = this.doc, o = this.opts, wrap, stage, i, faders;

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
  stage = wrap.querySelector('.lieq-stage');
  this.stage = stage;
  this.svg = wrap.querySelector('svg');

  this.el = {
    sArea: this.q('sArea'), sCore: this.q('sCore'), sFlow: this.q('sFlow'),
    tHalo: this.q('tHalo'), tCore: this.q('tCore'), tFlow: this.q('tFlow'),
    annLF: this.q('annLF'), annHF: this.q('annHF'),
    gapStem: this.q('gapStem'), gapTop: this.q('gapTop'), gapBot: this.q('gapBot'),
    hfDrop: this.q('hfDrop'), hfHalo: this.q('hfHalo'), hfDot: this.q('hfDot'),
    gapValue: this.q('gapValue')
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

  /* The teacher never moves: draw it once. */
  this.drawTeacher();
};

Instance.prototype.drawTeacher = function () {
  var ys = new Array(this.n), i, d;
  for (i = 0; i < this.n; i++) { ys[i] = this.y(this.teacher[i]); }
  d = catmullPath(this.xs, ys);
  if (this.el.tHalo) { this.el.tHalo.setAttribute('d', d); }
  if (this.el.tCore) { this.el.tCore.setAttribute('d', d); }
  if (this.el.tFlow) { this.el.tFlow.setAttribute('d', d); }
  this.teacherYs = ys;
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

Instance.prototype.bind = function () {
  var self = this, key, f;

  this.onDown = function (e) { self.pointerDown(e); };
  this.onMove = function (e) { self.pointerMove(e); };
  this.onUp = function (e) { self.pointerUp(e); };
  this.onKey = function (e) { self.keyDown(e); };

  for (key in this.faders) {
    if (!Object.prototype.hasOwnProperty.call(this.faders, key)) { continue; }
    f = this.faders[key];
    f.group.addEventListener('pointerdown', this.onDown, false);
    f.group.addEventListener('keydown', this.onKey, false);
  }

  /* Move / up / cancel live on the window so a drag survives the pointer
   * leaving the SVG even if setPointerCapture is unavailable. */
  var win = this.doc.defaultView || (typeof window !== 'undefined' ? window : null);
  this.win = win;
  if (win) {
    win.addEventListener('pointermove', this.onMove, false);
    win.addEventListener('pointerup', this.onUp, false);
    win.addEventListener('pointercancel', this.onUp, false);
  }
};

Instance.prototype.pointerDown = function (e) {
  var group = e.currentTarget,
      key = group.getAttribute('data-lieq-fader'),
      p, knobY;
  if (this.drag) { return; }
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
  var d = this.drag;
  if (!d) { return; }
  if (d.id !== undefined && e.pointerId !== undefined && e.pointerId !== d.id) { return; }
  if (e.cancelable) { e.preventDefault(); }
  this.setValue(d.key, this.valueFromY(this.toSvg(e).y + d.dy));
};

Instance.prototype.pointerUp = function (e) {
  var d = this.drag;
  if (!d) { return; }
  if (d.id !== undefined && e.pointerId !== undefined && e.pointerId !== d.id) { return; }
  try { d.group.releasePointerCapture(d.id); } catch (err) { /* ignore */ }
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
  if (lf !== undefined && lf !== null) {
    v = clamp(+lf || 0, 0, 1);
    if (v !== this.state.lf) { this.state.lf = v; changed = true; }
  }
  if (hf !== undefined && hf !== null) {
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
      vals = this.morph(st.lf, st.hf),
      i, d, area, conv, gap, dv,
      ax, tY, sY, fadeLF, fadeHF, key, f, knobY;

  for (i = 0; i < n; i++) { this.ys[i] = this.y(vals[i]); }
  d = catmullPath(this.xs, this.ys);
  area = d + 'L' + n1(this.xs[n - 1]) + ' ' + PLOT.b + 'L' + n1(this.xs[0]) + ' ' + PLOT.b + 'Z';

  if (el.sCore) { el.sCore.setAttribute('d', d); }
  if (el.sFlow) { el.sFlow.setAttribute('d', d); }
  if (el.sArea) { el.sArea.setAttribute('d', area); }

  /* the suppressed-vs-clean opacity language: the student brightens as it converges */
  conv = (st.lf + st.hf) / 2;
  if (el.sCore) { el.sCore.setAttribute('opacity', n1(0.62 + 0.38 * conv) + ''); }
  if (el.sFlow) { el.sFlow.setAttribute('opacity', n1(0.75 + 0.25 * conv) + ''); }

  /* peak gap readout — a property of the two drawn curves, nothing more */
  gap = 0;
  for (i = 0; i < n; i++) {
    dv = vals[i] - this.teacher[i];
    if (dv < 0) { dv = -dv; }
    if (dv > gap) { gap = dv; }
  }
  if (el.gapValue) { el.gapValue.textContent = (Math.round(gap * 100) / 100).toFixed(2); }

  /* annotation 1 — the measured low-frequency excess, as a dimension bar */
  fadeLF = clamp(1 - st.lf * 1.15, 0, 1);
  if (el.annLF) { el.annLF.setAttribute('opacity', n1(fadeLF) + ''); }
  i = this.annLFi;
  ax = this.xs[i]; sY = this.ys[i]; tY = this.teacherYs[i];
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

  if (typeof this.opts.onChange === 'function') {
    try { this.opts.onChange(st.lf, st.hf, this); } catch (err) { /* host's problem */ }
  }
  return this;
};

Instance.prototype.destroy = function () {
  var key, f;
  if (this.destroyed) { return; }
  this.destroyed = true;
  if (this.frame && this.win && this.win.cancelAnimationFrame) {
    this.win.cancelAnimationFrame(this.frame);
  }
  this.frame = 0;
  for (key in this.faders) {
    if (!Object.prototype.hasOwnProperty.call(this.faders, key)) { continue; }
    f = this.faders[key];
    f.group.removeEventListener('pointerdown', this.onDown, false);
    f.group.removeEventListener('keydown', this.onKey, false);
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
};

/* ==============================================================
 * 7.  Entry points
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
  defaults: DEFAULTS,
  curves: { teacher: parseCurve(TEACHER_SRC), student: parseCurve(STUDENT_SRC) },
  math: {
    clamp: clamp,
    lerp: lerp,
    smooth: smooth,
    split: split,
    morph: morph,
    catmullPath: catmullPath,
    parseCurve: parseCurve
  },
  geometry: {
    VB_W: VB_W, VB_H: VB_H, PLOT: PLOT, PLOT_HEAD: PLOT_HEAD,
    RACK: RACK, TRACK: TRACK, FADER_X: FADER_X, INSET: VB_INSET, LEGEND: LEGEND,
    ANN: { lfTextX: ANN_LF_TEXT_X, hfTextX: ANN_HF_TEXT_X, gapCap: GAP_CAP,
           lfFrac: ANN_LF_FRAC, hfFrac: ANN_HF_FRAC }
  }
};

});
