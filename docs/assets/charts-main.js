/*!
 * charts-main.js — LiEqualizer project page, main result charts.
 *
 * Native inline SVG + plain HTML tables. No framework, no bundler, no CDN,
 * no chart library. ES5-compatible. Safe to load with a plain <script src>.
 *
 * Public API (window.LiEQCharts):
 *   renderMainTable(el, opts)     3x3 cross-sensor matrix as three grouped-bar panels
 *   renderGainStrip(el, opts)     the nine LiEqualizer-over-LiSeer gains, sorted
 *   renderAblation(el, opts)      additive ablation, HTML table
 *   renderKDBaselines(el, opts)   distillation-objective comparison, HTML table
 *   renderAll(opts)               renders into the four default mount ids
 *   fallbackHTML(which)           plain data table markup for <noscript>
 *   DATA                          every hard-coded number, for inspection
 *
 * The first argument may be an element or an element id. A missing mount is a no-op (returns
 * null) and never throws.
 *
 * Numbers are transcribed from LiEqualizer_main.tex:
 *   Table 1  (\label{tab:ST results},  tex lines 619-671)
 *   Table 2  (\label{tab:kdbaselines}, tex lines 720-735)
 *   Table 3  (\label{tab:ablation},    tex lines 746-762)
 * See charts-main.README.md for the one place this file deliberately departs
 * from the printed tex (Table 2, the SK->nuScenes delta on the last row).
 */
(function (root) {
  'use strict';

  var doc = (typeof document !== 'undefined') ? document : null;

  /* ====================================================================== *
   * 1. DATA — hard-coded from the manuscript. Do not "tidy" these numbers. *
   * ====================================================================== */

  /* Training sources (Table 1 row groups, in the paper's order). */
  var SOURCES = [
    { key: 'SK', name: 'SemanticKITTI', short: 'SK', tag: 'only SK' },
    { key: 'NU', name: 'nuScenes', short: 'NU', tag: 'only NU' },
    { key: 'WA', name: 'Waymo', short: 'WA', tag: 'only Waymo' }
  ];

  /* Test sets (Table 1 columns, in the paper's order). */
  var TESTS = [
    { key: 'SK', name: 'SemanticKITTI', short: 'SK' },
    { key: 'NU', name: 'nuScenes', short: 'NU' },
    { key: 'WA', name: 'Waymo', short: 'WA' }
  ];

  /* Methods (Table 1 rows within a group).
   *
   * deltaRef records WHAT the bracketed delta printed in Table 1 is measured
   * against, because it is not the same for all three rows:
   *   'own-source'  P-baseline's green figures are the collapse against that
   *                 same row's own-sensor score (e.g. 19.4 vs 71.6 = -52.2).
   *   'p-baseline'  LiSeer and LiEqualizer's red figures are the improvement
   *                 over P-baseline in the same cell (same source, same test).
   * Never relabel them.
   */
  var METHODS = [
    { key: 'base', name: 'P-baseline', deltaRef: 'own-source' },
    { key: 'liseer', name: 'LiSeer-P-RSCL', deltaRef: 'p-baseline' },
    { key: 'lieq', name: 'LiEqualizer-P', deltaRef: 'p-baseline' }
  ];

  /* Table 1 — S -> U domain generalization, zero-shot mIoU.
   * MAIN[source][method][test] = { v: mIoU, d: the delta printed in Table 1 }
   */
  var MAIN = {
    SK: {
      base: { SK: { v: 71.6, d: null }, NU: { v: 19.4, d: -52.2 }, WA: { v: 17.3, d: -54.3 } },
      liseer: { SK: { v: 68.1, d: -3.5 }, NU: { v: 56.8, d: 37.4 }, WA: { v: 54.6, d: 37.3 } },
      lieq: { SK: { v: 75.5, d: 3.9 }, NU: { v: 58.3, d: 38.9 }, WA: { v: 64.4, d: 47.1 } }
    },
    NU: {
      base: { SK: { v: 20.5, d: -49.9 }, NU: { v: 70.4, d: null }, WA: { v: 18.1, d: -52.3 } },
      liseer: { SK: { v: 54.4, d: 33.9 }, NU: { v: 66.5, d: -3.9 }, WA: { v: 53.8, d: 35.7 } },
      lieq: { SK: { v: 55.3, d: 34.8 }, NU: { v: 76.1, d: 5.7 }, WA: { v: 61.4, d: 43.3 } }
    },
    WA: {
      base: { SK: { v: 24.7, d: -36.2 }, NU: { v: 5.4, d: -55.5 }, WA: { v: 60.9, d: null } },
      liseer: { SK: { v: 52.3, d: 27.6 }, NU: { v: 49.0, d: 43.6 }, WA: { v: 62.2, d: 1.3 } },
      lieq: { SK: { v: 63.8, d: 39.1 }, NU: { v: 58.5, d: 53.1 }, WA: { v: 79.2, d: 18.3 } }
    }
  };

  /* LiEqualizer-P minus LiSeer-P-RSCL for each of the nine cells of Table 1.
   * Written out rather than computed, so the page cannot drift from the table
   * through floating-point arithmetic. Sorted descending.
   * The cross-sensor maximum is +11.5 at Waymo -> SemanticKITTI (63.8 - 52.3),
   * which is the figure the abstract quotes. Waymo -> Waymo is larger (+17.0)
   * but is same-sensor, which is why the claim says "cross-sensor".
   */
  var GAINS = [
    { src: 'WA', test: 'WA', g: 17.0, same: true },
    { src: 'WA', test: 'SK', g: 11.5, same: false, callout: true },
    { src: 'SK', test: 'WA', g: 9.8, same: false },
    { src: 'NU', test: 'NU', g: 9.6, same: true },
    { src: 'WA', test: 'NU', g: 9.5, same: false },
    { src: 'NU', test: 'WA', g: 7.6, same: false },
    { src: 'SK', test: 'SK', g: 7.4, same: true },
    { src: 'SK', test: 'NU', g: 1.5, same: false },
    { src: 'NU', test: 'SK', g: 0.9, same: false }
  ];

  /* Table 3 — additive ablation. mark: 'best' = \textbf, 'second' = \underline,
   * exactly as the tex marks them (the tex marks the whole "74.6 (+6.5)" unit,
   * not just the mIoU, so the mark wraps value and delta together).
   * The SK->SK column carries two underlines: 74.6 occurs twice and the tex
   * underlines both.
   */
  var ABLATION = {
    number: 'Table 3',
    caption: 'What each part of the objective adds, under the standard single-pass protocol.',
    rowHead: 'Objective',
    cols: ['SK→SK', 'SK→Waymo', 'SK→nuScenes'],
    rows: [
      { label: 'LiSeer-P-SK, no distillation', ruleAfter: true,
        cells: [{ v: 68.1 }, { v: 54.6 }, { v: 56.8, mark: 'second' }] },
      { label: '+ LF alone',
        cells: [{ v: 74.5, d: 6.4 }, { v: 65.4, d: 10.8, mark: 'best' }, { v: 54.4, d: -2.4 }] },
      { label: '+ HF and soft label alone',
        cells: [{ v: 73.5, d: 5.4 }, { v: 63.7, d: 9.1 }, { v: 53.0, d: -3.8 }] },
      { label: '+ both branches, ungated',
        cells: [{ v: 73.8, d: 5.7 }, { v: 64.0, d: 9.4 }, { v: 53.3, d: -3.5 }] },
      { label: '+ gates, matching only',
        cells: [{ v: 74.6, d: 6.5, mark: 'second' }, { v: 64.1, d: 9.5 }, { v: 53.9, d: -2.9 }] },
      { label: '+ gates, suppressing only',
        cells: [{ v: 74.2, d: 6.1 }, { v: 64.2, d: 9.6 }, { v: 53.1, d: -3.7 }] },
      { label: 'with LF suppressed towards the teacher', indent: true, ruleAfter: true,
        cells: [{ v: 74.6, d: 6.5, mark: 'second' }, { v: 64.0, d: 9.4 }, { v: 53.7, d: -3.1 }] },
      { label: 'LiEqualizer (gates, matching and suppressing)', ours: true,
        cells: [{ v: 75.5, d: 7.4, mark: 'best' }, { v: 64.4, d: 9.8, mark: 'second' }, { v: 58.3, d: 1.5, mark: 'best' }] }
    ],
    note: 'The table is read downwards, from the undistilled student to the full objective, and each bracketed figure is the difference from the first row. That row is the randomized-sampling student of Table 1 trained on SemanticKITTI, quoted from the original publication. The two single-branch rows are alternatives rather than steps: one keeps the low-frequency branch alone, the other the high-frequency branch that shares its switch. The three rows after them hold both branches and differ in how much of the gate they carry.'
  };

  /* Table 2 — distillation objectives compared under one pipeline.
   *
   * One departure from the printed tex, flagged on the page and in the README:
   * the tex prints (+0.5) on the last row's SK->nuScenes cell, but 58.3 - 56.8
   * = 1.5 and Table 3 prints (+1.5) for the identical pair. The page shows
   * (+1.5) and says so in the footnote.
   */
  var KD = {
    number: 'Table 2',
    caption: 'Distillation objectives compared under one pipeline (source-domain and zero-shot mIoU).',
    rowHead: 'Objective',
    attachHead: 'Attached to',
    cols: ['SK→SK', 'SK→Waymo', 'SK→nuScenes'],
    rows: [
      { label: 'student (no distillation)', attached: '—',
        cells: [{ v: 68.1 }, { v: 54.6 }, { v: 56.8, mark: 'second' }] },
      { label: '+ feature mimicking [LiDAR Distillation]', attached: 'TPV planes',
        cells: [{ v: 73.7, d: 5.6 }, { v: 64.8, d: 10.2, mark: 'best' }, { v: 54.0, d: -2.8 }] },
      { label: '+ soft-label KL [LESS; Hinton]', attached: 'Voxel logits',
        cells: [{ v: 74.1, d: 6.0 }, { v: 64.1, d: 9.5 }, { v: 53.6, d: -3.2 }] },
      { label: '+ both, the two-term pointwise form', attached: 'Both',
        cells: [{ v: 73.6, d: 5.5 }, { v: 64.0, d: 9.4 }, { v: 53.9, d: -2.9 }] },
      { label: '+ PVKD', attached: 'Both',
        cells: [{ v: 73.8, d: 5.7 }, { v: 64.5, d: 9.9, mark: 'second' }, { v: 54.1, d: -2.7 }] },
      { label: '+ CWD', attached: 'TPV planes',
        cells: [{ v: 74.2, d: 6.1, mark: 'second' }, { v: 64.2, d: 9.6 }, { v: 53.7, d: -3.1 }] },
      { label: '+ LiEqualizer (ours)', attached: 'Both', ours: true,
        cells: [{ v: 75.5, d: 7.4, mark: 'best' }, { v: 64.4, d: 9.8 }, { v: 58.3, d: 1.5, mark: 'best', footnote: 1 }] }
    ],
    note: 'All competing objectives are implemented by us within this pipeline. Where the released code runs on this architecture we follow it and its default hyper-parameters; the substitutions we made are listed in Appendix E. The first row is the same student without any distillation, and the bracketed figures are differences against that row.',
    footnotes: ['Table 2 of the manuscript prints (+0.5) in this cell. 58.3 − 56.8 = 1.5, and Table 3 prints (+1.5) for the same pair, so the page shows (+1.5).']
  };

  /* ====================================================================== *
   * 2. TOKENS                                                              *
   * ====================================================================== */

  /* Series colours. Brand values from the paper's own figure palette
   * (TPV_SPECTRAL/scripts/tpvstyle.py) and from the dark-card tokens of the
   * sibling LiSeer page. Data colours are emitted as attributes so one page
   * can hold a light and a dark copy of the same chart. */
  var THEMES = {
    light: {
      series: { base: '#8f8d88', liseer: '#eb6834', lieq: '#2a78d6' },
      cross: '#2a78d6',
      same: '#86b6ef',
      diagFill: '#eef3f8',
      hatch: 'rgba(52,72,96,.20)',
      grid: 'rgba(52,72,96,.12)',
      axis: 'rgba(52,72,96,.28)',
      accent: '#2a78d6'
    },
    dark: {
      series: { base: '#8f8d88', liseer: '#e58236', lieq: '#56adf0' },
      cross: '#56adf0',
      same: '#2a78d6',
      diagFill: 'rgba(219,231,245,.07)',
      hatch: 'rgba(219,231,245,.18)',
      grid: 'rgba(219,231,245,.14)',
      axis: 'rgba(219,231,245,.30)',
      accent: '#56adf0'
    }
  };

  var CSS = [
    '.lq-fig{margin:0;max-width:var(--maxw,1180px);font-family:var(--sans,"Inter",system-ui,sans-serif);color:var(--text,#17202c);opacity:1}',
    '.lq-fig *{box-sizing:border-box}',
    '@media (prefers-reduced-motion: no-preference){',
    '@keyframes lq-fade{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}',
    '.lq-fig{animation:lq-fade .42s ease-out both}}',

    /* heads and captions */
    '.lq-eyebrow{display:inline-block;font-family:var(--mono,"JetBrains Mono",ui-monospace,monospace);font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:#377fba;margin:0 0 .5em}',
    '.lq-fig h3.lq-head{font-family:var(--disp,"Space Grotesk",var(--sans,"Inter",system-ui,sans-serif));font-size:clamp(17px,1.5vw,21px);line-height:1.25;font-weight:600;margin:0 0 .35em;color:var(--text,#17202c)}',
    '.lq-sub{font-size:14px;line-height:1.55;color:var(--muted,#647184);margin:0 0 1.1em;max-width:62ch}',
    '.lq-note{font-size:12.5px;line-height:1.6;color:var(--muted,#647184);margin:.9em 0 0;max-width:78ch}',
    '.lq-note b{font-weight:600;color:var(--text,#17202c)}',
    '.lq-fn{font-size:12px;line-height:1.6;color:var(--muted,#647184);margin:.45em 0 0;max-width:78ch}',
    '.lq-fn sup{color:#377fba;font-weight:600}',

    /* legend */
    '.lq-legend{display:flex;flex-wrap:wrap;gap:.45em 1.1em;align-items:center;margin:0 0 1em;padding:0;list-style:none;font-family:var(--mono,"JetBrains Mono",ui-monospace,monospace);font-size:11.5px;color:var(--muted,#647184)}',
    '.lq-legend li{display:flex;align-items:center;gap:.5em;margin:0}',
    '.lq-sw{width:14px;height:10px;border-radius:2px;flex:0 0 auto;display:inline-block}',
    '.lq-legend b{font-weight:500;color:var(--text,#17202c)}',

    /* panel grid */
    /* 1-up below 960px, 3-up above. The SVG is capped so a single-column panel
       never scales its type past ~1.4x, which keeps the 8.4px axis labels
       legible at 360px and at 1180px alike. */
    '.lq-panels{display:grid;grid-template-columns:1fr;gap:20px 18px}',
    '@media (min-width:960px){.lq-panels{grid-template-columns:repeat(3,minmax(0,1fr))}}',
    '.lq-panel{min-width:0}',
    '.lq-panel .lq-svg{max-width:420px}',
    '.lq-strip .lq-svg{max-width:900px}',
    '.lq-panel h4{font-family:var(--mono,"JetBrains Mono",ui-monospace,monospace);font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted,#647184);margin:0 0 .55em;font-weight:500}',
    '.lq-panel h4 b{color:var(--text,#17202c);font-weight:600;letter-spacing:.06em}',

    /* svg */
    '.lq-svg{display:block;width:100%;height:auto;overflow:visible;-webkit-tap-highlight-color:transparent}',
    '.lq-svg text{font-family:var(--mono,"JetBrains Mono",ui-monospace,monospace);font-variant-numeric:tabular-nums}',
    '.lq-tick,.lq-unit{fill:var(--muted,#647184)}',
    '.lq-xlab{fill:var(--text,#17202c)}',
    '.lq-xlab.is-diag{fill:var(--muted,#647184)}',
    '.lq-xtag{fill:var(--muted,#647184);letter-spacing:.06em}',
    '.lq-val{fill:var(--text,#17202c)}',
    '.lq-val.is-dim{fill:var(--muted,#647184)}',
    '.lq-rowlab{fill:var(--text,#17202c)}',
    '.lq-callout-t{fill:var(--muted,#647184);letter-spacing:.07em}',
    '.lq-svg[data-lq-theme="dark"] .lq-tick,.lq-svg[data-lq-theme="dark"] .lq-unit,',
    '.lq-svg[data-lq-theme="dark"] .lq-xtag,.lq-svg[data-lq-theme="dark"] .lq-callout-t,',
    '.lq-svg[data-lq-theme="dark"] .lq-xlab.is-diag,.lq-svg[data-lq-theme="dark"] .lq-val.is-dim{fill:#9cc8f3}',
    '.lq-svg[data-lq-theme="dark"] .lq-xlab,.lq-svg[data-lq-theme="dark"] .lq-val,',
    '.lq-svg[data-lq-theme="dark"] .lq-rowlab{fill:#dbe7f5}',
    '.lq-bar,.lq-gbar{shape-rendering:geometricPrecision}',
    '.lq-hit{fill:transparent;cursor:default}',

    /* hero stat for the gain strip */
    '.lq-hero{display:flex;flex-wrap:wrap;align-items:baseline;gap:.15em .7em;margin:0 0 .9em;padding:.75em .9em;border:1px solid var(--line,rgba(52,72,96,.12));border-left:3px solid #2a78d6;border-radius:0 12px 12px 0;background:var(--mist,#eef3f8)}',
    '.lq-hero-n{font-family:var(--sans,"Inter",system-ui,sans-serif);font-size:clamp(26px,3.2vw,38px);font-weight:650;line-height:1;color:#2a78d6;letter-spacing:-.01em}',
    '.lq-hero-l{font-family:var(--mono,"JetBrains Mono",ui-monospace,monospace);font-size:11.5px;letter-spacing:.07em;text-transform:uppercase;color:var(--muted,#647184)}',
    '.lq-hero-d{font-size:13px;color:var(--text,#17202c);flex:1 1 100%;margin-top:.15em}',
    '.lq-hero-d b{font-weight:600}',

    /* tables */
    '.lq-tablewrap{max-width:900px;overflow-x:auto;-webkit-overflow-scrolling:touch;border:1px solid var(--line,rgba(52,72,96,.12));border-radius:var(--r,18px);background:#fff;box-shadow:var(--shadow,0 18px 48px -30px rgba(38,61,91,.35))}',
    '.lq-tablewrap:focus-visible{outline:2px solid #2f78bf;outline-offset:2px}',
    '.lq-table{border-collapse:collapse;width:100%;min-width:430px;font-size:13.5px;line-height:1.45}',
    '.lq-table th,.lq-table td{padding:.62em .8em;text-align:right;white-space:nowrap;border-bottom:1px solid var(--line,rgba(52,72,96,.12))}',
    '.lq-table thead th:first-child,.lq-table thead th.lq-attach{text-align:left}',
    '.lq-table thead th{font-family:var(--mono,"JetBrains Mono",ui-monospace,monospace);font-size:11px;letter-spacing:.07em;text-transform:uppercase;font-weight:500;color:var(--muted,#647184);background:var(--mist,#eef3f8);border-bottom:1px solid var(--line,rgba(52,72,96,.12))}',
    '.lq-table tbody th{text-align:left;font-weight:400;color:var(--text,#17202c);white-space:normal;min-width:15ch}',
    '.lq-table td{font-family:var(--mono,"JetBrains Mono",ui-monospace,monospace);font-variant-numeric:tabular-nums;color:var(--text,#17202c)}',
    '.lq-table td.lq-attach{font-family:var(--sans,"Inter",system-ui,sans-serif);text-align:left;color:var(--muted,#647184);font-size:12.5px}',
    '.lq-table tbody tr:last-child th,.lq-table tbody tr:last-child td{border-bottom:0}',
    '.lq-table tr.lq-rule th,.lq-table tr.lq-rule td{border-bottom:1px solid rgba(52,72,96,.26)}',
    '.lq-table tr.lq-ours{background:rgba(42,120,214,.06)}',
    '.lq-table tr.lq-ours th{font-weight:600}',
    '.lq-table tr.lq-ours th:first-child{box-shadow:inset 3px 0 0 #2a78d6}',
    '.lq-table th.lq-indent{padding-left:2.1em;color:var(--muted,#647184)}',
    '.lq-best{font-weight:700}',
    '.lq-second{text-decoration:underline;text-decoration-thickness:1px;text-underline-offset:3px;text-decoration-color:rgba(52,72,96,.45)}',
    '.lq-d{margin-left:.35em;font-size:.92em}',
    '.lq-d-gain{color:#DC2626}',
    '.lq-d-drop{color:#15803D}',
    '.lq-table td sup{color:#377fba;font-weight:600;margin-left:.1em}',
    '@media (max-width:620px){',
    '.lq-table{min-width:0;font-size:12.5px}',
    '.lq-table th,.lq-table td{padding:.55em .55em}',
    '.lq-table tbody th{min-width:11ch;font-size:12px}',
    '.lq-table td .lq-d{display:block;margin-left:0}',
    '.lq-table td.lq-attach{display:none}.lq-table th.lq-attach{display:none}}',

    /* on-dark variant, for a figure sitting on the dark instrument card */
    '.lq-fig[data-lq-theme="dark"]{color:#dbe7f5}',
    '.lq-fig[data-lq-theme="dark"] h3.lq-head{color:#dbe7f5}',
    '.lq-fig[data-lq-theme="dark"] .lq-sub,.lq-fig[data-lq-theme="dark"] .lq-note,',
    '.lq-fig[data-lq-theme="dark"] .lq-fn,.lq-fig[data-lq-theme="dark"] .lq-legend,',
    '.lq-fig[data-lq-theme="dark"] .lq-panel h4,.lq-fig[data-lq-theme="dark"] .lq-hero-l{color:#9cc8f3}',
    '.lq-fig[data-lq-theme="dark"] .lq-note b,.lq-fig[data-lq-theme="dark"] .lq-legend b,',
    '.lq-fig[data-lq-theme="dark"] .lq-panel h4 b,.lq-fig[data-lq-theme="dark"] .lq-hero-d,',
    '.lq-fig[data-lq-theme="dark"] .lq-hero-d b{color:#dbe7f5}',
    '.lq-fig[data-lq-theme="dark"] .lq-eyebrow{color:#56adf0}',
    '.lq-fig[data-lq-theme="dark"] .lq-hero{background:rgba(86,173,240,.10);border-color:rgba(219,231,245,.16);border-left-color:#56adf0}',
    '.lq-fig[data-lq-theme="dark"] .lq-hero-n{color:#bfe0ff}',
    '.lq-fig[data-lq-theme="dark"] .lq-tablewrap{background:#0c1322;border-color:rgba(219,231,245,.16);box-shadow:none}',
    '.lq-fig[data-lq-theme="dark"] .lq-table thead th{background:rgba(219,231,245,.06);color:#9cc8f3;border-bottom-color:rgba(219,231,245,.16)}',
    '.lq-fig[data-lq-theme="dark"] .lq-table th,.lq-fig[data-lq-theme="dark"] .lq-table td{border-bottom-color:rgba(219,231,245,.12)}',
    '.lq-fig[data-lq-theme="dark"] .lq-table tbody th,.lq-fig[data-lq-theme="dark"] .lq-table td{color:#dbe7f5}',
    '.lq-fig[data-lq-theme="dark"] .lq-table td.lq-attach{color:#9cc8f3}',
    '.lq-fig[data-lq-theme="dark"] .lq-table tr.lq-ours{background:rgba(86,173,240,.10)}',
    '.lq-fig[data-lq-theme="dark"] .lq-table tr.lq-ours th:first-child{box-shadow:inset 3px 0 0 #56adf0}',
    '.lq-fig[data-lq-theme="dark"] .lq-d-gain{color:#FCA5A5}',
    '.lq-fig[data-lq-theme="dark"] .lq-d-drop{color:#4ADE80}',
    '.lq-fig[data-lq-theme="dark"] .lq-second{text-decoration-color:rgba(219,231,245,.5)}',
    '.lq-fig[data-lq-theme="dark"] .lq-fn sup,.lq-fig[data-lq-theme="dark"] .lq-table td sup{color:#56adf0}',

    /* screen-reader-only data tables */
    '.lq-sr{position:absolute!important;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0 0 0 0);clip-path:inset(50%);white-space:nowrap;border:0}',
    '.lq-srtable{width:1px;table-layout:fixed}',

    /* tooltip */
    '.lq-tip{position:fixed;z-index:60;pointer-events:none;opacity:0;transition:opacity .12s linear;max-width:260px;padding:.5em .65em;border-radius:10px;background:#0c1322;color:#dbe7f5;font-family:var(--mono,"JetBrains Mono",ui-monospace,monospace);font-size:11.5px;line-height:1.5;box-shadow:0 10px 30px -12px rgba(8,16,30,.7)}',
    '.lq-tip[data-on="1"]{opacity:1}',
    '.lq-tip b{color:#fff;font-weight:600}',
    '.lq-tip i{font-style:normal;color:#9cc8f3}',
    '@media (prefers-reduced-motion: reduce){.lq-tip{transition:none}}'
  ].join('\n');

  /* ====================================================================== *
   * 3. SMALL HELPERS                                                       *
   * ====================================================================== */

  var MINUS = '−';
  var ARROW = '→';
  var uidN = 0;

  function uid(p) { uidN += 1; return 'lq' + p + uidN; }
  function r2(n) { return Math.round(n * 100) / 100; }
  function f1(v) { return (Math.round(v * 10) / 10).toFixed(1); }
  function signed(v) { return (v > 0 ? '+' : (v < 0 ? MINUS : '')) + f1(Math.abs(v)); }

  function esc(s) {
    return String(s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  function byKey(list, key) {
    for (var i = 0; i < list.length; i++) { if (list[i].key === key) { return list[i]; } }
    return null;
  }
  function srcOf(k) { return byKey(SOURCES, k); }
  function testOf(k) { return byKey(TESTS, k); }
  function methodOf(k) { return byKey(METHODS, k); }

  function themeAttr(opts) {
    return (opts && opts.theme === 'dark') ? ' data-lq-theme="dark"' : '';
  }

  function theme(opts) {
    var name = (opts && opts.theme === 'dark') ? 'dark' : 'light';
    return { name: name, t: THEMES[name] };
  }

  function resolve(el) {
    if (!el) { return null; }
    if (typeof el === 'string') { return doc ? doc.getElementById(el) : null; }
    if (el.nodeType === 1) { return el; }
    return null;
  }

  function injectCSS() {
    if (!doc || doc.getElementById('lq-charts-main-css')) { return; }
    var st = doc.createElement('style');
    st.id = 'lq-charts-main-css';
    st.appendChild(doc.createTextNode(CSS));
    (doc.head || doc.documentElement).appendChild(st);
  }

  /* <rect> with a rounded data-end and a square baseline: the rect is pushed
   * rad px past the baseline and the group is clipped at the baseline. */
  function barRect(cls, attrs) {
    var out = ['<rect class="' + cls + '"'];
    for (var k in attrs) {
      if (attrs.hasOwnProperty(k) && attrs[k] !== null && attrs[k] !== undefined) {
        out.push(' ' + k + '="' + esc(attrs[k]) + '"');
      }
    }
    out.push('/>');
    return out.join('');
  }

  function hatchDef(id, color) {
    return '<pattern id="' + id + '" width="6" height="6" patternUnits="userSpaceOnUse"' +
      ' patternTransform="rotate(45)">' +
      '<line x1="0" y1="0" x2="0" y2="6" stroke="' + color + '" stroke-width="1.2"/>' +
      '</pattern>';
  }

  /* ====================================================================== *
   * 4. DATA TABLES (screen-reader view and <noscript> fallback)            *
   * ====================================================================== */

  function cellText(c) {
    var s = f1(c.v);
    if (c.d !== null && c.d !== undefined) { s += ' (' + signed(c.d) + ')'; }
    return s;
  }

  function mainDataTable(srOnly) {
    var h = [];
    h.push((srOnly ? '<div class="lq-sr"><table class="lq-srtable">' : '<table class="lq-table">'));
    h.push('<caption>Table 1. S' + ARROW + 'U domain generalization, zero-shot mIoU. ' +
      'Bracketed figures are the deltas printed in the manuscript: for P-baseline the drop ' +
      'against that row’s own-sensor score, for LiSeer-P and LiEqualizer-P the gain over ' +
      'P-baseline in the same cell.</caption>');
    h.push('<thead><tr><th scope="col">Trained on</th><th scope="col">Method</th>');
    for (var t = 0; t < TESTS.length; t++) {
      h.push('<th scope="col">Test on ' + esc(TESTS[t].name) + ' mIoU</th>');
    }
    h.push('</tr></thead><tbody>');
    for (var s = 0; s < SOURCES.length; s++) {
      var sk = SOURCES[s].key;
      for (var m = 0; m < METHODS.length; m++) {
        var mk = METHODS[m].key;
        h.push('<tr><th scope="row">' + esc(SOURCES[s].tag) + '</th><td class="lq-attach">' +
          esc(METHODS[m].name) + (mk === 'lieq' ? ' (ours)' : '') + '</td>');
        for (var c = 0; c < TESTS.length; c++) {
          var cell = MAIN[sk][mk][TESTS[c].key];
          h.push('<td>' + cellText(cell) +
            (sk === TESTS[c].key ? ' <span class="lq-sr">same sensor</span>' : '') + '</td>');
        }
        h.push('</tr>');
      }
    }
    h.push('</tbody></table>' + (srOnly ? '</div>' : ''));
    return h.join('');
  }

  function gainDataTable(srOnly) {
    var h = [];
    h.push((srOnly ? '<div class="lq-sr"><table class="lq-srtable">' : '<table class="lq-table">'));
    h.push('<caption>LiEqualizer-P minus LiSeer-P-RSCL for each of the nine cells of Table 1, ' +
      'sorted. The largest cross-sensor gain is +11.5 mIoU on Waymo' + ARROW +
      'SemanticKITTI.</caption>');
    h.push('<thead><tr><th scope="col">Trained on</th><th scope="col">Tested on</th>' +
      '<th scope="col">Pair</th><th scope="col">LiSeer-P mIoU</th>' +
      '<th scope="col">LiEqualizer-P mIoU</th><th scope="col">Gain</th></tr></thead><tbody>');
    for (var i = 0; i < GAINS.length; i++) {
      var r = GAINS[i];
      h.push('<tr><th scope="row">' + esc(srcOf(r.src).name) + '</th>' +
        '<td class="lq-attach">' + esc(testOf(r.test).name) + '</td>' +
        '<td class="lq-attach">' + (r.same ? 'same sensor' : 'cross-sensor') + '</td>' +
        '<td>' + f1(MAIN[r.src].liseer[r.test].v) + '</td>' +
        '<td>' + f1(MAIN[r.src].lieq[r.test].v) + '</td>' +
        '<td>' + signed(r.g) + '</td></tr>');
    }
    h.push('</tbody></table>' + (srOnly ? '</div>' : ''));
    return h.join('');
  }

  /* ====================================================================== *
   * 5. CHART 1 — the 3x3 cross-sensor matrix                               *
   * ====================================================================== */

  var P = {
    W: 304, H: 238,
    padL: 28, padR: 8, padT: 20, padB: 46,
    yMax: 80, ticks: [0, 20, 40, 60, 80],
    barW: 22, barGap: 2, rad: 4
  };
  P.plotW = P.W - P.padL - P.padR;        /* 268 */
  P.plotH = P.H - P.padT - P.padB;        /* 172 */
  P.y0 = P.padT + P.plotH;                /* 192 */
  P.band = P.plotW / TESTS.length;        /* 89.333 */
  P.inner = METHODS.length * P.barW + (METHODS.length - 1) * P.barGap; /* 70 */
  P.off = (P.band - P.inner) / 2;         /* 9.667 */

  function yOf(v) { return P.y0 - (v / P.yMax) * P.plotH; }

  function panelDesc(sk) {
    var src = srcOf(sk);
    var parts = ['Grouped bars, three methods per test set, zero-shot mIoU on a 0 to ' +
      P.yMax + ' scale. Student trained on ' + src.name + ' only.'];
    for (var t = 0; t < TESTS.length; t++) {
      var tk = TESTS[t].key;
      var seg = TESTS[t].name + (sk === tk ? ' (same sensor)' : ' (cross-sensor)') + ': ';
      var bits = [];
      for (var m = 0; m < METHODS.length; m++) {
        bits.push(METHODS[m].name + ' ' + f1(MAIN[sk][METHODS[m].key][tk].v));
      }
      parts.push(seg + bits.join(', ') + '.');
    }
    return parts.join(' ');
  }

  function tipFor(sk, mk, tk) {
    var src = srcOf(sk), tst = testOf(tk), met = methodOf(mk);
    var cell = MAIN[sk][mk][tk];
    var lines = ['<b>' + esc(met.name) + (mk === 'lieq' ? ' (ours)' : '') + '</b>'];
    lines.push('trained on ' + esc(src.name) + ' ' + ARROW + ' tested on ' + esc(tst.name) +
      (sk === tk ? ' <i>(same sensor)</i>' : ''));
    lines.push('<b>' + f1(cell.v) + '</b> mIoU');
    if (cell.d !== null && cell.d !== undefined) {
      if (met.deltaRef === 'own-source') {
        lines.push('<i>' + signed(cell.d) + ' vs its own sensor (' +
          f1(MAIN[sk][mk][sk].v) + ')</i>');
      } else {
        lines.push('<i>' + signed(cell.d) + ' vs P-baseline (' +
          f1(MAIN[sk].base[tk].v) + ')</i>');
      }
    }
    if (mk === 'lieq') {
      var over = null;
      for (var i = 0; i < GAINS.length; i++) {
        if (GAINS[i].src === sk && GAINS[i].test === tk) { over = GAINS[i].g; }
      }
      if (over !== null) {
        lines.push('<i>' + signed(over) + ' vs LiSeer-P (' +
          f1(MAIN[sk].liseer[tk].v) + ')</i>');
      }
    }
    return lines.join('||');
  }

  function matrixPanelSVG(sk, opts) {
    var th = theme(opts), T = th.t;
    var tid = uid('t'), did = uid('d'), cid = uid('c'), hid = uid('h');
    var src = srcOf(sk);
    var o = [];

    o.push('<svg class="lq-svg" viewBox="0 0 ' + P.W + ' ' + P.H + '" role="img"' +
      ' data-lq-theme="' + th.name + '" data-lq-panel="' + esc(sk) + '"' +
      ' preserveAspectRatio="xMidYMid meet" aria-labelledby="' + tid + ' ' + did + '">');
    o.push('<title id="' + tid + '">Zero-shot mIoU, student trained on ' + esc(src.name) +
      ' only</title>');
    o.push('<desc id="' + did + '">' + esc(panelDesc(sk)) + '</desc>');
    o.push('<defs><clipPath id="' + cid + '"><rect x="' + P.padL + '" y="' + P.padT +
      '" width="' + P.plotW + '" height="' + P.plotH + '"/></clipPath>' +
      hatchDef(hid, T.hatch) + '</defs>');

    /* same-sensor group backdrop: shaded + 45-degree hatch, so the cell the
       paper's claim is NOT about is marked by two channels, not colour alone. */
    for (var g = 0; g < TESTS.length; g++) {
      if (TESTS[g].key !== sk) { continue; }
      var bx = r2(P.padL + g * P.band);
      o.push('<rect class="lq-diag" x="' + bx + '" y="' + P.padT + '" width="' + r2(P.band) +
        '" height="' + P.plotH + '" fill="' + T.diagFill + '"/>');
      o.push('<rect class="lq-diag-h" x="' + bx + '" y="' + P.padT + '" width="' + r2(P.band) +
        '" height="' + P.plotH + '" fill="url(#' + hid + ')" opacity=".5"/>');
    }

    /* gridlines + y ticks */
    for (var k = 0; k < P.ticks.length; k++) {
      var ty = r2(yOf(P.ticks[k]));
      o.push('<line class="lq-grid" x1="' + P.padL + '" x2="' + r2(P.W - P.padR) +
        '" y1="' + ty + '" y2="' + ty + '" stroke="' + T.grid + '" stroke-width="1"/>');
      o.push('<text class="lq-tick" x="' + (P.padL - 6) + '" y="' + r2(ty + 2.8) +
        '" font-size="8" text-anchor="end">' + P.ticks[k] + '</text>');
    }
    o.push('<text class="lq-unit" x="' + (P.padL - 6) + '" y="12" font-size="7.5"' +
      ' text-anchor="end" letter-spacing=".06em">mIoU</text>');

    /* bars, clipped at the baseline so the rounded end is only on top */
    o.push('<g class="lq-barlayer" clip-path="url(#' + cid + ')">');
    for (var gi = 0; gi < TESTS.length; gi++) {
      var tk = TESTS[gi].key;
      for (var mi = 0; mi < METHODS.length; mi++) {
        var mk = METHODS[mi].key;
        var cell = MAIN[sk][mk][tk];
        var x = r2(P.padL + gi * P.band + P.off + mi * (P.barW + P.barGap));
        var yTop = r2(yOf(cell.v));
        o.push(barRect('lq-bar', {
          x: x, y: yTop, width: P.barW,
          height: r2(P.y0 - yTop + P.rad), rx: P.rad,
          fill: T.series[mk],
          'data-bar': '1', 'data-src': sk, 'data-method': mk, 'data-test': tk,
          'data-v': f1(cell.v)
        }));
      }
    }
    o.push('</g>');

    /* value on top of every bar: this chart stands in for a numeric table, so
       the numbers are the content, not decoration. */
    for (var gv = 0; gv < TESTS.length; gv++) {
      var tkv = TESTS[gv].key;
      for (var mv = 0; mv < METHODS.length; mv++) {
        var mkv = METHODS[mv].key;
        var cv = MAIN[sk][mkv][tkv];
        var cx = r2(P.padL + gv * P.band + P.off + mv * (P.barW + P.barGap) + P.barW / 2);
        o.push('<text class="lq-val' + (mkv === 'lieq' ? '' : ' is-dim') + '" x="' + cx +
          '" y="' + r2(yOf(cv.v) - 4) + '" font-size="8.6" text-anchor="middle"' +
          (mkv === 'lieq' ? ' font-weight="600"' : '') + '>' + f1(cv.v) + '</text>');
      }
    }

    /* baseline + group labels */
    o.push('<line class="lq-axis" x1="' + P.padL + '" x2="' + r2(P.W - P.padR) + '" y1="' +
      P.y0 + '" y2="' + P.y0 + '" stroke="' + T.axis + '" stroke-width="1"/>');
    for (var gl = 0; gl < TESTS.length; gl++) {
      var isDiag = TESTS[gl].key === sk;
      var lx = r2(P.padL + gl * P.band + P.band / 2);
      o.push('<text class="lq-xlab' + (isDiag ? ' is-diag' : '') + '" x="' + lx +
        '" y="' + (P.y0 + 14) + '" font-size="8.4" text-anchor="middle"' +
        (isDiag ? '' : ' font-weight="500"') + '>' + esc(TESTS[gl].name) + '</text>');
      o.push('<text class="lq-xtag" x="' + lx + '" y="' + (P.y0 + 25) +
        '" font-size="7.2" text-anchor="middle">' +
        (isDiag ? 'same sensor' : 'cross-sensor') + '</text>');
    }

    /* hover / tap targets last, so they sit on top of everything */
    for (var gh = 0; gh < TESTS.length; gh++) {
      var tkh = TESTS[gh].key;
      for (var mh = 0; mh < METHODS.length; mh++) {
        var mkh = METHODS[mh].key;
        var hx = r2(P.padL + gh * P.band + P.off + mh * (P.barW + P.barGap) - 1);
        o.push('<rect class="lq-hit" x="' + hx + '" y="' + P.padT + '" width="' +
          (P.barW + P.barGap) + '" height="' + P.plotH + '" data-tip="' +
          esc(tipFor(sk, mkh, tkh)) + '"/>');
      }
    }

    o.push('</svg>');
    return o.join('');
  }

  function legendHTML(items) {
    var h = ['<ul class="lq-legend">'];
    for (var i = 0; i < items.length; i++) {
      var it = items[i];
      h.push('<li><span class="lq-sw" style="' + it.style + '" aria-hidden="true"></span>' +
        '<b>' + esc(it.name) + '</b>' + (it.note ? ' ' + esc(it.note) : '') + '</li>');
    }
    h.push('</ul>');
    return h.join('');
  }

  function mainMatrixHTML(opts) {
    var th = theme(opts), T = th.t;
    var h = [];
    h.push('<figure class="lq-fig" data-lq-chart="main-matrix"' + themeAttr(opts) + '>');
    h.push('<figcaption>');
    h.push('<span class="lq-eyebrow">Table 1 ' + ARROW + ' live chart</span>');
    h.push('<h3 class="lq-head">One sensor in, three sensors out</h3>');
    h.push('<p class="lq-sub">Zero-shot mIoU. Each panel is a student trained on one dataset ' +
      'only and tested on all three. The hatched group is the sensor it was trained on; ' +
      'the two plain groups are the cross-sensor transfers the paper’s claim is about.</p>');
    h.push('</figcaption>');
    h.push(legendHTML([
      { name: 'P-baseline', note: 'single dataset', style: 'background:' + T.series.base },
      { name: 'LiSeer-P-RSCL', note: 'randomized sampling', style: 'background:' + T.series.liseer },
      { name: 'LiEqualizer-P', note: 'ours', style: 'background:' + T.series.lieq },
      { name: 'same sensor', note: '', style: 'background:' + T.diagFill +
        ';box-shadow:inset 0 0 0 1px ' + T.hatch +
        ';background-image:repeating-linear-gradient(45deg,' + T.hatch + ' 0 1px,transparent 1px 4px)' }
    ]));
    h.push('<div class="lq-panels">');
    for (var s = 0; s < SOURCES.length; s++) {
      h.push('<div class="lq-panel"><h4>trained on <b>' + esc(SOURCES[s].name) +
        '</b> only</h4>' + matrixPanelSVG(SOURCES[s].key, opts) + '</div>');
    }
    h.push('</div>');
    h.push('<p class="lq-note">Bars are read against the 0' + MINUS + '80 mIoU scale at the ' +
      'left of each panel. Trained on Waymo alone, the single-dataset baseline scores ' +
      '<b>60.9</b> on its own sensor and <b>5.4</b> on nuScenes. Hover or tap a bar for the ' +
      'delta the manuscript prints beside it.</p>');
    h.push(mainDataTable(true));
    h.push('</figure>');
    return h.join('');
  }

  /* ====================================================================== *
   * 6. CHART 2 — the gain strip                                            *
   * ====================================================================== */

  var G = {
    c: { W: 330, padL: 10, labX: 62, barX: 68, right: 46, rowH: 24, barH: 12,
         top: 34, tickY: 20, axisY: 27, labFs: 9.5, valFs: 10, footH: 36, abbrev: true },
    w: { W: 640, padL: 10, labX: 176, barX: 184, right: 56, rowH: 27, barH: 14,
         top: 38, tickY: 22, axisY: 29, labFs: 9.5, valFs: 10.5, footH: 22, abbrev: false }
  };
  G.gMax = 18;
  G.ticks = [0, 5, 10, 15];

  function gConf(bucket) {
    var g = (bucket === 'w') ? G.w : G.c;
    var out = {};
    for (var k in g) { if (g.hasOwnProperty(k)) { out[k] = g[k]; } }
    out.barMax = out.W - out.barX - out.right;
    out.rowsH = GAINS.length * out.rowH;
    out.H = out.top + out.rowsH + out.footH;
    return out;
  }

  function pairLabel(r, abbrev) {
    var a = srcOf(r.src), b = testOf(r.test);
    return abbrev ? (a.short + ARROW + b.short) : (a.name + ' ' + ARROW + ' ' + b.name);
  }

  function gainDesc() {
    var parts = ['Horizontal bars, one per cell of Table 1, sorted from the largest gain ' +
      'down. Scale 0 to ' + G.gMax + ' mIoU.'];
    for (var i = 0; i < GAINS.length; i++) {
      parts.push(pairLabel(GAINS[i], false) + ' ' + signed(GAINS[i].g) +
        (GAINS[i].same ? ' (same sensor)' : '') + '.');
    }
    parts.push('The largest cross-sensor gain is +11.5 on Waymo to SemanticKITTI, ' +
      '63.8 against 52.3.');
    return parts.join(' ');
  }

  function gainStripSVG(bucket, opts) {
    var th = theme(opts), T = th.t, c = gConf(bucket);
    var tid = uid('t'), did = uid('d'), cid = uid('c'), hid = uid('h');
    var o = [];
    var xOf = function (v) { return c.barX + (v / G.gMax) * c.barMax; };

    o.push('<svg class="lq-svg" viewBox="0 0 ' + c.W + ' ' + c.H + '" role="img"' +
      ' data-lq-theme="' + th.name + '" data-lq-bucket="' + esc(bucket) + '"' +
      ' preserveAspectRatio="xMidYMin meet" aria-labelledby="' + tid + ' ' + did + '">');
    o.push('<title id="' + tid + '">LiEqualizer-P minus LiSeer-P-RSCL, all nine cells</title>');
    o.push('<desc id="' + did + '">' + esc(gainDesc()) + '</desc>');
    o.push('<defs><clipPath id="' + cid + '"><rect x="' + c.barX + '" y="0" width="' +
      r2(c.W - c.barX) + '" height="' + c.H + '"/></clipPath>' +
      hatchDef(hid, T.hatch) + '</defs>');

    /* ticks + vertical grid */
    for (var k = 0; k < G.ticks.length; k++) {
      var tx = r2(xOf(G.ticks[k]));
      o.push('<line class="lq-grid" x1="' + tx + '" x2="' + tx + '" y1="' + c.axisY +
        '" y2="' + r2(c.top + c.rowsH) + '" stroke="' + T.grid + '" stroke-width="1"/>');
      o.push('<text class="lq-tick" x="' + tx + '" y="' + c.tickY + '" font-size="8"' +
        ' text-anchor="middle">' + (G.ticks[k] === 0 ? '0' : '+' + G.ticks[k]) + '</text>');
    }
    o.push('<text class="lq-unit" x="' + c.padL + '" y="' + c.tickY + '" font-size="7.5"' +
      ' text-anchor="start" letter-spacing=".06em">mIoU gain</text>');
    o.push('<line class="lq-axis" x1="' + c.barX + '" x2="' + c.barX + '" y1="' + c.axisY +
      '" y2="' + r2(c.top + c.rowsH) + '" stroke="' + T.axis + '" stroke-width="1"/>');

    /* rows */
    o.push('<g class="lq-barlayer" clip-path="url(#' + cid + ')">');
    for (var i = 0; i < GAINS.length; i++) {
      var r = GAINS[i];
      var cy = r2(c.top + i * c.rowH + c.rowH / 2);
      var w = r2((r.g / G.gMax) * c.barMax);
      var by = r2(cy - c.barH / 2);
      o.push(barRect('lq-gbar', {
        x: r2(c.barX - 4), y: by, width: r2(w + 4), height: c.barH, rx: 4,
        fill: r.same ? T.same : T.cross,
        'data-bar': '1', 'data-src': r.src, 'data-test': r.test, 'data-v': f1(r.g)
      }));
      if (r.same) {
        o.push(barRect('lq-gbar-h', {
          x: r2(c.barX - 4), y: by, width: r2(w + 4), height: c.barH, rx: 4,
          fill: 'url(#' + hid + ')', opacity: '.55'
        }));
      }
    }
    o.push('</g>');

    /* labels, values, callout */
    for (var j = 0; j < GAINS.length; j++) {
      var rr = GAINS[j];
      var ccy = r2(c.top + j * c.rowH + c.rowH / 2);
      var ww = r2((rr.g / G.gMax) * c.barMax);
      o.push('<text class="lq-rowlab' + (rr.same ? ' is-dim' : '') + '" x="' + c.labX +
        '" y="' + r2(ccy + 3.3) + '" font-size="' + c.labFs + '" text-anchor="end"' +
        (rr.same ? '' : ' font-weight="500"') + '>' + esc(pairLabel(rr, c.abbrev)) + '</text>');
      o.push('<text class="lq-val' + (rr.callout ? '' : ' is-dim') + '" x="' +
        r2(c.barX + ww + 6) + '" y="' + r2(ccy + 3.6) + '" font-size="' + c.valFs +
        '" text-anchor="start" font-weight="' + (rr.callout ? '700' : '500') + '">' +
        signed(rr.g) + '</text>');
      if (rr.callout) {
        o.push('<line x1="' + r2(c.barX + ww + 6) + '" x2="' + r2(c.barX + ww + 6 + 30) +
          '" y1="' + r2(ccy + 8) + '" y2="' + r2(ccy + 8) + '" stroke="' + T.accent +
          '" stroke-width="1.4"/>');
        if (bucket === 'w') {
          o.push('<text class="lq-callout-t" x="' + r2(c.barX + ww + 52) + '" y="' +
            r2(ccy + 3.6) + '" font-size="8.6" text-anchor="start">' +
            'largest cross-sensor gain</text>');
        }
      }
    }

    /* compact footer carries the callout the wide layout puts inline */
    if (bucket === 'c') {
      var fy = c.top + c.rowsH + 16;
      o.push('<text class="lq-callout-t" x="' + c.padL + '" y="' + fy + '" font-size="8.4">' +
        'largest cross-sensor gain +11.5 (Waymo' + ARROW + 'SK, 63.8 vs 52.3)</text>');
      o.push('<text class="lq-callout-t" x="' + c.padL + '" y="' + (fy + 11) +
        '" font-size="7.6">SK SemanticKITTI · NU nuScenes · WA Waymo</text>');
    }

    /* hit targets */
    for (var hh = 0; hh < GAINS.length; hh++) {
      var rh = GAINS[hh];
      o.push('<rect class="lq-hit" x="' + c.padL + '" y="' + r2(c.top + hh * c.rowH) +
        '" width="' + r2(c.W - c.padL * 2) + '" height="' + c.rowH + '" data-tip="' +
        esc('<b>' + pairLabel(rh, false) + '</b>||' +
          (rh.same ? '<i>same sensor</i>' : '<i>cross-sensor</i>') + '||' +
          'LiEqualizer-P <b>' + f1(MAIN[rh.src].lieq[rh.test].v) + '</b> vs LiSeer-P ' +
          f1(MAIN[rh.src].liseer[rh.test].v) + '||<b>' + signed(rh.g) + '</b> mIoU') +
        '"/>');
    }

    o.push('</svg>');
    return o.join('');
  }

  function gainStripHTML(bucket, opts) {
    var th = theme(opts), T = th.t;
    var h = [];
    h.push('<figure class="lq-fig" data-lq-chart="gain-strip"' + themeAttr(opts) + '>');
    h.push('<figcaption>');
    h.push('<span class="lq-eyebrow">Gain over LiSeer-P</span>');
    h.push('<h3 class="lq-head">What the spectral objective is worth, cell by cell</h3>');
    h.push('</figcaption>');
    h.push('<div class="lq-hero"><span class="lq-hero-n">+11.5</span>' +
      '<span class="lq-hero-l">largest cross-sensor gain</span>' +
      '<span class="lq-hero-d">Trained on Waymo, tested on SemanticKITTI: <b>63.8</b> mIoU ' +
      'against <b>52.3</b>. Waymo' + ARROW + 'Waymo is larger at +17.0, but that is the same ' +
      'sensor, which is why the claim says cross-sensor.</span></div>');
    h.push(legendHTML([
      { name: 'cross-sensor', note: '', style: 'background:' + T.cross },
      { name: 'same sensor', note: '', style: 'background:' + T.same +
        ';background-image:repeating-linear-gradient(45deg,' + T.hatch + ' 0 1.5px,transparent 1.5px 5px)' }
    ]));
    h.push('<div class="lq-strip">' + gainStripSVG(bucket, opts) + '</div>');
    h.push('<p class="lq-note">Every one of the nine cells moves upwards: the smallest gain ' +
      'is <b>+0.9</b> (nuScenes' + ARROW + 'SemanticKITTI), the largest <b>+17.0</b> ' +
      '(Waymo' + ARROW + 'Waymo, same sensor).</p>');
    h.push(gainDataTable(true));
    h.push('</figure>');
    return h.join('');
  }

  /* ====================================================================== *
   * 7. CHARTS 3 & 4 — the two tables                                       *
   * ====================================================================== */

  function markedCell(c) {
    var inner = '<span class="lq-numv">' + f1(c.v) + '</span>';
    if (c.d !== null && c.d !== undefined) {
      inner += '<span class="lq-d ' + (c.d >= 0 ? 'lq-d-gain' : 'lq-d-drop') + '">(' +
        signed(c.d) + ')</span>';
    }
    var out;
    if (c.mark === 'best') {
      out = '<strong class="lq-best">' + inner + '</strong>' +
        '<span class="lq-sr"> (best in column)</span>';
    } else if (c.mark === 'second') {
      out = '<span class="lq-second">' + inner + '</span>' +
        '<span class="lq-sr"> (second best in column)</span>';
    } else {
      out = inner;
    }
    if (c.footnote) { out += '<sup>' + c.footnote + '</sup>'; }
    return out;
  }

  function tableFigHTML(spec, opts) {
    var hasAttach = !!spec.attachHead;
    var wrapId = uid('w');
    var h = [];
    h.push('<figure class="lq-fig" data-lq-chart="' + esc(spec.slug) + '"' +
    themeAttr(opts) + '>');
    h.push('<figcaption>');
    h.push('<span class="lq-eyebrow">' + esc(spec.number) + '</span>');
    h.push('<h3 class="lq-head">' + esc(spec.head) + '</h3>');
    h.push('<p class="lq-sub">' + esc(spec.caption) + '</p>');
    h.push('</figcaption>');
    h.push('<div class="lq-tablewrap" tabindex="0" role="region" aria-labelledby="' + wrapId +
      '">');
    h.push('<table class="lq-table">');
    h.push('<caption id="' + wrapId + '" class="lq-sr">' + esc(spec.number + '. ' +
      spec.caption + ' Bold marks the best figure in a column, underline the second best, ' +
      'as in the manuscript. Bracketed figures are differences from the first row; they are ' +
      'printed in red when positive and green when negative, following the manuscript.') +
      '</caption>');
    h.push('<thead><tr><th scope="col">' + esc(spec.rowHead) + '</th>');
    if (hasAttach) { h.push('<th scope="col" class="lq-attach">' + esc(spec.attachHead) + '</th>'); }
    for (var ci = 0; ci < spec.cols.length; ci++) {
      h.push('<th scope="col">' + esc(spec.cols[ci]) + '</th>');
    }
    h.push('</tr></thead><tbody>');
    for (var ri = 0; ri < spec.rows.length; ri++) {
      var row = spec.rows[ri];
      var cls = [];
      if (row.ruleAfter) { cls.push('lq-rule'); }
      if (row.ours) { cls.push('lq-ours'); }
      h.push('<tr' + (cls.length ? ' class="' + cls.join(' ') + '"' : '') + '>');
      h.push('<th scope="row"' + (row.indent ? ' class="lq-indent"' : '') + '>' +
        esc(row.label) + '</th>');
      if (hasAttach) { h.push('<td class="lq-attach">' + esc(row.attached) + '</td>'); }
      for (var cj = 0; cj < row.cells.length; cj++) {
        h.push('<td>' + markedCell(row.cells[cj]) + '</td>');
      }
      h.push('</tr>');
    }
    h.push('</tbody></table></div>');
    h.push('<p class="lq-note">' + esc(spec.note) + '</p>');
    h.push('<p class="lq-fn"><b>Bold</b> is the best figure in a column and ' +
      '<span class="lq-second">underline</span> the second best. Bracketed figures are ' +
      'differences from the first row, shown <span class="lq-d-gain">red when positive</span> ' +
      'and <span class="lq-d-drop">green when negative</span>, as in the manuscript; the ' +
      'sign, not the colour, carries the direction.</p>');
    if (spec.footnotes) {
      for (var fi = 0; fi < spec.footnotes.length; fi++) {
        h.push('<p class="lq-fn"><sup>' + (fi + 1) + '</sup> ' + esc(spec.footnotes[fi]) + '</p>');
      }
    }
    h.push('</figure>');
    return h.join('');
  }

  function ablationHTML(opts) {
    var spec = ABLATION;
    spec.slug = 'ablation';
    spec.head = 'Each part, added one at a time';
    return tableFigHTML(spec, opts);
  }

  function kdBaselinesHTML(opts) {
    var spec = KD;
    spec.slug = 'kd-baselines';
    spec.head = 'The same teacher, a different objective';
    return tableFigHTML(spec, opts);
  }

  /* ====================================================================== *
   * 8. TOOLTIP (optional layer; the charts read without it)                *
   * ====================================================================== */

  var tipEl = null;
  var tipTimer = null;

  function tip() {
    if (!doc) { return null; }
    if (tipEl && tipEl.parentNode) { return tipEl; }
    tipEl = doc.createElement('div');
    tipEl.className = 'lq-tip';
    tipEl.setAttribute('aria-hidden', 'true');
    (doc.body || doc.documentElement).appendChild(tipEl);
    return tipEl;
  }

  function hideTip() {
    if (tipTimer) { root.clearTimeout(tipTimer); tipTimer = null; }
    if (tipEl) { tipEl.removeAttribute('data-on'); }
  }

  function showTip(payload, cx, cy) {
    var el = tip();
    if (!el) { return; }
    var lines = String(payload).split('||');
    el.innerHTML = lines.join('<br>');
    el.setAttribute('data-on', '1');
    var w = el.offsetWidth || 180, h = el.offsetHeight || 60;
    var vw = (root.innerWidth || 360), vh = (root.innerHeight || 640);
    var x = cx + 14, y = cy - h - 12;
    if (x + w > vw - 8) { x = Math.max(8, cx - w - 14); }
    if (y < 8) { y = Math.min(vh - h - 8, cy + 18); }
    el.style.left = Math.round(x) + 'px';
    el.style.top = Math.round(y) + 'px';
  }

  function tipPayload(node) {
    while (node && node !== doc) {
      if (node.getAttribute) {
        var t = node.getAttribute('data-tip');
        if (t) { return t; }
      }
      node = node.parentNode;
    }
    return null;
  }

  function on(el, type, fn, passive) {
    if (!el || !el.addEventListener) { return; }
    try { el.addEventListener(type, fn, passive ? { passive: true } : false); }
    catch (e) { el.addEventListener(type, fn, false); }
  }

  function wireTips(container) {
    if (!container || container.getAttribute('data-lq-tips') === '1') { return; }
    container.setAttribute('data-lq-tips', '1');

    on(container, 'mousemove', function (ev) {
      var p = tipPayload(ev.target);
      if (p) { showTip(p, ev.clientX, ev.clientY); } else { hideTip(); }
    });
    on(container, 'mouseleave', hideTip);
    on(container, 'touchstart', function (ev) {
      var t = ev.touches && ev.touches[0];
      var p = tipPayload(ev.target);
      if (!p || !t) { hideTip(); return; }
      showTip(p, t.clientX, t.clientY);
      if (tipTimer) { root.clearTimeout(tipTimer); }
      tipTimer = root.setTimeout(hideTip, 2800);
    }, true);

    if (!wireTips.global && doc) {
      wireTips.global = true;
      on(root, 'scroll', hideTip, true);
      on(doc, 'touchstart', function (ev) {
        if (!tipPayload(ev.target)) { hideTip(); }
      }, true);
    }
  }

  /* ====================================================================== *
   * 9. DOM RENDERERS                                                       *
   * ====================================================================== */

  function mountHTML(el, html) {
    var m = resolve(el);
    if (!m) { return null; }
    injectCSS();
    m.innerHTML = html;
    wireTips(m);
    return m;
  }

  function renderMainTable(el, opts) {
    try { return mountHTML(el, mainMatrixHTML(opts)); }
    catch (e) { return null; }
  }

  function renderAblation(el, opts) {
    try { return mountHTML(el, ablationHTML(opts)); }
    catch (e) { return null; }
  }

  function renderKDBaselines(el, opts) {
    try { return mountHTML(el, kdBaselinesHTML(opts)); }
    catch (e) { return null; }
  }

  /* The gain strip's row labels are SVG text, so a single viewBox cannot span
     360px to 1180px without the type going illegible at one end. Two layouts,
     switched on a width bucket, and re-rendered only when the bucket changes. */
  function bucketFor(width) { return width < 620 ? 'c' : 'w'; }

  function renderGainStrip(el, opts) {
    var m = resolve(el);
    if (!m) { return null; }
    var forced = opts && opts.bucket;
    var current = null;

    function draw() {
      var w = m.clientWidth || (m.getBoundingClientRect && m.getBoundingClientRect().width) || 0;
      var b = forced || bucketFor(w || 360);
      if (b === current) { return; }
      current = b;
      injectCSS();
      m.innerHTML = gainStripHTML(b, opts);
      wireTips(m);
    }

    try {
      draw();
      if (!forced && root) {
        if (typeof root.ResizeObserver === 'function') {
          var ro = new root.ResizeObserver(function () { draw(); });
          ro.observe(m);
        } else {
          var pending = null;
          on(root, 'resize', function () {
            if (pending) { root.clearTimeout(pending); }
            pending = root.setTimeout(function () { pending = null; draw(); }, 160);
          });
        }
      }
    } catch (e) { /* a mount that cannot be measured keeps whatever drew first */ }
    return m;
  }

  var DEFAULT_IDS = {
    main: 'lq-main-matrix',
    gain: 'lq-gain-strip',
    ablation: 'lq-ablation',
    kd: 'lq-kd-baselines'
  };

  function renderAll(opts) {
    var ids = (opts && opts.ids) || DEFAULT_IDS;
    return {
      main: renderMainTable(ids.main, opts),
      gain: renderGainStrip(ids.gain, opts),
      ablation: renderAblation(ids.ablation, opts),
      kd: renderKDBaselines(ids.kd, opts)
    };
  }

  function fallbackHTML(which) {
    if (which === 'gain' || which === 'gain-strip') { return gainDataTable(false); }
    if (which === 'ablation') { return tableFigHTML(ablationSpec(), null); }
    if (which === 'kd' || which === 'kd-baselines') { return tableFigHTML(kdSpec(), null); }
    return mainDataTable(false);
  }
  function ablationSpec() { ABLATION.slug = 'ablation'; ABLATION.head = 'Each part, added one at a time'; return ABLATION; }
  function kdSpec() { KD.slug = 'kd-baselines'; KD.head = 'The same teacher, a different objective'; return KD; }

  /* ====================================================================== *
   * 10. EXPORT + AUTO-INIT                                                 *
   * ====================================================================== */

  var API = {
    version: '1.0.0',
    renderMainTable: renderMainTable,
    renderGainStrip: renderGainStrip,
    renderAblation: renderAblation,
    renderKDBaselines: renderKDBaselines,
    renderAll: renderAll,
    fallbackHTML: fallbackHTML,
    /* string builders, usable without a DOM (headless checks, SSR, tests) */
    html: {
      mainMatrix: mainMatrixHTML,
      gainStrip: gainStripHTML,
      ablation: ablationHTML,
      kdBaselines: kdBaselinesHTML,
      matrixPanel: matrixPanelSVG,
      gainStripSVG: gainStripSVG,
      mainDataTable: mainDataTable,
      gainDataTable: gainDataTable
    },
    DATA: {
      sources: SOURCES, tests: TESTS, methods: METHODS,
      main: MAIN, gains: GAINS, ablation: ABLATION, kdBaselines: KD,
      themes: THEMES, defaultIds: DEFAULT_IDS
    },
    CSS: CSS
  };

  if (typeof module !== 'undefined' && module.exports) { module.exports = API; }
  if (root) { root.LiEQCharts = API; }

  function autoInit() {
    var tag = doc.currentScript;
    if (tag && tag.getAttribute('data-lq-no-autoinit') !== null) { return; }
    var go = function () { try { renderAll(); } catch (e) { /* no-op */ } };
    if (doc.readyState === 'loading') {
      doc.addEventListener('DOMContentLoaded', go, false);
    } else {
      go();
    }
  }

  if (doc) { autoInit(); }
})(typeof window !== 'undefined' ? window : (typeof globalThis !== 'undefined' ? globalThis : null));
