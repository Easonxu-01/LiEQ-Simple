/*!
 * motion.js — LiEqualizer project page, motion layer for the HTML-rendered figures.
 *
 * A thin layer over the figures that charts-main.js, charts-paper.js and
 * charts-native.js draw. It adds two things and nothing else:
 *
 *   1. an on-scroll reveal — bars grow from their own baseline, lines draw along
 *      their length, ribbons / cells / markers fade and settle, table rows stagger;
 *   2. pointer emphasis on the data marks — the mark under the pointer keeps its
 *      paint, its siblings step back, and the figure's own tooltip does the talking.
 *
 * It never draws, never re-numbers and never re-colours anything. Every animation
 * ends by removing the inline style it added, so a finished figure is byte-for-byte
 * the figure the chart module rendered: a screenshot, a print or a headless capture
 * of a settled page is identical to the no-JS page.
 *
 * Plain ES5. No framework, no bundler, no CDN, no build step. Load it with a plain
 * <script src> after the chart modules. Partner stylesheet: assets/motion.css.
 *
 * Public API (window.LiEQMotion):
 *   init(opts)      wire every figure present now; safe to call any number of times
 *   refresh()       rescan the document (same thing; call it after you render a figure)
 *   finishAll()     jump every figure to its final state at once and stop animating
 *   isInstant()     true when motion is suppressed for this page load
 *   settings        the timing / threshold knobs, mutable before init()
 *   version
 *
 * Forcing the final state (for headless screenshots and video frames) — any one of:
 *   <html data-lieq-motion="off">            before the script runs   (also: "instant", "none")
 *   window.LIEQ_MOTION_INSTANT = true;       before the script runs
 *   ?motion=off  in the page URL
 *   navigator.webdriver === true             (automation; override with data-lieq-motion="on")
 *   prefers-reduced-motion: reduce           (the user's own setting)
 *   LiEQMotion.finishAll()                   at any moment, e.g. just before capture
 *
 * Opt-in, off by default: data-lq-motion-keys on a figure root or its mount gives
 * that figure's hit targets one tab stop and arrow-key traversal. motion.js never
 * makes anything focusable without it, and it never intercepts Tab.
 */
(function (global) {
  'use strict';

  var VERSION = '1.0.0';
  var doc = global.document || null;

  /* ====================================================================== *
   * 1. KNOBS                                                               *
   * ====================================================================== */

  /* Durations in ms. The language is a scientific instrument: short, confident,
     easing out, slightly staggered, nothing bouncing and nothing overshooting. */
  var S = {
    startDelay: 110,      /* lets the page's own .reveal fade get under way     */
    panelStagger: 64,     /* between panels of one figure                       */
    panelStaggerCap: 5,   /* panels past this one share the last panel's delay  */
    sweep: 200,           /* left-to-right (or top-down) sweep inside a panel   */
    block: 460,           /* a bar growing from its baseline                    */
    draw: 620,            /* a line drawing along its length                    */
    fade: 300,            /* ribbons, heat-map cells, whiskers                  */
    point: 240,           /* markers and diamonds                              */
    value: 220,           /* the printed value labels                           */
    valueLag: 170,        /* value labels trail their own mark                  */
    row: 260,             /* a table row                                        */
    rowStagger: 34,
    panelFade: 420,       /* used instead of per-mark work on huge panels       */
    maxMarks: 600,        /* above this a panel reveals as one piece            */
    ease: 'cubic-bezier(.22,.61,.36,1)',
    easeDraw: 'cubic-bezier(.25,.6,.3,1)',
    threshold: 0.12,
    rootMargin: '0px 0px -8% 0px',
    watchdog: 2200,       /* §11: nothing is left half-drawn for longer       */
    repairAfter: 900,     /* ms after settling: the one look-and-repair pass    */
    dim: null             /* null = leave --lqm-dim to motion.css              */
  };

  /* ====================================================================== *
   * 2. WHAT THE THREE CHART MODULES ACTUALLY EMIT                          *
   * ====================================================================== *
   * Read off the modules, not guessed:
   *
   *   charts-main.js    <figure class="lq-fig" data-lq-chart="…">
   *                       <svg class="lq-svg">  rect.lq-bar / rect.lq-gbar  (data-bar="1", data-v)
   *                                             rect.lq-hit (data-tip="a||b")
   *                                             text.lq-val
   *                       <table class="lq-table"><tbody><tr>
   *   charts-paper.js   <div class="lqp" data-chart="…">
   *                       .lqp-panels > svg[data-panel]
   *                         g[data-series][data-kind] > path[data-part="median"|"ribbon"],
   *                                                     circle[data-part="marker"],
   *                                                     rect, rect.edge, .hit  (tip on __lqpTip)
   *                       .lqp-tip
   *   charts-native.js  <div class="lqc" data-chart="…">
   *                       .lqc-panels > svg
   *                         rect (+ rect[fill=hatch] + rect.edge) per bar, rect.cellsep,
   *                         path.dia, line.wk, path, circle, text.t-val/.t-n/.t-cell,
   *                         .hit  (tip on __lqcTip)
   *                       .lqc-tip
   */
  var ADAPTERS = [
    { key: 'main',   fig: 'figure.lq-fig[data-lq-chart]', panels: 'svg.lq-svg',
      hit: 'lq-hit', rows: 'table.lq-table>tbody>tr' },
    { key: 'paper',  fig: 'div.lqp[data-chart]',          panels: '.lqp-panels>svg',
      hit: 'hit',    rows: null },
    { key: 'native', fig: 'div.lqc[data-chart]',          panels: '.lqc-panels>svg',
      hit: 'hit',    rows: null }
  ];

  /* Chart furniture: frame, not data. Never touched, so axes, gridlines, ticks,
     parity rules, titles and annotations stay put while the data moves. */
  var SKIP = {
    'hit': 1, 'lq-hit': 1,
    'ax': 1, 'gr': 1, 'par': 1, 'soft': 1, 'shade': 1, 'hfedge': 1,
    'lq-grid': 1, 'lq-axis': 1, 'lq-diag': 1, 'lq-diag-h': 1,
    't-tick': 1, 't-pan': 1, 't-cat': 1, 't-sub': 1, 't-dim': 1, 't-par': 1,
    't-ax': 1, 't-anno': 1, 't-flag': 1, 't-key': 1,
    'lq-tick': 1, 'lq-unit': 1, 'lq-xlab': 1, 'lq-xtag': 1, 'lq-rowlab': 1,
    'lq-callout-t': 1, 'lq-sw': 1
  };

  /* The only text the reveal animates: the printed numbers, which belong to their
     mark. Everything else is a label and stays legible throughout. */
  var VALUE_TEXT = { 'lq-val': 1, 't-val': 1, 't-n': 1, 't-cell': 1 };

  /* Never descend into these: the visually hidden data tables (moving them once
     cost this page a 1884px horizontal overflow) and the insight equalizer, which
     owns its own interaction. */
  var NO_GO = ['lq-sr', 'lqp-sr', 'lqc-sr', 'lq-srtable', 'lieq-card', 'lieq-shell'];

  /* And never, ever touch a node inside one of these. All three modules put a
     clip rect and a hatch pattern in <defs>: a clip rect is a <rect> like any
     other, and a transform on it collapses the clip region, which would take a
     whole bar layer with it. */
  var NO_GO_TAGS = {
    'defs': 1, 'clippath': 1, 'pattern': 1, 'mask': 1, 'marker': 1,
    'symbol': 1, 'filter': 1, 'lineargradient': 1, 'radialgradient': 1
  };

  var MARK_ATTR = 'data-lqm-mark';
  var ON_ATTR = 'data-lqm-on';
  var FOCUS_ATTR = 'data-lqm-focus';
  var HOT_ATTR = 'data-lqm-hot';

  /* ====================================================================== *
   * 3. SMALL UTILITIES                                                     *
   * ====================================================================== */

  function clsOf(el) {
    if (!el || !el.getAttribute) { return ''; }
    /* SVG elements have no reliable .className string, so read the attribute. */
    var c = el.getAttribute('class');
    return c ? (' ' + c + ' ') : '';
  }

  function hasCls(el, name) {
    return clsOf(el).indexOf(' ' + name + ' ') >= 0;
  }

  function anyCls(el, map) {
    var c = clsOf(el);
    if (!c) { return null; }
    var parts = c.split(/\s+/), i;
    for (i = 0; i < parts.length; i++) {
      if (parts[i] && map[parts[i]]) { return parts[i]; }
    }
    return null;
  }

  function tag(el) {
    var t = el && (el.tagName || '');
    return String(t).toLowerCase();
  }

  function num(el, attr) {
    var v = el.getAttribute(attr);
    if (v === null || v === '') { return NaN; }
    return parseFloat(v);
  }

  function list(root, sel) {
    if (!root || !root.querySelectorAll) { return []; }
    var out = [], n, i;
    try { n = root.querySelectorAll(sel); } catch (e) { return []; }
    for (i = 0; i < n.length; i++) { out.push(n[i]); }
    return out;
  }

  function closestAttr(el, attr) {
    var n = el, i = 0;
    while (n && n.getAttribute && i < 40) {
      var v = n.getAttribute(attr);
      if (v !== null) { return v; }
      n = n.parentNode;
      i += 1;
    }
    return null;
  }

  function inNoGo(el, stop) {
    var n = el, i = 0, j;
    while (n && n !== stop && n.getAttribute && i < 40) {
      if (NO_GO_TAGS[tag(n)]) { return true; }
      for (j = 0; j < NO_GO.length; j++) {
        if (hasCls(n, NO_GO[j])) { return true; }
      }
      n = n.parentNode;
      i += 1;
    }
    return false;
  }

  function setStyle(el, prop, value) {
    try { el.style.setProperty(prop, value); }
    catch (e) { /* a style-less node: nothing to do */ }
  }

  function clearStyle(el, props) {
    var i;
    for (i = 0; i < props.length; i++) {
      try { el.style.removeProperty(props[i]); } catch (e) { /* ignore */ }
    }
    /* Removing the last property leaves an empty style="" behind in Chrome.
       Nothing renders differently, but a settled figure is supposed to be the
       module's own markup down to the attribute, so drop the husk — and only
       when it is empty, so a style the module itself set is never touched. */
    try {
      if (el.getAttribute && el.removeAttribute && el.getAttribute('style') === '') {
        el.removeAttribute('style');
      }
    } catch (e2) { /* ignore */ }
  }

  var CLEAN = ['transform', 'transform-origin', 'transform-box',
    'opacity', 'stroke-dasharray', 'stroke-dashoffset', 'will-change'];
  var CLEAN_T = ['transition'];

  function clamp01(v) {
    return v < 0 ? 0 : (v > 1 ? 1 : v);
  }

  function rAF(fn) {
    if (global.requestAnimationFrame) { return global.requestAnimationFrame(fn); }
    return global.setTimeout(fn, 16);
  }

  /* ====================================================================== *
   * 4. SHOULD ANYTHING MOVE AT ALL?                                        *
   * ====================================================================== */

  var FORCED_INSTANT = false;   /* set by finishAll() */

  function prefersReduced() {
    try {
      return !!(global.matchMedia &&
        global.matchMedia('(prefers-reduced-motion: reduce)').matches);
    } catch (e) { return false; }
  }

  function flagAttr() {
    if (!doc || !doc.documentElement) { return null; }
    var v = doc.documentElement.getAttribute('data-lieq-motion');
    return v === null ? null : String(v).toLowerCase();
  }

  function isInstant() {
    if (FORCED_INSTANT) { return true; }
    var f = flagAttr();
    if (f === 'on' || f === 'always') { return false; }
    if (f === 'off' || f === 'instant' || f === 'none') { return true; }
    if (global.LIEQ_MOTION_INSTANT === true) { return true; }
    if (prefersReduced()) { return true; }
    if (global.navigator && global.navigator.webdriver === true) { return true; }
    try {
      if (global.location && /(?:^|[?&])motion=(?:off|0|none|instant)\b/
        .test(String(global.location.search))) { return true; }
    } catch (e) { /* ignore */ }
    if (!('IntersectionObserver' in global)) { return true; }
    return false;
  }

  /* ====================================================================== *
   * 5. READING A FIGURE  (every geometry read for a figure happens here,   *
   *    once, before a single write)                                        *
   * ====================================================================== */

  /* Geometry in the panel's own user units. Read from attributes wherever the
     element has them, which is every bar, cell, marker, whisker and label the
     three modules draw — so the common case costs no layout at all. */
  function boxOf(el) {
    var t = tag(el), x, y, w, h, r;
    if (t === 'rect') {
      x = num(el, 'x'); y = num(el, 'y');
      w = num(el, 'width'); h = num(el, 'height');
    } else if (t === 'circle') {
      x = num(el, 'cx'); y = num(el, 'cy'); r = num(el, 'r');
      if (isNaN(r)) { r = 0; }
      x -= r; y -= r; w = 2 * r; h = 2 * r;
    } else if (t === 'ellipse') {
      var rx = num(el, 'rx'), ry = num(el, 'ry');
      x = num(el, 'cx') - rx; y = num(el, 'cy') - ry; w = 2 * rx; h = 2 * ry;
    } else if (t === 'line') {
      var x1 = num(el, 'x1'), x2 = num(el, 'x2'), y1 = num(el, 'y1'), y2 = num(el, 'y2');
      x = Math.min(x1, x2); y = Math.min(y1, y2);
      w = Math.abs(x2 - x1); h = Math.abs(y2 - y1);
    } else if (t === 'text') {
      x = num(el, 'x'); y = num(el, 'y'); w = 0; h = 0;
    } else {
      try {
        var b = el.getBBox();
        x = b.x; y = b.y; w = b.width; h = b.height;
      } catch (e) { return null; }
    }
    if (isNaN(x) || isNaN(y) || isNaN(w) || isNaN(h)) { return null; }
    return { x: x, y: y, w: w, h: h, cx: x + w / 2, cy: y + h / 2 };
  }

  function pathLength(el) {
    if (!el.getTotalLength) { return 0; }
    var len;
    try { len = el.getTotalLength(); } catch (e) { return 0; }
    if (!isFinite(len) || len <= 0 || len > 200000) { return 0; }
    return len;
  }

  /* kind: 'block' | 'stroke' | 'area' | 'point' | 'value' | null */
  function kindOf(el) {
    var t = tag(el);
    if (t === 'defs' || t === 'clippath' || t === 'pattern' || t === 'lineargradient' ||
        t === 'radialgradient' || t === 'stop' || t === 'title' || t === 'desc' ||
        t === 'g' || t === 'svg' || t === 'tspan' || t === 'style' || t === 'use' ||
        t === 'marker' || t === 'symbol' || t === 'filter' || t === 'mask') { return null; }
    if (anyCls(el, SKIP)) { return null; }

    if (t === 'text') {
      return anyCls(el, VALUE_TEXT) ? 'value' : null;
    }
    if (t === 'rect' || t === 'polygon' || t === 'polyline') { return 'block'; }
    if (t === 'circle' || t === 'ellipse') { return 'point'; }
    if (t === 'line') { return 'area'; }        /* whiskers, segments, flat refs: settle */
    if (t === 'path') {
      var fill = el.getAttribute('fill');
      var stroke = el.getAttribute('stroke');
      var dash = el.getAttribute('stroke-dasharray');
      if (fill === 'none' && stroke && stroke !== 'none' && !dash) { return 'stroke'; }
      return 'area';                             /* ribbons, diamonds, dashed lines */
    }
    return null;
  }

  /* Bars are the marks that share an edge, and that edge is the baseline. Tally the
     four candidate edges across the panel's rects: if most of them agree, the panel
     is a bar panel and the agreed line is where the bars grow from — which is also
     the zero line for the panels whose bars hang below it. If they do not agree (a
     heat map, a scatter of cells), there is no baseline and the blocks settle in
     instead of growing. */
  function detectAnchor(blocks) {
    if (blocks.length < 2) { return null; }
    var tallies = [{}, {}, {}, {}], keys = ['y2', 'y1', 'x1', 'x2'], i, j, v, k;
    for (i = 0; i < blocks.length; i++) {
      var b = blocks[i].box;
      var vals = [b.y + b.h, b.y, b.x, b.x + b.w];
      for (j = 0; j < 4; j++) {
        k = Math.round(vals[j] * 2) / 2;          /* half-unit tolerance */
        tallies[j][k] = (tallies[j][k] || 0) + 1;
      }
    }
    var best = null;
    for (j = 0; j < 4; j++) {
      for (v in tallies[j]) {
        if (!tallies[j].hasOwnProperty(v)) { continue; }
        if (!best || tallies[j][v] > best.n) {
          best = { n: tallies[j][v], at: parseFloat(v), which: keys[j] };
        }
      }
    }
    if (!best || best.n / blocks.length < 0.55) { return null; }
    return {
      axis: (best.which === 'y1' || best.which === 'y2') ? 'y' : 'x',
      at: best.at
    };
  }

  function collectPanel(svg, adapter) {
    var all, i, el, k, box;
    try { all = svg.getElementsByTagName('*'); } catch (e) { return null; }
    var marks = [], hits = [], blocks = [], strokes = 0;

    for (i = 0; i < all.length; i++) {
      el = all[i];
      if (hasCls(el, adapter.hit)) {
        if (inNoGo(el, svg)) { continue; }
        box = boxOf(el);
        if (box) { hits.push({ el: el, box: box }); }
        continue;
      }
      k = kindOf(el);
      if (!k) { continue; }
      if (inNoGo(el, svg)) { continue; }
      box = boxOf(el);
      if (!box) { continue; }
      var m = { el: el, kind: k, box: box, len: 0 };
      if (k === 'stroke') {
        m.len = pathLength(el);
        if (!m.len) { m.kind = 'area'; } else { strokes += 1; }
      }
      marks.push(m);
      if (m.kind === 'block') { blocks.push(m); }
    }

    if (!marks.length) { return null; }
    return {
      svg: svg,
      marks: marks,
      hits: hits,
      anchor: detectAnchor(blocks),
      hasStroke: strokes > 0,
      heavy: marks.length > S.maxMarks
    };
  }

  /* Pair each mark with the hit target that covers it, so hovering a hit can
     emphasise exactly the marks that hit stands for. Geometric, because the three
     modules label their marks three different ways (and charts-native does not
     label them at all): the smallest hit whose box contains the mark's centre
     wins. Only data-bearing marks are paired — lines and ribbons are context. */
  function pair(panel) {
    var i, j, m, h, chosen, area, tol = 0.75;
    for (i = 0; i < panel.hits.length; i++) {
      panel.hits[i].group = [];
    }
    for (i = 0; i < panel.marks.length; i++) {
      m = panel.marks[i];
      if (m.kind !== 'block' && m.kind !== 'point' && m.kind !== 'value') { continue; }
      chosen = null; area = Infinity;
      for (j = 0; j < panel.hits.length; j++) {
        h = panel.hits[j].box;
        if (m.box.cx < h.x - tol || m.box.cx > h.x + h.w + tol) { continue; }
        if (m.box.cy < h.y - tol || m.box.cy > h.y + h.h + tol) { continue; }
        var a = (h.w + 1) * (h.h + 1);
        if (a < area) { area = a; chosen = panel.hits[j]; }
      }
      if (chosen) {
        chosen.group.push(m.el);
        m.el.__lqmHit = chosen.el;
      }
    }
    for (i = 0; i < panel.hits.length; i++) {
      h = panel.hits[i];
      h.el.__lqmGroup = h.group.length ? h.group : null;
      h.el.__lqmPanel = panel.svg;
    }
  }

  /* ====================================================================== *
   * 6. CHOREOGRAPHY — one pass of arithmetic, no DOM access                *
   * ====================================================================== */

  function schedule(entry) {
    var p, i, m, total = 0, pi;
    for (pi = 0; pi < entry.panels.length; pi++) {
      p = entry.panels[pi];
      var pd = S.startDelay + Math.min(pi, S.panelStaggerCap) * S.panelStagger;
      p.delay = pd;

      if (p.heavy) {
        total = Math.max(total, pd + S.panelFade);
        continue;
      }

      /* the sweep runs along the bars, i.e. across x for upright bars and down y
         for the horizontal strips */
      var down = !!(p.anchor && p.anchor.axis === 'x');
      var lo = Infinity, hi = -Infinity, v;
      for (i = 0; i < p.marks.length; i++) {
        v = down ? p.marks[i].box.cy : p.marks[i].box.cx;
        if (v < lo) { lo = v; }
        if (v > hi) { hi = v; }
      }
      var span = (hi - lo) || 1;

      for (i = 0; i < p.marks.length; i++) {
        m = p.marks[i];
        v = down ? m.box.cy : m.box.cx;
        var t = clamp01((v - lo) / span);
        if (m.kind === 'stroke') {
          m.delay = pd + 40;
          m.dur = S.draw;
        } else if (m.kind === 'area') {
          m.delay = pd + 90 + Math.round(S.sweep * t * 0.5);
          m.dur = S.fade;
        } else if (m.kind === 'point') {
          m.delay = pd + Math.round(p.hasStroke
            ? S.draw * 0.45 + S.sweep * t * 0.3
            : S.sweep * t + S.block * 0.45);
          m.dur = S.point;
        } else if (m.kind === 'value') {
          m.delay = pd + Math.round(S.sweep * t) + S.valueLag;
          m.dur = S.value;
        } else {
          m.delay = pd + Math.round(S.sweep * t);
          m.dur = p.anchor ? S.block : S.fade;
        }
        total = Math.max(total, m.delay + m.dur);
      }
    }

    for (i = 0; i < entry.rows.length; i++) {
      entry.rows[i].delay = S.startDelay + i * S.rowStagger;
      entry.rows[i].dur = S.row;
      total = Math.max(total, entry.rows[i].delay + entry.rows[i].dur);
    }
    entry.total = total;
  }

  /* ====================================================================== *
   * 7. WRITES — start state, play, settle                                  *
   * ====================================================================== */

  function armPanel(p) {
    var i, m;
    if (p.heavy) {
      setStyle(p.svg, 'opacity', '0');
      return;
    }
    for (i = 0; i < p.marks.length; i++) {
      m = p.marks[i];
      if (m.kind === 'stroke') {
        setStyle(m.el, 'stroke-dasharray', m.len + 'px');
        setStyle(m.el, 'stroke-dashoffset', m.len + 'px');
      } else if (m.kind === 'block' && p.anchor) {
        /* grow from the baseline: scale about the shared edge, in the panel's own
           user units, so a bar collapses onto its axis and a bar that hangs below
           the zero line collapses onto that same line */
        if (p.anchor.axis === 'y') {
          setStyle(m.el, 'transform-origin', m.box.cx + 'px ' + p.anchor.at + 'px');
          setStyle(m.el, 'transform', 'scaleY(.015)');
        } else {
          setStyle(m.el, 'transform-origin', p.anchor.at + 'px ' + m.box.cy + 'px');
          setStyle(m.el, 'transform', 'scaleX(.015)');
        }
      } else {
        setStyle(m.el, 'opacity', '0');
      }
    }
  }

  function playPanel(p) {
    var i, m;
    if (p.heavy) {
      setStyle(p.svg, 'transition', 'opacity ' + S.panelFade + 'ms ' + S.ease +
        ' ' + p.delay + 'ms');
      setStyle(p.svg, 'opacity', '1');
      return;
    }
    for (i = 0; i < p.marks.length; i++) {
      m = p.marks[i];
      if (m.kind === 'stroke') {
        setStyle(m.el, 'transition', 'stroke-dashoffset ' + m.dur + 'ms ' +
          S.easeDraw + ' ' + m.delay + 'ms');
        setStyle(m.el, 'stroke-dashoffset', '0px');
      } else if (m.kind === 'block' && p.anchor) {
        setStyle(m.el, 'transition', 'transform ' + m.dur + 'ms ' + S.ease +
          ' ' + m.delay + 'ms');
        setStyle(m.el, 'transform', 'none');
      } else {
        setStyle(m.el, 'transition', 'opacity ' + m.dur + 'ms ' + S.ease +
          ' ' + m.delay + 'ms');
        setStyle(m.el, 'opacity', '1');
      }
    }
  }

  function armRows(entry) {
    var i;
    for (i = 0; i < entry.rows.length; i++) {
      /* opacity only: a transform on a table row is not reliable across browsers,
         and a row that fails to come back would hide a number */
      setStyle(entry.rows[i].el, 'opacity', '0');
    }
  }

  function playRows(entry) {
    var i, r;
    for (i = 0; i < entry.rows.length; i++) {
      r = entry.rows[i];
      setStyle(r.el, 'transition', 'opacity ' + r.dur + 'ms ' + S.ease + ' ' + r.delay + 'ms');
      setStyle(r.el, 'opacity', '1');
    }
  }

  /* charts-paper.js and charts-native.js fade their panels in by adding their own
     'in' class two animation frames after the figure is built. If those frames
     never come — which happens in headless Chrome under --virtual-time-budget, and
     is reproducible with this layer absent — the class never lands and the panels
     stay at the opacity:0 their 'anim' class gives them: a blank figure. Nothing
     here can prevent that, but a settled figure must never be an invisible one, so
     if the class is still missing when this layer finishes, put it there. It is
     exactly the class the module was going to set, and the module's own setCls
     writes the same string, so this cannot fight it. */
  var MODULE_IN = [['lqp-anim', 'lqp-in'], ['lqc-anim', 'lqc-in']];

  function ensureModuleIn(fig) {
    var i, c;
    if (!fig || !fig.getAttribute) { return; }
    for (i = 0; i < MODULE_IN.length; i++) {
      if (hasCls(fig, MODULE_IN[i][0]) && !hasCls(fig, MODULE_IN[i][1])) {
        c = fig.getAttribute('class') || '';
        fig.setAttribute('class', c + ' ' + MODULE_IN[i][1]);
      }
    }
  }

  function computedOpacity(el) {
    try {
      var v = global.getComputedStyle ? global.getComputedStyle(el).opacity : '1';
      var f = parseFloat(v);
      return isNaN(f) ? 1 : f;
    } catch (e) { return 1; }
  }

  /* The repair pass, for the hazard above. It looks before it touches: in a browser
     with a running clock every panel is already at opacity 1 by now and this does
     nothing at all. It is also what finishAll() runs, which is why calling that
     before a capture is the reliable way to photograph this page. */
  function repairPanels(entry) {
    var i, svgs = [], need = false;
    ensureModuleIn(entry.fig);
    for (i = 0; i < entry.panels.length; i++) { svgs.push(entry.panels[i].svg); }
    for (i = 0; i < svgs.length; i++) {
      if (computedOpacity(svgs[i]) < 0.999) { need = true; break; }
    }
    if (!need) { return false; }
    for (i = 0; i < svgs.length; i++) { setStyle(svgs[i], 'transition', 'none'); }
    try {
      if (entry.fig.getBoundingClientRect) { entry.fig.getBoundingClientRect(); }
    } catch (e) { /* ignore */ }
    for (i = 0; i < svgs.length; i++) { clearStyle(svgs[i], CLEAN_T); }
    return true;
  }

  function scheduleRepair(entry) {
    if (!global.setTimeout) { return; }
    if (entry.repair) { global.clearTimeout(entry.repair); }
    entry.repair = global.setTimeout(function () {
      entry.repair = null;
      repairPanels(entry);
    }, S.repairAfter);
  }

  /* The only state a settled figure is allowed to be in: the module's own. */
  /* Settling is not just "remove what we added". A transition that is still in
     flight when its declaration disappears keeps interpolating from the value it
     had reached, and an animation clock that is not advancing — headless Chrome
     under --virtual-time-budget is exactly that — leaves it there for ever. So:
     cancel every transition first, then drop the start state, then force one
     style resolution, and only then let the transitions back. Whatever the clock
     is doing, the figure is now in its finished state.
     One forced read per figure, after its reveal: no animation is in flight by
     then, so this costs nothing that the next frame would not have cost anyway.
     It also cancels the panel fade the async chart modules start themselves,
     which has the same problem for the same reason. */
  function settle(entry) {
    var pi, i, p, all = [];
    if (entry.timer) { global.clearTimeout(entry.timer); entry.timer = null; }
    if (entry.watch) { global.clearTimeout(entry.watch); entry.watch = null; }

    for (pi = 0; pi < entry.panels.length; pi++) {
      p = entry.panels[pi];
      all.push(p.svg);
      for (i = 0; i < p.marks.length; i++) { all.push(p.marks[i].el); }
    }
    for (i = 0; i < entry.rows.length; i++) { all.push(entry.rows[i].el); }

    for (i = 0; i < all.length; i++) { setStyle(all[i], 'transition', 'none'); }
    for (i = 0; i < all.length; i++) { clearStyle(all[i], CLEAN); }
    try {
      if (entry.fig.getBoundingClientRect) { entry.fig.getBoundingClientRect(); }
    } catch (e) { /* ignore */ }
    for (i = 0; i < all.length; i++) { clearStyle(all[i], CLEAN_T); }

    entry.state = 'done';
    ensureModuleIn(entry.fig);
    if (entry.fig.setAttribute) { entry.fig.setAttribute('data-lqm-state', 'done'); }
    if (entry.fig.parentNode) { entry.fig.parentNode.__lqmPlayed = true; }
    scheduleRepair(entry);
  }

  function play(entry) {
    if (entry.state !== 'armed') { return; }
    entry.state = 'playing';
    if (entry.watch) { global.clearTimeout(entry.watch); entry.watch = null; }
    if (entry.fig.setAttribute) { entry.fig.setAttribute('data-lqm-state', 'playing'); }

    /* The settle timer is set here, not inside the frame callbacks: a tab in the
       background gets no frames, and a figure that never got its end state must
       still end up in the finished state rather than stay half-drawn. settle()
       moves the state to 'done', which is what the frame callback below checks,
       so whichever runs second does nothing. */
    entry.timer = global.setTimeout(function () { settle(entry); }, entry.total + 180);

    /* two frames: the armed state has to be painted before the end state is set,
       or the browser has nothing to transition from */
    rAF(function () {
      rAF(function () {
        if (entry.state !== 'playing') { return; }
        var pi;
        for (pi = 0; pi < entry.panels.length; pi++) { playPanel(entry.panels[pi]); }
        playRows(entry);
      });
    });
  }

  /* ====================================================================== *
   * 8. THE SHARED READOUT                                                  *
   * ====================================================================== *
   * All three modules already carry a tooltip of their own, wired to the same hit
   * targets, and those tooltips are the figure's voice — they are built from the
   * module's own numbers. So on a pointer this layer stays silent and lets them
   * talk. The readout below exists for the two cases they do not cover: a figure
   * with no tooltip of its own, and the keyboard path, where no pointer event
   * ever fires.
   */
  var readoutEl = null;

  function readout() {
    if (!doc) { return null; }
    if (readoutEl && readoutEl.parentNode) { return readoutEl; }
    readoutEl = doc.createElement('div');
    readoutEl.className = 'lqm-readout';
    readoutEl.setAttribute('aria-hidden', 'true');
    (doc.body || doc.documentElement).appendChild(readoutEl);
    return readoutEl;
  }

  /* charts-main writes its tooltip text into data-tip, with || between lines and
     <b>/<i> for emphasis; charts-paper and charts-native hang plain text with \n
     on a property. Take either, render as text — never as markup. */
  function tipTextOf(el) {
    var n = el, i = 0, raw = null;
    while (n && n.getAttribute && i < 6) {
      if (n.__lqpTip) { raw = String(n.__lqpTip); break; }
      if (n.__lqcTip) { raw = String(n.__lqcTip); break; }
      var d = n.getAttribute('data-tip');
      if (d) {
        raw = String(d).replace(/\|\|/g, '\n').replace(/<[^>]*>/g, '');
        break;
      }
      n = n.parentNode;
      i += 1;
    }
    if (!raw) { return null; }
    var lines = raw.split('\n'), out = [], j;
    for (j = 0; j < lines.length; j++) {
      if (lines[j] !== '') { out.push(lines[j]); }
    }
    return out.length ? out : null;
  }

  function showReadout(lines, x, y) {
    var el = readout();
    if (!el) { return; }
    while (el.firstChild) { el.removeChild(el.firstChild); }
    var b = doc.createElement('b');
    b.appendChild(doc.createTextNode(lines[0]));
    el.appendChild(b);
    var i;
    for (i = 1; i < lines.length; i++) {
      var s = doc.createElement('span');
      s.appendChild(doc.createTextNode(lines[i]));
      el.appendChild(s);
      if (i < lines.length - 1) { el.appendChild(doc.createElement('br')); }
    }
    var vw = global.innerWidth || 360;
    el.style.left = Math.round(Math.max(90, Math.min(vw - 90, x))) + 'px';
    el.style.top = Math.round(Math.max(44, y)) + 'px';
    el.className = 'lqm-readout is-on';
  }

  function hideReadout() {
    if (readoutEl) { readoutEl.className = 'lqm-readout'; }
  }

  /* ====================================================================== *
   * 9. EMPHASIS                                                            *
   * ====================================================================== */

  var HOT = null;   /* { panel: svgEl, marks: [el], focus: bool } */

  function clearHot() {
    if (!HOT) { return; }
    var i;
    for (i = 0; i < HOT.marks.length; i++) {
      HOT.marks[i].removeAttribute(ON_ATTR);
      HOT.marks[i].removeAttribute(FOCUS_ATTR);
    }
    if (HOT.panel) { HOT.panel.removeAttribute(HOT_ATTR); }
    HOT = null;
    hideReadout();
  }

  function setHot(hit, viaFocus, x, y) {
    var group = hit.__lqmGroup, panel = hit.__lqmPanel, i;
    if (!group || !panel) { clearHot(); return; }
    if (HOT && HOT.hit === hit && HOT.focus === viaFocus) {
      if (!viaFocus && HOT.lines) { showReadout(HOT.lines, x, y); }
      return;
    }
    clearHot();
    for (i = 0; i < group.length; i++) {
      group[i].setAttribute(ON_ATTR, '1');
      if (viaFocus) { group[i].setAttribute(FOCUS_ATTR, '1'); }
    }
    panel.setAttribute(HOT_ATTR, '1');
    HOT = { hit: hit, panel: panel, marks: group, focus: viaFocus, lines: null };

    var entry = hit.__lqmEntry;
    var wants = viaFocus || !(entry && entry.hasTip);
    if (!wants) { return; }
    var lines = tipTextOf(hit);
    if (!lines) { return; }
    HOT.lines = lines;
    showReadout(lines, x, y);
  }

  function hitUnder(node) {
    var n = node, i = 0;
    while (n && n.getAttribute && i < 8) {
      if (n.__lqmGroup) { return n; }
      if (n.getAttribute(MARK_ATTR) && n.__lqmHit) { return n.__lqmHit; }
      n = n.parentNode;
      i += 1;
    }
    return null;
  }

  function onPointerOver(ev) {
    var hit = hitUnder(ev.target);
    if (!hit) { clearHot(); return; }
    setHot(hit, false,
      (ev.clientX === undefined ? 0 : ev.clientX),
      (ev.clientY === undefined ? 0 : ev.clientY) - 10);
  }

  function onFocusIn(ev) {
    var hit = hitUnder(ev.target);
    if (!hit) { clearHot(); return; }
    var x = 0, y = 0;
    try {
      var r = hit.getBoundingClientRect();
      x = r.left + r.width / 2;
      y = r.top + Math.min(r.height / 2, 18);
    } catch (e) { /* ignore */ }
    setHot(hit, true, x, y);
  }

  var listenersBound = false;

  function bindGlobal() {
    if (listenersBound || !doc || !doc.addEventListener) { return; }
    listenersBound = true;
    /* One delegated set for the whole page: no per-mark handlers, and nothing to
       re-bind when a module redraws a figure on resize. */
    try {
      doc.addEventListener('pointerover', onPointerOver, { passive: true });
      doc.addEventListener('pointercancel', clearHot, { passive: true });
      global.addEventListener('scroll', clearHot, { passive: true });
      global.addEventListener('blur', clearHot, { passive: true });
    } catch (e) {
      doc.addEventListener('pointerover', onPointerOver, false);
      global.addEventListener('scroll', clearHot, false);
    }
    doc.addEventListener('focusin', onFocusIn, false);
    doc.addEventListener('focusout', function () { if (HOT && HOT.focus) { clearHot(); } }, false);
    doc.addEventListener('keydown', onKeyDown, false);
  }

  /* ====================================================================== *
   * 10. OPTIONAL KEYBOARD TRAVERSAL  (data-lq-motion-keys, off by default) *
   * ====================================================================== *
   * One tab stop per figure, arrows between marks, Escape to drop the emphasis.
   * Tab and Shift+Tab are never intercepted, so focus is never trapped. Every
   * figure already ships a full screen-reader data table; this is for sighted
   * keyboard users who want the same readout a mouse gets.
   */
  function wantsKeys(fig) {
    if (!fig || !fig.getAttribute) { return false; }
    var v = closestAttr(fig, 'data-lq-motion-keys');
    return v !== null && v !== 'false' && v !== '0' && v !== 'off';
  }

  function enableKeys(entry) {
    var hits = entry.hits, i, h, lines;
    for (i = 0; i < hits.length; i++) {
      h = hits[i];
      if (!h.__lqmGroup) { continue; }
      h.setAttribute('tabindex', entry.keyIndex === i ? '0' : '-1');
      h.setAttribute('role', 'img');
      lines = tipTextOf(h);
      if (lines) { h.setAttribute('aria-label', lines.join('. ')); }
      h.__lqmKeyIndex = i;
    }
  }

  function moveKey(entry, from, step) {
    var hits = entry.hits, n = hits.length, i, j, h;
    for (i = 1; i <= n; i++) {
      j = (from + step * i) % n;
      if (j < 0) { j += n; }
      h = hits[j];
      if (h && h.__lqmGroup) {
        var k;
        for (k = 0; k < n; k++) {
          if (hits[k] && hits[k].setAttribute && hits[k].__lqmGroup) {
            hits[k].setAttribute('tabindex', k === j ? '0' : '-1');
          }
        }
        entry.keyIndex = j;
        try { h.focus(); } catch (e) { /* ignore */ }
        return true;
      }
    }
    return false;
  }

  function onKeyDown(ev) {
    var key = ev.key;
    if (key === 'Escape' || key === 'Esc') {
      if (HOT) {
        var el = HOT.hit;
        clearHot();
        if (el && el.blur) { try { el.blur(); } catch (e) { /* ignore */ } }
      }
      return;
    }
    var hit = hitUnder(ev.target);
    if (!hit || hit.__lqmKeyIndex === undefined) { return; }
    var entry = hit.__lqmEntry;
    if (!entry || !entry.keys) { return; }
    var step = 0;
    if (key === 'ArrowRight' || key === 'ArrowDown' || key === 'Right' || key === 'Down') { step = 1; }
    else if (key === 'ArrowLeft' || key === 'ArrowUp' || key === 'Left' || key === 'Up') { step = -1; }
    else if (key === 'Home') { step = 'home'; }
    else if (key === 'End') { step = 'end'; }
    else { return; }
    var ok;
    if (step === 'home') { ok = moveKey(entry, -1, 1); }
    else if (step === 'end') { ok = moveKey(entry, entry.hits.length, -1); }
    else { ok = moveKey(entry, hit.__lqmKeyIndex, step); }
    if (ok && ev.preventDefault) { ev.preventDefault(); }
  }

  /* ====================================================================== *
   * 11. WIRING ONE FIGURE                                                  *
   * ====================================================================== */

  var ENTRIES = [];
  var io = null;
  var ioSeen = false;   /* has the browser ever delivered an intersection record? */
  var ioDead = false;   /* it has had its chance and it has not */

  function onScreen(el) {
    if (!el || !el.getBoundingClientRect) { return true; }
    var r;
    try { r = el.getBoundingClientRect(); } catch (e) { return true; }
    var h = global.innerHeight || (doc && doc.documentElement && doc.documentElement.clientHeight) || 0;
    if (!h) { return true; }
    return r.bottom > 0 && r.top < h + 1;
  }

  function observer() {
    if (io || !('IntersectionObserver' in global)) { return io; }
    io = new global.IntersectionObserver(function (entries) {
      var i, en, entry;
      /* An IntersectionObserver reports on every target it is given, intersecting
         or not, as soon as it starts observing. So one callback of any kind is
         proof that this browser will tell us when a figure comes into view. */
      ioSeen = true;
      for (i = 0; i < entries.length; i++) {
        en = entries[i];
        if (!en.isIntersecting) { continue; }
        entry = en.target.__lqm;
        io.unobserve(en.target);
        if (entry) { play(entry); }
      }
    }, { threshold: S.threshold, rootMargin: S.rootMargin });
    return io;
  }

  function wire(fig, adapter) {
    if (!fig || !fig.getAttribute || fig.getAttribute('data-lqm')) { return null; }

    var panelEls = list(fig, adapter.panels), panels = [], i, p;
    for (i = 0; i < panelEls.length; i++) {
      p = collectPanel(panelEls[i], adapter);
      if (p) { pair(p); panels.push(p); }
    }

    var rows = [], rowEls = adapter.rows ? list(fig, adapter.rows) : [];
    for (i = 0; i < rowEls.length; i++) {
      if (!inNoGo(rowEls[i], fig)) { rows.push({ el: rowEls[i] }); }
    }

    /* The guard: a figure with nothing recognisable in it is left exactly alone —
       no attributes, no styles, no listeners. A fallback note, an error message or
       a chart drawn in some way this layer has never seen simply does not animate. */
    if (!panels.length && !rows.length) { return null; }

    var entry = {
      fig: fig, adapter: adapter, panels: panels, rows: rows, hits: [],
      state: 'idle', timer: null, watch: null, repair: null, total: 0,
      keys: false, keyIndex: 0,
      hasTip: !!(fig.querySelector &&
        (fig.querySelector('.lqp-tip') || fig.querySelector('.lqc-tip'))) ||
        closestAttr(fig, 'data-lq-tips') === '1'
    };

    fig.setAttribute('data-lqm', '1');
    fig.__lqm = entry;

    /* tag the marks: this, plus the three state attributes, is all motion.css
       has to go on — with the script absent nothing in it matches anything */
    var pi, j;
    for (pi = 0; pi < panels.length; pi++) {
      p = panels[pi];
      for (j = 0; j < p.marks.length; j++) {
        p.marks[j].el.setAttribute(MARK_ATTR, p.marks[j].kind);
      }
      for (j = 0; j < p.hits.length; j++) {
        p.hits[j].el.__lqmEntry = entry;
        entry.hits.push(p.hits[j].el);
      }
    }
    for (j = 0; j < rows.length; j++) {
      rows[j].el.setAttribute(MARK_ATTR, 'row');
    }

    if (wantsKeys(fig)) { entry.keys = true; enableKeys(entry); }
    bindGlobal();
    ENTRIES.push(entry);

    var played = fig.parentNode && fig.parentNode.__lqmPlayed;
    if (isInstant() || ioDead || played) {
      /* nothing is ever armed in this branch, so there is no state to undo and
         no frame in which the figure is incomplete */
      entry.state = 'done';
      ensureModuleIn(fig);
      fig.setAttribute('data-lqm-state', 'done');
      scheduleRepair(entry);
      return entry;
    }

    schedule(entry);
    for (pi = 0; pi < panels.length; pi++) { armPanel(panels[pi]); }
    armRows(entry);
    entry.state = 'armed';
    fig.setAttribute('data-lqm-state', 'armed');

    var ob = observer();
    if (ob) {
      ob.observe(fig);
      /* Headless Chrome under --virtual-time-budget produces no intersection
         callbacks at all, so a figure armed for a capture would be captured
         half-drawn. If nothing has observed this figure by now and it is in fact
         on screen, play it. A figure that is genuinely below the fold is left
         armed, which is the whole point of a scroll reveal. */
      entry.watch = global.setTimeout(function () {
        if (entry.state !== 'armed') { return; }
        if (!ioSeen) {
          /* Not one record has arrived, for any figure, since this page started:
             there is no intersection machinery here (headless Chrome under
             --virtual-time-budget behaves exactly like this). Rather than hold
             every figure in its start state for a capture, give up on the reveal
             and hand the whole page its finished state. */
          ioDead = true;
          finishAll();
          return;
        }
        if (onScreen(entry.fig)) { play(entry); }
      }, S.watchdog);
    } else {
      play(entry);
    }
    return entry;
  }

  /* ====================================================================== *
   * 12. SCANNING, AND KEEPING UP WITH RE-RENDERS                           *
   * ====================================================================== */

  function scan() {
    if (!doc || !doc.querySelectorAll) { return 0; }
    var n = 0, a, i, figs, j;
    for (a = 0; a < ADAPTERS.length; a++) {
      figs = list(doc, ADAPTERS[a].fig);
      for (i = 0; i < figs.length; i++) {
        j = wire(figs[i], ADAPTERS[a]);
        if (j) { n += 1; }
      }
    }
    return n;
  }

  /* A module that redraws on resize throws its old DOM away. Drop the entries
     that went with it, so neither this layer nor the observer holds on to
     detached nodes. */
  function prune() {
    var i, e;
    for (i = ENTRIES.length - 1; i >= 0; i--) {
      e = ENTRIES[i];
      if (e.fig && e.fig.parentNode) { continue; }
      if (e.timer) { global.clearTimeout(e.timer); e.timer = null; }
      if (e.watch) { global.clearTimeout(e.watch); e.watch = null; }
      if (e.repair) { global.clearTimeout(e.repair); e.repair = null; }
      if (io && e.fig) { try { io.unobserve(e.fig); } catch (err) { /* ignore */ } }
      if (HOT && HOT.panel && !HOT.panel.parentNode) { clearHot(); }
      ENTRIES.splice(i, 1);
    }
  }

  var mo = null, scanning = false;

  function scanSafe() {
    if (scanning) { return 0; }
    scanning = true;
    var found = 0;
    try { prune(); found = scan(); } catch (e) {
      if (global.console && global.console.warn) {
        global.console.warn('[LiEQMotion] scan: ' + (e && e.message ? e.message : e));
      }
    }
    /* drop the records our own attribute writes just generated, so the observer
       cannot feed itself */
    if (mo && mo.takeRecords) { try { mo.takeRecords(); } catch (e2) { /* ignore */ } }
    scanning = false;
    return found;
  }

  function watch() {
    if (mo || !global.MutationObserver || !doc || !doc.body) { return; }
    mo = new global.MutationObserver(function () { scanSafe(); });
    try {
      mo.observe(doc.body, { childList: true, subtree: true });
    } catch (e) { mo = null; }
  }

  /* ====================================================================== *
   * 13. API                                                                *
   * ====================================================================== */

  function apply(opts) {
    if (!opts) { return; }
    var k;
    for (k in opts) {
      if (opts.hasOwnProperty(k) && S.hasOwnProperty(k)) { S[k] = opts[k]; }
    }
  }

  function init(opts) {
    /* window.LIEQ_MOTION = { sweep: 160 } set before this script runs is read
       here, which is the only way to change a knob before the first figure is
       wired on a page that lets this file auto-initialise. */
    apply(global.LIEQ_MOTION);
    apply(opts);
    if (!doc) { return 0; }
    if (typeof S.dim === 'number' && doc.documentElement &&
        doc.documentElement.style && doc.documentElement.style.setProperty) {
      try { doc.documentElement.style.setProperty('--lqm-dim', String(S.dim)); }
      catch (e) { /* ignore */ }
    }
    var n = scanSafe();
    watch();
    return n;
  }

  function finishAll() {
    FORCED_INSTANT = true;
    var i;
    ioDead = true;
    for (i = 0; i < ENTRIES.length; i++) {
      if (ENTRIES[i].state !== 'done') { settle(ENTRIES[i]); }
      repairPanels(ENTRIES[i]);
    }
    clearHot();
    return ENTRIES.length;
  }

  var API = {
    version: VERSION,
    settings: S,
    init: init,
    refresh: function () { scanSafe(); return ENTRIES.length; },
    finishAll: finishAll,
    isInstant: isInstant,
    /* for tests and for anyone debugging a figure in the console */
    _entries: ENTRIES,
    _adapters: ADAPTERS,
    _kindOf: kindOf,
    _detectAnchor: detectAnchor
  };

  global.LiEQMotion = API;
  if (typeof module !== 'undefined' && module.exports) { module.exports = API; }

  if (doc) {
    if (doc.readyState === 'loading') {
      doc.addEventListener('DOMContentLoaded', function () { init(); }, false);
    } else {
      init();
    }
    /* the chart modules fetch their data, so figures keep arriving after load */
    if (global.addEventListener) {
      global.addEventListener('load', function () { scanSafe(); watch(); }, false);
    }
  }
})(typeof window !== 'undefined' ? window : this);
