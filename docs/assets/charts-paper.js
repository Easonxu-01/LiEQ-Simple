/*!
 * charts-paper.js - the manuscript's Figure 3 and Figure 4, redrawn as live inline SVG.
 *
 * No build step, no framework, no bundler, no CDN, no chart library. Vanilla ES5 syntax
 * plus inline SVG; the only runtime features beyond ES5 are fetch/XHR and Promise, both
 * feature-detected (without them the mount shows a short note instead of throwing).
 *
 * Fig. 3  per-band CKA to the Dense teacher, divided by the teacher's re-truncation
 *         noise ceiling: row a on the TPV xy feature plane, row b on the voxel logits.
 * Fig. 4  band-wise energy relative to the Dense teacher: shares (a, b), the real /
 *         pseudo split (c, d) and the paired LiEqualizer - LiSeer change (e, f).
 *
 * Data is NOT embedded. The three JSON files written by the engineer who produced the
 * published matplotlib figures stay the single source of truth and are fetched at
 * runtime from <script dir>/data/ (default 'assets/data/'):
 *
 *   fig3a_band_cka_feature_planes.json
 *   fig3b_band_cka_voxel_logits.json
 *   fig4_energy_real.json
 *
 * Public API: window.LiEQPaperCharts (also merged into window.LiEQCharts).
 * See charts-paper.README.md.
 */
;(function (global) {
  'use strict';

  var VERSION = '1.0.0';

  /* ------------------------------------------------------------------ files */

  var FILES = {
    fig3a: 'fig3a_band_cka_feature_planes.json',
    fig3b: 'fig3b_band_cka_voxel_logits.json',
    fig4: 'fig4_energy_real.json'
  };

  var DEFAULT_BASE = 'assets/data/';

  /* ------------------------------------------------------------------ style */

  var STYLE_ID = 'lqp-style';

  var CSS = [
    '.lqp{',
    '  position:relative;margin:0;padding:0;',
    '  --lqp-sans:var(--sans,\'Inter\',system-ui,sans-serif);',
    '  --lqp-disp:var(--disp,\'Space Grotesk\',var(--lqp-sans));',
    '  --lqp-mono:var(--mono,\'JetBrains Mono\',ui-monospace,monospace);',
    '  --lqp-ink:var(--text,#17202c);',
    '  --lqp-muted:var(--muted,#647184);',
    '  --lqp-hair:var(--line,rgba(52,72,96,.12));',
    '  --lqp-grid:rgba(52,72,96,.13);',
    '  --lqp-axis:rgba(52,72,96,.34);',
    '  --lqp-edge:rgba(52,72,96,.42);',
    '  --lqp-halo:#ffffff;',
    '  --lqp-surface:#ffffff;',
    '  --lqp-shade:#f3f2ef;',
    '  --lqp-eyebrow:#377fba;',
    '  --lqp-link:#2f78bf;',
    '  --lqp-hover:#58b9eb;',
    '  color:var(--lqp-ink);font-family:var(--lqp-sans);',
    '}',
    '.lqp[data-lqp-theme="dark"]{',
    '  --lqp-ink:#dbe7f5;--lqp-muted:#9cc8f3;',
    '  --lqp-hair:rgba(190,215,245,.18);--lqp-grid:rgba(190,215,245,.15);',
    '  --lqp-axis:rgba(190,215,245,.40);--lqp-edge:rgba(190,215,245,.42);',
    '  --lqp-halo:#0c1322;--lqp-surface:#0c1322;',
    '  --lqp-shade:rgba(190,215,245,.07);',
    '  --lqp-eyebrow:#9cc8f3;--lqp-link:#56adf0;--lqp-hover:#bfe0ff;',
    '}',
    '.lqp-fig{margin:0;padding:0}',
    '.lqp-row{margin:0 0 4px}',
    '.lqp-row+.lqp-row{margin-top:22px}',
    '.lqp-rowhead{display:flex;flex-wrap:wrap;gap:4px 14px;align-items:baseline;',
    '  margin:0 0 8px;font-family:var(--lqp-mono);font-size:10.5px;letter-spacing:.07em;',
    '  text-transform:uppercase;color:var(--lqp-eyebrow)}',
    '.lqp-rowhead span{text-transform:none;font-size:11px;letter-spacing:.02em;',
    '  font-weight:600}',
    '.lqp-rowhead i{font-style:normal;color:var(--lqp-muted);letter-spacing:.04em;',
    '  text-transform:none}',
    '.lqp-cols{display:flex;gap:14px;margin:0 0 6px;font-family:var(--lqp-mono);',
    '  font-size:10.5px;font-weight:700;letter-spacing:.06em;color:var(--lqp-ink)}',
    '.lqp-cols span{flex:1 1 0;min-width:0;text-align:center}',
    '.lqp-panels{display:grid;gap:14px;align-items:start;',
    '  grid-template-columns:repeat(var(--lqp-cols,1),minmax(0,1fr))}',
    '.lqp-panels>svg{display:block;width:100%;height:auto;overflow:visible;',
    '  max-width:var(--lqp-pmax,none);margin:0}',
    '.lqp-anim .lqp-panels>svg{opacity:0;transform:translateY(5px);',
    '  transition:opacity .45s ease,transform .45s ease}',
    '.lqp-anim.lqp-in .lqp-panels>svg{opacity:1;transform:none}',
    '.lqp-xlab{font-family:var(--lqp-mono);font-size:10px;color:var(--lqp-muted);',
    '  text-align:center;margin:2px 0 0}',
    '.lqp text{font-family:var(--lqp-mono)}',
    '.lqp .t-tick{font-size:9.5px;fill:var(--lqp-muted)}',
    '.lqp .t-pan{font-size:10.5px;font-weight:600;fill:var(--lqp-ink)}',
    '.lqp .t-sub{font-size:8.4px;fill:var(--lqp-muted)}',
    '.lqp .t-ax{font-size:9px;fill:var(--lqp-muted)}',
    '.lqp .t-anno{font-size:8.4px;fill:var(--lqp-muted);stroke:var(--lqp-halo);',
    '  stroke-width:2.4px;paint-order:stroke;stroke-linejoin:round}',
    '.lqp .t-flag{font-size:9.5px;stroke:var(--lqp-halo);stroke-width:2.4px;',
    '  paint-order:stroke;stroke-linejoin:round}',
    '.lqp .t-key{font-size:8.6px;fill:var(--lqp-muted)}',
    '.lqp .ax{fill:none;stroke:var(--lqp-axis);stroke-width:1}',
    '.lqp .gr{fill:none;stroke:var(--lqp-grid);stroke-width:1;shape-rendering:crispEdges}',
    '.lqp .shade{fill:var(--lqp-shade);stroke:none}',
    '.lqp .hfedge{fill:none;stroke:var(--lqp-axis);stroke-width:1;stroke-dasharray:2 3;',
    '  opacity:.7}',
    '.lqp .edge{fill:none;stroke:var(--lqp-edge);stroke-width:.7}',
    '.lqp .hit{fill:#ffffff;fill-opacity:0;stroke:none;pointer-events:all}',
    '.lqp-legend{display:flex;flex-wrap:wrap;gap:6px 14px;align-items:center;',
    '  margin:0 0 9px;font-family:var(--lqp-mono);font-size:11px;color:var(--lqp-muted)}',
    '.lqp-legend.is-row{margin:0 0 8px}',
    '.lqp-key{display:inline-flex;align-items:center;gap:6px;white-space:nowrap}',
    '.lqp-key svg{display:block;flex:0 0 auto}',
    '.lqp-axes{font-family:var(--lqp-mono);font-size:10.5px;line-height:1.5;',
    '  color:var(--lqp-muted);margin:0 0 9px}',
    '.lqp-pub{font-family:var(--lqp-mono);font-size:9.6px;line-height:1.55;',
    '  color:var(--lqp-muted);margin:10px 0 0}',
    '.lqp-note{font-family:var(--lqp-sans);font-size:12.5px;line-height:1.55;',
    '  color:var(--lqp-muted);margin:12px 0 0;max-width:80ch}',
    '.lqp-note b{font-family:var(--lqp-mono);font-size:10px;font-weight:600;',
    '  letter-spacing:.09em;text-transform:uppercase;color:var(--lqp-eyebrow);',
    '  margin-right:.45em}',
    '.lqp-foot{margin:10px 0 0}',
    '.lqp-toggle{font-family:var(--lqp-mono);font-size:10.5px;letter-spacing:.06em;',
    '  text-transform:uppercase;color:var(--lqp-eyebrow);background:transparent;',
    '  border:1px solid var(--lqp-hair);border-radius:999px;padding:5px 11px;cursor:pointer}',
    '.lqp-toggle:hover{color:var(--lqp-hover);border-color:var(--lqp-hover)}',
    '.lqp-toggle:focus-visible{outline:2px solid var(--lqp-link);outline-offset:2px}',
    '.lqp-sr{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;',
    '  clip:rect(0 0 0 0);clip-path:inset(50%);white-space:nowrap;border:0}',
    '.lqp-sr.is-open{position:static;width:auto;height:auto;margin:14px 0 0;',
    '  overflow:auto;clip:auto;clip-path:none;white-space:normal}',
    '.lqp table{border-collapse:collapse;font-family:var(--lqp-mono);font-size:11px;',
    '  width:100%}',
    '.lqp table+table{margin-top:16px}',
    '.lqp caption{text-align:left;font-size:11px;color:var(--lqp-muted);padding:0 0 6px}',
    '.lqp th,.lqp td{border-bottom:1px solid var(--lqp-hair);padding:3px 8px 3px 0;',
    '  text-align:right;white-space:nowrap}',
    '.lqp th:first-child,.lqp td:first-child{text-align:left}',
    '.lqp thead th{color:var(--lqp-muted);font-weight:600}',
    '.lqp-tip{position:absolute;z-index:6;left:0;top:0;pointer-events:none;opacity:0;',
    '  transform:translate(-50%,-112%);background:#0c1322;color:#dbe7f5;',
    '  font-family:var(--lqp-mono);font-size:10.5px;line-height:1.5;padding:6px 9px;',
    '  border-radius:8px;box-shadow:0 10px 24px -14px rgba(12,19,34,.8);',
    '  white-space:pre;max-width:19rem;transition:opacity .12s ease}',
    '.lqp-tip.is-on{opacity:1}',
    '.lqp-missing{border:1px dashed var(--lqp-hair);border-radius:12px;padding:16px 18px}',
    '.lqp-missing p{margin:0}',
    '.lqp-missing .h{font-family:var(--lqp-mono);font-size:10.5px;letter-spacing:.08em;',
    '  text-transform:uppercase;color:var(--lqp-eyebrow)}',
    '.lqp-missing .d{font-size:12.5px;line-height:1.55;color:var(--lqp-muted);',
    '  margin-top:5px}',
    '@media (prefers-reduced-motion:reduce){',
    '  .lqp-anim .lqp-panels>svg{opacity:1 !important;transform:none !important}',
    '  .lqp *{animation:none !important;transition:none !important}',
    '}'
  ].join('\n');

  /* ---------------------------------------------------------------- dom util */

  var SVGNS = 'http://www.w3.org/2000/svg';
  var UID = 0;

  function doc() {
    return global.document || null;
  }

  function warn(where, err) {
    if (global.console && global.console.warn) {
      global.console.warn('[LiEQPaperCharts] ' + where + ': ' +
        (err && err.message ? err.message : err));
    }
  }

  function uid(prefix) {
    UID += 1;
    return 'lqp-' + prefix + '-' + UID;
  }

  function injectCSS() {
    var D = doc();
    if (!D || !D.createElement) return;
    if (D.getElementById && D.getElementById(STYLE_ID)) return;
    var s = D.createElement('style');
    s.setAttribute('id', STYLE_ID);
    s.appendChild(D.createTextNode(CSS));
    (D.head || D.documentElement || D.body).appendChild(s);
  }

  function clear(node) {
    while (node && node.firstChild) node.removeChild(node.firstChild);
  }

  function hEl(tag, cls, parent, content) {
    var n = doc().createElement(tag);
    if (cls) n.setAttribute('class', cls);
    if (content !== undefined && content !== null && content !== '') {
      n.appendChild(doc().createTextNode(String(content)));
    }
    if (parent) parent.appendChild(n);
    return n;
  }

  function sEl(tag, attrs, parent) {
    var n = doc().createElementNS(SVGNS, tag);
    if (attrs) {
      for (var k in attrs) {
        if (attrs.hasOwnProperty(k) && attrs[k] !== null && attrs[k] !== undefined) {
          n.setAttribute(k, attrs[k]);
        }
      }
    }
    if (parent) parent.appendChild(n);
    return n;
  }

  function sTxt(parent, x, y, content, cls, attrs) {
    var a = attrs || {};
    a.x = r2(x);
    a.y = r2(y);
    if (cls) a['class'] = cls;
    var t = sEl('text', a, parent);
    t.appendChild(doc().createTextNode(String(content)));
    return t;
  }

  /* Multi-line SVG text: the JSON keeps literal \n in axis labels and annotations. */
  function sLines(parent, x, y, lines, cls, attrs, lh) {
    var a = attrs || {};
    a.x = r2(x);
    a.y = r2(y);
    if (cls) a['class'] = cls;
    var t = sEl('text', a, parent), i, ts;
    for (i = 0; i < lines.length; i++) {
      ts = sEl('tspan', { x: r2(x), dy: i === 0 ? 0 : r2(lh || 10) }, t);
      ts.appendChild(doc().createTextNode(String(lines[i])));
    }
    return t;
  }

  function sLine(parent, x1, y1, x2, y2, cls, attrs) {
    var a = attrs || {};
    a.x1 = r2(x1); a.y1 = r2(y1); a.x2 = r2(x2); a.y2 = r2(y2);
    if (cls) a['class'] = cls;
    return sEl('line', a, parent);
  }

  function sRect(parent, x, y, w, h, attrs) {
    var a = attrs || {};
    a.x = r2(x); a.y = r2(y);
    a.width = r2(Math.max(0, w)); a.height = r2(Math.max(0, h));
    return sEl('rect', a, parent);
  }

  function setStyle(node, str) {
    if (node && node.setAttribute) node.setAttribute('style', str);
  }

  function setCls(node, str) {
    if (node && node.setAttribute) node.setAttribute('class', str);
  }

  /* ---------------------------------------------------------------- numbers */

  function r2(v) {
    return Math.round(v * 100) / 100;
  }

  function trimZeros(s) {
    if (s.indexOf('.') < 0) return s;
    s = s.replace(/0+$/, '');
    if (s.charAt(s.length - 1) === '.') s = s.slice(0, -1);
    return s;
  }

  /* Python's "{:.<p>g}", so printed values match the published figure's style. */
  function gf(v, p) {
    if (v === null || v === undefined || !isFinite(v)) return 'n/a';
    if (v === 0) return '0';
    p = p || 4;
    var es = Math.abs(v).toExponential(p - 1);
    var parts = es.split('e');
    var e = parseInt(parts[1], 10);
    var sign = v < 0 ? '−' : '';
    if (e < -4 || e >= p) {
      var mant = trimZeros(parts[0]);
      var ae = Math.abs(e);
      return sign + mant + 'e' + (e < 0 ? '-' : '+') + (ae < 10 ? '0' + ae : String(ae));
    }
    var dec = p - 1 - e;
    if (dec < 0) dec = 0;
    return sign + trimZeros(Math.abs(v).toFixed(dec));
  }

  function linScale(d0, d1, r0, r1) {
    var m = (d1 - d0) === 0 ? 0 : (r1 - r0) / (d1 - d0);
    return function (v) { return r0 + (v - d0) * m; };
  }

  function clamp(v, lo, hi) {
    return v < lo ? lo : (v > hi ? hi : v);
  }

  function isNum(v) {
    return typeof v === 'number' && isFinite(v);
  }

  /* --------------------------------------------------------------- colours */

  /* The paper's own colours, used as-is in light theme. On the dark instrument card
     #52514e and #2a78d6 go nearly invisible, so dark theme swaps in the page's own
     on-dark siblings (same roles, same hue order). See README section 3. */
  var DARK_MAP = {
    '#2a78d6': '#56adf0',
    '#86b6ef': '#bfe0ff',
    '#eb6834': '#e58236',
    '#f5b896': '#f6bd8c',
    '#52514e': '#b7c4d4',
    '#8f8d88': '#93a3b6',
    '#bdbbb5': '#8796a8',
    '#5a95db': '#8cc6f6',
    '#0b0b0b': '#dbe7f5'
  };

  function colorOf(c, theme) {
    if (!c) return 'currentColor';
    if (theme !== 'dark') return c;
    var k = String(c).toLowerCase();
    return DARK_MAP[k] || c;
  }

  /* ---------------------------------------------------------------- a11y */

  function prefersReduced() {
    try {
      return !!(global.matchMedia &&
        global.matchMedia('(prefers-reduced-motion: reduce)').matches);
    } catch (e) {
      return false;
    }
  }

  function svgRoot(parent, w, h, title, desc, attrs) {
    var tid = uid('t'), did = uid('d'), a = attrs || {};
    a.viewBox = '0 0 ' + r2(w) + ' ' + r2(h);
    a.role = 'img';
    a['aria-labelledby'] = tid + ' ' + did;
    a.preserveAspectRatio = 'xMidYMid meet';
    a.focusable = 'false';
    var svg = sEl('svg', a, parent);
    var t = sEl('title', { id: tid }, svg);
    t.appendChild(doc().createTextNode(String(title)));
    var d = sEl('desc', { id: did }, svg);
    d.appendChild(doc().createTextNode(String(desc)));
    return svg;
  }

  function buildTable(host, caption, head, rows) {
    var tbl = hEl('table', null, host);
    hEl('caption', null, tbl, caption);
    var thead = hEl('thead', null, tbl);
    var tr = hEl('tr', null, thead), i, j, r;
    for (i = 0; i < head.length; i++) {
      hEl('th', null, tr, head[i]).setAttribute('scope', 'col');
    }
    var tb = hEl('tbody', null, tbl);
    for (i = 0; i < rows.length; i++) {
      r = hEl('tr', null, tb);
      for (j = 0; j < rows[i].length; j++) {
        if (j === 0) {
          hEl('th', null, r, rows[i][j]).setAttribute('scope', 'row');
        } else {
          hEl('td', null, r, rows[i][j]);
        }
      }
    }
    return tbl;
  }

  /* -------------------------------------------------------------- tooltips */

  var ACTIVE_TIP = null;
  var SCROLL_BOUND = false;

  function tipFor(node, text) {
    node.__lqpTip = text;
    return node;
  }

  function findTip(node, root) {
    var n = node, i = 0;
    while (n && n !== root && i < 8) {
      if (n.__lqpTip) return n;
      n = n.parentNode;
      i += 1;
    }
    return null;
  }

  function hideActiveTip() {
    if (ACTIVE_TIP) {
      setCls(ACTIVE_TIP, 'lqp-tip');
      ACTIVE_TIP = null;
    }
  }

  function bindScrollOnce() {
    if (SCROLL_BOUND || !global.addEventListener) return;
    SCROLL_BOUND = true;
    try {
      global.addEventListener('scroll', hideActiveTip, { passive: true });
    } catch (e) { /* ignore */ }
  }

  function attachTips(fig, host, tip) {
    if (!host || !host.addEventListener) return;
    var showing = null;

    function place(ev) {
      if (!fig.getBoundingClientRect) return;
      var box = fig.getBoundingClientRect();
      var cx = (ev.clientX !== undefined ? ev.clientX : 0) - box.left;
      var cy = (ev.clientY !== undefined ? ev.clientY : 0) - box.top;
      setStyle(tip, 'left:' + Math.round(cx) + 'px;top:' + Math.round(cy - 6) + 'px');
    }

    function show(target, ev) {
      if (showing === target) { place(ev); return; }
      showing = target;
      clear(tip);
      var lines = String(target.__lqpTip).split('\n'), i;
      for (i = 0; i < lines.length; i++) {
        if (i) hEl('br', null, tip);
        tip.appendChild(doc().createTextNode(lines[i]));
      }
      setCls(tip, 'lqp-tip is-on');
      ACTIVE_TIP = tip;
      place(ev);
    }

    function hide() {
      showing = null;
      setCls(tip, 'lqp-tip');
      if (ACTIVE_TIP === tip) ACTIVE_TIP = null;
    }

    function onMove(ev) {
      var t = findTip(ev.target, host);
      if (t) show(t, ev); else hide();
    }

    if (global.PointerEvent) {
      host.addEventListener('pointermove', onMove, { passive: true });
      host.addEventListener('pointerdown', onMove, { passive: true });
      host.addEventListener('pointerleave', hide, { passive: true });
      host.addEventListener('pointercancel', hide, { passive: true });
    } else {
      host.addEventListener('mousemove', onMove);
      host.addEventListener('mouseleave', hide);
      host.addEventListener('touchstart', function (ev) {
        var t0 = ev.touches && ev.touches[0];
        if (!t0) return;
        var t = findTip(ev.target, host);
        if (t) show(t, { clientX: t0.clientX, clientY: t0.clientY, target: ev.target });
      }, { passive: true });
      host.addEventListener('touchend', hide, { passive: true });
    }
    bindScrollOnce();
  }

  /* ------------------------------------------------------------- data load */

  var BASE = null;
  var CACHE = {};

  function scriptBase() {
    var D = doc(), src = null, all, i, s;
    if (!D) return null;
    if (D.currentScript && D.currentScript.getAttribute) {
      src = D.currentScript.getAttribute('src');
    }
    if (!src && D.getElementsByTagName) {
      all = D.getElementsByTagName('script') || [];
      for (i = all.length - 1; i >= 0; i--) {
        s = all[i].getAttribute ? all[i].getAttribute('src') : null;
        if (s && /charts-paper\.js(\?|#|$)/.test(s)) { src = s; break; }
      }
    }
    if (!src) return null;
    return src.replace(/[?#].*$/, '').replace(/[^\/]*$/, '') + 'data/';
  }

  function base() {
    if (API && API.dataBase) return API.dataBase;
    return BASE || DEFAULT_BASE;
  }

  function resolved(v) {
    if (!global.Promise) return null;
    return global.Promise.resolve(v);
  }

  function rejected(e) {
    if (!global.Promise) return null;
    return global.Promise.reject(e);
  }

  function loadJSON(url) {
    if (!global.Promise) return null;
    if (global.fetch) {
      return global.fetch(url, { credentials: 'same-origin' }).then(function (res) {
        if (!res || !res.ok) {
          throw new Error('HTTP ' + (res ? res.status : '?') + ' for ' + url);
        }
        return res.json();
      });
    }
    if (global.XMLHttpRequest) {
      return new global.Promise(function (ok, no) {
        var xhr = new global.XMLHttpRequest();
        xhr.open('GET', url, true);
        xhr.onreadystatechange = function () {
          if (xhr.readyState !== 4) return;
          if (xhr.status >= 200 && xhr.status < 300) {
            try { ok(JSON.parse(xhr.responseText)); } catch (e) { no(e); }
          } else {
            no(new Error('HTTP ' + xhr.status + ' for ' + url));
          }
        };
        xhr.onerror = function () { no(new Error('network error for ' + url)); };
        xhr.send(null);
      });
    }
    return rejected(new Error('no fetch and no XMLHttpRequest'));
  }

  function dataset(key) {
    if (!FILES[key]) return rejected(new Error('unknown data key ' + key));
    if (!CACHE[key]) {
      CACHE[key] = loadJSON(base() + FILES[key]);
      if (CACHE[key] && CACHE[key]['catch']) {
        /* a failed load must not poison the cache for a later retry */
        CACHE[key]['catch'](function () { CACHE[key] = null; });
      }
    }
    return CACHE[key];
  }

  function loadAll(keys) {
    if (!global.Promise) return null;
    var jobs = [], i;
    for (i = 0; i < keys.length; i++) jobs.push(dataset(keys[i]));
    return global.Promise.all(jobs).then(function (list) {
      var out = {}, j;
      for (j = 0; j < keys.length; j++) out[keys[j]] = list[j];
      return out;
    });
  }

  function fallback(mount, keys) {
    if (!mount) return;
    try {
      injectCSS();
      clear(mount);
      var names = [], i;
      for (i = 0; i < keys.length; i++) names.push(FILES[keys[i]]);
      var box = hEl('div', 'lqp lqp-missing', mount);
      hEl('p', 'h', box, 'Figure data unavailable');
      hEl('p', 'd', box, 'This figure is drawn live from ' + names.join(' and ') +
        ' under ' + base() + ', which could not be loaded. Serve the page over http(s) ' +
        '(the files are read with fetch, which a file:// page blocks), or read the figure ' +
        'in the paper PDF.');
    } catch (e) {
      warn('fallback', e);
    }
  }

  /* ------------------------------------------------------------- plumbing */

  function resolve(el) {
    if (!el) return null;
    if (typeof el === 'string') {
      var D = doc();
      if (!D || !D.getElementById) return null;
      el = D.getElementById(el.charAt(0) === '#' ? el.slice(1) : el);
    }
    return (el && el.appendChild) ? el : null;
  }

  function measure(mount, opts) {
    var w = (opts && opts.width) || 0;
    if (!w && mount) {
      w = mount.clientWidth || 0;
      if (!w && mount.getBoundingClientRect) {
        w = Math.round(mount.getBoundingClientRect().width || 0);
      }
    }
    if (!w || w < 140) w = (opts && opts.fallbackWidth) || 900;
    return Math.round(Math.min(w, 1180));
  }

  function themeOf(mount, opts) {
    if (opts && opts.theme) return opts.theme;
    var n = mount, i = 0, t;
    while (n && n.getAttribute && i < 8) {
      t = n.getAttribute('data-lqp-theme') || n.getAttribute('data-lqc-theme') ||
        n.getAttribute('data-lq-theme') || n.getAttribute('data-theme');
      if (t === 'dark' || t === 'light') return t;
      n = n.parentNode;
      i += 1;
    }
    return 'light';
  }

  function panelWidth(w, cols, gap, min, max) {
    var pw = Math.floor((w - gap * (cols - 1)) / cols);
    if (pw < min) pw = min;
    if (max && pw > max) pw = max;
    return pw;
  }

  function watch(mount, paint) {
    if (mount.__lqpWatching) return;
    var timer = null;
    function kick() {
      if (timer) global.clearTimeout(timer);
      timer = global.setTimeout(function () {
        timer = null;
        try { paint(); } catch (e) { warn('resize', e); }
      }, 180);
    }
    if (global.ResizeObserver) {
      try {
        var ro = new global.ResizeObserver(kick);
        ro.observe(mount);
        mount.__lqpWatching = ro;
        return;
      } catch (e) { /* fall through */ }
    }
    if (global.addEventListener) {
      global.addEventListener('resize', kick);
      mount.__lqpWatching = true;
    }
  }

  function run(el, key, keys, opts, draw) {
    var mount = resolve(el);
    if (!mount) return resolved(null);
    var o = opts || {};
    try {
      injectCSS();
    } catch (e) {
      warn('css', e);
    }
    if (!global.Promise) {
      fallback(mount, keys);
      return null;
    }
    var jobs = loadAll(keys);
    return jobs.then(function (sets) {
      var paint = function () {
        var w = measure(mount, o);
        var theme = themeOf(mount, o);
        var sig = key + '|' + theme + '|' + Math.round(w / 40);
        if (mount.__lqpSig === sig && !o.force) return mount.__lqpFig;
        clear(mount);
        var fig = draw(mount, w, theme, sets, o);
        mount.__lqpSig = sig;
        mount.__lqpFig = fig;
        return fig;
      };
      var out = paint();
      if (o.watch !== false) watch(mount, paint);
      return out;
    })['catch'](function (err) {
      warn(key, err);
      fallback(mount, keys);
      return null;
    });
  }

  /* -------------------------------------------------------------- scaffold */

  function scaffold(mount, cfg) {
    var D = doc();
    var fig = D.createElement('div');
    setCls(fig, 'lqp' + (prefersReduced() ? '' : ' lqp-anim'));
    fig.setAttribute('data-chart', cfg.key);
    if (cfg.theme === 'dark') fig.setAttribute('data-lqp-theme', 'dark');

    var top = hEl('div', 'lqp-legend', fig);
    hEl('p', 'lqp-axes', fig, cfg.axes);

    /* <figcaption> must be the first or last child of its <figure>. */
    var inner = hEl('figure', 'lqp-fig', fig);
    var body = hEl('div', 'lqp-body', inner);

    var note = hEl('figcaption', 'lqp-note', inner);
    hEl('b', null, note, cfg.noteLabel || 'Measurement basis');
    note.appendChild(D.createTextNode(cfg.note));

    var foot = hEl('div', 'lqp-foot', fig);
    var btn = hEl('button', 'lqp-toggle', foot, 'Show data table');
    btn.setAttribute('type', 'button');
    var tableId = uid('tbl');
    btn.setAttribute('aria-expanded', 'false');
    btn.setAttribute('aria-controls', tableId);

    var srBox = hEl('div', 'lqp-sr', fig);
    srBox.setAttribute('id', tableId);

    var tip = hEl('div', 'lqp-tip', fig);
    tip.setAttribute('aria-hidden', 'true');

    if (btn.addEventListener) {
      btn.addEventListener('click', function () {
        var open = btn.getAttribute('aria-expanded') === 'true';
        btn.setAttribute('aria-expanded', open ? 'false' : 'true');
        setCls(srBox, open ? 'lqp-sr' : 'lqp-sr is-open');
        clear(btn);
        btn.appendChild(D.createTextNode(open ? 'Show data table' : 'Hide data table'));
      });
    }

    mount.appendChild(fig);
    attachTips(fig, body, tip);

    if (!prefersReduced() && global.requestAnimationFrame) {
      global.requestAnimationFrame(function () {
        global.requestAnimationFrame(function () { setCls(fig, 'lqp lqp-anim lqp-in'); });
      });
    } else {
      setCls(fig, 'lqp lqp-in');
    }

    return { fig: fig, legend: top, body: body, note: note, table: srBox, tip: tip };
  }

  /* ---------------------------------------------------------------- legend */

  function keyBox(host, label, w) {
    var k = hEl('span', 'lqp-key', host);
    var svg = sEl('svg', {
      width: w || 22, height: 12, viewBox: '0 0 ' + (w || 22) + ' 12', 'aria-hidden': 'true'
    }, k);
    k.appendChild(doc().createTextNode(label));
    return svg;
  }

  function legendEntries(host, entries, theme) {
    var i, e, svg, col, mk;
    for (i = 0; i < entries.length; i++) {
      e = entries[i];
      col = colorOf(e.color, theme);
      svg = keyBox(host, e.label);
      if (e.kind === 'patch') {
        sEl('rect', { x: 0, y: 1, width: 21, height: 10, fill: col }, svg);
        sEl('rect', { x: 0, y: 1, width: 21, height: 10, 'class': 'edge' }, svg);
      } else {
        sEl('line', {
          x1: 0, y1: 6, x2: 21, y2: 6, stroke: col,
          'stroke-width': lw(e.linewidth_pt || 1.2),
          'stroke-dasharray': e.style === 'dashed' ? '3.8 2.5' : null
        }, svg);
        mk = e.marker;
        if (mk) {
          sEl('circle', {
            cx: 10.5, cy: 6, r: 2.9,
            fill: paperWhite(mk.face, 'var(--lqp-surface)', theme),
            stroke: paperWhite(mk.edge, 'var(--lqp-halo)', theme),
            'stroke-width': 0.9
          }, svg);
        }
      }
    }
    return host;
  }

  /* matplotlib points -> SVG user units at the house text scale. */
  function lw(pt) {
    return Math.round(pt * 1.2 * 100) / 100;
  }

  /* The figure's white marker faces/edges are "the page behind the figure", not a fixed
     white, so they follow the theme instead of the colour literal. */
  function paperWhite(c, token, theme) {
    if (!c) return null;
    return String(c).toLowerCase() === '#ffffff' ? token : colorOf(c, theme);
  }

  /* ----------------------------------------------------------- panel parts */

  function bandLabels(xa) {
    return (xa && xa.band_lower_edge_labels_all) ||
      ['0', '.01', '.03', '.08', '.15', '.25', '.40', '.60', '.85', '1.15', '1.50'];
  }

  function bandRange(xa, k) {
    var lab = bandLabels(xa);
    var lo = lab[k] === undefined ? String(k) : lab[k];
    var hi = lab[k + 1] === undefined ? '2.02' : lab[k + 1];
    return 'band ' + k + '  (λ ' + lo + '–' + hi + ')';
  }

  /* JetBrains Mono at 9.5px advances ~5.75px per glyph; the gutter has to hold the tick
     labels AND, where there is one, the rotated axis label, or they collide. */
  function tickWOf(labels) {
    var w = 0, i;
    for (i = 0; i < (labels || []).length; i++) {
      if (labels[i]) w = Math.max(w, String(labels[i]).length);
    }
    return w ? w * 5.75 + 7 : 7;
  }

  function labelLines(panel, hasYLabel) {
    if (!hasYLabel || !panel.y || !panel.y.label) return 0;
    return String(panel.y.label).split('\n').length;
  }

  function padLFor(panel, labels, hasYLabel) {
    var n = labelLines(panel, hasYLabel);
    return Math.ceil(tickWOf(labels) + (n ? n * 10.4 + 6 : 0) + 4);
  }

  function geomOf(panel, pw, ph, groupPadL, drawY, hasYLabel, labels) {
    var side = (panel.y && panel.y.tick_label_side) || 'left';
    var tickW = drawY ? tickWOf(labels || (panel.y && panel.y.tick_labels)) : 7;
    var pad;
    if (side === 'right') {
      pad = { l: 16, r: drawY ? Math.ceil(tickW + 6) : 14, t: 28, b: 32 };
    } else {
      pad = { l: groupPadL, r: 14, t: 28, b: 32 };
    }
    var g = {
      pad: pad,
      x0: pad.l,
      x1: pw - pad.r,
      y0: ph - pad.b,
      y1: pad.t,
      side: side,
      tickW: tickW,
      hasYLabel: hasYLabel
    };
    var xl = panel.x.lim, yl = panel.y.lim;
    g.xlim = xl; g.ylim = yl;
    g.xs = linScale(xl[0], xl[1], g.x0, g.x1);
    g.ys = linScale(yl[0], yl[1], g.y0, g.y1);
    g.unit = (g.x1 - g.x0) / (xl[1] - xl[0]);
    return g;
  }

  function clipFor(svg, g) {
    var id = uid('clip');
    var defs = sEl('defs', null, svg);
    var cp = sEl('clipPath', { id: id }, defs);
    sRect(cp, g.x0, g.y1, g.x1 - g.x0, g.y0 - g.y1, {});
    return 'url(#' + id + ')';
  }

  function drawFrame(svg, panel, g, theme, opts) {
    var xa = panel.x, ya = panel.y, i, t, px, py, lab;
    var hf = xa.hf_shading;

    if (hf) {
      var hx0 = g.xs(clamp(hf.from_x, g.xlim[0], g.xlim[1]));
      var hx1 = g.xs(clamp(hf.to_x, g.xlim[0], g.xlim[1]));
      sRect(svg, hx0, g.y1, hx1 - hx0, g.y0 - g.y1, {
        'class': 'shade', 'data-hf-from': hf.from_x, 'data-hf-to': hf.to_x
      });
      sLine(svg, hx0, g.y1, hx0, g.y0, 'hfedge');
    }

    for (i = 0; i < (ya.ticks || []).length; i++) {
      py = g.ys(ya.ticks[i]);
      sLine(svg, g.x0, py, g.x1, py, 'gr');
    }
    sLine(svg, g.x0, g.y0, g.x1, g.y0, 'ax');

    if (opts.drawY) {
      lab = opts.yTickLabels || ya.tick_labels || [];
      for (i = 0; i < (ya.ticks || []).length; i++) {
        py = g.ys(ya.ticks[i]);
        t = lab[i];
        if (t === undefined || t === null || t === '') continue;
        if (g.side === 'right') {
          sTxt(svg, g.x1 + 6, py + 3.2, t, 't-tick', { 'text-anchor': 'start' });
        } else {
          sTxt(svg, g.x0 - 6, py + 3.2, t, 't-tick', { 'text-anchor': 'end' });
        }
      }
    }

    for (i = 0; i < (xa.ticks || []).length; i++) {
      px = g.xs(xa.ticks[i]);
      sLine(svg, px, g.y0, px, g.y0 + 3, 'ax');
      t = (xa.tick_labels || [])[i];
      if (t === undefined || t === null || t === '') continue;
      sTxt(svg, px, g.y0 + 14, t, 't-tick', { 'text-anchor': 'middle' });
    }

    if (opts.drawY && g.hasYLabel && ya.label) {
      var lines = String(ya.label).split('\n');
      var block = lines.length * 10.4;
      var gx = g.side === 'right'
        ? g.x1 + g.tickW + block / 2 + 2
        : g.x0 - g.tickW - block / 2 - 2;
      var gy = (g.y0 + g.y1) / 2;
      var gg = sEl('g', { transform: 'translate(' + r2(gx) + ',' + r2(gy) + ') rotate(-90)' }, svg);
      var tt = sEl('text', { x: 0, y: 0, 'class': 't-ax', 'text-anchor': 'middle' }, gg);
      for (i = 0; i < lines.length; i++) {
        var ts = sEl('tspan', {
          x: 0, y: r2(((lines.length - 1) / 2 - i) * 10.4)
        }, tt);
        ts.appendChild(doc().createTextNode(lines[i]));
      }
    }
  }

  function drawHLine(parent, s, g, theme) {
    var py = g.ys(s.y);
    var grp = sEl('g', {
      'data-series': s.name, 'data-kind': 'hline', 'data-y': s.y
    }, parent);
    sLine(grp, g.x0, py, g.x1, py, null, {
      stroke: colorOf(s.color, theme),
      'stroke-width': lw(s.linewidth_pt || 1),
      'stroke-dasharray': s.style === 'dashed' ? '3.8 2.5' : null,
      fill: 'none'
    });
    return grp;
  }

  function drawLineSeries(parent, s, g, theme, xa, ctx) {
    var col = colorOf(s.color, theme);
    var d = s.draw || {};
    var med = s[d.line || 'q50'] || s.q50 || s.y || [];
    var lo = s[(d.ribbon || ['q25', 'q75'])[0]] || s.q25;
    var hi = s[(d.ribbon || ['q25', 'q75'])[1]] || s.q75;
    var xs = s.x || xa.band_index || [];
    var grp = sEl('g', {
      'data-series': s.name, 'data-kind': 'line', 'data-n': med.length
    }, parent);
    var i, path;

    if (lo && hi && lo.length === hi.length && lo.length === xs.length) {
      path = '';
      for (i = 0; i < xs.length; i++) {
        path += (i ? ' L' : 'M') + r2(g.xs(xs[i])) + ' ' + r2(g.ys(hi[i]));
      }
      for (i = xs.length - 1; i >= 0; i--) {
        path += ' L' + r2(g.xs(xs[i])) + ' ' + r2(g.ys(lo[i]));
      }
      path += ' Z';
      sEl('path', {
        d: path, fill: col, 'fill-opacity': d.ribbon_alpha === undefined ? 0.14 : d.ribbon_alpha,
        stroke: 'none', 'data-part': 'ribbon', 'clip-path': ctx.clip
      }, grp);
    }

    path = '';
    for (i = 0; i < xs.length; i++) {
      path += (i ? ' L' : 'M') + r2(g.xs(xs[i])) + ' ' + r2(g.ys(med[i]));
    }
    sEl('path', {
      d: path, fill: 'none', stroke: col, 'stroke-width': lw(d.linewidth_pt || 1.5),
      'stroke-linecap': 'round', 'stroke-linejoin': 'round',
      'data-part': 'median', 'clip-path': ctx.clip
    }, grp);

    var hollow = d.markers === 'hollow';
    for (i = 0; i < xs.length; i++) {
      sEl('circle', {
        cx: r2(g.xs(xs[i])), cy: r2(g.ys(med[i])), r: 2.9,
        fill: hollow ? 'var(--lqp-surface)' : col,
        stroke: hollow ? col : 'var(--lqp-halo)',
        'stroke-width': 0.9, 'data-part': 'marker', 'data-band': i
      }, grp);
      tipFor(sEl('circle', {
        cx: r2(g.xs(xs[i])), cy: r2(g.ys(med[i])), r: 9, 'class': 'hit'
      }, grp), s.name + '\n' + bandRange(xa, i) +
        '\nmedian  ' + gf(med[i], 5) +
        (lo && hi ? '\nIQR  ' + gf(lo[i], 4) + ' – ' + gf(hi[i], 4) : '') +
        (s.n_frames_finite ? '\n' + s.n_frames_finite[i] + ' frames' : ''));
    }
    return grp;
  }

  function drawBarSeries(parent, s, g, theme, xa, ctx) {
    var col = colorOf(s.color, theme);
    var xs = s.x || [], ys = s.y || [], bot = s.bottom || null;
    var w = (s.width || 0.36) * g.unit;
    var grp = sEl('g', {
      'data-series': s.name, 'data-kind': 'bar', 'data-n': ys.length,
      'data-stacked-on': s.stacked_on || null
    }, parent);
    var i, b, top, py0, py1, cx, tipTxt;

    for (i = 0; i < xs.length; i++) {
      if (!isNum(ys[i])) continue;
      b = bot ? bot[i] : 0;
      top = b + ys[i];
      py0 = g.ys(b);
      py1 = g.ys(top);
      cx = g.xs(xs[i]);
      sRect(grp, cx - w / 2, Math.min(py0, py1), w, Math.abs(py1 - py0), {
        fill: col, 'data-band': i, 'clip-path': ctx.clip
      });
      sRect(grp, cx - w / 2, Math.min(py0, py1), w, Math.abs(py1 - py0), {
        'class': 'edge', 'clip-path': ctx.clip
      });
      tipTxt = s.name + '\n' + bandRange(xa, i) + '\n' +
        (ctx.percent ? gf(ys[i], 4) + ' %' : gf(ys[i], 5));
      if (bot) tipTxt += '\nstacked on ' + gf(b, 4) + ', top ' + gf(top, 4);
      if (s.n_frames_finite) tipTxt += '\n' + s.n_frames_finite[i] + ' frames';
      /* the hit column is exactly the bar's own column: paired bars sit 0.02 units
         apart, so any padding here would steal the neighbour's hover */
      tipFor(sRect(grp, cx - w / 2, g.y1, w, g.y0 - g.y1, { 'class': 'hit' }), tipTxt);
    }
    return grp;
  }

  function drawSegmentSeries(parent, s, g, theme, xa, ctx) {
    var col = colorOf(s.color, theme);
    var ys = s.y || [], x0 = s.x0 || [], x1 = s.x1 || [];
    var grp = sEl('g', {
      'data-series': s.name, 'data-kind': 'segments', 'data-n': ys.length
    }, parent);
    var i, py;
    for (i = 0; i < ys.length; i++) {
      if (!isNum(ys[i])) continue;
      py = g.ys(ys[i]);
      sLine(grp, g.xs(x0[i]), py, g.xs(x1[i]), py, null, {
        stroke: col, 'stroke-width': lw(s.linewidth_pt || 1.1),
        'stroke-linecap': 'butt', fill: 'none',
        'data-band': i, 'clip-path': ctx.clip
      });
      tipFor(sRect(grp, g.xs(x0[i]), py - 6, g.xs(x1[i]) - g.xs(x0[i]), 12, { 'class': 'hit' }),
        s.name + '\n' + bandRange(xa, i) + '\n' + gf(ys[i], 5));
    }
    return grp;
  }

  function zOrder(series) {
    var dec = [], i;
    for (i = 0; i < series.length; i++) {
      dec.push({ s: series[i], z: series[i].zorder === undefined ? 3 : series[i].zorder, i: i });
    }
    dec.sort(function (a, b) { return (a.z - b.z) || (a.i - b.i); });
    var out = [];
    for (i = 0; i < dec.length; i++) out.push(dec[i].s);
    return out;
  }

  function drawAnnotations(svg, panel, g, theme) {
    var list = panel.annotations || [], i, a, px, py, lines, fs, anchor;
    for (i = 0; i < list.length; i++) {
      a = list[i];
      if (a.coords === 'axes_fraction') {
        px = g.x0 + a.x * (g.x1 - g.x0);
        py = g.y0 + a.y * (g.y1 - g.y0);
      } else {
        px = g.xs(a.x);
        py = g.ys(a.y);
      }
      if (a.offset_pt) {
        px += a.offset_pt[0] * 1.25;
        py -= a.offset_pt[1] * 1.25;
      }
      lines = String(a.text).split('\n');
      fs = a.text === '†' ? 't-flag' : 't-anno';
      anchor = a.ha === 'right' ? 'end' : (a.ha === 'center' ? 'middle' : 'start');
      if (a.va === 'top') py += 8;
      else if (a.va === 'bottom') py -= (lines.length - 1) * 10 + 1;
      else if (a.va === 'center') py += 3.2 - (lines.length - 1) * 5;
      sLines(svg, px, py, lines, fs, {
        'text-anchor': anchor,
        fill: colorOf(a.color, theme),
        'font-weight': a.bold ? 700 : null,
        'data-anno': a.text
      }, 10);
    }
  }

  function drawPanelLegend(svg, legend, g, theme) {
    if (!legend || !legend.entries || !legend.entries.length) return;
    var ent = legend.entries, i, e, wide = 0;
    for (i = 0; i < ent.length; i++) {
      wide = Math.max(wide, String(ent[i].label).length);
    }
    /* JetBrains Mono at 8.6px advances ~5.2px per glyph */
    var textW = wide * 5.2;
    var y = g.y1 + 6, x = g.x1 - 4;
    var sx = x - textW - 14;
    if (sx < g.x0 + 4) sx = g.x0 + 4;
    var grp = sEl('g', { 'data-part': 'panel-legend' }, svg);
    for (i = 0; i < ent.length; i++) {
      e = ent[i];
      sTxt(grp, x, y + i * 11 + 4, e.label, 't-key', { 'text-anchor': 'end' });
      sRect(grp, sx, y + i * 11 - 3.5, 9, 9, { fill: colorOf(e.color, theme) });
      sRect(grp, sx, y + i * 11 - 3.5, 9, 9, { 'class': 'edge' });
    }
  }

  function drawPanel(host, panel, cfg) {
    var pw = cfg.pw, ph = cfg.ph, theme = cfg.theme;
    var g = geomOf(panel, pw, ph, cfg.padL, cfg.drawY, cfg.hasYLabel,
      cfg.yTickLabels || (panel.y && panel.y.tick_labels));
    var svg = svgRoot(host, pw, ph, cfg.title, cfg.desc, {
      'data-panel': panel.id,
      'data-figure': cfg.figRef,
      'data-plot': [r2(g.x0), r2(g.x1), r2(g.y0), r2(g.y1)].join(','),
      'data-xlim': g.xlim.join(','),
      'data-ylim': g.ylim.join(',')
    });
    var ctx = { clip: clipFor(svg, g), percent: (panel.y && panel.y.unit) === 'percent' };

    drawFrame(svg, panel, g, theme, {
      drawY: cfg.drawY, yTickLabels: cfg.yTickLabels
    });

    sTxt(svg, g.x0, 14, panel.title, 't-pan');
    if (cfg.sub) sTxt(svg, g.x0, 24, cfg.sub, 't-sub');

    var ser = zOrder(panel.series || []), i, s;
    for (i = 0; i < ser.length; i++) {
      s = ser[i];
      if (s.kind === 'hline') drawHLine(svg, s, g, theme);
      else if (s.kind === 'line') drawLineSeries(svg, s, g, theme, panel.x, ctx);
      else if (s.kind === 'bar') drawBarSeries(svg, s, g, theme, panel.x, ctx);
      else if (s.kind === 'segments') drawSegmentSeries(svg, s, g, theme, panel.x, ctx);
    }

    drawAnnotations(svg, panel, g, theme);
    drawPanelLegend(svg, panel.legend, g, theme);
    return svg;
  }

  /* --------------------------------------------------------------- tables */

  function tableFor(host, caption, panels) {
    var xa = panels[0].x, lab = bandLabels(xa);
    var head = ['Panel · series'], rows = [], i, j, p, s, row, med, lo, hi, pct;
    for (i = 0; i < lab.length; i++) head.push(lab[i]);
    for (i = 0; i < panels.length; i++) {
      p = panels[i];
      pct = (p.y && p.y.unit) === 'percent';
      for (j = 0; j < (p.series || []).length; j++) {
        s = p.series[j];
        if (s.kind === 'hline') continue;
        if (s.kind === 'line') {
          med = s[(s.draw && s.draw.line) || 'q50'] || [];
          lo = s.q25; hi = s.q75;
          row = [p.title + ' — ' + s.name + ' (median)'];
          for (var m = 0; m < med.length; m++) row.push(gf(med[m], 4));
          rows.push(row);
          if (lo && hi) {
            row = [p.title + ' — ' + s.name + ' (IQR)'];
            for (m = 0; m < lo.length; m++) {
              row.push(gf(lo[m], 4) + '–' + gf(hi[m], 4));
            }
            rows.push(row);
          }
        } else {
          row = [p.title + ' — ' + s.name +
            (s.stacked_on ? ' (segment, stacked on ' + s.stacked_on + ')' : '')];
          for (m = 0; m < (s.y || []).length; m++) {
            row.push(pct ? gf(s.y[m], 4) + '%' : gf(s.y[m], 4));
          }
          rows.push(row);
        }
      }
    }
    buildTable(host, caption, head, rows);
  }

  /* ---------------------------------------------------------------- notes */

  var EPOCH_NOTE = 'The model labelled LiSeer — before distillation — is the ' +
    'epoch-21 checkpoint named in the released measurement file, while the manuscript’s ' +
    'scalar readings elsewhere are read from epoch 24.';

  function framesLine(p) {
    var f = (p && p.frames) || {};
    return (f.count || '?') + ' frames of ' + (p.dataset || 'SemanticKITTI') +
      ' sequence ' + (p.sequence || '08') + ' (' + (f.selection || 'see data file') + ').';
  }

  var CKA_LINE = 'Each curve is the linear CKA between the band-filtered model field and the ' +
    'band-filtered Dense-teacher field, divided per frame and band by the teacher’s own ' +
    're-truncation noise ceiling, so 1.00 is that ceiling; the line is the per-band median ' +
    'across frames (frames are the unit) and the ribbon the interquartile range.';

  var GAP_LINE = 'The right-hand panel is 100 × the median of (CKA_LiEQ − CKA_LiSeer) ' +
    '/ (CKA_ceiling − CKA_LiSeer) per band and region.';

  var HF_LINE = 'The shaded strip is the high-frequency region, λ ≥ 0.60 ' +
    '(bands 7–10).';

  var ROW_A_LINE = 'The field is the TPV feature plane xy at neck scale s0 — a 240 ' +
    '× 180 cylindrical grid of 128 channels, taken after backbone and neck and before ' +
    'FiLM, the plane the low-frequency distiller acts on — filtered by the 4-neighbour ' +
    'normalized grid Laplacian with square-root raised-cosine tight-frame windows, the same ' +
    'operator as that distiller.';

  var ROW_B_LINE = 'The field is the 8-class voxel logits on the teacher support Ω_T of ' +
    'the [8, 240, 180, 16] cylindrical grid, filtered by the 3-D support-subgraph Laplacian ' +
    'L_Ω — the operator used by the high-frequency distiller — with 11 ' +
    'band-pass filters by Jackson–Chebyshev approximation at order K = 1024.';

  var OMEGA_LINE = 'Statistics use the same 11-band partition, remove the per-channel mean ' +
    'and divide by the RMS inside Ω_T, and are restricted to Ω_T.';

  /* note for one row of Fig. 3 on its own */
  function note3one(d, key) {
    return [
      framesLine(d.protocol),
      key === 'fig3b' ? ROW_B_LINE : ROW_A_LINE,
      OMEGA_LINE, CKA_LINE, GAP_LINE, HF_LINE, EPOCH_NOTE
    ].join(' ');
  }

  function note3(a, b) {
    var pa = a.protocol;
    return [
      framesLine(pa),
      'Row a measures the TPV feature plane xy at neck scale s0 — a 240 × 180 ' +
      'cylindrical grid of 128 channels, taken after backbone and neck and before FiLM, the ' +
      'plane the low-frequency distiller acts on — filtered by the 4-neighbour normalized ' +
      'grid Laplacian with square-root raised-cosine tight-frame windows, the same operator as ' +
      'that distiller.',
      'Row b measures the 8-class voxel logits on the teacher support Ω_T of the ' +
      '[8, 240, 180, 16] cylindrical grid, filtered by the 3-D support-subgraph Laplacian ' +
      'L_Ω — the operator used by the high-frequency distiller — with 11 ' +
      'band-pass filters by Jackson–Chebyshev approximation at order K = 1024.',
      'Both rows use the same 11-band partition, remove the per-channel mean and divide by the ' +
      'RMS inside Ω_T, and restrict every statistic to Ω_T.',
      CKA_LINE, GAP_LINE, HF_LINE, EPOCH_NOTE
    ].join(' ');
  }

  function note4(d) {
    var p = d.protocol;
    return [
      framesLine(p),
      'The left column measures the TPV feature plane xy at neck scale s0 with the 4-neighbour ' +
      'normalized grid Laplacian, the low-frequency distiller’s operator; the right column ' +
      'measures the 8-class voxel logits on the teacher support Ω_T with the 3-D ' +
      'support-subgraph Laplacian L_Ω, the high-frequency distiller’s operator. Both ' +
      'use the same 11 bands, with the per-channel mean removed and the RMS over Ω_T ' +
      'divided out.',
      'Panels a and b plot log₂ of the band energy share relative to the Dense teacher, so ' +
      '0 is ×1: the line is the per-band median across frames and the ribbon the ' +
      'interquartile range.',
      'In panels c–f, real energy is the part of a student’s band energy that a ' +
      'cross-fitted ridge regression reproduces from the teacher’s own band-k field ' +
      '(outer 2-fold split by alternating 60° φ sectors, inner leave-one-sector-out ' +
      'choice of the ridge parameter) and pseudo energy is the remainder; both are divided by ' +
      'the teacher’s own band energy in that band, measured with the teacher’s input ' +
      're-truncated under a different 500,000-point seed. The panel axis labels abbreviate that ' +
      'denominator as “÷ Dense teacher”.',
      'Real energy is a lower bound and pseudo an upper bound, because only same-band linear ' +
      'relations count.',
      'The grey segments in c and d are the same quantity for the re-truncated teacher: the part ' +
      'the teacher itself still accounts for.',
      'In e and f the dark and light bars are separate medians of the per-frame LiEqualizer ' +
      '− LiSeer difference, not a stack, so their sum is not the median of the total change.',
      '† marks bands where the cross-scene null median R² exceeds 0.05, where real ' +
      'energy may be slightly overestimated.',
      EPOCH_NOTE
    ].join(' ');
  }

  /* ============================================================== Fig. 3 == */

  /* the published figure's own row sub-captions, word for word */
  function rowLabelFor(key, d) {
    if (key === 'fig3a') return '(a) on the feature planes';
    if (key === 'fig3b') return '(b) on the voxel logits';
    return d.paper_ref || '';
  }

  /* row b carries a tag of its own in the published figure ("voxel logits - 199
     frames"); for row a, which has none, state the same facts in the same shape */
  function rowTagFor(d) {
    var t = d.figure_texts || [], i, tag = null;
    for (i = 0; i < t.length; i++) {
      if (t[i].fontsize_pt && t[i].fontsize_pt <= 6.5 && t[i].ha === 'right') {
        tag = t[i].text;
        break;
      }
    }
    if (!tag) {
      tag = ((d.protocol && d.protocol.frames && d.protocol.frames.count) || '?') + ' frames';
    }
    return tag + ' · ' + (d.paper_ref || '');
  }

  function drawFig3Row(body, d, key, W, theme, opts) {
    var panels = d.panels || [];
    var cols = W >= 880 ? 3 : (W >= 620 ? 2 : 1);
    var pw = panelWidth(W, cols, 14, 232, 440);
    var ph = 236;
    var row = hEl('div', 'lqp-row', body);
    var head = hEl('div', 'lqp-rowhead', row);
    hEl('span', null, head, rowLabelFor(key, d));
    hEl('i', null, head, rowTagFor(d));
    var grid = hEl('div', 'lqp-panels', row);
    setStyle(grid, '--lqp-cols:' + cols + ';--lqp-pmax:' + pw + 'px');

    var leader = panels[0];
    var hasYLabel = pw >= 300;
    var i, p, shared, drawY, labs;

    /* one gutter for the whole row, wide enough for the widest of its panels, so the
       plot boxes line up and the rotated label never lands on the tick labels */
    var padL = 0;
    for (i = 0; i < panels.length; i++) {
      p = panels[i];
      if (((p.y && p.y.tick_label_side) || 'left') === 'right') continue;
      shared = !!(p.y && p.y.shared_with_panel);
      drawY = shared ? (cols === 1) : true;
      labs = shared && leader ? leader.y.tick_labels : p.y.tick_labels;
      padL = Math.max(padL, drawY ? padLFor(p, labs, hasYLabel && !shared) : 16);
    }

    for (i = 0; i < panels.length; i++) {
      p = panels[i];
      shared = !!(p.y && p.y.shared_with_panel);
      /* Only the leftmost panel of a shared group prints its y ticks - unless the grid
         has collapsed to one column, where a panel with no y labels is unreadable. */
      drawY = shared ? (cols === 1) : ((p.y.tick_label_side || 'left') !== 'none');
      drawPanel(grid, p, {
        pw: pw, ph: ph, theme: theme, padL: padL,
        drawY: drawY,
        hasYLabel: hasYLabel && !shared,
        yTickLabels: shared && leader ? leader.y.tick_labels : null,
        figRef: d.paper_ref,
        sub: cols === 1 ? rowLabelFor(key, d) : null,
        title: (d.paper_ref || 'Fig. 3') + ' ' + p.title.replace(/\s*↑\s*$/, '') +
          ' — ' + rowLabelFor(key, d),
        desc: d.description
      });
    }
    hEl('p', 'lqp-xlab', row, (panels[0] && panels[0].x.label) || '');
    return row;
  }

  function drawFig3(mount, W, theme, sets, opts) {
    var a = sets.fig3a, b = sets.fig3b;
    var ctx = scaffold(mount, {
      key: 'paper-fig3', theme: theme,
      axes: 'x: ' + (a.units && a.units.x) + ' · y: ' + (a.units && a.units.y_left_middle) +
        ' (left, middle), ' + (a.units && a.units.y_right) + ' (right)',
      note: note3(a, b)
    });
    legendEntries(ctx.legend, (a.figure_legends && a.figure_legends[0] &&
      a.figure_legends[0].entries) || [], theme);

    drawFig3Row(ctx.body, a, 'fig3a', W, theme, opts);
    drawFig3Row(ctx.body, b, 'fig3b', W, theme, opts);

    tableFor(ctx.table, 'Fig. 3a, TPV feature plane xy: per-band CKA to the Dense teacher ' +
      'divided by the noise ceiling, and the share of the LiSeer-to-ceiling gap closed by ' +
      'LiEqualizer. Columns are the 11 graph-frequency bands by lower λ edge; 4 ' +
      'significant digits.', a.panels);
    tableFor(ctx.table, 'Fig. 3b, voxel logits: the same quantities on the 8-class voxel ' +
      'logits over the teacher support.', b.panels);
    return ctx.fig;
  }

  function drawFig3Single(key) {
    return function (mount, W, theme, sets, opts) {
      var d = sets[key];
      var ctx = scaffold(mount, {
        key: 'paper-' + key, theme: theme,
        axes: 'x: ' + (d.units && d.units.x) + ' · y: ' +
          (d.units && d.units.y_left_middle) + ' (left, middle), ' +
          (d.units && d.units.y_right) + ' (right)',
        note: note3one(d, key)
      });
      legendEntries(ctx.legend, (d.figure_legends && d.figure_legends[0] &&
        d.figure_legends[0].entries) || [], theme);
      drawFig3Row(ctx.body, d, key, W, theme, opts);
      tableFor(ctx.table, (d.paper_ref || 'Fig. 3') + ': ' + d.description +
        ' Columns are the 11 graph-frequency bands by lower λ edge; 4 significant digits.',
        d.panels);
      return ctx.fig;
    };
  }

  /* ============================================================== Fig. 4 == */

  var FIG4_ROWS = [['a', 'b'], ['c', 'd'], ['e', 'f']];

  function panelById(d, id) {
    var i, ps = d.panels || [];
    for (i = 0; i < ps.length; i++) if (ps[i].id === id) return ps[i];
    return null;
  }

  function derivedLegend(panels) {
    var out = [], seen = {}, i, j, s;
    for (i = 0; i < panels.length; i++) {
      for (j = 0; j < (panels[i].series || []).length; j++) {
        s = panels[i].series[j];
        if (s.kind !== 'bar' || seen[s.name]) continue;
        seen[s.name] = 1;
        out.push({ label: s.name, kind: 'patch', color: s.color });
      }
    }
    return out;
  }

  function drawFig4(mount, W, theme, sets, opts) {
    var d = sets.fig4;
    var ctx = scaffold(mount, {
      key: 'paper-fig4', theme: theme,
      axes: 'x: ' + (d.units && d.units.x) + ' · y: a, b — ' +
        (d.units && d.units.y_a_b) + ' · c, d — ' + (d.units && d.units.y_c_d) +
        ' · e, f — ' + (d.units && d.units.y_e_f),
      note: note4(d)
    });

    var cols = W >= 720 ? 2 : 1;
    var pw = panelWidth(W, cols, 14, 268, 570);
    var ph = 252;
    var hasYLabel = pw >= 300;
    var legends = d.figure_legends || [];
    var ri, ci, row, head, grid, ids, p, i;

    /* one gutter for all six panels, so the three rows line up column by column */
    var padL = 0;
    for (i = 0; i < d.panels.length; i++) {
      padL = Math.max(padL, padLFor(d.panels[i], d.panels[i].y.tick_labels, hasYLabel));
    }

    /* column headers, as in the published figure, only while the columns exist */
    if (cols === 2) {
      var ch = hEl('div', 'lqp-cols', ctx.body);
      hEl('span', null, ch, (panelById(d, 'a') || {}).column || '');
      hEl('span', null, ch, (panelById(d, 'b') || {}).column || '');
    }

    for (ri = 0; ri < FIG4_ROWS.length; ri++) {
      ids = FIG4_ROWS[ri];
      row = hEl('div', 'lqp-row', ctx.body);
      head = hEl('div', 'lqp-legend is-row', row);
      if (ri === 0) {
        legendEntries(head, (legends[0] && legends[0].entries) || [], theme);
      } else if (ri === 1) {
        legendEntries(head, (legends[1] && legends[1].entries) || [], theme);
      } else {
        legendEntries(head, derivedLegend([panelById(d, 'e'), panelById(d, 'f')]), theme);
      }
      grid = hEl('div', 'lqp-panels', row);
      setStyle(grid, '--lqp-cols:' + cols + ';--lqp-pmax:' + pw + 'px');
      for (ci = 0; ci < ids.length; ci++) {
        p = panelById(d, ids[ci]);
        if (!p) continue;
        drawPanel(grid, p, {
          pw: pw, ph: ph, theme: theme, padL: padL,
          drawY: true, hasYLabel: hasYLabel,
          figRef: d.paper_ref,
          sub: cols === 1 ? p.column : null,
          title: (d.paper_ref || 'Fig. 4') + ' ' + p.title + ' — ' + p.column,
          desc: d.description
        });
      }
      hEl('p', 'lqp-xlab', row, (p && p.x.label) || '');
    }

    var texts = d.figure_texts || [];
    for (i = 0; i < texts.length; i++) {
      if (texts[i].fontsize_pt && texts[i].fontsize_pt <= 6.5) {
        hEl('p', 'lqp-pub', ctx.body, 'As published: ' + texts[i].text);
      }
    }

    tableFor(ctx.table, 'Fig. 4: band-wise energy relative to the Dense teacher. a, b are ' +
      'log₂ of the energy-share ratio (0 = ×1); c, d and e, f are divided by the ' +
      'teacher’s own band energy under a re-truncated input. Light bars in c and d are ' +
      'segment heights stacked on the matching dark bar. Columns are the 11 graph-frequency ' +
      'bands by lower λ edge; 4 significant digits.', d.panels);
    return ctx.fig;
  }

  /* --------------------------------------------------------------- public */

  function renderFig3(el, opts) {
    return run(el, 'paper-fig3', ['fig3a', 'fig3b'], opts, drawFig3);
  }

  function renderFig3a(el, opts) {
    return run(el, 'paper-fig3a', ['fig3a'], opts, drawFig3Single('fig3a'));
  }

  function renderFig3b(el, opts) {
    return run(el, 'paper-fig3b', ['fig3b'], opts, drawFig3Single('fig3b'));
  }

  function renderFig4(el, opts) {
    return run(el, 'paper-fig4', ['fig4'], opts, drawFig4);
  }

  var KEYS = {
    'paper-fig3': renderFig3,
    'paper-fig3a': renderFig3a,
    'paper-fig3b': renderFig3b,
    'paper-fig4': renderFig4
  };

  function autoInit(opts) {
    var D = doc(), out = {}, nodes, i, n, k;
    if (!D || typeof D.querySelectorAll !== 'function') return out;
    try {
      nodes = D.querySelectorAll('[data-lieq-paper-chart],[data-lieq-chart]');
    } catch (e) {
      warn('autoInit', e);
      return out;
    }
    for (i = 0; i < nodes.length; i++) {
      n = nodes[i];
      k = (n.getAttribute('data-lieq-paper-chart') || n.getAttribute('data-lieq-chart') || '');
      if (!KEYS[k]) continue;
      out[k] = KEYS[k](n, opts);
    }
    return out;
  }

  function renderAll(map, opts) {
    if (!map) return autoInit(opts);
    var out = {}, k;
    for (k in map) {
      if (map.hasOwnProperty(k) && KEYS[k]) out[k] = KEYS[k](map[k], opts);
    }
    return out;
  }

  var API = {
    version: VERSION,
    keys: ['paper-fig3', 'paper-fig3a', 'paper-fig3b', 'paper-fig4'],
    files: FILES,
    dataBase: null,
    renderFig3: renderFig3,
    renderFig3a: renderFig3a,
    renderFig3b: renderFig3b,
    renderFig4: renderFig4,
    renderAll: renderAll,
    autoInit: autoInit,
    load: dataset,
    note3: note3,
    note3one: note3one,
    note4: note4,
    _internals: {
      gf: gf, lw: lw, linScale: linScale, geomOf: geomOf, zOrder: zOrder,
      colorOf: colorOf, injectCSS: injectCSS, bandRange: bandRange, loadJSON: loadJSON,
      base: base, CSS: CSS
    }
  };

  BASE = scriptBase();
  if (BASE) API.dataBase = BASE;

  /* Own, unambiguous global. The two sibling chart modules on this page also publish
     window.LiEQCharts, so we never overwrite it: we fill in only keys nobody has
     defined yet, and whichever script loads last leaves all three reachable. */
  global.LiEQPaperCharts = API;
  var NS = global.LiEQCharts;
  if (!NS || typeof NS !== 'object') {
    NS = {};
    global.LiEQCharts = NS;
  }
  for (var ak in API) {
    if (API.hasOwnProperty(ak) && !Object.prototype.hasOwnProperty.call(NS, ak)) {
      NS[ak] = API[ak];
    }
  }
  if (typeof module === 'object' && module && module.exports) module.exports = API;

  var D0 = doc();
  if (D0 && D0.addEventListener) {
    if (D0.readyState === 'loading') {
      D0.addEventListener('DOMContentLoaded', function () { autoInit(); });
    } else {
      autoInit();
    }
  }
}(typeof window !== 'undefined' ? window : (typeof globalThis !== 'undefined' ? globalThis : this)));
