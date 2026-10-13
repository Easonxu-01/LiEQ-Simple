/*
 * charts-paper.test.js - assertions for assets/charts-paper.js.
 *
 * Runs in plain node with no DOM, no jsdom, no test framework, no network:
 *     node docs/assets/charts-paper.test.js
 *
 * It renders Fig. 3 and Fig. 4 into a hand-rolled DOM, with fetch reading the three
 * JSON files off disk, and then checks what the markup actually says: the panel and
 * series structure, and the pixel position of EVERY drawn value, recomputed from each
 * panel's own axis range. It also covers the wording the figure notes must carry, the
 * accessibility contract, and the failure paths.
 *
 * Re-run it after re-exporting assets/data/*.json: it reads the JSON, so it will tell
 * you whether the figure still draws what the data says.
 *
 * A passing run prints two [LiEQPaperCharts] warnings, one 404 and one "offline": they
 * are the module reporting the two failures the RESILIENCE section provokes on purpose.
 * The run ends in "552 / 552 checks passed" with exit code 0.
 */
'use strict';

/* ------------------------------------------------------------------ */
/* A DOM small enough to read and big enough to render charts into.     */
/* ------------------------------------------------------------------ */


function Node(tag, ns) {
  this.tagName = tag;
  this.namespaceURI = ns || null;
  this.attributes = {};
  this.childNodes = [];
  this.parentNode = null;
  this.nodeType = 1;
}
Node.prototype.setAttribute = function (k, v) { this.attributes[String(k)] = String(v); };
Node.prototype.getAttribute = function (k) {
  return Object.prototype.hasOwnProperty.call(this.attributes, k) ? this.attributes[k] : null;
};
Node.prototype.hasAttribute = function (k) {
  return Object.prototype.hasOwnProperty.call(this.attributes, k);
};
Node.prototype.removeAttribute = function (k) { delete this.attributes[k]; };
Node.prototype.appendChild = function (n) {
  if (n.parentNode) n.parentNode.removeChild(n);
  n.parentNode = this;
  this.childNodes.push(n);
  return n;
};
Node.prototype.removeChild = function (n) {
  var i = this.childNodes.indexOf(n);
  if (i >= 0) this.childNodes.splice(i, 1);
  n.parentNode = null;
  return n;
};
Node.prototype.addEventListener = function (type, fn) {
  (this.__h || (this.__h = {}))[type] = (this.__h[type] || []).concat([fn]);
};
Node.prototype.removeEventListener = function () {};
Node.prototype.dispatch = function (type, ev) {
  var hs = (this.__h || {})[type] || [], i;
  for (i = 0; i < hs.length; i++) hs[i].call(this, ev || {});
};
Node.prototype.getBoundingClientRect = function () {
  return { left: 0, top: 0, width: this.__w || 0, height: 0 };
};
Object.defineProperty(Node.prototype, 'firstChild', {
  get: function () { return this.childNodes.length ? this.childNodes[0] : null; }
});
Object.defineProperty(Node.prototype, 'textContent', {
  get: function () {
    var out = '', i, c;
    for (i = 0; i < this.childNodes.length; i++) {
      c = this.childNodes[i];
      out += c.nodeType === 3 ? c.data : c.textContent;
    }
    return out;
  }
});

function Text(data) {
  this.nodeType = 3;
  this.data = String(data);
  this.parentNode = null;
  this.childNodes = [];
}
Text.prototype.textContent = '';

/* --- query support: only the attribute selectors the module actually uses --- */
function walk(node, out) {
  var i, c;
  for (i = 0; i < node.childNodes.length; i++) {
    c = node.childNodes[i];
    if (c.nodeType !== 1) continue;
    out.push(c);
    walk(c, out);
  }
  return out;
}

function matches(el, sel) {
  sel = sel.trim();
  var m = /^\[([a-zA-Z0-9_-]+)(?:=["']?([^"'\]]*)["']?)?\]$/.exec(sel);
  if (m) return m[2] === undefined ? el.hasAttribute(m[1]) : el.getAttribute(m[1]) === m[2];
  var t = /^([a-zA-Z0-9]+)$/.exec(sel);
  if (t) return String(el.tagName).toLowerCase() === t[1].toLowerCase();
  var c = /^\.([a-zA-Z0-9_-]+)$/.exec(sel);
  if (c) return (' ' + (el.getAttribute('class') || '') + ' ').indexOf(' ' + c[1] + ' ') >= 0;
  return false;
}

function makeDocument() {
  var D = {
    readyState: 'complete',
    createElement: function (tag) { return new Node(tag, 'http://www.w3.org/1999/xhtml'); },
    createElementNS: function (ns, tag) { return new Node(tag, ns); },
    createTextNode: function (d) { return new Text(d); },
    getElementById: function (id) {
      var all = walk(D.documentElement, []), i;
      for (i = 0; i < all.length; i++) if (all[i].getAttribute('id') === id) return all[i];
      return null;
    },
    querySelectorAll: function (sel) {
      var parts = String(sel).split(','), all = walk(D.documentElement, []), out = [], i, j;
      for (i = 0; i < all.length; i++) {
        for (j = 0; j < parts.length; j++) {
          if (matches(all[i], parts[j])) { out.push(all[i]); break; }
        }
      }
      return out;
    },
    addEventListener: function () {},
    currentScript: null
  };
  D.documentElement = new Node('html');
  D.head = new Node('head');
  D.body = new Node('body');
  D.documentElement.appendChild(D.head);
  D.documentElement.appendChild(D.body);
  return D;
}

/* --- serialization, for the assertions that read the rendered markup --- */
var VOID = { br: 1, img: 1, input: 1, meta: 1, link: 1 };

function esc(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function serialize(node) {
  if (node.nodeType === 3) return esc(node.data);
  var tag = node.tagName, out = '<' + tag, k, i;
  for (k in node.attributes) {
    if (node.attributes.hasOwnProperty(k)) {
      out += ' ' + k + '="' + esc(node.attributes[k]).replace(/"/g, '&quot;') + '"';
    }
  }
  out += '>';
  if (VOID[tag]) return out;
  for (i = 0; i < node.childNodes.length; i++) out += serialize(node.childNodes[i]);
  return out + '</' + tag + '>';
}

/* --- the global the module closes over --- */
function makeWindow(opts) {
  opts = opts || {};
  var W = {
    document: makeDocument(),
    console: console,
    Promise: Promise,
    setTimeout: setTimeout,
    clearTimeout: clearTimeout,
    addEventListener: function () {},
    removeEventListener: function () {},
    matchMedia: function (q) {
      return { matches: !!opts.reducedMotion && /reduce/.test(q), media: q };
    }
  };
  if (opts.fetch !== false) W.fetch = opts.fetch;
  if (opts.rAF) W.requestAnimationFrame = function (fn) { fn(); };
  return W;
}

/* ------------------------------------------------------------------ */
/* The assertions.                                                     */
/* ------------------------------------------------------------------ */

var fs = require('fs');
var path = require('path');

var ASSETS = __dirname;
var DATA = path.join(ASSETS, 'data');
var MODULE = path.join(ASSETS, 'charts-paper.js');

var pass = 0, fail = 0, checks = 0;
var failures = [];

function ok(name, cond, detail) {
  checks++;
  if (cond) { pass++; return true; }
  fail++;
  failures.push(name + (detail ? '   (' + detail + ')' : ''));
  console.log('  FAIL ' + name + (detail ? '   (' + detail + ')' : ''));
  return false;
}
function head(s) { console.log('\n' + s); }
function line(s) { console.log('       ' + s); }

var TOL = 0.021; /* the module rounds every coordinate with Math.round(v*100)/100 */

function near(a, b, tol) {
  return Math.abs(a - b) <= (tol === undefined ? TOL : tol);
}

/* ------------------------------------------------------------------ set-up */

function makeFetch(log) {
  return function (url) {
    var file = String(url).replace(/^assets\/data\//, '');
    var p = path.join(DATA, file);
    if (log) log.push(url);
    return new Promise(function (resolve) {
      fs.readFile(p, 'utf8', function (err, txt) {
        if (err) { resolve({ ok: false, status: 404 }); return; }
        resolve({ ok: true, status: 200, json: function () { return Promise.resolve(JSON.parse(txt)); } });
      });
    });
  };
}

function freshModule(winOpts) {
  delete require.cache[MODULE];
  var W = makeWindow(winOpts);
  global.window = W;
  var API = require(MODULE);
  return { W: W, API: API };
}

function mountIn(W) {
  var m = W.document.createElement('div');
  W.document.body.appendChild(m);
  return m;
}

function json(name) {
  return JSON.parse(fs.readFileSync(path.join(DATA, name), 'utf8'));
}

var J3A = json('fig3a_band_cka_feature_planes.json');
var J3B = json('fig3b_band_cka_voxel_logits.json');
var J4 = json('fig4_energy_real.json');

/* --------------------------------------------------------------- DOM query */

function all(node) { return walk(node, []); }

function byTag(node, tag) {
  return all(node).filter(function (n) { return String(n.tagName).toLowerCase() === tag; });
}

function panelsOf(fig) {
  return byTag(fig, 'svg').filter(function (n) { return n.hasAttribute('data-panel'); });
}

function groupsOf(svg) {
  return byTag(svg, 'g').filter(function (n) { return n.hasAttribute('data-series'); });
}

function groupNamed(svg, name) {
  var gs = groupsOf(svg), i;
  for (i = 0; i < gs.length; i++) if (gs[i].getAttribute('data-series') === name) return gs[i];
  return null;
}

function kids(g, tag, attrName) {
  return byTag(g, tag).filter(function (n) {
    return attrName ? n.hasAttribute(attrName) : true;
  });
}

function textsWithClass(svg, cls) {
  return byTag(svg, 'text').filter(function (n) { return n.getAttribute('class') === cls; })
    .map(function (n) { return n.textContent; });
}

function scalesOf(svg) {
  var p = svg.getAttribute('data-plot').split(',').map(Number);
  var xl = svg.getAttribute('data-xlim').split(',').map(Number);
  var yl = svg.getAttribute('data-ylim').split(',').map(Number);
  return {
    x0: p[0], x1: p[1], y0: p[2], y1: p[3], xl: xl, yl: yl,
    xs: function (v) { return p[0] + (v - xl[0]) * (p[1] - p[0]) / (xl[1] - xl[0]); },
    ys: function (v) { return p[2] + (v - yl[0]) * (p[3] - p[2]) / (yl[1] - yl[0]); }
  };
}

function pathPoints(d) {
  var out = [], re = /[ML]\s*(-?[\d.]+)\s+(-?[\d.]+)/g, m;
  while ((m = re.exec(d)) !== null) out.push([parseFloat(m[1]), parseFloat(m[2])]);
  return out;
}

/* ------------------------------------------------------- per-panel checker */

function checkPanel(svg, panel, label) {
  var s = scalesOf(svg);
  var xa = panel.x;
  var gs = groupsOf(svg);
  var expect = panel.series.length;

  ok(label + ': ' + expect + ' series groups rendered', gs.length === expect,
    'got ' + gs.length + ' [' + gs.map(function (g) {
      return g.getAttribute('data-kind');
    }).join(',') + ']');

  /* the axis range the panel advertises must be the JSON's own */
  ok(label + ': x range from JSON', near(s.xl[0], xa.lim[0], 1e-9) && near(s.xl[1], xa.lim[1], 1e-9),
    s.xl.join(','));
  ok(label + ': y range from JSON', near(s.yl[0], panel.y.lim[0], 1e-9) &&
    near(s.yl[1], panel.y.lim[1], 1e-9), s.yl.join(','));

  var worst = 0, pts = 0;

  panel.series.forEach(function (ser) {
    var g = groupNamed(svg, ser.name);
    if (!ok(label + ': series "' + ser.name + '" present', !!g)) return;
    ok(label + ': "' + ser.name + '" kind ' + ser.kind, g.getAttribute('data-kind') === ser.kind,
      g.getAttribute('data-kind'));

    var i, dx, dy, r, segs, bad;

    if (ser.kind === 'hline') {
      var ln = byTag(g, 'line')[0];
      dy = Math.abs(parseFloat(ln.getAttribute('y1')) - s.ys(ser.y));
      worst = Math.max(worst, dy); pts++;
      ok(label + ': "' + ser.name + '" at y=' + ser.y, dy <= TOL, 'off by ' + dy.toFixed(3) + 'px');
      ok(label + ': "' + ser.name + '" spans the plot',
        near(parseFloat(ln.getAttribute('x1')), s.x0) && near(parseFloat(ln.getAttribute('x2')), s.x1));
      if (ser.style === 'dashed') {
        ok(label + ': "' + ser.name + '" is dashed', !!ln.getAttribute('stroke-dasharray'));
      }
      return;
    }

    if (ser.kind === 'line') {
      var med = ser[(ser.draw && ser.draw.line) || 'q50'];
      var medPath = byTag(g, 'path').filter(function (n) {
        return n.getAttribute('data-part') === 'median';
      })[0];
      var ribPath = byTag(g, 'path').filter(function (n) {
        return n.getAttribute('data-part') === 'ribbon';
      })[0];
      if (!ok(label + ': "' + ser.name + '" median path', !!medPath)) return;
      var P = pathPoints(medPath.getAttribute('d'));
      ok(label + ': "' + ser.name + '" median has ' + med.length + ' points',
        P.length === med.length, 'got ' + P.length);
      bad = null;
      for (i = 0; i < med.length && i < P.length; i++) {
        dx = Math.abs(P[i][0] - s.xs(ser.x[i]));
        dy = Math.abs(P[i][1] - s.ys(med[i]));
        worst = Math.max(worst, dx, dy); pts += 2;
        if (dx > TOL || dy > TOL) {
          bad = 'band ' + i + ' dx=' + dx.toFixed(3) + ' dy=' + dy.toFixed(3);
          break;
        }
      }
      ok(label + ': "' + ser.name + '" every median point at its axis position', !bad, bad);

      if (ok(label + ': "' + ser.name + '" quartile ribbon', !!ribPath)) {
        var R = pathPoints(ribPath.getAttribute('d'));
        ok(label + ': "' + ser.name + '" ribbon traces q75 out and q25 back',
          R.length === 2 * med.length, 'got ' + R.length);
        bad = null;
        for (i = 0; i < med.length && i < R.length; i++) {
          dy = Math.abs(R[i][1] - s.ys(ser.q75[i]));
          var dy2 = Math.abs(R[2 * med.length - 1 - i][1] - s.ys(ser.q25[i]));
          worst = Math.max(worst, dy, dy2); pts += 2;
          if (dy > TOL || dy2 > TOL) { bad = 'band ' + i; break; }
        }
        ok(label + ': "' + ser.name + '" ribbon edges are q25 and q75', !bad, bad);
        ok(label + ': "' + ser.name + '" ribbon alpha ' +
          (ser.draw.ribbon_alpha),
          Math.abs(parseFloat(ribPath.getAttribute('fill-opacity')) - ser.draw.ribbon_alpha) < 1e-9);
      }

      var marks = byTag(g, 'circle').filter(function (n) {
        return n.getAttribute('data-part') === 'marker';
      });
      ok(label + ': "' + ser.name + '" one marker per band', marks.length === med.length,
        'got ' + marks.length);
      var hits = byTag(g, 'circle').filter(function (n) {
        return n.getAttribute('class') === 'hit';
      });
      ok(label + ': "' + ser.name + '" one hover target per band', hits.length === med.length);
      ok(label + ': "' + ser.name + '" stroke is the JSON colour',
        medPath.getAttribute('stroke') === ser.color, medPath.getAttribute('stroke'));
      return;
    }

    if (ser.kind === 'bar') {
      var w = ser.width * (s.x1 - s.x0) / (s.xl[1] - s.xl[0]);
      bad = null;
      for (i = 0; i < ser.y.length; i++) {
        r = byTag(g, 'rect').filter(function (n) {
          return n.getAttribute('data-band') === String(i);
        })[0];
        if (!r) { bad = 'no rect for band ' + i; break; }
        var b = ser.bottom ? ser.bottom[i] : 0;
        var top = b + ser.y[i];
        var ey = Math.min(s.ys(b), s.ys(top));
        var eh = Math.abs(s.ys(top) - s.ys(b));
        var ex = s.xs(ser.x[i]) - w / 2;
        dx = Math.abs(parseFloat(r.getAttribute('x')) - ex);
        dy = Math.abs(parseFloat(r.getAttribute('y')) - ey);
        var dh = Math.abs(parseFloat(r.getAttribute('height')) - eh);
        var dw = Math.abs(parseFloat(r.getAttribute('width')) - w);
        worst = Math.max(worst, dx, dy, dh, dw); pts += 4;
        if (dx > TOL || dy > TOL || dh > TOL || dw > TOL) {
          bad = 'band ' + i + ' dx=' + dx.toFixed(3) + ' dy=' + dy.toFixed(3) +
            ' dh=' + dh.toFixed(3) + ' dw=' + dw.toFixed(3);
          break;
        }
        if (ser.bottom) {
          ok(label + ': "' + ser.name + '" band ' + i + ' sits on the bar below',
            near(parseFloat(r.getAttribute('y')) + parseFloat(r.getAttribute('height')),
              s.ys(b)), 'base should be ' + s.ys(b).toFixed(2));
        }
      }
      ok(label + ': "' + ser.name + '" every bar at its axis position and width', !bad, bad);
      ok(label + ': "' + ser.name + '" fill is the JSON colour',
        byTag(g, 'rect')[0].getAttribute('fill') === ser.color);
      if (ser.stacked_on) {
        ok(label + ': "' + ser.name + '" records its stack parent',
          g.getAttribute('data-stacked-on') === ser.stacked_on);
      }
      return;
    }

    if (ser.kind === 'segments') {
      segs = byTag(g, 'line').filter(function (n) { return n.hasAttribute('data-band'); });
      ok(label + ': "' + ser.name + '" ' + ser.y.length + ' short marks, not a line',
        segs.length === ser.y.length, 'got ' + segs.length);
      ok(label + ': "' + ser.name + '" drawn with no path element',
        byTag(g, 'path').length === 0);
      bad = null;
      for (i = 0; i < ser.y.length && i < segs.length; i++) {
        var L = segs[i];
        var d1 = Math.abs(parseFloat(L.getAttribute('y1')) - s.ys(ser.y[i]));
        var d2 = Math.abs(parseFloat(L.getAttribute('y2')) - s.ys(ser.y[i]));
        var d3 = Math.abs(parseFloat(L.getAttribute('x1')) - s.xs(ser.x0[i]));
        var d4 = Math.abs(parseFloat(L.getAttribute('x2')) - s.xs(ser.x1[i]));
        worst = Math.max(worst, d1, d2, d3, d4); pts += 4;
        if (d1 > TOL || d2 > TOL || d3 > TOL || d4 > TOL) {
          bad = 'band ' + i + ' dy=' + d1.toFixed(3) + ' dx0=' + d3.toFixed(3) +
            ' dx1=' + d4.toFixed(3);
          break;
        }
      }
      ok(label + ': "' + ser.name + '" every mark at its axis position and span', !bad, bad);
      return;
    }
  });

  /* high-frequency shading */
  if (xa.hf_shading) {
    var sh = byTag(svg, 'rect').filter(function (n) {
      return n.getAttribute('class') === 'shade';
    })[0];
    if (ok(label + ': high-frequency shading drawn', !!sh)) {
      var cl = function (v) { return Math.min(Math.max(v, s.xl[0]), s.xl[1]); };
      ok(label + ': shading starts at band 7 edge x=' + xa.hf_shading.from_x,
        near(parseFloat(sh.getAttribute('x')), s.xs(cl(xa.hf_shading.from_x))),
        sh.getAttribute('x') + ' vs ' + s.xs(cl(xa.hf_shading.from_x)).toFixed(2));
      ok(label + ': shading ends at x=' + xa.hf_shading.to_x + ' clamped to the axis',
        near(parseFloat(sh.getAttribute('x')) + parseFloat(sh.getAttribute('width')),
          s.xs(cl(xa.hf_shading.to_x))));
    }
  }

  /* x tick labels: only the labelled bands */
  var wantX = xa.tick_labels.filter(function (t) { return t !== ''; });
  var gotX = textsWithClass(svg, 't-tick').filter(function (t) {
    return wantX.indexOf(t) >= 0;
  });
  ok(label + ': x labels only on bands ' + (xa.labelled_bands || []).join(','),
    wantX.every(function (t) { return gotX.indexOf(t) >= 0; }) && gotX.length >= wantX.length,
    'want ' + wantX.join('|') + ' got ' + gotX.join('|'));

  /* role=img + title + desc */
  ok(label + ': role="img" with title and desc',
    svg.getAttribute('role') === 'img' && byTag(svg, 'title').length === 1 &&
    byTag(svg, 'desc').length === 1 && !!svg.getAttribute('aria-labelledby'));

  /* annotations (high-frequency label, dagger flags, real/pseudo arrows) */
  if ((panel.annotations || []).length) {
    var annos = byTag(svg, 'text').filter(function (n) { return n.hasAttribute('data-anno'); });
    ok(label + ': ' + panel.annotations.length + ' annotation(s) drawn',
      annos.length === panel.annotations.length, 'got ' + annos.length);
  }

  return { worst: worst, pts: pts };
}

/* ================================================================= run === */

head('LOAD');
var env = freshModule({ fetch: makeFetch([]) });
var API = env.API, W = env.W;
ok('module exports its own global', !!W.LiEQPaperCharts && W.LiEQPaperCharts === API);
ok('merged into window.LiEQCharts without clobbering', !!W.LiEQCharts &&
  W.LiEQCharts.renderFig4 === API.renderFig4);
ok('four render keys', API.keys.join(',') === 'paper-fig3,paper-fig3a,paper-fig3b,paper-fig4',
  API.keys.join(','));
ok('data is not embedded in the module',
  fs.readFileSync(MODULE, 'utf8').indexOf('0.921059') < 0 &&
  fs.readFileSync(MODULE, 'utf8').indexOf('48.492387') < 0);
line('module size: ' + Math.round(fs.statSync(MODULE).size / 1024) + ' KB, ' +
  fs.readFileSync(MODULE, 'utf8').split('\n').length + ' lines');

var m3 = mountIn(W), m4 = mountIn(W);
var worst = 0, pts = 0;

Promise.all([
  API.renderFig3(m3, { width: 1180, watch: false }),
  API.renderFig4(m4, { width: 1180, watch: false })
]).then(function (figs) {

  /* ------------------------------------------------------------- Fig. 3 -- */
  head('FIG. 3  (' + J3A.paper_ref + ' + ' + J3B.paper_ref + ')');
  var fig3 = figs[0];
  ok('fig 3 rendered', !!fig3);
  var P3 = panelsOf(fig3);
  ok('fig 3 draws 6 panels: two rows of three', P3.length === 6, 'got ' + P3.length);
  var rows = byTag(fig3, 'div').filter(function (n) {
    return n.getAttribute('class') === 'lqp-row';
  });
  ok('fig 3 has two rows', rows.length === 2, 'got ' + rows.length);

  var want3 = [
    ['3a', J3A, 0], ['3a', J3A, 1], ['3a', J3A, 2],
    ['3b', J3B, 0], ['3b', J3B, 1], ['3b', J3B, 2]
  ];
  want3.forEach(function (w, i) {
    var panel = w[1].panels[w[2]];
    var svg = P3[i];
    ok('panel ' + i + ' is ' + w[0] + '/' + panel.id,
      svg.getAttribute('data-panel') === panel.id, svg.getAttribute('data-panel'));
    var r = checkPanel(svg, panel, 'fig ' + w[0] + ' ' + panel.id);
    if (r) { worst = Math.max(worst, r.worst); pts += r.pts; }
  });

  head('FIG. 3  shared y axis');
  [0, 3].forEach(function (i) {
    var left = P3[i], mid = P3[i + 1], right = P3[i + 2];
    var want = J3A.panels[0].y.tick_labels;
    var lt = textsWithClass(left, 't-tick');
    var mt = textsWithClass(mid, 't-tick');
    var rt = textsWithClass(right, 't-tick');
    ok('row ' + (i ? 'b' : 'a') + ': leftmost panel prints ' + want.join('/'),
      want.every(function (t) { return lt.indexOf(t) >= 0; }), lt.join('|'));
    ok('row ' + (i ? 'b' : 'a') + ': middle panel (shared_with_panel) prints none of them',
      !want.some(function (t) { return mt.indexOf(t) >= 0; }), mt.join('|'));
    ok('row ' + (i ? 'b' : 'a') + ': gap panel prints its own percent ticks on the right',
      rt.indexOf('20%') >= 0 || rt.indexOf('100%') >= 0, rt.join('|'));
    var yl = byTag(left, 'text').filter(function (n) {
      return n.getAttribute('class') === 't-ax';
    });
    ok('row ' + (i ? 'b' : 'a') + ': y label keeps its line break',
      yl.length === 1 && byTag(yl[0], 'tspan').length === 2,
      yl.length ? byTag(yl[0], 'tspan').length + ' tspans' : 'no label');
  });

  /* ------------------------------------------------------------- Fig. 4 -- */
  head('FIG. 4  (' + J4.paper_ref + ')');
  var fig4 = figs[1];
  ok('fig 4 rendered', !!fig4);
  var P4 = panelsOf(fig4);
  ok('fig 4 draws 6 panels', P4.length === 6, 'got ' + P4.length);
  ok('fig 4 lays them out in three rows of two',
    byTag(fig4, 'div').filter(function (n) {
      return n.getAttribute('class') === 'lqp-row';
    }).length === 3);
  ok('fig 4 panel order is a,b,c,d,e,f',
    P4.map(function (p) { return p.getAttribute('data-panel'); }).join('') === 'abcdef',
    P4.map(function (p) { return p.getAttribute('data-panel'); }).join(''));
  ok('fig 4 titles keep their a-f letters',
    P4.every(function (p, i) {
      return textsWithClass(p, 't-pan')[0].indexOf('abcdef'.charAt(i)) === 0;
    }));
  ok('fig 3 titles carry no letter',
    P3.every(function (p) { return !/^[a-f]\s/.test(textsWithClass(p, 't-pan')[0]); }));

  J4.panels.forEach(function (panel, i) {
    var r = checkPanel(P4[i], panel, 'fig 4' + panel.id);
    if (r) { worst = Math.max(worst, r.worst); pts += r.pts; }
  });

  head('FIG. 4  the figure-specific kinds');
  var cSeg = groupNamed(P4[2], 'Teacher, re-truncated: reproducible part');
  ok('panel c segments are 11 separate <line> marks',
    byTag(cSeg, 'line').filter(function (n) { return n.hasAttribute('data-band'); }).length === 11);
  var s2 = scalesOf(P4[2]);
  var band5 = byTag(cSeg, 'line').filter(function (n) {
    return n.getAttribute('data-band') === '5';
  })[0];
  line('spot check  fig 4c segment band 5: y=' + J4.panels[2].series[4].y[5] +
    ' -> ' + s2.ys(J4.panels[2].series[4].y[5]).toFixed(2) + 'px, markup says ' +
    band5.getAttribute('y1'));
  line('spot check  fig 4c segment band 5 spans x 4.58..5.42 -> ' +
    s2.xs(4.58).toFixed(2) + '..' + s2.xs(5.42).toFixed(2) + 'px, markup says ' +
    band5.getAttribute('x1') + '..' + band5.getAttribute('x2'));

  var ePseudo = groupNamed(P4[4], 'Δ pseudo energy (LiEqualizer − LiSeer)');
  var r0 = byTag(ePseudo, 'rect').filter(function (n) {
    return n.getAttribute('data-band') === '0';
  })[0];
  var s4 = scalesOf(P4[4]);
  ok('panel e negative bar hangs below zero',
    near(parseFloat(r0.getAttribute('y')), s4.ys(0)) &&
    near(parseFloat(r0.getAttribute('height')), s4.ys(-0.170997) - s4.ys(0)),
    'y=' + r0.getAttribute('y') + ' h=' + r0.getAttribute('height') +
    ' zero at ' + s4.ys(0).toFixed(2));
  line('spot check  fig 4e pseudo band 0: ' + J4.panels[4].series[1].y[0] +
    ' -> top ' + s4.ys(0).toFixed(2) + 'px, bottom ' + s4.ys(-0.170997).toFixed(2) + 'px');

  var s0 = scalesOf(P3[0]);
  line('spot check  fig 3a left LiEqualizer band 0 median: ' +
    J3A.panels[0].series[3].q50[0] + ' -> ' +
    s0.ys(J3A.panels[0].series[3].q50[0]).toFixed(2) + 'px of plot [' +
    s0.y1 + '..' + s0.y0 + '] on y range ' + s0.yl.join('..'));
  var sg = scalesOf(P3[2]);
  var gb = byTag(groupNamed(P3[2], 'observed by both'), 'rect').filter(function (n) {
    return n.getAttribute('data-band') === '0';
  })[0];
  line('spot check  fig 3a gap bar band 0: ' + J3A.panels[2].series[0].y[0] +
    '% -> y ' + sg.ys(48.492387).toFixed(2) + 'px (zero at ' + sg.ys(0).toFixed(2) +
    '), markup y=' + gb.getAttribute('y') + ' h=' + gb.getAttribute('height'));

  /* ------------------------------------------------------------ legends -- */
  head('LEGENDS AND AXIS LINE');
  function legendKeys(scope) {
    return byTag(scope, 'span').filter(function (n) {
      return n.getAttribute('class') === 'lqp-key';
    }).map(function (n) { return n.textContent; });
  }
  var top3 = byTag(fig3, 'div').filter(function (n) {
    return n.getAttribute('class') === 'lqp-legend';
  })[0];
  var want3Keys = J3A.figure_legends[0].entries.map(function (e) { return e.label; });
  ok('fig 3 legend reproduces the published 4 entries in order',
    legendKeys(top3).join('|') === want3Keys.join('|'), legendKeys(top3).join('|'));
  ok('fig 3 axis line names both y quantities',
    byTag(fig3, 'p').some(function (p) {
      return p.getAttribute('class') === 'lqp-axes' &&
        p.textContent.indexOf(J3A.units.y_right) >= 0;
    }));
  var rowLegends = byTag(fig4, 'div').filter(function (n) {
    return n.getAttribute('class') === 'lqp-legend is-row';
  });
  ok('fig 4 has one legend per row', rowLegends.length === 3, 'got ' + rowLegends.length);
  ok('fig 4 row 1 legend is the published 3-entry line legend',
    legendKeys(rowLegends[0]).join('|') ===
    J4.figure_legends[0].entries.map(function (e) { return e.label; }).join('|'),
    legendKeys(rowLegends[0]).join('|'));
  ok('fig 4 row 2 legend is the published 6-entry real/pseudo legend',
    legendKeys(rowLegends[1]).join('|') ===
    J4.figure_legends[1].entries.map(function (e) { return e.label; }).join('|'),
    legendKeys(rowLegends[1]).join('|'));
  ok('fig 4 row 3 legend is derived from the e/f bar series themselves',
    legendKeys(rowLegends[2]).join('|') ===
    J4.panels[4].series.filter(function (s) { return s.kind === 'bar'; })
      .map(function (s) { return s.name; }).join('|'),
    legendKeys(rowLegends[2]).join('|'));
  ok('fig 4 prints the published column headers',
    byTag(fig4, 'div').some(function (n) {
      return n.getAttribute('class') === 'lqp-cols' &&
        n.textContent === J4.panels[0].column + J4.panels[1].column;
    }));
  ok('every row repeats the x-axis label from the JSON',
    byTag(fig4, 'p').filter(function (p) {
      return p.getAttribute('class') === 'lqp-xlab' &&
        p.textContent === J4.panels[0].x.label;
    }).length === 3);
  ok('every panel tags the figure it belongs to',
    P3.every(function (p) { return /^Fig\. 3[ab]$/.test(p.getAttribute('data-figure')); }) &&
    P4.every(function (p) { return p.getAttribute('data-figure') === 'Fig. 4'; }));

  /* ----------------------------------------------------------- captions -- */
  head('CAPTIONS AND NOTES');
  var note3 = byTag(fig3, 'figcaption')[0].textContent;
  var note4 = byTag(fig4, 'figcaption')[0].textContent;
  ok('fig 3 note states the frame count and selection',
    note3.indexOf('199 frames') >= 0 && note3.indexOf('round(linspace(0, 4070, 200))') >= 0);
  ok('fig 3 note names the field and the operator',
    note3.indexOf('TPV feature plane xy') >= 0 && note3.indexOf('voxel logits') >= 0 &&
    note3.indexOf('grid Laplacian') >= 0 && note3.indexOf('support-subgraph Laplacian') >= 0);
  ok('fig 3 note names the noise-ceiling denominator',
    note3.indexOf('re-truncation noise ceiling') >= 0);
  ok('fig 4 note says "divided by the teacher’s own band energy"',
    note4.indexOf('divided by the teacher’s own band energy') >= 0);
  ok('fig 4 note does NOT say "divided by Dense teacher"',
    note4.indexOf('divided by Dense teacher') < 0);
  ok('fig 4 note says the denominator input was re-truncated',
    note4.indexOf('re-truncated') >= 0);
  ok('fig 3 note discloses the epoch-21 checkpoint and the epoch-24 readings',
    note3.indexOf('epoch-21') >= 0 && note3.indexOf('epoch 24') >= 0);
  ok('fig 4 note discloses the epoch-21 checkpoint and the epoch-24 readings',
    note4.indexOf('epoch-21') >= 0 && note4.indexOf('epoch 24') >= 0);
  ok('the epoch disclosure is one sentence',
    /epoch-21 checkpoint[^.]*epoch 24\./.test(note4));
  ok('the JSON states epoch 21, so the note is not invented',
    J4.protocol.models[3].checkpoint.indexOf('epoch-21') >= 0,
    J4.protocol.models[3].checkpoint);
  ok('fig 4 note explains the dagger flag', note4.indexOf('†') >= 0 &&
    note4.indexOf('0.05') >= 0);
  ok('fig 4 keeps the published footnote verbatim',
    byTag(fig4, 'p').some(function (p) {
      return p.getAttribute('class') === 'lqp-pub' &&
        p.textContent.indexOf(J4.figure_texts[2].text) >= 0;
    }));

  /* ------------------------------------------------------ accessibility -- */
  head('ACCESSIBILITY');
  [['fig 3', fig3, 2], ['fig 4', fig4, 1]].forEach(function (t) {
    var fig = t[1];
    var btn = byTag(fig, 'button')[0];
    var sr = byTag(fig, 'div').filter(function (n) {
      return n.getAttribute('class') === 'lqp-sr';
    })[0];
    ok(t[0] + ': data table starts visually hidden', !!sr &&
      sr.getAttribute('class') === 'lqp-sr');
    ok(t[0] + ': toggle is a button with aria-expanded=false', !!btn &&
      btn.getAttribute('type') === 'button' && btn.getAttribute('aria-expanded') === 'false');
    ok(t[0] + ': toggle points at the table it controls', !!btn && !!sr &&
      btn.getAttribute('aria-controls') === sr.getAttribute('id'));
    ok(t[0] + ': ' + t[2] + ' data table(s) in the hidden box',
      byTag(sr, 'table').length === t[2], 'got ' + byTag(sr, 'table').length);
    byTag(sr, 'table').forEach(function (tb) {
      ok(t[0] + ': table header is 1 + 11 bands',
        byTag(byTag(tb, 'thead')[0], 'th').length === 12,
        'got ' + byTag(byTag(tb, 'thead')[0], 'th').length);
      ok(t[0] + ': every row has 12 cells',
        byTag(byTag(tb, 'tbody')[0], 'tr').every(function (tr) {
          return tr.childNodes.length === 12;
        }));
      ok(t[0] + ': row headers are scoped th',
        byTag(byTag(tb, 'tbody')[0], 'tr').every(function (tr) {
          return tr.childNodes[0].tagName === 'th' &&
            tr.childNodes[0].getAttribute('scope') === 'row';
        }));
    });
    btn.dispatch('click');
    ok(t[0] + ': the toggle reveals the table', sr.getAttribute('class') === 'lqp-sr is-open' &&
      btn.getAttribute('aria-expanded') === 'true' && btn.textContent === 'Hide data table');
    btn.dispatch('click');
    ok(t[0] + ': and hides it again', sr.getAttribute('class') === 'lqp-sr' &&
      btn.textContent === 'Show data table');
  });

  /* fig 3 tables: one row per non-hline series, plus an IQR row per line series */
  var sr3 = byTag(fig3, 'div').filter(function (n) {
    return n.getAttribute('class') === 'lqp-sr';
  })[0];
  var t3a = byTag(sr3, 'table')[0];
  var expRows = 0;
  J3A.panels.forEach(function (p) {
    p.series.forEach(function (s) {
      if (s.kind === 'hline') return;
      expRows += s.kind === 'line' ? 2 : 1;
    });
  });
  ok('fig 3a table has ' + expRows + ' rows (median + IQR per curve, one per bar series)',
    byTag(byTag(t3a, 'tbody')[0], 'tr').length === expRows,
    'got ' + byTag(byTag(t3a, 'tbody')[0], 'tr').length);
  var row0 = byTag(byTag(t3a, 'tbody')[0], 'tr').filter(function (tr) {
    return tr.textContent.indexOf('LiEqualizer (median)') >= 0;
  })[0];
  ok('fig 3a table prints the JSON value for LiEqualizer band 0',
    row0.childNodes[1].textContent === '0.9211', row0.childNodes[1].textContent);
  line('table cell check  LiEqualizer band 0 median 0.921059 -> "' +
    row0.childNodes[1].textContent + '" (4 significant digits)');

  /* -------------------------------------------------------- resilience -- */
  head('RESILIENCE');
  var e2 = freshModule({ fetch: makeFetch([]) });
  var p1 = e2.API.renderFig3(null, { width: 900 });
  var p2 = e2.API.renderFig4('no-such-id', { width: 900 });
  var e3 = freshModule({
    fetch: function () { return Promise.resolve({ ok: false, status: 404 }); }
  });
  var m = mountIn(e3.W);
  var e4 = freshModule({ fetch: function () { return Promise.reject(new Error('offline')); } });
  var m2 = mountIn(e4.W);
  var e5 = freshModule({ fetch: makeFetch([]), reducedMotion: true });
  var m5 = mountIn(e5.W);

  return Promise.all([
    p1, p2,
    e3.API.renderFig4(m, { width: 900, watch: false }),
    e4.API.renderFig3(m2, { width: 900, watch: false }),
    e5.API.renderFig4(m5, { width: 360, watch: false })
  ]).then(function (r) {
    ok('a missing mount returns null and does not throw', r[0] === null && r[1] === null);
    ok('a 404 renders a note in the mount instead of throwing', r[2] === null &&
      byTag(m, 'div').some(function (n) {
        return (n.getAttribute('class') || '').indexOf('lqp-missing') >= 0;
      }));
    ok('the 404 note names the file it could not read',
      m.textContent.indexOf('fig4_energy_real.json') >= 0);
    ok('a rejected fetch renders the same note', r[3] === null &&
      m2.textContent.indexOf('fig3a_band_cka_feature_planes.json') >= 0);
    ok('prefers-reduced-motion drops the animation class',
      (r[4].getAttribute('class') || '').indexOf('lqp-anim') < 0,
      r[4].getAttribute('class'));

    head('NARROW LAYOUT (360px)');
    var narrow = panelsOf(r[4]);
    ok('fig 4 still draws all six panels at 360px', narrow.length === 6);
    var sN = scalesOf(narrow[0]);
    ok('the plot box is still positive at 360px', sN.x1 > sN.x0 && sN.y0 > sN.y1,
      [sN.x0, sN.x1, sN.y0, sN.y1].join(','));
    ok('every panel keeps its y ticks when the grid collapses to one column',
      narrow.every(function (p) { return textsWithClass(p, 't-tick').length > 5; }));
    ok('one-column panels gain the column name as a subtitle',
      textsWithClass(narrow[0], 't-sub')[0] === J4.panels[0].column,
      textsWithClass(narrow[0], 't-sub')[0]);

    head('FIG. 3 SINGLE ROWS');
    var e6 = freshModule({ fetch: makeFetch([]) });
    var ma = mountIn(e6.W), mb = mountIn(e6.W);
    return Promise.all([
      e6.API.renderFig3a(ma, { width: 1180, watch: false }),
      e6.API.renderFig3b(mb, { width: 1180, watch: false })
    ]).then(function (rr) {
      ok('fig 3a alone draws three panels', panelsOf(rr[0]).length === 3);
      ok('fig 3b alone draws three panels', panelsOf(rr[1]).length === 3);
      ok('3a-alone note describes only the feature plane',
        byTag(rr[0], 'figcaption')[0].textContent.indexOf('TPV feature plane xy') >= 0 &&
        byTag(rr[0], 'figcaption')[0].textContent.indexOf('Jackson') < 0);
      ok('3b-alone note describes only the voxel logits',
        byTag(rr[1], 'figcaption')[0].textContent.indexOf('Jackson') >= 0);

      head('FETCH BEHAVIOUR');
      var log = [];
      var e7 = freshModule({ fetch: makeFetch(log) });
      var m7 = mountIn(e7.W), m8 = mountIn(e7.W);
      return Promise.all([
        e7.API.renderFig3(m7, { width: 1180, watch: false }),
        e7.API.renderFig3a(m8, { width: 1180, watch: false })
      ]).then(function () {
        ok('each JSON file is fetched once and shared between mounts',
          log.length === 2, log.join(' '));
        ok('fetched relative to assets/data/',
          log[0] === 'assets/data/fig3a_band_cka_feature_planes.json', log[0]);

        head('DARK THEME');
        var e8 = freshModule({ fetch: makeFetch([]) });
        var m9 = mountIn(e8.W);
        return e8.API.renderFig4(m9, { width: 1180, theme: 'dark', watch: false })
          .then(function (figD) {
            ok('dark theme sets the theme attribute',
              figD.getAttribute('data-lqp-theme') === 'dark');
            var g = groupNamed(panelsOf(figD)[0], 'LiEqualizer');
            var stroke = byTag(g, 'path').filter(function (n) {
              return n.getAttribute('data-part') === 'median';
            })[0].getAttribute('stroke');
            ok('on dark, #2a78d6 becomes the page\'s on-dark blue #56adf0',
              stroke === '#56adf0', stroke);

            /* -------------------------------------------------------- end */
            head('RESULT');
            console.log('  ' + pass + ' / ' + checks + ' checks passed');
            console.log('  ' + pts + ' rendered coordinates recomputed from the axis ranges; ' +
              'worst disagreement ' + worst.toFixed(4) + ' px (rounding tolerance ' +
              TOL + ')');
            if (fail) {
              console.log('\n  FAILURES:');
              failures.forEach(function (f) { console.log('   - ' + f); });
              process.exitCode = 1;
            }
          });
      });
    });
  });
}).catch(function (err) {
  console.log('\nHARNESS ERROR: ' + (err && err.stack ? err.stack : err));
  process.exitCode = 1;
});
