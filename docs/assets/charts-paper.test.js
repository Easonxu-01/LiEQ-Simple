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
  /* Rows b and d of Fig. 4 declare a broken y axis. Rebuild it here from what
     the panel states rather than trusting the module's own helper, so the test
     still measures the markup against an independently computed position. */
  var sp = svg.getAttribute('data-ysplit');
  var ys;
  if (sp) {
    var q = sp.split(',').map(Number);
    var zero = p[2] - q[2] * (p[2] - p[3]);
    ys = function (v) {
      if (v >= 0) return q[0] > 0 ? zero - (v / q[0]) * (zero - p[3]) : zero;
      return q[1] > 0 ? zero + (-v / q[1]) * (p[2] - zero) : zero;
    };
  } else {
    ys = function (v) { return p[2] + (v - yl[0]) * (p[3] - p[2]) / (yl[1] - yl[0]); };
  }
  return {
    x0: p[0], x1: p[1], y0: p[2], y1: p[3], xl: xl, yl: yl, split: sp || null,
    xs: function (v) { return p[0] + (v - xl[0]) * (p[1] - p[0]) / (xl[1] - xl[0]); },
    ys: ys
  };
}

/* the panels carry their size in the viewBox, not in a height attribute */
function vbH(svg) {
  return Number(String(svg.getAttribute('viewBox')).split(' ')[3]);
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

      /* the stack-top "energy distribution" lines carry no quartiles, and a
         ribbon drawn for them would be a ribbon of nothing */
      var wantRibbon = !!(ser.q25 && ser.q75);
      ok(label + ': "' + ser.name + '" ' +
        (wantRibbon ? 'has a quartile ribbon' : 'draws no ribbon without quartiles'),
        wantRibbon === !!ribPath);
      if (wantRibbon && ribPath) {
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
  fs.readFileSync(MODULE, 'utf8').indexOf(String(J3A.panels[0].series[3].q25[0])) < 0 &&
  fs.readFileSync(MODULE, 'utf8').indexOf(String(J4.panels[0].series[1].y[0])) < 0);
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
  ok('fig 3 draws 12 panels: two levels, three datasets, a gap strip each',
    P3.length === 12, 'got ' + P3.length);
  var rows = byTag(fig3, 'div').filter(function (n) {
    return n.getAttribute('class') === 'lqp-row';
  });
  ok('fig 3 has one row per representation level', rows.length === 2, 'got ' + rows.length);

  var want3 = [];
  [['3a', J3A], ['3b', J3B]].forEach(function (lv) {
    lv[1].panels.forEach(function (p, k) { want3.push([lv[0], lv[1], k]); });
  });
  ok('fig 3 panel order is the three datasets, then their gap strips',
    P3.map(function (p) { return p.getAttribute('data-panel'); }).join(',') ===
    'cka_SK,cka_NU,cka_WA,gap_SK,gap_NU,gap_WA,cka_SK,cka_NU,cka_WA,gap_SK,gap_NU,gap_WA',
    P3.map(function (p) { return p.getAttribute('data-panel'); }).join(','));
  want3.forEach(function (w, i) {
    var panel = w[1].panels[w[2]];
    var svg = P3[i];
    ok('panel ' + i + ' is ' + w[0] + '/' + panel.id,
      svg.getAttribute('data-panel') === panel.id, svg.getAttribute('data-panel'));
    var r = checkPanel(svg, panel, 'fig ' + w[0] + ' ' + panel.id);
    if (r) { worst = Math.max(worst, r.worst); pts += r.pts; }
  });

  head('FIG. 3  three datasets side by side');
  [0, 6].forEach(function (base) {
    var lvl = base ? 'b' : 'a';
    var src = base ? J3B : J3A;
    ['SemanticKITTI', 'nuScenes', 'Waymo'].forEach(function (name, k) {
      ok('row ' + lvl + ': column ' + k + ' is titled ' + name,
        textsWithClass(P3[base + k], 't-pan')[0] === name,
        textsWithClass(P3[base + k], 't-pan')[0]);
      var ct = textsWithClass(P3[base + k], 't-tick');
      ok('row ' + lvl + ' ' + name + ': the CKA panel prints its own y ticks',
        src.panels[k].y.tick_labels.every(function (t) { return ct.indexOf(t) >= 0; }),
        ct.join('|'));
      var gt = textsWithClass(P3[base + 3 + k], 't-tick');
      ok('row ' + lvl + ' ' + name + ': the gap strip prints percent ticks',
        gt.some(function (t) { return /%$/.test(t); }), gt.join('|'));
      ok('row ' + lvl + ' ' + name + ': the gap strip is shorter than the CKA panel',
        vbH(P3[base + 3 + k]) < vbH(P3[base + k]),
        vbH(P3[base + 3 + k]) + ' vs ' + vbH(P3[base + k]));
    });
    var yl = byTag(P3[base], 'text').filter(function (n) {
      return n.getAttribute('class') === 't-ax';
    });
    ok('row ' + lvl + ': y label keeps its line break',
      yl.length === 1 && byTag(yl[0], 'tspan').length === 2,
      yl.length ? byTag(yl[0], 'tspan').length + ' tspans' : 'no label');
  });

  head('FIG. 3  the merged support domain');
  ok('no panel still names a single support domain',
    !JSON.stringify([J3A.panels, J3B.panels]).match(/observed by (both|the teacher only)/i));
  ok('every CKA panel is normalised, with the ceiling at 1',
    [J3A, J3B].every(function (d) {
      return d.panels.slice(0, 3).every(function (p) {
        return p.series[0].kind === 'hline' && p.series[0].y === 1;
      });
    }));

  /* ------------------------------------------------------------- Fig. 4 -- */
  head('FIG. 4  (' + J4.paper_ref + ')');
  var fig4 = figs[1];
  ok('fig 4 rendered', !!fig4);
  var P4 = panelsOf(fig4);
  var FLAT4 = [].concat.apply([], J4.grid.rows);
  ok('fig 4 draws 12 panels: four readings across three datasets',
    P4.length === 12, 'got ' + P4.length);
  ok('fig 4 lays them out in four rows of three',
    byTag(fig4, 'div').filter(function (n) {
      return n.getAttribute('class') === 'lqp-row';
    }).length === 4);
  ok('fig 4 panel order follows the grid declared in the data',
    P4.map(function (p) { return p.getAttribute('data-panel'); }).join(',') ===
    FLAT4.join(','), P4.map(function (p) { return p.getAttribute('data-panel'); }).join(','));
  ok('fig 4 titles are the dataset names, three to a row',
    P4.every(function (p, i) {
      return textsWithClass(p, 't-pan')[0] ===
        ['SemanticKITTI', 'nuScenes', 'Waymo'][i % 3];
    }));
  ok('fig 4 names its four readings once each above the rows',
    J4.grid.row_titles.length === 4 &&
    byTag(fig4, 'span').filter(function (n) {
      return J4.grid.row_titles.indexOf(n.textContent) >= 0;
    }).length === 4);
  ok('fig 3 titles carry no a-f letter',
    P3.every(function (p) { return !/^[a-f]\s/.test(textsWithClass(p, 't-pan')[0]); }));

  FLAT4.forEach(function (id, i) {
    var panel = J4.panels.filter(function (p) { return p.id === id; })[0];
    var r = checkPanel(P4[i], panel, 'fig 4 ' + id);
    if (r) { worst = Math.max(worst, r.worst); pts += r.pts; }
  });

  head('FIG. 4  the broken y axis of the paired rows');
  ['b_SK', 'b_NU', 'b_WA', 'd_SK', 'd_NU', 'd_WA'].forEach(function (id) {
    var svg = P4[FLAT4.indexOf(id)];
    var panel = J4.panels.filter(function (p) { return p.id === id; })[0];
    var sa = panel.y.split_axis;
    var sp = svg.getAttribute('data-ysplit');
    ok(id + ': the panel states its split axis',
      sp === [sa.P, sa.N, sa.h].join(','), sp);
    var s = scalesOf(svg);
    var zero = s.y0 - sa.h * (s.y0 - s.y1);
    ok(id + ': zero sits at the fraction h of the plot height',
      near(s.ys(0), zero), s.ys(0).toFixed(2) + ' vs ' + zero.toFixed(2));
    ok(id + ': +P reaches the top of the plot', near(s.ys(sa.P), s.y1),
      s.ys(sa.P).toFixed(2) + ' vs ' + s.y1);
    ok(id + ': -N reaches the bottom of the plot', near(s.ys(-sa.N), s.y0),
      s.ys(-sa.N).toFixed(2) + ' vs ' + s.y0);
    var zl = byTag(groupNamed(svg, 'zero'), 'line')[0];
    ok(id + ': the zero rule is drawn at that height',
      near(parseFloat(zl.getAttribute('y1')), zero));
  });
  ok('h is shared across each paired row, so the datasets stay comparable',
    ['b_SK', 'b_NU', 'b_WA'].map(function (id) {
      return J4.panels.filter(function (p) { return p.id === id; })[0].y.split_axis.h;
    }).every(function (v, _, a) { return v === a[0]; }) &&
    ['d_SK', 'd_NU', 'd_WA'].map(function (id) {
      return J4.panels.filter(function (p) { return p.id === id; })[0].y.split_axis.h;
    }).every(function (v, _, a) { return v === a[0]; }));

  head('FIG. 4  unsupported bands and broken bars');
  var hatchTotal = 0, overTotal = 0;
  J4.panels.forEach(function (panel) {
    var svg = P4[FLAT4.indexOf(panel.id)];
    var want = panel.hatched_bands || [];
    var got = byTag(svg, 'rect').filter(function (n) {
      return n.hasAttribute('data-hatched');
    });
    var bands = {};
    got.forEach(function (n) { bands[n.getAttribute('data-hatched')] = 1; });
    hatchTotal += got.length;
    ok(panel.id + ': hatches exactly the unsupported bands [' + want.join(',') + ']',
      want.length === Object.keys(bands).length &&
      want.every(function (b) { return bands[String(b)]; }),
      Object.keys(bands).join(','));
    var sp = scalesOf(svg), plotW = sp.x1 - sp.x0;
    ok(panel.id + ': the hatch is on the bars, not behind them',
      got.every(function (n) {
        return /^url\(#lqp-hatch-/.test(n.getAttribute('fill')) &&
          parseFloat(n.getAttribute('width')) > 0 &&
          parseFloat(n.getAttribute('width')) < plotW / 4;
      }));
    var over = panel.overflow || [];
    var marks = byTag(svg, 'text').filter(function (n) {
      return n.hasAttribute('data-overflow');
    });
    overTotal += marks.length;
    ok(panel.id + ': ' + over.length + ' broken bar(s), each printing its true value',
      marks.length === over.length &&
      over.every(function (o) {
        return marks.some(function (m) {
          return m.textContent === String(Math.round(o.true_value * 100) / 100) ||
            m.textContent.indexOf(String(o.true_value).slice(0, 4)) === 0;
        });
      }), marks.map(function (m) { return m.textContent; }).join(','));
  });
  line('hatched bar rects: ' + hatchTotal + ' · broken-bar labels: ' + overTotal);
  ok('some band is hatched and some bar is broken, so both paths are exercised',
    hatchTotal > 0 && overTotal > 0);

  head('NOTHING IS PAINTED OUTSIDE ITS PANEL');
  /* a stack top of 10.24 on an axis that stops at 1.5 put a marker a thousand
     pixels above its panel, on top of the figure above it. Every mark that can
     leave the plot box must carry the panel's clip. */
  var unclipped = 0, offplot = [];
  [['fig 3', P3], ['fig 4', P4]].forEach(function (t) {
    t[1].forEach(function (svg) {
      var s = scalesOf(svg), pid = svg.getAttribute('data-panel');
      byTag(svg, 'circle').concat(byTag(svg, 'rect')).forEach(function (n) {
        var cy = n.hasAttribute('cy') ? parseFloat(n.getAttribute('cy'))
          : (n.hasAttribute('y') ? parseFloat(n.getAttribute('y')) : null);
        if (cy === null || isNaN(cy)) return;
        var out = cy < s.y1 - 12 || cy > s.y0 + 12;
        if (!out) return;
        offplot.push(t[0] + ' ' + pid + ' at y=' + cy.toFixed(1));
        if (!n.getAttribute('clip-path')) unclipped++;
      });
    });
  });
  ok('every mark that falls outside its plot box is clipped to it',
    unclipped === 0, unclipped + ' unclipped of ' + offplot.length +
    ' off-plot marks: ' + offplot.slice(0, 4).join(', '));
  line('off-plot marks found: ' + offplot.length + ', all clipped');

  var s0 = scalesOf(P3[0]);
  line('spot check  fig 3a SemanticKITTI LiEqualizer band 0 median: ' +
    J3A.panels[0].series[3].q50[0] + ' -> ' +
    s0.ys(J3A.panels[0].series[3].q50[0]).toFixed(2) + 'px of plot [' +
    s0.y1 + '..' + s0.y0 + '] on y range ' + s0.yl.join('..'));
  var sg = scalesOf(P3[3]);
  var gb = byTag(groupNamed(P3[3], 'Gap closed by LiEqualizer'), 'rect').filter(function (n) {
    return n.getAttribute('data-band') === '0';
  })[0];
  line('spot check  fig 3a SemanticKITTI gap bar band 0: ' +
    J3A.panels[3].series[0].y[0] + '% -> y ' +
    sg.ys(J3A.panels[3].series[0].y[0]).toFixed(2) + 'px (zero at ' +
    sg.ys(0).toFixed(2) + '), markup y=' + gb.getAttribute('y') +
    ' h=' + gb.getAttribute('height'));

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
  ok('fig 3 legend reproduces the published ' + want3Keys.length + ' entries in order',
    legendKeys(top3).join('|') === want3Keys.join('|'), legendKeys(top3).join('|'));
  ok('fig 3 axis line names both y quantities',
    byTag(fig3, 'p').some(function (p) {
      return p.getAttribute('class') === 'lqp-axes' &&
        p.textContent.indexOf(J3A.units.y_strip) >= 0;
    }));
  var fig4Legends = byTag(fig4, 'div').filter(function (n) {
    return n.getAttribute('class') === 'lqp-legend';
  });
  ok('fig 4 carries one legend for the whole figure, as published',
    fig4Legends.length === 1, 'got ' + fig4Legends.length);
  ok('fig 4 legend reproduces the published entries in order',
    legendKeys(fig4Legends[0]).join('|') ===
    J4.figure_legends[0].entries.map(function (e) { return e.label; }).join('|'),
    legendKeys(fig4Legends[0]).join('|'));
  ok('fig 4 legend explains the hatch and the broken bar',
    legendKeys(fig4Legends[0]).join('|').indexOf('unsupported by sparse scan') >= 0 &&
    legendKeys(fig4Legends[0]).join('|').indexOf('broken bar') >= 0);
  ok('fig 4 prints the three dataset headers',
    byTag(fig4, 'div').some(function (n) {
      return n.getAttribute('class') === 'lqp-cols' &&
        n.textContent === 'SemanticKITTInuScenesWaymo';
    }));
  ok('the x-axis label is printed once, under the last row',
    byTag(fig4, 'p').filter(function (p) {
      return p.getAttribute('class') === 'lqp-xlab' &&
        p.textContent === J4.figure_texts[0].text;
    }).length === 1);
  ok('every panel tags the figure it belongs to',
    P3.every(function (p) { return /^Fig\. 3[ab]$/.test(p.getAttribute('data-figure')); }) &&
    P4.every(function (p) { return p.getAttribute('data-figure') === 'Fig. 4'; }));

  /* ----------------------------------------------------------- captions -- */
  head('CAPTIONS AND NOTES');
  var note3 = byTag(fig3, 'figcaption')[0].textContent;
  var note4 = byTag(fig4, 'figcaption')[0].textContent;
  ok('fig 3 note states the per-dataset frame counts from the JSON',
    Object.keys(J3A.protocol.frames.per_dataset).every(function (k) {
      return note3.indexOf(J3A.protocol.frames.per_dataset[k] + ' on ' + k) >= 0;
    }), note3.slice(0, 120));
  ok('fig 3 note names the field and the operator',
    note3.indexOf('TPV feature plane xy') >= 0 && note3.indexOf('voxel logits') >= 0 &&
    note3.indexOf('grid Laplacian') >= 0 && note3.indexOf('support-subgraph Laplacian') >= 0);
  ok('fig 3 note names the noise-ceiling denominator',
    note3.indexOf('re-truncation noise ceiling') >= 0);
  ok('fig 3 note states the support domain plainly, with no version history',
    note3.indexOf('teacher support \u03a9_T as a single domain') >= 0 &&
    !/earlier version|used to|previously|merged here/i.test(note3));
  ok('fig 4 note carries no version history either',
    !/earlier version|used to|previously|merged here/i.test(note4));
  ok('fig 3 note states the gap-closed formula with the ceiling at 1',
    note3.indexOf('(1 − CKA_LiSeer-P)') >= 0);
  ok('fig 4 note says "divided by the teacher’s own band energy"',
    note4.indexOf('divided by the teacher’s own band energy') >= 0);
  ok('fig 4 note does NOT say "divided by Dense teacher"',
    note4.indexOf('divided by Dense teacher') < 0);
  ok('fig 4 note explains the hatch as unsupported structure, not attenuation',
    note4.indexOf('white diagonals') >= 0 &&
    note4.indexOf('absent from the observation rather than attenuated') >= 0);
  ok('fig 4 note explains the broken bar and says the true value is printed',
    note4.indexOf('cut by two strokes') >= 0 &&
    note4.indexOf('true value is printed') >= 0);
  ok('fig 4 note explains the broken y axis of rows b and d',
    note4.indexOf('broken y axis') >= 0 &&
    note4.indexOf('zero sits at the same height') >= 0);
  ok('fig 4 note warns that the paired bars are not a stack',
    note4.indexOf('not a stack') >= 0);
  // One name per model (2026-11-14): no internal run name, checkpoint number
  // or hash reaches the page, and no note implies a second LiSeer-P.
  ok('no note names an internal run or checkpoint',
    !/LiSeer[_-](?:SK|NU|WA)|curmap|retrain|\b[0-9a-f]{7,}\b|checkpoints:|LiEqualizer-P/
      .test(note3 + ' ' + note4), note4.slice(-180));
  ok('the data files carry no internal model names either',
    !/LiSeer[_-](?:SK|NU|WA)|curmap|retrain/
      .test(JSON.stringify(J3A) + JSON.stringify(J3B) + JSON.stringify(J4)));
  ok('fig 4 protocol no longer carries a models block',
    J4.protocol.models === undefined && J3A.protocol.models === undefined);
  ok('no note still claims an epoch-21 / epoch-24 mismatch',
    note3.indexOf('epoch-21') < 0 && note4.indexOf('epoch-21') < 0);
  ok('no note still explains a dagger flag that is no longer drawn',
    note4.indexOf('†') < 0 &&
    JSON.stringify(J4.panels).indexOf('flagged_bands') < 0);

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
  var wantCell = Number(J3A.panels[0].series[3].q50[0]).toPrecision(4)
    .replace(/0+$/, '').replace(/\.$/, '');
  ok('fig 3a table prints the JSON value for LiEqualizer band 0',
    row0.childNodes[1].textContent === wantCell,
    row0.childNodes[1].textContent + ' vs ' + wantCell);
  line('table cell check  LiEqualizer band 0 median ' +
    J3A.panels[0].series[3].q50[0] + ' -> "' +
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
    ok('fig 4 still draws all twelve panels at 360px', narrow.length === 12,
      'got ' + narrow.length);
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
      ok('fig 3a alone draws its six panels', panelsOf(rr[0]).length === 6,
        'got ' + panelsOf(rr[0]).length);
      ok('fig 3b alone draws its six panels', panelsOf(rr[1]).length === 6,
        'got ' + panelsOf(rr[1]).length);
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
            var g = groupNamed(panelsOf(figD)[0], 'LiEqualizer, energy distribution');
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
