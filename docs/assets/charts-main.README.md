# `charts-main.js` — the LiEqualizer result charts

The paper's **result** figures, re-created as live HTML and inline SVG rather than
embedded as bitmaps. Design artwork (`insight.jpg`, `framework.jpg`, `realdata.jpg`,
`LF.jpg`, `HF.jpg`, `LiSeer.jpg`, the real-vehicle photos) stays as images; the
numbers on this page are drawn by the browser.

One file, no build step, no framework, no bundler, no CDN. Vanilla ES5. Drop it in
with a `<script src>` and it works on GitHub Pages.

---

## 1. Integration

### 1.1 The page already loads the fonts

Nothing to add. The file only *uses* `--sans` / `--disp` / `--mono`; it never loads
a font or a stylesheet of its own.

### 1.2 Markup

Four mount elements. The ids below are the defaults, so with these the script
needs no init call at all:

```html
<section id="lq-main-matrix">
  <noscript><!-- paste the output of LiEQCharts.fallbackHTML('main') here --></noscript>
</section>

<section id="lq-gain-strip">
  <noscript><!-- LiEQCharts.fallbackHTML('gain') --></noscript>
</section>

<section id="lq-ablation">
  <noscript><!-- LiEQCharts.fallbackHTML('ablation') --></noscript>
</section>

<section id="lq-kd-baselines">
  <noscript><!-- LiEQCharts.fallbackHTML('kd') --></noscript>
</section>

<script src="assets/charts-main.js"></script>
```

That is the whole integration. The script auto-initialises on `DOMContentLoaded`
and fills whichever of the four mounts exist. A missing mount is a silent no-op.

There is **no stylesheet to include** — the script injects its own `<style>` once,
on first render, under the id `lq-charts-main-css`.

### 1.3 If you want to drive it yourself

Add `data-lq-no-autoinit` to the script tag and call the functions:

```html
<script src="assets/charts-main.js" data-lq-no-autoinit></script>
<script>
  LiEQCharts.renderMainTable('my-matrix-div');
  LiEQCharts.renderGainStrip('my-strip-div');
  LiEQCharts.renderAblation(document.querySelector('#abl'));
  LiEQCharts.renderKDBaselines('my-kd-div');
  // or, with custom ids:
  LiEQCharts.renderAll({ ids: { main:'a', gain:'b', ablation:'c', kd:'d' } });
</script>
```

Every renderer takes an **element or an element id**, returns the mount element,
and returns `null` instead of throwing when the element is missing.

### 1.4 Options

| option | values | effect |
|---|---|---|
| `theme` | `'light'` (default), `'dark'` | `'dark'` repaints the whole figure — SVG *and* the HTML copy, legend, table and footnotes — in the dark-card tokens. Use it for a chart placed on the `#0c1322` instrument card. |
| `bucket` | `'c'`, `'w'` | **`renderGainStrip` only.** Forces the compact or wide layout instead of measuring the container. |
| `ids` | object | **`renderAll` only.** Override the four default mount ids. |

```js
LiEQCharts.renderGainStrip('strip-on-the-dark-card', { theme: 'dark' });
```

### 1.5 `<noscript>`

`fallbackHTML(which)` returns plain, styled `<table>` markup carrying every number,
for `which` in `'main' | 'gain' | 'ablation' | 'kd'`. It is a build-time helper, not
a runtime one — **run it once and paste the result into the `<noscript>` blocks**:

```bash
node -e "var C=require('./docs/assets/charts-main.js');
         console.log(C.fallbackHTML('main'))"
```

(The same markup is what every chart already emits as its visually-hidden
screen-reader table, so the two can never disagree.)

---

## 2. What each function draws

### `renderMainTable(el, opts)` — Table 1, as a grouped bar chart

Three panels, one per training source, each with three groups (the three test
sets) of three bars (P-baseline, LiSeer-P-RSCL, LiEqualizer-P). 27 bars, every one
labelled with its mIoU.

* The y scale is **0–80 with ticks at 0 / 20 / 40 / 60 / 80**, the same ticks as the
  colour bar printed in the paper's Table 1 caption.
* **The diagonal is visually set apart**, because the paper's claim is about the
  off-diagonal cells: the same-sensor group gets a tinted backdrop *and* a 45°
  hatch *and* a `same sensor` tag under the axis, while the cross-sensor groups are
  left plain, full-strength and tagged `cross-sensor`. Three channels, so it never
  depends on colour alone.
* Layout: one column below 960px, three columns above. The SVG is capped at 420px
  so a single-column panel never scales its 8.4px axis type past ~1.4×. Verified
  legible at 360px and at 1180px.

### `renderGainStrip(el, opts)` — LiEqualizer over LiSeer, all nine cells

A sorted horizontal strip of the nine `LiEqualizer-P − LiSeer-P-RSCL` differences,
from **+17.0** down to **+0.9**, with the **cross-sensor maximum +11.5** called out
twice: as a hero stat above the strip (`+11.5 · largest cross-sensor gain · Waymo →
SemanticKITTI, 63.8 against 52.3`) and as an annotated row inside it.

The three same-sensor rows are drawn in the light brand blue with a hatch overlay
and named in the legend, so a reader cannot mistake `Waymo→Waymo +17.0` for the
cross-sensor claim.

**Two layouts.** SVG text does not have a CSS font size that survives a 3.6×
container range, so the strip measures its mount and picks a viewBox: compact
(330 wide, `WA→SK` labels, a footer legend for the two-letter codes) below 620px,
wide (640 wide, full dataset names, the callout inline) above. It re-renders only
when the bucket actually changes, via `ResizeObserver` where available and a
debounced `resize` listener otherwise.

### `renderAblation(el, opts)` — Table 3
### `renderKDBaselines(el, opts)` — Table 2

Real `<table>` markup styled with the page tokens. Read downwards from the
undistilled student; each bracketed figure is the difference from the first row.
The manuscript's **bold** (best in column) and <u>underline</u> (second best) are
preserved exactly, including on the bracketed deltas, which the tex marks together
with the value. Both tables carry their `\floatnote` verbatim.

---

## 3. Where the numbers come from

Everything is transcribed from `LiEqualizer_main.tex`:

| chart | source |
|---|---|
| `renderMainTable`, `renderGainStrip` | `\label{tab:ST results}`, tex lines 619–671 |
| `renderKDBaselines` | `\label{tab:kdbaselines}`, tex lines 720–735 |
| `renderAblation` | `\label{tab:ablation}`, tex lines 746–762 |

All of it lives in one place, `LiEQCharts.DATA`, and nothing is computed from
anything else at render time.

### 3.1 Two traps worth knowing about

**The deltas in Table 1 do not all mean the same thing.** The green figures on the
P-baseline rows are that row's collapse against *its own sensor* (nuScenes 19.4
against SemanticKITTI 71.6 = −52.2). The red figures on the LiSeer and LiEqualizer
rows are the gain over *P-baseline in the same cell*. The data carries a
`deltaRef` field per method for exactly this reason, and the tooltips label each
one with its own reference. Do not relabel them.

**The nine gains are written out, not computed.** `GAINS` holds `+11.5` as a
literal so the page can never drift from the table through floating-point
arithmetic. The verification script checks every one of them against
`LiEqualizer-P − LiSeer-P` in `MAIN`, so the two cannot silently disagree.

### 3.2 The one place this departs from the printed tex

Table 2's last row prints **`(+0.5)`** on the `SK→nuScenes` cell. But
`58.3 − 56.8 = 1.5`, the first row of that same column is `56.8`, the floatnote says
the bracketed figures are differences against that row, and **Table 3 prints
`(+1.5)` for the identical pair**.

The page shows **`(+1.5)`** and carries a numbered footnote under the table saying
that the manuscript prints `(+0.5)` there. Nothing is hidden. If you would rather
the page reproduce the typo, change the single `d: 1.5` in the `KD` object and drop
the `footnote: 1`.

**This is worth fixing in the manuscript too** — see the open question at the end.

Two smaller editorial calls, both flagged here rather than made silently:

* The tex row label is `+ \textbf{LiEqualizer} (ours)(ours)` — a duplicated
  `(ours)`. The page prints it once.
* The tex row label `+ both, Eq.~\eqref{eq:kdbaseline}` refers to a numbered
  equation. The page has no equation numbering, so it reads
  `+ both, the two-term pointwise form`.

---

## 4. Colours

Brand values are used exactly as given. The one addition is the P-baseline grey.

| role | light | dark card | source |
|---|---|---|---|
| LiEqualizer-P | `#2a78d6` | `#56adf0` | brand |
| LiSeer-P-RSCL | `#eb6834` | `#e58236` | brand |
| **P-baseline** | `#8f8d88` | `#8f8d88` | see below |
| same-sensor gain bar | `#86b6ef` | `#2a78d6` | brand light variant |
| gain delta (positive) | `#DC2626` | `#FCA5A5` | tex `gainRed` / `gainRedLight` |
| drop delta (negative) | `#15803D` | `#4ADE80` | tex `dropGreen` / `dropGreenLight` |

**On the P-baseline grey.** P-baseline is not one of the four named brand entities.
The paper's own 3D rendering of this table (`material/3Dchart_lieq.py`) paints it
`#a8a5a0`; this page uses `#8f8d88`, which is in the supplied token list, is one
step darker, and is the only one of the two that clears 3:1 contrast against the
`#f8fafc` page. `#52514e` was not reused because that token means *dense teacher*
elsewhere on the page.

**Palette validation.** Run against the `dataviz` validator as a 3-slot categorical
palette on the light surface:

```
[PASS] Lightness band      all 3 inside L 0.43–0.77
[FAIL] Chroma floor        #8f8d88 reads gray  (intended — see below)
[PASS] CVD separation      worst adjacent #eb6834 vs #8f8d88  ΔE 10.8 protan / 21.3 tritan
[PASS] Normal-vision floor worst adjacent ΔE 17.2
[PASS] Contrast vs surface all 3 >= 3:1
```

The chroma FAIL is deliberate and is the only one: P-baseline is the neutral
"no method" reference, and a grey is the correct encoding for it. The CVD margin
(10.8) clears the ΔE ≥ 8 target without needing the secondary encoding, and it has
one anyway — every bar is directly labelled with its value.

**On red-for-gain.** The manuscript colours gains red and drops green, which is the
reverse of the Western convention. The page follows the manuscript, but the sign
(`+` / `−`) is always printed, so colour is never the only channel, and a footnote
under each table says so explicitly.

---

## 5. Accessibility

* Every SVG is `role="img"` with a `<title>` and a `<desc>` wired through
  `aria-labelledby`. The `<desc>` is a prose summary that names every value, so a
  screen reader gets the whole chart from the summary alone.
* Every chart also carries a **visually-hidden `<table>`** with the exact numbers,
  wrapped in a clipping `<div>` (a bare `<table>` treats `width:1px` as a minimum
  and escapes `overflow:hidden` — this caused a real 1884px horizontal overflow at
  360px before it was fixed; there is now a regression check for it).
* Bold / underline marking in the two tables is **not visual-only**: each marked
  cell also carries hidden text, `(best in column)` or `(second best in column)`.
* `<caption>`, `scope="col"` and `scope="row"` throughout. The scrollable table
  wrapper is `tabindex="0"` with `role="region"`, so it can be scrolled from the
  keyboard.
* **Motion:** one 0.42s fade-in, declared inside
  `@media (prefers-reduced-motion: no-preference)`. Under reduced motion there is
  no animation at all, and the figure's base state is `opacity:1`, so it can never
  be stuck invisible if the animation does not run.
* **Tooltips are an extra, never a requirement.** Every number is already printed on
  the chart. Tooltips work on hover and on tap (tap elsewhere or wait 2.8s to
  dismiss) and they add only the manuscript's bracketed delta and its reference.

---

## 6. Verification

Three suites, all passing. Re-run them after any edit.

```
node --check docs/assets/charts-main.js
node       $S/verify-charts-main.js   # 108 assertions - markup, numbers, geometry
node       $S/verify-dom.js           #  16 assertions - mounting, against a DOM stub
python3 -I $S/verify-tex.py           #  13 assertions - SVG XML + a diff against the tex
```

The three scripts are **not part of the deployed page**. They live outside the repo,
in the session scratchpad:

```
S=/private/tmp/claude-501/-Users-eason-Documents-LiEqualizer/d1a88a99-a9f0-49ed-96c1-4bde53aff9a4/scratchpad
```

Copy them somewhere permanent if you want them in CI. They need only `node` and
`python3`; `verify-tex.py` additionally reads `LiEqualizer_main.tex` by absolute
path, so adjust that constant if the manuscript moves.

What is asserted:

* **Bar counts.** 27 `<rect class="lq-bar">` across the three panels, 9 per panel;
  9 `<rect class="lq-gbar">` in each gain-strip layout plus exactly 3 hatch
  overlays; 27 and 9 hit-target rects.
* **Every hard-coded number reaches the output.** A sweep over `DATA` builds a list
  of 144 figures — 27 Table 1 values, 24 Table 1 deltas, 9 gains, and every value
  and delta of Tables 2 and 3 — and checks each one appears in the rendered markup
  of all four charts. Each bar is additionally checked to carry the right
  `data-v` for its own method/test pair.
* **Against the tex, not against a summary.** A separate Python pass parses
  `\hmcell`, `\textbf` and `\underline` straight out of `LiEqualizer_main.tex` and
  diffs all 27 Table 1 values, all 24 deltas, and every value, delta and
  bold/underline mark of Tables 2 and 3 against `DATA`. The only permitted
  difference is the documented `(+0.5)` → `(+1.5)` one, which is asserted
  explicitly rather than merely tolerated.
* **Arithmetic.** All nine gains are checked to equal `LiEqualizer-P − LiSeer-P`;
  the cross-sensor maximum is checked to be `+11.5` at `Waymo → SemanticKITTI`.
* **Geometry.** Every bar is checked to sit inside the plot box and land on the
  clipped baseline; every value label in both gain-strip layouts is checked to fit
  inside its viewBox.
* **Structure.** All 8 rendered SVGs parse as well-formed XML; each has `role="img"`,
  a `viewBox`, and `<title>` + `<desc>` as its first two children with matching
  `aria-labelledby`; all 34 generated ids are unique so clip paths and hatch
  patterns cannot cross-contaminate; no SVG carries a fixed `width`.
* **No-throw contract.** `renderX(null)`, `renderX(undefined)`, `renderX('missing-id')`
  and `renderX({})` all return `null`.
* **DOM behaviour** (against a minimal DOM stub): all four mount, the stylesheet is
  injected exactly once across four renders, auto-init survives a page with no
  mounts, a 1180px mount picks the wide gain-strip layout and a 360px mount the
  compact one, and `theme:'dark'` reaches both the SVG and the HTML chrome.
* **Layout, in a real browser.** Rendered in headless Chrome and inspected at 1180px
  and at a true 360px (in an iframe — headless Chrome clamps its own window to
  500px, which will mislead you if you screenshot it directly).
  `document.scrollWidth === clientWidth` at 360 and 600, i.e. no horizontal
  overflow.

Two bugs were found and fixed this way, both invisible in the markup:

1. the visually-hidden tables blew the document out to 1884px at 360px;
2. the dark variant themed the SVG but not the surrounding HTML, so bold numbers
   in the caption rendered near-black on near-black.

---

## 7. Open questions for the author

1. **Table 2, last row, `SK→nuScenes`: `(+0.5)` or `(+1.5)`?** The page shows
   `(+1.5)` with a footnote. `58.3 − 56.8 = 1.5` and Table 3 prints `(+1.5)` for the
   same pair, so this looks like a typo in the manuscript rather than a different
   measurement — but it is your number, so please confirm, and fix it in the tex if
   it is indeed a typo.
2. **Table 2's caption** still ends `; numbers to be filled`. The page drops that
   parenthetical. The tex should probably drop it too.
3. **`(ours)(ours)`** is duplicated in the tex label of Table 2's last row.
4. **The `†` caveat.** The ablation prose says *"the rows marked † were trained with
   one sample per GPU and the last two with four"*, but no row in the tex actually
   carries a `†`. The page therefore does not mark any row, and the caveat is not
   shown. Tell me which rows should carry it and I will add the marker and the note.
5. **Table 2's row-label citations.** The page prints `[LiDAR Distillation]` and
   `[LESS; Hinton]` where the tex has `\cite{...}`. If the page will have a
   reference list, these should become links.
