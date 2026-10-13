# charts-native.js — spectral diagnostic charts, native inline SVG

Six figures from the paper's appendix, re-drawn as inline SVG by hand. One file, no
build step, no framework, no bundler, no CDN, no chart library, no network requests at
runtime. Vanilla ES5 — it parses and runs in anything that can run the rest of the page.

| | |
|---|---|
| File | `docs/assets/charts-native.js` (110 KB raw, 29 KB gzipped, 2 770 lines) |
| Of which data | ~50 KB — the float64 values embedded verbatim, so the page never fetches anything |
| Global | `window.LiEQSpectralCharts` (also merged into `window.LiEQCharts`, see §1) |
| Dependencies | none |

---

## 1. Integration

### The short way (recommended)

Give each mount a `data-lieq-chart` attribute and load the script. It renders everything
on `DOMContentLoaded` by itself.

```html
<section class="figure-block">
  <p class="eyebrow">Appendix D.1</p>
  <h3>Multi-scale diffusion profiles</h3>
  <div data-lieq-chart="qprofiles"></div>
</section>

<div data-lieq-chart="rolloff-hfshare"></div>
<div data-lieq-chart="class-retention"></div>
<div data-lieq-chart="gap-heatmap"></div>
<div data-lieq-chart="b0-trend"></div>
<div data-lieq-chart="band-profiles"></div>

<script src="assets/charts-native.js" defer></script>
```

The six keys are exactly: `qprofiles`, `rolloff-hfshare`, `class-retention`,
`gap-heatmap`, `b0-trend`, `band-profiles`.

### The explicit way

```html
<div id="fig-qprofiles"></div>
<div id="fig-rolloff"></div>
<div id="fig-retention"></div>
<div id="fig-heatmap"></div>
<div id="fig-b0"></div>
<div id="fig-bands"></div>

<script src="assets/charts-native.js" defer></script>
<script>
  document.addEventListener('DOMContentLoaded', function () {
    var C = window.LiEQSpectralCharts;
    C.renderQProfiles(document.getElementById('fig-qprofiles'));
    C.renderRolloffHfShare(document.getElementById('fig-rolloff'));
    C.renderClassRetention(document.getElementById('fig-retention'));
    C.renderGapHeatmap(document.getElementById('fig-heatmap'));
    C.renderB0Trend(document.getElementById('fig-b0'));
    C.renderBandProfiles(document.getElementById('fig-bands'));
  });
</script>
```

Each render function takes an element **or** an id string (`'fig-b0'` / `'#fig-b0'`), and
returns the figure element it built, or `null`. A missing or wrong element is never fatal:
the function returns `null` and logs one `console.warn`. Nothing throws, ever, so a chart
that is not on the page cannot take the page down with it.

The mount should be an empty `<div>`. Everything inside it is replaced on each render.
No stylesheet to include: the module injects one `<style id="lqc-style">` the first time
it runs, and never again.

### Options (second argument, all optional)

| option | default | meaning |
|---|---|---|
| `width` | measured | overrides the measured container width (useful for tests) |
| `fallbackWidth` | `900` | used when the container measures 0 (hidden tab, print, SSR) |
| `theme` | inherited | `'dark'` forces the on-dark palette; otherwise taken from a `data-lqc-theme` / `data-theme` attribute on the mount or any ancestor |
| `watch` | `true` | `false` disables the resize re-render |
| `force` | `false` | re-render even when the layout bucket has not changed |
| `barLabel` | `'printed'` | **class-retention only.** `'median'` prints the median above each bar instead of the published `max(p75, median, HFR^(c))`, and says so in the figure note. See §6. |

Other API: `renderAll({ 'qprofiles': el, ... }, opts)`, `autoInit(opts)`, `data` (the whole
embedded dataset), `noteText(key)`, `keys`, `version`.

### Living next to `charts-main.js`

The sibling model-comparison module on this page publishes `window.LiEQCharts` and has its
own `renderAll`. This file therefore owns **`window.LiEQSpectralCharts`** and, as a
convenience, copies its keys into `window.LiEQCharts` **only where that object has nothing
of that name yet** — it never overwrites a sibling's function, whichever script loads first.
Use `LiEQSpectralCharts` in page code and the two modules cannot collide. Everything else is
already namespaced apart: CSS classes `.lqc-*` (theirs are `.lq-*`), the injected
`<style id="lqc-style">` (theirs is `lq-charts-main-css`), and the auto-init attribute
`data-lieq-chart` (theirs is `data-lq-chart`).

---

## 2. Tokens it consumes

The module never hard-codes page colour or type. It reads the page's custom properties
with a fallback baked in, so it looks right even if it is dropped onto a bare page:

```
--text   #17202c    all chart ink
--muted  #647184    ticks, legend, figure note
--line   rgba(52,72,96,.12)   hairlines, table rules, button border
--sans   'Inter', system-ui, sans-serif        → figure notes (body copy)
--mono   'JetBrains Mono', ui-monospace, …     → ticks, legend, panel titles,
                                                 axis line, bar labels, buttons, tables
```

`--disp` (Space Grotesk) is deliberately **not** used: the module renders no `h1`/`h2`/`h3`,
so headings stay with the page. `--paper`, `--mist`, `--r`, `--shadow` and `--maxw` belong to
the card the chart sits in, not to the chart; the chart is transparent and fills its container
(capped at 1180 px of design width, matching `--maxw`).

Derived variables, all overridable per chart by setting them on the mount:

```
--lqc-ink --lqc-muted --lqc-hair --lqc-grid --lqc-axis
--lqc-edge --lqc-whisk --lqc-mark --lqc-halo
--lqc-eyebrow --lqc-link --lqc-hover
```

### Dark instrument card

Put the chart inside the dark widget and mark it (`data-lqc-theme`, `data-lq-theme` and
plain `data-theme` are all accepted, on the mount or on any ancestor):

```html
<div class="instrument" data-lqc-theme="dark">
  <div data-lieq-chart="qprofiles"></div>
</div>
```

The ink flips to `#dbe7f5` / `#9cc8f3`, grid and axis to translucent light, label halos to
`#0c1322`, the eyebrow and links to `#9cc8f3` / `#56adf0` / `#bfe0ff`. Data colours do not
change — see the next section.

---

## 3. The colour rule (do not override it)

In these six figures **colour encodes the dataset**, and dense vs sparse is encoded by
fill and line style, never by colour. That is the published mapping and the page must
match it:

| | |
|---|---|
| SemanticKITTI | `#4477AA` |
| Waymo | `#EE6677` |
| nuScenes | `#228833` |
| dense | solid fill, alpha 0.92, edge `#555555` — or solid line with round markers |
| sparse | **same dataset colour**, alpha 0.38 with `///` hatching — or dashed line with square markers |
| classes (b0-trend only) | matplotlib tab10: `#1f77b4 #ff7f0e #2ca02c #d62728 #9467bd #8c564b #e377c2` |
| gap heat map | RdBu_r, symmetric about 0, vmax 0.3820365948699273, red = dense excess |
| HFR diamonds / parity lines | black (near-white on the dark card) |

The brand blue `#2a78d6` and brand orange `#eb6834` apply to the *model* comparison charts
elsewhere on the page and appear nowhere in these six. There is a regression assertion for
exactly that.

Hatching is a real SVG `<pattern>` (vertical rules on a 5 px tile, `patternTransform:
rotate(45)`), one per dataset per panel, so it survives zoom and prints correctly.

---

## 4. What each chart draws

Every chart is built the same way:

```
<div class="lqc" data-chart="…">        ← positioning context for the tooltip
  <div class="lqc-legend">…</div>       ← mono, wraps on narrow screens
  <p class="lqc-axes">x: … · y: …</p>   ← axis names and units, once, in text
  <figure class="lqc-fig">
    <div class="lqc-panels">            ← CSS grid, 1–3 columns
      <svg role="img" aria-labelledby="…"><title>…</title><desc>…</desc>…</svg>
      …one <svg> per panel…
    </div>
    <figcaption class="lqc-note">…measurement basis…</figcaption>
  </figure>
  <div class="lqc-foot"><button class="lqc-toggle">Show data table</button></div>
  <div class="lqc-sr"><table>…every plotted value…</table></div>
  <div class="lqc-tip"></div>
</div>
```

**1. `renderQProfiles` — multi-scale diffusion profiles.** Three panels (density,
normal_z, surface_variation), six lines each: three datasets × dense (solid, circles) /
sparse (dashed, squares). **The x axis is the diffusion step t ∈ {1, 4, 16, 32}, not b0** —
equally spaced categorical ticks, as in the original. Linear y, nice domain [0.15, 0.35].
Each sparse series carries its DRD-JS mean (`{:.4f}`) in the tooltip and data table.

**2. `renderRolloffHfShare` — roll-off and true HF share.** Two panels: λ-85 roll-off
(median over frames) and the true high-frequency share P(λ ≥ 0.6) (mean over frames).
Each panel is three signal groups × three datasets × a dense/sparse pair = 18 bars.
Logarithmic y, from the decade below the smallest value up to max × 2.2 (the headroom
the original uses). Bar labels are rotated 90° at 3 significant digits, exactly the
published format, and are dropped automatically when a bar is narrower than 9 px.

**3. `renderClassRetention` — per-class retention.** Seven class groups × three datasets,
bar offsets ±0.26 and bar width 0.2392 of the group pitch, straight from the original
geometry. Bar height = **median**; whiskers = IQR p25–p75 with caps; black diamonds =
the independent full-spectrum estimate HFR^(c); `n=…` printed inside the foot of each bar;
black dashed parity line at 1; y axis 0 → 1.22 × max(p75, median, diamond) = 4.4036.
See §6 for the printed label. Below 560 px the whole chart flips to horizontal bars with
the class names on the left — the only way 21 bars and their labels stay readable at 360 px.

**4. `renderGapHeatmap` — dense-minus-sparse band gap.** One heat map per dataset, rows =
3 global signals + 7 classes, columns = the **merged 10 bands** (`merged_10_columns`), which
is what the published figure draws. `full_11_bands` is reference only and is not shipped in
the embedded data. Cells are RdBu_r, symmetric about 0 at vmax 0.382; cell text is G × 1000;
white text above 0.55 · vmax; NaN would print `N/A` in `#666` (there are no NaN cells in this
run). A gradient colour bar with ±382 ticks sits in the legend row.

**5. `renderB0Trend` — retention vs diffusion cutoff.** One panel per dataset, seven class
lines each, log-2 x axis at b0 ∈ {2, 8, 16, 32} (so the 2→8 gap is twice the 8→16 gap, as
on a log-2 axis it must be), shared linear y across the three panels, parity line at 1.
Colour here means **class**, not dataset; the dataset only colours the panel title.

**6. `renderBandProfiles` — 11-band energy profiles.** Nine panels (3 datasets × 3 signals),
11 bands × dense/sparse per panel. Logarithmic y, decade domain shared per dataset row
([1e-4, 1] for all three rows in this run). The dotted vertical rule sits at λ = 0.03: the
first two bands lie inside the Jackson kernel soft edge.

### Figure notes

Every chart carries a note under it, built from `caption_basis` in the source JSON, never
hand-written: frames per dataset and how they were selected, what dense and sparse mean,
the statistic for that figure, the spectrum, the label mapping and class convention, plus
that figure's own caveat. If the engineer re-exports the data with a different basis, the
notes change with it.

---

## 5. Accessibility, interaction, responsiveness

* **`role="img"` + `<title>` + `<desc>`** on every panel SVG, wired with `aria-labelledby`.
  The title names the panel; the description says what is plotted and over how many frames.
* **A visually-hidden data table per chart**, always in the DOM for screen readers, with a
  mono **Show data table** button (`aria-expanded`, `aria-controls`) that reveals it for
  everyone else. Tables carry every plotted value at the figure's own precision, plus the
  values that are not drawn (medians, IQRs, per-row n, DRD-JS, sparse/dense ratios).
* **Tooltips** work with mouse, pen and touch: `pointermove` + `pointerdown` where
  `PointerEvent` exists, `mousemove` + `touchstart` otherwise, both through one delegated
  listener per chart (no per-mark handlers). One shared `scroll` handler dismisses them,
  registered once for the whole module, so repeated re-renders cannot pile up listeners.
* **`prefers-reduced-motion`** is respected twice over: the only animation is a 0.45 s
  fade-in of the panels, it is not even attached when `matchMedia` reports a preference for
  reduced motion, and a `@media (prefers-reduced-motion: reduce)` block in the injected CSS
  neutralises every transition regardless.
* **Readable at 360 px.** The module measures the container and lays out in real pixels, so
  tick text is ~9.5 px at every width rather than a viewBox shrunk to illegibility. Panels
  reflow 3 → 2 → 1 column (roll-off 2 → 1; heat map 3 → 2 → 1), class retention flips to
  horizontal bars under 560 px, heat-map cell numbers drop to whole numbers under 30 px
  cells and disappear under 22 px (the colour, the tooltip and the table still carry them),
  and dense band/column labels thin out to every other label.
* **Re-renders on resize** via `ResizeObserver` (falling back to `window.resize`), debounced
  180 ms, and only when the layout bucket or a 40 px width quantum actually changed.
* Browser storage is not used at all; nothing is persisted, nothing is sent anywhere.

---

## 6. The class-retention label — read this one

In the published figure the number printed above each bar is **not** the median. It is
`max(p75, median, HFR^(c))` — the JSON ships it as `printed_label_value`. This module
reproduces the figure faithfully and prints `printed_label_value`, while the median is
always available three other ways:

* in the **tooltip** (`median R`, the IQR, `HFR^(c)`, the printed label and `n`, all at once),
* in the **data table**, which has separate `Printed label` and `Median R` columns,
* in the **figure note**, which states in so many words that the printed number is
  `max(p75, median, HFR^(c))` and not the median.

The bar *height* has always been the median — there is a regression assertion that the bar
top maps the median and is more than 5 px away from where the printed label would put it.

If LOCAL decides the web version should print the median instead, pass
`{ barLabel: 'median' }`; the note then says the rendering departs from the published figure.

---

## 7. Data provenance

Values come from
`results/S4/APPX_D_native_web_data_20261013/fig_native_data.json`, produced by
`export_web_data.py` from the run
`APPX_D_native_finalmap_20261012` (S4, 2026-10-12 21:02 → 10-13 00:17 SGT, exit 0),
200 frames per dataset, summary md5s `35b0185d…` / `c11aff1c…` / `f7ac07d9…`.

They are embedded as JS number literals written from Python `repr()`, i.e. the shortest
representation that round-trips, so the browser parses back bit-identical float64 — no
rounding happened anywhere between the figure script and the page. Display formats are
applied at draw time, per the format declared in the JSON for each figure
(`{:.3f}`, `{:.3g}`, `{:.4f}`, `G × 1000 {:.1f}`).

Not embedded, on purpose: `full_11_bands` (reference only, never drawn), and the
`caption_basis` fields that are identical across all six figures are hoisted into one
`meta` block (verified identical before hoisting; only `statistic` and `note` differ per
figure).

**To refresh after a new export**, the `DATA = {…}` literal at the top of the file is the only
thing that changes. It was generated mechanically from the JSON — dict order preserved,
numeric arrays inline, floats via `repr()` — so it can be regenerated the same way and
pasted back, or edited in place; nothing else in the file reads the JSON.

---

## 8. Verification

`node --check docs/assets/charts-native.js` → clean.

A headless harness (minimal DOM shim + SVG serializer, since no jsdom is available here)
renders all six charts at 1100 px and again at 360 px and asserts **217 checks, 0 failures**
(plus a separate coexistence probe):

* **Series counts per chart** — 3 panels × 6 polylines and 72 markers (q-profiles);
  2 panels × 18 bars with 18 hatched sparse bars (roll-off); 21 bars, 21 diamonds,
  63 whisker segments, 21 `n=` labels (retention); 3 × 10 × 10 = 300 cells (heat map);
  3 × 7 = 21 lines and 84 markers (b0-trend); 9 panels × 22 bars with 99 hatched
  (band profiles); and the matching table row counts 18 / 18 / 21 / 30 / 21 / 18.
* **Spot values at the declared precision** — `0.31645` and DRD-JS `0.0031`; roll-off
  `0.455` / `0.00965` / `0.106` / `0.0335`; retention printed `0.537` *and* median `0.411`,
  p25 `0.331`, diamond `0.417`, `n=7`; heat map `-356.8`; b0-trend `0.4518` / `0.4231`;
  bands `0.4682` / `0.8762`.
* **Formatting** — the `{:.Ng}` implementation was compared against Python's `%g` over
  **805 values** actually printed by the charts: 0 mismatches.
* **Geometry arithmetic** — the four y coordinates of a q-profiles line recomputed from the
  nice domain; the roll-off bar top recomputed on the log scale and its foot on the
  baseline; the retention bar top = median (and demonstrably *not* the printed label),
  bar width = 0.2392 × pitch, dataset offset = 0.26 × pitch, foot at zero; the heat-map cell
  fill = RdBu_r at `(G+vmax)/2vmax` with zero mapping to `rgb(247,247,247)`; the band-profile
  bar on a per-row decade domain [1e-4, 1]; log-2 spacing on the b0 axis (2→8 is exactly
  twice 8→16).
* **Coexistence** — with a sibling `window.LiEQCharts` pre-seeded, its `renderAll`,
  `version` and render functions survive intact while this module's are merged alongside,
  and `window.LiEQSpectralCharts` is this module either way.
* **Contract** — `<title>`/`<desc>`/`aria-labelledby` on every panel; the hidden table and
  the toggle present; the note carrying frames, statistic and label mapping; no `NaN`,
  `undefined` or `Infinity` anywhere in the markup; no brand blue/orange in any of the six;
  dataset colours used for 6 polylines each in q-profiles (colour = dataset, not side);
  `null` / unknown id / junk object mounts return `null` without throwing; re-render is
  idempotent; dark theme attribute honoured; one stylesheet injected.

Rendered in Chrome (headless) at 1240 px and at a true 360 px column, all six charts plus
the dark-card variant, and checked by eye. Tooltips and the table toggle were driven with
real `PointerEvent`s in that browser: the tooltip opens with the right text (including the
median alongside the printed label), positions itself, and closes on leave; the toggle flips
`aria-expanded`, the class and the button label.
