# charts-paper.js — the manuscript's Figure 3 and Figure 4, as live inline SVG

The two spectral figures from the paper body, redrawn by hand as inline SVG. One file, no
build step, no framework, no bundler, no CDN, no chart library. Vanilla ES5 syntax, so it
parses and runs in anything that can run the rest of the page.

| | |
|---|---|
| File | `docs/assets/charts-paper.js` (58 KB raw, 17 KB gzipped, 1 589 lines) |
| Tests | `docs/assets/charts-paper.test.js` — `node docs/assets/charts-paper.test.js` |
| Of which data | none — the three JSON files are fetched at runtime (see §7) |
| Global | `window.LiEQPaperCharts` (also merged into `window.LiEQCharts`, see §1) |
| Dependencies | none. `fetch`/`XMLHttpRequest` and `Promise` are feature-detected |
| Renders | Fig. 3 = 2 rows × 3 panels · Fig. 4 = 3 rows × 2 panels |

---

## 1. Integration

### The short way (recommended)

Give each mount a `data-lieq-paper-chart` attribute and load the script. It renders on
`DOMContentLoaded` by itself.

```html
<figure class="fig reveal">
  <div class="fig-media">
    <div id="lq-paper-fig3" data-lieq-paper-chart="paper-fig3"></div>
  </div>
  <figcaption><b>Fig 3.</b> Alignment, band by band. Agreement between the student's field
  and the teacher's within each band, scaled so that one is the agreement the teacher
  reaches with itself when its input is re-drawn under the same budget. Higher is better in
  all three columns.</figcaption>
</figure>

<figure class="fig reveal">
  <div class="fig-media">
    <div id="lq-paper-fig4" data-lieq-paper-chart="paper-fig4"></div>
  </div>
  <figcaption><b>Fig 4.</b> How much energy a band carries, and how much of it the teacher
  accounts for.</figcaption>
</figure>

<script src="assets/charts-paper.js" defer></script>
```

The mount ids are yours; the module keys off the attribute, not the id. The ids above
(`lq-paper-fig3`, `lq-paper-fig4`) follow the page's existing `lq-*` naming and are what the
explicit API below expects.

The four keys are:

| key | draws |
|---|---|
| `paper-fig3` | both rows: (a) feature planes and (b) voxel logits, 6 panels |
| `paper-fig3a` | row (a) alone, 3 panels |
| `paper-fig3b` | row (b) alone, 3 panels |
| `paper-fig4` | panels a–f, 6 panels |

`data-lieq-chart="paper-fig3"` works too: the sibling `charts-native.js` ignores keys it does
not own, and so does this module, so the two can share the attribute safely.

### The explicit way

```js
LiEQPaperCharts.renderFig3('lq-paper-fig3');
LiEQPaperCharts.renderFig4(document.getElementById('lq-paper-fig4'), { theme: 'dark' });
```

Every render function takes `(elementOrId, options)` and returns a **Promise** that resolves
to the figure element, or to `null` if the mount was missing or the data could not be loaded.
It never rejects and never throws.

```js
Promise.all([
  LiEQPaperCharts.renderFig3('lq-paper-fig3'),
  LiEQPaperCharts.renderFig4('lq-paper-fig4')
]).then(function (figs) { /* both are drawn */ });
```

### Options (second argument, all optional)

| option | default | meaning |
|---|---|---|
| `width` | measured | force the layout width in px instead of measuring the mount |
| `fallbackWidth` | `900` | used when the mount measures 0 (hidden tab, `display:none`) |
| `theme` | inherited | `'light'` or `'dark'`; otherwise read from the nearest `data-lqp-theme` / `data-lqc-theme` / `data-lq-theme` / `data-theme` ancestor |
| `watch` | `true` | re-draw on resize (ResizeObserver, debounced 180 ms) |
| `force` | `false` | redraw even when width and theme have not changed |

### Where it fetches from

The data path is taken from the script's own `src` at load time: `assets/charts-paper.js`
gives `assets/data/`. Override it before rendering if your layout differs:

```js
LiEQPaperCharts.dataBase = '/LiEQ-Simple/assets/data/';
```

### Living next to the other chart modules

`charts-main.js` and `charts-native.js` both publish `window.LiEQCharts`. This module
publishes its own `window.LiEQPaperCharts` and then fills into `window.LiEQCharts` only the
keys nobody has defined yet, so whichever script loads last leaves all three reachable. Its
CSS is namespaced `.lqp-*` with its own `<style id="lqp-style">`, injected once.

---

## 2. Tokens it consumes

The module reads the page's tokens and falls back to its own values when they are absent, so
it also works in isolation:

| token | used for | fallback |
|---|---|---|
| `--sans` | notes, body copy | `'Inter', system-ui, sans-serif` |
| `--mono` | every piece of chrome: eyebrows, ticks, legends, buttons, tables | `'JetBrains Mono', ui-monospace` |
| `--text` | ink | `#17202c` |
| `--muted` | ticks, notes, axis labels | `#647184` |
| `--line` | hairlines, table rules, button border | `rgba(52,72,96,.12)` |

All chrome is mono, as on the rest of the page. Nothing else is read, and nothing global is
written.

### Dark instrument card

Put the mount inside an element carrying `data-lqp-theme="dark"` (or pass `theme: 'dark'`)
and the figure flips to the card palette: ink `#dbe7f5`, dim `#9cc8f3`, links `#56adf0`,
hover `#bfe0ff`, hairlines at 18 % opacity, white marker faces become the card's own
background so hollow markers stay hollow.

---

## 3. The colour rule

**The series colours are the paper's.** They are read from the JSON, never hard-coded here:

| role | colour | where |
|---|---|---|
| LiEqualizer / after distillation | `#2a78d6` | lines, gap bars (observed by both), real energy |
| LiEqualizer light | `#86b6ef` | gap bars (teacher only), pseudo energy, Δ pseudo |
| LiSeer / before distillation | `#eb6834` | lines, real energy |
| LiSeer light | `#f5b896` | pseudo energy |
| Dense teacher / reference | `#52514e` | reference lines, the grey re-truncation marks |
| noise ceiling / re-truncated | `#8f8d88` | the `= 1` ceiling line in Fig. 3 |
| high-frequency shading | `#f3f2ef` | the λ ≥ .60 strip |

On the dark card those literals are swapped for the page's own on-dark siblings (`#2a78d6` →
`#56adf0`, `#eb6834` → `#e58236`, `#86b6ef` → `#bfe0ff`, `#f5b896` → `#f6bd8c`, the two greys
→ `#b7c4d4` / `#93a3b6`, the HF strip → a 7 % light wash). `#52514e` on `#0c1322` is
invisible, so this is a legibility fix, not a restyle: the hue order and the role of each
colour are unchanged. The map lives in `DARK_MAP` near the top of the file.

---

## 4. What the panels draw

### Fig. 3 — alignment, band by band

Two rows, each of three panels, labelled with the published figure's own sub-captions,
`(a) on the feature planes` and `(b) on the voxel logits`.

| panel | content |
|---|---|
| left, *Observed by both ↑* | CKA to the Dense teacher ÷ the noise ceiling, on Ω_S ∩ Ω_T |
| middle, *Observed by teacher only ↑* | the same on Ω_T \ Ω_S; **shares the left panel's y axis** |
| right, *Gap to teacher closed by LiEQ* | the share of the LiSeer-to-ceiling gap LiEqualizer closes, two bars per band |

Each curve draws its median as a line with markers and its interquartile range as a 14 %
ribbon. The `= 1` ceiling is a flat reference line, true by construction. The right panel
carries its y ticks on the **right**, as published, and its own two-swatch legend inside the
plot.

### Fig. 4 — energy per band

Three rows of two; the left column is the TPV feature plane, the right the voxel logits, and
the panel titles keep their `a`–`f` letters, as published.

| panel | content |
|---|---|
| a, b | log₂ of the band energy share relative to the Dense teacher (0 = ×1), median + IQR ribbon |
| c, d | the real / pseudo split, as two stacked bars per model, plus the grey re-truncation marks and the dashed `= 1` line |
| e, f | the paired LiEqualizer − LiSeer change in each part, as two **unstacked** bars per band around a zero line |

### The four series kinds

| kind | drawn as |
|---|---|
| `line` | `q50` as the path, `q25`–`q75` as a filled ribbon, one marker and one hover target per band. `markers: "hollow"` gives a hollow marker (teacher, sparse input) |
| `bar` | one rect per band, centred on the series' own `x` (which already carries the dodge offset), `width` in data units. A `bottom` array stacks it on the segment below; negative values hang below the baseline |
| `hline` | one flat reference line across the plot; `style: "dashed"` is honoured |
| `segments` | **one short horizontal mark per band**, from `x0[k]` to `x1[k]` at `y[k]` — not a line, not a bar. This is Fig. 4 c/d's "what the teacher itself still accounts for when its input is re-truncated" |

Every series group in the DOM carries `data-series="<name>"` and `data-kind="<kind>"`, and
every panel `<svg>` carries `data-panel`, `data-figure`, `data-plot="x0,x1,y0,y1"`,
`data-xlim` and `data-ylim`. That is what the verification harness reads, and it makes the
rendered figure inspectable in devtools.

### From the JSON, and honoured

* the 11 bands and their lower λ edges; x tick labels only on bands 0, 2, 4, 7, 10
* the high-frequency region: shading from x = 6.5 to 10.5 (clamped to the axis), a dotted
  edge at its start, and the `high frequency (λ ≥ .60)` annotation
* every annotation, in `data` or `axes_fraction` coordinates, with its point offset, its
  horizontal and vertical alignment and its line breaks — including the `†` null flags
* the shared-y marking: the middle panel of Fig. 3 prints no y tick labels, and its plot box
  is given the same gutter as the leftmost panel so the two are directly comparable
* y-axis labels keep their literal `\n` as two rotated lines, the first nearest the plot, as
  matplotlib stacks them
* the published figure legends, entry for entry, in order; Fig. 4's two legends sit above the
  rows they describe

---

## 5. The figure notes

Each figure carries a `<figcaption>` stating the measurement basis, built from the JSON's own
`protocol` block: the frame count and selection, the field, and the operator. Two sentences in
it are deliberate and should not be "tidied away":

1. **Fig. 4 c–f.** The note says the quantities are **"divided by the teacher's own band
   energy"**, with the teacher's input re-truncated under a different 500 000-point seed. It
   must not say "divided by Dense teacher": the denominator is the re-truncated teacher's
   band energy, not the dense teacher's. The panels' own axis labels, which come from the
   JSON and are drawn verbatim, abbreviate it as `÷ Dense teacher`; the note says so
   explicitly rather than leaving the two in silent contradiction.

2. **The epoch.** The model labelled *LiSeer* — before distillation — is the **epoch-21**
   checkpoint, which `protocol.models[].checkpoint` states in all three files, while the
   manuscript's scalar readings elsewhere are read from **epoch 24**. Both notes say this in
   one sentence. It is a real discrepancy in the record; the figure is honest about it rather
   than hiding it.

---

## 6. Accessibility, interaction, responsiveness

* every panel is `role="img"` with its own `<title>` and `<desc>`, wired through
  `aria-labelledby`
* every figure ships a **visually hidden data table** (`.lqp-sr`), one per row for Fig. 3 and
  one for Fig. 4, with `scope`'d row and column headers: 1 + 11 columns, the 11 graph-frequency
  bands by lower λ edge, values at 4 significant digits. Line series contribute a median row
  and an IQR row; stacked bar series are listed as segment heights with their parent named
* a `Show data table` / `Hide data table` button toggles it into view, with
  `aria-expanded` and `aria-controls` — the same contract as `charts-native.js`
* tooltips work with pointer **and** touch (`pointerdown`, `touchstart`), hide on scroll, and
  share one window listener no matter how many figures are on the page
* `prefers-reduced-motion: reduce` drops the entrance transition entirely
* the grid collapses 3 → 2 → 1 columns for Fig. 3 (at 880 px and 620 px) and 2 → 1 for Fig. 4
  (at 720 px). Verified at a true 360 px phone width: no panel escapes its container, no text
  escapes its panel, no page-level horizontal scroll
* a missing mount, a 404, a rejected fetch or a browser without `Promise` all end in the same
  place: a short note in the mount naming the file it could not read. Nothing throws

---

## 7. Data provenance

The module embeds **no data**. It fetches, at runtime, relative to its own location:

| file | figure |
|---|---|
| `data/fig3a_band_cka_feature_planes.json` | Fig. 3 row (a) |
| `data/fig3b_band_cka_voxel_logits.json` | Fig. 3 row (b) |
| `data/fig4_energy_real.json` | Fig. 4 |

These are the files the engineer who produced the published matplotlib figures exported; they
hold the exact values those figures draw, and they remain the single source of truth. Re-export
them and the page changes with no edit here. Each file is fetched once per page and shared
between mounts; a failed fetch is not cached, so a later render retries.

Because they are fetched, **the page must be served over http(s)**. Opened from `file://` the
browser blocks the read and each figure shows its fallback note. `python3 -m http.server` from
`docs/` is enough for a local look.

---

## 8. Verification

```
node --check docs/assets/charts-paper.js
node docs/assets/charts-paper.test.js
```

The test file sits next to the module, the way `eq-widget.test.js` sits next to
`eq-widget.js`: plain node, no jsdom, no framework, no network, runnable from any directory.
It renders both figures into a hand-rolled DOM with `fetch` reading the real JSON off disk,
then checks **552 assertions**, of which the substantive ones are:

* **Structure.** Fig. 3 draws 6 panels in 2 rows, Fig. 4 draws 6 in 3 rows, in the order
  a, b, c, d, e, f; Fig. 4's titles keep their letters and Fig. 3's carry none; each panel
  renders exactly as many series groups as its JSON panel declares, each with the declared
  `kind`.
* **Position.** For every panel, the harness reads the panel's advertised axis range, checks
  it against the JSON's `lim`, recomputes the expected pixel for **every drawn value** and
  compares it with the markup: 1 506 coordinates in all — median points, both ribbon edges,
  bar x / y / width / height, stack bases, segment ends, reference lines and the shading
  bounds. Worst disagreement: **0.005 px**, which is the module's own 2-decimal rounding.
* **Spot checks printed by the run.** Fig. 3a left, LiEqualizer band 0: median 0.921059 on
  the range 0…1.22 → y = 71.13 px in a plot spanning 28…204 px. Fig. 3a gap bar band 0:
  48.492387 % on 0…60 → y = 61.76 px, height 142.24 px with zero at 204 px. Fig. 4c segment
  band 5: 0.900432 → y = 104.74 px, spanning x 4.58…5.42 → 288.33…325.67 px. Fig. 4e pseudo
  band 0: −0.170997 → a bar hanging from y = 115.27 px (zero) to 145.12 px.
* **Shared y.** The leftmost panel of each Fig. 3 row prints `0.00`…`1.00`; the middle panel
  prints none of them; the gap panel prints its own percent ticks on the right; the y label
  keeps its line break as two tspans.
* **Captions.** The frame count and selection, the field and the operator appear; Fig. 4's
  note contains "divided by the teacher's own band energy" and does **not** contain "divided
  by Dense teacher"; both notes carry the epoch-21 / epoch-24 sentence, and the JSON is
  re-read to confirm epoch 21 is what the data file says.
* **Accessibility.** `role="img"` + title + desc per panel; the table starts hidden; the
  button toggles `aria-expanded`, the class and its own label; headers are 1 + 11 and every
  row has 12 cells with a `scope="row"` header; the LiEqualizer band-0 cell reads `0.9211`.
* **Resilience.** Missing mount → `null`; 404 → the fallback note naming the file; rejected
  fetch → the same; reduced motion → no animation class; at 360 px all six panels still draw,
  every panel keeps its y ticks and gains the column name as a subtitle.
* **Fetch.** Two mounts needing the same file produce one request; the URL is
  `assets/data/…`; dark theme maps `#2a78d6` to `#56adf0`.

Separately, the figures were rendered in headless Chrome at 1280 px, 768 px and a forced
360 px box. Measured there: 12 panels, 0 texts outside their panel box, 0 collisions between
the rotated y label and the tick labels, 0 overlapping x tick labels, no page-level horizontal
scroll. The published PDF's own text layer was extracted and compared label for label: the
legends, panel titles, tick labels, the `high frequency (λ ≥ .60)` annotations, the
`(a) on the feature planes` / `(b) on the voxel logits` sub-captions and the `voxel logits ·
199 frames` tag all match what this module draws.

---

## 9. Where it departs from the published figure, and why

1. **One legend for Fig. 3, not one per row.** The PDF repeats the four-entry legend above
   each row; the web figure states it once at the top, where it is always on screen.
2. **Fig. 4 e/f gain a legend.** The PDF relies on the in-panel `real ↑` / `pseudo ↓`
   annotations, which are drawn here too; a two-swatch legend was added above the row because
   the bars are the same blues as row c/d but mean something different.
3. **Type is re-set for screen.** The PDF's 6.3–8 pt sizes are not reproduced; chrome uses
   the sibling module's scale (9.5 px ticks, 10.5 px titles, 8.4 px annotations) so the two
   modules read as one page.
4. **Shared y axes are restored when the grid collapses.** At one column a panel with no y
   labels is unreadable, so the middle panel of each Fig. 3 row prints the leader's tick
   labels there. At two or three columns the JSON's `shared_with_panel` marking is obeyed
   exactly.
5. **The HF shading is `#f3f2ef` in light mode and a light wash in dark mode.** The paper's
   near-white would glare on the dark card.
6. **The x-axis label sits once under each row**, as in the PDF; when the grid collapses to
   one column it ends up under the last panel of the row rather than under each.
7. **`data-*` hooks were added** to the SVG (`data-panel`, `data-plot`, `data-series`, …).
   They are invisible, and they are what makes the figure verifiable.

---

## 10. For the page author

* The page's results section already calls two other figures **Fig 3** and **Fig 4** in its
  captions (the source-to-unseen matrix and the gain strip). These are the manuscript's
  Fig. 3 and Fig. 4. The two numbering schemes now collide — renumber one of them before
  publishing.
* Both figures are tall: Fig. 4 is six panels. Consider giving them their own section, or
  using `paper-fig3a` / `paper-fig3b` to place the two rows apart.
* If the page is served from a project sub-path and you load the script with an absolute URL,
  the data path follows the script automatically; only set `LiEQPaperCharts.dataBase` if you
  move the JSON somewhere else.
