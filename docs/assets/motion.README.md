# `motion.js` + `motion.css` — the figure motion layer

Two files, no dependencies, no build step. They sit **on top of** the figures that
`charts-main.js`, `charts-paper.js` and `charts-native.js` draw and add exactly two
things:

1. **an on-scroll reveal** — bars grow from their own baseline, lines draw along their
   length, ribbons / heat-map cells / markers fade and settle, table rows stagger in;
2. **pointer emphasis** — the mark under the pointer keeps its paint, its siblings
   step back, and the figure's own tooltip does the talking.

They never draw, re-number, re-colour or re-label anything. **This layer writes no
words of its own**: the one readout it can show is the chart module's own tooltip
text, re-rendered as plain text. It cannot introduce a caption, a label or a claim.

| | |
|---|---|
| Files | `docs/assets/motion.js` (1 214 lines), `docs/assets/motion.css` (159 lines) |
| Global | `window.LiEQMotion` |
| Dependencies | none. `IntersectionObserver`, `MutationObserver`, `PointerEvent` and `matchMedia` are all feature-detected |
| Touches | `figure.lq-fig[data-lq-chart]`, `div.lqp[data-chart]`, `div.lqc[data-chart]` |
| Never touches | `<defs>` content, the visually hidden data tables, the insight equalizer (`.lieq-card` / `.lieq-shell`), axes, gridlines, ticks, parity lines, titles, annotations |

---

## 1. Integration

```html
<!-- in <head>, after the font link -->
<link rel="stylesheet" href="assets/motion.css" />

<!-- in <body>, AFTER the three chart modules -->
<script src="assets/charts-main.js" defer></script>
<script src="assets/charts-native.js" defer></script>
<script src="assets/charts-paper.js" defer></script>
<script src="assets/motion.js" defer></script>
```

That is the whole integration. No element needs a new id, a new class or a new data
attribute. The script auto-initialises on `DOMContentLoaded`, and because
`charts-paper.js` and `charts-native.js` fetch their data and render later, it also
watches the document and wires each figure as it appears.

**Order matters only in one direction:** `motion.js` must come *after* the chart
modules so their tooltips are wired first. Loading it before them still works — the
figures are picked up by the document watcher — but then the first pointer emphasis
on a `charts-main` figure may show this layer's own readout instead of that module's
tooltip for one hover.

`motion.css` on its own does nothing. Every rule in it is keyed on an attribute that
only `motion.js` writes, so **with the stylesheet loaded and the script missing,
broken or disabled, every figure renders exactly as the chart modules drew it.** The
reveal's start state is applied by JS, never by CSS. Keep it that way.

---

## 2. Forcing the finished state (headless capture)

The page is captured with headless Chrome for screenshots and video frames. Any one
of these gives every figure its finished state with no animation at all:

| how | when to use it |
|---|---|
| `<html data-lieq-motion="off">` (also `"instant"`, `"none"`) | a capture build of the page |
| `window.LIEQ_MOTION_INSTANT = true;` before the script runs | the same, from script |
| `?motion=off` (also `motion=none` / `motion=0` / `motion=instant`) in the URL | **the easy one for a capture pipeline** — nothing to edit |
| `LiEQMotion.finishAll()` | from the automation, at any moment, just before the shot |
| `navigator.webdriver === true` | automatic: Puppeteer / Playwright / WebDriver are detected. Override with `data-lieq-motion="on"` |
| `prefers-reduced-motion: reduce` | the user's own setting, always obeyed |

Two things to know about capture:

* **A figure below the fold is deliberately left in its start state** until it is
  scrolled into view — that is what a scroll reveal is. A capture that stitches or
  uses `captureBeyondViewport` without scrolling must therefore use one of the
  switches above. A plain `--screenshot` only captures the viewport, so it is not
  affected, and Puppeteer / Playwright are detected automatically.
* **`--virtual-time-budget` freezes CSS transitions at their start value.** This layer
  is immune: its settle step is driven by a timer, not by a frame, and it cancels
  every transition before it drops its own styles (§6). It also cancels, and so
  repairs, the panel fade that `charts-paper.js` and `charts-native.js` start
  themselves, which has the same problem and can otherwise leave Fig 6 and Fig 7
  **blank** — that happens with this layer absent too, and is worth fixing in those
  modules. Because that fade can start *after* this layer has settled, the repair is
  checked once more `repairAfter` ms later, and again whenever `finishAll()` is
  called — which is why `LiEQMotion.finishAll()` immediately before the shot is the
  most reliable capture recipe of the lot. The check reads a computed opacity and
  does nothing unless something is actually wrong, so it costs nothing in a browser
  with a running clock.

---

## 3. The knobs

Set them **before** `DOMContentLoaded` — the simplest way is a global, read at init:

```html
<script>window.LIEQ_MOTION = { sweep: 160, block: 400, dim: .28 };</script>
<script src="assets/motion.js" defer></script>
```

or afterwards through `LiEQMotion.settings.*`, which affects figures wired from then
on (`threshold` and `rootMargin` are read once, when the observer is created).

| knob | default | what it does |
|---|---|---|
| `startDelay` | `110` | ms before anything moves, so the page's own `.reveal` fade is under way |
| `panelStagger` | `64` | ms between the panels of one figure |
| `panelStaggerCap` | `5` | panels after this one share the last panel's delay, so a nine-panel figure does not crawl |
| `sweep` | `200` | ms the left-to-right (or top-down) sweep takes across one panel |
| `block` | `460` | ms for a bar to grow |
| `draw` | `620` | ms for a line to draw |
| `fade` | `300` | ms for ribbons, cells, whiskers |
| `point` | `240` | ms for markers and diamonds |
| `value` / `valueLag` | `220` / `170` | the printed numbers: how long they fade, and how far they trail their own mark |
| `row` / `rowStagger` | `260` / `34` | table rows |
| `maxMarks` | `600` | above this many marks a panel reveals as one piece instead of mark by mark |
| `panelFade` | `420` | ms for that whole-panel fallback |
| `ease` / `easeDraw` | `cubic-bezier(.22,.61,.36,1)` / `(.25,.6,.3,1)` | easing out, no overshoot |
| `threshold` / `rootMargin` | `0.12` / `0px 0px -8% 0px` | when a figure counts as "in view" |
| `watchdog` | `2200` | ms before a figure that nothing has observed is dealt with (§6) |
| `repairAfter` | `900` | ms after settling, when the one look-and-repair pass over the panels runs (§6) |
| `dim` | `null` | overrides `--lqm-dim`; `null` leaves it to `motion.css` |

`motion.css` carries four custom properties you can restyle without touching JS:
`--lqm-dim` (how far siblings step back), `--lqm-ring`, `--lqm-chip`,
`--lqm-chip-ink`.

---

## 4. What it does to each kind of mark

The three modules label their marks three different ways — and `charts-native.js`
does not label them at all — so the layer classifies by tag, class and paint, and
pairs marks to hit targets geometrically. Nothing is guessed from an index.

| mark | recognised as | reveal |
|---|---|---|
| `rect` / `polygon` with paint (bars, heat-map cells, bar outlines, cell separators) | `block` | grows from the panel's baseline, or fades if the panel has no baseline |
| `path` with `fill="none"`, a stroke and no dash (median lines) | `stroke` | draws along its length (`stroke-dasharray` / `stroke-dashoffset`) |
| any other `path` (ribbons, diamonds, dashed lines), and `line` (whiskers, segments, flat references) | `area` | fades and settles |
| `circle` / `ellipse` (markers) | `point` | fades, after the line it sits on has drawn |
| `text.lq-val`, `text.t-val`, `text.t-n`, `text.t-cell` | `value` | fades in just behind its own mark |
| `table.lq-table tbody tr` | `row` | staggered fade. Opacity only — a transform on a table row is not reliable, and a row that failed to come back would hide a number |

**Where the baseline comes from.** The layer tallies the four edges of a panel's
rectangles. If 55 % or more of them share one edge, that edge is the baseline and the
blocks scale about it — which is why bars that hang *below* a zero line collapse onto
that same line and grow downwards, correctly. If no edge is shared, there is no
baseline (a heat map, a scatter of cells) and the blocks fade instead. Panels whose
bars are stacked may fall below the 55 % line and fade; that is intended, not a bug.

**Everything else is furniture** — axes, gridlines, ticks, parity and `= 1` reference
rules, panel titles, axis labels, annotations, legends, the hatched diagonal backdrop
— and is never touched, so the frame of the chart is already there while the data
arrives.

---

## 5. Hover and focus

* On `pointerover` of a hit target, that hit's own marks are emphasised and the rest
  of the **panel** steps back to `--lqm-dim` (0.34). Lines and ribbons do not dim:
  dimming the line a hovered marker sits on reads as a mistake, not as emphasis.
* **The figure's own tooltip is the readout.** All three modules already carry one,
  wired to the same hit targets and built from their own numbers, so on a pointer
  this layer stays silent and lets them talk.
* The shared readout (`.lqm-readout`) appears only where there is no module tooltip,
  and on the keyboard path, where no pointer event ever fires. It renders the
  module's tip text as **text** — `data-tip`'s `||` separators and `<b>` / `<i>`
  markup are flattened, never inserted as markup.
* Emphasis is pointer- and focus-driven only. **Nothing is made focusable by
  default.** If a figure (or its mount) carries `data-lq-motion-keys`, that figure's
  hit targets get one tab stop and arrow-key traversal (`←` `→` `↑` `↓`, `Home`,
  `End`, `Escape`), each named with its own tooltip text. `Tab` and `Shift+Tab` are
  never intercepted, so focus is never trapped. It is off by default because every
  figure already ships a complete screen-reader data table.

---

## 6. The things that go wrong, and what is done about them

| hazard | what the layer does |
|---|---|
| A clip rect or a hatch pattern line is a `<rect>` / `<line>` like any other — scaling a clip rect would collapse the clip and take a whole bar layer with it | nothing inside `defs`, `clipPath`, `pattern`, `mask`, `marker`, `symbol`, `filter` or a gradient is ever touched |
| A figure this layer does not recognise (a fallback note, a 404 message, a chart drawn some other way) | the guard: with no recognisable marks it gets no attribute, no style and no listener |
| A module redraws on resize and throws its DOM away | a `MutationObserver` re-wires the new DOM; a figure that already played is not re-animated; the old entry, its timers and its observation are dropped |
| The tab is in the background, so no animation frames arrive | the settle timer is set when the reveal starts, not inside a frame callback, so the figure cannot be left half-drawn |
| A transition is still in flight when its declaration is removed, and the clock is not advancing (headless virtual time) | settle cancels every transition, drops the start state, forces one style resolution, then lets transitions back |
| Nothing ever reports an intersection (no `IntersectionObserver`, or a browser that never calls back) | a per-figure watchdog: if **no** intersection record of any kind has arrived by `watchdog` ms, the reveal is abandoned and every figure is handed its finished state |
| A print or a screenshot lands mid-reveal | `@media print` in `motion.css` forces `opacity: 1`, `transform: none`, no dash on every mark; an `!important` in a stylesheet beats this layer's inline styles |
| Reading geometry per mark would thrash layout | geometry comes from the elements' own attributes (`x`/`y`/`width`/`cx`/…), so the common path forces no layout at all; `getBBox()` is a fallback for paths only, and all reads happen before any write |

---

## 7. Verification

Everything below was run against these files, with the figures the three real modules
produce.

```
node --check docs/assets/motion.js                     syntax
node $S/motion.selftest.js                             124 assertions, 0 failures
node $S/motion.selectors.js                             77 / 77 selectors confirmed
                                                        to occur in the real modules
```

`$S` is the session scratchpad; the two scripts are throwaway harnesses, not part of
the page. The self-test renders figures whose markup mirrors, attribute for
attribute, what each module emits, and asserts: the `defs` guard, the furniture
exclusions, every mark classification, the armed state and its baselines, hit
pairing, the emphasis in and out, the readout's markup stripping, the focus path,
instant mode, reduced motion, `?motion=off`, automation detection, the missing
`IntersectionObserver` path, `finishAll()`, the no-intersection watchdog, the panel
repair pass (and that a healthy panel is left alone), re-render pruning, the opt-in
keyboard traversal — and that **no inline style survives the reveal** anywhere.

In real headless Chrome, against the real page:

```
settled DOM == the no-JS DOM, once motion.js's own attributes are stripped   yes
settled screenshot vs the same page with ?motion=off, byte for byte          identical
                                                      pixel diff bounding box: none
collapsed upright bar: bottom edge vs its final bottom edge                  delta +0.00 px
collapsed horizontal bar: left edge vs its final left edge                   delta +0.00 px
hover: hovered bar opacity 1, sibling bar and its value label 0.34, axis 1   as designed
document scrollWidth == clientWidth (no horizontal overflow)                 at 1280 and at 485
the whole page, three consecutive headless runs: 12 figures, 2 575 marks,
  all "done", 0 inline styles left, every panel at opacity 1, no console error
```

(The 485 is headless Chrome refusing to make its own window narrower than ~500 px —
a true 360 px check has to be run in an iframe, as `charts-main.README.md` §6 notes.)

---

## 8. Open questions for whoever owns `index.html`

1. The two tags in §1 are the whole integration, and they are already in
   `index.html` — the stylesheet right after the font link, the script right after
   `charts-paper.js`. That is exactly the wiring everything below was verified
   against. These two files add nothing else to the page.
2. `charts-paper.js` and `charts-native.js` add their own panel-fade class two
   animation frames after they build a figure. When those frames do not arrive the
   class never lands and the panels stay at `opacity: 0`; when they do arrive but the
   clock is frozen, the fade sticks at 0. Both are reproducible with this layer
   absent — on the pristine page, Fig 6 and Fig 7 captured blank in a plain headless
   run. This layer repairs both, but it is a courtesy and it is timing-dependent;
   the right fix is in those two modules: set the class on a timer as well as on a
   frame, and give the panels `opacity: 1` whenever the class is already there.
