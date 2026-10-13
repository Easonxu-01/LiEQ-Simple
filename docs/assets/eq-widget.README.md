# `eq-widget.js` — the hero spectral equaliser

A self-contained interactive instrument for the LiEqualizer project page. Two vertical
faders, **LF EQUALIZER** and **HF EQUALIZER**, continuously morph the sparse student's
spectrum toward the dense teacher's — LF moves the low-frequency *envelope*, HF moves the
high-frequency *detail*. Both rest at 0, which is the undistilled student.

One file. Native inline SVG plus one injected `<style>`. No framework, no bundler, no build
step, no CDN, no chart library. ES5-compatible; safe to serve straight from GitHub Pages.

---

## 1. Integration

The page already carries everything it needs. For reference, this is the whole contract:

```html
<!-- in the hero's right-hand grid column -->
<figure class="instrument" id="eq-instrument">
  <div id="eq-mount"></div>
  <figcaption>
    <b>Drag the two faders.</b> The <span class="k-orange">sparse student's spectrum</span> is
    pulled toward the <span class="k-blue">dense teacher's</span>. <b>LF</b> restores the shape
    sampling weakened; <b>HF</b> removes the detail sampling faked. Both rest at zero — that is
    the undistilled student.
  </figcaption>
</figure>

<!-- before </body> -->
<script src="assets/eq-widget.js" defer></script>
```

That is all. On `DOMContentLoaded` the script mounts itself into every
`[data-lieq-equalizer]` element, and — if there are none — into `#eq-mount`. Nothing else to
call.

To mount it somewhere else, or to pass options:

```html
<div data-lieq-equalizer></div>
```

```js
var eq = LiEQEqualizer.init('#eq-mount', {
  lf: 0, hf: 0,
  xLabel: 'Graph frequency',
  onChange: function (lf, hf) { /* ... */ }
});
```

### What it injects

Into the mount element:

```html
<div class="lieq-shell">            <!-- or .lieq-card when chrome:true -->
  <div class="lieq-stage">
    <i class="lieq-scan"></i>       <!-- the 'powered on' scan sweep -->
    <svg class="lieq-svg" viewBox="0 0 470 280" role="group" aria-label="...">…</svg>
  </div>
</div>
```

Plus one `<style id="lieq-equalizer-css">` in `<head>`, once per document however many
widgets are on the page. Every CSS class and every SVG `id` is prefixed `lieq`, and the
`<defs>` ids are additionally suffixed with a per-instance counter, so two widgets on one
page cannot collide.

### The card around it

`chrome` defaults to `false` because **the page's own `.instrument` rule already paints the
dark card** — border-radius 25px, the two-layer `#16223a → #0c1322` background, the 34px
masked graph-paper `::before`, the figcaption hairline. The widget only adds the scan line
the sibling page had and the page's `.instrument` does not.

Pass `chrome: true` to get a standalone `.lieq-card` with the full shell (two-layer
background, four-part box-shadow, graph paper) for use outside the hero.

---

## 2. Options

Every option is optional. Strings are HTML-escaped before they reach the markup.

| option | default | what it does |
|---|---|---|
| `lf`, `hf` | `0`, `0` | initial fader positions, 0–1 |
| `chrome` | `false` | `true` also paints the dark card shell (see above) |
| `smoothRadius` | `3` | box-blur radius for the envelope/detail split |
| `smoothPasses` | `3` | how many blur passes |
| `xLabel` | `'Graph frequency'` | x-axis title |
| `yLabel` | `'Normalized spectral energy'` | rotated y-axis title |
| `lowTick`, `highTick` | `'LOW'`, `'HIGH'` | the two x-axis end ticks |
| `teacherLabel` | `'DENSE TEACHER'` | blue legend entry |
| `studentLabel` | `'SPARSE STUDENT'` | orange legend entry |
| `rackLabel` | `'SPECTRAL EQ'` | eyebrow above the faders |
| `readoutLabel` | `'PEAK GAP'` | label for the live readout |
| `lfAnnotation` | `['LOW-FREQUENCY','EXCESS']` | two lines; fades out as LF rises |
| `hfAnnotation` | `['SPURIOUS HIGH-FREQUENCY','RESPONSE']` | two lines; fades out as HF rises |
| `lfAria`, `hfAria` | see source | `aria-label` on each fader |
| `ariaLabel`, `desc` | see source | the `<svg>` name and its `<desc>` |
| `teacher`, `student` | built-in | override the curves: an array, or a comma-separated string, of y values on a uniform x grid. Both are resampled to the shorter length; 4 samples minimum. |
| `onChange` | `null` | `function (lf, hf, instance)`, called on every render |

Anything a caller throws inside `onChange` is swallowed, so a bad callback cannot break the
instrument.

## 3. Instance API

```js
var eq = LiEQEqualizer.init('#eq-mount');

eq.getValues()            // -> { lf: 0, hf: 0 }
eq.setValues(0.4, 0.8)    // set both; omit one with null
eq.setValue('lf', 0.4)    // set one
eq.morph(lf, hf)          // -> Array, the curve that pose would draw (pure, no DOM)
eq.render()               // force a synchronous redraw
eq.destroy()              // unbind everything and empty the mount
eq.element  eq.svg  eq.state  eq.teacher  eq.student
```

Module level:

```js
LiEQEqualizer.version                 // '1.0.0'
LiEQEqualizer.init(target, opts)      // -> instance | null   (never throws)
LiEQEqualizer.initAll(selector, opts) // -> [instance]
LiEQEqualizer.curves                  // { teacher: [...], student: [...] }
LiEQEqualizer.math                    // { clamp lerp smooth split morph catmullPath parseCurve }
LiEQEqualizer.geometry                // the viewBox layout constants
LiEQEqualizer.css                     // the stylesheet, as a string
LiEQEqualizer.renderMarkup(opts, uid) // the SVG source, no DOM needed
LiEQEqualizer.defaults
```

`init` returns `null` rather than throwing when the target is missing, is not an element, or
when the curves are unusable. The file also loads cleanly in node with no DOM at all — it
touches `document` only from the auto-init block, which is guarded.

---

## 4. How the morph works

Both curves are sampled on one shared, uniform x grid of 97 points. At init each is split
into a low-frequency envelope and a high-frequency residual, using three passes of a
radius-3 box moving average with clamped edges:

```
lo = smooth(y, 3, 3)
hi = y - lo                    // so lo + hi === y, exactly
```

Every frame the drawn curve is two independent linear interpolations:

```
y[i] = lerp(studentLo[i], teacherLo[i], lf) + lerp(studentHi[i], teacherHi[i], hf)
```

There is no cross term and no easing, so:

* the response is **continuous (无极)** — every intermediate fader position gives a
  proportional intermediate curve, and a step of `d` in `lf` moves sample `i` by exactly
  `d · (teacherLo[i] − studentLo[i])`;
* it is **monotone** in each fader at every sample;
* the four corners are exact — `(0,0)` is the student, `(1,1)` is the teacher, `(1,0)` is
  the teacher's envelope carrying the student's own wiggle, `(0,1)` the reverse;
* because the family is separately linear in each fader, its extremes over the whole unit
  square are attained at those four corners. The vertical scale is derived from exactly
  that, so no pose can ever overflow the frame.

The result is drawn as a uniform Catmull-Rom spline converted to cubic Béziers (one segment
per interval, end tangents clamped), with coordinates quantised to 0.1 user units. Render is
coalesced into one `requestAnimationFrame` per burst of pointer events.

**Releasing a fader does nothing.** There is no spring-back, no reset, no settle animation —
the value *is* the state. This is the one behaviour deliberately not copied from the sibling
page's optics bench, which always sprang home.

### Where the curves come from

Traced column by column off the left half of the paper's insight figure — the teacher from
the top-left panel (rescaled by the 1.046 factor that least-squares-fits the dashed teacher
drawn in the student's own axes, 2.5px RMS over 82 observed columns), the student from the
bottom-left panel. The student additionally carries a small decaying ripple over its right
half: the paper's *unsupported / spurious high-frequency response*, which is the thing the
HF fader exists to remove. Measured on the shipped arrays: the student's envelope runs
0.118 above the teacher's over the low third, and carries 1.49× the teacher's
high-frequency residual energy over the right half.

**These curves are illustrative.** They are a trace of a schematic figure, not measured
data, and the `PEAK GAP` readout is simply the largest vertical distance between the two
drawn curves. Nothing here should be quoted as a result.

---

## 5. Design

Colours, fonts and the card chrome follow the sibling LiSeer page, so the two read as one
family.

| | |
|---|---|
| teacher / reference | `#56adf0` core, `#bfe0ff` flow overlay, `#9cc8f3` label |
| student | `#e58236` core, `#f6bd8c` flow overlay, `#eab089` annotations |
| on-dark text | `#dbe7f5`, dimmed `#8d99ab`, focus `#58b9eb` |
| card | `#0c1322` base with a `#16223a` gradient terminus, radius 25px, padding `clamp(18px,2vw,25px)` |
| type | everything is `var(--mono)` — this is instrument chrome, not body copy |

Both curves use the sibling's **beam idiom**: a fat opaque core plus a thin bright dashed
"flow" overlay marching with `steps(18,end)`. The teacher additionally carries a soft
9-unit halo at 0.24 opacity, so when the two curves coincide at `lf=hf=1` you see the orange
line sitting inside a blue sheath rather than the blue simply vanishing. The student's core
and flow brighten from 0.62/0.75 toward full opacity as it converges, reusing the sibling's
"suppressed versus clean" opacity language.

The two annotations are live. `LOW-FREQUENCY EXCESS` is drawn as a **dimension bar** between
the two curves at the sample where the student's envelope most exceeds the teacher's; it
shrinks as you raise LF and fades out. `SPURIOUS HIGH-FREQUENCY RESPONSE` drops a leader
onto the live curve inside the ripple and fades out as you raise HF.

### Rest pose

`lf=0, hf=0` is composed to read as a finished static figure from the paper, because that is
what most visitors will see: legend, axes, both annotated defects, the gap readout, and both
faders at the bottom of their travel.

### Layout

viewBox `0 0 470 280` (aspect 1.679). Plot `48..334 × 46..214`, rack `352..462 × 26..252`,
faders at x 379 and 435 with 134 units of travel. Everything sits inside an 8-unit viewBox
inset. A `@media (max-width:760px)` block enlarges the type in user units so the labels stay
legible when the card shrinks to phone width; every size in it is collision-checked by the
test harness at both sizes.

---

## 6. Interaction and accessibility

* **Pointer Events throughout**, with `setPointerCapture` / `releasePointerCapture`, a
  `pointercancel` handler, and `pointerId` filtering so a second finger cannot hijack a
  drag. `touch-action:none` on each fader stops the page scrolling mid-drag. Move/up/cancel
  are also bound at the window so a drag survives both the pointer leaving the SVG and a
  browser that refuses capture on SVG elements.
* **Press the track** to jump the cap there; **press the cap** to drag it from where you
  grabbed it, with the offset preserved so it never jumps under your finger.
* **Touch targets** are 50 × 156 user units — about 34 × 106 CSS px on a 390px phone.
* **Each fader is a real `role="slider"`** with `tabindex="0"`, `aria-orientation="vertical"`,
  `aria-valuemin="0"`, `aria-valuemax="100"`, and live `aria-valuenow` / `aria-valuetext`.
* **Keyboard**: `↑ ↓ ← →` step ±0.05, with <kbd>Shift</kbd> ±0.2; `PageUp`/`PageDown` ±0.2;
  `Home` 0; `End` 1. Unknown keys fall through untouched.
* **Focus** is shown by a drawn ring element, not only a CSS `outline` (which several engines
  will not paint on SVG), progressively narrowed to `:focus-visible` where supported.
* The `<svg>` is `role="group"` with an `aria-label` and a long `<desc>` — *not* `role="img"`,
  which would have made the whole subtree presentational and the sliders unreachable. All
  static decoration is inside one `aria-hidden` group; the faders are deliberately outside it.
* `prefers-reduced-motion: reduce` stops the flow dashes, the scan sweep and the cap
  transition. There is no other motion: once you stop moving, the JS is completely quiet.

---

## 7. Tests

```
node docs/assets/eq-widget.test.js
```

84 assertions, no DOM, no framework, no dependencies. Covers the morph algebra (the four
corner identities, linearity, monotonicity, the corner bound on the drawn range), the
Catmull-Rom path, the fader travel and its pixel↔value round trip, label collisions at both
type sizes — read out of the exported stylesheet so they cannot drift from what the browser
paints — markup balance, the accessibility attributes, and the no-throw paths.

The envelope/detail claim is checked two ways. The exact one is the corner identity: at
`lf=1, hf=0` the output *is* `teacherLo + studentHi`, to 1.1e-16. The cross-check then takes
the output back apart with two instruments the widget does not itself use — a wider moving
average for the envelope, a second difference for the ripple — because re-running the
widget's own box blur over its own output would measure the blur as much as the result (a
box blur is not a projection: `S(S(x)) ≠ S(x)`).
