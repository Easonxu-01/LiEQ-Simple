# `eq-widget.js` — the hero spectral equaliser

A self-contained interactive instrument for the LiEqualizer project page. Two vertical
faders, **LF EQUALIZER** and **HF EQUALIZER**, continuously morph the sparse student's
profile toward the dense teacher's — LF moves the low-frequency *envelope*, HF moves the
high-frequency *detail*. Both rest at 0, which is the undistilled student. Either curve can
also be **plucked** with the pointer: pull it anywhere and it springs back like a cord.
Three preset pills jump the faders to the three states of the paper's insight figure.

The plot is in the **spatial domain** — x is a spatial coordinate and y a signal/feature
value, exactly as the source figure's own axis is labelled. At full travel the student lands
*close to but not on* the teacher: a 4% residual is left, as the paper's own measurements
show.

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
  xLabel: 'Spatial position',
  onChange: function (lf, hf) { /* ... */ }
});
```

### What it injects

Into the mount element:

```html
<div class="lieq-shell">            <!-- or .lieq-card when chrome:true -->
  <div class="lieq-stage">
    <i class="lieq-scan"></i>       <!-- the 'powered on' scan sweep -->
    <svg class="lieq-svg" viewBox="0 0 480 248" role="group" aria-label="...">…</svg>
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
| `xLabel` | `'Spatial position'` | x-axis title — the source figure's own label |
| `yLabel` | `'Signal / feature response'` | rotated y-axis title |
| `lowTick`, `highTick` | `'0'`, `'X MAX'` | the two x-axis end ticks |
| `residual` | `0.04` | what full travel CANNOT recover, in signal units (see §4) |
| `scrubLabel` | `'GAP HERE'` | readout label while the scrub cursor is up |
| `hint` | `['PLUCK','A CURVE']` | two lines, top right; retires on the first pluck |
| `presets` | 3 pills | `[{label, lf, hf, aria}]` |
| `caption` | the disclosure line | see §4 |
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
LiEQEqualizer.version                 // '2.0.0'
LiEQEqualizer.init(target, opts)      // -> instance | null   (never throws)
LiEQEqualizer.initAll(selector, opts) // -> [instance]
LiEQEqualizer.caption                 // the illustrative-not-measured line (print it)
LiEQEqualizer.curves                  // { teacher: [...], student: [...] }
LiEQEqualizer.constants               // residual + pluck spring constants
LiEQEqualizer.math                    // { clamp lerp smooth split morph reach bump spring
                                      //   catmullPath parseCurve }
LiEQEqualizer.PluckState              // the DOM-free pluck/spring state machine
LiEQEqualizer.Instance                // the constructor (its prototype is testable)
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
y[i] = lerp(studentLo[i], teacherLo[i], lf · reach)
     + lerp(studentHi[i], teacherHi[i], hf · reach)
```

There is no cross term and no easing, so:

* the response is **continuous (无极)** — every intermediate fader position gives a
  proportional intermediate curve, and a step of `d` in `lf` moves sample `i` by exactly
  `d · (teacherLo[i] − studentLo[i])`;
* it is **monotone** in each fader at every sample;
* the four corners are exact — `(0,0)` is the student; `(1,0)` is the teacher's envelope
  carrying the student's own wiggle, `(0,1)` the reverse; `(1,1)` is the **limit curve**,
  not the teacher (next section);
* because the family is separately linear in each fader, its extremes over the whole unit
  square are attained at those four corners. The vertical scale is derived from exactly
  that, so no pose can ever overflow the frame.

The result is drawn as a uniform Catmull-Rom spline converted to cubic Béziers (one segment
per interval, end tangents clamped), with coordinates quantised to 0.1 user units. Render is
coalesced into one `requestAnimationFrame` per burst of pointer events.

### The residual: full travel does not reach the teacher

`FULL_SCALE_RESIDUAL = 0.04`. The faders travel over

```
reach = 1 − residual / max|student − teacher|          // 0.91389 on the shipped curves
```

of the student-to-teacher path, so the limit curve is

```
L = teacher + (1 − reach) · (student − teacher)        // teacher + 8.6% of the original error
```

and `max|L − teacher|` is *exactly* `0.04` — 4% of the plotted 0..1 range, 91.4% of the
original 0.4645 peak gap closed. The `PEAK GAP` readout therefore reads `0.04` at full
travel, never `0.00`. Because `reach` only shortens the path, the family stays exactly
linear, monotone and continuous in each fader.

This is both a design choice and an honesty one: the paper reports a distilled student
*close to* the dense teacher, never equal to it, and a widget whose two curves snapped into
one line would claim more than the paper does.

### Releasing a fader does nothing. Releasing a *curve* springs back.

Two different gestures, deliberately opposite:

| gesture | on release |
|---|---|
| **fader** (drag, press-the-track, keyboard, preset) | nothing. No spring-back, no reset, no settle. The value *is* the state. |
| **curve pluck** (drag either curve directly) | springs back to the shape the faders imply, with a short underdamped settle. |

A pluck cannot move a fader: the whole pluck path runs through `PluckState`, which has no
reference to `lf` or `hf`, and the three `Instance` methods that drive it
(`pluckBegin` / `pluckTo` / `pluckRelease`) never read or write `this.state`. Once the cord
settles, `PluckState` clears itself and the drawn array is bit-for-bit `morph(lf, hf)` again.

```
deform[i] += amp · ½(1 + cos(π · (i − i₀)/w))          // raised cosine, w = 0.14·(n−1) ≈ 13.4
amp(t)     = a₀ e^(−ζω₀t) [cos(ω_d t) + (ζω₀/ω_d) sin(ω_d t)]   // ζ = 0.34, 7.2 Hz
```

The bump is smooth, not a spike (the apex rises 1.4% of the pull per sample) and strictly
local (nothing outside ±13.4 samples moves). The spring is time-based, so a dropped frame
changes nothing; it overshoots 32% of the pull, crosses zero twice, and is below the
`0.0015` settle threshold — a quarter of a user unit on screen — after **192 ms**. Under
`prefers-reduced-motion` the pluck is simply dropped on release instead of animated.

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

The module says so out loud, in one line the page should print under the instrument:

```js
LiEQEqualizer.caption   // also eq.caption, and overridable as opts.caption
```

> Illustrative, not measured: both curves are traced from the paper's insight figure, and
> the student's high-frequency ripple is synthesised so the HF fader has something to remove.

The same sentence is appended to the `<svg>`'s `<desc>`, so a screen reader gets it too.

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

viewBox `0 0 480 248` (aspect **1.935**, was 1.679). Plot `46..346 × 30..200`, rack
`362..472 × 20..236`, faders at x 389 and 445 with 134 units of travel. Below the axis sit
the three preset pills (left) and the readout (right). Everything sits inside an 8-unit
viewBox inset.

**Footprint.** Measured in headless Chrome at a 1440px viewport, in the page's own hero
grid (`minmax(0,.78fr) minmax(560px,1.22fr)`), the card paints 639 px wide:

| | SVG | card |
|---|---|---|
| this widget at 470 × 280 | 587 × 350 | 639 × **480** |
| this widget at 480 × 248 | 587 × 303 | 639 × **433** |
| sibling LiSeer `.splitter` | 587 × 350 | 639 × **444** |

So the card lost 46 px of height (−10%) and now sits 11 px *under* the sibling's, which is
what "match the sibling's footprint" was asked for. The grid was not touched; only the
widget's own aspect was. A `@media (max-width:760px)` block enlarges the type in user units so the labels stay
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
* **Touch targets**: faders 50 × 156 user units — about 33 × 104 CSS px on a 390px phone.
  The pluck band is a 22-unit invisible stroke along each curve (15 CSS px on a phone, 27 in
  the desktop hero); the preset pills carry a 24-unit hit rect (16 / 29 CSS px). The pills
  are the smallest target on the card and are a convenience: everything they do is also on
  the faders, which are keyboard-reachable.
* **Plucking**: `pointerdown` on either curve's band grabs the nearest sample, `pointermove`
  pulls it, `pointerup` releases it to the spring. The gesture takes pointer capture, is
  filtered by `pointerId`, is cancelled by `pointercancel`, and sets `touch-action:none` on
  the band only — so the page still scrolls everywhere else on the card.
* **Scrub**: moving the pointer across the plot raises a dotted cursor with a dot on each
  curve and switches the readout from `PEAK GAP` to `GAP HERE` — the local distance between
  the two drawn curves at that sample.
* **Presets** are `role="button"` with `tabindex="0"`, `aria-pressed` reflecting whether the
  faders currently sit on that pose, Enter/Space keyboard activation and their own focus
  ring. They jump; they never rebound.
* **Each fader is a real `role="slider"`** with `tabindex="0"`, `aria-orientation="vertical"`,
  `aria-valuemin="0"`, `aria-valuemax="100"`, and live `aria-valuenow` / `aria-valuetext`.
* **Keyboard**: `↑ ↓ ← →` step ±0.05, with <kbd>Shift</kbd> ±0.2; `PageUp`/`PageDown` ±0.2;
  `Home` 0; `End` 1. Unknown keys fall through untouched.
* **Focus** is shown by a drawn ring element, not only a CSS `outline` (which several engines
  will not paint on SVG), progressively narrowed to `:focus-visible` where supported.
* The `<svg>` is `role="group"` with an `aria-label` and a long `<desc>` — *not* `role="img"`,
  which would have made the whole subtree presentational and the sliders unreachable. All
  static decoration is inside one `aria-hidden` group; the faders are deliberately outside it.
* `prefers-reduced-motion: reduce` stops the flow dashes, the scan sweep, the hint pulse and
  every transition, and makes a released pluck drop instead of spring. Once you stop moving,
  the JS is completely quiet.

---

## 7. Tests

```
node docs/assets/eq-widget.test.js
```

138 assertions, no DOM, no framework, no dependencies. Covers the morph algebra (the four
corner identities, linearity, monotonicity, the corner bound on the drawn range), the
residual (`max|c₁₁ − teacher| − 0.04 = 3.5e-17`), the pluck kernel and spring (deform,
locality, smoothness, overshoot, settle time, and that a settled pluck returns the array
*bit-for-bit*), the proof that `pluckBegin`/`pluckTo`/`pluckRelease` leave `lf` and `hf`
untouched, the Catmull-Rom path, the footprint, the fader travel and its pixel↔value round
trip, label collisions at both type sizes — read out of the exported stylesheet so they
cannot drift from what the browser paints — markup balance, the accessibility attributes,
the caption, and the no-throw paths.

A second, browser-side pass (headless Chrome, synthetic `PointerEvent`s against the real
handlers) checks the parts a node harness cannot see: that a curve drag rewrites the path
and leaves both `aria-valuenow`s alone, that the curve is byte-identical again after the
spring, that the faders stay where keyboard and pointer leave them, that the presets drive
both faders and mark themselves `aria-pressed`, that the scrub cursor and its readout come
up, and that `destroy()` empties the mount — 29 assertions, no uncaught errors.

The envelope/detail claim is checked two ways. The exact one is the corner identity: at
`lf=1, hf=0` the output *is* `teacherLo + studentHi`, to 1.1e-16. The cross-check then takes
the output back apart with two instruments the widget does not itself use — a wider moving
average for the envelope, a second difference for the ripple — because re-running the
widget's own box blur over its own output would measure the blur as much as the result (a
box blur is not a projection: `S(S(x)) ≠ S(x)`).
