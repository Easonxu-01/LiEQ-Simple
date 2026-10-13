# `eq-widget.js` — the hero spatial-profile equaliser

A self-contained interactive instrument for the LiEqualizer project page. Two vertical
faders, **LF / SHAPE** and **HF / DETAIL**, continuously morph the sparse student's profile
toward the dense teacher's. Both rest at 0, which is the undistilled student. Either curve
can also be **plucked** with the pointer: pull it anywhere and it springs back like a cord.
Three preset pills jump the faders to the three states of the paper's insight figure.

One file. Native inline SVG plus one injected `<style>`. No framework, no bundler, no build
step, no CDN, no chart library. ES5-compatible; safe to serve straight from GitHub Pages.

## The one fact that governs everything here

**The plot is in the SPATIAL domain.** x is a spatial position, y is a signal / feature
response — exactly as the source figure's own axis is labelled. The *height* of a curve
therefore carries **no frequency information whatsoever**, and nothing in this widget labels
a region of a curve with a frequency, a spectrum or a band. What the two words mean on a
spatial plot is what they mean in the paper's figure:

| | in the spatial domain |
|---|---|
| **low frequency** | the **envelope** — the gross shape of the curve |
| **high frequency** | the fine **spiky ripple** riding on top of that envelope |

So **LF moves the shape** and **HF grows the fine detail**. The LF/HF fader captions are
kept (they name the two distillation branches in the paper); every other label on the plot
is a plain spatial statement.

---

## What changed in version 3

Three defects in v2 are fixed. All three were visible on the live page.

### Defect 1 — the curve semantics were backwards

v2 synthesised a high-frequency **ripple on the student** and left the **teacher smooth**,
then had the HF fader *remove* that ripple. The paper's figure is the other way round: the
dense scan carries dense fine detail, the sparse scan lost it, and panel 3 shows the student
*acquiring* the teacher's texture. v3 rebuilds the model as four generated arrays:

```
teacherEnvelope   one dominant hump plus subordinate shoulders
teacherRipple     dense, sharp, varied spikes — detail a DENSE scan keeps
studentEnvelope   a different, WRONG shape: two humps where the dense scan has one
studentRipple     5% of the teacher's texture — what survived sparse sampling
```

and the drawn student at `(lf, hf)` is exactly

```
env[i]    = lerp(studentEnvelope[i], teacherEnvelope[i], lf)
ripple[i] = lerp(studentRipple[i],   teacherRipple[i],   hf)
y[i]      = env[i] + ripple[i]
```

At rest the student is smooth and its shape is wrong; LF pulls the shape onto the dense
scan's and leaves it smooth; HF grows the spikes until the texture matches. Measured by the
harness: the teacher's second difference is **15.4x** the student's at rest, and HF
multiplies the student's ripple energy by **19x**.

### Defect 2 — the in-plot callouts were domain-wrong

`LOW-FREQUENCY EXCESS` and `SPURIOUS HIGH-FREQUENCY RESPONSE` pinned a frequency to a region
of a spatial curve, which is meaningless. **Both are deleted**, options and all. In their
place, two callouts that are true as spatial statements and that change wording as the
faders move:

| | at rest | past `ANN_DONE_AT` (0.55) |
|---|---|---|
| left bay | `SHAPE THE SPARSE / SCAN GOT WRONG` | `SHAPE BROUGHT / BACK IN LINE` |
| right bay | `DETAIL THE SPARSE SCAN / NEVER RECORDED` | `DETAIL BACK FROM / THE DENSE SCAN` |

The SHAPE callout drops a dimension bar between the two drawn curves at the sample where the
sparse shape departs upward most (`curves.shapeDepartureIndex`); the bar shrinks to nothing
as LF closes it. The DETAIL callout brackets `ANN_DETAIL_FROM..ANN_DETAIL_TO` — a stretch
chosen because the two **shapes agree there to better than 0.06**, so the only visible
difference in that stretch really is the fine detail.

The x axis also lost its `X MAX` tick: that axis is a spatial coordinate with no units in
the source figure, so it now carries tick *marks* and the title `Spatial position` and no
numbers at all. `lowTick` / `highTick` still exist as options and default to `''`.

### Defect 3 — the teacher could not be grabbed where the curves run close

v2 painted two invisible `.lieq-grab` strokes, 22 user units wide, `pointer-events:stroke`,
with the **student's painted last**. Wherever the curves came within ~11 units of each other
the student's band covered the teacher's. Measured with real pointer events: **3 of 9** drags
along the teacher moved it; at the 6 failures `elementFromPoint` returned the student's band.

v3 deletes both bands and paints **one transparent hit surface** over the plot. On
`pointerdown` it picks the **nearest curve in y** at that x — `math.pickCurve`, a pure
function the harness probes directly — and plucks that one. The same function runs on
`pointermove`, so the curve that *would* be grabbed is highlighted first (thicker, brighter
stroke plus a ring on the curve at the pointer), and the cursor is `grab` / `grabbing`.
Ties go to the teacher. The harness now measures **9 of 9** on each curve, and **9 of 9 at
every sample where the two curves are within 11 units**.

Discoverability, which was the user's first report ("the curves cannot be dragged"):

* a bordered pill at the top right of the plot, `⇕ DRAG EITHER CURVE`, with a slow nudge;
* it **retires permanently** the first time anyone plucks (`data-lieq-plucked` on the stage);
* `cursor:grab` only when a curve is actually in reach, so the cursor never lies;
* the hover highlight + grab ring, which say *which* curve you are about to take.

---

## Integration

**Unchanged.** The page contract is still:

```html
<div id="eq-mount"></div>
<script src="assets/eq-widget.js" defer></script>
```

`autoInit()` mounts into every `[data-lieq-equalizer]`, or into `#eq-mount` if there are
none. `index.html` also copies `window.LiEQEqualizer.caption` into
`[data-eq-caption]` on load; that still works and is still the single source of the
disclosure wording.

### Two strings in `index.html` are now stale

This file does not own `index.html`, so they are listed here for whoever does:

1. The no-JS fallback inside `<figcaption>` still reads *"…the student's high-frequency
   ripple is synthesised so the HF fader has something to remove."* That is v2's inverted
   story. JS overwrites it on load, so it is only visible with JS off — but it should be
   replaced with the new `caption` text:

   > Illustrative, not measured: both profiles are drawn procedurally after the paper's
   > insight figure — one dominant shape carrying fine detail for the dense scan, a smoothed
   > two-hump shape for the sparse one — and not plotted from logged features.

2. The same `<figcaption>` says *"**HF** removes the detail sampling faked."* That is the
   paper's description of the HF branch and is correct about the paper; it is **not** what
   this widget's HF fader does, which is to grow the dense scan's fine detail on the sparse
   profile. Suggested: *"**LF** restores the shape sampling weakened, **HF** restores the
   fine detail sampling lost."*

---

## Options

Everything is optional; these are the defaults.

| option | default | what it does |
|---|---|---|
| `lf`, `hf` | `0`, `0` | starting fader positions |
| `chrome` | `false` | also paint the dark card shell (`.lieq-card`) |
| `residual` | `0.04` | signal units left between student and teacher at full travel |
| `samples` | `181` | samples on the shared x grid |
| `xLabel` | `Spatial position` | x axis title |
| `yLabel` | `Signal / feature response` | y axis title |
| `lowTick`, `highTick` | `''`, `''` | optional end ticks; empty means none is drawn |
| `teacherLabel`, `studentLabel` | `DENSE TEACHER`, `SPARSE STUDENT` | legend |
| `rackLabel` | `SPECTRAL EQ` | the rack eyebrow (the instrument's name, not a region label) |
| `readoutLabel`, `scrubLabel` | `PEAK GAP`, `GAP HERE` | the two readout modes |
| `hint` | `DRAG EITHER CURVE` | the pluck pill's text |
| `shapeAnnotation`, `shapeAnnotationDone` | see above | two lines each |
| `detailAnnotation`, `detailAnnotationDone` | see above | two lines each |
| `lfSub`, `hfSub` | `SHAPE`, `DETAIL` | the caption under each fader |
| `lfAria`, `hfAria`, `ariaLabel`, `desc` | see source | accessibility strings |
| `caption` | `LiEQEqualizer.caption` | the illustrative-not-measured line |
| `presets` | BYPASS / LF / LF+HF | the pills |
| `teacher`, `student` | `null` | host-supplied curves; split into shape + detail with `smoothRadius` / `smoothPasses` |
| `onChange(lf, hf, inst)` | `null` | called at the end of every render |

`lfAnnotation` and `hfAnnotation` **no longer exist**: they were the two frequency-domain
callouts, and they are gone rather than renamed.

## Instance API

```js
var eq = LiEQEqualizer.init('#eq-mount', { lf: 0.4 });
eq.getValues();            // -> {lf: 0.4, hf: 0}
eq.setValues(1, 1);        // jump both faders
eq.setValue('hf', 0.25);   // one fader
eq.morph(0.5, 0.5);        // the curve that pose would draw — pure, no DOM
eq.envAt(0.5);             // that pose's ENVELOPE alone
eq.destroy();              // unbind everything, empty the mount
```

Module surface: `init` `initAll` `autoInit` `renderMarkup` `css` `caption` `defaults`
`presets` `Instance` `PluckState` `curves` `constants` `math` `geometry` — unchanged from
v2 except that `curves` now also carries `teacherEnvelope`, `teacherRipple`,
`studentEnvelope`, `studentRipple`, `n`, `vmax`, `reach` and `shapeDepartureIndex`, and
`math` gains `linePath` `areaPath` `bracketPath` `lobeSum` `edgeWindow` `lcg` `buildRipple`
`buildModel` `modelVmax` `shapeDepartureIndex` `pickCurve`.

---

## The curve model

Everything that shapes the two profiles is a named constant in section 1 of the source.
There is no table of typed coordinates: the four arrays are generated, so a change of mind
is a change of one number.

```js
TEACHER_LOBES        Gaussian lobes: {c: centre, w: half-width, a: height} over u in [0,1]
STUDENT_LOBES        the sparse scan's two wrong humps + the shoulders it did get right
EDGE_TAPER   0.035   both curves window to exactly 0 at both ends
RIPPLE               the spike train: amp, minH, accentEvery, accentGain, wing,
                     strideMin/Max, flip, maskPow, edge, seed
STUDENT_RIPPLE_KEEP  0.05
FULL_SCALE_RESIDUAL  0.04
N_SAMPLES    181
```

**Why the spikes are built the way they are.** A soft sine would not read as "detail": the
ripple is a train of single-sample excursions with two short wings, spaced 2–4 samples
apart, mostly alternating in sign, with every 7th spike taller — so it reads as structure
rather than as noise or as a comb. Heights come from a seeded LCG, never `Math.random`, so
every visitor and the test harness see the same curve. The amplitude is masked by the
envelope to the power `maskPow`, because detail rides on signal: where there is no signal
there is no detail, and both curves stay non-negative so the filled areas behave.

**Why the student keeps 5% rather than 0.** A sparse scan is not perfectly smooth, and a
scaled copy of the teacher's own texture makes the HF fader exactly a gain on it — which
makes the drawn ripple energy provably monotone in `hf`. At 5% of a 0.082 spike it is 0.66
user units on screen: invisible, which is the point.

**Both curves are drawn as filled areas with a solid outline**, as the paper figure draws
them, and the two live curves are **polylines** (`math.linePath`), not splines: at 1.67 units
per sample the envelope still reads as a smooth curve, and a spike keeps a sharp tip instead
of the rounded overshoot a Catmull-Rom would put on it. The dashed reference envelope, which
has no spikes to blunt, is still a spline.

### The dashed reference

The paper overlays a dashed copy of the teacher on the student's panel. Both profiles share
one pair of axes here, so a dashed copy of the teacher's *drawn* curve would vanish inside
the teacher's own ink. What is drawn instead is the teacher's **envelope** — the shape the
LF fader is aiming at — dashed, *on top* of both curves, fading out as LF arrives
(`0.72 -> 0` over the first 85% of LF travel). It is the one line on the plot that is not
already there, and it is exactly what LF closes.

### The residual

The faders travel over `reach = 1 - residual / peakGap` of the student-to-teacher path, so
at `lf = hf = 1` the largest remaining difference is exactly `FULL_SCALE_RESIDUAL = 0.04`:
the student lands *near* the teacher, never on it, as the paper's own measurements show.
With the v3 curves `peakGap = 0.877`, so `reach = 0.954` and full travel closes 95.4% of a
0.805 shape error. The `PEAK GAP` readout therefore reads `0.04` at full travel, never
`0.00`.

---

## Layout

viewBox `480 x 248`, plot `46..346 x 30..200`, rack `362..472`. The two callouts live in the
only two bays the curves leave empty, and the harness proves it: for every pose on the fader
square it computes the topmost curve ink under each label box and asserts the label clears
it.

| bay | contents | clearance at phone type sizes |
|---|---|---|
| plot top-left | SHAPE callout, baselines 44 / 55 | text bottom 55, nearest ink 75 |
| plot top-right | hint pill 34–47, DETAIL callout 60 / 71 | text bottom 71, nearest ink 135 |

The SHAPE leader is vertical by construction: the dimension bar sits under its own text
block, so no leader ever crosses the drawing. The DETAIL bracket is recomputed every frame
as `min(both curves over the stretch) - 9`, clamped to `ANN_DETAIL.minY = 82` so it can
never climb into the text.

---

## Interaction and accessibility

* **Faders** — pointer drag, or keyboard: arrows ±0.05, shift-arrows / PageUp/PageDown ±0.2,
  Home / End. `role="slider"` with live `aria-valuenow` / `aria-valuetext`. They do not
  spring back; the value *is* the state.
* **Curves** — one hit surface, nearest curve wins, `setPointerCapture` plus window-level
  `pointermove` / `pointerup` so a drag survives leaving the SVG. A pluck **never** touches a
  fader: `PluckState` cannot see `lf` or `hf`, and the harness asserts it.
* **Grab radius** — 14 user units for a mouse (~17 CSS px on the hero, ~10 on a phone),
  26 for a touch or a pen, searched over ±2 samples so a steep flank still catches.
* **Presets** — `role="button"`, Enter / Space, `aria-pressed` tracks the faders.
* **Reduced motion** — the scan line, the hint nudge and every transition are dropped, and a
  released pluck snaps home instead of ringing.
* **Phone** — one `@media (max-width:760px)` block grows the type in user units; it must stay
  **last** in the sheet because the harness parses its tail.

**One known trade-off.** The hit surface carries `touch-action:none` over the whole plot, so
a touch that *starts* inside the plot scrolls nothing — that is what makes a vertical pluck
possible on a phone at all. v2 had the same property over most of the plot (two 22-unit
bands). If page scrolling inside the plot ever matters more than plucking on touch, change
`.lieq-hitfield` to `touch-action:pan-y` and accept that a touch pluck will fight the
scroller.

---

## Tests

```
node docs/assets/eq-widget.test.js     # 278 assertions, plain node, no DOM, no framework
node --check docs/assets/eq-widget.js
```

Sections: the four arrays and the spatial-domain claims (smooth student vs spiky teacher,
spike density / sharpness / height variation, two humps vs one); the residual; the fader
semantics (LF strictly closes the envelope distance, HF strictly raises the ripple energy,
neither interferes with the other); the morph algebra, continuity and monotonicity; the
pluck kernel and spring; **the hit test**, including the 9-of-9 measurement and every sample
where the curves are within 11 units; the path builders; the geometry, including
label-versus-ink clearance at both type sizes; a render-path smoke test driven with stub
nodes (every `setAttribute` the draw path makes); the markup and accessibility; the caption;
and robustness.

Assertions that encoded v2's inverted semantics were **replaced, not deleted**, and each
replacement says so in its name:

| v2 assertion | v3 replacement |
|---|---|
| "student envelope exceeds teacher at low frequency" | "the student OVERSHOOTS the dense shape in one place" / "falls far SHORT of it where the dense scan peaks" — same geometry, no frequency claim |
| "student carries more high-frequency energy on the right half" | "the teacher is an order of magnitude kinkier than the student" + the whole *DEFECT 1* section, inverted |
| "LF alone leaves the wiggle the student's own" | "LF alone leaves the student SMOOTH: it does not grow detail" |
| "HF alone puts the wiggle on the teacher's" | "HF alone DOES grow the fine detail" |
| "both annotations from the paper figure are present" (matched `LOW-FREQUENCY` / `SPURIOUS HIGH-FREQUENCY`) | three assertions that those exact strings are **gone**, plus one that no callout, axis title, tick, legend or readout contains `FREQ`, `SPECTR` or `BAND` |
| "it names what was synthesised" (`ripple is synthesised`) | "it names what is illustrative: both profiles are generated" + one that the old inverted sentence is gone |
| end-tick collision checks against `X MAX` | "the meaningless X MAX tick is gone, and no tick pretends to be a coordinate" |
| "the pluck band is 22 user units wide" (a note) | the grab-radius note, plus the one-hit-surface and no-`lieq-grab` assertions |

`split()` and `smooth()` are still exported and still tested: a host may pass its own
`teacher` / `student` curves, which are split into shape + detail exactly as v2 did.
