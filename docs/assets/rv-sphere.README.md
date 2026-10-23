# `rv-sphere.js` — the on-vehicle wall, carried on a 3D cap

Thirty-six clips, twelve per platform. Three buttons choose a platform; its twelve
clips face the camera as one tight block and play at once, and the other twenty-four are
parked at ±120° on the same sphere showing their stills, stopped. Click any clip to
enlarge it in the page's own lightbox; click a parked platform, or its button, and the
sphere turns to it.

One CSS file, one JS file, no framework, no bundler, no build step, no CDN. ES5. Safe to
serve straight from GitHub Pages.

## The one fact that governs everything here

**The clips are letterbox strips, not tiles.** Measured with ffprobe on what the page
ships:

| platform | resolution | ratio |
|---|---|---|
| TH-CS55 | 1100 × 406 | 2.71 : 1 |
| TH-AVT | 1100 × 332 | 3.31 : 1 |
| TH-Courier | 1100 × 340 | 3.24 : 1 |

The first version laid them on concentric rings — one in the middle, three on a triangle,
six on a hexagon. **That cannot be made gapless at any radius.** Six strips on a ring at a
readable size subtend about 24° each, cover 144° of 360, and leave the rest as holes;
pushing the ring in to close them puts it inside the middle clip. Tuning the radii against
nearest-neighbour distance made every clip close to *a* neighbour while leaving a 288px
void directly left and right of the middle one, because the triangle had no member near 0°
or 180° to fill it. Rings are the wrong shape for strips.

**Rows are the right shape.** Three columns, four rows, every clip the same size. Every row
is the same width by construction, so the block has no ragged side, and the only gaps are
the gutter.

The clips are never cropped to a friendlier ratio. Each one is a side-by-side panel —
camera, segmentation, planning — and cropping would cut a panel off.

## The layout

`ROWS`, `MID_W`, `GUTTER` and `TILT_R` in section 1 of the source are the whole thing.

| | |
|---|---|
| `ROWS` | four rows of three, all at scale 1 |
| `MID_W` | 520 — one clip's width; every other distance is solved from it |
| `GUTTER` | 3px, the hairline between neighbours |
| `TILT_R` | 2100 — the radius of the cap, which sets how far the clips turn |

The block comes to about **1515 × 769** at the tallest aspect inside an **1880 × 800**
stage, leaving roughly 180px either side for the parked platforms to show behind it.

### Touching on a curved surface is not touching on a flat one

Every clip is turned to face out of the sphere through its own position — `rotateY(atan(x/R))`,
`rotateX(-atan(y/R))` — and dropped away from the camera by `d²/2R`. Coplanar rectangles
read as a poster however they are spaced; this is what makes the block read as curvature.

But turning a clip by `lon` narrows what the camera sees of it by `cos(lon)`, and the cap
pushes it back so the page's perspective shrinks it again. **A row laid out to touch on the
plane opens visible gaps the moment it is posed.** Each row is therefore walked outward
from the middle with every clip placed where its *projected* edge meets its neighbour's, by
a short fixed point — the position depends on the pose and the pose on the position, and a
couple of dozen passes settle it to well under a pixel. The row pitch is solved the same
way. Measured by the harness: **3.3px between clips in a row, 2.2–3.0px between rows**,
against a 3px gutter.

The vertical radii are scaled per platform (`ARN_REF / arn`). Solve one set of distances
against the tallest aspect and the other two platforms are left with vertical gaps about a
fifth of a clip high, which is exactly the sparseness the packing exists to remove.

One residual is deliberate: the outer rows sit further back, so the camera sees them about
**4px narrower in 1515**. That is the cap's own foreshortening. A block whose rows matched
to the pixel would be a flat poster.

## What keeps it smooth

1. **Twelve decoders, never thirty-six.** Only the facing platform's clips carry a `src`; a
   parked clip has none at all and costs one decoded still. On a switch the arriving twelve
   are armed as the turn starts, started `START_IN` before it lands, and the leaving twelve
   released `DROP_AFT` after. Twelve clips at 1100 × 332 and **10 fps** is about 44
   Mpixel/s, well under one 1080p30 stream.
2. **Nothing animates but transform and opacity**, so no frame touches layout. The
   responsive scale is one 2D transform on a wrapper **outside** the perspective root,
   because a scale inside a `preserve-3d` chain flattens it in Safari.
3. **Out of sight is out of mind.** An `IntersectionObserver` stops and releases everything
   when the wall leaves the viewport and restores the facing platform when it returns.
4. **Touch gets one decoder.** `decoderCap()` returns 1 on a touch device or a small screen,
   where a platform may refuse the fourth `<video>` outright. Below `MIN_WIDTH` (940px) the
   block does not mount at all and the page keeps its original grid.

## Two bugs worth not reintroducing

**The face must not be hit-testable.** `.rvs-face` is a full-size box on its own plane and
the clips sit on a cap *behind* that plane, so in 3D the face stands in front of every one
of them and swallows every click: `elementFromPoint` returns `.rvs-face` and the click falls
through to the page's lightbox handler. The face carries `pointer-events:none` and the slots
take the pointer back. Symptom: clicking a clip does nothing, except for whichever clip
happens to sit at z = 0.

**No `backface-visibility` on `.rvs-hit`.** Chrome stops hit-testing a transformed element
that carries it inside a `preserve-3d` chain, so every clip with a `rotateY` became
unclickable. The button is covered by its own card from behind anyway.

Both were invisible to a test that clicked with `element.click()`, which bypasses hit
testing. Drive a real pointer, or assert on `document.elementFromPoint`.

## The motion, and where it comes from

Borrowed from the **onetake** motion library.

**Springs, not hand-picked béziers.** onetake's rule for interface motion is that it should
feel *native* rather than *animated*: SwiftUI's `Spring(duration d, bounce b)` is a damped
oscillator at ζ = 1 − b, ω = 2π/d. Sampling it over one duration makes **d cancel**, so each
curve is a pure function of its bounce alone and reusable at any length. Both are emitted as
CSS `linear()` with a `cubic-bezier` fallback: `--ease-smooth` (bounce 0) for the 120° turn,
where an overshoot on a move that large reads as a slip rather than as weight, and
`--ease-snappy` (bounce 0.15, 0.03% over) for a clip settling after one.

**`ribbon` — "panels switch with a ring-out, words lag behind", for "options change under the
hand".** That is the platform buttons. The caption drops out at once, is rewritten at
`WORD_LAG` (52%) of the turn and returns on the snappy curve, so a platform is named just
*after* its clips have arrived.

**Carry.** onetake scores a boundary 0.00 when nothing survives it. The turn is one
continuous rotation of a world that never disappears.

**No flourish that is not the product's own behaviour.** No sparkles, no specks, no idle
drift — onetake counts drift as motion and keeps it off the rests, so a block nobody is
touching is perfectly still.

## Integration

```html
<link rel="stylesheet" href="assets/rv-sphere.css" />
<script src="assets/rv-sphere.js" defer></script>
```

That is the whole contract. **This file owns no content.** It reads the thirty-six tiles
`index.html` already ships — their `data-src`, their `.webp` still, their `data-cap`, their
accent, their platform, their beam count — builds the block from what it found, and only
then hides the original grid and the original control bar. With JS off, or below
`MIN_WIDTH`, nothing here runs and the page keeps the wall it always had. The clip list
therefore still has exactly one home, in `index.html`.

Every clip keeps the page's own `data-clip` hook, so a click reaches the existing lightbox.
Clicking a parked platform anywhere turns the sphere to it instead.

## Module surface

```js
LiEQSphere.init('#wall')   // -> instance, or null if it declined to mount
LiEQSphere.autoInit()      // mounts #wall; runs itself on DOMContentLoaded
LiEQSphere.SLOTS           // the twelve {ring, x, y, z, lon, lat, s}
LiEQSphere.ROWS            // the row table
LiEQSphere.buildSlots(arn, count)   // pure; SLOTS is one call of it
LiEQSphere.pose(x, y)      // {z, lon, lat, f} for a point on the cap
LiEQSphere.constants       // every number in section 1
```

| constant | value | what it is |
|---|---|---|
| `DESIGN_W`, `DESIGN_H` | 1880, 800 | the design space one scale maps from |
| `MID_W` | 520 | one clip's width; the block is solved from it |
| `GUTTER` | 3 | the hairline between neighbours |
| `TILT_R` | 2100 | the cap radius, so about 14° at the outer clips |
| `PERSP` | 3200 | the page's perspective; keep in step with `.rvs-stage` |
| `RADIUS` | 1700 | how far a parked platform sits from the sphere's centre |
| `FACE_STEP` | 120 | three platforms, 360° between them |
| `SPIN_MS` | 1150 | the turn from one platform to the next |
| `SLOT_MS` | 860 | how long a clip takes to settle after a turn |
| `WORD_LAG` | 0.52 | how far into a turn the caption is rewritten |
| `START_IN` | 240 | playback starts this long before the turn lands |
| `DROP_AFT` | 320 | sources released this long after it lands |
| `MIN_WIDTH` | 940 | below this the block does not mount |

## Accessibility

The platform buttons are real `<button>`s with `aria-pressed`, every clip is a focusable
button carrying the tile's original `aria-label`, and the readout is a `role="status"`.
Under `prefers-reduced-motion` the turn is instant, the caption does not slide, and
**nothing autoplays** — the stills stand and the Pause button starts as Play.

## One known trade-off

A parked platform shows its stills through a back plate carrying the same `.webp` turned the
right way up by its own 180°, dimmed, drained and defocused. The alternative — letting the
real backface show — renders every parked still **mirrored**, which reads as a mistake
rather than as depth. The cost is one extra `<img>` per clip, lazy and never decoded until
it is near the viewport.

## Tests

```
node docs/assets/rv-sphere.test.js    # 138 assertions, plain node, no DOM
node --check docs/assets/rv-sphere.js
```

The harness is geometry, because geometry is the part that can be wrong in a way nobody
notices until a visitor sees two clips overlapping on one platform. It checks the row table;
that a platform short of a full set drops seats from the **last** row rather than leaving a
hole mid-block; that neighbours touch to within the gutter **after projection**, in a row and
between rows, at all three real aspect ratios; that every row is flush with every other to
within the cap's own foreshortening; that the block fits the stage and still leaves room for
the parked platforms; that each clip carries a pose with the right sign, since getting either
backwards turns the cap into a saddle; and that the timings are ordered — playback starting
inside the turn, sources released after it, the caption rewritten late enough to lag.
