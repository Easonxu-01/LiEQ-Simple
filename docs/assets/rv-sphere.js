/*!
 * rv-sphere.js — LiEqualizer project page, the on-vehicle wall on a 3D sphere.
 *
 * Thirty clips, ten per platform. Three buttons at the top choose a platform;
 * that platform's ten clips face the camera and play at once, and the other
 * twenty are parked at +/-120deg on the same sphere showing their stills,
 * stopped. Inside a face the ten sit at three depths:
 *
 *   ring 0   one clip at the centre, largest
 *   ring 1   three clips on an equilateral triangle, smaller
 *   ring 2   six clips on a regular hexagon, smallest
 *
 * Click an outer clip and it trades places with the centre one, both travelling
 * along an arc that lifts toward the camera at mid-flight so they pass around
 * each other rather than through. Click the centre clip and the page's own
 * lightbox opens it. Click a parked platform, or its button, and the sphere
 * turns to it: the ten arriving clips are armed and started, the ten leaving
 * are stopped and have their sources released.
 *
 * THE RINGS ARE ELLIPSES. The clips are letterbox strips — 2.71:1 on CS55,
 * 3.31:1 on AVT, 3.24:1 on Courier — so six tiles spaced 60deg apart on a true
 * circle overlap left-and-right long before they overlap top-and-bottom. Each
 * ring therefore carries its own rx and ry while the ANGLES stay exactly
 * 120deg and 60deg apart, which is what makes the triangle and the hexagon read
 * as regular. RINGS below is the whole layout; rv-sphere.test.js proves that no
 * two of the ten overlap at any of the three aspect ratios, and reports the
 * tightest pair, so a change of mind here is a change of one number and a
 * re-run of the harness.
 *
 * WHAT KEEPS IT SMOOTH. Three things, in order of how much they matter:
 *
 *   1. Ten decoders, never thirty. Only the facing platform's clips carry a
 *      src. A parked clip has none at all, so it costs one decoded still. On a
 *      switch the arriving ten are armed at once, started when the turn ends,
 *      and the leaving ten are released a beat later.
 *   2. Nothing animates but transform and opacity, so no frame touches layout.
 *      The responsive scale is ONE 2D transform on a wrapper outside the
 *      perspective root, because a scale inside a preserve-3d chain flattens it
 *      in Safari.
 *   3. Out of sight is out of mind. An IntersectionObserver stops and releases
 *      everything when the wall leaves the viewport and restores the facing
 *      platform when it comes back.
 *
 * PROGRESSIVE ENHANCEMENT. This file owns no content. It reads the thirty tiles
 * the page already ships — their mp4, their .webp still, their caption, their
 * accent, their platform — builds the sphere from what it found, and only then
 * hides the original grid. With JS off, or on a viewport too narrow for ten
 * letterbox strips on a dome, the page keeps the grid it always had and nothing
 * here runs. index.html therefore needs two lines, a <link> and a <script>, and
 * no markup of its own.
 *
 * ES5. No framework, no bundler, no build step, no CDN. Safe to serve straight
 * from GitHub Pages.
 */
(function (window, document) {
  'use strict';

  /* ====================================================== 0. small helpers */

  function el(tag, cls, attrs) {
    var n = document.createElement(tag), k;
    if (cls) n.className = cls;
    if (attrs) for (k in attrs) if (attrs.hasOwnProperty(k)) n.setAttribute(k, attrs[k]);
    return n;
  }

  function reduced() {
    return !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  }

  /* ======================================================== 1. constants */

  /* the design space every number below is written in. One 2D scale maps it to
     whatever width the column actually has. */
  /* Three columns of MID_W plus the gutters come to about 1794, and four rows
     to about 889 at the reference aspect. The design space hugs that: the whole
     point of the block is the footage, so the margin is only wide enough to let
     the two parked platforms show at the edges. */
  var DESIGN_W = 1880,
      DESIGN_H = 800;

  var
      /* How far a parked platform sits from the sphere's centre. It also decides
         how far out the two parked faces land on screen: larger radius pushes
         them wider and further back, where they read as depth rather than as
         clutter beside the facing platform's outer clips. */
      RADIUS = 1700,
      FACE_STEP = 120;         /* three platforms, 360deg between them */

  /* THE LAYOUT IS THREE ROWS, NOT THREE RINGS.
     Four clips above, three across the middle, four below. The row counts are
     what make the mosaic close: three middle clips of width W and four outer
     clips of width 0.75W span the same 3W, so all three rows end flush at both
     edges and the block has no ragged side. Concentric rings cannot do this
     with letterbox strips -- six clips on a ring at a readable size subtend
     about 24deg each, cover 144deg of 360, and leave the rest as holes. That is
     why the first version looked sparse however its radii were tuned.

     The middle row is the largest and its centre seat is where a click brings a
     clip home. */
  var ROWS = [
    {n: 3, s: 1.00, row: -1.5},
    {n: 3, s: 1.00, row: -0.5},
    {n: 3, s: 1.00, row:  0.5},
    {n: 3, s: 1.00, row:  1.5}
  ];

  var GUTTER = 3;              /* the hairline between neighbours, in design px */
  /* Width of one clip. Every other distance in the block is solved from it, so
     this one number sizes the whole thing. The stage is wider than the block on
     purpose: the margin either side is where the two parked platforms show
     behind it. */
  var MID_W  = 520;

  /* The radius of the cap the clips are laid on, and the only source of their
     individual poses: a clip at (x, y) is turned to face out of the sphere
     through its own position rather than left parallel to its neighbours.
     Coplanar rectangles read as a poster however they are spaced. At 1520 the
     outermost clips stand about 24deg off the screen, which is enough to read
     as curvature and little enough to keep them watchable. */
  var TILT_R = 2100;

  /* The perspective the page applies, repeated here because the packing has to
     know it. Keep in step with .rvs-stage in rv-sphere.css. */
  var PERSP = 3200;

  /* The aspect the row widths were solved against. */
  var ARN_REF = 1100 / 406;

  /* A clip's pose and its projected scale, both of which follow from where it
     is. CSS rotateY maps (0,0,1) to (sin, 0, cos) and rotateX to (0, -sin, cos),
     so the longitude is +atan(x/R) and the latitude -atan(y/R); the cap drops
     the clip away by d^2/2R and the page's perspective scales whatever sits at
     that depth by P/(P - z). */
  function pose(x, y) {
    var z = -(x * x + y * y) / (2 * TILT_R);
    return {
      z: z,
      lon: Math.atan(x / TILT_R),
      lat: -Math.atan(y / TILT_R),
      f: PERSP / (PERSP - z)
    };
  }

  /* TOUCHING ON A CURVED SURFACE IS NOT TOUCHING ON A FLAT ONE. Turning a clip
     by lon narrows what the camera sees of it by cos(lon), and the cap pushes it
     back, so a row laid out to touch on the plane opens visible gaps the moment
     it is posed. Each row is therefore walked outward from the middle with every
     clip placed where its PROJECTED edge meets its neighbour's, by a short fixed
     point: the position depends on the pose and the pose on the position, and a
     couple of dozen passes settle it to well under a pixel. */
  function solveRow(n, hw, y) {
    var xs = new Array(n), i, k, x, p, want, mid = Math.floor((n - 1) / 2), start;

    function projEdge(x, dir) {
      var q = pose(x, y);
      return x * q.f + dir * hw * Math.cos(q.lon) * q.f;
    }

    if (n % 2) {
      xs[mid] = 0;
      start = mid;
    } else {
      x = hw;
      for (k = 0; k < 24; k++) {
        p = pose(x, y);
        x = (GUTTER / 2 + hw * Math.cos(p.lon) * p.f) / p.f;
      }
      xs[mid] = -x; xs[mid + 1] = x; start = mid + 1;
    }

    for (i = start + 1; i < n; i++) {
      want = projEdge(xs[i - 1], +1) + GUTTER;
      x = xs[i - 1] + 2 * hw;
      for (k = 0; k < 24; k++) {
        p = pose(x, y);
        x = (want + hw * Math.cos(p.lon) * p.f) / p.f;
      }
      xs[i] = x;
      xs[n - 1 - i] = -x;
    }
    return xs;
  }

  /* the same walk in y, so the three rows meet as cleanly as the clips in them */
  /* the row pitch, solved the same way: one clip's projected height plus the
     gutter. Every row holds clips of the same size, so one pitch serves all. */
  function solveRowY(hh) {
    var y = 2 * hh, k, p, f0 = pose(0, 0).f;
    for (k = 0; k < 24; k++) {
      p = pose(0, y);
      y = (hh * f0 + GUTTER + hh * Math.cos(p.lat) * p.f) / p.f;
    }
    return y;
  }

  /* One flat list of slots, middle row first so slot 0 is the clip at the centre
     of the block. `count` lets a platform short of a full set give up a seat in
     the bottom row rather than leave a hole in the mosaic. */
  function buildSlots(arn, count) {
    var A = arn || ARN_REF,
        hwMid = MID_W / 2,
        hwOut = MID_W * ROWS[1].s / 2,
        dy = solveRowY(hwMid / A),
        out = [], r, i, row, n, xs, y, hw, q;

    for (r = 0; r < ROWS.length; r++) {
      row = ROWS[r];
      n = row.n;
      /* a platform short of a full set gives up seats from the last row rather
         than leaving a hole somewhere in the middle of the block */
      if (count) {
        var before = 0, q;
        for (q = 0; q < r; q++) before += ROWS[q].n;
        n = Math.max(0, Math.min(n, count - before));
        if (!n) continue;
      }
      hw = MID_W * row.s / 2;
      y = row.row * dy;
      xs = solveRow(n, hw, y);
      for (i = 0; i < n; i++) {
        q = pose(xs[i], y);
        out.push({
          ring: r,                       /* 0 middle, 1 top, 2 bottom */
          x: Math.round(xs[i]),
          y: Math.round(y),
          z: Math.round(q.z),
          lon: +(q.lon * 180 / Math.PI).toFixed(2),
          lat: +(q.lat * 180 / Math.PI).toFixed(2),
          s: row.s
        });
      }
    }
    return out;
  }

  var SLOTS = buildSlots(ARN_REF);     /* 3 + 4 + 4 = 11 */

  var SPIN_MS  = 1150,         /* the turn from one platform to the next */
      SLOT_MS  = 860,          /* how long a clip takes to settle after a turn */
      START_IN = 240,          /* play starts this long before the turn lands */
      DROP_AFT = 320,          /* sources released this long after it lands */
      WORD_LAG = 0.52;         /* the caption is rewritten this far into a turn */

  var MIN_WIDTH = 940;         /* below this the block is abandoned, not shrunk */

  /* ============================================ 3. reading what the page has */

  /* Every fact the sphere shows already exists in the wall's own markup. This
     reads it rather than restating it, so the clip list has exactly one home. */
  function readWall(wall) {
    var groups = wall.querySelectorAll('.wall-group'), out = [], g, i, j;
    for (i = 0; i < groups.length; i++) {
      g = groups[i];
      var head = g.querySelector('.wall-head h4'),
          idx  = g.querySelector('.wall-head .idx'),
          key  = g.querySelector('.wall-stats span.k'),
          blurb= g.querySelector('.wall-blurb'),
          tiles= g.querySelectorAll('.tile'),
          clips= [];
      for (j = 0; j < tiles.length; j++) {
        var t = tiles[j],
            btn = t.querySelector('.tile-media'),
            img = t.querySelector('img.still'),
            vid = t.querySelector('video'),
            no  = t.querySelector('.tile-no');
        if (!btn || !img) continue;
        clips.push({
          mp4:  (vid && vid.getAttribute('data-src')) || btn.getAttribute('data-clip') || '',
          still: img.getAttribute('src') || '',
          cap:  btn.getAttribute('data-cap') || '',
          aria: btn.getAttribute('aria-label') || '',
          no:   no ? (no.textContent || '').replace(/\/.*$/, '') : String(j + 1),
          w:    parseInt(img.getAttribute('width'), 10) || 1100,
          h:    parseInt(img.getAttribute('height'), 10) || 340
        });
      }
      if (!clips.length) continue;
      out.push({
        group: g,
        key:   g.getAttribute('data-group') || ('p' + i),
        label: head ? (head.textContent || '').trim() : '',
        index: idx ? (idx.textContent || '').trim() : String(i + 1),
        beams: key ? (key.textContent || '').trim() : '',
        blurb: blurb ? (blurb.textContent || '').trim() : '',
        accent: (g.style.getPropertyValue('--ac') || '#56adf0').trim(),
        clips: clips
      });
    }
    return out;
  }

  /* ========================================================= 4. the faces */

  function Face(platform, n, total) {
    this.data = platform;
    this.n = n;
    /* slot -> clip index. Swapping two clips is one exchange in here. */
    this.order = [];
    this.cards = [];
    this.live = false;
    this.armed = false;

    var ar = platform.clips[0].w + '/' + platform.clips[0].h,
        arn = platform.clips[0].w / platform.clips[0].h;
    this.slots = buildSlots(arn, platform.clips.length);

    var node = el('div', 'rvs-face');
    node.setAttribute('data-face', platform.key);
    node.style.setProperty('--face-deg', (n * FACE_STEP) + 'deg');
    node.style.setProperty('--radius', RADIUS + 'px');
    node.style.setProperty('--ac', platform.accent);
    node.style.setProperty('--ar', ar);
    node.style.setProperty('--arn', arn.toFixed(4));
    this.node = node;

    for (var i = 0; i < this.slots.length && i < platform.clips.length; i++) {
      this.order.push(i);
      this.cards.push(this.makeCard(platform.clips[i], i, total));
      node.appendChild(this.cards[i].slot);
    }
    this.place(true);
  }

  Face.prototype.makeCard = function (clip, i, total) {
    var slot = el('div', 'rvs-slot');
    slot.setAttribute('data-clip-n', String(i));

    var lift  = el('div', 'rvs-lift'),
        card  = el('div', 'rvs-card'),
        front = el('div', 'rvs-plate is-front'),
        back  = el('div', 'rvs-plate is-back');

    /* the still is the only thing a parked clip costs */
    var imgF = el('img', null, {
      src: clip.still, alt: '', 'aria-hidden': 'true',
      loading: 'lazy', decoding: 'async',
      width: String(clip.w), height: String(clip.h)
    });
    var imgB = el('img', null, {
      src: clip.still, alt: '', 'aria-hidden': 'true',
      loading: 'lazy', decoding: 'async',
      width: String(clip.w), height: String(clip.h)
    });

    /* no src until this platform is the one facing the camera */
    var vid = el('video', null, {
      muted: '', loop: '', playsinline: '', preload: 'none',
      disablepictureinpicture: '', tabindex: '-1', 'aria-hidden': 'true'
    });
    vid.muted = true;
    vid.setAttribute('data-src', clip.mp4);

    /* The still underneath is the fallback, so the video only fades in once it
       genuinely has a frame. Without this gate a clip that fails to decode --
       an unsupported build, a dropped request -- would raise an empty element
       over a perfectly good poster and the slot would go black. */
    vid.addEventListener('loadeddata', function () { slot.setAttribute('data-ready', '1'); });
    vid.addEventListener('error', function () { slot.removeAttribute('data-ready'); });

    var tagNo = el('span', 'rvs-tag at-tl');
    tagNo.textContent = clip.no + '/' + total;
    var tagName = el('span', 'rvs-tag at-br');
    tagName.textContent = this.data.label;
    var tagBack = el('span', 'rvs-tag at-tl');
    tagBack.textContent = this.data.label;

    /* the lightbox the page already owns reads data-clip off whatever was
       clicked, so the centre clip keeps working by carrying the same hook */
    var hit = el('button', 'rvs-hit', {
      type: 'button',
      'data-clip': clip.mp4,
      'data-cap': clip.cap,
      'aria-label': clip.aria
    });

    /* the arrow says which way an outer clip is about to travel; the word on
       the back says the dark panels at the sides are clickable at all */
    var badge = el('span', 'rvs-badge', {'aria-hidden': 'true'});
    badge.innerHTML = '&#9974;';
    var turn = el('span', 'rvs-turn', {'aria-hidden': 'true'});
    turn.textContent = 'CLICK TO TURN';

    front.appendChild(imgF); front.appendChild(vid);
    front.appendChild(tagNo); front.appendChild(tagName);
    back.appendChild(imgB); back.appendChild(tagBack); back.appendChild(turn);
    card.appendChild(front); card.appendChild(back);
    lift.appendChild(card); lift.appendChild(hit); lift.appendChild(badge);
    slot.appendChild(lift);

    return {slot: slot, video: vid, hit: hit, clip: clip};
  };

  /* write every clip's slot. `hard` skips the transition, for the first paint. */
  Face.prototype.place = function (hard) {
    for (var s = 0; s < this.order.length; s++) this.placeOne(s, hard);
  };

  Face.prototype.placeOne = function (slotIndex, hard) {
    var c = this.cards[this.order[slotIndex]], sl = this.slots[slotIndex], st = c.slot.style;
    if (hard) st.transition = 'none';
    st.setProperty('--x', String(sl.x));
    st.setProperty('--y', String(sl.y));
    st.setProperty('--z', String(sl.z));
    st.setProperty('--lon', sl.lon + 'deg');
    st.setProperty('--lat', sl.lat + 'deg');
    st.setProperty('--s', sl.s.toFixed(3));
    c.slot.setAttribute('data-ring', String(sl.ring));
    if (hard) {
      /* read once so the "none" is committed before it is taken away */
      void c.slot.offsetWidth;
      st.transition = '';
    }
  };

  /* --- playback --------------------------------------------------------- */

  Face.prototype.arm = function (limit) {
    var n = (limit == null) ? this.cards.length : limit, i, c;
    for (i = 0; i < this.cards.length; i++) {
      c = this.cards[i];
      /* the centre clip is armed first, so a capped device still plays the one
         clip the eye is on */
      var wanted = (i === this.order[0]) || (this.playableRank(i) < n);
      if (wanted && !c.video.getAttribute('src')) {
        c.video.setAttribute('src', c.video.getAttribute('data-src'));
        try { c.video.load(); } catch (e) {}
      }
    }
    this.armed = true;
  };

  /* the centre first, then the inner ring, then the outer: the order a capped
     device should spend its decoders in */
  Face.prototype.playableRank = function (clipIndex) {
    for (var s = 0; s < this.order.length; s++) if (this.order[s] === clipIndex) return s;
    return 99;
  };

  Face.prototype.play = function () {
    this.live = true;
    this.node.setAttribute('data-active', '1');
    for (var i = 0; i < this.cards.length; i++) {
      var c = this.cards[i];
      if (!c.video.getAttribute('src')) continue;
      c.slot.setAttribute('data-live', '1');
      var p = c.video.play();
      if (p && p['catch']) p['catch'](function () {});
    }
  };

  Face.prototype.stop = function () {
    this.live = false;
    for (var i = 0; i < this.cards.length; i++) {
      this.cards[i].slot.removeAttribute('data-live');
      try { this.cards[i].video.pause(); } catch (e) {}
    }
  };

  Face.prototype.disarm = function () {
    this.stop();
    this.node.removeAttribute('data-active');
    for (var i = 0; i < this.cards.length; i++) {
      var v = this.cards[i].video;
      if (!v.getAttribute('src')) continue;
      v.removeAttribute('src');
      this.cards[i].slot.removeAttribute('data-ready');
      try { v.load(); } catch (e) {}     /* the decoder goes back to the pool */
    }
    this.armed = false;
  };

  /* ========================================================= 5. the sphere */

  function Sphere(wall, platforms) {
    var self = this;
    this.wall = wall;
    this.platforms = platforms;
    this.total = 0;
    for (var i = 0; i < platforms.length; i++) this.total += platforms[i].clips.length;

    this.reduce = reduced();
    this.cap = this.decoderCap();
    this.current = 0;
    this.spin = 0;                 /* kept continuous, never wrapped, so the
                                      sphere always turns the short way */
    this.spinning = 0;
    this.visible = true;
    this.paused = this.reduce;     /* reduced motion does not autoplay */

    this.root = el('div', 'rvs-root');
    this.root.setAttribute('data-rvs', '1');
    this.scaler = el('div', 'rvs-scaler');
    this.stage = el('div', 'rvs-stage');
    this.sphereNode = el('div', 'rvs-sphere');
    this.sphereNode.style.setProperty('--radius', RADIUS + 'px');
    this.sphereNode.style.setProperty('--spin-ms', SPIN_MS + 'ms');

    this.faces = [];
    for (i = 0; i < platforms.length; i++) {
      var f = new Face(platforms[i], i, this.total);
      f.node.style.setProperty('--slot-ms', SLOT_MS + 'ms');
      this.faces.push(f);
      this.sphereNode.appendChild(f.node);
    }

    this.stage.appendChild(this.sphereNode);
    this.scaler.appendChild(this.stage);
    this.root.appendChild(this.scaler);

    /* the gesture nobody guesses. It rides in the caption row rather than
       floating over the block, because the stage now hugs the clips and a pill
       at its foot would sit on the bottom row. It retires for good at the first
       click: a hint that outstays the lesson it teaches is clutter. */
    this.hint = el('div', 'rvs-hint');
    var hintText = el('span');
    hintText.textContent = 'CLICK ANY CLIP TO ENLARGE IT';
    this.hint.appendChild(hintText);

    this.bar = this.makeBar();
    this.caption = this.makeCaption();
    this.caption.appendChild(this.hint);

    /* in before the groups, so the sphere is what the section leads with */
    var anchor = wall.querySelector('.wall-group');
    wall.insertBefore(this.bar, anchor);
    wall.insertBefore(this.root, anchor);
    wall.insertBefore(this.caption, anchor);

    /* the page's own bar would now be a second set of controls */
    var old = wall.querySelector('.wall-bar');
    if (old) old.style.display = 'none';
    for (i = 0; i < platforms.length; i++) platforms[i].group.style.display = 'none';

    this.root.addEventListener('click', function (ev) { self.onClick(ev); });
    this.resize();
    addEventListener('resize', function () { self.resize(); }, {passive: true});
    this.watch();
    this.select(0, false);
  }

  /* Ten simultaneous decoders is nothing for 1100x332 at 10fps on a desktop GPU
     and is a gamble on a phone, where the platform may refuse the fourth one
     outright. Touch or a small screen therefore gets the centre clip alone. */
  Sphere.prototype.decoderCap = function () {
    var touch = ('ontouchstart' in window) ||
                (navigator.maxTouchPoints > 0 && !window.matchMedia('(min-width:1100px)').matches);
    return touch ? 1 : SLOTS.length;
  };

  Sphere.prototype.makeBar = function () {
    var bar = el('div', 'rvs-bar'),
        seg = el('div', 'rvs-seg', {role: 'group', 'aria-label': 'Choose a vehicle platform'}),
        i;
    this.tabs = [];
    for (i = 0; i < this.platforms.length; i++) {
      var p = this.platforms[i],
          b = el('button', null, {type: 'button', 'data-go': String(i), 'aria-pressed': i ? 'false' : 'true'});
      b.innerHTML = '';
      b.appendChild(document.createTextNode(p.label.replace(/^TH-/, '')));
      var cnt = el('b');
      cnt.textContent = String(p.clips.length);
      b.appendChild(cnt);
      seg.appendChild(b);
      this.tabs.push(b);
    }
    bar.appendChild(seg);
    bar.appendChild(el('span', 'rvs-spacer'));
    this.liveEl = el('span', 'rvs-live', {role: 'status', 'aria-live': 'off'});
    bar.appendChild(this.liveEl);
    this.pauseEl = el('button', 'rvs-btn', {type: 'button', 'data-pause': '1', 'aria-pressed': 'false'});
    this.pauseEl.innerHTML = '&#10073;&#10073; Pause';
    bar.appendChild(this.pauseEl);
    var self = this;
    this.pauseEl.addEventListener('click', function () { self.togglePause(); });
    return bar;
  };

  Sphere.prototype.makeCaption = function () {
    var c = el('div', 'rvs-caption');
    this.capN = el('span', 'n'); this.capN.setAttribute('aria-hidden', 'true');
    this.capH = el('h4');
    this.capB = el('span', 'beams');
    /* the blurb is gone: this block is here for the footage, and a paragraph
       under it is a paragraph of space the footage does not get */
    this.capP = null;
    c.appendChild(this.capN); c.appendChild(this.capH); c.appendChild(this.capB);
    return c;
  };

  Sphere.prototype.resize = function () {
    var w = this.root.parentNode ? this.root.parentNode.clientWidth : DESIGN_W,
        k = Math.min(1, w / DESIGN_W);
    this.k = k;
    this.scaler.style.setProperty('--k', k.toFixed(4));
    this.root.style.setProperty('--card-w', MID_W + 'px');
    this.root.style.height = Math.round(DESIGN_H * k) + 'px';
  };

  /* out of sight, out of mind: thirty stills stay, ten decoders do not */
  Sphere.prototype.watch = function () {
    var self = this;
    if (!window.IntersectionObserver) return;
    new IntersectionObserver(function (es) {
      for (var i = 0; i < es.length; i++) {
        var on = es[i].isIntersecting;
        if (on === self.visible) continue;
        self.visible = on;
        if (on) self.resume(); else self.faces[self.current].disarm();
      }
    }, {rootMargin: '220px 0px'}).observe(this.root);
  };

  Sphere.prototype.resume = function () {
    var f = this.faces[this.current];
    f.arm(this.cap);
    if (!this.paused) f.play();
    this.readout();
  };

  Sphere.prototype.togglePause = function () {
    this.paused = !this.paused;
    this.pauseEl.innerHTML = this.paused ? '&#9654; Play' : '&#10073;&#10073; Pause';
    this.pauseEl.setAttribute('aria-pressed', this.paused ? 'true' : 'false');
    if (this.paused) this.faces[this.current].stop();
    else { this.faces[this.current].arm(this.cap); this.faces[this.current].play(); }
    this.readout();
  };

  /* --- choosing a platform ---------------------------------------------- */

  Sphere.prototype.select = function (n, animate) {
    var self = this, i, outgoing = this.faces[this.current];
    if (n === this.current && this.faces[n].armed) return;

    /* the short way round: step the continuous angle by at most 180deg */
    var delta = n - this.current;
    while (delta > 1.5) delta -= this.faces.length;
    while (delta < -1.5) delta += this.faces.length;
    this.spin -= delta * FACE_STEP;
    this.current = n;

    for (i = 0; i < this.tabs.length; i++) {
      this.tabs[i].setAttribute('aria-pressed', i === n ? 'true' : 'false');
    }
    var p = this.platforms[n];
    this.root.style.setProperty('--ac', p.accent);
    this.bar.style.setProperty('--ac', p.accent);
    this.caption.style.setProperty('--ac', p.accent);
    this.wall.style.setProperty('--ac', p.accent);
    var writeCaption = function () {
      self.capN.textContent = p.index;
      self.capH.textContent = p.label;
      self.capB.textContent = p.beams;
    };

    var incoming = this.faces[n];
    if (outgoing !== incoming) outgoing.stop();
    if (this.visible) incoming.arm(this.cap);

    var ms = (animate === false || this.reduce) ? 1 : SPIN_MS;
    this.sphereNode.style.setProperty('--spin-ms', ms + 'ms');
    this.sphereNode.style.setProperty('--spin', this.spin + 'deg');

    if (ms > 1) {
      this.spinning++;
      this.root.setAttribute('data-spinning', '1');
      /* the words lag the panels: the caption drops out at once, is rewritten
         about halfway through the turn and returns on the snappy curve, so a
         platform is named just after its clips have arrived */
      this.caption.className = 'rvs-caption is-out';
      window.setTimeout(function () {
        if (self.current !== n) return;
        writeCaption();
        self.caption.className = 'rvs-caption';
      }, Math.round(ms * WORD_LAG));
    } else {
      writeCaption();
      this.caption.className = 'rvs-caption';
    }

    /* the arriving clips start just before the turn lands, so the first frame
       is already moving when it settles */
    window.setTimeout(function () {
      if (self.current !== n || !self.visible) return;
      if (!self.paused) incoming.play();
      self.readout();
    }, Math.max(0, ms - START_IN));

    window.setTimeout(function () {
      if (ms > 1 && --self.spinning <= 0) {
        self.spinning = 0;
        self.root.removeAttribute('data-spinning');
      }
      /* released only once the face is genuinely out of view */
      if (outgoing !== self.faces[self.current]) outgoing.disarm();
    }, ms + DROP_AFT);
  };

  Sphere.prototype.readout = function () {
    if (!this.liveEl) return;
    var f = this.faces[this.current],
        p = this.platforms[this.current],
        live = 0, i;
    for (i = 0; i < f.cards.length; i++) if (f.cards[i].slot.getAttribute('data-live')) live++;
    this.liveEl.innerHTML = 'Facing <i>' + p.label + '</i> <b>' +
      (this.paused ? 'held' : live + ' of ' + f.cards.length + ' live') + '</b>';
  };

  /* --- one delegated click ---------------------------------------------- */

  Sphere.prototype.onClick = function (ev) {
    var t = ev.target;
    if (!t || !t.closest) return;

    /* a parked platform, clicked anywhere on it, turns to the front */
    var face = t.closest('.rvs-face');
    if (face) {
      for (var i = 0; i < this.faces.length; i++) {
        if (this.faces[i].node !== face) continue;
        if (i !== this.current) {
          ev.preventDefault();
          ev.stopPropagation();          /* not a lightbox click */
          this.select(i, true);
          return;
        }
        break;
      }
    }

    /* On the facing platform every clip answers the same way: the click goes
       through to the page's own lightbox, which is what both a single and a
       double click are now for. The block has no centre seat to trade places
       with, so there is nothing else a click could mean. */
    var slot = t.closest('.rvs-slot');
    if (slot && face === this.faces[this.current].node) {
      this.root.setAttribute('data-hinted', '1');
      this.hint.className = 'rvs-hint is-out';
    }
  };

  /* the buttons are in the bar, which is not inside .rvs-root */
  Sphere.prototype.bindTabs = function () {
    var self = this;
    for (var i = 0; i < this.tabs.length; i++) {
      (function (n, b) {
        b.addEventListener('click', function () { self.select(n, true); });
      }(i, this.tabs[i]));
    }
  };

  /* ============================================================ 6. mount */

  function init(wall) {
    wall = (typeof wall === 'string') ? document.querySelector(wall) : wall;
    if (!wall || wall.getAttribute('data-rvs-on')) return null;

    /* a dome of letterbox strips needs width; below MIN_WIDTH the page keeps
       the grid it already has, which is the better answer on a phone anyway */
    if (window.innerWidth < MIN_WIDTH) return null;

    var platforms = readWall(wall);
    if (platforms.length < 2) return null;

    var s = new Sphere(wall, platforms);
    s.bindTabs();
    wall.setAttribute('data-rvs-on', '1');
    return s;
  }

  function autoInit() {
    var w = document.getElementById('wall');
    if (w) init(w);
  }

  window.LiEQSphere = {
    init: init,
    autoInit: autoInit,
    SLOTS: SLOTS,
    ARN_REF: ARN_REF,
    ROWS: ROWS,
    pose: pose,
    constants: {
      DESIGN_W: DESIGN_W, DESIGN_H: DESIGN_H,
      RADIUS: RADIUS, FACE_STEP: FACE_STEP, TILT_R: TILT_R,
      PERSP: PERSP, GUTTER: GUTTER, MID_W: MID_W,
      SPIN_MS: SPIN_MS, SLOT_MS: SLOT_MS,
      START_IN: START_IN, DROP_AFT: DROP_AFT, MIN_WIDTH: MIN_WIDTH,
      WORD_LAG: WORD_LAG
    },
    buildSlots: buildSlots
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', autoInit);
  } else {
    autoInit();
  }

}(window, document));
