// Track centerline generation + track-relative frame. Mirrors python_training/envs/raycast_car.py
// (centerline_from_spec, TrackBatch.build, nearest_index, track_frame) so observations match training.

export class TrackGeometry {
  constructor(spec, nPoints) {
    this.spec = spec;
    this.halfWidth = spec.half_width;
    const N = (this.N = nPoints);

    // Dense polar curve
    const dense = 4096;
    const px = new Float64Array(dense);
    const py = new Float64Array(dense);
    for (let i = 0; i < dense; i++) {
      const th = (2 * Math.PI * i) / dense;
      let r = 1;
      for (const [k, a, phi] of spec.harmonics) r += a * Math.cos(k * th + phi);
      r *= spec.radius;
      px[i] = r * Math.cos(th) * spec.aspect;
      py[i] = r * Math.sin(th);
    }
    // Arc-length resample (np.interp on the closed polyline)
    const cum = new Float64Array(dense + 1);
    for (let i = 0; i < dense; i++) {
      const j = (i + 1) % dense;
      cum[i + 1] = cum[i] + Math.hypot(px[j] - px[i], py[j] - py[i]);
    }
    const total = cum[dense];
    this.cx = new Float32Array(N);
    this.cy = new Float32Array(N);
    this.elev = new Float32Array(N);
    let seg = 0;
    for (let i = 0; i < N; i++) {
      const target = (total * i) / N;
      while (cum[seg + 1] < target) seg++;
      const f = (target - cum[seg]) / (cum[seg + 1] - cum[seg]);
      const j = (seg + 1) % dense;
      this.cx[i] = px[seg] + (px[j] - px[seg]) * f;
      this.cy[i] = py[seg] + (py[j] - py[seg]) * f;
      const u = target / total;
      let h = 0;
      for (const [k, amp, phi] of spec.elevation || []) h += amp * Math.sin(2 * Math.PI * k * u + phi);
      this.elev[i] = h;
    }

    // Tangents (central difference), left normals, arc position, boundaries
    this.tx = new Float32Array(N);
    this.ty = new Float32Array(N);
    this.s = new Float32Array(N);
    let acc = 0;
    for (let i = 0; i < N; i++) {
      const n = (i + 1) % N;
      const p = (i - 1 + N) % N;
      let dx = this.cx[n] - this.cx[p];
      let dy = this.cy[n] - this.cy[p];
      const l = Math.hypot(dx, dy);
      this.tx[i] = dx / l;
      this.ty[i] = dy / l;
      this.s[i] = acc;
      acc += Math.hypot(this.cx[n] - this.cx[i], this.cy[n] - this.cy[i]);
    }
    this.length = acc;
    this.nx = this.ty.map((v) => -v);
    this.ny = this.tx.slice();
  }

  wrap(i) {
    return ((i % this.N) + this.N) % this.N;
  }

  /** Boundary point at index i on side (+1 = left, -1 = right) at lateral distance `w` (defaults to half width). */
  edge(i, side, w = this.halfWidth) {
    i = this.wrap(i);
    return { x: this.cx[i] + this.nx[i] * w * side, y: this.cy[i] + this.ny[i] * w * side };
  }

  nearestIndex(x, y, prev, window = 12) {
    let best = prev;
    let bestD = Infinity;
    for (let o = -window; o <= window; o++) {
      const i = this.wrap(prev + o);
      const d = (this.cx[i] - x) ** 2 + (this.cy[i] - y) ** 2;
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    return best;
  }

  globalNearest(x, y) {
    let best = 0;
    let bestD = Infinity;
    for (let i = 0; i < this.N; i++) {
      const d = (this.cx[i] - x) ** 2 + (this.cy[i] - y) ** 2;
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    return best;
  }

  /** Arc position s, signed lateral offset (m, +left) and heading error (rad) at index i. */
  frame(i, x, y, yaw) {
    const dx = x - this.cx[i];
    const dy = y - this.cy[i];
    return {
      s: this.s[i] + dx * this.tx[i] + dy * this.ty[i],
      offset: dx * this.nx[i] + dy * this.ny[i],
      headingErr: yaw - Math.atan2(this.ty[i], this.tx[i]),
    };
  }

  /** Approximate distance to centerline + nearest elevation, for terrain shaping (coarse brute force). */
  distanceInfo(x, y, stride = 4) {
    let best = 0;
    let bestD = Infinity;
    for (let i = 0; i < this.N; i += stride) {
      const d = (this.cx[i] - x) ** 2 + (this.cy[i] - y) ** 2;
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    for (let o = -stride; o <= stride; o++) {
      const i = this.wrap(best + o);
      const d = (this.cx[i] - x) ** 2 + (this.cy[i] - y) ** 2;
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    return { dist: Math.sqrt(bestD), index: best, elev: this.elev[best] };
  }
}
