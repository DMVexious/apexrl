// Real-time LIDAR radar: 11-ray fan (forward = up), free-space polygon, proximity colours,
// wind vector arrow and a strip of the live 19-dim observation fed to the network.
import { RAY_ANGLES, RAY_MAX } from '../core/SimConfig.js';

const COLORS = { safe: '#00ffcc', warn: '#ffaa00', danger: '#ff0055' };
// per-feature scale for the observation strip (matches the network's internal normalization)
const OBS_SCALE = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 40, 10, 3, 1, 1, 1, 1, 1];

export class SensorOverlay {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.resize();
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const { width, height } = this.canvas.getBoundingClientRect();
    this.w = width || 300;
    this.h = height || 150;
    this.canvas.width = this.w * dpr;
    this.canvas.height = this.h * dpr;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  draw(rays, windLat, obs) {
    const { ctx, w, h } = this;
    ctx.clearRect(0, 0, w, h);
    const ox = w / 2;
    const oy = h - 22;
    const R = Math.min(w / 2 - 8, oy - 6);

    // range rings
    ctx.lineWidth = 1;
    for (const f of [0.25, 0.5, 0.75, 1]) {
      ctx.strokeStyle = `rgba(0,243,255,${f === 1 ? 0.35 : 0.12})`;
      ctx.beginPath();
      ctx.arc(ox, oy, R * f, Math.PI, 2 * Math.PI);
      ctx.stroke();
    }

    const pts = RAY_ANGLES.map((a, i) => {
      const r = (rays[i] / RAY_MAX) * R;
      return [ox - Math.sin(a) * r, oy - Math.cos(a) * r];
    });

    // free-space polygon
    const grad = ctx.createRadialGradient(ox, oy, 0, ox, oy, R);
    grad.addColorStop(0, 'rgba(0,243,255,0.30)');
    grad.addColorStop(1, 'rgba(0,243,255,0.04)');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.moveTo(ox, oy);
    for (const [x, y] of pts) ctx.lineTo(x, y);
    ctx.closePath();
    ctx.fill();

    // rays
    pts.forEach(([x, y], i) => {
      const n = rays[i] / RAY_MAX;
      const c = n < 0.2 ? COLORS.danger : n < 0.5 ? COLORS.warn : COLORS.safe;
      ctx.strokeStyle = c;
      ctx.globalAlpha = 0.85;
      ctx.beginPath();
      ctx.moveTo(ox, oy);
      ctx.lineTo(x, y);
      ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.fillStyle = c;
      ctx.shadowColor = c;
      ctx.shadowBlur = 8;
      ctx.beginPath();
      ctx.arc(x, y, 2.6, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
    });

    // car glyph
    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.moveTo(ox, oy - 8);
    ctx.lineTo(ox - 5, oy + 5);
    ctx.lineTo(ox + 5, oy + 5);
    ctx.closePath();
    ctx.fill();

    // lateral wind arrow (+ pushes left)
    if (Math.abs(windLat) > 0.02) {
      const len = windLat * 40;
      const y = oy + 12;
      ctx.strokeStyle = ctx.fillStyle = '#bff6ff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(ox, y);
      ctx.lineTo(ox - len, y);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(ox - len, y);
      ctx.lineTo(ox - len + Math.sign(len) * 5, y - 4);
      ctx.lineTo(ox - len + Math.sign(len) * 5, y + 4);
      ctx.fill();
      ctx.font = '600 9px Rajdhani, sans-serif';
      ctx.fillText('WIND', ox + (len > 0 ? 8 : -30), y + 3);
    }

    // observation strip: 19 tiny bipolar bars along the bottom-left
    if (obs) {
      const bw = 4;
      const x0 = 6;
      const yb = h - 4;
      for (let i = 0; i < obs.length; i++) {
        const v = Math.max(-1, Math.min(1, obs[i] / OBS_SCALE[i]));
        ctx.fillStyle = i < 11 ? 'rgba(0,243,255,0.8)' : i < 14 ? 'rgba(255,0,85,0.85)' : 'rgba(255,170,0,0.85)';
        ctx.fillRect(x0 + i * (bw + 1), yb - Math.max(v, 0) * 14, bw, Math.abs(v) * 14 || 1);
      }
    }
  }
}
