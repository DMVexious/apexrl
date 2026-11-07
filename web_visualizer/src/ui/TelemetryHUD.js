// Glassmorphic telemetry HUD: binds live vehicle + inference values to DOM gauges every frame,
// and owns panel collapsing (persisted in localStorage) and the hide-all toggle.
const $ = (id) => document.getElementById(id);
const STORAGE_KEY = 'apex-rl:collapsed-panels';

const fmtTime = (t) => {
  if (!Number.isFinite(t)) return '--:--.---';
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  return `${String(m).padStart(2, '0')}:${s.toFixed(3).padStart(6, '0')}`;
};

export class TelemetryHUD {
  constructor({ onPanelToggle } = {}) {
    this.el = {
      speed: $('val-speed'),
      boost: $('boost-tag'),
      slip: $('bar-slip'),
      slipVal: $('val-slip'),
      throttle: $('bar-throttle'),
      throttleVal: $('val-throttle'),
      steer: $('bar-steer'),
      steerVal: $('val-steer'),
      hb: $('bar-handbrake'),
      hbVal: $('val-handbrake'),
      latency: $('val-latency'),
      fps: $('val-fps'),
      backend: $('val-backend'),
      mode: $('val-mode'),
      lap: $('val-lap'),
      lapMini: $('val-laptime-mini'),
      lapTime: $('val-laptime'),
      lastLap: $('val-lastlap'),
      bestLap: $('val-bestlap'),
      crashes: $('val-crashes'),
      obstacles: $('val-obstacles'),
      mu: $('val-mu'),
      load: $('val-load'),
      map: $('val-map'),
    };
    this.lastText = new Map();
    this.initPanels(onPanelToggle);
  }

  initPanels(onToggle) {
    let saved = [];
    try {
      saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
    } catch {
      saved = [];
    }
    const panels = document.querySelectorAll('.glass-panel[data-panel]');
    const persist = () => {
      const collapsed = [...panels].filter((p) => p.classList.contains('collapsed')).map((p) => p.dataset.panel);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(collapsed));
    };
    panels.forEach((p) => {
      if (saved.includes(p.dataset.panel)) p.classList.add('collapsed');
      p.querySelector('.min-btn')?.addEventListener('click', (e) => {
        e.stopPropagation();
        p.classList.toggle('collapsed');
        persist();
        onToggle?.(p.dataset.panel, !p.classList.contains('collapsed'));
      });
    });
  }

  toggleHidden() {
    document.body.classList.toggle('hud-hidden');
  }

  text(el, v) {
    if (!el || this.lastText.get(el) === v) return;
    this.lastText.set(el, v);
    el.textContent = v;
  }

  /** Centre-split bar for bipolar values in [-1, 1]. */
  split(el, v) {
    if (!el) return;
    const w = Math.min(Math.abs(v), 1) * 50;
    el.style.left = v >= 0 ? '50%' : `${50 - w}%`;
    el.style.width = `${w}%`;
    el.classList.toggle('neg', v < 0);
  }

  update(d) {
    const e = this.el;
    this.text(e.speed, Math.round(Math.abs(d.speed) * 3.6).toString());
    e.boost.classList.toggle('on', d.boosting);
    e.slip.style.width = `${Math.min(d.slip, 1) * 100}%`;
    e.slip.classList.toggle('hot', d.slip > 0.4);
    this.text(e.slipVal, d.slip.toFixed(2));
    this.split(e.throttle, d.throttle);
    this.text(e.throttleVal, (d.throttle >= 0 ? '+' : '') + d.throttle.toFixed(2));
    this.split(e.steer, d.steer);
    this.text(e.steerVal, (d.steer >= 0 ? 'L ' : 'R ') + Math.abs(d.steer * 35).toFixed(1) + '°');
    e.hb.style.width = `${Math.min(d.handbrake, 1) * 100}%`;
    this.text(e.hbVal, d.handbrake.toFixed(2));
    this.text(e.latency, d.latency.toFixed(2));
    this.text(e.fps, Math.round(d.fps).toString());
    this.text(e.backend, d.backend);
    this.text(e.mode, d.mode);
    e.mode.classList.toggle('manual', d.mode !== 'AI PILOT');
    this.text(e.lap, String(d.lap));
    const lt = fmtTime(d.lapTime);
    this.text(e.lapTime, lt);
    this.text(e.lapMini, lt);
    this.text(e.lastLap, fmtTime(d.lastLap));
    this.text(e.bestLap, fmtTime(d.bestLap));
    this.text(e.crashes, String(d.crashes));
    this.text(e.obstacles, String(d.obstacleHits));
    this.text(e.mu, d.mu.toFixed(2));
    this.text(e.load, `${d.load.toFixed(2)} g`);
    this.text(e.map, d.mapLabel);
  }
}
