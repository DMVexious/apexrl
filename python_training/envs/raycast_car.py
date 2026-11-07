"""Vectorized (batched) car physics, track geometry and raycast sensors in pure PyTorch.

Everything here runs on whatever device the tensors live on (CUDA or CPU), so thousands of
environments are simulated in lock-step without Python loops.

The model is a dynamic bicycle model with saturating (tanh) tyre curves, a friction circle on
the rear axle (so throttle / handbrake break traction), rear-wheel drive, aero drag, rolling
resistance and a lateral wind force. The web visualizer applies *the same* force equations to
a Rapier rigid body, so a policy trained here transfers to the browser.

Conventions: 2D plane (x, y), yaw measured CCW from +x, body frame = (forward, left).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

G = 9.81
CONFIG_PATH = Path(__file__).resolve().parents[2] / "shared" / "sim_config.json"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------------------------
# Track geometry
# --------------------------------------------------------------------------------------------
def centerline_from_spec(spec: dict, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Closed centerline from a polar harmonic spec, resampled uniformly by arc length.

    r(theta) = R * (1 + sum_k a_k cos(k theta + phi_k)); x is stretched by `aspect`.
    Returns (points [N,2], elevation [N]). Mirrored exactly in web_visualizer/src/entities/Tracks.js.
    """
    dense = 4096
    theta = np.linspace(0.0, 2.0 * np.pi, dense, endpoint=False)
    r = np.ones_like(theta)
    for k, a, phi in spec["harmonics"]:
        r += a * np.cos(k * theta + phi)
    r *= spec["radius"]
    pts = np.stack([r * np.cos(theta) * spec["aspect"], r * np.sin(theta)], axis=1)

    seg = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    targets = np.linspace(0.0, total, n_points, endpoint=False)
    closed = np.concatenate([pts, pts[:1]], axis=0)
    x = np.interp(targets, cum, closed[:, 0])
    y = np.interp(targets, cum, closed[:, 1])
    u = targets / total
    elev = np.zeros(n_points)
    for k, amp, phi in spec.get("elevation", []):
        elev += amp * np.sin(2.0 * np.pi * k * u + phi)
    return np.stack([x, y], axis=1), elev


def min_turn_radius(points: np.ndarray) -> float:
    p_prev, p_next = np.roll(points, 1, axis=0), np.roll(points, -1, axis=0)
    a = np.linalg.norm(points - p_prev, axis=1)
    b = np.linalg.norm(p_next - points, axis=1)
    c = np.linalg.norm(p_next - p_prev, axis=1)
    cross = np.abs((points - p_prev)[:, 0] * (p_next - points)[:, 1] - (points - p_prev)[:, 1] * (p_next - points)[:, 0])
    curvature = 2.0 * cross / np.maximum(a * b * c, 1e-9)
    return float(1.0 / max(curvature.max(), 1e-9))


def random_track_spec(rng: np.random.Generator) -> dict:
    """Random but drivable closed track (rejects specs whose tightest turn is too sharp)."""
    while True:
        half_width = float(rng.uniform(6.0, 8.5))
        spec = {
            "radius": float(rng.uniform(90, 140)),
            "aspect": float(rng.uniform(1.0, 1.6)),
            "half_width": half_width,
            "harmonics": [],
            "elevation": [],
        }
        for k in (2, 3, 4, 5):
            amp = float(rng.uniform(0.0, 0.75 / (k * k)))
            spec["harmonics"].append([k, amp, float(rng.uniform(0, 2 * np.pi))])
        hilliness = float(rng.uniform(0.0, 1.0))  # some tracks flat, some very hilly
        for k in (1, 2, 3, 5, 7, 10):
            amp = float(rng.uniform(0.0, 1.0)) * hilliness * 9.0 / k
            spec["elevation"].append([k, amp, float(rng.uniform(0, 2 * np.pi))])
        pts, elev = centerline_from_spec(spec, 512)
        grade, _ = elevation_derivatives(pts, elev)
        if min_turn_radius(pts) > 2.2 * half_width and np.abs(grade).max() < 0.13:
            return spec


def elevation_derivatives(points: np.ndarray, elev: np.ndarray):
    """Road grade dh/ds and vertical curvature d2h/ds2 (central differences along the loop)."""
    ds = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1).mean()
    grade = (np.roll(elev, -1) - np.roll(elev, 1)) / (2 * ds)
    vcurv = (np.roll(elev, -1) - 2 * elev + np.roll(elev, 1)) / ds**2
    return grade, vcurv


@dataclass
class TrackBatch:
    """K tracks stacked into tensors: center/tangent/normal [K,N,2], s [K,N], length/half_width [K]."""

    center: torch.Tensor
    tangent: torch.Tensor
    normal: torch.Tensor
    s: torch.Tensor
    length: torch.Tensor
    half_width: torch.Tensor
    left: torch.Tensor
    right: torch.Tensor
    elev: torch.Tensor  # [K,N] road height (m)
    grade: torch.Tensor  # [K,N] dh/ds
    vcurv: torch.Tensor  # [K,N] d2h/ds2 (negative on crests)

    @property
    def n_points(self) -> int:
        return self.center.shape[1]

    @staticmethod
    def build(specs: list[dict], n_points: int, device) -> "TrackBatch":
        centers, widths, elevs, grades, vcurvs = [], [], [], [], []
        for spec in specs:
            pts, elev = centerline_from_spec(spec, n_points)
            grade, vcurv = elevation_derivatives(pts, elev)
            centers.append(pts)
            widths.append(spec["half_width"])
            elevs.append(elev)
            grades.append(grade)
            vcurvs.append(vcurv)
        c = torch.tensor(np.stack(centers), dtype=torch.float32, device=device)
        nxt = torch.roll(c, -1, dims=1)
        prv = torch.roll(c, 1, dims=1)
        t = nxt - prv
        t = t / t.norm(dim=-1, keepdim=True)
        n = torch.stack([-t[..., 1], t[..., 0]], dim=-1)  # left normal
        seg = (nxt - c).norm(dim=-1)
        s = torch.cumsum(seg, dim=1) - seg
        length = seg.sum(dim=1)
        hw = torch.tensor(widths, dtype=torch.float32, device=device)
        left = c + n * hw[:, None, None]
        right = c - n * hw[:, None, None]
        f32 = lambda a: torch.tensor(np.stack(a), dtype=torch.float32, device=device)
        return TrackBatch(c, t, n, s, length, hw, left, right, f32(elevs), f32(grades), f32(vcurvs))


# --------------------------------------------------------------------------------------------
# Car dynamics
# --------------------------------------------------------------------------------------------
class CarParams:
    def __init__(self, cfg: dict):
        car = cfg["car"]
        self.m = car["mass"]
        self.iz = car["yaw_inertia"]
        self.lf = car["lf"]
        self.lr = car["lr"]
        self.half_width = car["half_width"]
        self.max_steer = math.radians(car["max_steer_deg"])
        self.steer_rate = car["steer_rate"]
        self.drive_force = car["drive_force"]
        self.brake_force = car["brake_force"]
        self.hb_force = car["handbrake_force"]
        self.hb_cut = car["handbrake_grip_cut"]
        self.drag = car["drag"]
        self.k_tire = car["tire_stiffness"]
        self.min_slip_speed = car["min_slip_speed"]
        self.wind_force = car["wind_force"]


def car_forces(p: CarParams, vx, vy, r, steer, throttle, hb, mu, drive, roll, rear_grip, wind_lat, mass_scale, load=1.0):
    """Body-frame forces. Returns (front_fx, front_fy, rear_fx, rear_fy, extra_fx, extra_fy) where
    front forces are already rotated into the body frame and act at +lf, rear at -lr, extra at the CoM.
    `load` = normal force / (m g): < 1 over crests (car goes light, less grip), > 1 in compressions.
    Same equations as web_visualizer/src/core/PhysicsWorld.js::computeTireForces."""
    m = p.m * mass_scale
    L = p.lf + p.lr
    fz_f = m * G * p.lr / L * load
    fz_r = m * G * p.lf / L * load

    v_ref = torch.clamp(vx.abs(), min=p.min_slip_speed)
    alpha_f = steer - torch.atan2(vy + p.lf * r, v_ref)
    alpha_r = -torch.atan2(vy - p.lr * r, v_ref)

    mu_r_long = mu * rear_grip
    mu_r_lat = mu_r_long * (1.0 - p.hb_cut * hb)

    roll_sign = torch.tanh(vx / 0.5)
    f_drive = p.drive_force * drive * torch.clamp(throttle, 0.0, 1.0)
    f_brake = -p.brake_force * torch.clamp(-throttle, 0.0, 1.0) * roll_sign
    f_hb = -p.hb_force * hb * roll_sign

    fx_r_cap = mu_r_long * fz_r
    fx_r = torch.maximum(torch.minimum(f_drive + 0.4 * f_brake + f_hb, fx_r_cap), -fx_r_cap)
    fx_f_cap = mu * fz_f
    fx_f = torch.maximum(torch.minimum(0.6 * f_brake, fx_f_cap), -fx_f_cap)

    fy_f = mu * fz_f * torch.tanh(p.k_tire * alpha_f)
    fy_r_max = torch.sqrt(torch.clamp((mu_r_lat * fz_r) ** 2 - fx_r**2, min=0.0) + (0.1 * mu_r_lat * fz_r) ** 2)
    fy_r = fy_r_max * torch.tanh(p.k_tire * alpha_r)

    cs, sn = torch.cos(steer), torch.sin(steer)
    front_fx = fx_f * cs - fy_f * sn
    front_fy = fy_f * cs + fx_f * sn

    extra_fx = -roll * m * G * load * roll_sign - p.drag * vx * vx.abs()
    extra_fy = p.wind_force * wind_lat
    return front_fx, front_fy, fx_r, fy_r, extra_fx, extra_fy


def slope_terms(grade, vcurv, heading_err, vx):
    """Gravity components along the car's forward/left axes and the normal-load factor on a 3D road.
    grade/vcurv are taken along the track tangent; the car is yawed by heading_err relative to it."""
    inv = torch.rsqrt(1.0 + grade * grade)  # cos(pitch of the road)
    g_long = -G * grade * torch.cos(heading_err) * inv
    g_lat = G * grade * torch.sin(heading_err) * inv
    v_track = vx * torch.cos(heading_err)
    load = torch.clamp(inv + v_track * v_track * vcurv / G, 0.0, 2.0)
    return g_long, g_lat, load


def integrate(p: CarParams, st: dict, steer, throttle, hb, env: dict, wind_lat, dt: float, slope=None, boost=0.0):
    """Semi-implicit Euler step of the bicycle model, in place on the state dict.
    slope = (g_long, g_lat, load) from slope_terms(), or None for a flat road.
    boost = extra forward force (N) from a boost pad; acts like a jet, independent of tyre grip."""
    vx, vy, r = st["vx"], st["vy"], st["r"]
    g_long, g_lat, load = slope if slope is not None else (0.0, 0.0, 1.0)
    ffx, ffy, rfx, rfy, efx, efy = car_forces(
        p, vx, vy, r, steer, throttle, hb, env["mu"], env["drive"], env["roll"], env["rear_grip"], wind_lat, env["mass"], load
    )
    m = p.m * env["mass"]
    iz = p.iz * env["mass"]
    ax = (ffx + rfx + efx + boost) / m + g_long + r * vy
    ay = (ffy + rfy + efy) / m + g_lat - r * vx
    rdot = (p.lf * ffy - p.lr * rfy) / iz
    vx = torch.clamp(vx + ax * dt, min=-3.0)
    vy = vy + ay * dt
    r = r + rdot * dt
    yaw = st["yaw"] + r * dt
    c, s = torch.cos(yaw), torch.sin(yaw)
    st["x"] = st["x"] + (vx * c - vy * s) * dt
    st["y"] = st["y"] + (vx * s + vy * c) * dt
    st["yaw"], st["vx"], st["vy"], st["r"] = yaw, vx, vy, r


# --------------------------------------------------------------------------------------------
# Sensors
# --------------------------------------------------------------------------------------------
def nearest_index(tracks: TrackBatch, track_id, x, y, prev_idx, window: int = 12):
    """Local search for the nearest centerline point around the previous index."""
    N = tracks.n_points
    offs = torch.arange(-window, window + 1, device=x.device)
    cand = (prev_idx[:, None] + offs[None, :]) % N
    pts = tracks.center[track_id[:, None], cand]  # [B,W,2]
    d2 = (pts[..., 0] - x[:, None]) ** 2 + (pts[..., 1] - y[:, None]) ** 2
    return cand.gather(1, d2.argmin(dim=1, keepdim=True)).squeeze(1)


def global_nearest_index(tracks: TrackBatch, track_id, x, y):
    pts = tracks.center[track_id]  # [B,N,2]
    d2 = (pts[..., 0] - x[:, None]) ** 2 + (pts[..., 1] - y[:, None]) ** 2
    return d2.argmin(dim=1)


def track_frame(tracks: TrackBatch, track_id, idx, x, y, yaw):
    """Returns (arc position s, signed lateral offset [m, +left], heading error [rad])."""
    c = tracks.center[track_id, idx]
    t = tracks.tangent[track_id, idx]
    n = tracks.normal[track_id, idx]
    dx, dy = x - c[:, 0], y - c[:, 1]
    s = tracks.s[track_id, idx] + dx * t[:, 0] + dy * t[:, 1]
    offset = dx * n[:, 0] + dy * n[:, 1]
    heading_err = yaw - torch.atan2(t[:, 1], t[:, 0])
    return s, offset, heading_err


def cast_rays(tracks: TrackBatch, track_id, idx, x, y, yaw, ray_angles, max_dist: float, back: int = 12, ahead: int = 40, stride: int = 2):
    """Distance from the car to the track walls along each ray [B,R]. Only wall segments from `back`
    to `ahead` centerline points around the car are tested (rays span -90..+90 deg, so nothing behind
    the car is needed), using every `stride`-th boundary vertex (chord error of a few cm)."""
    N = tracks.n_points
    offs = torch.arange(-back, ahead, stride, device=x.device)
    i0 = (idx[:, None] + offs[None, :]) % N
    i1 = (i0 + stride) % N
    tid = track_id[:, None]
    a = torch.cat([tracks.left[tid, i0], tracks.right[tid, i0]], dim=1)  # [B,S,2]
    b = torch.cat([tracks.left[tid, i1], tracks.right[tid, i1]], dim=1)
    e = b - a

    ang = yaw[:, None] + ray_angles[None, :]  # [B,R]
    d = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)  # [B,R,2]

    ap = a - torch.stack([x, y], dim=-1)[:, None, :]  # [B,S,2]
    # cross(u, v) = u.x v.y - u.y v.x ; shapes broadcast to [B,R,S]
    dx, dy = d[..., 0][:, :, None], d[..., 1][:, :, None]
    ex, ey = e[..., 0][:, None, :], e[..., 1][:, None, :]
    apx, apy = ap[..., 0][:, None, :], ap[..., 1][:, None, :]
    denom = dx * ey - dy * ex
    safe = torch.where(denom.abs() < 1e-8, torch.full_like(denom, 1e-8), denom)
    t = (apx * ey - apy * ex) / safe
    u = (apx * dy - apy * dx) / safe
    valid = (denom.abs() > 1e-8) & (t > 0.0) & (u >= 0.0) & (u <= 1.0)
    t = torch.where(valid, t, torch.full_like(t, max_dist))
    return torch.clamp(t.min(dim=-1).values, max=max_dist)


def cast_rays_circles(x, y, yaw, ray_angles, cx, cy, cr, max_dist: float):
    """Distance along each ray to the nearest circular obstacle (vertical cylinders on the road).
    cx, cy, cr: [B,M] centres/radii; inactive obstacles have cr == 0. Returns [B,R]."""
    ang = yaw[:, None] + ray_angles[None, :]  # [B,R]
    dx, dy = torch.cos(ang)[:, :, None], torch.sin(ang)[:, :, None]
    fx = (x[:, None] - cx)[:, None, :]  # [B,1,M]  origin - centre
    fy = (y[:, None] - cy)[:, None, :]
    b = fx * dx + fy * dy  # [B,R,M]
    c = fx * fx + fy * fy - (cr * cr)[:, None, :]
    disc = b * b - c
    t = -b - torch.sqrt(torch.clamp(disc, min=0.0))
    valid = (disc >= 0.0) & (t > 0.0) & (cr[:, None, :] > 0.0)
    t = torch.where(valid, t, torch.full_like(t, max_dist))
    return torch.clamp(t.min(dim=-1).values, max=max_dist)
