"""Batched multi-terrain racing environment (pure PyTorch, CUDA-ready) + a Gymnasium wrapper.

Observation (19, raw physical units — normalization lives inside the policy network):
    [0..10]  wall distance along 11 rays, -90deg (right) .. +90deg (left), metres, max 50
    [11]     forward velocity vx (m/s)          [12] lateral velocity vy (m/s, +left)
    [13]     yaw rate (rad/s, +CCW)
    [14,15]  cos / sin of heading error vs. centerline tangent
    [16]     centerline offset / half width, clipped to [-1, 1] (+left)
    [17]     surface friction coefficient mu
    [18]     lateral wind component [-1, 1] (+ pushes left)
Action (3): steer [-1,1] (+left, x35deg), throttle/brake [-1,1], handbrake [0,1] (negatives clipped).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .raycast_car import (
    CarParams,
    TrackBatch,
    cast_rays,
    cast_rays_circles,
    global_nearest_index,
    integrate,
    load_config,
    nearest_index,
    random_track_spec,
    slope_terms,
    track_frame,
)

TERRAIN_NAMES = ["grass", "desert", "windy", "ice"]
OBS_DIM = 19
ACT_DIM = 3


def wind_lateral(t, yaw, wmax, phi0, phase):
    """Deterministic gusting wind field, identical to web_visualizer PhysicsWorld.windAt()."""
    phi = phi0 + 0.15 * torch.sin(0.05 * t)
    strength = wmax * torch.clamp(0.6 + 0.3 * torch.sin(0.4 * t + phase) + 0.1 * torch.sin(1.7 * t + 2.0 * phase), 0.0, 1.0)
    return strength * torch.sin(phi - yaw)


class VecCarEnv:
    def __init__(
        self,
        num_envs: int = 1024,
        device: str | torch.device = "cpu",
        n_random_tracks: int = 64,
        terrains: list[str] | None = None,
        fixed_tracks_only: bool = False,
        max_steps: int = 3000,
        domain_randomization: bool = True,
        seed: int = 0,
        terrain_weights: list[float] | None = None,
        crash_penalty: float = 5.0,
        wall_penalty: float = 0.0,
        course: bool = True,
        obstacle_penalty: float = 0.0,
        obstacle_crash_penalty: float | None = None,
    ):
        self.cfg = load_config()
        self.p = CarParams(self.cfg)
        self.B = num_envs
        self.device = torch.device(device)
        self.max_steps = max_steps
        self.dr = domain_randomization
        self.crash_penalty = crash_penalty
        self.wall_penalty = wall_penalty  # per-step penalty ramping up inside the last 2 m before wall contact
        self.obstacle_penalty = obstacle_penalty  # per-step penalty ramping up inside 3 m of an obstacle's surface
        # Terminal penalty for hitting an obstacle (replaces crash_penalty for those crashes); defaults to crash_penalty.
        self.obstacle_crash_penalty = crash_penalty if obstacle_crash_penalty is None else obstacle_crash_penalty
        self.gen = torch.Generator(device="cpu").manual_seed(seed)

        hz = self.cfg["control_hz"]
        self.dt = 1.0 / hz
        self.substeps = self.cfg["physics_substeps"]
        sens = self.cfg["sensors"]
        self.ray_max = sens["ray_max_dist"]
        self.ob_inflate = sens.get("obstacle_inflate", 0.0)  # body-width clearance in the obstacle ray test
        # Rays are denser straight ahead (±10°, ±20°) so obstacles can't hide between them.
        angles = sens.get("ray_angles_deg") or torch.linspace(sens["ray_min_deg"], sens["ray_max_deg"], sens["ray_count"]).tolist()
        self.ray_angles = torch.deg2rad(torch.tensor(angles, dtype=torch.float32)).to(self.device)

        # Track pool: the four hand-designed map tracks (ids 0..3, same order as TERRAIN_NAMES) + random ones.
        rng = np.random.default_rng(seed)
        specs = [self.cfg["tracks"][name] for name in TERRAIN_NAMES]
        if not fixed_tracks_only:
            specs += [random_track_spec(rng) for _ in range(n_random_tracks)]
        self.tracks = TrackBatch.build(specs, self.cfg["track_points"], self.device)
        self.n_tracks = len(specs)
        self.fixed_tracks_only = fixed_tracks_only

        terr = self.cfg["terrains"]
        self.terrain_ids = [TERRAIN_NAMES.index(t) for t in (terrains or TERRAIN_NAMES)]
        w = torch.tensor(terrain_weights if terrain_weights else [1.0] * len(self.terrain_ids), dtype=torch.float32)
        self.terrain_probs = w / w.sum()  # sampling weights over self.terrain_ids
        self.pinned_terrain = None  # optional LongTensor [B] forcing each env's terrain (evaluation)
        tab = lambda key: torch.tensor([terr[n][key] for n in TERRAIN_NAMES], device=self.device)
        self.t_mu, self.t_drive, self.t_roll = tab("mu"), tab("drive"), tab("roll")
        self.t_rear, self.t_wind = tab("rear_grip"), tab("wind")

        z = lambda: torch.zeros(self.B, device=self.device)
        zi = lambda: torch.zeros(self.B, dtype=torch.long, device=self.device)
        self.st = {k: z() for k in ("x", "y", "yaw", "vx", "vy", "r")}
        self.steer = z()
        self.prev_steer_cmd = z()
        self.time = z()
        self.steps = zi()
        self.stuck = z()
        self.s_prev = z()
        self.idx = zi()
        self.track_id = zi()
        self.terrain = zi()
        self.env = {k: z() for k in ("mu", "drive", "roll", "rear_grip", "mass")}
        self.wind_max, self.wind_phi0, self.wind_phase = z(), z(), z()

        # Course: obstacles (vertical cylinders on the road) + boost pads, re-randomized every episode.
        cc = self.cfg["course"]
        self.course_enabled = course
        self.cc = cc
        M, P = cc["obstacle_slots"], cc["boost_slots"]
        zm = lambda k: torch.zeros(self.B, k, device=self.device)
        self.ob_idx = torch.zeros(self.B, M, dtype=torch.long, device=self.device)
        self.ob_x, self.ob_y, self.ob_r = zm(M), zm(M), zm(M)  # ob_r == 0 -> inactive
        self.pad_idx = torch.zeros(self.B, P, dtype=torch.long, device=self.device)
        self.pad_lat, self.pad_on = zm(P), torch.zeros(self.B, P, dtype=torch.bool, device=self.device)
        self.pad_force = z()
        cm = [cc["maps"][n] for n in TERRAIN_NAMES]
        self.t_ob_frac = torch.tensor([m["obstacles"] / M for m in cm], device=self.device)
        self.t_ob_rmin = torch.tensor([m["radius"][0] for m in cm], device=self.device)
        self.t_ob_rmax = torch.tensor([m["radius"][1] for m in cm], device=self.device)
        self.t_pad_frac = torch.tensor([m["boost_pads"] / P for m in cm], device=self.device)
        self.t_pad_force = torch.tensor([m["boost_force"] for m in cm], device=self.device, dtype=torch.float32)
        self.reset_all()

    # ------------------------------------------------------------------ helpers
    def _rand(self, n, lo=0.0, hi=1.0):
        return (torch.rand(n, generator=self.gen) * (hi - lo) + lo).to(self.device)

    def _randint(self, n, hi):
        return torch.randint(0, hi, (n,), generator=self.gen).to(self.device)

    def reset_all(self):
        self._reset(torch.ones(self.B, dtype=torch.bool, device=self.device))
        return self._observe()

    def _reset(self, mask: torch.Tensor):
        ids = mask.nonzero(as_tuple=True)[0]
        n = ids.numel()
        if n == 0:
            return
        if self.pinned_terrain is not None:
            terrain = self.pinned_terrain[ids]
        else:
            terr_choices = torch.tensor(self.terrain_ids, device=self.device)
            pick = torch.multinomial(self.terrain_probs, n, replacement=True, generator=self.gen).to(self.device)
            terrain = terr_choices[pick]
        if self.fixed_tracks_only:
            track = terrain.clone()
        else:
            use_map_track = self._rand(n) < 0.35
            track = torch.where(use_map_track, terrain, self._randint(n, self.n_tracks))
        N = self.tracks.n_points
        idx = self._randint(n, N)
        hw = self.tracks.half_width[track]
        c = self.tracks.center[track, idx]
        t = self.tracks.tangent[track, idx]
        nrm = self.tracks.normal[track, idx]
        lat = self._rand(n, -0.45, 0.45) * (hw - self.p.half_width)
        self.st["x"][ids] = c[:, 0] + nrm[:, 0] * lat
        self.st["y"][ids] = c[:, 1] + nrm[:, 1] * lat
        self.st["yaw"][ids] = torch.atan2(t[:, 1], t[:, 0]) + self._rand(n, -0.2, 0.2)

        jitter = (lambda: self._rand(n, 0.9, 1.1)) if self.dr else (lambda: torch.ones(n, device=self.device))
        mu = self.t_mu[terrain] * jitter()
        self.env["mu"][ids] = mu
        self.env["drive"][ids] = self.t_drive[terrain] * jitter()
        self.env["roll"][ids] = self.t_roll[terrain]
        self.env["rear_grip"][ids] = self.t_rear[terrain]
        self.env["mass"][ids] = jitter()
        base_wind = self.t_wind[terrain]
        small_wind = self._rand(n, 0.0, 0.2) if self.dr else torch.zeros(n, device=self.device)
        self.wind_max[ids] = torch.maximum(base_wind * jitter(), small_wind)
        self.wind_phi0[ids] = self._rand(n, 0, 2 * math.pi)
        self.wind_phase[ids] = self._rand(n, 0, 2 * math.pi)

        self.st["vx"][ids] = self._rand(n, 0.0, 1.0) * 14.0 * torch.sqrt(mu / 0.85)
        self.st["vy"][ids] = 0.0
        self.st["r"][ids] = 0.0
        self.steer[ids] = 0.0
        self.prev_steer_cmd[ids] = 0.0
        self.time[ids] = self._rand(n, 0, 100)
        self.steps[ids] = 0
        self.stuck[ids] = 0.0
        self.idx[ids] = idx
        self.track_id[ids] = track
        self.terrain[ids] = terrain
        s, _, _ = track_frame(self.tracks, track, idx, self.st["x"][ids], self.st["y"][ids], self.st["yaw"][ids])
        self.s_prev[ids] = s
        self._reset_course(ids, track, terrain, idx)

    def _reset_course(self, ids, track, terrain, spawn_idx):
        """Obstacles in evenly spaced slots (random position inside the slot, random lateral offset that
        always leaves >= lateral_gap of road free), clear of the spawn point; boost pads likewise.
        Training randomizes the density per episode; evaluation uses each map's configured counts.
        Mirrored by web_visualizer/src/core/Course.js."""
        n = ids.numel()
        cc, N = self.cc, self.tracks.n_points
        M, P = cc["obstacle_slots"], cc["boost_slots"]
        if not self.course_enabled:
            self.ob_r[ids] = 0.0
            self.pad_on[ids] = False
            return
        hw = self.tracks.half_width[track][:, None]
        rnd = lambda k: torch.rand(n, k, generator=self.gen).to(self.device)
        # --- obstacles
        slot = torch.arange(M, device=self.device)[None, :].float()
        o_idx = ((slot + rnd(M) * 0.6 + 0.2) * N / M).long() % N
        frac = self._rand(n, 0.0, 0.85) if self.dr else self.t_ob_frac[terrain]
        active = rnd(M) < frac[:, None]
        rel = (o_idx - spawn_idx[:, None]) % N
        active &= (rel >= cc["spawn_clear_ahead_pts"]) & (rel <= N - cc["spawn_clear_behind_pts"])
        if self.dr:  # training: any size across all maps' ranges
            rmin = float(self.t_ob_rmin.min())
            rmax = float(self.t_ob_rmax.max())
        else:
            rmin, rmax = self.t_ob_rmin[terrain][:, None], self.t_ob_rmax[terrain][:, None]
        r = rmin + (rmax - rmin) * rnd(M)
        lat_max = torch.clamp(hw - r - cc["lateral_gap"], min=0.0)
        lat = (rnd(M) * 2 - 1) * lat_max
        tid = track[:, None]
        c = self.tracks.center[tid, o_idx]
        nrm = self.tracks.normal[tid, o_idx]
        self.ob_idx[ids] = o_idx
        self.ob_x[ids] = c[..., 0] + nrm[..., 0] * lat
        self.ob_y[ids] = c[..., 1] + nrm[..., 1] * lat
        self.ob_r[ids] = torch.where(active, r, torch.zeros_like(r))
        # --- boost pads (never placed just before an obstacle)
        pslot = torch.arange(P, device=self.device)[None, :].float()
        p_idx = ((pslot + rnd(P) * 0.6 + 0.2) * N / P).long() % N
        pfrac = self._rand(n, 0.0, 1.0) if self.dr else self.t_pad_frac[terrain]
        on = rnd(P) < pfrac[:, None]
        gap = (o_idx[:, None, :] - p_idx[:, :, None]) % N  # [n,P,M] obstacle position relative to pad
        conflict = (active[:, None, :] & ((gap < 30) | (gap > N - 4))).any(-1)
        self.pad_idx[ids] = p_idx
        self.pad_on[ids] = on & ~conflict
        self.pad_lat[ids] = (rnd(P) * 2 - 1) * torch.clamp(hw - cc["boost_half_width"] - 0.5, min=0.0)
        self.pad_force[ids] = self.t_pad_force[terrain]

    def _boost(self, offset):
        """Boost force (N) for cars whose centre is over an active pad."""
        N = self.tracks.n_points
        seg = self.tracks.length[self.track_id] / N
        rel = (self.idx[:, None] - self.pad_idx) % N
        over = (
            self.pad_on
            & (rel.float() * seg[:, None] < self.cc["boost_length"])
            & ((offset[:, None] - self.pad_lat).abs() < self.cc["boost_half_width"])
        )
        return over.any(-1).float() * self.pad_force

    def _obstacle_clearance(self):
        """Distance (m) from the car's two footprint circles to the nearest obstacle surface."""
        st, cc = self.st, self.cc
        c, s = torch.cos(st["yaw"]), torch.sin(st["yaw"])
        best = torch.full((self.B,), 1e3, device=self.device)
        for k in (cc["car_collision_offset"], -cc["car_collision_offset"]):
            px = (st["x"] + c * k)[:, None]
            py = (st["y"] + s * k)[:, None]
            d = torch.sqrt((px - self.ob_x) ** 2 + (py - self.ob_y) ** 2) - self.ob_r - cc["car_collision_radius"]
            d = torch.where(self.ob_r > 0, d, torch.full_like(d, 1e3))
            best = torch.minimum(best, d.min(-1).values)
        return best

    def _obstacle_hit(self):
        """Car approximated by two circles (front/rear) vs. obstacle cylinders."""
        st, cc = self.st, self.cc
        hit = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        c, s = torch.cos(st["yaw"]), torch.sin(st["yaw"])
        for k in (cc["car_collision_offset"], -cc["car_collision_offset"]):
            px = (st["x"] + c * k)[:, None]
            py = (st["y"] + s * k)[:, None]
            d2 = (px - self.ob_x) ** 2 + (py - self.ob_y) ** 2
            hit |= ((d2 < (self.ob_r + cc["car_collision_radius"]) ** 2) & (self.ob_r > 0)).any(-1)
        return hit

    def _wind_lat(self):
        return wind_lateral(self.time, self.st["yaw"], self.wind_max, self.wind_phi0, self.wind_phase)

    def _observe(self, noise: bool | None = None):
        noise = self.dr if noise is None else noise
        st = self.st
        rays = cast_rays(self.tracks, self.track_id, self.idx, st["x"], st["y"], st["yaw"], self.ray_angles, self.ray_max)
        if self.course_enabled:
            sense_r = torch.where(self.ob_r > 0, self.ob_r + self.ob_inflate, self.ob_r)
            obstacle_rays = cast_rays_circles(st["x"], st["y"], st["yaw"], self.ray_angles, self.ob_x, self.ob_y, sense_r, self.ray_max)
            rays = torch.minimum(rays, obstacle_rays)
        _, offset, he = track_frame(self.tracks, self.track_id, self.idx, st["x"], st["y"], st["yaw"])
        hw = self.tracks.half_width[self.track_id]
        vel = torch.stack([st["vx"], st["vy"], st["r"]], dim=1)
        if noise:
            rays = torch.clamp(rays + torch.randn(rays.shape, generator=self.gen).to(self.device) * 0.25, 0.0, self.ray_max)
            vel = vel + torch.randn(vel.shape, generator=self.gen).to(self.device) * torch.tensor([0.1, 0.05, 0.01], device=self.device)
        obs = torch.cat(
            [
                rays,
                vel,
                torch.cos(he)[:, None],
                torch.sin(he)[:, None],
                torch.clamp(offset / hw, -1.0, 1.0)[:, None],
                self.env["mu"][:, None],
                torch.clamp(self._wind_lat(), -1.0, 1.0)[:, None],
            ],
            dim=1,
        )
        return obs

    # ------------------------------------------------------------------ step
    @torch.no_grad()
    def step(self, action: torch.Tensor):
        """Returns obs, reward, terminated, truncated, info{final_obs, ...}. Done envs auto-reset."""
        p = self.p
        a = torch.clamp(action.to(self.device), -1.0, 1.0)
        steer_cmd = a[:, 0] * p.max_steer
        throttle = a[:, 1]
        hb = torch.clamp(a[:, 2], 0.0, 1.0)

        max_delta = p.steer_rate * self.dt
        self.steer = self.steer + torch.clamp(steer_cmd - self.steer, -max_delta, max_delta)
        wind = self._wind_lat()
        _, off0, he0 = track_frame(self.tracks, self.track_id, self.idx, self.st["x"], self.st["y"], self.st["yaw"])
        boost = self._boost(off0) if self.course_enabled else 0.0
        grade = self.tracks.grade[self.track_id, self.idx]
        vcurv = self.tracks.vcurv[self.track_id, self.idx]
        sub_dt = self.dt / self.substeps
        for _ in range(self.substeps):
            slope = slope_terms(grade, vcurv, he0, self.st["vx"])
            integrate(p, self.st, self.steer, throttle, hb, self.env, wind, sub_dt, slope, boost)
        self.time += self.dt
        self.steps += 1

        st = self.st
        self.idx = nearest_index(self.tracks, self.track_id, st["x"], st["y"], self.idx)
        s, offset, he = track_frame(self.tracks, self.track_id, self.idx, st["x"], st["y"], st["yaw"])
        L = self.tracks.length[self.track_id]
        ds = torch.remainder(s - self.s_prev + 0.5 * L, L) - 0.5 * L
        self.s_prev = s
        hw = self.tracks.half_width[self.track_id]

        hit_obstacle = self._obstacle_hit() if self.course_enabled else torch.zeros_like(offset, dtype=torch.bool)
        crashed = (offset.abs() > (hw - p.half_width)) | hit_obstacle
        speed = torch.sqrt(st["vx"] ** 2 + st["vy"] ** 2)
        self.stuck = torch.where(speed < 1.0, self.stuck + self.dt, torch.zeros_like(self.stuck))
        backwards = torch.cos(he) < -0.5
        truncated = ((self.steps >= self.max_steps) | (self.stuck > 3.0) | (backwards & (speed > 3.0))) & ~crashed

        dsteer = a[:, 0] - self.prev_steer_cmd
        self.prev_steer_cmd = a[:, 0]
        reward = (
            0.1 * ds
            - 0.02 * (offset / hw) ** 2
            - 0.02 * dsteer**2
            - 0.003 * hb
            - self.wall_penalty * torch.clamp((offset.abs() - (hw - p.half_width - 2.0)) / 2.0, 0.0, 1.0)
            - torch.where(hit_obstacle, self.obstacle_crash_penalty, self.crash_penalty * crashed.float())
        )
        if self.course_enabled and self.obstacle_penalty > 0:
            reward = reward - self.obstacle_penalty * torch.clamp(1.0 - self._obstacle_clearance() / 3.0, 0.0, 1.0)

        done = crashed | truncated
        info = {"progress": ds, "speed": speed, "crashed": crashed, "obstacle_hit": hit_obstacle, "terrain": self.terrain.clone()}
        if done.any():
            info["final_obs"] = self._observe(noise=False)
            self._reset(done)
        return self._observe(), reward, crashed, truncated, info


try:  # Optional Gymnasium single-env wrapper (training uses VecCarEnv directly for speed).
    import gymnasium as gym
    from gymnasium import spaces

    class ApexCarEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self, terrain: str = "grass", **kw):
            super().__init__()
            self.vec = VecCarEnv(num_envs=1, terrains=[terrain], fixed_tracks_only=True, domain_randomization=False, **kw)
            hi = np.full(OBS_DIM, np.inf, dtype=np.float32)
            self.observation_space = spaces.Box(-hi, hi, dtype=np.float32)
            self.action_space = spaces.Box(
                np.array([-1, -1, 0], dtype=np.float32), np.array([1, 1, 1], dtype=np.float32), dtype=np.float32
            )

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            if seed is not None:
                self.vec.gen.manual_seed(seed)
            return self.vec.reset_all()[0].cpu().numpy(), {}

        def step(self, action):
            obs, rew, term, trunc, info = self.vec.step(torch.as_tensor(action, dtype=torch.float32)[None])
            return obs[0].cpu().numpy(), float(rew[0]), bool(term[0]), bool(trunc[0]), {}

    gym.register(id="ApexRL-Car-v0", entry_point=ApexCarEnv)
except ImportError:  # pragma: no cover
    ApexCarEnv = None
