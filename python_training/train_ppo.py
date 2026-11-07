"""PPO training for APEX-RL on thousands of batched environments (CUDA if available, else CPU).

    python train_ppo.py                       # defaults: 1024 envs, 60M steps, auto device
    python train_ppo.py --total-steps 20e6 --num-envs 512
    python train_ppo.py --resume checkpoints/latest.pt

At the end (and on Ctrl+C) the best checkpoint is exported to
web_visualizer/public/weights/car_policy.onnx via export_onnx.py.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from envs import ACT_DIM, OBS_DIM, TERRAIN_NAMES, VecCarEnv
from envs.raycast_car import load_config

ROOT = Path(__file__).resolve().parent


class ActorCritic(nn.Module):
    def __init__(self, state_dim=OBS_DIM, action_dim=ACT_DIM):
        super().__init__()
        # Raw physical observations are scaled inside the network, so the exported ONNX model
        # consumes the exact 19 values the browser measures (metres, m/s, rad/s, ...).
        scale = torch.tensor(load_config()["observation_scale"], dtype=torch.float32)
        self.register_buffer("obs_scale", scale)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim),
            nn.Tanh(),  # Output bounded [-1, 1]
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        with torch.no_grad():
            self.actor[5].weight.mul_(0.01)
            self.actor[5].bias.zero_()
            self.critic[4].weight.mul_(0.1)

    def normalize(self, state):
        return torch.clamp(state / self.obs_scale, -5.0, 5.0)

    def forward(self, state):
        x = self.normalize(state)
        return self.actor(x), self.critic(x).squeeze(-1)

    def act_deterministic(self, state):
        return self.actor(self.normalize(state))


@torch.no_grad()
def evaluate(model: ActorCritic, device, steps: int = 1800, per_terrain: int = 32):
    """Deterministic rollout on the four map tracks. Returns {terrain: (avg speed along track m/s, crashes/min)}."""
    env = VecCarEnv(num_envs=per_terrain * 4, device=device, fixed_tracks_only=True, domain_randomization=False, seed=123)
    group = torch.arange(env.B, device=env.device) // per_terrain
    env.pinned_terrain = group
    obs = env.reset_all()
    progress = torch.zeros(4, device=env.device)
    crashes = torch.zeros(4, device=env.device)
    for _ in range(steps):
        obs, _, term, _, info = env.step(model.act_deterministic(obs))
        progress.index_add_(0, group, info["progress"])
        crashes.index_add_(0, group, term.float())
    secs = steps * env.dt
    return {
        name: (float(progress[i] / per_terrain / secs), float(crashes[i] / per_terrain / (secs / 60)))
        for i, name in enumerate(TERRAIN_NAMES)
    }


def eval_score(res: dict, crash_weight: float = 0.5) -> float:
    # Normalise speed per terrain (ice is naturally slow) and punish crashes (per minute) by crash_weight.
    ref = {"grass": 30.0, "desert": 22.0, "windy": 28.0, "ice": 10.0}
    return sum(res[k][0] / ref[k] - crash_weight * res[k][1] for k in res) / len(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-steps", type=float, default=60e6)
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument("--rollout", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.0)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--out", default=str(ROOT / "checkpoints"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--crash-penalty", type=float, default=5.0)
    ap.add_argument("--wall-penalty", type=float, default=0.0, help="per-step penalty within 2 m of the walls")
    ap.add_argument("--obstacle-crash-penalty", type=float, default=None,
                    help="terminal penalty for hitting an obstacle (default: same as --crash-penalty)")
    ap.add_argument("--eval-crash-weight", type=float, default=0.5,
                    help="checkpoint score = mean(speed/ref) - weight * crashes/min; higher favours safety over speed")
    ap.add_argument("--obstacle-penalty", type=float, default=0.0, help="per-step penalty within 3 m of an obstacle")
    ap.add_argument("--terrain-weights", type=float, nargs=4, default=None, metavar=("GRASS", "DESERT", "WINDY", "ICE"),
                    help="relative sampling weights of the four terrains (default: uniform)")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device != "auto" else "cpu"))
    if device.type == "cpu":
        torch.set_num_threads(max(1, torch.get_num_threads()))
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[apex-rl] device={device} envs={args.num_envs} total_steps={args.total_steps:.3g}")

    env = VecCarEnv(num_envs=args.num_envs, device=device, seed=args.seed, terrain_weights=args.terrain_weights,
                   crash_penalty=args.crash_penalty, wall_penalty=args.wall_penalty,
                   obstacle_penalty=args.obstacle_penalty, obstacle_crash_penalty=args.obstacle_crash_penalty)
    model = ActorCritic().to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    start_update, best = 0, -1e9
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        best = -1e9  # re-establish: the eval protocol may differ from the run that wrote the checkpoint
        # Resume by env-steps so the env count / rollout length may differ from the original run.
        resume_steps = ck.get("steps", ck.get("update", 0) * 1024 * 64)

    B, T = args.num_envs, args.rollout
    if args.resume:
        start_update = int(resume_steps // (B * T))
        print(f"[apex-rl] resumed {args.resume} at {resume_steps/1e6:.2f}M steps (update {start_update})")
    n_updates = int(args.total_steps // (B * T))
    obs_buf = torch.zeros(T, B, OBS_DIM, device=device)
    act_buf = torch.zeros(T, B, ACT_DIM, device=device)
    logp_buf = torch.zeros(T, B, device=device)
    rew_buf = torch.zeros(T, B, device=device)
    done_buf = torch.zeros(T, B, device=device)
    val_buf = torch.zeros(T, B, device=device)

    obs = env.reset_all()
    t0 = time.time()
    ep_stats = {"crash": 0, "done": 0}
    try:
        for update in range(start_update, n_updates):
            frac = 1.0 - update / n_updates
            for g in opt.param_groups:
                g["lr"] = args.lr * max(frac, 0.05)

            speed_acc = 0.0
            for t in range(T):
                with torch.no_grad():
                    mean, value = model(obs)
                    dist = Normal(mean, model.log_std.exp())
                    action = dist.sample()
                    logp = dist.log_prob(action).sum(-1)
                next_obs, reward, term, trunc, info = env.step(action)
                if trunc.any():
                    with torch.no_grad():
                        _, v_final = model(info["final_obs"])
                    reward = reward + args.gamma * v_final * trunc.float()
                obs_buf[t], act_buf[t], logp_buf[t] = obs, action, logp
                rew_buf[t], val_buf[t] = reward, value
                done_buf[t] = (term | trunc).float()
                ep_stats["crash"] += int(term.sum())
                ep_stats["done"] += int((term | trunc).sum())
                speed_acc += float(info["speed"].mean())
                obs = next_obs

            with torch.no_grad():
                _, last_val = model(obs)
                adv = torch.zeros_like(rew_buf)
                gae = torch.zeros(B, device=device)
                for t in reversed(range(T)):
                    nv = last_val if t == T - 1 else val_buf[t + 1]
                    nonterm = 1.0 - done_buf[t]
                    delta = rew_buf[t] + args.gamma * nv * nonterm - val_buf[t]
                    gae = delta + args.gamma * args.lam * nonterm * gae
                    adv[t] = gae
                ret = adv + val_buf

            b_obs, b_act = obs_buf.reshape(-1, OBS_DIM), act_buf.reshape(-1, ACT_DIM)
            b_logp, b_adv, b_ret = logp_buf.reshape(-1), adv.reshape(-1), ret.reshape(-1)
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
            N = b_obs.shape[0]
            for _ in range(args.epochs):
                perm = torch.randperm(N, device=device)
                for i in range(0, N, args.minibatch):
                    mb = perm[i : i + args.minibatch]
                    mean, value = model(b_obs[mb])
                    dist = Normal(mean, model.log_std.exp())
                    logp = dist.log_prob(b_act[mb]).sum(-1)
                    ratio = (logp - b_logp[mb]).exp()
                    pg = -torch.min(ratio * b_adv[mb], torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * b_adv[mb]).mean()
                    vloss = 0.5 * (value - b_ret[mb]).pow(2).mean()
                    ent = dist.entropy().sum(-1).mean()
                    loss = pg + 0.5 * vloss - args.ent * ent
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    # Clip actor and critic separately: with a joint norm, large value errors (e.g. after a big
                    # crash penalty) dominate the norm and the clip silently shrinks the policy update to ~0.
                    nn.utils.clip_grad_norm_(list(model.actor.parameters()) + [model.log_std], 0.5)
                    nn.utils.clip_grad_norm_(model.critic.parameters(), 0.5)
                    opt.step()
            with torch.no_grad():
                model.log_std.clamp_(-2.5, 0.0)

            steps_done = (update + 1) * B * T
            if update % 5 == 0:
                sps = (update + 1 - start_update) * B * T / (time.time() - t0)
                cr = ep_stats["crash"] / max(ep_stats["done"], 1)
                print(
                    f"upd {update:5d} | steps {steps_done/1e6:7.2f}M | sps {sps:8.0f} | rew/step {rew_buf.mean():+.4f} "
                    f"| speed {speed_acc/T:5.1f} m/s | crash% {100*cr:5.1f} | std {model.log_std.exp().mean():.3f}",
                    flush=True,
                )
                ep_stats = {"crash": 0, "done": 0}

            if (update + 1) % args.eval_every == 0 or update == n_updates - 1:
                model.eval()
                res = evaluate(model, device)
                model.train()
                score = eval_score(res, args.eval_crash_weight)
                print("  eval: " + " | ".join(f"{k} {v[0]:.1f}m/s {v[1]:.2f}crash/min" for k, v in res.items()) + f" | score {score:.3f}", flush=True)
                ck = {"model": model.state_dict(), "opt": opt.state_dict(), "update": update + 1, "steps": steps_done, "best": best, "eval": res}
                torch.save(ck, out / "latest.pt")
                if score > best:
                    best = score
                    ck["best"] = best
                    torch.save(ck, out / "best.pt")
                    print(f"  ** new best ({best:.3f}) saved", flush=True)
    except KeyboardInterrupt:
        print("Interrupted — saving latest.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "update": update, "steps": update * B * T, "best": best}, out / "latest.pt")

    if not args.no_export:
        export_to_onnx(str(out / ("best.pt" if (out / "best.pt").exists() else "latest.pt")))


def export_to_onnx(checkpoint: str, model_path: str | None = None):
    from export_onnx import export

    export(checkpoint, model_path)


if __name__ == "__main__":
    main()
