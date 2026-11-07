"""Export a trained PPO checkpoint to ONNX for onnxruntime-web, then validate it.

    python export_onnx.py                                  # checkpoints/best.pt -> web weights
    python export_onnx.py --checkpoint checkpoints/latest.pt --out car_policy.onnx

The exported graph is: raw 19-dim observation -> scale -> actor MLP -> [steer, throttle, clamp(handbrake, 0, 1)].
Input name 'input_states' [batch, 19], output name 'output_actions' [batch, 3].
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT.parent / "web_visualizer" / "public" / "weights" / "car_policy.onnx"


class ExportPolicy(nn.Module):
    def __init__(self, ac):
        super().__init__()
        self.ac = ac

    def forward(self, input_states):
        a = self.ac.act_deterministic(input_states)
        return torch.cat([a[:, :2], torch.clamp(a[:, 2:3], 0.0, 1.0)], dim=1)


def export(checkpoint: str | None = None, out: str | None = None) -> Path:
    from train_ppo import ActorCritic

    checkpoint = checkpoint or str(ROOT / "checkpoints" / "best.pt")
    out_path = Path(out) if out else DEFAULT_OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ac = ActorCritic()
    ck = torch.load(checkpoint, map_location="cpu")
    ac.load_state_dict(ck["model"])
    ac.eval()
    policy = ExportPolicy(ac).eval()
    dummy = torch.randn(1, 19, dtype=torch.float32)

    kwargs = dict(
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input_states"],
        output_names=["output_actions"],
        dynamic_axes={"input_states": {0: "batch_size"}, "output_actions": {0: "batch_size"}},
    )
    try:
        torch.onnx.export(policy, (dummy,), str(out_path), dynamo=False, **kwargs)
    except TypeError:  # older torch without the dynamo flag
        torch.onnx.export(policy, (dummy,), str(out_path), **kwargs)

    # Inline any external-data sidecar so the browser only needs one file.
    import onnx

    model = onnx.load(str(out_path), load_external_data=True)
    onnx.checker.check_model(model)
    onnx.save_model(model, str(out_path), save_as_external_data=False)
    sidecar = out_path.with_name(out_path.name + ".data")
    if sidecar.exists():
        sidecar.unlink()

    validate(policy, out_path)
    print(f"Model successfully exported to {out_path} ({out_path.stat().st_size/1024:.1f} KiB)")
    return out_path


def validate(policy: nn.Module, path: Path):
    import onnxruntime as ort

    from envs import VecCarEnv

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    env = VecCarEnv(num_envs=256, domain_randomization=False, fixed_tracks_only=True, seed=7)
    x = env._observe().numpy().astype(np.float32)
    ref = policy(torch.from_numpy(x)).detach().numpy()
    got = sess.run(["output_actions"], {"input_states": x})[0]
    err = float(np.abs(ref - got).max())
    assert got.shape == (256, 3), got.shape
    assert err < 1e-4, f"ONNX mismatch: max abs err {err}"
    assert got[:, 2].min() >= 0.0
    print(f"Validation OK — max |torch - onnxruntime| = {err:.2e} on 256 real observations")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    export(a.checkpoint, a.out)
