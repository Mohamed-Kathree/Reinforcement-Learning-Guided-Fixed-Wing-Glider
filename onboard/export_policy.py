"""
export_policy.py -- DEV PC ONLY (needs stable-baselines3 + torch). Turns an SB3
PPO checkpoint and its VecNormalize statistics into a numpy .npz the Pi can run
without torch (onboard/policy.py).

    python -m onboard.export_policy \
        --model checkpoints_v24_run4/best_model.zip \
        --vecnorm checkpoints_v24_run4/vecnormalize_best.pkl \
        --out onboard/model/run4.npz

Stored: the policy MLP's Linear layers (weights + biases, in order), the
activation, the action head, the observation normalisation (mean, var, clip,
epsilon), n_stack, obs_version, the action bounds, and a SHA-256 of the source
files. The value network is not exported (not needed to act).
tests/test_onboard.py checks numpy == SB3 deterministic actions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def export(model_path: str, vecnorm_path: str, out: str) -> dict:
    import torch
    from stable_baselines3 import PPO

    model = PPO.load(model_path, device="cpu")
    pol = model.policy
    if type(pol.features_extractor).__name__ != "FlattenExtractor":
        raise ValueError("only FlattenExtractor policies are supported")
    if getattr(pol, "squash_output", False):
        raise ValueError("squashed-output policies are not supported")
    layers, act = [], None
    for m in pol.mlp_extractor.policy_net:
        if isinstance(m, torch.nn.Linear):
            layers.append((m.weight.detach().numpy().astype(np.float32), m.bias.detach().numpy().astype(np.float32)))
        elif isinstance(m, torch.nn.Tanh):
            act = act or "tanh"
            if act != "tanh":
                raise ValueError("mixed activations")
        elif isinstance(m, torch.nn.ReLU):
            act = act or "relu"
            if act != "relu":
                raise ValueError("mixed activations")
        else:
            raise ValueError(f"unsupported layer {m}")
    head = pol.action_net
    with open(vecnorm_path, "rb") as f:
        vn = pickle.load(f)
    obs_version = int(getattr(model, "obs_version", 1))
    obs_dim = {1: 12, 2: 17}[obs_version]
    n_obs = int(model.observation_space.shape[0])
    meta = {
        "source_model": os.path.relpath(model_path, ROOT).replace("\\", "/"),
        "source_vecnorm": os.path.relpath(vecnorm_path, ROOT).replace("\\", "/"),
        "model_sha256": sha(model_path), "vecnorm_sha256": sha(vecnorm_path),
        "activation": act, "n_layers": len(layers), "obs_version": obs_version, "obs_dim": obs_dim,
        "n_stack": n_obs // obs_dim, "norm_obs": bool(vn.norm_obs), "clip_obs": float(vn.clip_obs),
        "epsilon": float(vn.epsilon),
        "action_low": model.action_space.low.tolist(), "action_high": model.action_space.high.tolist(),
    }
    arrays = {f"w{i}": w for i, (w, _) in enumerate(layers)}
    arrays.update({f"b{i}": b for i, (_, b) in enumerate(layers)})
    arrays["w_head"] = head.weight.detach().numpy().astype(np.float32)
    arrays["b_head"] = head.bias.detach().numpy().astype(np.float32)
    arrays["obs_mean"] = vn.obs_rms.mean.astype(np.float64)
    arrays["obs_var"] = vn.obs_rms.var.astype(np.float64)
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez(out, **arrays)
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints_v24_run4/best_model.zip")
    ap.add_argument("--vecnorm", default="checkpoints_v24_run4/vecnormalize_best.pkl")
    ap.add_argument("--out", default="onboard/model/run4.npz")
    a = ap.parse_args(argv)
    meta = export(a.model, a.vecnorm, a.out)
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
