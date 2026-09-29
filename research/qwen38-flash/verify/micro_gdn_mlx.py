#!/usr/bin/env python3
"""Dump stage-level GDN tensors from the MLX port, fed the reference input.

Runs in the mlx env (/Users/kearm/mlx/.venv, repo on sys.path). Loads
tiny-gdn through the port's real path, replays GatedDeltaNet.__call__
stage by stage on the identical x_in captured by micro_gdn_ref.py, and
also runs the MLX ops kernel on the reference's exact kernel inputs.
Writes micro-mlx.npz.
"""

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, "/Users/kearm/mlx-lm")

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_lm.models.gated_delta import compute_g, gated_delta_ops, normalize_qk
from mlx_lm.models.qwen4_exp import Model, ModelArgs

HERE = Path(__file__).parent
NAME = "tiny-gdn"

mx.set_default_device(mx.cpu)

ref = np.load(HERE / "micro-ref.npz")

config = json.loads((HERE / NAME / "config.json").read_text())
args_m = ModelArgs.from_dict(config)
model = Model(args_m)

weights = {}
for shard in sorted(glob.glob(str(HERE / NAME / "*.safetensors"))):
    weights.update(mx.load(shard))
weights = model.sanitize(weights)
model.load_weights(list(weights.items()))
mx.eval(model.parameters())

gdn = None
for name, mod in model.named_modules():
    if mod.__class__.__name__ == "GatedDeltaNet":
        gdn = mod
        print(f"found {name}")
        break
assert gdn is not None

x = mx.array(ref["x_in"])
B, S, _ = x.shape
out: dict = {}

mixed_qkv = gdn.in_proj_qkv(x)
z = gdn.in_proj_z(x).reshape(B, S, gdn.n_v, gdn.dv)
b = gdn.in_proj_b(x)
a = gdn.in_proj_a(x)

conv_state = mx.zeros((B, gdn.conv_kernel_size - 1, gdn.conv_dim), dtype=x.dtype)
conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)
conv_out = nn.silu(gdn.conv1d(conv_input))
out["conv_out"] = np.array(conv_out)

q, k, v = mx.split(conv_out, [gdn.key_dim, 2 * gdn.key_dim], axis=-1)
q = q.reshape(B, S, gdn.n_k, gdn.dk)
k = k.reshape(B, S, gdn.n_k, gdn.dk)
v = v.reshape(B, S, gdn.n_v, gdn.dv)
out["q_raw"] = np.array(q)
out["k_raw"] = np.array(k)
out["v"] = np.array(v)

q_norm, k_norm = normalize_qk(q, k, inv_scale=gdn.dk**-0.5, eps=1e-6)
out["q_norm_port"] = np.array(q_norm)
out["k_norm_port"] = np.array(k_norm)

beta = mx.sigmoid(b)
g = compute_g(gdn.A_log, a, gdn.dt_bias)
out["beta"] = np.array(beta)
out["g"] = np.array(g)

core, _ = gated_delta_ops(q_norm, k_norm, v, g, beta, None, None)
out["core_port"] = np.array(core)

# Feed the reference's exact kernel inputs through the MLX ops kernel.
q_ref = mx.array(ref["kd_q"])
k_ref = mx.array(ref["kd_k"])
v_ref = mx.array(ref["kd_v"])
g_ref = mx.exp(mx.array(ref["kd_g"]))  # kernel boundary passes log-decay; ops takes linear
beta_ref = mx.array(ref["kd_beta"])
rep = gdn.n_v // gdn.n_k
q_ref_u = q_ref[:, :, 0::rep]
k_ref_u = k_ref[:, :, 0::rep]
qn2, kn2 = normalize_qk(q_ref_u, k_ref_u, inv_scale=gdn.dk**-0.5, eps=1e-6)
out["q_norm_fromref"] = np.array(qn2)
core2, _ = gated_delta_ops(qn2, kn2, v_ref, g_ref, beta_ref, None, None)
out["core_fromref"] = np.array(core2)

# Full module output for the identical input.
gdn_out = gdn(x, None, None)
out["gdn_out"] = np.array(gdn_out)

np.savez(HERE / "micro-mlx.npz", **out)
print("stages:", {k: v.shape for k, v in out.items()})
