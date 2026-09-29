#!/usr/bin/env python3
"""Dump stage-level GDN tensors from the transformers reference.

Runs in the reference env (.venv-ref). Hooks the first GDN layer of
tiny-gdn, monkeypatches the chunk kernel boundary, and saves every
intermediate needed to localize MLX divergence. Writes micro-ref.npz.
"""

import json
from pathlib import Path

import numpy as np
import torch

import transformers.models.qwen4_exp.modeling_qwen4_exp as mq
from transformers import Qwen4ExpForConditionalGeneration

HERE = Path(__file__).parent
NAME = "tiny-gdn"

cap: dict = {}
_orig_chunk = mq.torch_chunk_gated_delta_rule


def spy_chunk(*args, **kwargs):
    out, state = _orig_chunk(*args, **kwargs)
    cap["kd_q"] = args[0].detach().float().numpy() if len(args) > 0 else kwargs["query"].detach().float().numpy()
    cap["kd_k"] = args[1].detach().float().numpy() if len(args) > 1 else kwargs["key"].detach().float().numpy()
    cap["kd_v"] = args[2].detach().float().numpy() if len(args) > 2 else kwargs["value"].detach().float().numpy()
    cap["kd_g"] = kwargs["g"].detach().float().numpy()
    cap["kd_beta"] = kwargs["beta"].detach().float().numpy()
    cap["kd_core"] = out.detach().float().numpy()
    return out, state


mq.torch_chunk_gated_delta_rule = spy_chunk

model = Qwen4ExpForConditionalGeneration.from_pretrained(HERE / NAME).eval()

gdn = None
for name, mod in model.named_modules():
    if mod.__class__.__name__ == "Qwen4ExpTextGatedDeltaNet":
        gdn = mod
        print(f"hooked {name}")
        break
assert gdn is not None


def fwd_hook(module, inputs, output):
    cap["x_in"] = inputs[0].detach().float().numpy()
    cap["gdn_out"] = output.detach().float().numpy()


gdn.register_forward_hook(fwd_hook)

prompts = json.loads((HERE / f"{NAME}-prompts.json").read_text())
ids = prompts["dense"]
with torch.no_grad():
    model(input_ids=torch.tensor([ids]), use_cache=False)

np.savez(
    HERE / "micro-ref.npz",
    **cap,
)
print("stages:", {k: v.shape for k, v in cap.items()})
