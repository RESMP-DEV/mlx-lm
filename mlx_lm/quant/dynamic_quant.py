# Copyright © 2025 Apple Inc.

import argparse
import copy
import json
import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from tqdm import tqdm

from mlx_lm.quant.utils import load_data
from mlx_lm.tuner.losses import kl_div_loss
from mlx_lm.tuner.trainer import grad_checkpoint
from mlx_lm.utils import (
    compute_bits_per_weight,
    load,
    quantize_model,
    save,
)

MODE_DEFAULTS = {
    "affine": (64, 4),
    "mxfp4": (32, 4),
    "mxfp8": (32, 8),
    "nvfp4": (16, 4),
}


def _quant_dequant(w, bits, group_size, mode):
    w_q, scales, *biases = mx.quantize(w, bits=bits, group_size=group_size, mode=mode)
    return mx.dequantize(
        w_q,
        scales=scales,
        biases=biases[0] if biases else None,
        bits=bits,
        group_size=group_size,
        mode=mode,
    )


def _can_quantize(module, low_group_size, high_group_size, low_mode, high_mode):
    if not hasattr(module, "weight") or not hasattr(module, "to_quantized"):
        return False
    if isinstance(module, nn.Embedding):
        return False
    weight = module.weight
    if weight.ndim != 2:
        return False
    group_sizes = (
        {low_group_size} if low_mode == high_mode else {low_group_size, high_group_size}
    )
    return all(weight.shape[-1] % group_size == 0 for group_size in group_sizes)


def _resolve_quant_params(mode, bits, group_size, default_bits, default_group_size):
    mode_group_size, mode_bits = MODE_DEFAULTS[mode]
    if bits is None:
        bits = default_bits if mode == "affine" else mode_bits
    if group_size is None:
        group_size = default_group_size if mode == "affine" else mode_group_size
    required = (mode_bits, mode_group_size)
    if mode != "affine" and (bits, group_size) != required:
        raise ValueError(
            f"{mode} requires bits={required[0]} and group_size={required[1]}"
        )
    return bits, group_size


def eval_ppl(model, data, batch_size=8):
    all_loss = 0.0
    ntoks = 0
    for s in range(0, len(data), batch_size):
        batch = data[s : s + batch_size]
        logits = model(batch[:, :-1]).astype(mx.float32)
        losses = nn.losses.cross_entropy(logits, batch[:, 1:])
        all_loss += losses.sum().item()
        ntoks += losses.size
    ppl = math.exp(all_loss / ntoks)
    return ppl


def estimate_sensitivities(
    model,
    data,
    low_bits,
    low_group_size,
    high_bits,
    high_group_size,
    low_mode="affine",
    high_mode="affine",
    batch_size: int = 4,
    gradient_accum_dtype: mx.Dtype = mx.float32,
    gradient_checkpoint: bool = False,
):
    layers = tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module)
    layers = {
        k: l
        for k, l in layers
        if _can_quantize(l, low_group_size, high_group_size, low_mode, high_mode)
    }
    if not layers:
        return []

    q_model = copy.deepcopy(model)
    q_layers = copy.deepcopy(layers)
    for l in q_layers.values():
        l.weight = _quant_dequant(l.weight, low_bits, low_group_size, low_mode)
        # Freeze everything but the quantizable weight
        l.freeze()
        l.unfreeze(keys=["weight"])
    q_model.freeze()
    q_model.update_modules(tree_unflatten(list(q_layers.items())))

    def loss_fn(batch, targets):
        return kl_div_loss(q_model(batch), targets).mean()

    if gradient_checkpoint:
        grad_checkpoint(q_model.layers[0])

    grad_accum = tree_map(
        lambda x: mx.zeros(x.shape, dtype=gradient_accum_dtype),
        q_model.trainable_parameters(),
    )
    for s in tqdm(
        range(0, len(data), batch_size),
        total=len(data) // batch_size,
        desc="Estimating sensitivities",
    ):
        batch = data[s : s + batch_size]
        targets = model(batch)
        mx.eval(targets)
        _, grads = nn.value_and_grad(q_model, loss_fn)(batch, targets)
        grad_accum = tree_map(lambda x, y: x + y, grad_accum, grads)
        del grads
        mx.eval(grad_accum)

    def compute_sensitivity(gradient, low_q_weight, original_weight):
        n_batches = (len(data) + batch_size - 1) // batch_size
        gradient = gradient / n_batches
        high_q_weight = _quant_dequant(
            original_weight, high_bits, high_group_size, high_mode
        )
        param_size = original_weight.size / 1e6
        alignment = (gradient * (low_q_weight - high_q_weight)).sum()
        return alignment / param_size

    sensitivities = tree_map(
        compute_sensitivity,
        grad_accum,
        q_model.parameters(),
        model.parameters(),
    )
    mx.eval(sensitivities)

    sensitivities = [(k[:-7], s.item()) for k, s in tree_flatten(sensitivities)]

    return sensitivities


def estimate_threshold(
    model,
    sensitivities,
    target_bpw,
    low_bits,
    low_group_size,
    high_bits,
    high_group_size,
    low_mode="affine",
    high_mode="affine",
):
    def predicate(p, m, high_threshold):
        if not _can_quantize(m, low_group_size, high_group_size, low_mode, high_mode):
            return False
        if sensitivities[p] > high_threshold:
            return {
                "bits": high_bits,
                "group_size": high_group_size,
                "mode": high_mode,
            }
        return True

    # Binary search for the threshold
    sens_vals = list(sensitivities.values())
    min_threshold = min(sens_vals)
    max_threshold = max(sens_vals)
    tolerance = 1e-3 * (max_threshold - min_threshold)
    while (max_threshold - min_threshold) > tolerance:
        mid = (max_threshold + min_threshold) / 2
        class_predicate = lambda p, m: predicate(p, m, mid)
        q_model = copy.deepcopy(model)
        nn.quantize(
            q_model,
            group_size=low_group_size,
            bits=low_bits,
            mode=low_mode,
            class_predicate=class_predicate,
        )
        bpw = compute_bits_per_weight(q_model)
        if bpw > target_bpw:
            min_threshold = mid
        else:
            max_threshold = mid

    return (max_threshold + min_threshold) / 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="Qwen/Qwen3-0.6B-base")
    parser.add_argument(
        "--mlx-path", default="mlx_model", help="Path to save the model"
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--sensitivities",
        type=str,
        default=None,
        help="Path to a pre-computed sensitivity JSON file.",
    )
    parser.add_argument(
        "--target-bpw", type=float, default=5.0, help="Target bits per weight."
    )
    parser.add_argument("--low-bits", type=int, default=None)
    parser.add_argument("--low-group-size", type=int, default=None)
    parser.add_argument("--high-bits", type=int, default=None)
    parser.add_argument("--high-group-size", type=int, default=None)
    parser.add_argument(
        "--low-mode",
        choices=tuple(MODE_DEFAULTS),
        default="affine",
    )
    parser.add_argument(
        "--high-mode",
        choices=tuple(MODE_DEFAULTS),
        default="affine",
    )
    parser.add_argument(
        "--mxfp4",
        action="store_true",
        help="use MXFP8 for sensitive layers and MXFP4 for the rest",
    )
    parser.add_argument(
        "--report-ppl",
        action="store_true",
        help="Compute the perplexity of the base and quantized models.",
    )
    parser.add_argument(
        "--grad-checkpoint",
        action="store_true",
        help="Use gradient checkpointing to reduce memory use.",
    )
    parser.add_argument(
        "--accumulation-dtype",
        default="float32",
        choices=["float32", "bfloat16"],
        help="What type to use to accumulate the gradients for the sensitivities",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trusting remote code for tokenizer/model loading.",
    )
    args = parser.parse_args()

    if args.mxfp4:
        args.low_mode = "mxfp4"
        args.high_mode = "mxfp8"
        args.low_bits = None
        args.low_group_size = None
        args.high_bits = None
        args.high_group_size = None

    try:
        args.low_bits, args.low_group_size = _resolve_quant_params(
            args.low_mode,
            args.low_bits,
            args.low_group_size,
            default_bits=4,
            default_group_size=64,
        )
        args.high_bits, args.high_group_size = _resolve_quant_params(
            args.high_mode,
            args.high_bits,
            args.high_group_size,
            default_bits=5,
            default_group_size=64,
        )
    except ValueError as e:
        parser.error(str(e))

    if (args.low_mode, args.low_bits, args.low_group_size) == (
        args.high_mode,
        args.high_bits,
        args.high_group_size,
    ):
        parser.error("low and high quantization settings must differ")

    mx.distributed.init()
    model, tokenizer, config = load(
        args.model,
        return_config=True,
        trust_remote_code=args.trust_remote_code,
    )

    if args.sensitivities is None:
        mx.random.seed(args.seed)
        data = load_data(tokenizer, num_samples=-1, sequence_length=512)

        sensitivities = estimate_sensitivities(
            model,
            data,
            args.low_bits,
            args.low_group_size,
            args.high_bits,
            args.high_group_size,
            args.low_mode,
            args.high_mode,
            gradient_accum_dtype=getattr(mx, args.accumulation_dtype),
            gradient_checkpoint=args.grad_checkpoint,
        )
        model_name = args.model.replace("/", "_")
        with open(f"{model_name}_sensitivities.json", "w") as fid:
            json.dump(sensitivities, fid)
    else:
        with open(args.sensitivities, "r") as fid:
            sensitivities = json.load(fid)

    sensitivities = dict(sensitivities)
    mx.random.seed(args.seed)
    data = load_data(tokenizer, num_samples=-1, sequence_length=512)

    if args.report_ppl:
        ppl = eval_ppl(model, data)
        print(f"Original PPL: {ppl:.3f}")

    threshold = estimate_threshold(
        model,
        sensitivities,
        target_bpw=args.target_bpw,
        low_bits=args.low_bits,
        low_group_size=args.low_group_size,
        high_bits=args.high_bits,
        high_group_size=args.high_group_size,
        low_mode=args.low_mode,
        high_mode=args.high_mode,
    )

    def quant_predicate(p, m):
        if not _can_quantize(
            m,
            args.low_group_size,
            args.high_group_size,
            args.low_mode,
            args.high_mode,
        ):
            return False
        if sensitivities[p] > threshold:
            return {
                "bits": args.high_bits,
                "group_size": args.high_group_size,
                "mode": args.high_mode,
            }
        return True

    model, config = quantize_model(
        model,
        config,
        group_size=args.low_group_size,
        bits=args.low_bits,
        mode=args.low_mode,
        quant_predicate=quant_predicate,
    )

    if args.report_ppl:
        ppl = eval_ppl(model, data)
        print(f"Quantized PPL: {ppl:.3f}")

    save(
        args.mlx_path,
        args.model,
        model,
        tokenizer,
        config,
    )
    print(f"Peak memory used: {mx.get_peak_memory() / 1000**3:.3f}GB")


if __name__ == "__main__":
    main()
