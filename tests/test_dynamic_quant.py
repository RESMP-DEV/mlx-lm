# Copyright © 2026 Apple Inc.

import unittest


import mlx.core as mx
import mlx.nn as nn

from mlx_lm.quant import dynamic_quant
from mlx_lm.quant.dynamic_quant import estimate_sensitivities
from mlx_lm.utils import compute_bits_per_weight, quantize_model


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.low = nn.Linear(64, 16, bias=False)
        self.high = nn.Linear(64, 16, bias=False)

    def __call__(self, x):
        return self.low(x) + self.high(x)


class TokenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 8, bias=False)

    def __call__(self, x):
        return self.fc(x.reshape(-1, 64)).reshape(*x.shape[:-1], 8)


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Embedding(256, 16)

    def __call__(self, x):
        return self.fc(x)


class TestDynamicQuant(unittest.TestCase):
    def test_quant_dequant_mxfp4(self):
        weight = mx.random.normal((4, 64))
        values = dynamic_quant._quant_dequant(weight, 4, 32, "mxfp4")
        self.assertEqual(values.shape, weight.shape)
        self.assertEqual(values.dtype, mx.bfloat16)
        self.assertTrue(mx.all(mx.isfinite(values)))

    def test_resolve_params(self):
        self.assertEqual(
            dynamic_quant._resolve_quant_params(
                "affine", None, None, default_bits=4, default_group_size=64
            ),
            (4, 64),
        )
        self.assertEqual(
            dynamic_quant._resolve_quant_params(
                "mxfp4", None, None, default_bits=4, default_group_size=64
            ),
            (4, 32),
        )
        with self.assertRaises(ValueError):
            dynamic_quant._resolve_quant_params(
                "mxfp4", 8, 32, default_bits=4, default_group_size=64
            )

    def test_mxfp4_mxfp8_threshold(self):
        model = TinyModel()
        sensitivities = {"low": -1.0, "high": 1.0}
        threshold = dynamic_quant.estimate_threshold(
            model,
            sensitivities,
            5.0,
            4,
            32,
            8,
            32,
            low_mode="mxfp4",
            high_mode="mxfp8",
        )
        self.assertLessEqual(threshold, 1.0)

        def predicate(path, module):
            if sensitivities[path] > threshold:
                return {"bits": 8, "group_size": 32, "mode": "mxfp8"}
            return True

        model, config = quantize_model(
            model,
            {},
            group_size=32,
            bits=4,
            mode="mxfp4",
            quant_predicate=predicate,
        )
        self.assertEqual(model.low.mode, "mxfp4")
        self.assertEqual(model.high.mode, "mxfp8")
        self.assertEqual(config["quantization"]["mode"], "mxfp4")
        self.assertEqual(
            config["quantization"]["high"],
            {"bits": 8, "group_size": 32, "mode": "mxfp8"},
        )
        self.assertGreater(compute_bits_per_weight(model), 4.2)

    def test_estimate_mxfp4_sensitivities(self):
        model = TokenModel()
        data = mx.random.normal((2, 4, 64))
        sensitivities = estimate_sensitivities(
            model,
            data,
            4,
            32,
            8,
            32,
            low_mode="mxfp4",
            high_mode="mxfp8",
            batch_size=2,
        )
        self.assertEqual(len(sensitivities), 1)
        self.assertEqual(sensitivities[0][0], "fc")
        self.assertTrue(mx.all(mx.isfinite(mx.array(sensitivities[0][1]))))

    def test_mxfp4_skips_incompatible_layers(self):
        model = SmallModel()
        data = mx.random.uniform(0, 256, (2, 8)).astype(mx.uint32)
        sensitivities = estimate_sensitivities(
            model,
            data,
            4,
            32,
            8,
            32,
            low_mode="mxfp4",
            high_mode="mxfp8",
            batch_size=2,
        )
        self.assertEqual(sensitivities, [])

        def predicate(path, module):
            return dynamic_quant._can_quantize(module, 32, 32, "mxfp4", "mxfp8")

        model, config = quantize_model(
            model,
            {},
            group_size=32,
            bits=4,
            mode="mxfp4",
            quant_predicate=predicate,
        )
        self.assertFalse(hasattr(model.fc, "scales"))
        self.assertEqual(model.fc.weight.dtype, mx.float32)
        self.assertEqual(config["quantization"]["mode"], "mxfp4")


if __name__ == "__main__":
    unittest.main()
