"""Compare per-tensor weights between EXO deepseek_v4_0731 loader and OMLX deepseek_v4 loader."""

from __future__ import annotations
from pathlib import Path
import json
import mlx.core as mx
import mlx.nn as nn
from exo.worker.engines.mlx.deepseek_v4_0731_loader import load_exo_model
from exo.worker.engines.mlx.deepseek_v4_0731_model import normalize_deepseek_v4_0731_weights
from mlx_lm.models.deepseek_v4 import Model as PinnedDeepseekV4Model, ModelArgs

MODEL_PATH = Path("/Users/jared/.cache/huggingface/hub/models/Jundot--DeepSeek-V4-Flash-0731-oQ4e-mtp")

def compare_weights():
    # 1. Load via EXO loader
    print("--- Loading EXO model ---")
    exo_model, _ = load_exo_model(MODEL_PATH, lazy=True)

    # 2. Load weights and sanitize directly
    print("--- Loading reference model ---")
    with open(MODEL_PATH / "config.json") as f:
        config = json.load(f)
    args = ModelArgs.from_dict(config)
    omlx_model = PinnedDeepseekV4Model(args)

    raw_weights = {}
    for p in MODEL_PATH.glob("*.safetensors"):
        raw_weights.update(mx.load(str(p)))

    # Apply 0731 norm + base sanitize
    norm_weights = normalize_deepseek_v4_0731_weights(raw_weights, args)
    sanitized = omlx_model.sanitize(norm_weights)
    omlx_model.load_weights(list(sanitized.items()))

    # Compare Layer 0 tensors
    exo_layer0 = exo_model.layers[0]
    omlx_layer0 = omlx_model.layers[0]

    tensors_to_check = [
        ("attn.wo_a.weight", getattr(exo_layer0.attn.wo_a, "weight", None), getattr(omlx_layer0.attn.wo_a, "weight", None)),
        ("attn.wo_a.scales", getattr(exo_layer0.attn.wo_a, "scales", None), getattr(omlx_layer0.attn.wo_a, "scales", None)),
        ("attn.wq_b.weight", getattr(exo_layer0.attn.wq_b, "weight", None), getattr(omlx_layer0.attn.wq_b, "weight", None)),
        ("attn.wq_b.scales", getattr(exo_layer0.attn.wq_b, "scales", None), getattr(omlx_layer0.attn.wq_b, "scales", None)),
        ("attn.attn_sink", getattr(exo_layer0.attn, "attn_sink", None), getattr(omlx_layer0.attn, "attn_sink", None)),
    ]

    for hc_attr in ["hc_attn", "hc_ffn", "attn_hc", "ffn_hc"]:
        exo_hc = getattr(exo_layer0, hc_attr, None)
        omlx_hc = getattr(omlx_layer0, hc_attr, None)
        if exo_hc is not None or omlx_hc is not None:
            tensors_to_check.append((
                f"{hc_attr}.gamma",
                getattr(exo_hc, "gamma", None) if exo_hc else None,
                getattr(omlx_hc, "gamma", None) if omlx_hc else None
            ))

    print("\n=== LAYER 0 TENSOR COMPARISON ===")
    for name, exo_arr, omlx_arr in tensors_to_check:
        print(f"\nTensor: {name}")
        if exo_arr is None:
            print("  EXO: None")
        else:
            print(f"  EXO: shape={tuple(exo_arr.shape)}, dtype={exo_arr.dtype}")
        if omlx_arr is None:
            print("  OMLX: None")
        else:
            print(f"  OMLX: shape={tuple(omlx_arr.shape)}, dtype={omlx_arr.dtype}")

        if isinstance(exo_arr, mx.array) and isinstance(omlx_arr, mx.array):
            if exo_arr.shape != omlx_arr.shape:
                print(f"  *** SHAPE MISMATCH: EXO {exo_arr.shape} vs OMLX {omlx_arr.shape}")
            else:
                mx.eval(exo_arr, omlx_arr)
                equal = bool(mx.array_equal(exo_arr, omlx_arr).item())
                diff = float(mx.max(mx.abs(exo_arr.astype(mx.float32) - omlx_arr.astype(mx.float32))).item())
                print(f"  Equal: {equal}, Max diff: {diff}")

if __name__ == "__main__":
    compare_weights()
