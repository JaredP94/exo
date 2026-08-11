#!/usr/bin/env python3
"""Do the two GPUs agree on attention? The invariant the float32 fix rests on.

The DeepSeek V4 0731 decode collapse was caused by bfloat16
`scaled_dot_product_attention` producing materially different output on M4 Max
versus M5 Max — around 2.9 in max-abs from bitwise-identical q/k/v. Softmax is
contractive and cannot amplify identical inputs by that much, so the two devices
are not merely rounding differently; they are taking different kernel paths. EXO
works around it by running attention in float32.

That makes cross-device float32 agreement a load-bearing invariant rather than a
nicety. An MLX upgrade that reshuffles kernel selection could break it, and the
symptom would be another week of incoherent output. This checks it directly.

Run on the two-node cluster, one process per machine:

    mlx.launch --hosts <m4-host>,<m5-host> scripts/check_cross_device_sdpa.py

Running both ranks on ONE host makes the check vacuous — same GPU, so agreement
is trivial. The script cannot detect that, so it prints the hostname per rank for
you to confirm they differ.

Exit code is non-zero if float32 agreement exceeds tolerance. The bfloat16 figure
is reported for context and does NOT gate: its divergence is the known upstream
defect, not a regression in EXO.

NOT TESTED ON HARDWARE: written without MLX available.
"""

from __future__ import annotations

import argparse
import socket
import sys
import zlib

import mlx.core as mx
from mlx_lm.models.base import scaled_dot_product_attention

# float32 reduction-order noise across devices was measured at ~2.6e-3. Ten times
# that leaves headroom for a different machine pairing without admitting the
# bfloat16 failure mode, which was three orders of magnitude larger.
DEFAULT_TOLERANCE = 2.5e-2


def deterministic(shape: tuple[int, ...], salt: int, dtype: mx.Dtype) -> mx.array:
    """Bit-identical values on any device, without using the RNG.

    `mx.random` is not guaranteed to produce the same bits across GPU
    architectures, which would make every difference below unattributable. Integer
    arithmetic is exact and IEEE division is exactly specified, so this is
    reproducible everywhere.
    """
    total = 1
    for dim in shape:
        total *= dim
    idx = mx.arange(total, dtype=mx.int32)
    numerator = ((idx * 7919 + salt * 104729) % 2003) - 1001
    return (numerator.astype(mx.float32) / 1001.0).reshape(shape).astype(dtype)


def _gather(value: mx.array, group: mx.distributed.Group) -> mx.array:
    return mx.distributed.all_gather(value.reshape(1, -1), group=group)


def _format_bfloat16_rank_summary(
    rank: int,
    minimum: float,
    maximum: float,
    all_finite: bool,
) -> str:
    return (
        f"rank {rank} bfloat16 output: min={minimum:.6g} max={maximum:.6g} "
        f"all_finite={all_finite}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    args = ap.parse_args()

    group = mx.distributed.init()
    n, rank = group.size(), group.rank()
    host = socket.gethostname()

    if n != 2:
        print(
            f"[rank {rank}] group size {n}, not 2. This check needs one process on "
            f"each machine; a single-rank run proves nothing.",
            file=sys.stderr,
        )
        return 1

    shape = (1, args.heads, args.seq_len, args.head_dim)
    scale = float(args.head_dim) ** -0.5

    # Confirm both ranks really are computing from the same inputs before
    # attributing any output difference to the kernel.
    q32 = deterministic(shape, 1, mx.float32)
    k32 = deterministic(shape, 2, mx.float32)
    v32 = deterministic(shape, 3, mx.float32)
    fingerprint = mx.stack([mx.sum(q32), mx.sum(k32), mx.sum(v32)]).astype(mx.float32)
    gathered_inputs = _gather(fingerprint, group)
    mx.eval(gathered_inputs)

    out32 = scaled_dot_product_attention(
        q32, k32, v32, cache=None, scale=scale, mask=None
    )
    bf = mx.bfloat16
    out16 = scaled_dot_product_attention(
        q32.astype(bf),
        k32.astype(bf),
        v32.astype(bf),
        cache=None,
        scale=scale,
        mask=None,
    ).astype(mx.float32)
    mx.eval(out32, out16)

    gathered32 = _gather(out32, group)
    gathered16 = _gather(out16, group)
    # crc32, not hash(): Python randomises string hashing per process, so two
    # processes on the SAME host would produce different values and the
    # vacuous-run warning below would never fire.
    hosts = mx.distributed.all_gather(
        mx.array([float(zlib.crc32(host.encode()) % 1_000_000)]).reshape(1, -1),
        group=group,
    )
    mx.eval(gathered32, gathered16, hosts)

    if rank != 0:
        return 0

    print(f"rank 0 host: {host}")
    same_host = bool(mx.all(hosts[0] == hosts[1]).item())
    if same_host:
        print(
            "  WARNING: both ranks report the same host fingerprint. If this is one\n"
            "  machine the comparison is vacuous — same GPU, trivial agreement."
        )

    inputs_match = bool(mx.all(gathered_inputs[0] == gathered_inputs[1]).item())
    print(f"inputs bit-identical across ranks: {inputs_match}")
    if not inputs_match:
        print(
            "  The two ranks did not build the same q/k/v, so no output difference\n"
            "  below is attributable to the attention kernel. Fix this first."
        )
        return 1

    d32 = float(mx.max(mx.abs(gathered32[0] - gathered32[1])).item())
    d16 = float(mx.max(mx.abs(gathered16[0] - gathered16[1])).item())
    scale_of = float(mx.mean(mx.abs(gathered32[0])).item())

    print(f"\nmean |attention output|      : {scale_of:.6g}")
    print(
        f"cross-device max-abs, float32 : {d32:.6g}   (gates, tol {args.tolerance:g})"
    )
    print(f"cross-device max-abs, bfloat16: {d16:.6g}   (context only)")
    for peer_rank in range(n):
        peer_bfloat16 = gathered16[peer_rank]
        minimum = float(mx.min(peer_bfloat16).item())
        maximum = float(mx.max(peer_bfloat16).item())
        all_finite = bool(mx.all(mx.isfinite(peer_bfloat16)).item())
        print(_format_bfloat16_rank_summary(peer_rank, minimum, maximum, all_finite))

    if d16 > 10 * args.tolerance:
        print(
            "\n  bfloat16 divergence is present, as expected. This is the upstream\n"
            "  defect EXO works around by running attention in float32; it is not a\n"
            "  regression. Worth an MLX issue and an MLX version pin."
        )
    else:
        print(
            "\n  bfloat16 now agrees across devices. Either the machines changed or\n"
            "  MLX fixed the kernel — re-evaluate whether the float32 cast, which\n"
            "  roughly doubles attention working memory, is still needed."
        )

    if d32 > args.tolerance:
        print(
            f"\nFAIL: float32 attention differs by {d32:.6g} across devices, above "
            f"{args.tolerance:g}.\nEXO's fix assumes float32 agreement; if that no "
            f"longer holds, 2-rank decode is unsound again regardless of the patch."
        )
        return 1
    print("\nPASS: float32 attention agrees across devices within tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
