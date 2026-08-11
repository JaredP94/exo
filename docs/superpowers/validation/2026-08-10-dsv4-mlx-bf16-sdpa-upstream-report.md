# MLX bfloat16 SDPA cross-device divergence

## Summary

On an M4 Max and an M5 Max, bfloat16 `scaled_dot_product_attention` produced
materially different output from bitwise-identical inputs. The probe used shape
`(1, 8, 64, 128)` and inputs built by integer arithmetic in `[-1, 1]`.

The cross-device maximum absolute difference was `1.14062` in bfloat16 and
`0.000787497` in float32. Mean absolute output was `0.381306`. Both ranks were
finite and bounded: rank 0 was within `[-0.507812, 0.511719]` and rank 1 was
within `[-0.6875, 0.679688]`. This is neither overflow nor NaN. A 1.14 delta
between operands bounded near 0.69 requires near-opposite signs at the worst
element, while rank 1's range is consistently wider; the evidence points to a
structural kernel difference rather than ordinary reduction-order noise.

The reproducer is `scripts/check_cross_device_sdpa.py`, run with
`mlx.launch --hosts <m4>,<m5>`. Both hosts used MLX source commit `cc3f3e60`
from wheels built one day apart. The reported measurement was taken after
byte-copying one wheel to both hosts. Future reproduction must pin one
identical wheel, not only a source commit, because kernel selection is the
defect under discussion.

## Acceptance-run context note

The DSV4 acceptance run completed cold prefill measurements through 32K tokens.
The 64K attempt rebooted a node before completion, consistent with a context
RAM-pressure failure but without surviving host logs to prove OOM. Future work
should investigate OMLX TurboQuant as a way to reduce context-management RAM
before retrying 64K.
