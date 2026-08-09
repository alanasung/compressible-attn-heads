"""Parameter, FLOP, and eager wall-clock accounting (M4-honest)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np


def count_attention_flops(seq_len: int, n_heads: int, head_dim: int, *, programmatic_heads: int = 0) -> dict[str, int]:
    # Learned heads: QK matmul 2*s^2*d per head roughly; programmatic skip QK.
    learned = n_heads - programmatic_heads
    qk = learned * seq_len * seq_len * head_dim
    av = n_heads * seq_len * seq_len * head_dim
    return {
        "qk_flops": int(qk),
        "av_flops": int(av),
        "total_attn_flops": int(qk + av),
        "programmatic_heads": int(programmatic_heads),
        "learned_heads": int(learned),
    }


def measure_eager_matmul(*, size: int = 512, reps: int = 5) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    a = rng.normal(size=(size, size))
    b = rng.normal(size=(size, size))
    t0 = time.perf_counter()
    for _ in range(reps):
        _ = a @ b
    elapsed = time.perf_counter() - t0
    return {
        "eager_matmul_seconds": float(elapsed),
        "label": "eager-mode lower bound, not a production CUDA/FlashAttention claim",
        "size": size,
        "reps": reps,
    }


def efficiency_report(*, params_before: int, params_after: int, n_heads: int, converted: int, seq_len: int = 64, head_dim: int = 64) -> dict[str, Any]:
    flops = count_attention_flops(seq_len, n_heads, head_dim, programmatic_heads=converted)
    wall = measure_eager_matmul(size=min(256, seq_len * 4), reps=3)
    return {
        "params_before": params_before,
        "params_after": params_after,
        "params_removed": params_before - params_after,
        "param_reduction_frac": (params_before - params_after) / max(params_before, 1),
        "flops": flops,
        "wall_clock": wall,
    }
