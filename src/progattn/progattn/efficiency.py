"""Parameter, FLOP, and eager wall-clock accounting (M4-honest)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np


def count_attention_flops(
    seq_len: int,
    n_heads: int,
    head_dim: int,
    *,
    programmatic_heads: int = 0,
    structural_removal: bool | None = None,
) -> dict[str, Any]:
    # Learned heads: QK matmul; programmatic skip QK only if structurally removed
    # or a kernel exists. Masking alone does not skip the fused matmul.
    if structural_removal is None:
        # Backward-compatible default: callers that pass programmatic_heads
        # intend the structural-removal FLOP model.
        structural_removal = programmatic_heads > 0
    if structural_removal:
        learned = n_heads - programmatic_heads
        qk = learned * seq_len * seq_len * head_dim
        note = "FLOP estimate assumes structural Q/K removal for converted heads."
    else:
        learned = n_heads
        qk = learned * seq_len * seq_len * head_dim
        note = (
            "Masking does not reduce QK FLOPs in fused GPT-2 attention; "
            "full QK cost retained."
        )
    av = n_heads * seq_len * seq_len * head_dim
    return {
        "qk_flops": int(qk),
        "av_flops": int(av),
        "total_attn_flops": int(qk + av),
        "programmatic_heads": int(programmatic_heads),
        "learned_heads": int(learned),
        "structural_removal": bool(structural_removal),
        "note": note,
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


def efficiency_report(
    *,
    params_before: int,
    params_after: int,
    n_heads: int,
    converted: int,
    seq_len: int = 64,
    head_dim: int = 64,
    surgery_kind: str = "structural_removal",
) -> dict[str, Any]:
    structural = surgery_kind == "structural_removal"
    # Never claim param reduction for masking.
    if not structural:
        params_after = params_before
    flops = count_attention_flops(
        seq_len,
        n_heads,
        head_dim,
        programmatic_heads=converted if structural else 0,
        structural_removal=structural,
    )
    wall = measure_eager_matmul(size=min(256, seq_len * 4), reps=3)
    removed = params_before - params_after
    return {
        "params_before": params_before,
        "params_after": params_after,
        "params_removed": removed,
        "param_reduction_frac": removed / max(params_before, 1),
        "surgery_kind": surgery_kind,
        "claims_parameter_reduction": bool(structural and removed > 0),
        "flops": flops,
        "wall_clock": wall,
        "note": (
            "Parameter reduction reported only for structural_removal. "
            "Masking reports params_removed=0."
            if structural
            else "Masking≠removal: params_removed forced to 0."
        ),
    }
