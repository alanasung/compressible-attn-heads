"""Parameter, FLOP, and measured wall-clock accounting (M4-honest)."""

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
        "measured": True,
    }


def measure_forward_wall_clock(
    runtime: Any,
    *,
    text: str = "The capital of France is Paris and the river is the Seine.",
    warmup: int = 1,
    reps: int = 3,
    device_label: str | None = None,
) -> dict[str, Any]:
    """Time eager forward passes on a loaded/monkeypatched model (CPU/MPS/CUDA)."""
    import torch

    tok = runtime.tokenizer
    model = runtime.model
    enc = tok(text, return_tensors="pt")
    enc = {k: v.to(runtime.device) for k, v in enc.items()}
    # Warmup
    with torch.no_grad():
        for _ in range(max(0, warmup)):
            _ = model(**enc)
        if str(runtime.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif str(runtime.device) == "mps" and hasattr(torch, "mps"):
            try:
                torch.mps.synchronize()
            except Exception:  # noqa: BLE001
                pass
        t0 = time.perf_counter()
        for _ in range(max(1, reps)):
            _ = model(**enc)
        if str(runtime.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif str(runtime.device) == "mps" and hasattr(torch, "mps"):
            try:
                torch.mps.synchronize()
            except Exception:  # noqa: BLE001
                pass
        elapsed = time.perf_counter() - t0
    return {
        "forward_seconds_total": float(elapsed),
        "forward_seconds_per_rep": float(elapsed / max(1, reps)),
        "reps": int(reps),
        "warmup": int(warmup),
        "device": device_label or str(getattr(runtime, "device", "unknown")),
        "measured": True,
        "label": "measured eager forward wall-clock (not a production kernel claim)",
    }


def compare_wall_clock(
    *,
    seconds_before: float | None,
    seconds_after: float | None,
    measured: bool,
) -> dict[str, Any]:
    """Speedup claims require measured before/after timings."""
    if (
        not measured
        or seconds_before is None
        or seconds_after is None
        or seconds_before <= 0
        or seconds_after <= 0
    ):
        return {
            "speedup": None,
            "claims_speedup": False,
            "measured": False,
            "note": "No speedup claim without measured before/after wall-clock timings.",
        }
    speedup = float(seconds_before / seconds_after)
    return {
        "seconds_before": float(seconds_before),
        "seconds_after": float(seconds_after),
        "speedup": speedup,
        "claims_speedup": bool(speedup > 1.0 + 1e-6),
        "measured": True,
        "note": (
            "Speedup claimed only from measured timings."
            if speedup > 1.0 + 1e-6
            else "Measured timings do not show speedup; claim withheld."
        ),
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
    wall_clock_before: dict[str, Any] | None = None,
    wall_clock_after: dict[str, Any] | None = None,
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
    # Prefer measured model forwards; fall back to eager matmul diagnostic only.
    if wall_clock_before and wall_clock_after and wall_clock_before.get("measured"):
        wall = {
            "before": wall_clock_before,
            "after": wall_clock_after,
            "label": "measured_forward",
            "measured": True,
        }
        cmp = compare_wall_clock(
            seconds_before=wall_clock_before.get("forward_seconds_per_rep"),
            seconds_after=wall_clock_after.get("forward_seconds_per_rep"),
            measured=True,
        )
    else:
        wall = measure_eager_matmul(size=min(256, seq_len * 4), reps=3)
        wall["measured"] = True
        wall["label"] = wall.get("label", "eager-matmul")
        # Eager matmul is a diagnostic lower bound, NOT a surgery speedup claim.
        cmp = compare_wall_clock(
            seconds_before=None,
            seconds_after=None,
            measured=False,
        )
        cmp["note"] = (
            "Eager matmul diagnostic only; never claims surgery/masking speedup "
            "without measured before/after forwards."
        )
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
        "wall_clock_comparison": cmp,
        "claims_speedup": bool(cmp.get("claims_speedup", False)),
        "note": (
            "Parameter reduction reported only for structural_removal. "
            "Masking reports params_removed=0. "
            "Speedup claimed only with measured before/after wall-clock."
            if structural
            else "Masking≠removal: params_removed forced to 0. No speedup without measured timings."
        ),
    }
