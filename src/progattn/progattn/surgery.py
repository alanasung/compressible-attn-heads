"""Genuine GPT-2 fused c_attn Q/K column surgery (structural, not masking)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SurgeryPlan:
    layer: int
    converted_heads: list[int]
    n_heads: int
    head_dim: int
    hidden: int

    @property
    def surviving_heads(self) -> list[int]:
        conv = set(self.converted_heads)
        return [h for h in range(self.n_heads) if h not in conv]


def _slice_indices(heads: list[int], head_dim: int) -> np.ndarray:
    idxs = []
    for h in heads:
        start = h * head_dim
        idxs.extend(range(start, start + head_dim))
    return np.asarray(idxs, dtype=np.int64)


def surgically_narrow_c_attn(
    weight: np.ndarray,
    bias: np.ndarray | None,
    plan: SurgeryPlan,
) -> dict[str, Any]:
    """Slice Q and K columns for converted heads out of fused c_attn.

    GPT-2 packs [Q|K|V] along the out dimension. Genuine removal rebuilds a
    narrower projection for surviving Q/K heads while keeping V intact (or
    optionally narrowed — here V stays full so value pathways remain).
    """
    hidden, three = weight.shape[0], weight.shape[1]
    if three != 3 * plan.hidden:
        # allow weight shaped (hidden, 3*hidden)
        if weight.shape[0] == 3 * plan.hidden and weight.shape[1] == plan.hidden:
            weight = weight.T
            hidden, three = weight.shape[0], weight.shape[1]
        else:
            raise ValueError(
                f"expected fused c_attn shape (hidden, 3*hidden); got {weight.shape}"
            )
    q_end = plan.hidden
    k_end = 2 * plan.hidden
    q = weight[:, :q_end]
    k = weight[:, q_end:k_end]
    v = weight[:, k_end:]
    keep = _slice_indices(plan.surviving_heads, plan.head_dim)
    q_new = q[:, keep]
    k_new = k[:, keep]
    # Rebuild fused surviving QK + full V
    new_w = np.concatenate([q_new, k_new, v], axis=1)
    new_b = None
    if bias is not None:
        bq, bk, bv = bias[:q_end], bias[q_end:k_end], bias[k_end:]
        new_b = np.concatenate([bq[keep], bk[keep], bv], axis=0)
    params_before = int(weight.size + (0 if bias is None else bias.size))
    params_after = int(new_w.size + (0 if new_b is None else new_b.size))
    return {
        "weight": new_w,
        "bias": new_b,
        "params_before": params_before,
        "params_after": params_after,
        "params_removed": params_before - params_after,
        "surviving_heads": plan.surviving_heads,
        "converted_heads": list(plan.converted_heads),
        "note": "masking is not removal; parameter counts measured on rebuilt module",
    }


def numerical_equivalence_check(
    masked_logits: np.ndarray,
    surgical_logits: np.ndarray,
    *,
    atol: float = 1e-4,
) -> dict[str, Any]:
    max_abs = float(np.max(np.abs(masked_logits - surgical_logits)))
    return {
        "max_abs_diff": max_abs,
        "equivalent": bool(max_abs <= atol),
        "atol": atol,
    }


def demo_surgery(*, n_heads: int = 12, head_dim: int = 64, converted: list[int] | None = None) -> dict[str, Any]:
    converted = converted or [0, 3, 6]
    hidden = n_heads * head_dim
    rng = np.random.default_rng(0)
    w = rng.normal(0, 0.02, size=(hidden, 3 * hidden))
    b = rng.normal(0, 0.02, size=(3 * hidden,))
    plan = SurgeryPlan(layer=0, converted_heads=converted, n_heads=n_heads, head_dim=head_dim, hidden=hidden)
    result = surgically_narrow_c_attn(w, b, plan)
    # Equivalence on a toy projection: surviving columns match exactly.
    keep = _slice_indices(plan.surviving_heads, head_dim)
    toy_x = rng.normal(0, 1, size=(4, hidden))
    qk_surv_old = np.concatenate([toy_x @ w[:, keep], toy_x @ w[:, hidden + keep]], axis=-1)
    qk_surv_new = np.concatenate(
        [toy_x @ result["weight"][:, : len(keep)], toy_x @ result["weight"][:, len(keep) : 2 * len(keep)]],
        axis=-1,
    )
    eq = numerical_equivalence_check(qk_surv_old, qk_surv_new)
    result["equivalence"] = eq
    # Don't return giant arrays in metrics path
    return {
        "params_before": result["params_before"],
        "params_after": result["params_after"],
        "params_removed": result["params_removed"],
        "surviving_heads": result["surviving_heads"],
        "converted_heads": result["converted_heads"],
        "equivalence": eq,
    }
