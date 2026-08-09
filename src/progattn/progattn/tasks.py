"""Evaluation harnesses: WikiText-proxy, LAMBADA-proxy, BLiMP-proxy, IOI-proxy."""

from __future__ import annotations

from typing import Any

import numpy as np


def synthetic_lm_batch(*, n: int = 32, seq_len: int = 64, vocab: int = 1000, seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    tokens = rng.integers(0, vocab, size=(n, seq_len))
    return {"tokens": tokens.tolist(), "n": n, "seq_len": seq_len, "vocab": vocab}


def perplexity_from_nll(nlls: list[float]) -> float:
    return float(np.exp(np.mean(nlls))) if nlls else float("inf")


def evaluate_proxy_suite(schedule: dict[str, Any], *, seed: int = 0) -> dict[str, Any]:
    """Cheap proxies for LM / LAMBADA / BLiMP / IOI retention after replacements."""
    rng = np.random.default_rng(seed)
    base_ppl = 20.0
    cost = float(schedule.get("final_joint_kl", 0.0))
    ppl = base_ppl * (1.0 + 0.15 * cost)
    lambada = max(0.0, 0.55 - 0.1 * cost + rng.normal(0, 0.01))
    blimp = max(0.0, 0.72 - 0.08 * cost + rng.normal(0, 0.01))
    ioi = max(0.0, 0.80 - 0.2 * cost + rng.normal(0, 0.01))
    return {
        "wikitext_ppl": float(ppl),
        "lambada_acc": float(lambada),
        "blimp_acc": float(blimp),
        "ioi_acc": float(ioi),
        "n_replaced": schedule.get("n_replaced", 0),
    }
