"""Evaluation harnesses: real NLL/perplexity on loaded model + proxy task suite."""

from __future__ import annotations

from typing import Any

import numpy as np

from .model_runtime import try_load_causal_lm


def synthetic_lm_batch(*, n: int = 32, seq_len: int = 64, vocab: int = 1000, seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    tokens = rng.integers(0, vocab, size=(n, seq_len))
    return {"tokens": tokens.tolist(), "n": n, "seq_len": seq_len, "vocab": vocab}


def perplexity_from_nll(nlls: list[float]) -> float:
    return float(np.exp(np.mean(nlls))) if nlls else float("inf")


def evaluate_proxy_suite(schedule: dict[str, Any], *, seed: int = 0) -> dict[str, Any]:
    """Cheap proxies for LM / LAMBADA / BLiMP / IOI retention after replacements.

    ``wikitext_ppl`` here is a schedule-cost proxy kept for backward-compatible
    unit tests; the measured path uses ``measure_perplexity`` / ``evaluate_suite``.
    """
    rng = np.random.default_rng(seed)
    cost = float(schedule.get("final_joint_kl", 0.0))
    base_ppl = 20.0
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
        "task_metrics_mode": "proxy",
    }


def measure_perplexity(
    *,
    model_name: str,
    revision: str | None = None,
    texts: list[str] | None = None,
    force_synthetic: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    """Next-token NLL perplexity on a loaded model (fail closed unless synthetic)."""
    if force_synthetic or not model_name or model_name in {"x", "none", "synthetic", "missing"}:
        rng = np.random.default_rng(seed)
        nlls = list(rng.normal(3.0, 0.1, size=16))
        return {
            "wikitext_ppl": perplexity_from_nll(nlls),
            "mean_nll": float(np.mean(nlls)),
            "n_tokens": 16,
            "mode": "synthetic",
            "is_synthetic": True,
            "note": "Synthetic NLL for smoke only; not a model measurement.",
        }

    runtime = try_load_causal_lm(model_name, revision=revision, force_synthetic=False)
    if runtime is None:
        raise RuntimeError(
            f"Could not load {model_name!r} for perplexity. "
            "Set force_synthetic=true for smoke only."
        )

    import torch
    import torch.nn.functional as F

    if not texts:
        texts = [
            "The quick brown fox jumps over the lazy dog near the river bank.",
            "Children played outside while birds flew over the quiet village.",
            "Scientists measured the signal carefully before writing the report.",
            "A small language model can still reveal attention head structure.",
        ]

    nlls: list[float] = []
    n_tokens = 0
    with torch.no_grad():
        for text in texts:
            enc = runtime.tokenizer(text, return_tensors="pt")
            enc = {k: v.to(runtime.device) for k, v in enc.items()}
            input_ids = enc["input_ids"]
            if input_ids.shape[-1] < 2:
                continue
            out = runtime.model(input_ids=input_ids)
            logits = out.logits[:, :-1, :]
            targets = input_ids[:, 1:]
            log_probs = F.log_softmax(logits, dim=-1)
            token_nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            nlls.extend(token_nll.float().cpu().numpy().reshape(-1).tolist())
            n_tokens += int(token_nll.numel())

    return {
        "wikitext_ppl": perplexity_from_nll(nlls),
        "mean_nll": float(np.mean(nlls)) if nlls else float("inf"),
        "n_tokens": n_tokens,
        "mode": "model",
        "is_synthetic": False,
        "model_name": model_name,
        "revision": runtime.revision,
        "note": "Local next-token NLL on held-out short texts (pilot substitute for full WikiText stream).",
    }


def evaluate_suite(
    schedule: dict[str, Any],
    *,
    seed: int = 0,
    model_name: str = "synthetic",
    revision: str | None = None,
    force_synthetic: bool = False,
) -> dict[str, Any]:
    ppl = measure_perplexity(
        model_name=model_name,
        revision=revision,
        force_synthetic=force_synthetic,
        seed=seed,
    )
    proxy = evaluate_proxy_suite(schedule, seed=seed)
    return {
        **proxy,
        "wikitext_ppl": ppl["wikitext_ppl"],
        "mean_nll": ppl["mean_nll"],
        "ppl_mode": ppl["mode"],
        "ppl_is_synthetic": ppl["is_synthetic"],
        "ppl_note": ppl["note"],
        "n_replaced": schedule.get("n_replaced", 0),
    }
