"""Evaluation harnesses: real NLL/perplexity + local downstream fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .model_runtime import try_load_causal_lm

_FIXTURE_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "fixtures" / "downstream_mini.json"
)


def synthetic_lm_batch(*, n: int = 32, seq_len: int = 64, vocab: int = 1000, seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    tokens = rng.integers(0, vocab, size=(n, seq_len))
    return {"tokens": tokens.tolist(), "n": n, "seq_len": seq_len, "vocab": vocab}


def perplexity_from_nll(nlls: list[float]) -> float:
    return float(np.exp(np.mean(nlls))) if nlls else float("inf")


def load_downstream_fixtures(path: Path | None = None) -> dict[str, Any]:
    p = path or _FIXTURE_PATH
    with p.open() as f:
        return json.load(f)


def evaluate_proxy_suite(schedule: dict[str, Any], *, seed: int = 0) -> dict[str, Any]:
    """Cheap proxies for LM / LAMBADA / BLiMP / IOI retention after replacements.

    ``wikitext_ppl`` here is a schedule-cost proxy kept for backward-compatible
    unit tests; the measured path uses ``measure_perplexity`` / ``evaluate_suite``.
    ``claims_downstream`` is always false for proxy mode.
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
        "claims_downstream": False,
        "note": "Schedule-cost proxy; not a downstream claim.",
    }


def _mean_nll(runtime: Any, text: str) -> float | None:
    import torch
    import torch.nn.functional as F

    enc = runtime.tokenizer(text, return_tensors="pt")
    enc = {k: v.to(runtime.device) for k, v in enc.items()}
    input_ids = enc["input_ids"]
    if input_ids.shape[-1] < 2:
        return None
    with torch.no_grad():
        out = runtime.model(input_ids=input_ids)
        logits = out.logits[:, :-1, :]
        targets = input_ids[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        token_nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return float(token_nll.mean().item())


def _next_token_logprob(runtime: Any, prompt: str, continuation: str) -> float | None:
    """Average log-prob of ``continuation`` tokens given ``prompt``."""
    import torch
    import torch.nn.functional as F

    tok = runtime.tokenizer
    prompt_ids = tok(prompt, return_tensors="pt")["input_ids"].to(runtime.device)
    full = tok(prompt + " " + continuation, return_tensors="pt")["input_ids"].to(runtime.device)
    if full.shape[-1] <= prompt_ids.shape[-1]:
        return None
    with torch.no_grad():
        out = runtime.model(input_ids=full)
        logits = out.logits[:, :-1, :]
        targets = full[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        token_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    # Score only continuation span.
    start = max(0, int(prompt_ids.shape[-1]) - 1)
    cont = token_lp[0, start:]
    if cont.numel() == 0:
        return None
    return float(cont.mean().item())


def evaluate_local_fixture_suite(
    schedule: dict[str, Any],
    *,
    model_name: str,
    revision: str | None = None,
    force_synthetic: bool = False,
    fixture_path: Path | None = None,
    runtime: Any | None = None,
) -> dict[str, Any]:
    """Score bundled LAMBADA/BLiMP/IOI-style fixtures on a loaded model.

    ``claims_downstream`` is True only for ``task_metrics_mode=local_fixture``.
    Smoke / synthetic path keeps proxy metrics and refuses the claim.
    """
    if force_synthetic or not model_name or model_name in {"x", "none", "synthetic", "missing"}:
        proxy = evaluate_proxy_suite(schedule, seed=0)
        proxy["task_metrics_mode"] = "proxy"
        proxy["claims_downstream"] = False
        proxy["fixture_corpus"] = None
        return proxy

    if runtime is None:
        runtime = try_load_causal_lm(model_name, revision=revision, force_synthetic=False)
    if runtime is None:
        raise RuntimeError(
            f"Could not load {model_name!r} for local fixture suite. "
            "Set force_synthetic=true for smoke only."
        )

    fixtures = load_downstream_fixtures(fixture_path)

    lambada_hits = 0
    lambada_n = 0
    for row in fixtures.get("lambada", []):
        ctx = str(row["context"])
        target = str(row["target"])
        lp = _next_token_logprob(runtime, ctx, target)
        # Prefer target over a wrong distractor built from reversed letters.
        distractor = target[::-1] if len(target) > 1 else "zzz"
        lp_bad = _next_token_logprob(runtime, ctx, distractor)
        if lp is None or lp_bad is None:
            continue
        lambada_n += 1
        lambada_hits += int(lp > lp_bad)

    blimp_hits = 0
    blimp_n = 0
    for row in fixtures.get("blimp", []):
        good = _mean_nll(runtime, str(row["good"]))
        bad = _mean_nll(runtime, str(row["bad"]))
        if good is None or bad is None:
            continue
        blimp_n += 1
        blimp_hits += int(good < bad)  # lower NLL = preferred

    ioi_hits = 0
    ioi_n = 0
    for row in fixtures.get("ioi", []):
        prompt = str(row["prompt"])
        lp_ok = _next_token_logprob(runtime, prompt, str(row["correct"]))
        lp_bad = _next_token_logprob(runtime, prompt, str(row["incorrect"]))
        if lp_ok is None or lp_bad is None:
            continue
        ioi_n += 1
        ioi_hits += int(lp_ok > lp_bad)

    scored = lambada_n + blimp_n + ioi_n
    if scored == 0:
        raise RuntimeError(
            "Local fixture suite scored 0 items; tokenizer/model produced no usable spans."
        )

    return {
        "lambada_acc": float(lambada_hits / max(lambada_n, 1)),
        "blimp_acc": float(blimp_hits / max(blimp_n, 1)),
        "ioi_acc": float(ioi_hits / max(ioi_n, 1)),
        "lambada_n": lambada_n,
        "blimp_n": blimp_n,
        "ioi_n": ioi_n,
        "n_replaced": schedule.get("n_replaced", 0),
        "task_metrics_mode": "local_fixture",
        "claims_downstream": True,
        "fixture_corpus": fixtures.get("corpus", "local_downstream_mini"),
        "model_name": model_name,
        "revision": getattr(runtime, "revision", revision),
        "note": "Hub-free local fixture suite; claims_downstream=true only in this mode.",
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

    nlls: list[float] = []  # type: ignore[no-redef]
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
    runtime: Any | None = None,
) -> dict[str, Any]:
    ppl = measure_perplexity(
        model_name=model_name,
        revision=revision,
        force_synthetic=force_synthetic,
        seed=seed,
    )
    if force_synthetic:
        tasks = evaluate_proxy_suite(schedule, seed=seed)
    else:
        tasks = evaluate_local_fixture_suite(
            schedule,
            model_name=model_name,
            revision=revision,
            force_synthetic=False,
            runtime=runtime,
        )
    return {
        **tasks,
        "wikitext_ppl": ppl["wikitext_ppl"],
        "mean_nll": ppl["mean_nll"],
        "ppl_mode": ppl["mode"],
        "ppl_is_synthetic": ppl["is_synthetic"],
        "ppl_note": ppl["note"],
        "n_replaced": schedule.get("n_replaced", 0),
        "claims_downstream": bool(tasks.get("claims_downstream", False)),
        "task_metrics_mode": tasks.get("task_metrics_mode", "proxy"),
    }
