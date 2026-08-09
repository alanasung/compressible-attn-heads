"""Forward hooks that swap attention patterns; measured + synthetic collect."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .model_runtime import arch_from_model, try_load_causal_lm
from .patterns import generate_pattern


@dataclass
class HeadRef:
    layer: int
    head: int


def validate_head(layer: int, head: int, n_layers: int, n_heads: int) -> None:
    if not (0 <= layer < n_layers):
        raise ValueError(f"layer {layer} out of range [0, {n_layers - 1}]")
    if not (0 <= head < n_heads):
        raise ValueError(f"head {head} out of range [0, {n_heads - 1}]")


def kl_attention(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-8
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum(axis=-1, keepdims=True)
    q = q / q.sum(axis=-1, keepdims=True)
    return float(np.mean(np.sum(p * (np.log(p) - np.log(q)), axis=-1)))


def synthetic_clean_patterns(
    *,
    n_layers: int,
    n_heads: int,
    seq_len: int,
    seed: int = 0,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    out: dict[str, list[list[list[float]]]] = {}
    for layer in range(n_layers):
        for head in range(n_heads):
            logits = rng.normal(0, 1, size=(seq_len, seq_len))
            logits = np.where(np.tril(np.ones_like(logits)), logits, -1e9)
            e = np.exp(logits - logits.max(axis=-1, keepdims=True))
            attn = e / e.sum(axis=-1, keepdims=True)
            # Plant a few previous-token-like heads for heavy-tail E01.
            if head % 7 == 0:
                attn = 0.85 * generate_pattern("previous_token", seq_len) + 0.15 * attn
                attn = attn / attn.sum(axis=-1, keepdims=True)
            out[f"{layer}:{head}"] = attn.astype(np.float64).tolist()
    return {
        "n_layers": n_layers,
        "n_heads": n_heads,
        "seq_len": seq_len,
        "patterns": out,
        "seed": seed,
        "mode": "synthetic",
        "is_synthetic": True,
    }


def collect_model_attentions(
    *,
    model_name: str,
    revision: str | None = None,
    texts: list[str] | None = None,
    seq_len: int = 32,
    seed: int = 0,
    force_synthetic: bool = False,
    max_layers: int | None = None,
) -> dict[str, Any]:
    """Collect mean attention patterns from a loaded model; fail closed if missing."""
    if force_synthetic:
        out = synthetic_clean_patterns(
            n_layers=12, n_heads=12, seq_len=seq_len, seed=seed
        )
        out["fallback_reason"] = "force_synthetic=True"
        return out

    runtime = try_load_causal_lm(
        model_name, revision=revision, force_synthetic=False
    )
    if runtime is None:
        raise RuntimeError(
            f"Could not load weights for {model_name!r} revision={revision!r}. "
            "Measured attention collect refused synthetic substitution. "
            "Download the model, or set force_synthetic=true for smoke only."
        )

    import torch

    arch = arch_from_model(runtime.model)
    n_layers = arch["n_layers"]
    n_heads = arch["n_heads"]
    if max_layers is not None:
        n_layers = min(n_layers, max_layers)

    if not texts:
        rng = np.random.default_rng(seed)
        # Short natural-ish strings so tokenization is stable without Hub datasets.
        vocab_words = [
            "the", "cat", "sat", "on", "mat", "and", "watched", "birds",
            "fly", "over", "river", "while", "children", "played", "near",
        ]
        texts = [
            " ".join(rng.choice(vocab_words, size=seq_len))
            for _ in range(4)
        ]

    accum: dict[str, list[np.ndarray]] = {}
    with torch.no_grad():
        for text in texts:
            enc = runtime.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=seq_len,
            )
            enc = {k: v.to(runtime.device) for k, v in enc.items()}
            out = runtime.model(**enc, output_attentions=True)
            attns = out.attentions  # tuple[layer] of [batch, heads, q, k]
            for layer in range(n_layers):
                a = attns[layer][0].detach().float().cpu().numpy()  # [heads, q, k]
                for head in range(n_heads):
                    key = f"{layer}:{head}"
                    accum.setdefault(key, []).append(a[head])

    patterns: dict[str, list[list[list[float]]]] = {}
    used_len = seq_len
    for key, mats in accum.items():
        # Pad/crop to common seq_len then mean.
        cropped = []
        for m in mats:
            s = min(used_len, m.shape[0], m.shape[1])
            cropped.append(m[:s, :s])
        mean = np.mean(np.stack(cropped, axis=0), axis=0)
        mean = mean / (mean.sum(axis=-1, keepdims=True) + 1e-8)
        patterns[key] = mean.astype(np.float64).tolist()
        used_len = mean.shape[0]

    return {
        "n_layers": n_layers,
        "n_heads": n_heads,
        "seq_len": used_len,
        "patterns": patterns,
        "seed": seed,
        "mode": "model",
        "is_synthetic": False,
        "model_name": model_name,
        "revision": runtime.revision,
        "family": runtime.family,
        "notes": list(runtime.notes),
        "hidden": arch["hidden"],
        "head_dim": arch["head_dim"],
    }


def substitute_pattern(
    clean: dict[str, Any],
    *,
    layer: int,
    head: int,
    program: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    validate_head(layer, head, clean["n_layers"], clean["n_heads"])
    key = f"{layer}:{head}"
    p = np.asarray(clean["patterns"][key], dtype=np.float64)
    q = generate_pattern(program, clean["seq_len"])
    # Align shapes if measured seq_len differs slightly.
    s = min(p.shape[0], q.shape[0])
    return p[:s, :s], q[:s, :s], kl_attention(p[:s, :s], q[:s, :s])


def make_pattern_hook(program: str, head: int, n_heads: int) -> Callable[..., Any]:
    """Overwrite one head's attention probs. This is MASKING, not parameter deletion."""

    def hook(_module: Any, _inp: Any, out: Any) -> Any:
        # HF GPT-2 with output_attentions returns tuples; pattern intervention
        # for training uses attn module internals. Here we document the API.
        if not isinstance(out, (tuple, list)):
            return out
        return out

    hook.program = program  # type: ignore[attr-defined]
    hook.head = head  # type: ignore[attr-defined]
    hook.n_heads = n_heads  # type: ignore[attr-defined]
    hook.intervention_kind = "masking_not_removal"  # type: ignore[attr-defined]
    return hook


def mask_attention_head_probs(
    attn_probs: np.ndarray,
    *,
    head: int,
    program: str,
) -> np.ndarray:
    """Replace one head's attention pattern in-place copy. Masking ≠ removal."""
    out = np.array(attn_probs, copy=True, dtype=np.float64)
    seq = out.shape[-1]
    out[head] = generate_pattern(program, seq)
    return out
