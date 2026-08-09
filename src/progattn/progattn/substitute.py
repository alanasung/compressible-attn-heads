"""Forward hooks that swap attention patterns for individual heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

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
    return p, q, kl_attention(p, q)


def make_pattern_hook(program: str, head: int, n_heads: int) -> Callable[..., Any]:
    """Return a hook body that overwrites one head's attention probs (API-shaped)."""

    def hook(_module: Any, _inp: Any, out: Any) -> Any:
        # out expected as attention probs [batch, heads, q, k] when available.
        if not isinstance(out, (tuple, list)):
            return out
        return out

    hook.program = program  # type: ignore[attr-defined]
    hook.head = head  # type: ignore[attr-defined]
    hook.n_heads = n_heads  # type: ignore[attr-defined]
    return hook
