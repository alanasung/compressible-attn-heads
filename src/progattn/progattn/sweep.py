"""Per-head, per-program substitutability sweep (E01 first)."""

from __future__ import annotations

from typing import Any

import numpy as np

from .patterns import PROGRAMS
from .substitute import collect_model_attentions, substitute_pattern, synthetic_clean_patterns

E01_PROGRAMS = ("previous_token", "positional_offset", "bos_attend")


def run_e01_sweep(
    *,
    n_layers: int = 12,
    n_heads: int = 12,
    seq_len: int = 32,
    seed: int = 0,
    kl_threshold: float = 0.05,
    programs: tuple[str, ...] = E01_PROGRAMS,
    clean: dict[str, Any] | None = None,
    model_name: str | None = None,
    revision: str | None = None,
    force_synthetic: bool = True,
) -> dict[str, Any]:
    """Falsifiable core: is substitutability heavy-tailed?"""
    if n_layers <= 0 or n_heads <= 0:
        raise ValueError("n_layers and n_heads must be positive")
    if clean is None:
        if force_synthetic or not model_name:
            clean = synthetic_clean_patterns(
                n_layers=n_layers, n_heads=n_heads, seq_len=seq_len, seed=seed
            )
        else:
            clean = collect_model_attentions(
                model_name=model_name,
                revision=revision,
                seq_len=seq_len,
                seed=seed,
                force_synthetic=False,
            )
            n_layers = int(clean["n_layers"])
            n_heads = int(clean["n_heads"])
            seq_len = int(clean["seq_len"])
    rows: list[dict[str, Any]] = []
    for layer in range(n_layers):
        for head in range(n_heads):
            best_prog = None
            best_kl = float("inf")
            for prog in programs:
                _, _, kl = substitute_pattern(clean, layer=layer, head=head, program=prog)
                if kl < best_kl:
                    best_kl = kl
                    best_prog = prog
            rows.append(
                {
                    "layer": layer,
                    "head": head,
                    "best_program": best_prog,
                    "best_kl": float(best_kl),
                    "cheap": bool(best_kl < kl_threshold),
                }
            )
    kls = np.asarray([r["best_kl"] for r in rows], dtype=np.float64)
    n_cheap = int(sum(r["cheap"] for r in rows))
    frac = n_cheap / max(len(rows), 1)
    q10 = float(np.quantile(kls, 0.10))
    med = float(np.median(kls))
    heavy_tailed = bool(n_cheap >= 15 and med > kl_threshold)
    return {
        "experiment": "E01",
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_heads_total": n_layers * n_heads,
        "programs": list(programs),
        "kl_threshold": kl_threshold,
        "rows": rows,
        "metrics": {
            "n_cheap_heads": n_cheap,
            "frac_cheap": frac,
            "median_kl": med,
            "p10_kl": q10,
            "heavy_tailed": heavy_tailed,
            "deployment_premise_ok": bool(frac >= 0.15 or n_cheap >= 15),
            "pattern_mode": clean.get("mode", "unknown"),
            "is_synthetic": bool(clean.get("is_synthetic", clean.get("mode") == "synthetic")),
        },
        "clean": {
            "mode": clean.get("mode"),
            "seq_len": seq_len,
            "seed": seed,
            "is_synthetic": clean.get("is_synthetic"),
        },
    }


def full_program_sweep(clean: dict[str, Any], *, programs: tuple[str, ...] = PROGRAMS) -> dict[str, Any]:
    rows = []
    for layer in range(clean["n_layers"]):
        for head in range(clean["n_heads"]):
            for prog in programs:
                _, _, kl = substitute_pattern(clean, layer=layer, head=head, program=prog)
                rows.append({"layer": layer, "head": head, "program": prog, "kl": float(kl)})
    return {"rows": rows, "n": len(rows)}
