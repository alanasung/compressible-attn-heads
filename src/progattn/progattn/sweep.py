"""Per-head, per-program substitutability sweep (E01 first)."""

from __future__ import annotations

from typing import Any

import numpy as np

from .patterns import PROGRAMS
from .substitute import (
    collect_model_attentions,
    intervention_next_token_kl,
    substitute_pattern,
    synthetic_clean_patterns,
)

E01_PROGRAMS = ("previous_token", "positional_offset", "bos_attend")

# M4 pilot cap on live next-token KL re-ranks.
_DEFAULT_LIVE_RERANK = 16


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
    live_rerank_budget: int = _DEFAULT_LIVE_RERANK,
) -> dict[str, Any]:
    """Falsifiable core: is substitutability heavy-tailed?

    Pattern-space KL schedules candidates. When a measured runtime is available,
    top candidates are re-ranked by live next-token KL under executable pattern
    intervention. Smoke / ``force_synthetic`` stays on pattern KL and stamps
    ``kl_space=pattern`` without claiming behavioral substitutability.
    """
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
    # Prefer arch from provided/collected clean patterns.
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
                    "pattern_kl": float(best_kl),
                    "rank_kl": float(best_kl),
                    "kl_space": "pattern",
                    "cheap": bool(best_kl < kl_threshold),
                }
            )

    is_synthetic = bool(clean.get("is_synthetic", clean.get("mode") == "synthetic"))
    use_next_token = (
        (not force_synthetic)
        and (not is_synthetic)
        and bool(model_name)
        and model_name not in {"x", "none", "synthetic", "missing"}
    )
    kl_space = "pattern"
    n_live = 0
    if use_next_token:
        ordered = sorted(rows, key=lambda r: float(r["pattern_kl"]))
        budget = max(0, min(int(live_rerank_budget), len(ordered)))
        for row in ordered[:budget]:
            live = intervention_next_token_kl(
                model_name=str(model_name),
                revision=revision,
                layer=int(row["layer"]),
                head=int(row["head"]),
                program=str(row["best_program"]),
                force_synthetic=False,
            )
            ntk = float(live.get("next_token_kl", float("nan")))
            row["next_token_kl"] = ntk
            if ntk == ntk:  # not NaN
                row["rank_kl"] = ntk
                row["kl_space"] = "next_token"
                n_live += 1
        if n_live > 0:
            kl_space = "next_token"

    kls = np.asarray([r["rank_kl"] for r in rows], dtype=np.float64)
    pattern_kls = np.asarray([r["pattern_kl"] for r in rows], dtype=np.float64)
    n_cheap = int(sum(r["cheap"] for r in rows))
    frac = n_cheap / max(len(rows), 1)
    q10 = float(np.quantile(pattern_kls, 0.10))
    med = float(np.median(pattern_kls))
    heavy_tailed = bool(n_cheap >= 15 and med > kl_threshold)
    behavioral_claim = bool(kl_space == "next_token" and not is_synthetic and n_live > 0)
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
            "median_rank_kl": float(np.median(kls)),
            "heavy_tailed": heavy_tailed,
            "deployment_premise_ok": bool(frac >= 0.15 or n_cheap >= 15),
            "pattern_mode": clean.get("mode", "unknown"),
            "is_synthetic": is_synthetic,
            "kl_space": kl_space,
            "n_live_next_token": n_live,
            "behavioral_substitutability_claimed": behavioral_claim,
        },
        "clean": {
            "mode": clean.get("mode"),
            "seq_len": seq_len,
            "seed": seed,
            "is_synthetic": is_synthetic,
        },
        "kl_space": kl_space,
        "is_synthetic": is_synthetic,
    }


def full_program_sweep(clean: dict[str, Any], *, programs: tuple[str, ...] = PROGRAMS) -> dict[str, Any]:
    rows = []
    for layer in range(clean["n_layers"]):
        for head in range(clean["n_heads"]):
            for prog in programs:
                _, _, kl = substitute_pattern(clean, layer=layer, head=head, program=prog)
                rows.append(
                    {
                        "layer": layer,
                        "head": head,
                        "program": prog,
                        "kl": float(kl),
                        "kl_space": "pattern",
                        "is_synthetic": bool(
                            clean.get("is_synthetic", clean.get("mode") == "synthetic")
                        ),
                    }
                )
    return {"rows": rows, "n": len(rows), "kl_space": "pattern"}
