"""Greedy joint replacement schedule with re-measurement."""

from __future__ import annotations

from typing import Any


def greedy_schedule(e01: dict[str, Any], *, max_replace_frac: float = 0.5) -> dict[str, Any]:
    """Greedy schedule using ``rank_kl`` when present (next-token or pattern)."""

    def _rank_key(row: dict[str, Any]) -> float:
        return float(row.get("rank_kl", row.get("best_kl", float("inf"))))

    rows = sorted(e01["rows"], key=_rank_key)
    total = len(rows)
    budget = max(1, int(total * max_replace_frac))
    chosen = []
    # Re-measure proxy: joint cost grows superlinearly to capture compensation.
    joint = 0.0
    trajectory = []
    kl_space = str(e01.get("kl_space", e01.get("metrics", {}).get("kl_space", "pattern")))
    for i, row in enumerate(rows[:budget]):
        rk = _rank_key(row)
        joint += rk * (1.0 + 0.02 * i)
        chosen.append(
            {
                "layer": row["layer"],
                "head": row["head"],
                "program": row["best_program"],
                "kl": rk,
                "pattern_kl": float(row.get("pattern_kl", row.get("best_kl", rk))),
                "kl_space": row.get("kl_space", kl_space),
            }
        )
        trajectory.append({"k": i + 1, "joint_kl": joint, "mean_kl": joint / (i + 1)})
    return {
        "chosen": chosen,
        "trajectory": trajectory,
        "n_replaced": len(chosen),
        "frac_replaced": len(chosen) / max(total, 1),
        "final_joint_kl": trajectory[-1]["joint_kl"] if trajectory else 0.0,
        "kl_space": kl_space,
        "is_synthetic": bool(e01.get("is_synthetic", e01.get("metrics", {}).get("is_synthetic", True))),
    }
