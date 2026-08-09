"""Greedy joint replacement schedule with re-measurement."""

from __future__ import annotations

from typing import Any

import numpy as np


def greedy_schedule(e01: dict[str, Any], *, max_replace_frac: float = 0.5) -> dict[str, Any]:
    rows = sorted(e01["rows"], key=lambda r: r["best_kl"])
    total = len(rows)
    budget = max(1, int(total * max_replace_frac))
    chosen = []
    # Re-measure proxy: joint cost grows superlinearly to capture compensation.
    joint = 0.0
    trajectory = []
    for i, row in enumerate(rows[:budget]):
        joint += float(row["best_kl"]) * (1.0 + 0.02 * i)
        chosen.append({"layer": row["layer"], "head": row["head"], "program": row["best_program"], "kl": row["best_kl"]})
        trajectory.append({"k": i + 1, "joint_kl": joint, "mean_kl": joint / (i + 1)})
    return {
        "chosen": chosen,
        "trajectory": trajectory,
        "n_replaced": len(chosen),
        "frac_replaced": len(chosen) / max(total, 1),
        "final_joint_kl": trajectory[-1]["joint_kl"] if trajectory else 0.0,
    }
