"""Soft-blend and straight-through differentiable relaxations."""

from __future__ import annotations

from typing import Any

import numpy as np


def soft_blend(p_learned: np.ndarray, p_prog: np.ndarray, gate: float) -> np.ndarray:
    g = float(np.clip(gate, 0.0, 1.0))
    out = (1 - g) * p_learned + g * p_prog
    out = out / np.maximum(out.sum(axis=-1, keepdims=True), 1e-8)
    return out


def straight_through(p_learned: np.ndarray, p_prog: np.ndarray, hard: bool = True) -> np.ndarray:
    # Forward: programmatic; backward (simulated): learned surrogate.
    forward = p_prog if hard else soft_blend(p_learned, p_prog, 0.5)
    return forward


def anneal_gate(steps: int = 20, start: float = 0.0, end: float = 1.0) -> list[float]:
    if steps < 2:
        return [end]
    return [float(start + (end - start) * i / (steps - 1)) for i in range(steps)]


def compare_relaxations(
    p_learned: np.ndarray,
    p_prog: np.ndarray,
    *,
    steps: int = 20,
) -> dict[str, Any]:
    gates = anneal_gate(steps)
    soft_traj = []
    for g in gates:
        blended = soft_blend(p_learned, p_prog, g)
        # Stall detector: gate mid-range with learned still load-bearing.
        learned_mass = float(np.mean(np.abs(blended - p_prog)))
        soft_traj.append({"gate": g, "distance_to_prog": learned_mass})
    ste = straight_through(p_learned, p_prog, hard=True)
    return {
        "soft_blend_trajectory": soft_traj,
        "soft_final_distance": soft_traj[-1]["distance_to_prog"],
        "ste_distance": float(np.mean(np.abs(ste - p_prog))),
        "failure_modes": {
            "soft_blend": "gate can stall mid-range leaving learned QK load-bearing",
            "straight_through": "surrogate gradient biased vs forward programmatic pattern",
        },
    }
