"""Soft-blend and straight-through differentiable relaxations."""

from __future__ import annotations

from typing import Any

import numpy as np

from .patterns import generate_pattern


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
    """Pattern-space anneal demo (smoke / diagnostic). Not a live-weight claim."""
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
        "anneal_mode": "pattern_demo",
        "is_synthetic": True,
        "failure_modes": {
            "soft_blend": "gate can stall mid-range leaving learned QK load-bearing",
            "straight_through": "surrogate gradient biased vs forward programmatic pattern",
        },
    }


def live_soft_anneal(
    *,
    model_name: str,
    revision: str | None = None,
    layer: int = 0,
    head: int = 0,
    program: str = "previous_token",
    steps: int = 5,
    text: str = "The capital of France is",
    force_synthetic: bool = False,
    runtime: Any | None = None,
) -> dict[str, Any]:
    """Annealed soft blend of learned vs programmatic attention on a loaded model.

    Smoke keeps ``anneal_mode=pattern_demo``. Measured path stamps
    ``anneal_mode=live_weight`` with a next-token KL trajectory vs gate.
    """
    if force_synthetic or not model_name or model_name in {"x", "none", "synthetic", "missing"}:
        seq = 16
        p = generate_pattern("uniform", seq)
        q = generate_pattern(program, seq)
        demo = compare_relaxations(p, q, steps=steps)
        demo["anneal_mode"] = "pattern_demo"
        demo["is_synthetic"] = True
        demo["note"] = "Pattern-space anneal demo only; not live-weight."
        return demo

    from .model_runtime import arch_from_model, try_load_causal_lm
    from .substitute import gated_pattern_next_token_kl, validate_head

    if runtime is None:
        runtime = try_load_causal_lm(model_name, revision=revision, force_synthetic=False)
    if runtime is None:
        raise RuntimeError(
            f"Could not load {model_name!r} for live soft-anneal. "
            "Set force_synthetic=true for smoke only."
        )

    arch = arch_from_model(runtime.model)
    validate_head(layer, head, arch["n_layers"], arch["n_heads"])
    gates = anneal_gate(steps)
    traj: list[dict[str, Any]] = []
    for g in gates:
        kl = gated_pattern_next_token_kl(
            runtime=runtime,
            layer=layer,
            head=head,
            program=program,
            gate=g,
            text=text,
            n_heads=arch["n_heads"],
        )
        traj.append(
            {
                "gate": g,
                "next_token_kl": float(kl["next_token_kl"]),
                "distance_to_prog": float(1.0 - g),  # diagnostic complement
            }
        )

    return {
        "soft_blend_trajectory": traj,
        "soft_final_distance": traj[-1]["distance_to_prog"] if traj else None,
        "soft_final_next_token_kl": traj[-1]["next_token_kl"] if traj else None,
        "anneal_mode": "live_weight",
        "is_synthetic": False,
        "layer": layer,
        "head": head,
        "program": program,
        "model_name": model_name,
        "revision": getattr(runtime, "revision", revision),
        "note": "Live soft-anneal: next-token KL under gated pattern blend on loaded weights.",
        "failure_modes": {
            "soft_blend": "gate can stall mid-range leaving learned QK load-bearing",
            "straight_through": "surrogate gradient biased vs forward programmatic pattern",
        },
    }
