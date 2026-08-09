from __future__ import annotations
import numpy as np

def evaluate_extra(cfg, run_dir, y, prob):
    # substitutability proxy: fraction of heads with low "KL" stand-in
    kl = np.abs(prob - 0.5)
    frac = float(np.mean(kl < 0.05))
    return {
        "substitutable_head_fraction": frac,
        "mean_kl_proxy": float(kl.mean()),
        "parameters_removed": 0,
        "notes": "E01 sweep proxy on synthetic features; real KL requires GPT-2 surgery path",
    }

