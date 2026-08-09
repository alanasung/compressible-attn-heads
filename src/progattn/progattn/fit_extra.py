from __future__ import annotations
def fit_extra(cfg, run_dir, x, y, prob):
    return {"surgery_mode": "mask_not_removal_until_rebuild", "n_heads_considered": int(x.shape[0])}

