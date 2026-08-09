"""Domain stages: E01 sweep first, then schedule/surgery/efficiency."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig

from .efficiency import efficiency_report
from .patterns import generate_pattern
from .relax import compare_relaxations
from .schedule import greedy_schedule
from .substitute import synthetic_clean_patterns
from .surgery import demo_surgery
from .sweep import run_e01_sweep
from .tasks import evaluate_proxy_suite, synthetic_lm_batch
from ._util import ensure_dir, read_json, stage_result, write_json


def _seed(cfg: DictConfig) -> int:
    return int(getattr(cfg.run, "seed", 0))


def stage_build_dataset(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    n = int(getattr(cfg.data, "n_items", 128))
    batch = synthetic_lm_batch(n=n, seq_len=32, seed=_seed(cfg))
    out = ensure_dir(run_dir / "artifacts" / "dataset")
    write_json(out / "lm_batch.json", batch)
    # Also record GPT-2 shaped architecture for E01
    arch = {"n_layers": 12, "n_heads": 12, "seq_len": 32, "note": "valid layers are 0..11"}
    write_json(out / "arch.json", arch)
    payload = stage_result(task="build_dataset", seed=_seed(cfg), n=n, metrics={"seq_len": 32, **arch})
    write_json(out / "results.json", payload)
    return payload


def stage_collect(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    arch = read_json(run_dir / "artifacts" / "dataset" / "arch.json")
    clean = synthetic_clean_patterns(
        n_layers=int(arch["n_layers"]),
        n_heads=int(arch["n_heads"]),
        seq_len=int(arch["seq_len"]),
        seed=_seed(cfg),
    )
    out = ensure_dir(run_dir / "artifacts" / "collect")
    write_json(out / "clean_patterns.json", clean)
    payload = stage_result(
        task="collect",
        seed=_seed(cfg),
        n=clean["n_layers"] * clean["n_heads"],
        metrics={"mode": clean["mode"], "n_heads_total": clean["n_layers"] * clean["n_heads"]},
    )
    payload["is_synthetic"] = True
    write_json(out / "results.json", payload)
    return payload


def stage_fit(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    # E01 must report before anything else is motivated.
    e01 = run_e01_sweep(n_layers=12, n_heads=12, seq_len=32, seed=_seed(cfg))
    schedule = greedy_schedule(e01, max_replace_frac=0.25)
    surgery = demo_surgery(converted=[c["head"] for c in schedule["chosen"][:3]] or [0, 3, 6])
    clean = read_json(run_dir / "artifacts" / "collect" / "clean_patterns.json")
    key = next(iter(clean["patterns"]))
    p = np.asarray(clean["patterns"][key], dtype=np.float64)
    q = generate_pattern("previous_token", clean["seq_len"])
    relax = compare_relaxations(p, q, steps=10)
    out = ensure_dir(run_dir / "artifacts" / "fit")
    write_json(out / "e01.json", {k: v for k, v in e01.items() if k != "rows"} | {"n_rows": len(e01["rows"]), "rows_head": e01["rows"][:20]})
    write_json(out / "e01_rows.json", e01["rows"])
    write_json(out / "schedule.json", schedule)
    write_json(out / "surgery.json", surgery)
    write_json(out / "relax.json", relax)
    metrics = {
        **e01["metrics"],
        "n_replaced": schedule["n_replaced"],
        "params_removed": surgery["params_removed"],
        "surgery_equivalent": surgery["equivalence"]["equivalent"],
        "soft_final_distance": relax["soft_final_distance"],
    }
    payload = stage_result(task="fit", seed=_seed(cfg), n=e01["n_heads_total"], metrics=metrics)
    write_json(out / "results.json", payload)
    return payload


def stage_evaluate(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    schedule = read_json(run_dir / "artifacts" / "fit" / "schedule.json")
    surgery = read_json(run_dir / "artifacts" / "fit" / "surgery.json")
    tasks = evaluate_proxy_suite(schedule, seed=_seed(cfg))
    eff = efficiency_report(
        params_before=surgery["params_before"],
        params_after=surgery["params_after"],
        n_heads=12,
        converted=len(surgery["converted_heads"]),
    )
    out = ensure_dir(run_dir / "artifacts" / "evaluate")
    write_json(out / "tasks.json", tasks)
    write_json(out / "efficiency.json", eff)
    metrics = {**tasks, "param_reduction_frac": eff["param_reduction_frac"], "qk_flops": eff["flops"]["qk_flops"]}
    payload = stage_result(task="evaluate", seed=_seed(cfg), n=schedule["n_replaced"], metrics=metrics)
    write_json(out / "results.json", payload)
    return payload


def stage_report(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    fit = read_json(run_dir / "artifacts" / "fit" / "results.json")
    ev = read_json(run_dir / "artifacts" / "evaluate" / "results.json")
    metrics = {
        "e01_heavy_tailed": fit["metrics"].get("heavy_tailed"),
        "e01_n_cheap_heads": fit["metrics"].get("n_cheap_heads"),
        "deployment_premise_ok": fit["metrics"].get("deployment_premise_ok"),
        "wikitext_ppl": ev["metrics"].get("wikitext_ppl"),
        "params_removed": fit["metrics"].get("params_removed"),
        "note": "E01 gates later stages; masking is not reported as parameter removal",
    }
    out = ensure_dir(run_dir / "artifacts" / "report")
    payload = stage_result(task="report", seed=_seed(cfg), n=1, metrics=metrics)
    write_json(out / "results.json", payload)
    return payload
