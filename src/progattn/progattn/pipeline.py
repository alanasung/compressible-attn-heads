"""Domain stages: E01 sweep first, then schedule/surgery/efficiency."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig

from ._util import ensure_dir, read_json, stage_result, write_json
from .efficiency import efficiency_report
from .patterns import generate_pattern
from .relax import compare_relaxations
from .schedule import greedy_schedule
from .substitute import collect_model_attentions, synthetic_clean_patterns
from .surgery import demo_surgery, live_gpt2_surgery
from .sweep import run_e01_sweep
from .tasks import evaluate_suite, synthetic_lm_batch


def _seed(cfg: DictConfig) -> int:
    return int(getattr(cfg.run, "seed", 0))


def _force_synthetic(cfg: DictConfig) -> bool:
    if bool(getattr(cfg, "force_synthetic", False)):
        return True
    exp = getattr(cfg, "experiment", None)
    if exp is not None and str(getattr(exp, "name", "")) == "smoke":
        return True
    run = getattr(cfg, "run", None)
    if run is not None and str(getattr(run, "profile", "")) in {"smoke", "debug"}:
        return True
    data = getattr(cfg, "data", None)
    if data is not None and str(getattr(data, "name", "")) == "synthetic":
        return True
    return False


def _model_name(cfg: DictConfig) -> str:
    model = getattr(cfg, "model", None)
    return str(getattr(model, "name", "openai-community/gpt2") if model is not None else "openai-community/gpt2")


def _revision(cfg: DictConfig) -> str | None:
    model = getattr(cfg, "model", None)
    rev = getattr(model, "revision", None) if model is not None else None
    return str(rev) if rev not in (None, "", "null") else None


def stage_build_dataset(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    n = int(getattr(cfg.data, "n_items", 128))
    force = _force_synthetic(cfg)
    batch = synthetic_lm_batch(n=n, seq_len=32, seed=_seed(cfg))
    out = ensure_dir(run_dir / "artifacts" / "dataset")
    write_json(out / "lm_batch.json", batch)
    # GPT-2 small arch is the E01 subject; measured collect may overwrite.
    arch = {
        "n_layers": 12,
        "n_heads": 12,
        "seq_len": 32,
        "note": "valid layers are 0..11; measured path may refine from loaded config",
        "model_name": _model_name(cfg),
        "revision": _revision(cfg),
        "force_synthetic": force,
    }
    write_json(out / "arch.json", arch)
    payload = stage_result(
        task="build_dataset",
        seed=_seed(cfg),
        n=n,
        metrics={"seq_len": 32, **{k: arch[k] for k in ("n_layers", "n_heads", "force_synthetic")}},
    )
    write_json(out / "results.json", payload)
    return payload


def stage_collect(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    arch = read_json(run_dir / "artifacts" / "dataset" / "arch.json")
    force = _force_synthetic(cfg)
    if force:
        clean = synthetic_clean_patterns(
            n_layers=int(arch["n_layers"]),
            n_heads=int(arch["n_heads"]),
            seq_len=int(arch["seq_len"]),
            seed=_seed(cfg),
        )
    else:
        clean = collect_model_attentions(
            model_name=_model_name(cfg),
            revision=_revision(cfg),
            seq_len=int(arch["seq_len"]),
            seed=_seed(cfg),
            force_synthetic=False,
        )
        # Persist measured arch.
        arch.update(
            {
                "n_layers": clean["n_layers"],
                "n_heads": clean["n_heads"],
                "seq_len": clean["seq_len"],
                "hidden": clean.get("hidden"),
                "head_dim": clean.get("head_dim"),
                "family": clean.get("family"),
            }
        )
        write_json(run_dir / "artifacts" / "dataset" / "arch.json", arch)

    out = ensure_dir(run_dir / "artifacts" / "collect")
    write_json(out / "clean_patterns.json", clean)
    payload = stage_result(
        task="collect",
        seed=_seed(cfg),
        n=clean["n_layers"] * clean["n_heads"],
        metrics={
            "mode": clean["mode"],
            "n_heads_total": clean["n_layers"] * clean["n_heads"],
            "force_synthetic": force,
            "is_synthetic": bool(clean.get("is_synthetic", force)),
        },
    )
    payload["is_synthetic"] = bool(clean.get("is_synthetic", force))
    write_json(out / "results.json", payload)
    return payload


def stage_fit(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    force = _force_synthetic(cfg)
    clean = read_json(run_dir / "artifacts" / "collect" / "clean_patterns.json")
    e01 = run_e01_sweep(
        n_layers=int(clean["n_layers"]),
        n_heads=int(clean["n_heads"]),
        seq_len=int(clean["seq_len"]),
        seed=_seed(cfg),
        clean=clean,
        force_synthetic=force,
    )
    schedule = greedy_schedule(e01, max_replace_frac=0.25)
    converted = [c["head"] for c in schedule["chosen"][:3]] or [0, 3, 6]
    if force:
        surgery = demo_surgery(converted=converted)
        surgery["mode"] = "synthetic"
        surgery["is_synthetic"] = True
    else:
        surgery = live_gpt2_surgery(
            model_name=_model_name(cfg),
            revision=_revision(cfg),
            layer=0,
            converted=converted,
            force_synthetic=False,
        )
    key = next(iter(clean["patterns"]))
    p = np.asarray(clean["patterns"][key], dtype=np.float64)
    q = generate_pattern("previous_token", clean["seq_len"])
    s = min(p.shape[0], q.shape[0])
    relax = compare_relaxations(p[:s, :s], q[:s, :s], steps=10)
    out = ensure_dir(run_dir / "artifacts" / "fit")
    write_json(
        out / "e01.json",
        {k: v for k, v in e01.items() if k != "rows"}
        | {"n_rows": len(e01["rows"]), "rows_head": e01["rows"][:20]},
    )
    write_json(out / "e01_rows.json", e01["rows"])
    write_json(out / "schedule.json", schedule)
    write_json(out / "surgery.json", surgery)
    write_json(out / "relax.json", relax)
    metrics = {
        **e01["metrics"],
        "n_replaced": schedule["n_replaced"],
        "params_removed": surgery["params_removed"],
        "surgery_kind": surgery.get("surgery_kind", "structural_removal"),
        "surgery_equivalent": surgery["equivalence"]["equivalent"],
        "masking_params_removed": surgery.get("masking_contrast", {}).get("params_removed", 0),
        "soft_final_distance": relax["soft_final_distance"],
        "surgery_mode": surgery.get("mode"),
    }
    payload = stage_result(task="fit", seed=_seed(cfg), n=e01["n_heads_total"], metrics=metrics)
    write_json(out / "results.json", payload)
    return payload


def stage_evaluate(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    force = _force_synthetic(cfg)
    schedule = read_json(run_dir / "artifacts" / "fit" / "schedule.json")
    surgery = read_json(run_dir / "artifacts" / "fit" / "surgery.json")
    tasks = evaluate_suite(
        schedule,
        seed=_seed(cfg),
        model_name=_model_name(cfg),
        revision=_revision(cfg),
        force_synthetic=force,
    )
    eff = efficiency_report(
        params_before=surgery["params_before"],
        params_after=surgery["params_after"],
        n_heads=int(read_json(run_dir / "artifacts" / "collect" / "clean_patterns.json")["n_heads"]),
        converted=len(surgery["converted_heads"]),
        surgery_kind=str(surgery.get("surgery_kind", "structural_removal")),
    )
    out = ensure_dir(run_dir / "artifacts" / "evaluate")
    write_json(out / "tasks.json", tasks)
    write_json(out / "efficiency.json", eff)
    metrics = {
        **tasks,
        "param_reduction_frac": eff["param_reduction_frac"],
        "qk_flops": eff["flops"]["qk_flops"],
        "claims_parameter_reduction": eff["claims_parameter_reduction"],
        "surgery_kind": eff["surgery_kind"],
    }
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
        "ppl_mode": ev["metrics"].get("ppl_mode"),
        "params_removed": fit["metrics"].get("params_removed"),
        "masking_params_removed": fit["metrics"].get("masking_params_removed"),
        "surgery_kind": fit["metrics"].get("surgery_kind"),
        "note": (
            "E01 gates later stages; masking is never reported as parameter removal; "
            "structural fused-QKV surgery counts real Q/K column deletion."
        ),
    }
    out = ensure_dir(run_dir / "artifacts" / "report")
    payload = stage_result(task="report", seed=_seed(cfg), n=1, metrics=metrics)
    write_json(out / "results.json", payload)
    return payload
