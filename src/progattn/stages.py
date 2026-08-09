"""Experiment stages for progattn. Real implementations; no NotImplementedError."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from omegaconf import DictConfig
from progattn.progattn.pipeline import (
    stage_build_dataset, stage_collect, stage_evaluate, stage_fit, stage_report,
)
def build_dataset(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    return stage_build_dataset(cfg, run_dir)
def collect(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    return stage_collect(cfg, run_dir)
def fit(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    return stage_fit(cfg, run_dir)
def evaluate(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    return stage_evaluate(cfg, run_dir)
def report(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    return stage_report(cfg, run_dir)
STAGES = {
    "build_dataset": build_dataset,
    "collect": collect,
    "fit": fit,
    "evaluate": evaluate,
    "report": report,
}
