"""P4: force_synthetic smoke-only; measured path monkeypatched (no Hub)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from progattn.progattn.pipeline import _force_synthetic, stage_collect, stage_fit
from progattn.progattn.substitute import collect_model_attentions
from progattn.progattn.surgery import demo_surgery, masking_report
from progattn.progattn.tasks import measure_perplexity
from progattn.progattn.efficiency import efficiency_report


def test_force_synthetic_smoke_only():
    smoke = SimpleNamespace(
        force_synthetic=True,
        experiment=SimpleNamespace(name="smoke"),
        data=SimpleNamespace(name="synthetic"),
        run=SimpleNamespace(profile="smoke"),
    )
    pilot = SimpleNamespace(
        force_synthetic=False,
        experiment=SimpleNamespace(name="pilot"),
        data=SimpleNamespace(name="pilot"),
        run=SimpleNamespace(profile="pilot"),
    )
    assert _force_synthetic(smoke) is True
    assert _force_synthetic(pilot) is False


def test_collect_respects_force_flag(tmp_path):
    cfg = SimpleNamespace(
        force_synthetic=True,
        experiment=SimpleNamespace(name="smoke"),
        run=SimpleNamespace(seed=0, profile="smoke"),
        data=SimpleNamespace(n_items=16, name="synthetic"),
        model=SimpleNamespace(name="openai-community/gpt2", revision="deadbeef"),
    )
    from progattn.progattn.pipeline import stage_build_dataset

    stage_build_dataset(cfg, tmp_path)
    with patch("progattn.progattn.substitute.try_load_causal_lm") as load:
        out = stage_collect(cfg, tmp_path)
        load.assert_not_called()
    assert out["is_synthetic"] is True
    assert out["metrics"]["force_synthetic"] is True


def test_measured_collect_fail_closed():
    with patch("progattn.progattn.substitute.try_load_causal_lm", return_value=None):
        try:
            collect_model_attentions(
                model_name="openai-community/gpt2",
                force_synthetic=False,
            )
            ok = False
        except RuntimeError:
            ok = True
        assert ok


def test_masking_not_removal_accounting():
    mask = masking_report(n_heads=12, converted=[0, 3], hidden=768, head_dim=64)
    assert mask["params_removed"] == 0
    assert mask["surgery_kind"] == "masking"
    structural = demo_surgery(converted=[0, 3])
    assert structural["params_removed"] > 0
    assert structural["masking_contrast"]["params_removed"] == 0
    eff_mask = efficiency_report(
        params_before=100,
        params_after=80,
        n_heads=12,
        converted=2,
        surgery_kind="masking",
    )
    assert eff_mask["params_removed"] == 0
    assert eff_mask["claims_parameter_reduction"] is False


def test_perplexity_synthetic_smoke():
    out = measure_perplexity(model_name="gpt2", force_synthetic=True, seed=0)
    assert out["is_synthetic"] is True
    assert out["wikitext_ppl"] > 1.0


def test_perplexity_fail_closed():
    with patch("progattn.progattn.tasks.try_load_causal_lm", return_value=None):
        try:
            measure_perplexity(model_name="openai-community/gpt2", force_synthetic=False)
            ok = False
        except RuntimeError:
            ok = True
        assert ok


def test_fit_records_masking_contrast(tmp_path):
    cfg = SimpleNamespace(
        force_synthetic=True,
        experiment=SimpleNamespace(name="smoke"),
        run=SimpleNamespace(seed=0, profile="smoke"),
        data=SimpleNamespace(n_items=16, name="synthetic"),
        model=SimpleNamespace(name="openai-community/gpt2", revision="deadbeef"),
    )
    from progattn.progattn.pipeline import stage_build_dataset

    stage_build_dataset(cfg, tmp_path)
    stage_collect(cfg, tmp_path)
    out = stage_fit(cfg, tmp_path)
    assert out["metrics"]["masking_params_removed"] == 0
    assert out["metrics"]["params_removed"] > 0


def test_pilot_yaml_flags():
    from pathlib import Path

    pilot = (Path(__file__).resolve().parents[1] / "configs" / "experiment" / "pilot.yaml").read_text()
    smoke = (Path(__file__).resolve().parents[1] / "configs" / "experiment" / "smoke.yaml").read_text()
    assert "force_synthetic: false" in pilot
    assert "override /model: gpt2" in pilot
    assert "force_synthetic: true" in smoke
