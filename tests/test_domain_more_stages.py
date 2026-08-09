from types import SimpleNamespace
from progattn.stages import STAGES
import json

def cfg():
    return SimpleNamespace(run=SimpleNamespace(seed=2, profile="pilot"), data=SimpleNamespace(n_items=16), model=SimpleNamespace(name="gpt2"), eval=SimpleNamespace(layers=[2,4,6]))

def test_fit_writes_e01(tmp_path):
    run = tmp_path / "r"
    c = cfg()
    STAGES["build_dataset"](c, run)
    STAGES["collect"](c, run)
    STAGES["fit"](c, run)
    assert (run / "artifacts/fit/e01_rows.json").is_file()
    res = json.loads((run / "artifacts/fit/results.json").read_text())
    assert "n_cheap_heads" in res["metrics"]

def test_evaluate_efficiency(tmp_path):
    run = tmp_path / "r"
    c = cfg()
    for s in ("build_dataset", "collect", "fit", "evaluate"):
        STAGES[s](c, run)
    eff = json.loads((run / "artifacts/evaluate/efficiency.json").read_text())
    assert eff["params_removed"] >= 0

def test_arch_valid_layers(tmp_path):
    run = tmp_path / "r"
    STAGES["build_dataset"](cfg(), run)
    arch = json.loads((run / "artifacts/dataset/arch.json").read_text())
    assert arch["n_layers"] == 12
