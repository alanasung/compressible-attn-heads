from types import SimpleNamespace
from progattn.stages import STAGES
import json

def cfg():
    return SimpleNamespace(run=SimpleNamespace(seed=0,profile="smoke"), data=SimpleNamespace(n_items=16), model=SimpleNamespace(name="gpt2"), eval=SimpleNamespace(layers=[2,4,6]))

def test_registry():
    assert set(STAGES)=={"build_dataset","collect","fit","evaluate","report"}

def test_e2e(tmp_path):
    run=tmp_path/"r"; c=cfg()
    for n in ["build_dataset","collect","fit","evaluate","report"]:
        out=STAGES[n](c,run)
        assert out["task"]==n and "metrics" in out

def test_e01_artifact(tmp_path):
    run=tmp_path/"r"; c=cfg()
    for n in ["build_dataset","collect","fit"]:
        STAGES[n](c,run)
    assert (run/"artifacts/fit/e01_rows.json").is_file()
