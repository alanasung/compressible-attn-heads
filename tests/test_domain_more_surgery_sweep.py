import numpy as np
from progattn.progattn.surgery import numerical_equivalence_check, demo_surgery
from progattn.progattn.sweep import full_program_sweep, run_e01_sweep
from progattn.progattn.substitute import kl_attention, substitute_pattern, synthetic_clean_patterns

def test_equivalence_fail():
    a = np.zeros((2, 2))
    b = np.ones((2, 2))
    assert numerical_equivalence_check(a, b, atol=1e-6)["equivalent"] is False

def test_full_sweep_rows():
    clean = synthetic_clean_patterns(n_layers=2, n_heads=2, seq_len=8, seed=0)
    out = full_program_sweep(clean)
    assert out["n"] == 2 * 2 * 7

def test_substitute_returns_kl():
    clean = synthetic_clean_patterns(n_layers=2, n_heads=2, seq_len=8, seed=0)
    p, q, kl = substitute_pattern(clean, layer=0, head=0, program="uniform")
    assert kl == kl_attention(p, q)

def test_demo_surgery_surviving():
    out = demo_surgery(converted=[0])
    assert 0 in out["converted_heads"]
    assert 0 not in out["surviving_heads"]

def test_e01_seed_deterministic():
    a = run_e01_sweep(n_layers=2, n_heads=2, seq_len=8, seed=0)
    b = run_e01_sweep(n_layers=2, n_heads=2, seq_len=8, seed=0)
    assert a["metrics"]["median_kl"] == b["metrics"]["median_kl"]
