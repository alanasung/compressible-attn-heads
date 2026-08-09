from progattn.progattn.sweep import E01_PROGRAMS, run_e01_sweep
from progattn.progattn.substitute import validate_head, synthetic_clean_patterns, substitute_pattern, kl_attention

def test_e01():
    e = run_e01_sweep(n_layers=3, n_heads=3, seq_len=10, seed=0)
    assert len(e["rows"]) == 9
    assert set(E01_PROGRAMS) == {"previous_token", "positional_offset", "bos_attend"}

def test_invalid_layer():
    import pytest
    with pytest.raises(ValueError):
        validate_head(12, 0, 12, 12)

def test_valid_layer():
    validate_head(11, 0, 12, 12)

def test_kl():
    clean = synthetic_clean_patterns(n_layers=2, n_heads=2, seq_len=8, seed=0)
    p,q,kl = substitute_pattern(clean, layer=0, head=0, program="uniform")
    assert abs(kl - kl_attention(p,q)) < 1e-9

def test_seed_stable():
    a = run_e01_sweep(n_layers=2, n_heads=2, seq_len=8, seed=0)
    b = run_e01_sweep(n_layers=2, n_heads=2, seq_len=8, seed=0)
    assert a["metrics"]["median_kl"] == b["metrics"]["median_kl"]
