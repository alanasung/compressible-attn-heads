import numpy as np
from progattn.progattn.patterns import generate_pattern
from progattn.progattn.relax import anneal_gate, soft_blend, compare_relaxations
from progattn.progattn.schedule import greedy_schedule
from progattn.progattn.sweep import run_e01_sweep

def test_greedy():
    sch = greedy_schedule(run_e01_sweep(n_layers=3,n_heads=3,seq_len=10,seed=0), max_replace_frac=0.5)
    assert sch["n_replaced"] >= 1

def test_blend():
    p=generate_pattern("uniform",8); q=generate_pattern("bos_attend",8)
    assert np.allclose(soft_blend(p,q,0), p)
    assert np.allclose(soft_blend(p,q,1), q)

def test_anneal():
    g=anneal_gate(5); assert g[0]==0 and g[-1]==1

def test_compare():
    p=generate_pattern("uniform",8); q=generate_pattern("previous_token",8)
    assert "soft_blend_trajectory" in compare_relaxations(p,q,steps=5)
