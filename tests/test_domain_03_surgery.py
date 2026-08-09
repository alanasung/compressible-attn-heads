import numpy as np
from progattn.progattn.surgery import SurgeryPlan, demo_surgery, surgically_narrow_c_attn, numerical_equivalence_check

def test_removal():
    out = demo_surgery(n_heads=8, head_dim=16, converted=[0,2])
    assert out["params_removed"] > 0
    assert out["equivalence"]["equivalent"]

def test_narrow():
    h=8; d=16; hidden=h*d
    w=np.zeros((hidden,3*hidden)); b=np.zeros(3*hidden)
    out = surgically_narrow_c_attn(w,b, SurgeryPlan(0,[1],h,d,hidden))
    assert out["weight"].shape[1] < w.shape[1]

def test_equiv_false():
    assert numerical_equivalence_check(np.zeros((2,2)), np.ones((2,2)))["equivalent"] is False
