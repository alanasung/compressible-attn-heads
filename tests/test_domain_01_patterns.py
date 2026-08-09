import pytest, numpy as np
from progattn.progattn.patterns import PROGRAMS, generate_pattern

@pytest.mark.parametrize("name", list(PROGRAMS))
def test_norm(name):
    p = generate_pattern(name, 12)
    assert np.allclose(p.sum(1), 1.0, atol=1e-6)
    assert np.isfinite(p).all()

def test_causal():
    p = generate_pattern("uniform", 8)
    assert np.allclose(np.triu(p, 1), 0)

def test_unknown():
    with pytest.raises(KeyError):
        generate_pattern("nope", 4)

def test_prev():
    assert generate_pattern("previous_token", 5)[3,2] == 1.0
