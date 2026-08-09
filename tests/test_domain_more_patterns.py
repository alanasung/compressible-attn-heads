import pytest
import numpy as np
from progattn.progattn.patterns import PROGRAMS, generate_pattern

@pytest.mark.parametrize("name", PROGRAMS)
def test_program_finite(name):
    p = generate_pattern(name, 10)
    assert np.isfinite(p).all()

@pytest.mark.parametrize("n", [1, 2, 7, 33])
def test_sizes(n):
    p = generate_pattern("positional_decay", n)
    assert p.shape == (n, n)

def test_bos_attends_col0():
    p = generate_pattern("bos_attend", 6)
    # each row puts mass on position 0 among causal support
    assert p[:, 0].min() > 0
