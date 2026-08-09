def test_imports():
    from progattn.progattn import patterns
    assert patterns.PROGRAMS

def test_stages():
    from progattn.stages import STAGES
    assert callable(STAGES["fit"])
