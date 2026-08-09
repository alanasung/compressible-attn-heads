def test_imports():
    from progattn.progattn import patterns, substitute, sweep, schedule, relax, surgery, tasks, efficiency, pipeline
    assert patterns.PROGRAMS

def test_stages():
    from progattn.stages import STAGES
    assert callable(STAGES["fit"])
