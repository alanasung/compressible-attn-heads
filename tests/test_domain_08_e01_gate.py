from progattn.progattn.sweep import run_e01_sweep

def test_metrics_keys():
    m = run_e01_sweep(n_layers=2,n_heads=2,seq_len=8,seed=0)["metrics"]
    for k in ("n_cheap_heads","frac_cheap","median_kl","deployment_premise_ok","heavy_tailed"):
        assert k in m
