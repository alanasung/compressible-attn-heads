from progattn.progattn.tasks import evaluate_proxy_suite, perplexity_from_nll, synthetic_lm_batch
from progattn.progattn.efficiency import count_attention_flops, efficiency_report

def test_ppl():
    assert perplexity_from_nll([0,0]) == 1.0

def test_batch():
    assert synthetic_lm_batch(n=4, seq_len=8, seed=0)["n"] == 4

def test_proxy():
    assert "wikitext_ppl" in evaluate_proxy_suite({"final_joint_kl":0.2,"n_replaced":2}, seed=0)

def test_flops():
    a=count_attention_flops(32,12,64,programmatic_heads=0)
    b=count_attention_flops(32,12,64,programmatic_heads=6)
    assert b["qk_flops"] < a["qk_flops"]

def test_eff_label():
    assert "eager" in efficiency_report(
        params_before=100, params_after=80, n_heads=12, converted=3
    )["wall_clock"]["label"]
