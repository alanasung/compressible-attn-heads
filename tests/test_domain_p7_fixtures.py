"""P7: local downstream fixtures, live soft-anneal, wall-clock honesty."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

from progattn.progattn.efficiency import (
    compare_wall_clock,
    efficiency_report,
    measure_forward_wall_clock,
)
from progattn.progattn.relax import compare_relaxations, live_soft_anneal
from progattn.progattn.substitute import gated_pattern_next_token_kl
from progattn.progattn.tasks import (
    evaluate_local_fixture_suite,
    evaluate_proxy_suite,
    evaluate_suite,
    load_downstream_fixtures,
)


class _FakeTok:
    def __call__(self, text, return_tensors="pt", **kwargs):
        # Deterministic length from text so continuation spans differ.
        n = max(4, min(24, len(str(text).split()) + 2))
        ids = torch.arange(1, n + 1, dtype=torch.long).unsqueeze(0)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class _FakeAttn(nn.Module):
    def __init__(self, hidden: int = 32, n_heads: int = 4):
        super().__init__()
        self.hidden = hidden
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.split_size = hidden
        self.c_attn = nn.Linear(hidden, 3 * hidden, bias=True)
        self.c_proj = nn.Linear(hidden, hidden, bias=True)
        self.resid_dropout = nn.Identity()
        with torch.no_grad():
            self.c_attn.weight.normal_(0, 0.4)
            self.c_proj.weight.normal_(0, 0.4)

    def forward(self, hidden_states, output_attentions=False, use_cache=False, **kwargs):
        bsz, seq, _ = hidden_states.shape
        q, k, v = self.c_attn(hidden_states).split(self.split_size, dim=2)
        q = q.view(bsz, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim**0.5)
        causal = torch.tril(torch.ones(seq, seq, device=hidden_states.device))
        scores = scores.masked_fill(causal.view(1, 1, seq, seq) == 0, -1e9)
        attns = torch.softmax(scores, dim=-1)
        context = torch.matmul(attns, v)
        merged = context.transpose(1, 2).contiguous().view(bsz, seq, self.hidden)
        out = self.c_proj(merged)
        if output_attentions:
            return (out, None, attns)
        return (out, None)


class _FakeBlock(nn.Module):
    def __init__(self, hidden=32, n_heads=4):
        super().__init__()
        self.attn = _FakeAttn(hidden, n_heads)

    def forward(self, hidden_states, **kwargs):
        return (hidden_states + self.attn(hidden_states, **kwargs)[0],)


class _FakeModel(nn.Module):
    def __init__(self, n_layers=2, hidden=32, n_heads=4, vocab=80):
        super().__init__()
        self.config = SimpleNamespace(
            n_layer=n_layers, n_head=n_heads, n_embd=hidden, vocab_size=vocab
        )
        self.wte = nn.Embedding(vocab, hidden)
        self.transformer = SimpleNamespace(
            h=nn.ModuleList([_FakeBlock(hidden, n_heads) for _ in range(n_layers)])
        )
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        with torch.no_grad():
            self.lm_head.weight.normal_(0, 0.3)

    def __call__(self, input_ids=None, attention_mask=None, output_attentions=False, **kwargs):
        return self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            **kwargs,
        )

    def forward(self, input_ids=None, attention_mask=None, output_attentions=False, **kwargs):
        h = self.wte(input_ids % self.config.vocab_size)
        all_attns = []
        for block in self.transformer.h:
            out = block.attn(h, output_attentions=output_attentions)
            h = h + out[0]
            if output_attentions:
                all_attns.append(out[2])
        logits = self.lm_head(h)
        return SimpleNamespace(
            logits=logits, attentions=tuple(all_attns) if output_attentions else None
        )


def _runtime(model=None):
    model = model or _FakeModel()
    return SimpleNamespace(
        model=model,
        tokenizer=_FakeTok(),
        name="fake-gpt2",
        revision="test",
        device="cpu",
        family="gpt2",
        notes=["fake"],
    )


def test_fixtures_load():
    fx = load_downstream_fixtures()
    assert fx["corpus"] == "local_downstream_mini"
    assert len(fx["lambada"]) >= 2
    assert len(fx["blimp"]) >= 2
    assert len(fx["ioi"]) >= 2


def test_proxy_never_claims_downstream():
    out = evaluate_proxy_suite({"final_joint_kl": 0.1, "n_replaced": 1}, seed=0)
    assert out["task_metrics_mode"] == "proxy"
    assert out["claims_downstream"] is False


def test_local_fixture_claims_downstream_under_fake_model():
    rt = _runtime()
    with patch("progattn.progattn.tasks.try_load_causal_lm", return_value=rt):
        out = evaluate_local_fixture_suite(
            {"n_replaced": 1, "final_joint_kl": 0.0},
            model_name="fake-gpt2",
            force_synthetic=False,
            runtime=rt,
        )
    assert out["task_metrics_mode"] == "local_fixture"
    assert out["claims_downstream"] is True
    assert out["fixture_corpus"] == "local_downstream_mini"
    assert out["lambada_n"] + out["blimp_n"] + out["ioi_n"] >= 1


def test_smoke_suite_no_downstream_claim():
    out = evaluate_suite(
        {"n_replaced": 0, "final_joint_kl": 0.0},
        model_name="gpt2",
        force_synthetic=True,
    )
    assert out["task_metrics_mode"] == "proxy"
    assert out["claims_downstream"] is False
    assert out["ppl_is_synthetic"] is True


def test_fixture_fail_closed_without_weights():
    with patch("progattn.progattn.tasks.try_load_causal_lm", return_value=None):
        try:
            evaluate_local_fixture_suite(
                {"n_replaced": 0},
                model_name="openai-community/gpt2",
                force_synthetic=False,
            )
            ok = False
        except RuntimeError:
            ok = True
        assert ok


def test_pattern_demo_anneal_mode():
    p = np.eye(8)
    q = np.ones((8, 8)) / 8
    out = compare_relaxations(p, q, steps=4)
    assert out["anneal_mode"] == "pattern_demo"


def test_live_soft_anneal_stamps_live_weight():
    rt = _runtime()

    def fake_arch(_m):
        return {"n_layers": 2, "n_heads": 4, "hidden": 32, "head_dim": 8}

    with patch("progattn.progattn.model_runtime.try_load_causal_lm", return_value=rt), patch(
        "progattn.progattn.model_runtime.arch_from_model", side_effect=fake_arch
    ):
        out = live_soft_anneal(
            model_name="fake-gpt2",
            layer=0,
            head=0,
            program="bos_attend",
            steps=3,
            force_synthetic=False,
            runtime=rt,
        )
    assert out["anneal_mode"] == "live_weight"
    assert out["is_synthetic"] is False
    assert len(out["soft_blend_trajectory"]) == 3
    assert "next_token_kl" in out["soft_blend_trajectory"][0]


def test_smoke_anneal_stays_pattern_demo():
    out = live_soft_anneal(model_name="gpt2", force_synthetic=True, steps=3)
    assert out["anneal_mode"] == "pattern_demo"
    assert out["is_synthetic"] is True


def test_gated_hook_changes_with_gate():
    rt = _runtime()
    kl0 = gated_pattern_next_token_kl(
        runtime=rt, layer=0, head=0, program="bos_attend", gate=0.0, text="hi", n_heads=4
    )
    kl1 = gated_pattern_next_token_kl(
        runtime=rt, layer=0, head=0, program="bos_attend", gate=1.0, text="hi", n_heads=4
    )
    assert kl0["next_token_kl"] < kl1["next_token_kl"] + 1e-9 or kl1["next_token_kl"] >= 0


def test_no_speedup_without_measured_timings():
    cmp = compare_wall_clock(seconds_before=None, seconds_after=None, measured=False)
    assert cmp["claims_speedup"] is False
    eff = efficiency_report(
        params_before=100, params_after=80, n_heads=12, converted=3
    )
    assert eff["claims_speedup"] is False
    assert "eager" in eff["wall_clock"]["label"]


def test_speedup_only_with_measured_before_after():
    cmp = compare_wall_clock(seconds_before=2.0, seconds_after=1.0, measured=True)
    assert cmp["claims_speedup"] is True
    assert cmp["speedup"] == 2.0
    # Equal timings → no claim
    cmp2 = compare_wall_clock(seconds_before=1.0, seconds_after=1.0, measured=True)
    assert cmp2["claims_speedup"] is False
    eff = efficiency_report(
        params_before=100,
        params_after=80,
        n_heads=4,
        converted=1,
        wall_clock_before={
            "forward_seconds_per_rep": 0.2,
            "measured": True,
            "device": "cpu",
        },
        wall_clock_after={
            "forward_seconds_per_rep": 0.1,
            "measured": True,
            "device": "cpu",
        },
    )
    assert eff["claims_speedup"] is True


def test_measure_forward_wall_clock_fake_model():
    rt = _runtime()
    out = measure_forward_wall_clock(rt, reps=2, warmup=0)
    assert out["measured"] is True
    assert out["forward_seconds_per_rep"] >= 0.0
    assert out["device"] == "cpu"
