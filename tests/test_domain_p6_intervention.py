"""P6: executable pattern intervention, next-token ranking stamps, surgery honesty."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

from progattn.progattn.schedule import greedy_schedule
from progattn.progattn.substitute import intervention_next_token_kl, make_pattern_hook
from progattn.progattn.sweep import run_e01_sweep
from progattn.progattn.surgery import live_gpt2_surgery


class _FakeTok:
    def __call__(self, text, return_tensors="pt", **kwargs):
        ids = torch.arange(1, 9, dtype=torch.long).unsqueeze(0)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class _FakeAttn(nn.Module):
    """GPT-2-shaped attention: c_attn / c_proj so pattern hooks rebuild logits."""

    def __init__(self, hidden: int = 32, n_heads: int = 4):
        super().__init__()
        self.hidden = hidden
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.split_size = hidden
        self.c_attn = nn.Linear(hidden, 3 * hidden, bias=True)
        self.c_proj = nn.Linear(hidden, hidden, bias=True)
        self.resid_dropout = nn.Identity()
        # Non-identity init so pattern replacement moves outputs.
        with torch.no_grad():
            self.c_attn.weight.normal_(0, 1.2)
            self.c_proj.weight.normal_(0, 1.2)
            self.c_attn.bias.normal_(0, 0.3)

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
        present = None
        if output_attentions:
            return (out, present, attns)
        return (out, present)


class _FakeBlock(nn.Module):
    def __init__(self, hidden=32, n_heads=4):
        super().__init__()
        self.attn = _FakeAttn(hidden, n_heads)
        self.ln = nn.Identity()

    def forward(self, hidden_states, **kwargs):
        attn_out = self.attn(hidden_states, **kwargs)
        h = attn_out[0] if isinstance(attn_out, tuple) else attn_out
        return (self.ln(h + hidden_states),)


class _FakeModel(nn.Module):
    def __init__(self, n_layers=2, hidden=32, n_heads=4, vocab=50):
        super().__init__()
        self.config = SimpleNamespace(
            n_layer=n_layers, n_head=n_heads, n_embd=hidden, vocab_size=vocab
        )
        self.wte = nn.Embedding(vocab, hidden)
        self.transformer = SimpleNamespace(h=nn.ModuleList([_FakeBlock(hidden, n_heads) for _ in range(n_layers)]))
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
        h = self.wte(input_ids)
        all_attns = []
        for block in self.transformer.h:
            out = block.attn(h, output_attentions=output_attentions)
            if output_attentions:
                h = h + out[0]
                all_attns.append(out[2])
            else:
                h = h + (out[0] if isinstance(out, tuple) else out)
        logits = self.lm_head(h)
        return SimpleNamespace(logits=logits, attentions=tuple(all_attns) if output_attentions else None)


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


def test_make_pattern_hook_changes_attn_output():
    attn = _FakeAttn()
    x = torch.randn(1, 6, 32)
    with torch.no_grad():
        clean = attn(x, output_attentions=True)
    handle = attn.register_forward_hook(make_pattern_hook("bos_attend", head=0, n_heads=4))
    try:
        with torch.no_grad():
            intervened = attn(x, output_attentions=True)
    finally:
        handle.remove()
    assert isinstance(clean, tuple) and isinstance(intervened, tuple)
    delta = (clean[0] - intervened[0]).abs().max().item()
    assert delta > 1e-4


def test_intervention_next_token_kl_nonzero_under_fake_model():
    torch.manual_seed(0)
    np.random.seed(0)
    rt = _runtime(_FakeModel())
    with patch("progattn.progattn.substitute.try_load_causal_lm", return_value=rt):
        out = intervention_next_token_kl(
            model_name="fake-gpt2",
            layer=0,
            head=0,
            program="bos_attend",
            force_synthetic=False,
        )
    assert out["is_synthetic"] is False
    assert out["kl_space"] == "next_token"
    # Executable pattern replace must move next-token mass (CI-stable floor).
    assert out["next_token_kl"] > 1e-8


def test_intervention_fail_closed_without_weights():
    with patch("progattn.progattn.substitute.try_load_causal_lm", return_value=None):
        try:
            intervention_next_token_kl(model_name="openai-community/gpt2", force_synthetic=False)
            ok = False
        except RuntimeError:
            ok = True
        assert ok


def test_smoke_stamps_pattern_kl_space():
    out = intervention_next_token_kl(model_name="gpt2", force_synthetic=True)
    assert out["is_synthetic"] is True
    assert out["kl_space"] == "pattern"


def test_e01_rerank_stamps_next_token_kl_space():
    clean = {
        "n_layers": 1,
        "n_heads": 2,
        "seq_len": 8,
        "patterns": {
            "0:0": np.eye(8).tolist(),
            "0:1": (np.ones((8, 8)) / 8).tolist(),
        },
        "mode": "model",
        "is_synthetic": False,
    }

    def fake_live(**kwargs):
        # Distinct KL per head so ranking moves.
        return {
            "next_token_kl": 0.5 + 0.1 * int(kwargs["head"]),
            "kl_space": "next_token",
            "is_synthetic": False,
        }

    with patch(
        "progattn.progattn.sweep.intervention_next_token_kl", side_effect=fake_live
    ):
        e01 = run_e01_sweep(
            clean=clean,
            model_name="fake-gpt2",
            force_synthetic=False,
            live_rerank_budget=4,
        )
    assert e01["kl_space"] == "next_token"
    assert e01["metrics"]["kl_space"] == "next_token"
    assert e01["metrics"]["is_synthetic"] is False
    assert e01["metrics"]["behavioral_substitutability_claimed"] is True
    assert e01["rows"][0]["kl_space"] == "next_token"
    assert "next_token_kl" in e01["rows"][0]
    sched = greedy_schedule(e01, max_replace_frac=1.0)
    assert sched["kl_space"] == "next_token"
    assert sched["chosen"][0]["kl_space"] == "next_token"


def test_e01_smoke_stays_pattern_no_behavioral_claim():
    e01 = run_e01_sweep(n_layers=2, n_heads=2, seq_len=8, seed=0, force_synthetic=True)
    assert e01["kl_space"] == "pattern"
    assert e01["metrics"]["kl_space"] == "pattern"
    assert e01["metrics"]["is_synthetic"] is True
    assert e01["metrics"]["behavioral_substitutability_claimed"] is False
    assert all(r["kl_space"] == "pattern" for r in e01["rows"])


class _Conv1DLike(nn.Module):
    """HF GPT-2 Conv1D layout: weight [in, out] = [hidden, 3*hidden]."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden, 3 * hidden) * 0.02)
        self.bias = nn.Parameter(torch.zeros(3 * hidden))

    def forward(self, x):
        return x @ self.weight + self.bias


def test_surgery_eval_live_zero_qk_stamp():
    hidden, n_heads, head_dim = 32, 4, 8
    model = _FakeModel(n_layers=1, hidden=hidden, n_heads=n_heads)
    attn = model.transformer.h[0].attn
    attn.c_attn = _Conv1DLike(hidden)
    attn.c_proj = nn.Linear(hidden, hidden, bias=True)
    attn.head_dim = head_dim
    attn.split_size = hidden
    rt = _runtime(model)
    rt.family = "gpt2"

    def fake_arch(_m):
        return {"n_layers": 1, "n_heads": n_heads, "hidden": hidden, "head_dim": head_dim}

    with patch("progattn.progattn.model_runtime.try_load_causal_lm", return_value=rt), patch(
        "progattn.progattn.model_runtime.arch_from_model", side_effect=fake_arch
    ):
        out = live_gpt2_surgery(
            model_name="fake-gpt2",
            converted=[0],
            force_synthetic=False,
        )
    assert out["surgery_eval"] == "live_zero_qk"
    assert out["ppl_before"]["mode"] == "in_memory_pre_zero_qk"
    assert out["ppl_after"]["mode"] == "in_memory_post_zero_qk"
    assert "not reload" in out["note"].lower()
