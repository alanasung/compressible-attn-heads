"""Local causal-LM loading with explicit synthetic fallback (no silent Hub in tests)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RuntimeModel:
    model: Any
    tokenizer: Any
    name: str
    revision: str | None
    device: str
    family: str
    notes: list[str]


def infer_family(model_name: str) -> str:
    low = model_name.lower()
    if "gpt2" in low:
        return "gpt2"
    if "pythia" in low or "gpt-neox" in low:
        return "gpt_neox"
    if "qwen" in low or "llama" in low or "olmo" in low or "gemma" in low:
        return "llama_like"
    return "unknown"


def try_load_causal_lm(
    model_name: str,
    *,
    revision: str | None = None,
    device: str | None = None,
    force_synthetic: bool = False,
) -> RuntimeModel | None:
    if force_synthetic or not model_name or model_name in {"x", "none", "synthetic", "missing"}:
        return None
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception:
        return None

    if revision is None:
        try:
            from progattn.models.registry import get_model_spec

            revision = get_model_spec(model_name).revision
        except Exception:
            revision = None

    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    try:
        tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision)
        model.to(device)
        model.eval()
        family = infer_family(model_name)
        return RuntimeModel(
            model=model,
            tokenizer=tok,
            name=model_name,
            revision=revision,
            device=device,
            family=family,
            notes=[f"loaded {model_name} revision={revision or 'default'} on {device} family={family}"],
        )
    except Exception:
        return None


def arch_from_model(model: Any) -> dict[str, int]:
    cfg = model.config
    n_layers = int(getattr(cfg, "n_layer", getattr(cfg, "num_hidden_layers", 0)))
    n_heads = int(getattr(cfg, "n_head", getattr(cfg, "num_attention_heads", 0)))
    hidden = int(getattr(cfg, "n_embd", getattr(cfg, "hidden_size", 0)))
    head_dim = hidden // max(n_heads, 1)
    return {
        "n_layers": n_layers,
        "n_heads": n_heads,
        "hidden": hidden,
        "head_dim": head_dim,
    }
