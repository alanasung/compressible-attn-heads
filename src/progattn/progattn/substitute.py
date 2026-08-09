"""Forward hooks that swap attention patterns; measured + synthetic collect."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .model_runtime import arch_from_model, try_load_causal_lm
from .patterns import generate_pattern


@dataclass
class HeadRef:
    layer: int
    head: int


def validate_head(layer: int, head: int, n_layers: int, n_heads: int) -> None:
    if not (0 <= layer < n_layers):
        raise ValueError(f"layer {layer} out of range [0, {n_layers - 1}]")
    if not (0 <= head < n_heads):
        raise ValueError(f"head {head} out of range [0, {n_heads - 1}]")


def kl_attention(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-8
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum(axis=-1, keepdims=True)
    q = q / q.sum(axis=-1, keepdims=True)
    return float(np.mean(np.sum(p * (np.log(p) - np.log(q)), axis=-1)))


def synthetic_clean_patterns(
    *,
    n_layers: int,
    n_heads: int,
    seq_len: int,
    seed: int = 0,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    out: dict[str, list[list[list[float]]]] = {}
    for layer in range(n_layers):
        for head in range(n_heads):
            logits = rng.normal(0, 1, size=(seq_len, seq_len))
            logits = np.where(np.tril(np.ones_like(logits)), logits, -1e9)
            e = np.exp(logits - logits.max(axis=-1, keepdims=True))
            attn = e / e.sum(axis=-1, keepdims=True)
            # Plant a few previous-token-like heads for heavy-tail E01.
            if head % 7 == 0:
                attn = 0.85 * generate_pattern("previous_token", seq_len) + 0.15 * attn
                attn = attn / attn.sum(axis=-1, keepdims=True)
            out[f"{layer}:{head}"] = attn.astype(np.float64).tolist()
    return {
        "n_layers": n_layers,
        "n_heads": n_heads,
        "seq_len": seq_len,
        "patterns": out,
        "seed": seed,
        "mode": "synthetic",
        "is_synthetic": True,
    }


def collect_model_attentions(
    *,
    model_name: str,
    revision: str | None = None,
    texts: list[str] | None = None,
    seq_len: int = 32,
    seed: int = 0,
    force_synthetic: bool = False,
    max_layers: int | None = None,
) -> dict[str, Any]:
    """Collect mean attention patterns from a loaded model; fail closed if missing."""
    if force_synthetic:
        out = synthetic_clean_patterns(
            n_layers=12, n_heads=12, seq_len=seq_len, seed=seed
        )
        out["fallback_reason"] = "force_synthetic=True"
        return out

    runtime = try_load_causal_lm(
        model_name, revision=revision, force_synthetic=False
    )
    if runtime is None:
        raise RuntimeError(
            f"Could not load weights for {model_name!r} revision={revision!r}. "
            "Measured attention collect refused synthetic substitution. "
            "Download the model, or set force_synthetic=true for smoke only."
        )

    import torch

    arch = arch_from_model(runtime.model)
    n_layers = arch["n_layers"]
    n_heads = arch["n_heads"]
    if max_layers is not None:
        n_layers = min(n_layers, max_layers)

    if not texts:
        rng = np.random.default_rng(seed)
        # Short natural-ish strings so tokenization is stable without Hub datasets.
        vocab_words = [
            "the", "cat", "sat", "on", "mat", "and", "watched", "birds",
            "fly", "over", "river", "while", "children", "played", "near",
        ]
        texts = [
            " ".join(rng.choice(vocab_words, size=seq_len))
            for _ in range(4)
        ]

    accum: dict[str, list[np.ndarray]] = {}
    with torch.no_grad():
        for text in texts:
            enc = runtime.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=seq_len,
            )
            enc = {k: v.to(runtime.device) for k, v in enc.items()}
            out = runtime.model(**enc, output_attentions=True)
            attns = out.attentions  # type: ignore[attr-defined]
            for layer in range(n_layers):
                a = attns[layer][0].detach().float().cpu().numpy()  # [heads, q, k]
                for head in range(n_heads):
                    key = f"{layer}:{head}"
                    accum.setdefault(key, []).append(a[head])

    patterns: dict[str, list[list[list[float]]]] = {}
    used_len = seq_len
    for key, mats in accum.items():
        # Pad/crop to common seq_len then mean.
        cropped = []
        for m in mats:
            s = min(used_len, m.shape[0], m.shape[1])
            cropped.append(m[:s, :s])
        mean = np.mean(np.stack(cropped, axis=0), axis=0)
        mean = mean / (mean.sum(axis=-1, keepdims=True) + 1e-8)
        patterns[key] = mean.astype(np.float64).tolist()
        used_len = mean.shape[0]

    return {
        "n_layers": n_layers,
        "n_heads": n_heads,
        "seq_len": used_len,
        "patterns": patterns,
        "seed": seed,
        "mode": "model",
        "is_synthetic": False,
        "model_name": model_name,
        "revision": runtime.revision,
        "family": runtime.family,
        "notes": list(runtime.notes),
        "hidden": arch["hidden"],
        "head_dim": arch["head_dim"],
    }


def substitute_pattern(
    clean: dict[str, Any],
    *,
    layer: int,
    head: int,
    program: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    validate_head(layer, head, clean["n_layers"], clean["n_heads"])
    key = f"{layer}:{head}"
    p = np.asarray(clean["patterns"][key], dtype=np.float64)
    q = generate_pattern(program, clean["seq_len"])
    # Align shapes if measured seq_len differs slightly.
    s = min(p.shape[0], q.shape[0])
    return p[:s, :s], q[:s, :s], kl_attention(p[:s, :s], q[:s, :s])


def _pattern_tensor(program: str, seq: int, *, device: Any, dtype: Any) -> Any:
    import torch

    pat = generate_pattern(program, seq)
    return torch.tensor(pat, device=device, dtype=dtype)


def _intervene_gpt2_style(
    module: Any,
    hidden_states: Any,
    out: Any,
    program: str,
    head: int,
    n_heads: int,
    gate: float = 1.0,
) -> Any:
    """Rebuild attn_output via program pattern × values (affects logits).

    ``gate`` in [0, 1] soft-blends learned attentions with the program pattern
    on the target head (1.0 = full program replace).
    """
    import torch

    split_size = int(getattr(module, "split_size", hidden_states.shape[-1]))
    head_dim = int(getattr(module, "head_dim", split_size // max(n_heads, 1)))
    query_states, key_states, value_states = module.c_attn(hidden_states).split(
        split_size, dim=2
    )
    # [batch, heads, seq, head_dim]
    bsz, seq, _ = query_states.shape
    value_states = value_states.view(bsz, seq, n_heads, head_dim).transpose(1, 2)

    # Prefer original attentions when present; otherwise build a soft causal baseline.
    base_attns = None
    if isinstance(out, tuple) and len(out) >= 3 and out[2] is not None and torch.is_tensor(out[2]):
        base_attns = out[2]
    if base_attns is None:
        # Causal uniform fallback when weights were not returned.
        mask = torch.tril(torch.ones(seq, seq, device=hidden_states.device, dtype=hidden_states.dtype))
        base_attns = (mask / mask.sum(dim=-1, keepdim=True)).view(1, 1, seq, seq).expand(
            bsz, n_heads, seq, seq
        )

    new_attns = base_attns.clone()
    if not (0 <= head < new_attns.shape[1]):
        return out
    t = _pattern_tensor(program, seq, device=new_attns.device, dtype=new_attns.dtype)
    g = float(max(0.0, min(1.0, gate)))
    learned = new_attns[:, head, :, :]
    blended = (1.0 - g) * learned + g * t.unsqueeze(0)
    blended = blended / blended.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    new_attns[:, head, :, :] = blended

    # context: [batch, heads, seq, head_dim] -> merge -> project
    context = torch.matmul(new_attns, value_states)
    merged = context.transpose(1, 2).contiguous().view(bsz, seq, n_heads * head_dim)
    attn_output = module.c_proj(merged)
    if hasattr(module, "resid_dropout"):
        attn_output = module.resid_dropout(attn_output)

    present = out[1] if isinstance(out, tuple) and len(out) > 1 else None
    if isinstance(out, tuple) and len(out) >= 3:
        return (attn_output, present, new_attns) + out[3:]
    if isinstance(out, tuple):
        return (attn_output, present) + out[2:]
    return attn_output


def _intervene_value_cache(
    module: Any,
    out: Any,
    program: str,
    head: int,
) -> Any:
    """Rebuild merged head output from ``module._progattn_values`` × new pattern."""
    import torch

    if not isinstance(out, tuple) or len(out) < 3 or out[2] is None:
        return out
    attns = out[2]
    values = getattr(module, "_progattn_values", None)
    if not torch.is_tensor(attns) or attns.dim() != 4 or not torch.is_tensor(values):
        return out
    if not (0 <= head < attns.shape[1]):
        return out
    seq = int(attns.shape[-1])
    new_attns = attns.clone()
    new_attns[:, head, :, :] = _pattern_tensor(
        program, seq, device=attns.device, dtype=attns.dtype
    ).unsqueeze(0)
    # values: [batch, heads, seq, head_dim]
    context = torch.matmul(new_attns, values)
    bsz, n_heads, seq_len, head_dim = context.shape
    merged = context.transpose(1, 2).contiguous().view(bsz, seq_len, n_heads * head_dim)
    if hasattr(module, "c_proj"):
        merged = module.c_proj(merged)
    return (merged, out[1], new_attns) + out[3:]


def make_pattern_hook(
    program: str,
    head: int,
    n_heads: int,
    *,
    gate: float = 1.0,
) -> Callable[..., Any]:
    """Executable pattern intervention for GPT-2-style attention modules.

    Replaces the probabilities used for ``attn @ V`` (or rebuilds ``attn_output``
    from program pattern × values) so next-token logits change. Mutating only the
    debug attentions tensor is insufficient and is not used as the primary path.

    ``gate`` soft-blends learned vs program on the target head (1.0 = full replace).
    This is MASKING / soft intervention, not parameter deletion.
    """

    def hook(module: Any, inp: Any, out: Any) -> Any:
        import torch

        hidden = None
        if isinstance(inp, (tuple, list)) and inp:
            hidden = inp[0]
        elif torch.is_tensor(inp):
            hidden = inp

        # Preferred: GPT-2 fused c_attn / c_proj path — rebuild projected output.
        if (
            hidden is not None
            and torch.is_tensor(hidden)
            and hasattr(module, "c_attn")
            and hasattr(module, "c_proj")
        ):
            return _intervene_gpt2_style(
                module, hidden, out, program, head, n_heads, gate=gate
            )

        # Test / generic path: values cached on the module during forward.
        if getattr(module, "_progattn_values", None) is not None:
            return _intervene_value_cache(module, out, program, head)

        # Last resort: edit returned attentions only (does NOT affect logits).
        if isinstance(out, tuple) and len(out) >= 3 and out[2] is not None:
            attns = out[2]
            if torch.is_tensor(attns) and attns.dim() == 4 and 0 <= head < attns.shape[1]:
                seq = int(attns.shape[-1])
                new_attns = attns.clone()
                t = _pattern_tensor(
                    program, seq, device=attns.device, dtype=attns.dtype
                ).unsqueeze(0)
                g = float(max(0.0, min(1.0, gate)))
                blended = (1.0 - g) * new_attns[:, head, :, :] + g * t
                blended = blended / blended.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                new_attns[:, head, :, :] = blended
                return (out[0], out[1], new_attns) + out[3:]
        return out

    hook.program = program  # type: ignore[attr-defined]
    hook.head = head  # type: ignore[attr-defined]
    hook.n_heads = n_heads  # type: ignore[attr-defined]
    hook.gate = float(gate)  # type: ignore[attr-defined]
    hook.intervention_kind = "masking_not_removal"  # type: ignore[attr-defined]
    hook.executable = True  # type: ignore[attr-defined]
    return hook


def gated_pattern_next_token_kl(
    *,
    runtime: Any,
    layer: int,
    head: int,
    program: str,
    gate: float,
    text: str,
    n_heads: int,
) -> dict[str, Any]:
    """Next-token KL(clean || gated-blend) on an already-loaded runtime."""
    import torch
    import torch.nn.functional as F

    model = runtime.model
    tok = runtime.tokenizer
    enc = tok(text, return_tensors="pt")
    enc = {k: v.to(runtime.device) for k, v in enc.items()}

    with torch.no_grad():
        clean = model(**enc, output_attentions=True)
        clean_logits = clean.logits[0, -1].float()
        clean_p = F.softmax(clean_logits, dim=-1)

    block = None
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        block = model.transformer.h[layer].attn
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        block = getattr(model.model.layers[layer], "self_attn", None)
    if block is None:
        return {"next_token_kl": float("nan"), "gate": float(gate)}

    handle = block.register_forward_hook(
        make_pattern_hook(program, head, n_heads, gate=gate)
    )
    try:
        with torch.no_grad():
            intervened = model(**enc, output_attentions=True)
            int_logits = intervened.logits[0, -1].float()
            int_p = F.softmax(int_logits, dim=-1)
    finally:
        handle.remove()

    eps = 1e-8
    kl = float(torch.sum(clean_p * (torch.log(clean_p + eps) - torch.log(int_p + eps))).item())
    return {"next_token_kl": kl, "gate": float(gate), "kl_space": "next_token"}


def mask_attention_head_probs(
    attn_probs: np.ndarray,
    *,
    head: int,
    program: str,
) -> np.ndarray:
    """Replace one head's attention pattern in-place copy. Masking ≠ removal."""
    out = np.array(attn_probs, copy=True, dtype=np.float64)
    seq = out.shape[-1]
    out[head] = generate_pattern(program, seq)
    return out


def intervention_next_token_kl(
    *,
    model_name: str,
    revision: str | None = None,
    layer: int = 0,
    head: int = 0,
    program: str = "previous_token",
    text: str = "The capital of France is",
    force_synthetic: bool = False,
) -> dict[str, Any]:
    """Measure next-token KL between clean and pattern-intervened forward passes."""
    if force_synthetic:
        # Deterministic smoke: pattern-space KL as stand-in, stamped synthetic.
        seq = 16
        p = generate_pattern("uniform", seq)
        q = generate_pattern(program, seq)
        return {
            "next_token_kl": kl_attention(p, q),
            "mode": "synthetic",
            "is_synthetic": True,
            "kl_space": "pattern",
            "intervention_kind": "masking_not_removal",
            "note": "Synthetic pattern-space KL stand-in; not a live logit KL.",
        }

    runtime = try_load_causal_lm(model_name, revision=revision, force_synthetic=False)
    if runtime is None:
        raise RuntimeError(
            f"Could not load {model_name!r} for intervention KL. "
            "Set force_synthetic=true for smoke only."
        )

    import torch
    import torch.nn.functional as F

    arch = arch_from_model(runtime.model)
    validate_head(layer, head, arch["n_layers"], arch["n_heads"])
    tok = runtime.tokenizer
    model = runtime.model
    enc = tok(text, return_tensors="pt")
    enc = {k: v.to(runtime.device) for k, v in enc.items()}

    with torch.no_grad():
        clean = model(**enc, output_attentions=True)
        clean_logits = clean.logits[0, -1].float()
        clean_p = F.softmax(clean_logits, dim=-1)

    # Prefer attn module on the block when present (GPT-2).
    block = None
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        block = model.transformer.h[layer].attn
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        block = getattr(model.model.layers[layer], "self_attn", None)
    if block is None:
        return {
            "next_token_kl": float("nan"),
            "mode": "unavailable",
            "is_synthetic": False,
            "kl_space": "next_token",
            "intervention_kind": "masking_not_removal",
            "note": "No attention module found for hook installation.",
        }

    handle = block.register_forward_hook(make_pattern_hook(program, head, arch["n_heads"]))
    try:
        with torch.no_grad():
            intervened = model(**enc, output_attentions=True)
            int_logits = intervened.logits[0, -1].float()
            int_p = F.softmax(int_logits, dim=-1)
    finally:
        handle.remove()

    # KL(clean || intervened)
    eps = 1e-8
    kl = float(torch.sum(clean_p * (torch.log(clean_p + eps) - torch.log(int_p + eps))).item())
    return {
        "next_token_kl": kl,
        "mode": "model",
        "is_synthetic": False,
        "kl_space": "next_token",
        "intervention_kind": "masking_not_removal",
        "layer": layer,
        "head": head,
        "program": program,
        "model_name": model_name,
        "revision": runtime.revision,
        "note": "Live next-token KL under attention-pattern masking (not param deletion).",
    }
