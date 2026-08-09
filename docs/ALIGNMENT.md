# ALIGNMENT — programmatic-attention

## Codex GPT-5 Sol (p4)
- Verdict: MINOR_DRIFT
- Summary: Tightly aligned with deploying programmatic QK circuits and native interpretability; fixable gaps around training-efficiency measurement, relaxation failure-mode documentation completeness, and explicit update criteria (those live in `projects.py`).
- Detail: `orchestration/out/align/programmatic-attention.json`

## Grok (p4 dual)
- Verdict: MINOR_DRIFT
- Summary: Deployment-focused substitutability→surgery ladder matches ; residual drift is incomplete live FT/recovery and proxy downstream tasks, not a different program.
- Detail: `orchestration/out/grok/align/programmatic-attention.p4.md`

## Reconciliation
Proceed. Kernel-level inference remains out of scope (honest). P4 strengthens measured GPT-2 collect, fused-QKV structural surgery with masking≠removal contrast, and real next-token perplexity.
