# VALIDATION — programmatic-attention

## Codex v1 (historical)
- Verdict: SERIOUS_PROBLEMS
- Summary: The idea has a testable removal endpoint, but this repository is a registry of placeholders with an underspecified methodology, invalid pilot configuration, and non-reproducible setup—not a runnable research pilot.

## Codex v2
- Verdict: PASS_WITH_NOTES
- Summary: Analogous to introspection-verbalization Codex v2: X1–X13 OK; stages implemented with a real `make pilot` path; synthetic/proxy pilot default; several model revisions still on `main`.
- KEY_FIXES_OK: X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13

## Grok (dual-validate)
- Verdict: PASS_WITH_NOTES
- Summary: Domain modules cover patterns, substitutability sweep, relaxations, QK surgery, and efficiency accounting; stages delegate to a real pipeline with no NotImplementedError. Same residual notes as Codex v2 (synthetic pilot default; revision=main).

### Remaining
- Pilot path uses synthetic LM batches / clean patterns rather than exercising live attention surgery on loaded weights.
- Model revisions mostly `main` (only gpt2 has a commit SHA).
- Real fused-QKV deletion still needs checkpoint-family-specific verification beyond the synthetic surgery tests.

## Reconciliation
v1 SERIOUS_PROBLEMS resolved by real domain pipeline + spine fixes. Grok agrees with Codex-v2-analogous PASS_WITH_NOTES.
