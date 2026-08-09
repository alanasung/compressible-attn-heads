# VALIDATION — programmatic-attention

## Codex (p4)
- Verdict: SERIOUS_PROBLEMS
- Summary: Codex wants executable pattern-swap hooks, next-token KL under intervention, installed surgical modules for post-removal PPL, and full differentiable FT — beyond the local measurable pilot once measured collect, fused-QKV accounting, and honest masking≠removal are in place.
- Detail: `orchestration/out/validate/programmatic-attention.json`

## Grok (p4 dual)
- Verdict: PASS_WITH_NOTES
- Summary: Measured GPT-2 attention collect and live fused-QKV surgery are pilot-default and fail-closed; masking reports `params_removed=0` while structural slice counts real Q/K deletion; efficiency refuses false param-reduction claims; evaluate measures next-token NLL on a loaded model; pilot subject is pinned GPT-2; domain P4 tests pass under Hub monkeypatch.
- Detail: `orchestration/out/grok/validate/programmatic-attention.p4.md`

## KEY_FIXES (p4)
| Fix | Status |
|---|---|
| Live attention collect on loaded weights | OK (`substitute.collect_model_attentions`) |
| `force_synthetic` smoke-only; pilot measured default | OK (pilot `false` + gpt2, smoke `true`) |
| Fail-closed measured collect / surgery / PPL | OK (`RuntimeError` when weights missing) |
| Family-specific fused-QKV; masking≠removal both | OK (`surgically_narrow_c_attn` + `masking_report`) |
| Efficiency no param reduction without removal | OK (`claims_parameter_reduction` gated) |
| Substitutability ladder + real perplexity | PARTIAL (E01+schedule; real NLL; LAMBADA/BLiMP/IOI proxy) |
| Pinned revisions (gpt2 + qwen/pythia/llama) | OK (yaml + registry SHAs) |
| Hub-free domain tests | OK (`test_domain_p4_measured.py`, 61 domain tests) |

## Remaining (compute / scale — not empty stages)
- Surgical module is sliced and counted with toy equivalence; not yet installed as a custom forward for end-to-end post-removal PPL.
- Soft-blend / STE remain pattern-space demos, not full live-weight anneal→retrain.
- LAMBADA / BLiMP / IOI stay schedule-cost proxies; PPL uses short held-out texts (not full WikiText stream).
- `make_pattern_hook` is a documented masking API stub (masking≠removal correctly labeled).
- Codex SERIOUS_PROBLEMS accepted as frontier-purity / full-training residual notes under the local-pilot lens.

## Reconciliation
Grok PASS_WITH_NOTES on the measurable core. Codex SERIOUS_PROBLEMS remains on executable intervention training and post-install surgical evaluation — recorded as residual scale notes, not missing stages. Domain tests pass (61).

## P5 rigor pass (measured prior work-critical paths)

- Live / measured paths preferred; synthetic remains smoke-only with honesty stamps.
- Claim gating tightened where proxies previously looked like evidence.
- Domain tests green without Hub downloads.

## P6 executable intervention + ranking honesty

| Fix | Status |
|---|---|
| Pattern hook rebuilds `attn_output` (program × values / GPT-2 `c_attn` path) | OK (`make_pattern_hook` executable) |
| Live `intervention_next_token_kl` moves logits under FakeModel | OK (Hub-free unit test) |
| E01 schedules on pattern KL then re-ranks with next-token KL when measured | OK (`kl_space`, `rank_kl`) |
| Smoke / `force_synthetic` stays `kl_space=pattern`; no behavioral claim | OK |
| Surgery eval prefers in-memory zero-QK NLL (`surgery_eval=live_zero_qk`) | OK |
| Fail-closed measured path; synthetic smoke-only | OK |

Residual: full narrower Conv1D module install for end-to-end post-removal PPL; LAMBADA/BLiMP remain schedule proxies.

## P7 rigor pass (local fixtures + live anneal + wall-clock honesty)

| Fix | Status |
|---|---|
| Bundled Hub-free downstream mini fixtures | OK (`data/fixtures/downstream_mini.json`) |
| `task_metrics_mode=local_fixture`; `claims_downstream` only then | OK (`tasks.evaluate_local_fixture_suite`) |
| PPL path unchanged (real NLL / synthetic smoke) | OK (`measure_perplexity`) |
| Live soft-anneal one head; `anneal_mode=live_weight` vs `pattern_demo` | OK (`relax.live_soft_anneal`, gated hook) |
| Wall-clock: never claim speedup without measured timings | OK (`efficiency.compare_wall_clock`, FakeModel CI) |
| Hub-free P7 domain tests | OK (`tests/test_domain_p7_fixtures.py`) |

Residual (scale): full Conv1D surgical module install for true post-removal wall-clock/PPL; licensed WikiText/LAMBADA streams.

