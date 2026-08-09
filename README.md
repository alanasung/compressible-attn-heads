<p align="center">
  <h1 align="center">Deploying Programmatic Attention in Transformers</h1>
  <p align="center"><strong>Rank heads by substitutability with hand-written programs, then surgically install those programs for real efficiency gains.</strong></p>
  </p>

---

## Overview

This repository implements experimental profiles for **Deploying Programmatic Attention in Transformers**. Config, caching, hooks, metrics, ablations, reporting, and CI are built for reproducible local pilots on small open-weight models.

Hypothesis (one line): Head substitutability is heavy-tailed and predictable, and more importantly it survives deployment: heads identified as substitutable can have their Q and K projections physically deleted and be retrained with a fixed programmatic pattern, recovering most of the perplexity cost that hard post-hoc substitution incurs, while genuinely reducing parameters rather than merely masking a still-computed pattern.

## Motivation

Interpretability and safety claims fail in practice for boring engineering
reasons: unpinned weights, chat templates skipped, invalid layer indices,
intervals that span zero treated as nulls, and stages that raise
`NotImplementedError`. This repo treats those as first-class bugs.

## Architecture / Pipeline

```mermaid
flowchart LR
  cfg[Hydra config] --> seed[set_seed]
  seed --> data[build dataset]
  data --> model[load pinned model]
  model --> stages[experiment stages]
  stages --> cache[artifact cache]
  stages --> eval[evaluation harness]
  eval --> agg[aggregate]
  agg --> tables[MD + LaTeX tables]
  agg --> figs[PDF/SVG/PNG figures]
```

| Stage | Module | Output |
|---|---|---|
| Compose config | `configs/` + `progattn.configs` | resolved `config.yaml` |
| Build data | `progattn.data` | splits + manifest |
| Load model | `progattn.models` | `LoadedModel` + resolved commit |
| Run stages | `scripts/run_experiment.py` | per-stage JSON |
| Aggregate | `progattn.reporting` | `results.json` + tables + figures |

## Results

| Experiment | Metric | Value | Provenance |
|---|---|---:|---|
| smoke | config compose | pass | unit / CI |
| pilot | harness recovery | pending | labelled synthetic until measured |

**Provenance.** No measured number in this table comes from a full model run on
private data. Synthetic harness-validation outputs are labelled
`is_synthetic: true` and must not be reported as empirical results.

## Repository guide

```
.
├── configs/           # Hydra groups + experiment presets
├── src/progattn/       # installable library (print-free)
├── scripts/           # CLIs with argparse / hydra
├── tests/             # ≥30 modules; tiny random GPT-2 only
├── data/              # manifests only
├── docs/              # DESIGN.md, HARDWARE.md
├── TASK.md            # research plan + DAG
└── Makefile           # install, lint, test, ci, pilot, doctor
```

| Command | Purpose |
|---|---|
| `make install-dev` | editable install + pinned requirements |
| `make test` | full unit suite |
| `make ci` | lint + test + typecheck + api-contract + coverage |
| `make pilot` | end-to-end pilot profile |
| `make doctor` | environment / device report |

## Status

Shared spine is in place. Domain-specific stages land behind the experiment
registry and must pass the harness-validation script before any measured claim.

## Related work

- Complexity bar: Critical Data PRIMED-AI / RecursiveJEPA engineering standard

## Citation

```bibtex
@misc{programmatic_attention_pilot,
  title        = {Deploying Programmatic Attention in Transformers},
  author       = {Alana Sung},
  year         = {2026},
  howpublished = {Technical report},
}
```

## License

MIT. Model weights and third-party datasets retain their upstream licenses.

---

<p align="center">Built for reproducible interpretability pilots on Apple Silicon and CI CPUs.</p>

## Design constraints (short)

1. Library code has zero `print`, zero `argparse`, zero `__main__`.
2. Every result JSON carries `task`, `seed`, `git_sha`, `n`.
3. Model revisions are pinned; load path records the resolved commit.
4. Chat templates are applied when available; the path is recorded.
5. MPS sets `PYTORCH_ENABLE_MPS_FALLBACK` and records the flag.
6. CI spanning zero is inconclusive; report MDE and run TOST before null claims.
7. Pilot `n_items >= 512`.
8. Layer indices are validated against `n_layers`.

## Hardware note

Torch model forward passes may use MPS on Apple Silicon. Sklearn, numpy,
pandas, and matplotlib figure generation run on CPU (see `docs/HARDWARE.md`).

## Config composition

```bash
python scripts/run_config_smoke_test.py experiment=pilot model=gpt2 seed=7
python scripts/run_experiment.py experiment=baseline eval.layers=[2,4,6]
```

## Ablations

Ablations live under `src/progattn/ablation/` and return structured dicts. Presets
mirror them under `configs/experiment/ablation_*.yaml`.

## Reporting

```bash
python scripts/aggregate_results.py
python scripts/make_tables.py
python scripts/make_figures.py
```

One aggregation command regenerates Markdown and booktabs LaTeX from raw JSONs.

## Contributing

See `CONTRIBUTING.md`. Open work goes in `TASK.md` / GitHub issues — never as
`TODO` comments in library code.
