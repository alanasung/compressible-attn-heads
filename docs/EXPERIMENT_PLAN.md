# Experiment plan — Attention Heads That Collapse to Tiny Programs

Stage-by-stage design. Each stage is registered in `src/progattn/stages.py`
and appears in `python -m progattn stages`.

## Stages

| stage | responsibility |
|---|---|
| `patterns` | programmatic QK pattern generators |
| `substitute` | forward hooks that swap attention patterns |
| `sweep` | per-head, per-program substitutability sweep |
| `schedule` | greedy joint replacement schedule |
| `relax` | soft-blend and straight-through differentiable variants |
| `surgery` | annealing the gate and physically deleting Q/K projections |
| `tasks` | wikitext, LAMBADA, BLiMP, and IOI evaluation harnesses |
| `efficiency` | wall-clock, FLOP, and parameter-count accounting |

## Execution order

Stages form a linear dependency chain by default; the runner resolves the order
topologically, so a stage may be run alone and its prerequisites are pulled in
automatically:

```bash
python -m progattn run -c configs/pilot.yaml --stage efficiency
```

## Controls and their purpose

- Perplexity on a narrow corpus understates damage. Evaluate on a held-out mixed corpus and at least one structured task.
- A head can look substitutable in isolation but not jointly, since heads compensate. The greedy schedule must re-measure after each replacement.
- Masking a pattern while still computing Q and K proves nothing about deployment. Only measured parameter and FLOP removal counts as evidence.

## Decision rules

Pre-registered: if fewer than 15% of heads accept a programmatic pattern at under 0.05 nats of KL, the deployment premise is weak and the project pivots to characterizing why. If retrained-with-removal perplexity comes within 2% of baseline at 25% of heads converted, that is a strong positive and justifies scaling to a larger model.

## Reproducibility

Every run records a manifest with the git sha, a config fingerprint, resolved
device and dtype, package versions, per-stage timings, and metrics. Seeds are
set across python, numpy, and torch. Known determinism limits are recorded in
the manifest rather than assumed away: MPS does not support
`torch.use_deterministic_algorithms`, so small numeric drift between runs is
expected and should not be read as an effect.

## Scale

The pilot profile is what actually runs on the target machine. The full profile
describes the intended scaled-up run. When reporting any result, state which
profile produced it; a pilot-scale null is weaker evidence than a full-scale
null and the writeup must not blur them.
