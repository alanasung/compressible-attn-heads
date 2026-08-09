# Proposal: Substitutability Ladder for Programmatic QK Circuits

**Target project.** Deploying Programmatic Attention in Real Transformers
**Mentor.** Belinda Li (Anthropic)
**Research areas.** Mechanistic interpretability
**Posting.** https://sparai.org/projects/f26/reci1DhApjFAtQx7L

## Summary

Rank every attention head by how cheaply it can be replaced with a hand-written program, then make that substitution survive training.

## Hypothesis

Head substitutability is heavy-tailed and predictable, and more importantly it survives deployment: heads identified as substitutable can have their Q and K projections physically deleted and be retrained with a fixed programmatic pattern, recovering most of the perplexity cost that hard post-hoc substitution incurs, while genuinely reducing parameters rather than merely masking a still-computed pattern.

A hypothesis worth testing has to be able to lose. This one loses if the
measurements below come back null, and the design is built so that a null is
reportable rather than a dead end.

## Research questions

1. Per head, what is the next-token KL divergence when its attention pattern is replaced by a programmatic proxy (previous-token, induction, positional decay, delimiter-attending, BOS sink, uniform)?
2. Does a greedy schedule over per-head cost let us replace a large fraction of heads before perplexity degrades noticeably?
3. Hard-fixing a pattern kills the QK gradient. Do a gated soft blend with annealing, or a straight-through estimator, keep training stable? This is the mentor's stated application question, answered empirically.
4. After annealing, can the learned QK branch be deleted outright so the head is natively programmatic, and what does that cost?

## Method

1. Load a small pretrained Transformer and cache clean attention patterns.
2. Implement a library of programmatic pattern generators.
3. Sweep every (head, program) pair, measuring KL and downstream accuracy.
4. Build the substitutability ladder and a greedy replacement schedule.
5. Fine-tune under two differentiable relaxations, soft-blend and straight-through, and compare against hard substitution.
6. Anneal the blend gate to zero and then physically remove the Q and K projections for converted heads, reporting the parameter reduction and the perplexity cost of true removal.
7. Record wall-clock and FLOP deltas so the efficiency claim is measured rather than assumed.

## Measurements

- per-head KL to the unmodified model
- perplexity as a function of heads replaced
- downstream task accuracy retention on LAMBADA, BLiMP, IOI
- parameters actually removed after QK deletion
- measured wall-clock and FLOP change

## Threats to validity

- Perplexity on a narrow corpus understates damage. Evaluate on a held-out mixed corpus and at least one structured task.
- A head can look substitutable in isolation but not jointly, since heads compensate. The greedy schedule must re-measure after each replacement.
- Masking a pattern while still computing Q and K proves nothing about deployment. Only measured parameter and FLOP removal counts as evidence.

## Datasets

| role | dataset |
|---|---|
| `language_modeling` | wikitext-103-raw-v1 held-out split for perplexity |
| `structured_task` | IOI (indirect object identification) for circuit-level behavior that attention substitution should plausibly break |
| `downstream` | LAMBADA (last-word prediction) and BLiMP as accuracy checks that are sensitive to long-range attention |

## Relaxation failure modes

| method | failure mode |
|---|---|
| `soft_blend` | The gate can stall at an intermediate value, leaving the opaque learned QK branch permanently load-bearing. The head then looks converted but is not interpretable and nothing can be deleted. |
| `straight_through` | The surrogate gradient is biased, so the forward pass uses the programmatic pattern while the backward pass optimizes a different function. This can diverge, or converge to weights tuned for a pattern the model never actually runs. |

## Workstream choice

This repo commits to the TRAINING workstream, with the post-hoc substitutability sweep serving only as the head-selection instrument that the training work needs. The mentor's framing is a bridge from explainability to deployment, so the endpoint here is a model trained with programmatic QK heads whose learned Q and K projections have been removed, not a hook-based intervention on a frozen model. Kernel-level inference efficiency is explicitly out of scope and named as such.

## Pre-registered update thresholds

Pre-registered: if fewer than 15% of heads accept a programmatic pattern at under 0.05 nats of KL, the deployment premise is weak and the project pivots to characterizing why. If retrained-with-removal perplexity comes within 2% of baseline at 25% of heads converted, that is a strong positive and justifies scaling to a larger model.

## Feasibility

The pilot is written for an Apple M4 with 10 cores, unified memory, the PyTorch
MPS backend, no CUDA device, and no configured API keys. Model choices are
capped accordingly (openai-community/gpt2, Qwen/Qwen2.5-0.5B-Instruct). The
`full` profile documents the scaled-up version of the same experiment for when a
real GPU is available, so the reduction in scale is explicit rather than hidden.

## Relationship to the posting

This proposal was checked against the mentor's verbatim posting by an
independent model before implementation began. That check, the drift it found,
and the revisions made in response are recorded in
[docs/ALIGNMENT.md](ALIGNMENT.md).
