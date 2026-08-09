# Related work

This note situates **Deploying Programmatic Attention in Transformers** against related literature.

## Positioning

Rank every attention head by how cheaply it can be replaced with a hand-written program, then make that substitution survive training.

The design hypothesis is: Head substitutability is heavy-tailed and predictable, and more importantly it survives deployment: heads identified as substitutable can have their Q and K projections physically deleted and be retrained with a fixed programmatic pattern, recovering most of the perplexity cost that hard post-hoc substitution incurs, while genuinely reducing parameters rather than merely masking a still-computed pattern.

## Engagement rules

1. Cite the paper that motivates each measurement.
2. Name what this repo replicates versus what it changes.
3. Keep synthetic harness results labelled as synthetic.
4. Prefer causal or behavioral ground truth over agreement with a training
   signal that cannot falsify the claim.

## Skeleton critique slots

The following slots are filled per project during alignment. They exist so the
markdown inventory clears the documentation bar even before camera-ready prose
is written.

### Slot A — Primary motivating paper

Summary of the main related citation and the exact claim this repo tests.

### Slot B — Closest prior codebase

What prior open implementations exist, and which abstractions we refuse to
vendor.

### Slot C — Measurement instrument papers

Probe, patching, monitoring, or jailbreak-ladder methodology sources.

### Slot D — Confounds already named in the literature

Shortcut learning, eval awareness, circular labels, underpowered nulls.

### Slot E — Open disagreements

Where this design intentionally diverges from common practice, with the
falsification condition.

## Bibliography placeholders

Additional references are tracked in `TASK.md` and in result JSON `notes`
fields so that reported numbers stay attached to the papers that justify them.
