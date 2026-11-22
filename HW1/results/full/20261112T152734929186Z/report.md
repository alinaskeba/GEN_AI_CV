# HW1: RealNVP results

## Process
Trained four unconditional multi-scale RealNVP architectures from scratch on MNIST and Fashion-MNIST using fixed train/validation splits, uniform dequantization, the preprocessing Jacobian, Adam, cosine decay, gradient clipping and equal epoch budgets. Checkpoints and architectures were selected by validation bpd; only the selected models were evaluated on the test sets.

## Results

### mnist

| Experiment | Validation bpd | Best epoch | Parameters |
|---|---:|---:|---:|
| dense_mixing | 1.315428 | 40 | 1,615,740 |
| wide_baseline | 1.322854 | 40 | 1,615,260 |
| sparse_mixing | 1.335895 | 40 | 1,615,340 |
| compact_baseline | 1.526617 | 40 | 292,712 |

Validation winner: **dense_mixing**; test bpd **1.313278**, NLL **-3633.748 nats/image**.
Wide minus compact validation bpd: -0.203762. Sparse minus wide: +0.013041. Dense minus wide: -0.007427. Dense minus sparse: -0.020468. Negative means better likelihood.
Compact versus wide changes width and depth together. Sparse/dense comparisons share the wide backbone and full training budget.

### fashion_mnist

| Experiment | Validation bpd | Best epoch | Parameters |
|---|---:|---:|---:|
| dense_mixing | 3.165900 | 40 | 1,615,740 |
| wide_baseline | 3.170187 | 40 | 1,615,260 |
| sparse_mixing | 3.175409 | 40 | 1,615,340 |
| compact_baseline | 3.433495 | 35 | 292,712 |

Validation winner: **dense_mixing**; test bpd **3.168043**, NLL **-2625.818 nats/image**.
Wide minus compact validation bpd: -0.263308. Sparse minus wide: +0.005222. Dense minus wide: -0.004286. Dense minus sparse: -0.009508. Negative means better likelihood.
Compact versus wide changes width and depth together. Sparse/dense comparisons share the wide backbone and full training budget.

## What worked well?
Invertibility and analytic Jacobians passed numerical checks. The measured comparisons above identify which architecture choices improved likelihood in this seed and budget. Checkpoint selection uses validation rather than assuming the final epoch is best.

## What did not turn out as expected?
More capacity or more mixing may not win: the signed differences report negative experiments too. Earlier development showed ambiguous and disconnected digits despite improving bpd; inspect the final panels to assess whether those artifacts persist in these fresh runs.

## Why might quality remain limited?
Likelihood and visual realism differ. Limited spatial context, independent Gaussian split priors and finite training time are possible constraints, not proven causes. A widening train/validation gap suggests overfitting; both curves still improving suggests a larger budget may help. A single seed cannot establish significance. Out-of-range pixels from the contracted logit are clipped only for display and their fraction is recorded.

## How could results improve?
Repeat the best configuration across several seeds, train longer if validation still improves, or test global-context conditioners and conditional split priors as separate ablations. Temperature 0.7 changes the sampling distribution and may trade diversity for sharpness; it does not improve likelihood.

## Visual observations
After full training, inspect the final panels and add your own concrete observations on recognizable digits or clothing silhouettes, broken strokes, disconnected parts, noisy backgrounds and variety. Do not infer visual quality solely from bpd or full class coverage from one grid. Generations are unconditional examples, not reconstructions of the displayed originals.

## Artifacts and limitations
The result directory contains validation_comparison.csv, test_results.csv, environment.json, learning curves, per-run configuration and loss histories, intermediate generations, and originals beside best checkpoints. Reported bpd is a dequantization bound rather than exact discrete likelihood. Test sets were inspected during earlier development.