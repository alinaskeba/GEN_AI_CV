# AE / VAE result audit

Checked the saved AE run 20261112T183733962880Z and VAE run 20261112T184058340619Z against the current notebook code and actual split indices.

## Split checks
- mnist: 50,000 training / 10,000 validation / 10,000 official test images; train/validation source-index overlap = 0. Exact original pixel patterns shared across train/validation: 0; training/official test: 0.
- fashion_mnist: 50,000 training / 10,000 validation / 10,000 official test images; train/validation source-index overlap = 0. Exact original pixel patterns shared across train/validation: 0; training/official test: 0.

## Findings
- No code-level train/validation/test leakage found: optimization uses training batches; checkpoints use validation only; official test files are loaded separately. Labels are not training inputs. The AE mean-image reference is fit on training data only.
- Checked all 12 AE selected epochs and both VAE selected epochs against history minima. VAE checkpoint epoch metadata matches the recorded selections.
- Saved metrics are finite. AE PSNR matches the documented dataset-MSE formula. VAE negative ELBO equals reconstruction NLL plus KL within floating-point tolerance; KL is positive in the reported dataset averages.
- The standard VAE uses dynamically binarized data and reports a valid Bernoulli observation-model bound. Gray-input MSE is a separate out-of-training-input-distribution diagnostic. Decoder probability grids are conditional means, not binary observation samples; both are labeled.
- Prior grids visually contain recognizable MNIST digits and clothing silhouettes, with blur and some ambiguous forms. This is not evidence of leakage or proof that individual training examples were never memorized.

## Caveats and safeguards
- Exact duplicate pixel patterns may already occur in the official data. The overlap counts above are not caused by reusing split indices. No content-level deduplication was applied; changing official splits now would invalidate direct comparisons with earlier runs.
- Test sets were inspected in earlier development. Treat this as an iterative homework benchmark, not a fresh untouched holdout; do not tune the next model based on test scores.
- This audit checks code, stored metadata and arithmetic and loads actual checkpoints; it does not re-run the original full GPU training or guarantee perfect generalization.
- A flow posterior cannot use the old analytic diagonal-Gaussian KL after transformation. The next section must use log q_K = log q_0 - log|det J| and a Monte Carlo log-density ratio.

## Submission cleanup note (2026-11-13)

This is a historical audit. Its original completed AE/VAE runs were archived under the Git-ignored local submission_cleanup directory when the runs referenced by the final notebook outputs were selected. See the [current submission review](../../../SUBMISSION_REVIEW.md) and [completed-results index](../../README.md) for the retained runs. The original audit findings and metrics above were not rewritten.
