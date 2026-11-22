# Beta-VAE exploration

## Protocol
Compared beta values [0.5, 1.0, 4.0, 8.0] on MNIST and Fashion-MNIST with the same 32-dimensional convolutional VAE, seed, split, initialization, optimizer and 30-epoch budget. Beta is constant during each run. Checkpoint selection uses each run's validation reconstruction NLL + beta * KL, never test data. Observations are dynamically binarized.

## Results
| Dataset | Beta | Gray MSE | Recon NLL | KL | Standard -ELBO | Active units (validation) | Prior pixel diversity |
|---|---:|---:|---:|---:|---:|---:|---:|
| mnist | 0.5 | 0.006360 | 61.587 | 35.645 | 97.232 | 20 | 0.099261 |
| mnist | 1 | 0.009116 | 72.615 | 23.043 | 95.658 | 14 | 0.092374 |
| mnist | 4 | 0.018186 | 103.659 | 9.367 | 113.025 | 14 | 0.082659 |
| mnist | 8 | 0.029530 | 132.627 | 4.825 | 137.451 | 5 | 0.051775 |
| fashion_mnist | 0.5 | 0.017166 | 214.498 | 24.045 | 238.543 | 24 | 0.129051 |
| fashion_mnist | 1 | 0.018464 | 221.045 | 15.948 | 236.993 | 20 | 0.124261 |
| fashion_mnist | 4 | 0.021895 | 239.966 | 6.716 | 246.682 | 6 | 0.133997 |
| fashion_mnist | 8 | 0.024790 | 254.363 | 4.384 | 258.747 | 4 | 0.121264 |

## Reconstruction quality
Use the measured MSE/reconstruction NLL and matched-image panels to describe the cost or benefit of increasing beta. Weighted objectives should not be compared directly across beta values. Standard ELBO and importance-weighted estimates are recorded separately with unit KL weight; they are bounds/estimates, not exact NLL.

## Latent usage and disentanglement
Active units use validation variance of posterior means >0.01. Per-coordinate KL and correlation are descriptive diagnostics, not semantic disentanglement scores. Traversal dimensions are chosen by validation mean KL, not by test scores or attractive images. Ground-truth independent factors are unavailable here; no MIG/DCI claim is made. Individual axes may permute/rotate across runs.

## Generation diversity
The reported diversity is mean pairwise per-pixel MSE of 64 fixed-seed prior decoder means. It detects identical outputs but can reward unrealistic variation; it is not class coverage or semantic diversity. Inspect the full prior grids before drawing conclusions. Different beta values can change the quality/diversity tradeoff without a monotonic trend.

## Visual traversal findings to complete
For each dataset, identify concrete dimensions/rows where changing one coordinate affects stroke thickness, slant, garment width, shape or several attributes together. Compare zero-prior and validation-image anchors. Other coordinates are fixed; values are overwritten on the same -3 to 3 prior scale. Far-away values can leave the learned aggregate-posterior region.

## What worked / what did not?
Record whether beta=1, weak regularization or stronger regularization gives the best observed tradeoff. Note inactive dimensions, blur, reduced variety or mixed changes when present. More regularization is not proof of better factors; collapsed dimensions can produce nearly unchanged traversals.

## Limitations and improvements
One seed and a fixed budget limit reliability. MNIST/Fashion-MNIST do not give independent generative factors. A stronger study would repeat seeds and use a factor-annotated dataset with a justified metric. Test sets have already been inspected during development; selection in this section uses validation only.

## Artifacts
validation_comparison.csv, test_results.csv, latent_diagnostics.csv, beta_tradeoffs.png, dataset curves and reconstruction/generation comparisons. Per-run folders contain history/config, validation_latent_dimensions.csv, selected traversal IDs, anchor panels and zero/encoded-anchor traversal figures. Weights are stored locally outside Git.