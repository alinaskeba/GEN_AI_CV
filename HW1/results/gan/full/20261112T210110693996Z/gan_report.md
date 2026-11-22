# Vanilla GAN report

## Process
Trained independent unconditional MLP GANs on MNIST and Fashion-MNIST. Used 100-dimensional normal noise, tanh generator outputs, real images scaled to [-1,1], discriminator logits, BCE-with-logits and the non-saturating generator objective. Adam settings and budgets match between datasets. Each batch makes one D update followed by one G update, with gradients isolated correctly.

## Recorded final training diagnostics
| Dataset | Epochs | Train G loss | Train D loss | Validation D(real) | Validation D(fake) | Fixed-batch pixel variation |
|---|---:|---:|---:|---:|---:|---:|
| mnist | 40 | 1.0340 | 0.5937 | 0.469 | 0.305 | 0.119504 |
| fashion_mnist | 40 | 0.8571 | 0.6590 | 0.594 | 0.525 | 0.159301 |

## What worked well?
Use the fixed/fresh grids to identify recognizable digits or clothing silhouettes and whether quality improves over epochs. The numerical checks verify both networks update and the generator step propagates through a parameter-frozen discriminator. Do not conclude that low adversarial loss means good generation.

## What did not turn out as expected?
Record blurry or malformed shapes, repeated outputs, unstable sample quality or confidence imbalance if observed. Cite actual epochs/seeds and compare both datasets. The fully connected baseline may struggle with spatial structure, especially detailed clothing.

## Mode-collapse inspection
Inspect fixed_epoch and fresh_epoch grids across time and final seeds 201/202/203. Describe repeated near-identical shapes, disappearance of variety or missing recognizable categories when visible. The pairwise pixel-MSE proxy detects low pixel variation but is not semantic coverage and can be inflated by noise. Small grids cannot prove all classes are covered or exclude mode collapse.

## Why might results be limited?
Adversarial training can oscillate or become imbalanced; D may become too confident or G may exploit a narrow set of outputs. Limited MLP spatial inductive bias and finite training budget are possible factors, not proven causes. Compare the loss/confidence trajectories with sample sequences instead of interpreting either alone.

## How could results improve?
Test a convolutional architecture, learning-rate/update balance or longer training as controlled experiments. Conditional labels, PatchGAN and feature matching are the next homework sections, not implemented in this vanilla baseline. Repeat seeds before making strong claims.

## Visual findings to complete
For each dataset, identify specific generated batches/epochs that show improvement or instability. State whether repeated shapes are visible and compare final samples to real references. Include counterexamples rather than choosing only attractive outputs.

## Artifacts and limitations
run_summary.csv, test_discriminator_diagnostics.csv, environment.json, per-dataset config.json/history.csv, fixed/fresh progress grids, loss/confidence curves, pixel-variation plots, progress montage, real/generated panels and three additional final seeds. Final checkpoint selection is fixed by epoch budget, not adversarial loss. These metrics are not likelihood or perceptual quality scores. Test sets were inspected during earlier homework development.