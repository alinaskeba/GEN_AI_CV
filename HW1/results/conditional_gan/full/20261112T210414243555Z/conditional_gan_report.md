# Conditional and Advanced GAN Variants

## Protocol
Trained five matched variants from scratch on MNIST and Fashion-MNIST: unconditional control, cGAN, cGAN+PatchGAN, cGAN+feature matching, and cGAN+PatchGAN+feature matching. Shared MLP generator and convolutional discriminator parameter shapes, seed, split, optimizer, labels/noise sequence and budget. Unconditional controls zero the conditioning channels. Part 1 remains a historical all-MLP baseline, not the matched conditioning ablation.

## What changed?
G concatenates one-hot class labels with noise. D concatenates spatial one-hot maps with images. Global D applies BCE after averaging the logit map; PatchGAN averages BCE across 14x14 logits with 14x14 receptive fields. Feature matching adds weight-10 mean-squared differences between real/fake class-specific mean features to G loss; D still uses only adversarial BCE. Real features are detached, fake-feature gradients reach G, and both branches use the same current frozen D.

## Recorded diagnostics
| Dataset | Variant | Final train G adv | Final train FM | Final train D | Mean within-class pixel MSE (3 seeds) |
|---|---|---:|---:|---:|---:|
| mnist | unconditional_control | 1.1426 | 0.000000 | 0.5292 | 0.000091 |
| mnist | cgan | 0.7789 | 0.000000 | 0.6533 | 0.000051 |
| mnist | cgan_patch | 0.7542 | 0.000000 | 0.6690 | 0.000027 |
| mnist | cgan_feature_matching | 0.8051 | 0.008897 | 0.6418 | 0.000169 |
| mnist | cgan_patch_feature_matching | 0.7565 | 0.000841 | 0.6700 | 0.000070 |
| fashion_mnist | unconditional_control | 0.7585 | 0.000000 | 0.6667 | 0.129644 |
| fashion_mnist | cgan | 0.7018 | 0.000000 | 0.6912 | 0.065375 |
| fashion_mnist | cgan_patch | 0.6962 | 0.000000 | 0.6923 | 0.088551 |
| fashion_mnist | cgan_feature_matching | 0.7019 | 0.000428 | 0.6905 | 0.077069 |
| fashion_mnist | cgan_patch_feature_matching | 0.7001 | 0.000111 | 0.6910 | 0.079243 |

## Conditioning and label correctness
Compare cGAN with the matched unconditional control. Rows indicate requested labels, not predicted labels. Describe whether generated content actually follows digit/clothing classes using real class references. D is not a semantic class-accuracy evaluator.

## PatchGAN effect
Compare cGAN_patch with cGAN. Describe local details and global shape consistency. The patch receptive field is local; improved texture need not improve entire-object structure. Do not interpret lower patch BCE as a universal quality gain.

## Feature matching effect
Compare cGAN_feature_matching with cGAN and the combined variant with cgan_patch. Describe sharpness, repetition and stability. Mean feature matching may help aggregate statistics while still allowing repetition; feature scales differ between trained discriminators.

## Diversity and mode collapse
Inspect multiple fixed/fresh rows over time and final seeds. A generator can produce one prototype per class and still collapse within classes. The recorded within-class pixel-MSE proxy measures pixel variation, not semantic diversity; noise can inflate it. Small class grids do not establish full mode coverage.

## What worked / what did not?
Add concrete dataset-specific findings from all_variants panels and progress grids. Record incorrect requested classes, merged strokes, missing garment parts, blur, repeated prototypes or unstable epochs when observed. Include negative experiments rather than assuming the combined model must win.

## Improvements and limitations
Repeat seeds, tune FM weight using validation/development evidence, or test a convolutional generator as a separate change. This is one seed and finite budget. Discriminator/FM diagnostics are not likelihood, FID or class accuracy. Final checkpoints use fixed epoch budgets, not test-based selection. Test sets have been inspected during earlier development.

## Artifacts
run_summary.csv, test_discriminator_diagnostics.csv, within_class_diversity.csv, per-run config/history, fixed/fresh class grids, final seeds, variant curves, real class references, final per-variant grids and all_variants panels. Weights are stored locally outside Git.