# GAN random-vector and display audit

Read-only CPU audit of the executed GAN notebook and actual trained generator weights. No notebook or trained checkpoint was edited.

## Input randomness and display indexing
- Training uses a fresh torch.randn batch separately for the D step and G step. Capturing two consecutive steps confirmed 8/8 distinct input vectors in each vanilla, conditional and progressive update-function probe.
- The conditional display bank has 80 distinct 100-dimensional normal vectors. Only labels repeat eight times per class; the noise vectors are not repeated.
- Fixed seed 123 intentionally reproduces the same latent bank across epochs and variants. Fresh seeds 1000+epoch and final seeds 201/202/203 provide different banks. Private display generators do not reset the global training RNG.
- Sentinel-image checks passed for vanilla, class-organized and RGB progressive grids: every source image index is used exactly once.

## Trained conditional generators
| Dataset | Variant | Unique z | Unique float images | Unique rounded uint8 images | Within-class pixel MSE | New noise / same class MSE |
|---|---|---:|---:|---:|---:|---:|
| mnist | unconditional_control | 80/80 | 80/80 | 80/80 | 0.00008510 | 0.00010869 |
| mnist | cgan | 80/80 | 80/80 | 80/80 | 0.00005526 | 0.00006087 |
| mnist | cgan_patch | 80/80 | 80/80 | 80/80 | 0.00002769 | 0.00002554 |
| mnist | cgan_feature_matching | 80/80 | 80/80 | 80/80 | 0.00015403 | 0.00015104 |
| mnist | cgan_patch_feature_matching | 80/80 | 80/80 | 80/80 | 0.00006752 | 0.00007003 |
| fashion_mnist | unconditional_control | 80/80 | 80/80 | 80/80 | 0.13522735 | 0.14235553 |
| fashion_mnist | cgan | 80/80 | 80/80 | 80/80 | 0.07117827 | 0.06839973 |
| fashion_mnist | cgan_patch | 80/80 | 80/80 | 80/80 | 0.09385644 | 0.07997209 |
| fashion_mnist | cgan_feature_matching | 80/80 | 80/80 | 80/80 | 0.07302192 | 0.07390467 |
| fashion_mnist | cgan_patch_feature_matching | 80/80 | 80/80 | 80/80 | 0.07471469 | 0.07521775 |

## Interpretation
No code-level reuse of one noise vector per row/batch or duplicated display indexing was found. Different z can nevertheless produce nearly identical images: this is a learned generator behavior consistent with mode collapse, not proof that inputs or plot cells are duplicated. Float-unique or quantized-unique images can still be perceptually near-identical.
The strong MNIST repetition remains a genuine diversity limitation. See noise_audit.json for per-class uniqueness and the relative response to changing labels versus changing noise.

## Scope
CPU generator streams differ from CUDA even at the same seed; the measurements above use fresh CPU inputs with saved learned weights. They do not reproduce old CUDA pixels bit for bit. These tests rule out the inspected indexing/RNG mistakes, not every possible training failure.