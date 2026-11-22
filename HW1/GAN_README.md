# HW1 — Generative Adversarial Networks

[GAN.ipynb](GAN.ipynb) presents Vanilla and conditional GAN experiments on MNIST/Fashion-MNIST, followed by progressive generation on CIFAR-10. The [results index](results/README.md) links to saved curves, metrics and image panels.

## 1. Vanilla GAN

The unconditional MLP generator maps 100-dimensional standard normal noise through widths 256→512→1024→784 and tanh. The discriminator maps flattened images through 512→256→1 with LeakyReLU hidden activations and a final logit. Real pixels are scaled to [-1,1].

Training used BCE-with-logits, a non-saturating generator loss and one D/G update per batch. Fake images were detached for the D update; D parameters were frozen during the G update while gradients passed through D. Both datasets were trained independently for 40 epochs with batch size 128 and Adam at 2e-4, betas (0.5, 0.999).

Saved diagnostics include train/validation losses, discriminator confidence, fixed/fresh noise batches, pixel variation, image progression and three additional final seeds. The final fixed-epoch model was reported rather than the model with the lowest G/D loss.

Reference: [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661).

## 2. Conditional GAN, PatchGAN and feature matching

Five variants shared parameter shapes, seed, split, initialization, optimizer and a 40-epoch budget on both datasets:

| Variant | Class input | D scoring | Feature matching weight |
|---|---|---|---:|
| Unconditional control | Zero vectors | Global | 0 |
| cGAN | One-hot labels | Global | 0 |
| cGAN + PatchGAN | One-hot labels | Patchwise | 0 |
| cGAN + feature matching | One-hot labels | Global | 10 |
| Combined variant | One-hot labels | Patchwise | 10 |

G concatenates Gaussian noise with ten label values. D receives images and broadcast label maps. The shared convolutional discriminator produces a 14×14 logit map with 14×14 receptive fields. Global scoring applies BCE after averaging logits; patch scoring averages BCE across individual patch logits. The matched unconditional control uses this same backbone, unlike the all-MLP Vanilla GAN.

Feature matching adds a generator loss between class-specific means of pooled discriminator features. Real features are detached; fake-feature gradients reach G through the frozen D. D still uses only adversarial BCE.

Class-organized grids use eight independent noise vectors per requested class and the same latent bank across variants. Within-class pixel MSE and extra seeds help identify repeated prototypes. Requested labels are not verified predictions, and pixel variation is not semantic diversity or class accuracy.

The saved MNIST models show substantial mode collapse: conditional variants often repeat one prototype per class, while the unconditional control largely repeats one shape. Fashion-MNIST has recognizable silhouettes but also noise and ambiguous classes. The modifications did not remove all collapse. The [noise and display audit](results/audit/20261113T210846160248Z/noise_audit.md) found distinct noise vectors and unique plot indices, supporting a model-behaviour explanation for repetition.

References: [Conditional GAN](https://arxiv.org/abs/1411.1784), [PatchGAN inspiration](https://arxiv.org/abs/1611.07004), [Feature matching](https://arxiv.org/abs/1606.03498).

## 3. Progressive Growing GAN

An unconditional CIFAR-10 model grew through **4→8→16→32** pixel stages. Each stage added active generator/critic blocks with ToRGB/FromRGB paths and alpha blending. Previous weights and Adam state were preserved. Future blocks were registered but received no gradients until activation.

The compact PGGAN-inspired implementation used equalized linear/convolution layers, generator PixelNorm, tanh RGB outputs and WGAN-GP. Gradient penalty and critic drift weights were 10 and 0.001. Each batch used one critic/G update with Adam at 1e-3, betas (0, 0.99). The critic had no batch-dependent normalization.

| Resolution | Fade epochs | Stable epochs | Batch size |
|---|---:|---:|---:|
| 4×4 | 0 | 4 | 128 |
| 8×8 | 4 | 4 | 128 |
| 16×16 | 4 | 8 | 64 |
| 32×32 | 8 | 16 | 64 |

The run completed 48 epochs. Alpha moved from 0 to 1 during fade-in and stayed at 1 during stabilization. The official training set was split into 45,000 training and 5,000 validation images; 10,000 test images were separate. Real images were downsampled to the active resolution.

Saved artifacts show losses, gradient penalty, alpha, resolution evolution, fade evolution, real/generated comparisons and extra seeds. Low-resolution outputs use nearest-neighbour enlargement for display. Final samples show varied colours and some object-like shapes, with blur and ambiguity remaining. There is no matched fixed-resolution control, so the run does not prove that progressive growth caused better quality or stability.

References: [Progressive Growing of GANs](https://arxiv.org/abs/1710.10196), [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).

## Evaluation and reproduction

MNIST/Fashion-MNIST experiments used fixed 50,000/10,000 train/validation splits. Test data were used for final image references and discriminator diagnostics, not optimization or checkpoint selection. GAN losses and critic diagnostics are not likelihoods, FID, exact Wasserstein distances or semantic coverage scores.

For a full run, install the [root requirements](../requirements.txt), start the kernel in `HW1/` and run cells in order with `SMOKE_TEST=False`. The notebooks do not require `torchvision` or previous weights. CIFAR-10 is downloaded from its official binary archive and verified against its published MD5.

Results are stored under `results/gan/full/`, `results/conditional_gan/full/` and `results/progressive_gan/full/`, each with a run ID. Checkpoints are saved under the Git-ignored `../.local/hw1/<section>/checkpoints/<run-id>/` directories.
