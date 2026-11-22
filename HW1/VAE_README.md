# HW1 — Autoencoders and VAE

[VAE.ipynb](VAE.ipynb) contains four studies on MNIST and Fashion-MNIST. The [results index](results/README.md) links to their metrics and visual comparisons.

## 1. Autoencoder compression

A deterministic MLP autoencoder compared latent sizes **2, 8, 16, 32, 64 and 128**. The encoder uses 784→256→128→latent layers; the decoder reverses these widths and ends with sigmoid. Each model was trained from scratch for 20 epochs with Adam and MSE. Checkpoints were selected by validation MSE.

Test evaluation includes MSE, MAE and aggregate PSNR. A training-set mean-image predictor provides a simple reference. Error-versus-size plots and same-image reconstruction panels show how the bottleneck affects quality. Larger latent spaces generally reduced error, with smaller gains at higher sizes.

The ratio 784/latent size describes dimensionality reduction, not compressed file size. Float32 latent storage and shared decoder costs are separate considerations.

## 2. Variational autoencoder

A convolutional encoder and decoder used a 32-dimensional diagonal-Gaussian posterior and standard normal prior. Training minimized negative ELBO: pixel-summed reconstruction BCE plus latent-summed Gaussian KL with weight 1.

The Bernoulli observation model used dynamically binarized images, with fixed validation/test seeds. Likelihood bounds therefore describe binary observations rather than original grayscale intensities. Both models were trained for 30 epochs, and checkpoints were selected by validation negative ELBO.

Test results include reconstruction NLL, KL, negative ELBO and a K=16 importance-weighted bound estimate in nats/image. These are bounds or estimates, not exact NLL. Grayscale-input reconstruction MSE is reported separately. Generation panels show both decoder probability maps and sampled Bernoulli images.

References: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114), [Importance Weighted Autoencoders](https://arxiv.org/abs/1509.00519).

## 3. Flow-VAE

Six conditional affine RealNVP coupling layers transformed the 32-dimensional posterior. Alternating masks and encoder mean/log-variance context allowed a more flexible distribution. The density calculation included the log-Jacobian: `log qK = log q0 - sum(log|det J|)`.

The KL term used the Monte Carlo ratio `log qK - log p(zK)`, rather than the Gaussian analytic formula after transformation. A zero-flow control and six-flow model shared the backbone, split, seed, Adam learning rate 5e-4 and 30-epoch budget. Checkpoint selection used validation negative ELBO; final evaluation used K=16 importance samples.

The flow improved the measured importance-weighted bound on both datasets in this run. This does not establish a general visual advantage. The prior remained standard normal: unconditional generation sampled it and decoded directly, without applying the image-conditioned posterior flow. Deterministic reconstructions used a transformed-base-mean proxy.

References: [Variational Inference with Normalizing Flows](https://arxiv.org/abs/1505.05770), [Real NVP](https://arxiv.org/abs/1605.08803).

## 4. Beta-VAE

Beta values **0.5, 1, 4 and 8** were compared with the same convolutional VAE, 32-dimensional latent space, splits, initialization, optimizer and 30-epoch budget. Training used `reconstruction_nll + beta * kl`. Each checkpoint was selected by its own validation beta objective; weighted objectives were not ranked across beta values.

The analysis reports reconstruction error, unweighted KL, ordinary ELBO, importance-weighted bounds and prior-generation pixel variation. Latent diagnostics include per-coordinate KL, posterior-mean variance, active units (variance > 0.01) and correlations. Traversals varied one coordinate from -3 to 3 with the others fixed, using zero-prior and validation-image anchors. Eight dimensions were selected by validation KL; statistics were saved for all 32.

Higher beta increased reconstruction error and reduced active-unit counts. Traversals show changes in shape and style, but several properties can change together. These datasets do not provide independent ground-truth factors, so the diagnostics do not prove semantic disentanglement. Pixel variation also does not measure semantic diversity or class coverage.

References: [Understanding disentangling in beta-VAE](https://arxiv.org/abs/1804.03599), [Limitations of unsupervised disentanglement](https://arxiv.org/abs/1811.12359).

## Evaluation protocol

The experiments used fixed 50,000/10,000 train/validation splits and separate official test sets. The [historical split and metric audit](results/audit/20261112T191701803177Z/audit.md) found no overlapping split indices or exact training-versus-validation/test image matches for either dataset. This does not rule out memorization. Test sets had been inspected during development; model selection used validation metrics.

## Artifacts and reproduction

Results are stored in `results/autoencoder/full/`, `results/vae/full/`, `results/flow_vae/full/` and `results/beta_vae/full/`, each under a run ID. They include metric CSVs, configurations, histories, learning curves, reconstruction/generation panels and latent diagnostics where relevant.

For a full run, install the [root requirements](../requirements.txt), start the kernel in `HW1/` and run cells in order with `SMOKE_TEST=False`. Datasets are cached under `../.local/datasets/`; checkpoints are stored under `../.local/hw1/<section>/checkpoints/<run-id>/`. No previous weights or `torchvision` installation are required.
