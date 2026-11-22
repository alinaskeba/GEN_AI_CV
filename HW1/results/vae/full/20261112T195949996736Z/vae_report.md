# Variational Autoencoder report

## Process
Trained the same convolutional VAE from scratch on MNIST and Fashion-MNIST, with latent size 32, standard Gaussian prior, diagonal Gaussian posterior and KL weight 1. Used a Bernoulli decoder on dynamically binarized images, fixed validation/test randomness, validation negative-ELBO checkpoint selection and a shared training budget.

## Held-out measurements
| Dataset | Best epoch | Recon NLL | KL | Negative ELBO | Negative IW estimate | Gray-input MSE |
|---|---:|---:|---:|---:|---:|---:|
| mnist | 29 | 73.717 | 22.412 | 96.129 | 92.161 | 0.009818 |
| fashion_mnist | 30 | 221.530 | 15.704 | 237.233 | 234.679 | 0.018563 |

Likelihood-related quantities are nats per binary image. The negative importance-weighted value is a finite Monte Carlo estimate of a likelihood bound, not exact NLL. Gray-input MSE is a separate deterministic diagnostic, not the observation likelihood. Scores are not directly comparable to the grayscale autoencoder MSE objective or the RealNVP dequantized density task.

## What worked well?
Use the recorded curves and table to identify how the same model learned each dataset. Inspect recognizable digit strokes or garment silhouettes in prior-generated probability grids. Generation uses z drawn from the prior rather than encoding a training image.

## What did not turn out as expected?
Record blurry details, merged strokes, unclear garment parts, low variety or noisy binary samples if actually observed. Bernoulli pixel draws can look noisier than the displayed decoder means; these are different visualizations of the same observation model.

## Why might results be limited?
The ELBO balances reconstruction against matching a simple prior. A diagonal approximate posterior and conditionally independent Bernoulli pixels can limit the model; binarization removes grayscale information, especially relevant to clothing textures. These are hypotheses rather than proven causes. Inspect train/validation curves for under-training or overfitting and KL for limited latent usage. One seed and small grids do not establish statistical significance or full class coverage.

## How could results improve?
Repeat seeds, train longer if validation still improves, or compare latent sizes and a more expressive decoder in controlled experiments. A continuous observation distribution could model original grayscale intensities directly. Flow-VAE and beta-VAE are later sections, not part of this standard-VAE run.

## Visual dataset comparison to complete
Describe concrete differences between MNIST and Fashion-MNIST reconstruction panels and prior samples. Relate those observations to the measured reconstruction/KL tradeoff without assuming lower negative ELBO guarantees better visual quality.

## Artifacts
validation_comparison.csv, test_results.csv, environment.json, per-dataset config.json/history.csv, loss curves, best reconstructions, prior probability grids, Bernoulli observation samples, and datasets_generation_comparison.png. Weights are stored locally outside Git.