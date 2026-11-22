# Autoencoder Compression report

## Process
Trained a deterministic MLP encoder-decoder on MNIST and Fashion-MNIST. Compared latent sizes [2, 8, 16, 32, 64, 128]. Kept splits, optimizer, epoch budget and batch order fixed. Selected checkpoints by validation MSE and evaluated every preset latent size on the test set. Labels were not model inputs.

## Results

### mnist

| Latent size | Validation MSE | Test MSE | Test MAE | Aggregate PSNR (dB) | Dimension ratio |
|---|---:|---:|---:|---:|---:|
| 2 | 0.039003 | 0.038845 | 0.091319 | 14.11 | 392.00x |
| 8 | 0.018873 | 0.018482 | 0.050581 | 17.33 | 98.00x |
| 16 | 0.012048 | 0.011701 | 0.036879 | 19.32 | 49.00x |
| 32 | 0.009366 | 0.009069 | 0.031133 | 20.42 | 24.50x |
| 64 | 0.008928 | 0.008654 | 0.030271 | 20.63 | 12.25x |
| 128 | 0.007940 | 0.007676 | 0.028283 | 21.15 | 6.12x |

Lowest validation MSE: d=128, 0.007940. Mean-image test reference MSE: 0.067470.
The ratio 784/d is dimensionality reduction. A float32 code costs 4d bytes, excluding decoder weights, quantization and file compression.

### fashion_mnist

| Latent size | Validation MSE | Test MSE | Test MAE | Aggregate PSNR (dB) | Dimension ratio |
|---|---:|---:|---:|---:|---:|
| 2 | 0.026953 | 0.026848 | 0.092773 | 15.71 | 392.00x |
| 8 | 0.014669 | 0.014627 | 0.063909 | 18.35 | 98.00x |
| 16 | 0.012693 | 0.012677 | 0.059340 | 18.97 | 49.00x |
| 32 | 0.011927 | 0.011923 | 0.057464 | 19.24 | 24.50x |
| 64 | 0.011949 | 0.011929 | 0.057377 | 19.23 | 12.25x |
| 128 | 0.011746 | 0.011708 | 0.056454 | 19.32 | 6.12x |

Lowest validation MSE: d=128, 0.011746. Mean-image test reference MSE: 0.086642.
The ratio 784/d is dimensionality reduction. A float32 code costs 4d bytes, excluding decoder weights, quantization and file compression.

## What worked well?
Use the measured tables and curves to identify which codes reconstruct best and how they compare with the mean-image reference. Checkpoint selection and shared input images make the comparisons consistent.

## What did not turn out as expected?
Check whether error decreases monotonically with latent size. Larger codes may still optimize poorly under a fixed budget. Inspect whether lower MSE also improves thin details or mainly average pixel accuracy; describe actual examples.

## Why might quality be limited?
Small bottlenecks discard information; MSE can favor smooth averages. An MLP lacks the spatial inductive bias of convolutions. Changing latent size also changes parameter count. One seed and a fixed budget limit conclusions. Inspect train/validation curves for under-training or overfitting.

## How could results improve?
Try a convolutional autoencoder, more training if validation still improves, or a different reconstruction objective as separate controlled experiments. For storage compression, test quantized codes and include shared model overhead.

## Visual observations to complete
Compare the smallest and largest codes on identical images. Describe recognizable strokes or clothing silhouettes, blur and missing details. Compare MNIST with Fashion-MNIST using the saved reconstruction panels.

## Artifacts and limitations
validation_comparison.csv, test_results.csv, reconstruction_error_vs_latent.png, dataset learning curves, reconstruction comparisons and per-run configurations/history/progress figures. Weights are local and ignored by Git. This is a deterministic autoencoder: reconstruction MSE is not likelihood or ELBO.