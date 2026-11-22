# Flow-VAE: conditional RealNVP posterior

## Process
Trained a zero-flow control and a six-layer conditional RealNVP posterior VAE from scratch on MNIST and Fashion-MNIST. Both use the same convolutional backbone, latent size, seed, split, optimizer and budget. Only the posterior flow differs; parameter count is consequently larger. The generative prior is standard normal. The Bernoulli observation model uses dynamic binarization.

## Correct density and objective
For zK=f(z0; x), log qK=log q0-sum(log|det J|). Train with reconstruction NLL+log qK-log p(zK), averaged over examples. Do not reuse analytic Gaussian KL after transformation or clamp sampled log-density ratios. Test importance weights also use qK and the Jacobian. Generation samples the standard normal generative prior directly; no image-conditioned posterior flow is applied without an image.

## Results
| Dataset | Model | Recon NLL | MC KL | Negative ELBO | Negative IW estimate | Gray-proxy MSE |
|---|---|---:|---:|---:|---:|---:|
| mnist | standard_control | 69.654 | 25.730 | 95.384 | 91.575 | 0.007705 |
| mnist | flow_vae | 70.213 | 23.346 | 93.559 | 89.968 | 0.007960 |
| fashion_mnist | standard_control | 222.898 | 15.105 | 238.003 | 235.474 | 0.020265 |
| fashion_mnist | flow_vae | 221.653 | 15.236 | 236.889 | 234.623 | 0.017157 |
mnist: flow minus control negative importance-weighted estimate = -1.607 nats/image. Negative favors flow in this run. This is a report of preset models, not test-based hyperparameter selection.
fashion_mnist: flow minus control negative importance-weighted estimate = -0.850 nats/image. Negative favors flow in this run. This is a report of preset models, not test-based hyperparameter selection.

## What worked well?
Conditional inverses, analytic Jacobians, identity initialization, log-density change of variables and gradients passed checks. The tables measure whether the richer posterior helps the shared generative training protocol. Inspect prior images to assess visual benefit separately.

## What did not meet expectations?
Record if the flow fails to improve the bound, reconstructs less clearly, or leaves ambiguous generated images. Added flexibility does not guarantee improvement in a single fixed-budget run.

## Why might quality remain limited?
The prior and observation decoder remain simple even with a richer posterior. Bernoulli binarization discards grayscale texture. Monte Carlo KL introduces variance, optimization may be incomplete, and flow layers add capacity. A transformed base mean is not the exact posterior mean; its reconstruction MSE is only a diagnostic. One seed does not establish statistical significance.

## How could it improve?
Repeat seeds, inspect convergence, and ablate flow depth or conditioner width with a matched budget. A learned flow prior is a different extension and would change generation; do not confuse it with this posterior-flow model.

## Visual reflection to complete
Compare recognizable digits/garment silhouettes, missing parts, blur and diversity between the matched control and flow model using the saved panels. Do not infer complete class coverage from one grid.

## Artifacts and caveats
validation_comparison.csv, test_results.csv, per-run configuration/history, learning curves, full prior grids, binary observation samples, matched reconstruction and generation comparisons. Likelihood quantities are estimates/bounds on binarized images, not exact continuous grayscale likelihood. Test sets have already been inspected in prior development; checkpoint selection here uses validation only.