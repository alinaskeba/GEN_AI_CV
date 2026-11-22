# Computer Vision Homeworks

Implementations, experiments and results for the computer vision course.

## Homework 1: Generative Models

| Topic | Notebook | Experiment overview |
|---|---|---|
| Normalizing flows | [RealNVP](HW1/RealNVP.ipynb) | [Architecture and experiments](HW1/README.md) |
| Autoencoders and VAE | [AE, VAE, flow-VAE and beta-VAE](HW1/VAE.ipynb) | [Methods and evaluation](HW1/VAE_README.md) |
| GAN | [Vanilla, conditional, advanced and progressive GAN](HW1/GAN.ipynb) | [Methods and evaluation](HW1/GAN_README.md) |

The notebooks contain saved experiment outputs. The [results index](HW1/results/README.md) links to metrics, learning curves, reconstructions and generated samples for all eight experiment sections.

## Environment and reproduction

Use Python 3.11 or newer with a compatible PyTorch build. Install dependencies in the environment used by the notebook kernel:

```bash
python -m pip install -r requirements.txt
```

GPU training requires a CUDA-enabled PyTorch build and `torch.cuda.is_available()` returning `True`. The notebooks do not require `torchvision`. Recorded package versions and device information are available in each run's `environment.json`.

Start notebook kernels with `HW1/` as their working directory and run cells in order. Full experiments use `SMOKE_TEST=False`. Each execution trains new models and writes results to a new run directory. No previous weights are required. Datasets and checkpoints are stored under the Git-ignored `.local/` directory.

## Structure

```text
requirements.txt
HW1/
  RealNVP.ipynb
  VAE.ipynb
  GAN.ipynb
  README.md
  VAE_README.md
  GAN_README.md
  results/
    README.md
    selected_runs.json
    full/                 # RealNVP
    autoencoder/full/
    vae/full/
    flow_vae/full/
    beta_vae/full/
    gan/full/
    conditional_gan/full/
    progressive_gan/full/
```
