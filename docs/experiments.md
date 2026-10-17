# Earlier grayscale experiments

Historical setup and validation evidence, retained for comparison with the RGB
zoom work. Commands run from the repository root.

## Earlier unconditional experiment

This experiment uses **Warp to render antialiased black-and-white
Mandelbrot crops**, and **PyTorch to train an unconditional pixel-space DiT**
with the clean-image prediction flow loss from
[Modality Forcing, Eq. (7)](https://arxiv.org/abs/2606.13676).
The single-image objective is adapted here; this is not a reproduction of the
paper's multimodal model or pretrained text-to-image system.

## Black-and-white experiment on blacknoir

Work in `/home/blacknoir/Projects/mandelbrot-diffusion`.

```sh
uv run python mandelbrot.py --bw --aa-side 4
uv run python train.py --config configs/flow64.json
uv run python sample.py runs/dit-flow64-balanced-1000/checkpoint.pt --steps 50
```

The default config now uses `data/bw64-balanced-1000`: 900 training and 100
validation images at 64 x 64. Earlier datasets remain available for comparison.
Generation refuses to overwrite an existing directory. To generate another
dataset, select a new output directory and update the config's `data`.

Black means unescaped within the iteration budget, white means escaped, and
gray averages the coverage of boundary pixels. The renderer evaluates a 4 x 4
grid of subpixel samples using double-precision Warp arithmetic, then resolves
the 16 samples with a second Warp kernel. It rejects views where float64
coordinate spacing approaches the subpixel spacing. Reference-orbit deep zoom is not implemented for this grayscale renderer.

The revised sampler uses boundary anchors and exact marginal quotas across
six black-coverage bins spanning 20% to 80%, plus four horizontal span ranges:
0.05-0.12 (10%), 0.12-0.35 (25%), 0.35-1.0 (35%), and 1.0-2.5 (30%).
A coarse pilot rejects unsuitable candidates before antialiased rendering.
Iteration budgets are doubled from 1024 up to 16384 until successive coverage
images differ by at most 0.0005 on average. Difficult crops receive additional
iterations before rejection. This agreement is not proof of membership.

Train/validation anchor groups are disjoint and conjugate anchors stay together;
large crop footprints can overlap, so this is not a spatially disjoint test of
generalization. Exact duplicate quantized images are excluded across both splits.
Metadata, source hashes, quotas, rejection counts, and iteration budgets are in
`manifest.json`. `dataset.py export` creates individual PNGs, previews, and a ZIP
containing the tensor files and metadata. The generated 1000-image dataset has
mean black coverage 49.67%, with every crop between 20% and 80% black.

```sh
uv run python dataset.py prepare --output data/new-balanced --train-count 900 --val-count 100 --seed 20261015
uv run python dataset.py export data/new-balanced
```

## DiT and flow objective

`model.py` implements the [DiT architecture](https://www.wpeebles.com/DiT) using
plain PyTorch: non-overlapping patch embedding, fixed 2D sine/cosine positions,
multi-head self-attention, MLPs, and timestep-conditioned adaLN-Zero blocks.
Defaults are patch size 4, width 192, depth 6, six heads, and MLP ratio 4,
giving 256 tokens per image. The model directly predicts clean pixels (one channel by default, or RGB
with `"channels": 3` in the model config); there is no VAE, class conditioning, or learned variance head.

Images are mapped from [0,1] to [-1,1]. For clean image `x0`, Gaussian noise
`epsilon`, and noise level `t`:

```text
xt = (1-t)*x0 + t*epsilon                 # t=0 clean, t=1 noise
x0_hat = DiT(xt, t)
loss = mean(((x0_hat-xt)/t - (x0-epsilon))**2)
     = mean(((x0_hat-x0)/t)**2)
```

The timestep distribution is a standard logit-normal followed by the shift
`1.1*t / (1 + 0.1*t)`. Training clamps t to a minimum of 0.05 as an explicit
numerical choice for this small baseline, since the loss weights errors by
1/t squared. Sampling integrates `dx/dt=(x-x0_hat)/t` from 1 to 0 using 50
backward-time Euler steps. Only the final image is clipped for display.
The solver evaluates positive t; its final update reaches t=0 without division
by zero. The final sampling steps can fall below the training clamp, which is
a baseline tradeoff to revisit with longer runs and solver comparisons.

`train.py` uses AdamW, gradient clipping, EMA, fixed-noise validation, CSV
logging, dataset hashes, and checkpoints containing optimizer and RNG state.
Resume uses the total step count:

```sh
uv run python train.py --steps 40 --output runs/dit-flow64-smoke
uv run python train.py --steps 42 --output runs/dit-flow64-smoke \
  --resume runs/dit-flow64-smoke/checkpoint.pt
```

Architecture is stored in each checkpoint and reconstructed by the sampler.
Earlier unconditional U-Net smoke checkpoints are retired. New U-Net checkpoints
use the paired zoom formulation described above.
These historical smoke runs verify execution, not learned sample quality.
They do not describe the later RGB single-pair overfit.

The default DiT has
4,165,456 trainable parameters. A 40-step CUDA smoke run reduced fixed-noise
validation loss from 9.157 to 0.391. Resuming for two additional steps produced
model and EMA tensors exactly equal to a continuous 42-step run. Sampling
produced 16 finite grayscale images with 50 Euler steps. These early EMA
samples are execution checks, not evidence of useful fractal generation.
