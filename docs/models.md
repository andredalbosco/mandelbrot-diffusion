# Models and training

Run commands from the repository root. For the million-image dataset, see
[fast training](training-speed.md) and the [JiT recipe audit](jit-training-audit.md).

## Mixed zoom and unconditional training

```sh
uv run python train.py --config configs/zoom4x-smooth-100k-jit-1m.json
# Resume after interruption:
uv run python train.py --config configs/zoom4x-smooth-100k-jit-1m.json --resume runs/jit-1m-smooth-100k/checkpoint.pt
```

The mask-aware JiT consumes noisy RGB, fixed upsampled evidence, and one coverage
channel. It predicts clean RGB for interior, edge, corner, and unconditional
examples using the same weights. Missing evidence is neutral zero internally.
The 100k configuration holds out 10k examples and trains on 90k, using streamed
batches of 64, BF16, AdamW, EMA, and a provisional 20,000-update budget.

For clean target `x`, Gaussian noise `e`, and `t = sigmoid(N(-0.8, 0.8²))`:

```text
z = (1-t)*e + t*x
prediction = model(z, t, evidence, mask)
loss = mean(((prediction-x) / max(1-t, 0.05))²)
```

Sampling defaults to 16 Euler steps for both conditional and unconditional modes.
Both solvers clamp the velocity denominator consistently with training.
Use `--solver heun --steps 50` for JiT-style Heun integration, with a final Euler
step (99 model evaluations without guidance). Training and checkpoint weights
are unchanged.

This is clean-image prediction with velocity-equivalent rectified-flow weighting
and endpoint clamping. One Euler step returns the prediction directly, but this
objective does not guarantee good one-step generation. Fixed validation previews
compare one and sixteen steps. Preview columns are input, coverage, one step,
sixteen steps, target; rows group interior, edge, corner, unconditional examples.
Validation loss uses 256 fixed examples balanced across those modes.

```sh
# Unconditional generation:
uv run python sample.py runs/jit-1m-smooth-100k/checkpoint.pt --steps 16 --output output/unconditional.png
# Conditional generation using saved evidence and masks:
uv run python sample.py runs/jit-1m-smooth-100k/checkpoint.pt --input data/zoom4x-smooth-blue-gold-100k-128/val.pt --steps 16 --output output/zoom.png
```

Checkpoints include model/EMA/optimizer/RNG state and dataset SHA256 hashes.
`loss.csv`, `status.json`, and numbered previews record progress. Older fully
observed configurations below retain their original paired formulation.

Generation and training refuse to overwrite existing outputs; resume a run or
choose a new output directory.

## Small JiT variants

Both variants refine a 32×32 crop upsampled to 128×128 into a rendered 4× zoom.
The source image is concatenated with the evolving flow state throughout sampling.
Both use 8×8 patches, a bottleneck patch embedding, RoPE, Q/K normalization,
RMSNorm, SwiGLU and clean RGB prediction. Class-context tokens are omitted.

| Config | Parameters | Width | Blocks | Heads | Patch bottleneck |
|---|---:|---:|---:|---:|---:|
| `configs/zoom4x-128-jit-1m.json` | 1,031,806 | 128 | 3 | 4 | 64 |
| `configs/zoom4x-128-jit-10m.json` | 9,866,080 | 256 | 8 | 8 | 128 |

These configurations use the legacy fully observed paired interface. The new mixed dataset
at `data/zoom4x-multiscale-128` requires the mask-aware training interface above.
For a compatible fully observed dataset, train either configuration:

```bash
uv run python train.py --config configs/zoom4x-128-jit-1m.json
uv run python train.py --config configs/zoom4x-128-jit-10m.json
```

These are conditional adaptations of [JiT](https://github.com/LTH14/JiT), with
native PyTorch fused attention and no extra model dependency. The loss uses
velocity-equivalent weighting of clean-image error, clamping `1-t` at 0.05;
the paired path starts from the upsampled source rather than Gaussian noise.
Configurations retain the 20,000-step pilot budget. No convergence claim is made.

## 4x zoom setups

Both models take an evolving RGB image and a fixed upsampled RGB crop, and
predict three velocity channels at 128 x 128. For input `y` and target `x`:

```text
t ~ Uniform(0, 1)
zt = (1-t)*y + t*x
loss = mean((model(zt, t, y) - (x-y))**2)
```

Inference starts at `y` and integrates from t=0 to t=1 with 16 Euler steps by
default. There is no noise initialization. Images use [-1,1] internally and
[0,1] on disk and for display. Only the final sample is clipped.

| Setup | Architecture | Parameters |
| --- | --- | ---: |
| `configs/zoom4x-128-dit.json` | Patch 4, width 256, 8 blocks, 8 heads; 1024 tokens | 9,768,240 |
| `configs/zoom4x-128-unet.json` | Widths 64/128/256/256, two residual blocks per level, attention at 32 and 16 pixels | 14,856,835 |

Both concatenate state and condition into six input channels. The U-Net uses
timestep scale/shift modulation; DiT uses adaLN-Zero. Both configurations use
CUDA BF16 autocast, batch 32, AdamW at 0.0002, EMA 0.999, gradient clipping 1.0,
and a provisional 20,000-step budget. These are starting configurations, not
trained models or parameter-matched architecture comparisons.

A short synthetic CUDA check on the RTX 4090 measured peak allocated memory
of 3.44 GiB for DiT and 7.61 GiB for U-Net at batch 32. This included AdamW,
EMA, forward/backward and BF16 autocast, but excluded the full dataset and
checkpoint loading. Four optimizer steps and finite sampled outputs verify
execution only; they do not establish learned zoom quality or sustained speed.

Dataset formats and split limitations are documented in [datasets](datasets.md).

For compatible fully observed pairs, the older setups run as follows:

```sh
uv run python train.py --config configs/zoom4x-128-dit.json
uv run python train.py --config configs/zoom4x-128-unet.json
```

To sample a zoom checkpoint, provide a `.pt` file containing `images`: already
upsampled float32 conditions `[N,3,128,128]` in [0,1]. Sampling does not crop a
parent image automatically and uses all conditions in that file.

```sh
uv run python sample.py runs/dit-zoom4x-128/checkpoint.pt \
  --input data/zoom-conditions.pt --output output/dit-zoom.png
uv run python sample.py runs/unet-zoom4x-128/checkpoint.pt \
  --input data/zoom-conditions.pt --output output/unet-zoom.png
```
