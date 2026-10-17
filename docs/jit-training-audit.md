# JiT training audit

Checked 2026-10-17 against local `model.py`, `train.py`, the old 100k config,
and `configs/zoom4x-smooth-1m-jit-1m-fast.json`. The fast config preserves the
old optimizer/objective settings: performance changes are not a claim that
these hyperparameters are optimal. Our model is a small conditional adaptation,
not the paper's JiT-B.

## Architecture

| Detail | Our mask-aware 1M model | Official JiT-B/16 |
|---|---|---|
| Parameters | 1,048,640 (old checkpoint: 1,035,902) | Approximately 131M |
| Resolution / RGB range | 128×128 / [-1,1] | 256×256 / [-1,1] |
| Patch / image tokens | 8×8 / 256 | 16×16 / 256 |
| Depth / width / heads | 3 / 128 / 4 | 12 / 768 / 12 |
| Head dimension | 32 | 64 |
| Patch input / output dimensions | 448 (7×8²) / 192 | 768 / 768 |
| Patch bottleneck | 64 | 128 |
| MLP | SwiGLU; explicit intermediate 352 (old checkpoint: 341) | SwiGLU, ratio 4; intermediate 2048 |
| Position / attention | Fixed 2D sin-cos + RoPE; Q/K RMSNorm | Same families |
| Block norm / modulation | RMSNorm eps 1e-6; adaLN-Zero-style gates | RMSNorm eps 1e-6; adaLN-Zero-style gates |
| Final norm | PyTorch RMSNorm, explicit eps 1e-6 | Explicit eps 1e-6 |
| Time embedding | 256 sinusoidal features, raw t; two-layer SiLU MLP | Same construction |
| Initialization | Xavier projections; time weights std .02; zero gates/output | Same recipe; also initializes class embeddings |
| Conditioning | Concatenate noisy RGB + evidence RGB + mask | Class embedding added to time embedding |
| Context tokens | None | 32 class tokens inserted at block index 4 |
| Dropout | 0 | 0 (larger H/G use 0.2) |
| Attention implementation | PyTorch SDPA | Explicit FP32 attention logits and softmax |

Sources: [official architecture](https://github.com/LTH14/JiT/blob/main/model_jit.py),
[normalization and position utilities](https://github.com/LTH14/JiT/blob/main/util/model_util.py),
[paper](https://arxiv.org/html/2511.13720v1).

The local `t/1000` cancels the shared helper's multiplication by 1000: JiT sees
raw t, as upstream does. This is not an accidental thousand-fold time shift.
The final-norm epsilon matches upstream. The attention numerical implementation
differs; it is not an established cause of blur. Neither was changed by the speed work. Matching block names does not make checkpoints compatible.

The fast config now explicitly sets `mlp_hidden_size=352`, overriding the ratio-derived
341. The fused gate/value projection is 704 wide. Historical configs without this
field retain the original dimensions, so their checkpoints still load. The new
config writes `runs/jit-1m-smooth-1m-patch32`; it cannot resume the 341-wide checkpoint
without an explicit migration. No migration or training restart was performed.
Resident-data checks favored 352 over both 341 and 384 by about 1.4%, not a large
throughput gain. See `docs/training-speed.md` for the matched measurements.
The active dataset is now the separate `uint16-patch32` cache: 128x128 targets,
32x32 conditioning cores, and metadata-derived coverage. Both train and validation
use clamped-edge bilinear reconstruction before the 128x128 model input. Target
quantization and changed crop-edge conditioning are explicit dataset changes,
not an exact continuation of the old run. The original float32 dataset is retained.


## Objective, optimization, and evaluation

| Detail | Ours (old → fast config where changed) | JiT reference |
|---|---|---|
| Prediction | Clean RGB x | Clean x |
| Interpolation | z=(1-t)e+t*x; fixed evidence/mask | z=(1-t)e+t*x |
| Loss | Mean squared x error divided by max(1-t,.05)² | Equivalent velocity-space MSE |
| Time distribution | sigmoid(N(-.8,.8²)) | Same |
| Noise standard deviation | 1, in training and sampling | image_size/256 |
| Optimizer | Fused AdamW | AdamW in code; Adam in paper |
| Betas / epsilon | (.9,.999) / 1e-8 | (.9,.95) / 1e-8 |
| Weight decay | .01, all parameters | 0 |
| LR / schedule | 2e-4 / constant after warmup | 2e-4 at global batch 1024 / constant |
| Warmup | 500 updates | 5 epochs |
| Global batch | 64 | 1024 (128/GPU in default code) |
| Budget | 20k → 100k updates | 200 ablation / 600 main epochs |
| Training examples | 90k → 750k selected per pass (900k stored) | ImageNet training set |
| Effective passes | 14.22 → 8.53 selected passes | See epoch budget above |
| Precision | BF16 autocast, FP32 weights/optimizer | BF16 autocast |
| Gradient clipping | Global norm 1; finite checks each update | No clipping in training loop |
| EMA | One copy, .999 | Paper sweeps .9996/.9998/.9999; code tracks two |
| Unconditional cases | 10% training selection (stored dataset: 25%) | 10% random class-label dropout |
| Other modes | 60% interior, 15% edge, 15% corner | Class-conditioned generation |
| Augmentation | None | Center crop and random horizontal flip |
| Sampling | Euler16 previews; optional Heun50 | Heun50, last step Euler |
| Time grid / division clamp | Uniform [0,1] / .05 | Same |
| Guidance | None | CFG sweep 1–4; optional interval .1–1 |
| Final clipping | Clamp displayed/exported image to [0,1] | Generator returns unbounded state; image export handles conversion |
| Evaluation | 256 fixed mode-balanced examples + 16-image gallery | FID/IS on 50k samples; tuning uses 8k |
| Shuffling | Replacement sampling → shuffled shards and rows | Shuffled distributed sampler, drop final incomplete batch |
| Checkpoint state | Model, EMA, optimizer, RNG; fast loader adds consumed-row cursor | Model, EMA, optimizer, epoch |

Sources: [JiT Table 9](https://arxiv.org/html/2511.13720v1#A1),
[training configuration](https://github.com/LTH14/JiT/blob/main/main_jit.py),
[training loop](https://github.com/LTH14/JiT/blob/main/engine_jit.py),
[objective and sampling](https://github.com/LTH14/JiT/blob/main/denoiser.py).
The code's `blr=5e-5` scales with total batch/256 unless absolute `lr` is supplied;
therefore “use JiT's LR” does not uniquely mean 2e-4 at our batch size.

## Decisions before a quality run

- Preserve the objective, patch size, and mixed conditioning interface.
- Compare betas (.9,.95) and zero decay against our current baseline. Those
  paper settings are not silently enabled by the throughput config.
- Test noise std .5 versus 1 in separate fresh runs. At 128, .5 is an
  extrapolation of the paper's resolution rule, not a reported Mandelbrot result.
  Noise strength must be implemented consistently in training and sampling;
  it is currently fixed at 1 in our conditional flow.
- EMA is measured in updates, not examples. With .999 and batch64 its half-life
  is roughly 44k examples; copying .9999 from a batch1024 run would not preserve
  the same smoothing horizon.
- 100k updates is a pilot, not convergence. Keep the 1M baseline; test 10M capacity
  under controlled budgets. The old 10M config lacks the mask channel and cannot
  be used directly with this mixed dataset.
- Validation now includes depth stratification, the upsampled-input baseline, and
  fixed Heun galleries. Pixel error alone is unsuitable for unconditional quality.
  Nearby train/validation camera footprints can overlap: this is not an unseen-region test.
- Follow [Karpathy's training recipe](https://karpathy.github.io/2019/04/25/recipe/):
  inspect actual tensors, overfit a diverse small batch, compare baselines, then
  scale. The successful 32-view memorization test is not generalization evidence.
