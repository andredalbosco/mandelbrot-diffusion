# JiT timing on RTX 4090

## Implemented 1M and 10M variants

The production `model.py` now includes two conditional JiT configurations:

| Config | Parameters | Compiled batch-1 forward | Eager training pairs/s, batch 32 |
|---|---:|---:|---:|
| `configs/zoom4x-128-jit-1m.json` | 1,031,806 | 0.199 ms | 4,795 |
| `configs/zoom4x-128-jit-10m.json` | 9,866,080 | 0.467 ms | 1,708 |

These use patch 8, native PyTorch RMSNorm and fused SDPA. The small variant
uses width 128 / depth 3 / 4 heads / bottleneck 64; the larger uses width
256 / depth 8 / 8 heads / bottleneck 128. The timings below refer to the
earlier imported-upstream experiment; they are not substituted for these new
measurements. Raw native-implementation results are in
`output/benchmarks/20261016-jit/production-variants.json`.

Validation: 40 repository tests passed, including a clean-prediction oracle for
paired-flow sampling at 1/4/16 steps and a JiT conditional learning check.
Both production configurations completed a two-update CUDA smoke, resume to
update 3, EMA checkpoint loading and four-step sampling. These verify the
pipeline, not learning quality. No full training experiment has been launched.

The deterministic paired path and velocity-weighted clean prediction remain
experimental. Intermediate training states contain target information, and the
`1/(1-t)^2` loss weighting favors near-clean states. Evaluate actual sampling
from the upsampled source, including held-out locations and repeated zooms;
training MSE alone cannot establish success. A one-pass residual predictor and
a flow starting from a noisy upsampled source are useful future comparisons.

## Earlier upstream-backbone comparison

Measured October 16, 2026, on the local RTX 4090, at 128×128 RGB.
Small conditional JiT is fast enough to investigate: about **0.42–0.45 ms per
compiled forward pass**, or **6.3–6.8 ms for a 16-evaluation image** at batch 1.
These are execution measurements, not evidence of successful zoom learning.

## Single-image inference

Wall-clock medians, CUDA synchronized, GPU-resident inputs and outputs.
Compilation startup is excluded. Every model uses BF16 autocast with FP32
parameters; compiled runs use `torch.compile(mode="reduce-overhead", fullgraph=True)`.

| Model | Trainable parameters | Patch tokens | Eager forward | Compiled forward | Compiled 16-step image |
|---|---:|---:|---:|---:|---:|
| Existing DiT, patch 4 | 9.77M | 1,024 | 1.482 ms | 0.559 ms | 9.559 ms |
| DiT, patch 16 control | 10.32M | 64 | 1.473 ms | 0.447 ms | 6.873 ms |
| Existing U-Net | 14.86M | — | 3.067 ms | 1.138 ms | 18.613 ms |
| Small JiT, patch 8 | 9.87M | 256 | 3.472 ms | 0.445 ms | 6.821 ms |
| Small JiT, patch 16 | 10.16M | 64 | 3.317 ms | 0.419 ms | 6.324 ms |
| Conditional JiT-B, patch 16 | 130.43M | 64 | 5.199 ms | 1.352 ms | 22.681 ms |

Small JiTs use width 256, depth 8, 8 heads, bottleneck width 128, and MLP
ratio 4. JiT-B uses width 768, depth 12 and 12 heads. The DiT controls use
width 256, depth 8, 8 heads and MLP ratio 4.

The small patch-16 JiT is 2.7× faster per compiled forward than the compiled
U-Net, and its 16-step loop is 2.9× faster. Against the compiled patch-4 DiT,
the corresponding speedups are 1.3× and 1.5×. The patch-16 DiT control is
almost as fast: these results do not establish a large JiT-specific speedup.
JiT's clean-image prediction and large patches are also learning choices that
need a separate quality comparison.

## Training and batched inference

Training is **eager**, batch 32, including forward/backward, AdamW, gradient
clipping, EMA, and scalar synchronization. Throughput uses mean step time.
Peak allocated memory includes model, optimizer, EMA, activations and the
small input pool, but excludes other processes and allocator-reserved memory.

| Model | Training pairs/s | Training peak allocated | Compiled batch-32 forward, total | Compiled batch-32 16-step loop, total |
|---|---:|---:|---:|---:|
| DiT patch 4 | 724 | 3.45 GiB | 7.752 ms | 123.884 ms |
| DiT patch 16 | 3,838 | 0.45 GiB | 0.791 ms | 13.655 ms |
| U-Net | 240 | 7.60 GiB | 17.293 ms | 272.139 ms |
| Small JiT patch 8 | 1,434 | 1.57 GiB | 1.719 ms | 30.791 ms |
| Small JiT patch 16 | 1,683 | 0.58 GiB | 0.894 ms | 15.561 ms |
| Conditional JiT-B patch 16 | 671 | 3.78 GiB | 4.386 ms | 74.340 ms |

The script repeats one rendered main-view/centered-4×-zoom pair on the GPU.
Timing weights receive 5 warmup updates and 30 measured updates, then are
discarded. This is not a representative training dataset or a convergence
experiment. Training losses differ: existing models predict velocity, whereas
JiT predicts the clean target with velocity-equivalent weighting, clamped at
`1-t=0.05`. Inference uses 16 Euler evaluations without classifier-free guidance.

## What was adapted

The source is the authors' MIT-licensed [PyTorch JiT implementation](https://github.com/LTH14/JiT),
pinned to revision `cbc743a2ada5e9762697da2c83f8c4f8379e8c17`. Unmodified
source files and license are retained under `output/benchmarks/jit-source/`.
The benchmark imports those blocks and makes these explicit adaptations:

- Concatenate RGB state and RGB image condition: six channels in, three out.
- Remove class embeddings and class-context tokens, retaining time conditioning.
- Preserve JiT's bottleneck patch embedding, RoPE, RMSNorm, SwiGLU, and adaptive modulation.
- Replace explicit attention matrix multiplication with PyTorch SDPA.
- Remove upstream block-level compile decorators to measure eager execution and whole-model compilation separately.

This is a conditional JiT adaptation, not an unchanged pretrained JiT checkpoint.
No pretrained weights were downloaded. The published class-conditional ImageNet
quality results therefore do not apply to these timing models.

## Validation and limits

All six models completed finite loss/gradient checks. Compiled and eager
predictions were compared after the timing updates, with tolerances `atol=0.03`,
`rtol=0.03`; maximum observed absolute difference was 0.0078125. A separate
BF16 attention check against the upstream implementation passed at `atol=0.02`,
`rtol=0.02`, with maximum difference 0.0078125.

Forward measurements use 10 warmups and 50 timed repetitions; 16-step loops
use 2 warmups and 5 timed repetitions. Each shape/mode is warmed independently.
First compiled calls took approximately 3–15 seconds in this environment;
this is not a cold-cache compilation benchmark. GPU desktop/viewer/streaming
processes stayed running, with roughly 3 GiB occupied and 19% GPU activity
before the experiment. Short-run variance is visible in the raw means/p90s.
These are workstation timings, not isolated hardware limits. File saving,
rendering, display, CPU transfers and training compilation are excluded.

For the next quality experiment, both small JiTs are worth retaining. Patch 8
has four times as many spatial tokens as patch 16 for nearly the same batch-1
latency; whether this preserves Mandelbrot detail better remains untested.

## Reproduce

From the repository root:

```bash
uv pip install --python .venv/bin/python einops==0.8.2
.venv/bin/python docs/benchmark_jit.py
```

The script downloads its pinned upstream sources if absent. Existing production
model/training configurations are unchanged. Raw baseline and JiT runs are
retained separately; their merged record is
`output/benchmarks/20261016-jit/combined.json`.
