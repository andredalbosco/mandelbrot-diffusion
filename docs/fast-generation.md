# Fast Mandelbrot generation study — RTX 4090, 2026-10-16

**Measured result:** the separate batched backend produces approximately **5,900 main-view pairs/s**, **1,000 shallow-boundary pairs/s**, and **115–138 deep pairs/s**, including new batch construction and transfer to CPU. Deep data production improves about **4–6×** over the previously measured viewer-based path; exact gains depend on location and sampling density.

Implementation: [`mandelbrot_fast.py`](../mandelbrot_fast.py). The existing viewer and training entry points are unchanged. This is an experimental generation backend, not an automatic replacement or a claim of globally optimal performance.

## Workload

- RTX 4090, 128×128 RGB output, four samples per output-pixel area, normalized tent filter of radius one output pixel, fixed cosine palette, z0=0, 512 iterations for the headline cases.
- 32 distinct parent/child pairs per batch: 64 rendered views. A pair contains a parent image, its central 32×32 crop enlarged bilinearly to 128×128, and a separately rendered 4× zoom target.
- Centers vary across a local region. The first parent matches the scalar workload; later entries are not copies. The results therefore describe representative nearby views, not an exactly identical expanded copy of the scalar benchmark.
- Ten timed GPU-output repetitions after two warmups. The separate end-to-end measurement uses five repetitions after two warmups and includes fresh metadata, allocations and GPU→CPU transfer. No region search, rejection, iteration-convergence checks or filesystem writes.
- The desktop and viewer remained active, as in the previous benchmark. Compilation is excluded. Figures are short-run wall-clock averages; cold start and mixed-dataset throughput will differ.

## Four-sample production

| Region / selected method | Previous viewer path, pairs/s | Batched GPU-output pairs/s | Fresh CPU-output pairs/s |
|---|---:|---:|---:|
| main / hybrid32 | 1,046 | 9,241 | 5,872 |
| shallow_boundary / float64 | 370 | 1,128 | 1,010 |
| deep_boundary / perturb64 | 27 | 115 | 115 |
| saved_view / perturb32 | 23 | 145 | 138 |

The GPU-output column excludes buffer construction and CPU transfer. The CPU-output column includes them; disk serialization remains separate. In the main view those costs reduce throughput appreciably, while deep views remain dominated by numerical computation.

At the measured deep CPU-output rates, **10,000 pairs take approximately 73–87 seconds**, before search/rejection and storage. This compares with about 6–7 minutes for these two specific earlier four-sample workloads.

## Higher sampling density

64 samples per output-pixel area, batch eight pairs / sixteen views, guarded FP32 perturbation with direct-DD fallback, tent reconstruction:

| Region | Previous ms/pair | Fast ms/pair, GPU output | Fast pairs/s | 10,000-pair projection |
|---|---:|---:|---:|---:|
| deep_boundary | 440.0 | 99.8 | 10.0 | 16.6 min |
| saved_view | 541.3 | 72.2 | 13.9 | 12.0 min |

These projections exclude fresh-batch allocation, CPU transfer, disk output and dataset selection; unlike the four-sample CPU-output column, this table measures the GPU-output path only.

## What made it faster

1. **Minimal filter halo.** The viewer renders an overscan cache for navigation. Independent training renders do not need it. At four-sample density, sample dimensions fall from 384×384 to 260×260: **2.18× fewer samples**, retaining the tent filter and its necessary border.
2. **Batched views.** One launch uses indices `[batch, sample_y, sample_x]`. Each view has its own center, span and iteration limit. This amortizes launches and improves occupancy for small images; it does not multiply GPU capacity indefinitely.
3. **Fused orbit/color evaluation.** Fixed cosine RGB is written directly, removing the full renderer’s intermediate escape/styled-escape buffers and recoloring machinery.
4. **Mixed precision.** `hybrid32` uses FP32 for ordinary trajectories, estimates accumulated error and queues uncertain samples for FP64 recomputation. It helps broad views; it falls back on every sample at the tested shallow boundary, where direct FP64 is faster.
5. **Tiled deep references.** A DD reference orbit per sample tile supports local perturbations in FP32 or FP64. Uncertain samples use direct DD. Small tiles reduce perturbation distance but increase reference construction cost; 64×64 sample tiles were a useful tested compromise, not a universal optimum.
6. **Compacted fallback queue.** Uncertain pixels are collected into a GPU queue and recomputed together. This reduces partially active expensive warps. Queue generation, reference construction and fallback computation are included in reported timings; no CPU readback chooses the fallback count.

## Numerical evidence and tradeoffs

- **39 tests pass**, including independent 70-digit Decimal escape counts for deep batches, fallback behavior, distinct views, and counter ordering on a nondefault Torch stream.
- Eight spread-out entries in each selected four-sample batch were additionally checked against scalar rendering; deep references force direct DD instead of reusing the perturbation path. The selected deep modes had **zero escape-count differences** in those checks and worst per-image mean RGB error below **2.7×10⁻⁷**.
- Ordinary FP64 and hybrid paths differ at rare sensitive boundary samples: the worst checked shallow entry differed at **0.075% of samples**, with mean RGB error **1.62×10⁻⁴**. Individual pixel-channel differences can reach about 0.14. They are not byte-identical reference replacements.
- At 64 samples, the selected FP32 perturbation mode had zero count differences across the four checked entries per region. An alternate FP64 perturbation mode had one count mismatch in a saved-view entry; the error estimates are **heuristic, not certified intervals**.
- Raw `float32` reaches about **225,000 images/s** for a main-view batch of 32, but changes about **0.15%** of sample escape counts there, and **2.6%** in the medium boundary view (RGB MAE about 0.0057). This is an explicit preview experiment, not the proposed training-label backend.
- The earlier `deep_2048` timing workload is entirely unescaped at that budget and scale. Its speed is not evidence of fast production of useful detailed images, so it is omitted from the headline table.
- An initial prototype run failed numerical checks due to shared mutable Warp state initialization. Its artifact is marked INVALID and excluded from all reported rates. Later validated runs use independent state variables.

![Reference, fast rendering and amplified error](../output/benchmarks/20261016T004616Z/comparison.png)

For background on floating-point operation ordering and error-free arithmetic, see [NVIDIA’s floating-point documentation](https://docs.nvidia.com/cuda/archive/12.2.2/floating-point/index.html) and [Shewchuk’s adaptive-precision paper](https://people.eecs.berkeley.edu/~jrs/papers/robustr.pdf). Those references explain numerical principles; correctness evidence for this implementation is the tests and measured comparisons above.

## Current bottleneck and next step

The earlier DiT/U-Net trainers consume approximately 706/239 pair presentations per second. Broad and shallow generation can now exceed those rates. **Deep rendering still limits online generation**, but precomputing 10,000 four-sample pairs costs around 1–2 minutes in these workloads and is amortized across repeated epochs. At 64 samples, production takes around 12–17 minutes, comparable to the approximately 16-minute DiT training projection.

The next substantial deep-rendering speedups would require more than launch tuning: reducing DD fallback cost/frequency or introducing validated series/linear approximations that skip long orbit prefixes. Neither is implemented here. CUDA graphs could reduce small-batch host overhead, but are unlikely to change the current deep bottleneck materially.

First integrate the selected experimental modes into an offline preview generator, retain configurable sample density and numerical checks, and inspect diverse pairs. A larger benchmark covering actual selected regions is needed before predicting full-dataset throughput.

## Use and reproduce

```python
from mandelbrot import View
from mandelbrot_fast import FastRenderer

renderer = FastRenderer([View(), View(center=-0.75 + 0.1j, span=0.05)],
                        precision="float64", side=2)
images = renderer.render()  # CUDA [2,128,128,3], reused storage
```

The experimental backend supports z0=0, fixed cosine colors, a fixed input batch and at most 65,536 iterations. Direct float32/float64 reject coordinate zooms outside their precision margin. For deep batches choose `perturb32` or `perturb64`; `guard="bailout"` keeps the bailout uncertainty/cancellation checks while removing the stricter absolute-error early rejection. The default guard remains `strict`.

```sh
uv run python docs/benchmark_fast.py --pairs --cases main shallow_boundary deep_boundary saved_view --batches 32 --tiles 64 --modes hybrid32 float64 perturb32 perturb64 --guard bailout --repeats 10 --end-to-end
uv run python docs/benchmark_fast.py --pairs --cases deep_boundary saved_view --batches 8 --tiles 64 --guard bailout --side 8 --repeats 5
uv run python -m unittest discover -s tests -q
```

Artifacts:

- [Four-sample pairs, extra quality checks and CPU-output timings](../output/benchmarks/20261016T004616Z/fast.json).
- [64-sample pair timings](../output/benchmarks/20261016T004442Z/fast.json).
- [Direct/mixed-precision image sweep](../output/benchmarks/20261016T004217Z/fast.json).
- [Compaction experiment](../output/benchmarks/20261016T004054Z/fast.json).
- [Original training/production comparison](throughput.md).


## Dataset integration

`dataset.py prepare-zoom` now uses `DatasetRenderer` in `mandelbrot_fast.py`.
It measures fresh construction, rendering, and CPU transfer for FP64, hybrid32,
perturb32, and perturb64 on up to eight selected views per scale. The fastest
accepted mode is selected: worst checked per-image RGB MAE ≤0.0005 and sample
escape-count mismatch fraction ≤0.001 against same-budget FP64. These are
numerical tolerances, not proof of exact classification or sampling convergence.
Production renders batch views sharing a mode together. Each non-FP64 production
batch checks one rotating entry against FP64 and recomputes the entire group in
FP64 on failure. Records preserve the actual modes used for parent and target.

Boundary search now evaluates many independent paths per launch at each depth,
using batched 24×24 escape-count pilots and breadth-first traversal.

The new `data/zoom4x-fast-128` contains 900/100 RGB examples, seven balanced depths,
8192 iterations and four samples per pixel. Measured generation phases:

| Phase | Seconds |
|---|---:|
| Batched boundary search | 0.67 |
| Backend calibration | 24.90 |
| Production rendering, checks and CPU transfer | 5.68 |
| Matched direct-FP64 rendering and CPU transfer, optional extra baseline | 5.29 |
| Tensor and preview saving | 0.43 |
| Total, including baseline, calibration and setup | 37.63 |

Only scale 1 selected hybrid32 in this run; other scales selected FP64. All 33
production spot checks passed. This workload shows **no rendering speedup** over
direct FP64 after grouping/check overhead. At these spans the DD perturbation
backends are expensive and some fail the numerical acceptance threshold. The
earlier very-deep benchmark results do not transfer automatically to this corpus.
The previous sequential search took 10.66 seconds for 1000 examples with the same
quotas/budget; breadth-first candidate locations differ, so this is indicative
search improvement rather than an identical-camera rendering comparison.

Generation refuses to overwrite an existing dataset. Use a new output directory.
`dataset_viewer.py` exports all datasets into `dataset_viewer.html` plus local
lossless PNG sprite sheets. No server or external assets are required.

A subsequent full audit rerendered all 1000 examples with direct FP64, reprojected
all evidence, and compared masks and geometry. Worst per-image RGB MAE was
1.32e-8, maximum channel error 3.58e-7, masks matched exactly, and camera alignment
error was below 4.45e-16. Results are saved in the dataset's `quality.json` and
included in the HTML viewer. This is a same-budget backend comparison, not an
iteration/sampling convergence check.
