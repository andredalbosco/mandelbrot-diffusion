# Pipeline throughput — RTX 4090, 2026-10-16

**Conclusion:** precompute and reuse zoom pairs. Fresh deep Mandelbrot rendering is slower than either trainer, especially at higher sampling densities. DiT is about 3× faster than U-Net in the current configurations. These are execution measurements, not learning or output-quality results.

## Setup

- Local RTX 4090, 24 GiB; PyTorch 2.11.0+cu128; CUDA runtime 12.8.
- Both supplied 128×128 RGB configs, batch 32, eager BF16 autocast with FP32 weights, AdamW and EMA. No compilation or model tuning.
- Training: 5 warmup updates, 30 timed updates per architecture. All finite loss/gradient checks passed. Disposable weights were discarded.
- The GPU-resident pool contains 80 entries: five actual rendered zoom pairs repeated 16 times. This measures shape-dependent execution, not dataset diversity or convergence.
- Warmed rendering: 10 repeats after 2 warmups. Fresh image/pair timings force orbit recomputation; cached pan timings are reported separately in the JSON.
- A pair includes a 128×128 parent, centered 32×32 crop bilinearly enlarged to 128×128, and a separately rendered 4× zoom target. Fixed cosine coloring; tent reconstruction including the viewer overscan margin.
- The desktop/viewer stayed open. Initial background GPU utilization was 20% and device memory use about 3 GiB. Other GPU activity was not isolated; these short-run results are approximate.

## Training

| Model | Parameters | Mean ms/update | Pairs/s | Peak allocated GiB | 20,000 updates |
|---|---:|---:|---:|---:|---:|
| dit | 9,768,240 | 45.3 | 706 | 3.46 | 15.1 min |
| unet | 14,856,835 | 133.7 | 239 | 7.69 | 44.6 min |

Training timings include indexing, time sampling, forward/loss, finite checks, backward, gradient clipping, optimizer, EMA, and loss scalar synchronization. CSV filesystem writes and scheduled evaluation/checkpoints are excluded. Peak allocation includes the small benchmark pool, model, EMA, optimizer state and activations; it excludes other processes/driver memory. A full 10,000-pair float32 resident corpus adds approximately 3.63 GiB relative to this pool.

For an assumed 1,000-pair validation split, evaluating model + EMA took 0.87 s for DiT and 3.03 s for U-Net. With the existing 500-step evaluation interval and 1,000-step checkpoint interval, the 20,000-step projections become approximately **15.7 min DiT / 46.6 min U-Net**, plus startup, CSV writes and any long-run drift. Validation was measured separately using initialized weights and repeated main-view tensors. GPU-to-file checkpoint saves took approximately 87 / 134 ms (151 / 227 MiB), using OS buffering.

## Fresh Mandelbrot production

| Location | Iterations | 4 samples: mean ms/pair | 16 samples | 64 samples |
|---|---:|---:|---:|---:|
| main | 512 | 0.96 | 1.44 | 3.64 |
| shallow_boundary | 512 | 2.70 | 8.20 | 29.62 |
| deep_boundary | 512 | 37.15 | 119.78 | 439.96 |
| deep_2048 | 2048 | 65.36 | 231.47 | 892.53 |
| saved_view | 512 | 43.26 | 143.21 | 541.32 |

Four-sample fresh production is about **1,046 pairs/s main**, **371 pairs/s shallow**, and **15–27 pairs/s deep**; the saved view produces about **23 pairs/s**. Cached pans resolve in roughly 0.05–0.09 ms headlessly, but that is not fresh-data production. The GUI also pays presentation/transfer costs.

At the measured deep workloads, 10,000 pairs would take **6–11 min at four samples**, **20–39 min at 16**, or **73–149 min at 64**. These are uniform-workload projections, not predictions for a mixed dataset. Search, rejection, iteration convergence checks, additional high-quality reference renders, and CPU/file output are excluded.

## Model image generation

16 Euler steps, BF16 autocast; inference only, without file output. These briefly updated benchmark models do not demonstrate useful generated detail.

| Model | Single image median | Batch 32 median | Batch throughput |
|---|---:|---:|---:|
| dit | 23.0 ms | 212.0 ms | 151.5 images/s |
| unet | 48.9 ms | 762.0 ms | 42.1 images/s |

Only three timed inference repeats were used. The U-Net single-image mean was 55.8 ms versus a 48.9 ms median, indicating timing variation. Direct mathematical rendering is much faster for easy views. Neural sampling can be faster than expensive deep/high-sample renders, but matching their quality has not been established.

## Transfer and storage

- Copying an 80-pair, 30 MiB GPU tensor to CPU: 3.16 ms mean.
- Saving a CPU 30 MiB tensor: 13.32 ms mean, about 2,252 MiB/s into the OS page cache. No fsync: this is not physical-disk sustained throughput.
- One 128×128 main-view PNG: 2.24 ms mean CPU encoding/write. PNG cost depends on content. Training uses tensors, so individual PNG export is optional.

## What to do next

1. Precompute the small preview set, inspect quality, then reuse tensors for training. Rendering and training share the 4090; their independently measured rates cannot be assumed to overlap.
2. If the selected dataset emphasizes deep/high-sample views, optimize data production first: remove the viewer-sized overscan from independent dataset renders, then investigate batched rendering. Retain the necessary reconstruction-filter halo.
3. Use DiT as the faster initial experiment. Compare learned quality later; speed alone does not establish the better model.
4. Keep the full dataset cached. At batch 32 × 20,000 updates, training consumes 640,000 pair presentations: a 10,000-pair corpus is reused about 64 times, amortizing generation.

## Reproduce

```sh
uv run python docs/benchmark_pipeline.py
```

Raw results, complete configs, exact view coordinates and source hashes: [results.json](../output/benchmarks/20261016T002404Z/results.json). The script defaults to a new timestamped output directory and retains no benchmark model checkpoint.
