# Training throughput on the RTX 4090

Measured 2026-10-17 with PyTorch 2.11.0+cu128, 128×128 RGB, the mask-aware
1,035,902-parameter JiT, BF16, AdamW, gradient clipping and EMA. Other desktop
and Mandelbrot viewer processes remained running. Results are workload-specific.

## Run

The active config prepares a fresh 1,048,640-parameter JiT with SwiGLU width 352,
RGB16 targets and 32x32 conditioning patches. It uses the converted million-image
dataset, batch64, 10% unconditional sampling, compiled BF16 training, and 100,000
updates (6.4M examples, 8.53 selected-data passes). Original datasets and the paused
341-wide checkpoint remain intact. The historical measurements below describe
the earlier architecture unless explicitly marked aligned.

```sh
# Requires the completed compact dataset; checks without starting training:
uv run python scripts/preflight_training.py --config configs/zoom4x-smooth-1m-jit-1m-fast.json --report output/training-speed/training-ready.json
# Start the new run when ready:
uv run python train.py --config configs/zoom4x-smooth-1m-jit-1m-fast.json
# Resume the NEW run after it has saved a checkpoint:
uv run python train.py --config configs/zoom4x-smooth-1m-jit-1m-fast.json \
  --resume runs/jit-1m-smooth-1m-patch32/checkpoint.pt
```

Do not resume the old 341-wide checkpoint with this changed architecture/dataset.
Its original config is `runs/jit-1m-smooth-1m-fast/config.json`. The old training
process remains paused; it should not run concurrently with the new production run.

## Measurements

| Experiment | Batch | Images/s | Scope |
|---|---:|---:|---|
| Original eager model + original loader | 64 | 3,221 | 100 updates, one shard's working set |
| Compiled model + original loader | 64 | 5,725 | Same loader benchmark |
| Compiled, GPU-resident synthetic tensors | 64 | 21,804 | Compute ceiling; excludes disk/transfer |
| Compiled, GPU-resident synthetic tensors | 128 | 23,873 | Compute ceiling |
| Compiled, GPU-resident synthetic tensors | 256 | 20,930 | Compute ceiling |
| Final compiled model + GPU shard prefetch | 64 | 7,212 | 2,000 updates, 128k real examples, finite checks |
| Actual trainer, resumed 2,000-update segment | 64 | approximately 5,700 | Includes compile, validation, previews and checkpoints in loop |

The original-loader compile comparison is about 1.78× at the same batch size.
The final loader benchmark is about 2.24× the initial baseline, but it traverses
multiple shards rather than one: this is not a cache-controlled speedup estimate.
The full trainer measurement excludes startup metadata scans and initial previews.
It includes its first compiled training invocation. It is not a full-epoch average.

Raw benchmark records: `output/training-speed/*.json`. Training status and curves:
`runs/training-throughput-smoke/status.json` and `loss.csv`.
The resumed segment starts at update 2000; elapsed time restarts for that segment.
The complete 4k smoke run processed 256k examples and validation flow loss fell
from 1.435492 to 0.114669. This checks learning/execution, not convergence or quality.

## Changes

- Compile the training loss and model with CUDA graphs. Existing sampling and
  checkpoint interfaces retain ordinary module state dictionaries.
- Shuffle shard order and selected rows deterministically each pass. The current
  config sets `unconditional_fraction: 0.1`: each pass includes all 675k conditional
  rows and a seeded, rotating subset of 75k unconditional rows. The stored 900k
  training rows remain unchanged. Cross-boundary batches are joined. This is
  shard-local shuffling, not a global random permutation.
  The throughput measurements below predate this mixture change; it still reads
  the complete source shards, so I/O per selected example increases.
- Keep the current and prefetched float32 shard on CUDA; gather minibatches
  there. Read the next shard in a background thread, stage it in reusable pinned
  buffers, and transfer/normalize on a separate CUDA stream.
- Validate finite [0,1] source values once per loaded shard per process, using
  GPU reductions. Keep per-update finite loss/gradient checks and clipping.
- Cache the fixed validation/gallery tensors on CUDA. Write CSV every50 updates
  in the fast config; retain evaluation/checkpoint cadence of500 updates.
- Save the consumed sampler cursor in checkpoints, separately from prefetch.
  Record images/s, peak allocated CUDA memory and shard-wait time.

Measured peak allocated CUDA memory for the final loader benchmark was 7.76 GiB
(excludes other processes and reserved memory). Two pinned staging buffers use
about 7.7 GiB of host RAM for the existing 9000-row shards, in addition to file
cache and metadata. Larger shards or a much larger model need a fresh memory
budget; this path is tuned to the current dataset/4090.

Shard copies from the OS cache took roughly .22–.35 s; later disk-backed copies
were roughly .77–.92 s, plus about .30 s for CUDA transfer/validation/normalization.
Cache state was not forcibly cleared. This explains why the 22k compute ceiling
is not sustained on the approximately460 GB dataset. No source quantization,
resizing, target modifications or sample omission was used.

The manifest digest identifies the sharded dataset and its recorded file hashes.
Startup does not rehash every file; modifying shard contents without updating
the manifest is not cryptographically detected by resume. Contents still undergo
shape/range validation. Treat generated shards as immutable.

Compilation can change floating-point rounding and random-number sequences versus
eager execution; this is not a bitwise reproduction of an old eager run. Sampler
state restores exact row order. Optimizer, LR, EMA and loss settings remain those
of the old config, and are explicit in the new config. See the
[JiT comparison](jit-training-audit.md) for deliberate differences and pending
quality experiments.

## Reproduce

```sh
uv run python scripts/benchmark_training.py --real-data \
  --output output/training-speed/baseline-repeat.json
uv run python scripts/benchmark_training.py --compile --real-data \
  --output output/training-speed/compile-repeat.json
uv run python scripts/benchmark_training.py --compile --compile-mode reduce-overhead \
  --sharded --checked --steps 2000 --output output/training-speed/stream-repeat.json
uv run python -m unittest tests.test_training_stream
```

The benchmark times AdamW, backward, clipping and EMA after15 warmup steps;
`--checked` includes the production finite checks. Without `--sharded` or
`--real-data`, it is explicitly a synthetic GPU-resident benchmark.


## Kernel-library experiments (2026-10-17)

The main baseline was left paused. Isolated batch-64 runs used GPU-resident
synthetic 128x128 tensors, the same 1M architecture, BF16, fused AdamW,
gradient clipping and EMA. Timings include forward/backward/optimizer; exclude
compilation, data loading, validation and checkpointing. Desktop/viewer processes
remained active. These are speed experiments, not learning-quality experiments.

Profiler traces confirmed the default compiled path uses PyTorch FlashAttention
forward/backward. An initial backend sweep incorrectly reused cached compiled
FlashAttention graphs even under different `sdpa_kernel` contexts. Discard
`research-cudnn.json`, `research-efficient.json` and their first profile timings
as backend comparisons. The `*-fresh.json` runs disable compiler caches and
verify actual cuDNN SDPA and CUTLASS memory-efficient attention kernels.
The benchmark now disables compiler caches for explicit backend selection.

Liger 0.8.4 was installed with no dependencies into
`output/training-speed/research-deps`, leaving the training environment unchanged.
Only block RMSNorm plus scale/shift was replaced; Q/K normalization, RoPE, final
normalization and SwiGLU stayed unchanged. Nonzero-gate BF16 parity checks passed:
maximum output absolute difference 0.001953125, maximum parameter-gradient
absolute difference 3.8147e-6. These checks used two examples and tolerances,
not a convergence test. In-place backward was disabled for residual safety.
The compiled Liger run emitted Triton analysis warnings but completed; its result
is specific to this integration/version, not a claim about all Liger kernels.

Reproduce (one GPU process at a time):

```sh
.venv/bin/python scripts/benchmark_training.py --compile \
  --compile-mode reduce-overhead --checked --profile --steps 1000 \
  --output output/training-speed/research-profile-auto.json
# Compare --compile-mode max-autotune or --attention cudnn / efficient.
PYTHONPATH=output/training-speed/research-deps .venv/bin/python \
  scripts/benchmark_training.py --compile --compile-mode reduce-overhead \
  --checked --liger-modulation --steps 1000 \
  --output output/training-speed/research-liger.json
```

`--checked` preserves per-step loss/gradient finite checks. Omitting it is only a
synchronization diagnostic, not an equivalent safety policy. CUDA-graph private
allocations may not be reflected fully in the reset peak-memory field; do not use
these benchmark memory numbers to budget training VRAM.

| Variant | Images/s | Timed runs |
|---|---:|---:|
| Compiled PyTorch, checked | 18,824 | 2 |
| max-autotune, checked | 19,456 | 2 |
| cuDNN SDPA, checked | 17,917 | 1 |
| Memory-efficient SDPA, checked | 18,153 | 1 |
| Liger modulated RMSNorm, checked | 16,480 | 1 |
| Compiled PyTorch, unchecked diagnostic | 23,624 | 1 |

Compiler comparison uses two 2,000-update repeats per mode. Other variants use
one 1,000-update run each. Autotuning improves this compute-only benchmark by
approximately 3.4%; no production configuration was changed.


## Loader and synchronization follow-up

The trainer now checks finite loss and gradient norm together after backward and
clipping, before optimizer/EMA updates. This removes one host synchronization per
step. A nonfinite loss may now enter backward before rejection; neither optimizer
nor EMA updates run when either check fails. No checks were removed.

`ShardBatches(select_rows=True)` optionally gathers only the mixture-selected rows
into pinned staging, in ascending file order, then restores the original random
order on the GPU. Values remain float32 and sample order/resume state are unchanged.
Every selected subset is validated because unconditional selections change between
passes. Unselected rows are not validated in this mode. Enable in the trainer with
`"select_rows": true`; it is off by default and the fast config was not changed.

| 1,000-update experiment, batch 64 | Images/s | Peak allocated GiB |
|---|---:|---:|
| Resident data, separate checks | 19,063 | — |
| Resident data, combined check | 20,176 | — |
| Shards, separate checks, first pass | 7,241 | 7.76 |
| Shards, selected rows + combined check, first pass | 8,247 | 6.48 |
| Shards, separate checks, warm repeat | 9,754 | 7.76 |
| Shards, selected rows + combined check, warm repeat | 10,240 | 6.48 |
| Shards, full rows + combined check, warm repeat | 10,403 | 7.76 |

All shard runs used the same seed, 10% unconditional sampling, and finished with
identical reported loss (0.1312330067). Exact-value CPU tests cover selected/full
row equivalence across epoch boundaries and checkpoint resume. Cache state was
not forcibly controlled; the first-pass difference is not a causal speedup
estimate. The warm runs support a modest synchronization gain, while row selection
is useful for memory rather than a demonstrated speed improvement. Keep the full
row loader as default. Raw artifacts are `output/training-speed/loader-*.json`
and `compute-{combined,separate}.json`.

The main baseline remained paused. These source changes apply on the next process
start, not by sending SIGCONT to the already-loaded trainer. Original dataset
files, checkpoints, numeric precision and training recipe were preserved.


## Pipeline references and next experiments

Research checked 2026-10-17. Local PufferLib reports 4.0.0; inspected read-only.
Its `src/vecenv.h` allocates pinned observation/action buffers, assigns environment
observation pointers into those buffers, and uses per-buffer asynchronous CUDA
copies. There are still stream synchronizations where CPU environments require
GPU actions. This is not disk-to-GPU zero-copy. `src/pufferlib.cu` captures forward,
loss, backward and optimizer update in the training graph.

- [PufferLib 4.0 engineering](https://puffer.ai/blog/engineering-4.0/): persistent,
  contiguous allocations, fused operations, per-buffer streams, broad graph capture.
  Borrow the buffer ownership and graph scope; its RL throughput is not comparable
  to image diffusion and does not establish a speedup for us.
- [modded-nanoGPT pinned slots](https://github.com/KellerJordan/modded-nanogpt/blob/master/track_1_short/perf/pinned_batches.py):
  a reusable pinned ring guarded by upload-completion events. The inspected
  implementation allocates fresh device tensors; it is not a fully static device
  ring. [Training loop](https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt.py)
  also separates graph execution and optimizer scheduling.
- [FFCV tuning](https://docs.ffcv.io/parameter_tuning.html): planned caching and
  quasi-random access for datasets larger than RAM. File layout and traversal
  matter as much as transfer calls. Our shuffled-shard traversal already captures
  part of this idea. Adoption requires preserving paired float tensors and masks.
- [DALI tuning](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/advanced_topics_performance_tuning.html):
  preallocated processing buffers and adjustable queue depth hide variable latency,
  with an explicit memory cost. JPEG decoding is not our current bottleneck.
- [llm.c loader](https://github.com/karpathy/llm.c/blob/master/llmc/dataloader.h):
  compact uint16 token storage and simple reusable buffers. Token encoding is
  lossless; quantizing our floating-point RGB is a separate numerical decision.

Recommended experiments, not yet implemented:

1. Split file read-ahead, host staging and GPU upload into independently scheduled
   stages. Test a bounded ring of smaller blocks against current whole-shard
   prefetch, with CUDA completion events protecting buffer reuse. Preserve sampling
   order and consumed-row checkpoint state. Avoid replacing sequential reads with
   arbitrary per-image disk reads. Deeper queues hide jitter, not insufficient
   sustained bandwidth.
2. Prototype compact storage on a subset. Current rows contain 448 KiB of expanded
   float32 RGB evidence, target and mask. Test reversible compression first; separately
   evaluate uint16 fixed-point RGB (rounding error <= 1/(2*65535) in [0,1]). This is
   quantization, not equivalent to the current dataset. Masks can potentially be
   reconstructed from crop coordinates with parity validation. For future generation,
   preserve the small parent crop plus interpolation halo and reconstruct enlarged
   evidence on GPU; those parent pixels are not retained in current files.
3. Benchmark broader graph capture including optimizer/EMA after data throughput
   improves. Retain rejection of nonfinite updates: blindly capturing the current
   Python finite-check branch is not valid. Do not change optimizer/objective just
   to reproduce another project's speed number.

Current readiness: CPU preflight checked every shard's metadata and the 16k
checkpoint. An isolated resume to 16,001 passed, advanced sampler by 64 rows,
preserved the original validation baseline, and left the main checkpoint unchanged.
Report: `output/training-speed/preflight.json`; smoke: `runs/preflight-resume-16001`.
Main run remains paused pending an actual training start.


## Staged transfers

`staging_rows=256` uses two fixed-size pinned host slots. A worker reads a
sequential block into one slot and queues its GPU copy while filling the other.
CUDA events prevent overwriting a slot before its previous transfer finishes.
The full GPU shards remain allocated for random-access minibatches; this change
reduces host staging and overlaps CPU copying with upload inside each shard.
It does not independently prefetch multiple disk shards or eliminate disk reads.

At 128x128 float32 RGB/evidence/mask, host staging is 224 MiB instead of 7.69 GiB
for two 9,000-row shards. Dataset values, shuffled row order, normalization,
conditional mixture and consumed-row checkpoint state are preserved. Validation
still checks finite [0,1] values before a shard is used. Selective-row staging
remains optional and disabled; the new path also supports it.

Seven targeted tests pass, including CUDA event-protected slot reuse, short final
blocks, shard/epoch crossings, selected-row compatibility, exact resume, invalid
values, and compiled loss/gradients. Baseline and staged 1,000-update runs ended
with the identical reported loss 0.13123300671577454. Timing excludes startup,
evaluation and checkpointing; desktop processes remained active.

Use `--staging-rows 256` in `scripts/benchmark_training.py`. Zero keeps the old
whole-shard staging path. The fast training config retains zero/default staging;
resume permits this performance-only change because sampler state is unchanged.


Longer checks used 4,000 updates / 256,000 examples and traversed a working set
larger than system RAM (~140 GB of shard files, including prefetch). Results:

| Loader | Images/s | Pinned host staging |
|---|---:|---:|
| Whole-shard baseline | 6,190 | 7.69 GiB |
| 256-row staging | 5,875 | 224 MiB |
| 1,024-row staging | 6,216 | 896 MiB |
| Sequential-per-field prototype (discarded) | 3,771 | 224 MiB |

Short 1,000-update runs favored staged buffers (11.7–12.0k vs 9.6–10.6k images/s),
but that gain did not carry over to the larger working set. Cache state was not
forcibly controlled; each long variant had one run. The 0.4% difference at 1,024
rows is not evidence of a speedup. Retain whole-shard staging as the training
default and expose staged buffers only as a lower-host-memory option. The final
implementation uses blocks spanning all fields; the slower per-field experiment
was reverted. All long runs ended with identical reported loss 0.1034384668.

A separate real resume from step 16,000 to 16,001 using staged buffers produced
bitwise-identical model and EMA state dictionaries and identical sampler state
compared with the original loader. See `staged-resume-parity.json` and
`runs/preflight-staged-16001`. Main training remains paused. No dataset conversion,
precision change, or alteration of the conditioning mixture was made.


## Disk-backed compact-cache experiment (2026-10-17)

The sustained bottleneck remains feeding the model: about 6.2k images/s across
large float32 datasets versus about 21k with resident tensors. Small optimizer
improvements do not address that gap.

`scripts/prepare_compact_cache.py` builds a separate uint16-unorm cache, retaining
all original float32 files. This is explicitly lossy fixed-point encoding, decoded
on the GPU before the existing float32 normalization. Inputs, targets and fractional
masks use 16 bits each. Eight shards contain 72,000 physical training rows; sampling
at 10% unconditional selects 60,000 rows per pass. File size fell from 33.04 GB to
16.53 GB. Maximum measured error in [0,1] was 7.689e-6. Validation still reads originals.

The comparison used identical sample order, batch 64, 750 measured updates plus
15 warmup updates, compiled loss, clipping, combined finite checks, fused AdamW
and EMA. Before each process, POSIX_FADV_DONTNEED was requested for every train
file in that variant. This is advisory, not a guaranteed cold cache; measured Linux
/proc/self/io read_bytes below confirms substantial physical reads. Prefetch can
cross timing boundaries. Timings exclude setup, evaluation and checkpoints.
Desktop workloads and the paused training process remained present.

| Run | Images/s | Disk GB during timing | Disk GB including setup |
|---|---:|---:|---:|
| float | 6,674 | 21.83 | 30.13 |
| float-confirm | 6,530 | 21.68 | 29.94 |
| compact | 9,870 | 11.35 | 15.55 |
| compact-repeat | 9,956 | 11.14 | 15.28 |

Results and losses: `output/training-speed/compact-io-summary.json`; full records
are `compact-io-*.json`. The float-confirm run replaces float-repeat in the table
because a test process may have overlapped float-repeat. No tests ran alongside
the four reported measurements. These are disk-backed subset experiments, not
full-epoch training results or proof of unchanged learned quality.

Eight targeted tests pass, including compact decoding on CPU/CUDA, fractional
masks, sparse fetches, selected-row staging, shard boundaries, and exact sampler
resume relative to the decoded fixed-point values. Original float32 values are
not bitwise preserved. Staging buffers also check dtype before reuse.

Keep the production recipe and paused checkpoint unchanged. The next useful
validation is the compact format across the full working set and a matched
training-quality comparison. Do not infer that further model-kernel tuning will
remove the remaining data-feed gap.


## Conditioning representation analysis (2026-10-17)

A better first representation change than parent sharing is storing only the
34x34 RGB parent support patch for each 32x32 crop. The existing 4x bilinear
sampling maps output centers to parent coordinates left-0.375 through left+31.375,
so source indices left-1 through left+32 are needed. Crop coordinates determine
coverage; unconditional examples use neutral evidence and zero coverage.
A 16-bit RGB target plus 16-bit RGB patch is 102.773 KiB, versus 224 KiB for the
current seven-channel compact example. Metadata is excluded. This does not alter
model input size: evidence is reconstructed to 128x128 before the forward pass.

One audited shard had 9,000 distinct parent cameras for 9,000 rows. Parent sharing
would require a different sampling scheme. Full parents are not retained in the
current dataset. However, the small patch can be recovered offline from existing
evidence: multiply evidence by coverage to undo normalization, then invert the
separable bilinear map with a pseudoinverse. Its 128x34 matrix has condition number
6.269. This recovery supports the original crop only, not arbitrary new crops.

Measured on all 9,000 rows of shard zero: quantized-patch reconstruction on CPU
had maximum error 7.662e-6; GPU reconstruction with metadata-derived coverage
had maximum evidence error 8.256e-6 and mask error 3.338e-6. All four modes were
included. See conditioning-patch-shard-audit.json and conditioning-patch-gpu-audit.json
under output/training-speed. This is a numerical experiment, not a production loader.

GPU-only patch reconstruction measured 0.0618 ms per batch of 64. A separate
interior-only pinned-memory transfer/decode microbenchmark measured 0.945 ms
for full fields versus 0.476 ms for target plus patch. It excludes disk access,
CPU gathering, boundary-mask construction, model work and overlap with training;
it is not a prediction of end-to-end throughput. Main training remains paused.

On 256 randomly selected targets from the same shard, zlib level 1 compressed
16-bit RGB by 1.53x, or 1.57x after separating byte planes. These are size probes,
not GPU-decompression measurements. nvCOMP offers GPU codecs and Python APIs:
https://docs.nvidia.com/cuda/nvcomp/ . Its Blackwell dedicated decompression-engine
figures do not describe the RTX 4090. Target compression is a second experiment,
after small-patch conditioning; targets then account for about 93% of raw bytes.

Escape fields remain useful for recoloring but four 16-bit subpixel escape values
already cost 8 bytes/pixel versus 6 for RGB16. Color each sample before filtering.
Reference-orbit reconstruction has a different purpose: compact descriptions of
new views, with variable iteration cost competing against training. It is not the
first intervention for this static dataset.


## Conditioning borders and aligned SwiGLU (2026-10-17)

The user's preferred compact-conditioning format is 32x32. A 32x32 core plus
one-pixel filter halo is 34x34; the latter was an exact-filter-support experiment,
not a JiT architectural requirement. A plain 32x32 format can instead clamp its
crop edges before upsampling. It then changes boundary evidence relative to the
old full-parent reprojection; train and evaluation must use the same convention.
For partial views, observation coverage must still respect the physical parent
boundary. At the time of that analysis the production dataset/loader had not yet been
converted to 32x32 conditioning; see the subsequent preparation results below. The storage difference versus 34x34 is only 792 bytes per
RGB16 example, about 0.75% of target-plus-patch storage. Neither format changes
our model's 128x128 image size, 8x8 token patches, or 256 image tokens.

References:
- [JiT paper](https://arxiv.org/abs/2511.13720): uses 16x16 tokens on 256 images
  and 32x32 tokens on 512 images. These are non-overlapping network patches,
  not low-resolution conditioning tiles.
- [Official JiT model](https://github.com/LTH14/JiT/blob/main/model_jit.py):
  kernel_size equals stride in patch embedding; SwiGLU uses int(2*hidden/3).
  JiT-B's 768-wide backbone gives intermediate 2048 without extra rounding.
- [SR3, section 2.4](https://ar5iv.labs.arxiv.org/html/2104.07636): upsample
  the low-resolution image and concatenate it with the noisy target; uses
  bicubic interpolation. Supports the conditioning design, not a claim about
  exact tile borders or a 32-versus-34 speed comparison.
- [NVIDIA NPPDx area operations](https://nvidia.github.io/NPPDx/api/processing_steps_area.html):
  resize and neighborhood filters use halos around their core tiles.
- [GPU Gems 2, chapter 21](https://developer.nvidia.com/gpugems/gpugems2/part-iii-high-quality-rendering/chapter-21-high-quality-antialiased-rasterization):
  filter support determines tile padding.

Aligned-width benchmark: compiled BF16 resident-data training, batch64,
1,000 measured updates after warmup, clipping/finite checks/fused AdamW/EMA.
Two runs per width, second sweep in reverse order; no concurrent test processes.
These measurements exclude disk loading, evaluation and checkpointing. They
compare different model capacities/initializations, not training quality.

| SwiGLU width | Parameters | First images/s | Repeat images/s |
|---|---:|---:|---:|
| 341 | 1,035,902 | 23,575 | 23,564 |
| 352 | 1,048,640 | 23,888 | 23,886 |
| 384 | 1,085,696 | 23,536 | 23,480 |

Select 352 (a multiple of 32), producing a 704-wide gate/value projection.
384 was tested, not selected: its extra work did not improve throughput.
The active fast config explicitly records this width; legacy configs keep
ratio-derived dimensions for checkpoint compatibility. New runs use a separate
output directory. The arithmetic-estimate and benchmark scripts honor the new
setting. Model tests cover both aligned widths and legacy state-dict loading;
the real paused step-16000 model checkpoint also loaded strictly on CPU.
Main training remains paused; no claim of an end-to-end dataset speedup is made.


## Final training preparation (2026-10-17)

The million-image patch32 cache completed: 900,000 training / 100,000 validation
examples in 100 shard pairs, 104.609 GB including metadata. All source pixels were
checked for finite [0,1] values during conversion; each recovered-conditioning
fit passed the 1e-5 residual threshold (observed maximum 1.326e-7). Both splits
use the same crop-clamping and coverage-normalization convention. Originals remain.

The active recipe is the 1,048,640-parameter model with explicit SwiGLU width 352,
batch64, 100,000 updates, 500-step warmup, BF16 compiled loss, AdamW, EMA .999,
clipping1 and 10% unconditional selection. Full preflight reports 750,000 selected
training rows per pass and 256 fixed validation examples. The source fingerprints,
manifest/config digests, launch command and results are in
`output/training-speed/training-ready.json`.

Sixteen targeted model, CUDA loader, resume and evaluation tests passed. An isolated
64-update run saved and resumed to 65; model, EMA, optimizer, CUDA RNG and sampler
matched an uninterrupted 65-update run exactly. A further isolated check reached
600 updates, crossing the full learning-rate warmup without nonfinite values;
validation loss decreased from 1.498358 to 0.156607. This establishes learning and
startup health, not trained generation quality or full-run convergence.

The final 4,000-update / 256,000-example benchmark reached 13,675.5 images/s in
18.72 seconds, with 31.38 GB of measured physical disk reads and 4.56 GiB peak
PyTorch GPU allocation. Each train file received fsync and advisory DONTNEED before
the process started. These are actual data-loading plus optimizer-update timings;
setup, validation, previews and checkpointing are excluded. This is one final
measurement, not a full-epoch average. Details: `patch32-final-throughput.json`.

Production training was not started. Its output is `runs/jit-1m-smooth-1m-patch32`.
The old baseline process remains paused and its checkpoint is preserved. The
600-step check is under `runs/preflight-patch32-warmup`, separate from production.
