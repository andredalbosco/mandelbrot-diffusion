# Datasets

Run commands from the repository root. Existing datasets are preserved.

## Compact training cache

`data/zoom4x-smooth-blue-gold-1m-128-patch32` is derived from the original million
views. Both train and validation use uint16-unorm 128x128 RGB targets and 32x32 RGB
conditioning patches. Coverage is reconstructed from the camera metadata; masks
are not stored. GPU bilinear upsampling uses `align_corners=False` and clamped crop
edges. RGB is normalized by interpolated coverage; absent evidence is neutral.
This changes conditioning at crop borders compared with the original full-parent
sampling convention, but leaves target geometry, split and record order intact.

The converter recovers the original bilinear support offline and checks its
reprojection residual before retaining the 32x32 core. Target quantization has
maximum error approximately 7.7e-6 in [0,1]. Originals are never overwritten.

```sh
uv run python scripts/prepare_training_cache.py \
  --source data/zoom4x-smooth-blue-gold-1m-128 \
  --output data/zoom4x-smooth-blue-gold-1m-128-patch32
```

Conversion writes an incomplete manifest until every train/validation shard has
finished. Repeating the command resumes at the last committed shard. SHA256 hashes
are recorded for converted files. Existing resumed cache files are checked for
presence/size, not rehashed; the final preparation audit may rehash independently.
The loader refuses incomplete datasets. Approximate tensor storage is 102 KiB per
example versus 448 KiB in the original float32 representation, excluding metadata.

## Generate RGB zoom examples

```sh
uv run python dataset.py prepare-zoom --output data/new-deep-128 --max-depth 12
# Smooth blue/gold, 100k distinct camera views (90k train / 10k validation):
uv run python dataset.py prepare-zoom --output data/zoom4x-smooth-blue-gold-100k-128 --train-count 90000 --val-count 10000 --max-depth 12 --seed 20261017 --palette smooth --batch-size 128
# Optional additional matched FP64 benchmark pass:
uv run python dataset.py prepare-zoom --output data/new-measured-128 --benchmark-baseline
```

The generator batches boundary-guided search and calibrates numerical backends
by scale. Production checks can fall back to FP64. Default output is 128×128 RGB,
four samples/pixel, 8,192 iterations, 900 training and 100 validation examples.
Initial anchors receive continuous subcell offsets to avoid exhausting a finite
set of shallow views. Depth 0 is the main view; depth N has span 3.5/4**N. Conditional refinement remains 4×.

`train.pt` and `val.pt` contain `inputs`, `targets`, `images` (target alias),
`masks`, and per-example `records`. Images are float tensors in [0,1]. Evidence
modes include interior, edge, corner, and unconditional. Masks indicate coarse
observation coverage; white is not an exact target pixel to preserve.

`manifest.json` records the configuration, source hashes, sampling policy, render
modes, numerical checks, and timings. Calibration contributes to generation time.
Where present, `quality.json` records additional reference comparisons. Finite
sampling and iteration budgets do not guarantee convergence.

Initial boundary-anchor groups differ across splits, but broad views and crop
footprints may overlap. These pilot splits do not establish geographical
generalization. The legacy paired trainer rejects partial or absent evidence;
the conditional JiT trainer supports all evidence modes. See [models](models.md).

## Available RGB pilots

| Dataset | Examples | Absolute depths | Notes |
|---|---:|---|---|
| `zoom4x-10k-128` | 9,000/1,000 | 13: 1×–16,777,216× | Boundary-guided sampling, fixed FP64 rendering; browser default |
| `zoom4x-deep-128` | 900/100 | 13: 1×–16,777,216× | Extended depth, calibrated rendering |
| `zoom4x-fast-128` | 900/100 | 7: 1×–4,096× | Batched search and backend selection |
| `zoom4x-multiscale-128` | 900/100 | 7: 1×–4,096× | Direct FP64 predecessor |
| `zoom4x-128` | 900/100 | Continuous spans | Earlier pilot with skewed train/validation scales |
| `zoom4x-main-single` | 1/1 | Main view | Identical train/validation pair; memorization only |

Earlier grayscale datasets and their generation/export commands are documented
in [experiments](experiments.md).

## Browser viewer

```sh
uv run python dataset_viewer.py
xdg-open dataset_viewer.html
```

The HTML browser includes available datasets and splits, depth/mode filters, a
target gallery, aligned input/target/mask panels, camera records, and measurements.
It works offline; keep the generated `data/dataset-viewer/` sprite sheets at their
repository-relative location. Rebuild after adding datasets. Previews are 8-bit;
training tensors retain floating-point precision.

Sobel and overlay modes currently score **RGB luminance**, after one 3×3 binomial
blur at 128×128. Gain affects display only. Highest/lowest-score ordering changes
browsing, not sampling. Palette boundaries and aliasing can affect these scores.
The OpenGL viewer's G mode instead uses the smooth escape-time field.

The 10k dataset also has Mandelbrot-specific **Coloring scheme** previews:
smooth log escape-time in blue/white/gold, histogram equalization, post-escape
orbit stripes, and a cross-shaped orbit trap. **Compare schemes** shows the same
target in every style. Both the input and target are recolored, with a shared
histogram mapping per pair; masks and missing neutral-gray evidence are retained.
Colors are evaluated at spatial samples before the existing tent filter.
The original tensor dataset, Sobel scores, and Sobel/overlay views are unchanged.
Stripes can alias strongly at four samples per pixel, particularly in deep views.

To regenerate these static previews without changing training data:

```sh
uv run python dataset_viewer.py --palettes zoom4x-10k-128
```

Generated preview metadata records the methods and original-RGB comparison.
Normal HTML exports preserve the additional schemes. See [references](references.md)
for the coloring methods; the blue/gold and teal/copper gradients are local presets.

## Desktop viewer and notebook

```sh
uv run python dataset_viewwer.py
uv run python dataset_viewwer.py --dataset data/zoom4x-deep-128 --split val
uv run --group notebooks jupyter lab notebooks/
```

The desktop viewer uses PySide6, follows DPI scaling, and opens maximized. Browse
thumbnails or use arrow keys; paired datasets provide an optional difference
panel. The notebook browser includes galleries, metadata, and split audits.

## Million-image run

```sh
uv run python scripts/generate_zoom_shards.py --output data/zoom4x-smooth-blue-gold-1m-128
```

Generates 100 resumable shards of 9,000 training and 1,000 validation pairs,
with only 100 overview targets across the entire run. Remaining depth bins have
continuous scale jitter and wider camera offsets. A global spatial index rejects
cameras within 1% of view width and 1% of scale across shards and splits. This
filters near-identical cameras, not all visually similar fractal structures.

Each committed `shard-NNNN/` contains the usual `train.pt`, `val.pt`, manifest,
and previews. The root manifest lists committed counts and SHA256 hashes;
`complete` becomes true only after all one million pairs are committed. Repeating
the command checks completed shards and reconstructs the deduplication index.
Interrupted partial directories are retained separately. Float32 storage requires
approximately 460 GB. The trainer now accepts the root manifest and streams all shards. See
[fast training](training-speed.md) for the configuration and memory requirements.

## Sample-location heat map

`uv run python scripts/export_sample_heatmap.py` exports all committed camera
records to `output/dataset-browser/sample-heatmap.html` with compact binary
metadata. Open through the dataset HTTP server. Filter by split, nominal
depth and input mode; scroll to zoom and drag to pan. Click a nearby center to
inspect its coordinates and target footprint, then zoom to that sample.
Heat values count centers per bin, not footprint area. Log/linear color scales
normalize to the current visible peak. The background is an approximate
128-iteration reference, independent of the training renderer.

## Conditioning mix (2026-10-17)

New generation uses 60% interior, 15% edge, 15% corner and 10% unconditional,
with integer rounding for small splits. Generator manifests record this recipe
and shard generation refuses to resume a dataset with a different recipe.

The existing million-image dataset still physically contains 25% unconditional
examples. The fast training config now sets `unconditional_fraction: 0.1`.
Each pass selects 675,000 conditional and 75,000 unconditional training rows;
the unconditional subset changes deterministically between passes. This changes
the training mixture without deleting or rerendering stored images. Validation
still uses its explicitly mode-balanced evaluation subset. Checkpoint resume
rejects changes to the training fraction; start a new run for the revised mix.
