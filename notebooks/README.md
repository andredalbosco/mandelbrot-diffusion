# Notebooks

Run these in order, or independently from a fresh kernel:

1. [Zoom pairs](01_zoom_pairs.ipynb): real parent renders, movable 32x32 crops,
   128x128 targets, coordinate alignment, iteration-budget sensitivity, and an
   antialiasing comparison from 1x1 through 16x16 samples per pixel.
2. [Rectified flow](02_rectified_flow.ipynb): interpolation, target velocities,
   and an oracle check of the repository's Euler solver. No learned model is
   used in the oracle demonstration.
3. [DiT and U-Net](03_models.ipynb): actual configured architectures, parameter
   counts, one CUDA optimizer step, EMA sampling, and memory measurements.
4. [Dataset browser](04_dataset_browser.ipynb): interactive dataset/split/sample
   selection, paired input/target differences, galleries, rendering metadata,
   coverage/scale distributions, and spatial-overlap audits. Opens the new
   900/100-example multiscale RGB dataset by default (seven depths, 1×–4,096×).

From the repository root:

```sh
uv run --group notebooks jupyter lab notebooks/
```

Select the Python kernel from this repository's `.venv` when using another
notebook editor. The rendering and model notebooks require CUDA. The notebooks locate the repository from
either its root or the `notebooks/` working directory and import the existing
renderer and models. The dataset browser is CPU-only and uses `ipywidgets`,
included in the notebooks dependency group. Its saved static previews are
visible even without interactive widget support.

Saved outputs illustrate actual execution. Rerunning replaces those outputs;
it does not write datasets or checkpoints. These are small demonstrations,
not evidence of trained zoom quality. The first RGB pilot contains 1,000 examples; the larger collection remains pending.

The rendering demonstrations use 8x8 spatial supersampling (64 samples per output pixel),
averaging RGB values after coloring. The 16x16 comparison is a higher-sample
reference, not a guarantee of convergence. Display uses nearest-neighbor
interpolation so smoothing in the notebook viewer does not conceal aliasing.
The dataset browser shows the stored samples and their recorded rendering
settings; the RGB overfit pair uses four samples per pixel.
