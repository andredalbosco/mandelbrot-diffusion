# mandelbrot-diffusion

Explore the Mandelbrot set with a Python/Warp renderer and train small pixel-space
models to refine 4× zooms. Rendering and model experiments run locally on NVIDIA
CUDA; development measurements use an RTX 4090.

## Run

```sh
uv sync
uv run python mandelbrot.py --watch
```

Scroll to zoom, drag to pan, and press **G** for the escape-time gradient view.
The OpenGL app restores its camera, colors, and window settings after restarting.
Output defaults to 128×128 with four samples per pixel and tent reconstruction.
See [rendering](docs/rendering.md) for palettes, precision, and Moonlight setup.

## Inspect datasets

```sh
uv run python dataset_viewer.py
xdg-open dataset_viewer.html
```

The browser shows datasets, zoom depths, evidence modes, inputs, targets, and
coverage masks. To generate a new 13-depth RGB pilot:

```sh
uv run python dataset.py prepare-zoom --output data/new-deep-128 --max-depth 12
```

See [datasets](docs/datasets.md) for formats, viewers, and split limitations.

## Model status

JiT variants around 1M and 10M parameters predict clean 128×128 RGB images.
The mask-aware 1M setup uses noise-to-image rectified flow with fixed evidence,
sharing weights across full/partial 4× zoom inputs and unconditional generation.
One-step and sixteen-step samples are evaluated separately; one-step quality
is not guaranteed by the training objective.

```sh
uv run python train.py --config configs/zoom4x-smooth-1m-jit-1m-fast.json
```

Resume an interrupted run with `--resume runs/<run-name>`. Extend a completed
run with additional optimizer updates:

```sh
uv run python train.py --resume runs/jit-1m-smooth-1m-patch32 --additional-steps 100000
```

This restores the saved configuration, weights, EMA, optimizer, random states,
and dataset position. Warmup does not restart and the learning rate is unchanged.
`--steps` instead sets the total budget including previous updates. Do not run
two trainers in the same output directory simultaneously.
Use `--max-seconds 3600` to cap a session at one hour of the training loop;
startup and final evaluation/checkpoint saving add some time. The run stops at
whichever comes first: its update budget or its time limit.

[JiT recipe audit](docs/jit-training-audit.md) · [Fast training](docs/training-speed.md) ·
[Model setups](docs/models.md) · [Earlier experiments](docs/experiments.md) ·
[Executed notebooks](notebooks/README.md)

## Explore the trained model

```sh
uv run python explore.py --checkpoint runs/jit-10m-smooth-1m-patch32-pilot/checkpoint.pt
```

Open `http://127.0.0.1:8766`. Click an image to zoom 4×; use arrows to pan,
Back to retrace, and New noise to compare samples. The viewer shows the input,
50-step Heun prediction, and reference render. Only the initial view uses a rendered seed; subsequent views condition on the
previous model output. The reference render is comparison-only. To access it from
another machine, bind `--host` to the workstation's private network address.

A fully client-side **tinygrad/WebGPU explorer** lives in [web/](web/README.md).
It runs recursive 50-step Heun sampling on the browser GPU, with no inference server.

## Measurements and tests

[Viewer performance](docs/performance.md) · [Generation](docs/fast-generation.md) ·
[Training throughput](docs/throughput.md) · [JiT benchmarks](docs/jit-benchmark.md)

```sh
uv run python -m unittest discover -s tests -v
```

Core files: `mandelbrot.py` and `mandelbrot_fast.py` render samples; `viewer.py`
displays them; `dataset.py` prepares data; `model.py`, `train.py`, and `sample.py`
implement the learning experiments. Generated data and results live in ignored
`data/`, `runs/`, and `output/` directories.

## References

- [JiT](https://arxiv.org/abs/2511.13720) and [Rectified Flow](https://arxiv.org/abs/2209.03003): model formulation.
- [Fraktaler 3 / Deep Zoom](https://mathr.co.uk/web/deep-zoom.html): perturbation, rebasing, and BLA.
- [FractalShark](https://github.com/mattsaccount364/FractalShark): CUDA rendering and linear approximation.
- [Imagina](https://github.com/5E-324/Imagina) and [Bert Baron's explorer](https://github.com/bertbaron/mandelbrot): approximation, recovery, and progressive rendering.
- [PBRT: Image Reconstruction](https://pbr-book.org/4ed/Sampling_and_Reconstruction/Image_Reconstruction): sampling and reconstruction filters.

[Reference notes](docs/references.md)
