# Mandelbrot explorer

[Open the viewer](https://andredalbosco.github.io/mandelbrot-diffusion/).

Click either image to zoom 4×. The first click loads the model automatically.
Left: rendered Mandelbrot reference. Right: 10M JiT, 5-step Heun, local WebGPU.
Only the initial model image is rendered; every subsequent zoom uses its own
previous prediction. Back and Reset restore both views together.

```sh
python3 -m http.server 8767
```

Open http://localhost:8767 in a WebGPU-capable browser.
Weights (EMA step 220079) are included in `assets/weights.safetensors`.

The reference uses 8192 iterations, four samples per pixel, tent filtering and
the dataset's smooth blue/gold palette. Shallow views use WebGPU; deeper views
use a float64 browser worker. Beyond its coordinate precision limit, the left
panel explicitly becomes unavailable while learned zooms can continue.
Finite iteration budgets and numerical precision limit the reference as well.

Training lives on `main`; experiments and the exporter live on `research`.
[tinygrad](https://github.com/tinygrad/tinygrad) · [JiT](https://arxiv.org/abs/2511.13720)
