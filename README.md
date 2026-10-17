# Mandelbrot explorer

[Open the viewer](https://andredalbosco.github.io/mandelbrot-diffusion/).

Click **Load model**, then click the image to zoom 4×. Inference runs locally
with tinygrad/WebGPU: 10M parameters, 5-step Heun, 128×128 RGB.
Only the initial view is rendered; subsequent zooms use the model’s own output.

```sh
python3 -m http.server 8767
```

Open http://localhost:8767 in a WebGPU-capable browser.
The included EMA weights (step 220079) are in `assets/weights.safetensors`.
Training lives on `main`; experiments and the exporter live on `research`.
Generated patterns may drift or lose structure under repeated zooms.

[tinygrad](https://github.com/tinygrad/tinygrad) · [JiT](https://arxiv.org/abs/2511.13720)
