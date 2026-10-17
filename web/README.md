# Browser Mandelbrot explorer

A static, client-side version of the recursive 4× explorer. tinygrad exports the
trained JiT into WGSL compute kernels and a JavaScript WebGPU runner. The browser
loads approximately 39 MB of float32 EMA weights. No Python, CUDA, or inference
API is used at runtime. The server only delivers static files.

Open the private HTTPS deployment:
https://blacknoir.tail353b14.ts.net/mandelbrot-web/

Use a WebGPU-capable browser (Chrome recommended). HTTPS or localhost is required.
Click **Load model**, then click the generated image to zoom. **Back** restores
an earlier image, **Reset** restores the rendered seed, and **New variation**
resamples the current conditioning with a different seed.

The initial seed is a 128×128 reference render in smooth blue and gold. Every
subsequent view crops a 32×32 patch from the previous generated image, expands it
with bilinear reconstruction and a coverage mask, and samples 5-step Heun
(4 predictor/corrector steps followed by one Euler step, 9 network evaluations).
Generated images stay float32 between zooms; canvas quantization is display-only.
Sampling uses deterministic seeded Gaussian noise and the checkpoint's endpoint
clamp of 0.05. Patterns can drift from the exact Mandelbrot set.

## Files

- `model.py`: inference-only tinygrad translation of the trained architecture.
- `export.py`: checkpoint conversion, PyTorch parity check, WebGPU export.
- `assets/`: exported kernels, weights, manifest, seed and parity fixtures.
- `worker.js`: WebGPU loading and Heun integration off the UI thread.
- `app.js`, `index.html`, `style.css`: recursive zoom interface.

## Rebuild

Run from the repository root using its Python environment. Use the exact tinygrad
revision below; `extra.export_model` is a source-tree module, not a stable API.
Dawn is needed on the export machine, not on the browser client.

```sh
git clone https://github.com/tinygrad/tinygrad.git /tmp/mandelbrot-tinygrad
git -C /tmp/mandelbrot-tinygrad checkout 002630241913a57abccf646b092878ec8a9b882f
curl -fL https://github.com/wpmed92/pydawn/releases/download/v0.1.6/libwebgpu_dawn.so -o /tmp/libwebgpu_dawn.so
PYTHONPATH=/tmp/mandelbrot-tinygrad DEV=WEBGPU WEBGPU_PATH=/tmp/libwebgpu_dawn.so \
  .venv/bin/python web/export.py
```

The default checkpoint is `runs/jit-10m-smooth-1m-patch32-pilot/checkpoint.pt`.
Override with `--checkpoint`. Exported metadata includes its SHA256 and step.
The bundled seed image is preserved across model exports.

For a local static deployment, copy this folder and run:

```sh
python3 -m http.server 8767 --directory web
```

Then open http://localhost:8767. A remote plain-HTTP IP address will not enable
WebGPU. The private deployment uses Tailscale HTTPS; no public Funnel is enabled.

## Validation

The native tinygrad WebGPU forward pass was compared to PyTorch FP32 on a fixed
noisy input at t=0.4 with partial conditioning coverage. Maximum absolute error:
2.861e-6. This is forward-pass parity, not a guarantee that sampled trajectories
match across GPUs. Browser sampling is float32 whereas the Python viewer uses
bfloat16 autocast. Append `?test` and load the model to run the same parity fixture
inside your browser; the result appears in the status line.

Chrome WebGPU on the RTX 4090 also passed this fixture (max error 2.861e-6,
mean error 3.780e-7). Two successive 50-step zooms, Back and Reset were exercised
in a real browser. One zoom took 1.30 seconds on that machine; Mac performance
has not been measured. The export script works around upstream shape-based
kernel-name collisions, and materializes the final NCHW output layout.

## References

- [tinygrad exporter](https://github.com/tinygrad/tinygrad/blob/002630241913a57abccf646b092878ec8a9b882f/extra/export_model.py), MIT; see `tinygrad-LICENSE`.
- [JiT](https://arxiv.org/abs/2511.13720).

An additional 50-step browser sample succeeded with the browser context offline
after model loading (1.53 seconds), confirming there is no inference-server request.
