# mandelbrot-diffusion

PyTorch + NVIDIA Warp Mandelbrot image generation. The Warp kernel in
`mandelbrot.py` computes `z ← z² + c`, using double precision for coordinates
and iteration, and writes directly into reusable PyTorch tensors.

## Run

```sh
uv sync
uv run python mandelbrot.py
```

Linux x86-64 uses PyTorch's CUDA 12.8 wheels. Device selection prefers CUDA
when both PyTorch and Warp can access it, otherwise CPU. On macOS, Warp runs
on CPU; this renderer does not use Metal/MPS. The first render compiles the
kernel and takes longer than subsequent renders.

Initialize the view and resolution from the command line:

```sh
uv run python mandelbrot.py --width 1280 --height 960 \
  --center -0.743643887037151 0.131825904205330 --span 0.003 --iterations 1500
uv run python mandelbrot.py --z0 0.2 0.1
uv run python mandelbrot.py --headless --output output/mandelbrot.png
```

Scroll to zoom around the pointer; left-drag to pan. Edit **Center**, **Span**,
**Steps**, or **z₀** and press Enter to apply. Complex fields accept values
such as `-0.5+0.1j`. **R** / Reset restores the initial view, **S** / Save PNG
saves the image, and **Q** / Escape closes the window. Changing z₀ explores
the parameter plane for a different starting orbit; the standard Mandelbrot
set uses z₀ = 0. Window resizing scales the image; `--width` and `--height`
set its sampling resolution. Double precision still has a finite zoom limit.

On blacknoir, show the window in the existing Xpra session:

```sh
cd ~/Projects/mandelbrot-diffusion
DISPLAY=:100 XAUTHORITY="$HOME/.Xauthority" uv run python mandelbrot.py
```

## Use from PyTorch

```python
from mandelbrot import Mandelbrot, View

renderer = Mandelbrot(width=256, height=256, device="auto",
                      view=View(center=-0.5+0j, span=3.5, iterations=512))
rgb = renderer.render()                       # float32 [H, W, 3], range [0, 1]
sample = rgb.permute(2, 0, 1).contiguous()      # [C, H, W] for a model
other = renderer.render(View(center=-0.75+0.1j, span=0.1)).clone()
counts = renderer.counts                      # int32 [H, W]
smooth = renderer.escape                      # float32 [H, W], -1 if unescaped
```

RGB, counts, and smooth escape buffers are overwritten on every render;
use `.clone()` to retain a frame. An orbit is unescaped if it has not passed
radius 2 within the iteration budget; this is a finite approximation to set
membership. CUDA launches follow the current PyTorch stream through Warp's
[stream interoperability](https://nvidia.github.io/warp/latest/user_guide/interoperability/pytorch.html),
so subsequent PyTorch operations on that stream consume the output directly.
The renderer provides forward image generation; it does not implement
autograd through escape-time decisions.

## Validate

```sh
uv run python -m unittest discover -s tests -v
```

Tests compare escape counts and smooth values with an independent scalar
reference, check CPU/CUDA agreement and non-default CUDA stream ordering,
and exercise the viewer's controls without opening a window.
