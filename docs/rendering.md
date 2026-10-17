# Rendering and OpenGL viewer

Deep zooms now use tiled DD references, FP32 perturbation, and compact DD fallback,
while keeping the fixed-space pan cache, palette options, and G escape-gradient
mode. See [viewer performance](performance.md) for matched measurements.

The application uses Warp's `OpenGLRenderer` with ImGui controls. The Python
viewer in `viewer.py` transfers rendered RGB through a CUDA-registered OpenGL
buffer, without copying frames to CPU memory. Navigation and controls update
once per frame; the loop is capped at 60 Hz. Matplotlib remains for notebooks
and image export, and is not used for the interactive window.

```sh
uv run python mandelbrot.py --palette 3b1b
```

Color output defaults to 128 × 128 with four samples per output-pixel area
(`--aa-side 2`). The default `--filter tent` keeps an overscanned sample grid
anchored in Mandelbrot space while panning. Each output pixel uses a normalized
weighted sum of neighboring sample colors, with tent radius one output pixel.
The filter reuses neighboring samples; it does not evaluate 16 new orbits per
pixel. Palette changes recolor the cached grid. Leaving the cached region
rebuilds it in integer sample increments; changing zoom resets the grid spacing.
Zoom-level blending is not implemented. `--filter box` restores view-aligned
subpixel averaging. The reusable `ColorRenderer` API keeps box filtering by
default for existing dataset and notebook callers; `--bw` retains its coverage
averaging. Tent filtering follows the sampling/reconstruction distinction in
[PBRT §8.8](https://www.pbr-book.org/4ed/Sampling_and_Reconstruction/Image_Reconstruction).

The right panel has nine coloring schemes, with controls specific to each:

| Scheme | Controls |
| --- | --- |
| Cosine | Phase, iteration period |
| 3b1b original, Fire, Grayscale | Phase, linear/sqrt/log mapping, min/max range, repeat |
| 3b1b adaptive | Phase, percentile tail clipping, adaptation time |
| Smooth · blue & gold | Phase, cycles per log escape time |
| Histogram equalized | Phase, equalization strength |
| Orbit stripes | Phase, stripe contrast |
| Orbit trap · cross | Phase, cycles per distance decade |

**Reset coloring** restores the selected scheme's controls without moving the
camera. **Reset view** returns to the launch view. **Reset all parameters**
restores the main Mandelbrot view, 512 iterations, z0=0, default cosine coloring,
and default gradient controls. Resolution, sampling, and window geometry are
unchanged. Controls scroll independently from the image. All color settings
persist across watch restarts; adding new controls preserves older saved values.

Unescaped samples stay black. Color changes run on the background CUDA worker,
before tent reconstruction. Smooth and histogram modes reuse the escape field.
Stripes and traps lazily replay escaped orbits once per cached sample grid, then
share those fields across palette changes and panning. That initial replay can
be expensive at deep locations; it uses double-double coordinates at deep zooms.

Smooth uses a cyclic blue/white/gold gradient over log(1+escape). Histogram uses
the same gradient with a 512-bin interpolated CDF of visible escaped samples,
excluding the pan-cache margin. Strength blends equalization with a linear
mapping of the visible log-escape range. Unlike the paired dataset previews,
the interactive histogram is computed for one visible view.

Stripes implement [mathr's five-octave post-escape variant](https://www.mathr.co.uk/web/stripe-colouring.html),
not an average over the entire orbit. Cross traps use the smallest distance
to either coordinate axis over z1 through escape, mapped logarithmically to a
teal/copper gradient. Four samples per pixel can still alias fine stripes.

The 3b1b preset uses the palette and escape coordinate from
[3b1b/manim](https://github.com/3b1b/manim/tree/fafa083a4fb274bba9cabde0b6e2f50ba6da0622):
`n + log(2)/log(abs(z)) + abs(c)/2`, with zero-based escape index `n`, followed
by square root and palette interpolation over [1.5,8]. Other palettes use our
existing refined smooth escape value. This matches the Mandelbrot color
mapping, not Manim's complete rendering pipeline. Raise the color range when
deep zooms flatten to the final color, or select **3b1b adaptive**. The adaptive
variant is our extension: it maps the visible square-root escape values between
their 2nd and 98th percentiles by default. The range eases with a 0.35-second
time constant, continuing on cached samples after navigation stops. Set the
adaptation time to zero for immediate updates. Empty and constant fields remain
finite; overscan samples do not affect the range.

For automatic restart after edits:

```sh
uv run python mandelbrot.py --watch --palette 3b1b
```

The built-in watcher closes and waits for the previous viewer process group
before restarting, watches `mandelbrot.py`, `viewer.py`, `deep_zoom.py`, and `mandelbrot_fast.py`, and prevents duplicate
watchers for the same checkout/display. Ctrl+C closes its viewer. The application
opens maximized on the command's display. Viewport, iteration budget, initial z,
palette controls, and window geometry are saved on close and restored on reopen,
including watch reloads. State is stored per checkout/display under
`~/.local/state/mandelbrot-diffusion/` (or `$XDG_STATE_HOME` when set).

Color rendering automatically switches from float64 to double-double arithmetic
when pixels approach the ordinary coordinate precision margin. The Python Warp
kernel in `deep_zoom.py` stores each value as a high/low pair of float64 values,
with floating-point contraction disabled for its error-free transforms.
At 1024 pixels wide near unit-scale coordinates, the span guard is approximately
`2.6e-26`, versus approximately `7.3e-12` before. This is a finite precision
extension. Deep z0=0 views up to 65,536 iterations use tiled references and
perturbation with compact DD recovery; other supported configurations retain
the earlier path.

Centers retain 50-digit Decimal coordinates during panning, cursor-centered
zoom, text entry, and state serialization; the GPU kernel uses approximately
31 significant digits. The CLI accepts center components as strings to retain
their digits. Iteration limits remain independently adjustable. Grayscale
coverage rendering still uses its earlier float64 precision guard.
On Moonlight's display `:101`, launch with NVIDIA OpenGL offload enabled:

```sh
DISPLAY=:101 XAUTHORITY="$HOME/.Xauthority" \
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
uv run python mandelbrot.py --watch --palette 3b1b
```

CUDA-OpenGL interoperability requires an NVIDIA-backed OpenGL context. The
Moonlight service uses the same offload settings; plain `:101` otherwise uses
Mesa software rendering on this workstation.

## Escape-time gradient

Press **G** or use **Escape-time Sobel [G]** to switch between palette RGB and
Sobel gradients of smooth escape time. Unescaped samples are masked. Auto
contrast is enabled by default; disable it to use the manual Gradient scale
(smaller is brighter). Contrast changes display only, not sampling scores.
Mode and scale persist across restarts.


## Use from PyTorch

```python
from mandelbrot import ColorRenderer, View

renderer = ColorRenderer(width=256, height=256, device="cuda",
                      view=View(center=-0.5+0j, span=3.5, iterations=512))
rgb = renderer.render()                       # float32 [H, W, 3], range [0, 1]
sample = rgb.permute(2, 0, 1).contiguous()      # [C, H, W] for a model
other = renderer.render(View(center=-0.75+0.1j, span=0.1)).clone()
counts = renderer.counts                      # int32 [H, W]
smooth = renderer.escape                      # float32 [H, W], -1 if unescaped
```

For antialiased black-and-white coverage, use `GrayscaleRenderer` directly:

```python
from mandelbrot import GrayscaleRenderer, View, Viewer

renderer = GrayscaleRenderer(width=256, height=256, side=4)
coverage = renderer.render(View(center=-0.75+0.1j, span=0.2))  # float32 [H, W]
# render() without a view reuses the most recent view.
Viewer(renderer).show()
```

The viewer displays the two-dimensional coverage buffer with a grayscale
colormap; no RGB adapter or repeated channels are needed.

RGB, counts, and smooth escape buffers are overwritten on every render;
use `.clone()` to retain a frame. An orbit is unescaped if it has not passed
radius 2 within the iteration budget; this is a finite approximation to set
membership. CUDA launches follow the current PyTorch stream through Warp's
[stream interoperability](https://nvidia.github.io/warp/latest/user_guide/interoperability/pytorch.html),
so subsequent PyTorch operations on that stream consume the output directly.
The renderer provides forward image generation; it does not implement
autograd through escape-time decisions.
