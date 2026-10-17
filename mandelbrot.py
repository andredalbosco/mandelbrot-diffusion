"""Warp Mandelbrot renderer with reusable PyTorch buffers and a small viewer."""

import argparse
from dataclasses import dataclass
from decimal import Decimal, localcontext
import math
import time
from pathlib import Path

import torch
import warp as wp

from viewer import Viewer


@wp.kernel(enable_backward=False)
def escape_time_kernel(
    center_x: wp.float64,
    center_y: wp.float64,
    span: wp.float64,
    z0_x: wp.float64,
    z0_y: wp.float64,
    max_iterations: int,
    counts: wp.array2d(dtype=wp.int32),
    escape: wp.array2d(dtype=wp.float32),
    styled_escape: wp.array2d(dtype=wp.float32),
):
    y, x = wp.tid()
    height = counts.shape[0]
    width = counts.shape[1]
    # Square pixels sampled at their centers; row zero is the top of the view.
    step = span / wp.float64(width)
    cx = center_x + (wp.float64(x) + wp.float64(0.5) - wp.float64(width) * wp.float64(0.5)) * step
    cy = center_y + (wp.float64(height) * wp.float64(0.5) - wp.float64(y) - wp.float64(0.5)) * step
    zx = z0_x
    zy = z0_y
    n = int(0)

    # These interior tests apply only to the standard z0 = 0 Mandelbrot set.
    if z0_x == wp.float64(0.0) and z0_y == wp.float64(0.0):
        dx = cx - wp.float64(0.25)
        bulb_x = cx + wp.float64(1.0)
        q = dx * dx + cy * cy
        if q * (q + cx - wp.float64(0.25)) <= wp.float64(0.25) * cy * cy:
            n = max_iterations
        elif bulb_x * bulb_x + cy * cy <= wp.float64(0.0625):
            n = max_iterations

    radius_squared = zx * zx + zy * zy
    while n < max_iterations and radius_squared <= wp.float64(4.0):
        next_x = zx * zx - zy * zy + cx
        zy = wp.float64(2.0) * zx * zy + cy
        zx = next_x
        radius_squared = zx * zx + zy * zy
        n += 1

    counts[y, x] = n
    smooth = float(-1.0)
    styled = float(-1.0)
    if radius_squared > wp.float64(4.0):
        # 3b1b's escape coordinate, before the additional potential refinement.
        styled = float(wp.float64(n - 1) + wp.log(wp.float64(2.0)) / (wp.float64(0.5) * wp.log(radius_squared))
                       + wp.float64(0.5) * wp.sqrt(cx * cx + cy * cy))
        # Refine the potential after bailout without changing escape counts.
        # Stop before squaring could overflow for unusually large inputs.
        refined_n = n
        for _ in range(3):
            if radius_squared < wp.float64(1.0e100):
                next_x = zx * zx - zy * zy + cx
                zy = wp.float64(2.0) * zx * zy + cy
                zx = next_x
                radius_squared = zx * zx + zy * zy
                refined_n += 1
        # Continuous escape time: n + 1 - log2(log(|z|)).
        smooth = float(wp.float64(refined_n + 1) - wp.log(wp.float64(0.5) * wp.log(radius_squared)) / wp.log(wp.float64(2.0)))
    escape[y, x] = smooth
    styled_escape[y, x] = styled


@wp.kernel(enable_backward=False)
def palette_kernel(
    period: float,
    offset: float,
    escape: wp.array2d(dtype=wp.float32),
    image: wp.array3d(dtype=wp.float32),
):
    y, x = wp.tid()
    smooth = escape[y, x]
    r = float(0.0)
    g = float(0.0)
    b = float(0.0)
    if smooth != float(-1.0):
        phase = float(6.283185307179586) * (smooth / period + offset)
        r = float(0.5) + float(0.5) * wp.cos(phase)
        g = float(0.5) + float(0.5) * wp.cos(phase + float(2.1))
        b = float(0.5) + float(0.5) * wp.cos(phase + float(4.2))
    image[y, x, 0] = r
    image[y, x, 1] = g
    image[y, x, 2] = b


@wp.kernel(enable_backward=False)
def escape_mask_kernel(
    center_x: wp.float64, center_y: wp.float64, span: wp.float64,
    z0_x: wp.float64, z0_y: wp.float64, iterations: int,
    samples: wp.array2d(dtype=wp.float32),
):
    y, x = wp.tid()
    height, width = samples.shape[0], samples.shape[1]
    step = span / wp.float64(width)
    cx = center_x + (wp.float64(x) + wp.float64(0.5) - wp.float64(width) * wp.float64(0.5)) * step
    cy = center_y + (wp.float64(height) * wp.float64(0.5) - wp.float64(y) - wp.float64(0.5)) * step
    zx, zy = z0_x, z0_y
    n = int(0)
    if z0_x == wp.float64(0.0) and z0_y == wp.float64(0.0):
        dx = cx - wp.float64(0.25)
        q = dx * dx + cy * cy
        bulb_x = cx + wp.float64(1.0)
        if q * (q + dx) <= wp.float64(0.25) * cy * cy:
            n = iterations
        elif bulb_x * bulb_x + cy * cy <= wp.float64(0.0625):
            n = iterations
    radius_squared = zx*zx + zy*zy
    while n < iterations and radius_squared <= wp.float64(4.0):
        next_x = zx*zx - zy*zy + cx
        zy = wp.float64(2.0)*zx*zy + cy
        zx = next_x
        radius_squared = zx*zx + zy*zy
        n += 1
    value = float(0.0)
    if radius_squared > wp.float64(4.0):
        value = float(1.0)
    samples[y, x] = value


@wp.kernel(enable_backward=False)
def average_samples_kernel(side: int, samples: wp.array2d(dtype=wp.float32),
                   coverage: wp.array2d(dtype=wp.float32)):
    y, x = wp.tid()
    total = float(0.0)
    for sy in range(side):
        for sx in range(side):
            total += samples[y*side+sy, x*side+sx]
    coverage[y, x] = total / float(side*side)


@wp.kernel(enable_backward=False)
def average_colors_kernel(side: int, samples: wp.array3d(dtype=wp.float32),
                          image: wp.array3d(dtype=wp.float32)):
    y, x = wp.tid()
    for channel in range(3):
        total = float(0.0)
        for sy in range(side):
            for sx in range(side):
                total += samples[y*side+sy, x*side+sx, channel]
        image[y, x, channel] = total / float(side*side)


@dataclass(frozen=True)
class View:
    center: complex = -0.5 + 0j
    span: float = 3.5  # Horizontal extent in the complex plane.
    iterations: int = 512
    z0: complex = 0j
    center_exact: tuple[str, str] | None = None

    def __post_init__(self):
        center = self.center
        exact = self.center_exact
        if isinstance(center, str):
            text = center.replace(" ", "").strip("()")
            center = complex(text)
            if text.endswith("j"):
                split = next((i for i in range(len(text)-2, 0, -1)
                              if text[i] in "+-" and text[i-1] not in "eE"), None)
                real, imag = (text[:split], text[split:-1]) if split else ("0", text[:-1])
                imag = {"": "1", "+": "1", "-": "-1"}.get(imag, imag)
                exact = (str(Decimal(real)), str(Decimal(imag)))
            else:
                exact = (str(Decimal(text)), "0")
        if exact is None or complex(float(exact[0]), float(exact[1])) != center:
            exact = (str(center.real), str(center.imag))
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "center_exact", exact)
        values = (self.center.real, self.center.imag, self.span, self.z0.real, self.z0.imag)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("View parameters must be finite")
        if self.span <= 0:
            raise ValueError("Span must be positive")
        if type(self.iterations) is not int or not 1 <= self.iterations <= 2**31 - 1:
            raise ValueError("Iterations must be an integer between 1 and 2147483647")

    def extent(self, width, height):
        half_width = self.span / 2
        half_height = half_width * height / width
        return (self.center.real - half_width, self.center.real + half_width,
                self.center.imag - half_height, self.center.imag + half_height)

    @property
    def center_text(self):
        real, imag = self.center_exact
        return f"({real}{'' if imag.startswith('-') else '+'}{imag}j)"

    def shifted(self, offset, span=None):
        from dataclasses import replace
        with localcontext() as context:
            context.prec = 50
            real = Decimal(self.center_exact[0]) + Decimal(str(offset.real))
            imag = Decimal(self.center_exact[1]) + Decimal(str(offset.imag))
        return replace(self, center=complex(float(real), float(imag)),
                       center_exact=(str(real), str(imag)), span=self.span if span is None else span)


def renderer_devices(device):
    """Resolve matching PyTorch and Warp devices without allocating image buffers."""
    wp.init()
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("Rendering requires a CUDA device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device, wp.device_from_torch(device)


@wp.kernel(enable_backward=False)
def tent_resolve(samples: wp.array3d(dtype=wp.float32), side: int,
                 offset_x: float, offset_y: float, image: wp.array3d(dtype=wp.float32)):
    y, x = wp.tid()
    # Coordinates measured from the first cached sample, in sample spacings.
    sx = offset_x + (float(x) + 0.5) * float(side) - 0.5
    sy = offset_y + (float(y) + 0.5) * float(side) - 0.5
    ix, iy = int(wp.floor(sx)), int(wp.floor(sy))
    total = wp.vec3(0.0)
    weight_sum = float(0.0)
    # Radius is one OUTPUT pixel: four samples across each axis at side=2.
    for j in range(iy-side+1, iy+side+1):
        for i in range(ix-side+1, ix+side+1):
            weight = wp.max(0.0, 1.0-wp.abs(float(i)-sx)/float(side)) * wp.max(0.0, 1.0-wp.abs(float(j)-sy)/float(side))
            total += weight * wp.vec3(samples[j, i, 0], samples[j, i, 1], samples[j, i, 2])
            weight_sum += weight
    for channel in range(3):
        image[y, x, channel] = total[channel] / weight_sum


class ColorRenderer:
    """Render Mandelbrot escape-time colors into reusable CUDA buffers.

    render() returns float32 RGB [height, width, 3] in [0, 1]. For box filtering, counts and
    escape attributes contain per-sample int32 iteration counts and float32
    smooth escape times, at [height*side, width*side]. Each subpixel is colored
    before resolving the image. Unescaped samples have escape = -1 and RGB = 0. Buffers are reused:
    clone tensors when retaining multiple frames. This is a forward renderer.
    Tent filtering instead resolves a separate overscanned world grid; its orbit
    and sample buffers live in _grid, while image retains the output dimensions.
    """

    styles = {
        "cosine": "Cosine", "3b1b": "3b1b original", "3b1b_adaptive": "3b1b adaptive",
        "smooth": "Smooth · blue & gold", "histogram": "Histogram equalized",
        "stripes": "Orbit stripes", "orbit_trap": "Orbit trap · cross",
        "fire": "Fire", "grayscale": "Grayscale",
    }
    color_fields = ("style", "mapping", "period", "phase", "lower", "upper", "repeat",
                    "smooth_scale", "histogram_strength", "stripe_contrast", "trap_scale",
                    "adaptive_clip", "adaptive_seconds")
    color_defaults = dict(style="cosine", mapping="sqrt", period=2*math.pi/.15,
                          phase=0., lower=1.5, upper=8., repeat=False,
                          smooth_scale=.7, histogram_strength=1., stripe_contrast=1.,
                          trap_scale=.2, adaptive_clip=2., adaptive_seconds=.35)

    def __init__(self, width=1024, height=768, device="cuda", view=None, side=1, filter="box", pan_margin=True):
        if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive integers")
        if type(side) is not int or not 1 <= side <= 16:
            raise ValueError("AA side must be an integer from 1 to 16")
        if filter not in ("box", "tent"):
            raise ValueError("Unknown reconstruction filter")
        self.filter = filter
        self.pan_margin = pan_margin
        self._grid = None
        self._grid_key = None
        self.device, self.warp_device = renderer_devices(device)
        self.width, self.height = width, height
        self.side = side
        self.sample_width, self.sample_height = width*side, height*side
        self.view = view if view is not None else View()
        for name, value in self.color_defaults.items():
            setattr(self, name, value)
        self.counts = torch.empty((self.sample_height, self.sample_width), dtype=torch.int32, device=self.device)
        self.escape = torch.empty((self.sample_height, self.sample_width), dtype=torch.float32, device=self.device)
        self.styled_escape = torch.empty_like(self.escape)
        self.image = torch.empty((height, width, 3), dtype=torch.float32, device=self.device)
        self.samples = (self.image if side == 1 else
                        torch.empty((self.sample_height, self.sample_width, 3), dtype=torch.float32, device=self.device))
        # Wrap once: Warp and PyTorch share the same storage.
        self._outputs = [wp.from_torch(tensor) for tensor in (self.counts, self.escape, self.image)]
        self._styled_escape = wp.from_torch(self.styled_escape)
        self._samples = wp.from_torch(self.samples)
        self._adaptive_range = None
        self._adaptive_time = None
        self._adaptive_target_key = None
        self.adaptive_pending = False
        self._orbit_fields = None
        self._orbit_key = None
        self.precision = "float64"
        self._reference_orbit = None
        self._reference_length = wp.empty(1, dtype=wp.int32, device=self.warp_device)
        # Blue/gold stops and escape coordinate from 3b1b/manim:
        # https://github.com/3b1b/manim/tree/fafa083a4fb274bba9cabde0b6e2f50ba6da0622
        # manimlib/mobject/fractals.py and shaders/mandelbrot_fractal.wgsl.
        palettes = {
            "3b1b": ["00065c", "061e7e", "0c37a0", "205abc", "4287d3",
                     "d9ede4", "f0f9e4", "ba9f6a", "573706"],
            "fire": ["08051a", "38104a", "861745", "d93629", "ff8b26", "fff2b0"],
            "grayscale": ["000000", "ffffff"],
            "blue_gold": ["000750", "1446b4", "50aae6", "ebfaff", "ffb92d", "822d0a", "000750"],
            "teal_copper": ["05141c", "147d8c", "d2f0e1", "fab450", "962d14", "1e0c1e", "05141c"],
        }
        self.palettes = {
            name: torch.tensor([[int(color[i:i+2], 16) / 255 for i in (0, 2, 4)] for color in colors],
                               device=self.device)
            for name, colors in palettes.items()
        }

    def minimum_span(self, view):
        extent = max(abs(view.center.real), abs(view.center.imag), 1.0)
        return 32 * 2.0**-100 * extent * self.sample_width

    def validate_view(self, view):
        if view.span < self.minimum_span(view):
            raise ValueError("View exceeds the double-double precision margin")

    def render(self, view=None):
        view = self.view if view is None else view
        self.validate_view(view)
        if self.filter == "tent":
            return self._render_grid(view)
        self._orbit_key = None
        self._adaptive_target_key = None
        stream = wp.stream_from_torch(torch.cuda.current_stream(self.device))
        extent = max(abs(view.center.real), abs(view.center.imag), 1.0)
        if (view.span < 32 * math.ulp(extent) * self.sample_width
                and view.z0 == 0j and view.iterations <= 65536):
            # Reuse the dataset backend on exactly the viewer's sample lattice.
            # No extra halo/resolve: the parent renderer owns its pan cache,
            # palette and reconstruction. Keep both escape fields for G/3b1b.
            from mandelbrot_fast import FastRenderer
            self._tiled_backend = FastRenderer([view],width=self.sample_width,
                                              height=self.sample_height,side=1,
                                              precision="perturb32",tile=64,guard="bailout",
                                              halo=False,device=str(self.device))
            self._tiled_backend.render(resolve=False)
            self.counts.copy_(self._tiled_backend.counts[0])
            self.escape.copy_(self._tiled_backend.escape[0])
            self.styled_escape.copy_(self._tiled_backend.styled_escape[0])
            self.precision = "Tiled DD reference + FP32 perturbation + compact DD fallback"
            self.view = view
            return self.recolor()
        if view.span < 32 * math.ulp(extent) * self.sample_width:
            self.precision = "DD reference + float64 perturbation"
            from deep_zoom import escape_time_dd_kernel, reference_orbit_kernel, perturbation_kernel
            with localcontext() as context:
                context.prec = 50
                def split(value):
                    high = float(value)
                    return wp.vec2d(high, float(value-Decimal.from_float(high)))
                cx, cy = (split(Decimal(value)) for value in view.center_exact)
                step = split(Decimal(str(view.span))/Decimal(self.sample_width))
            # A bounded reference allocation; exceptionally long or unstable
            # trajectories automatically use the direct double-double fallback.
            limit = min(view.iterations, 65536)
            if self._reference_orbit is None or len(self._reference_orbit) < limit+1:
                self._reference_orbit = wp.empty(limit+1, dtype=wp.vec4d, device=self.warp_device)
            wp.launch(reference_orbit_kernel, dim=1,
                      inputs=[cx, cy, view.z0.real, view.z0.imag, limit],
                      outputs=[self._reference_orbit, self._reference_length],
                      device=self.warp_device, stream=stream)
            wp.launch(perturbation_kernel, dim=(self.sample_height, self.sample_width),
                      inputs=[view.center.real, view.center.imag, view.span/self.sample_width, view.iterations,
                              self._reference_orbit, self._reference_length],
                      outputs=[*self._outputs[:2], self._styled_escape],
                      device=self.warp_device, stream=stream)
            wp.launch(escape_time_dd_kernel, dim=(self.sample_height, self.sample_width),
                      inputs=[cx, cy, step, view.z0.real, view.z0.imag, view.iterations, True],
                      outputs=[*self._outputs[:2], self._styled_escape],
                      device=self.warp_device, stream=stream)
            self.view = view
            return self.recolor()
        self.precision = "float64"
        wp.launch(
            escape_time_kernel,
            dim=(self.sample_height, self.sample_width),
            inputs=[view.center.real, view.center.imag, view.span, view.z0.real, view.z0.imag, view.iterations],
            outputs=[*self._outputs[:2], self._styled_escape],
            device=self.warp_device,
            stream=stream,
        )
        self.view = view
        return self.recolor()

    def _render_grid(self, view):
        key = (view.span, view.iterations, view.z0)
        if self._grid_key != key:
            self._grid_key = key
            self._grid_anchor = view
            with localcontext() as context:
                context.prec = 50
                self._grid_step = Decimal(str(view.span)) / Decimal(self.sample_width)
            self._grid_origin = None
            # A quarter-view margin avoids orbit evaluations on most pan frames.
            margin = max(2*self.side, (max(self.sample_width, self.sample_height)+3)//4 if self.pan_margin else 0)
            self._grid = ColorRenderer(self.sample_width+2*margin, self.sample_height+2*margin,
                                       device=str(self.device))
        grid = self._grid
        with localcontext() as context:
            context.prec = 50
            dx = (Decimal(view.center_exact[0])-Decimal(self._grid_anchor.center_exact[0])) / self._grid_step
            dy = -(Decimal(view.center_exact[1])-Decimal(self._grid_anchor.center_exact[1])) / self._grid_step
            origin = self._grid_origin
            ox = float(dx-origin[0]) if origin else 0.0
            oy = float(dy-origin[1]) if origin else 0.0
            mx = (grid.width-self.sample_width)/2
            my = (grid.height-self.sample_height)/2
            if origin is None or abs(ox) > mx-self.side or abs(oy) > my-self.side:
                # Integer moves preserve the lattice; Decimal keeps deep anchors.
                origin = (round(dx), round(dy))
                self._grid_origin = origin
                real = Decimal(self._grid_anchor.center_exact[0]) + origin[0]*self._grid_step
                imag = Decimal(self._grid_anchor.center_exact[1]) - origin[1]*self._grid_step
                cache_view = View(center=f"({real}{'' if imag < 0 else '+'}{imag}j)",
                                  span=float(self._grid_step*grid.width),
                                  iterations=view.iterations, z0=view.z0)
                grid.render(cache_view)
                ox, oy = float(dx-origin[0]), float(dy-origin[1])
        self._grid_offset = (mx+ox, my+oy)
        self.view = view
        self.precision = grid.precision
        return self.recolor()

    def color_settings(self):
        return {name: getattr(self, name) for name in self.color_fields}

    def reset_colors(self):
        """Restore only the controls used by the selected coloring scheme."""
        fields = {
            "cosine": ("period",), "smooth": ("smooth_scale",),
            "histogram": ("histogram_strength",), "stripes": ("stripe_contrast",),
            "orbit_trap": ("trap_scale",),
            "3b1b_adaptive": ("adaptive_clip", "adaptive_seconds"),
        }.get(self.style, ("mapping", "lower", "upper", "repeat"))
        self.set_colors(**{name: self.color_defaults[name] for name in ("phase", *fields)})

    def set_colors(self, **updates):
        """Validate controls without launching GPU work on the UI thread."""
        unknown = updates.keys() - self.color_fields
        if unknown:
            raise ValueError(f"Unknown color controls: {', '.join(unknown)}")
        values = self.color_settings() | {k: v for k, v in updates.items() if v is not None}
        if values["style"] not in self.styles or values["mapping"] not in ("linear", "sqrt", "log"):
            raise ValueError("Unknown color style or value mapping")
        for name in self.color_fields[2:]:
            if not math.isfinite(values[name]):
                raise ValueError(f"{name} must be finite")
        if values["period"] <= 0 or values["lower"] >= values["upper"]:
            raise ValueError("Need period > 0 and color range min < max")
        for name in ("smooth_scale", "stripe_contrast", "trap_scale"):
            if values[name] <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= values["histogram_strength"] <= 1:
            raise ValueError("Histogram strength must be in [0, 1]")
        if not 0 <= values["adaptive_clip"] < 50 or values["adaptive_seconds"] < 0:
            raise ValueError("Need tail clip in [0, 50) and adaptation seconds >= 0")
        if values["style"] != self.style:
            self._adaptive_range = self._adaptive_time = None
        for name, value in values.items():
            setattr(self, name, value)

    def _visible(self, field):
        """View statistics exclude the overscan used for cached panning."""
        if self.filter != "tent" or self._grid is None:
            return field
        x, y = (round(v) for v in self._grid_offset)
        return field[y:y+self.sample_height, x:x+self.sample_width]

    def _adaptive_limits(self, values, valid):
        key = (self.view, self.adaptive_clip)
        if self._adaptive_target_key != key:
            visible = self._visible(values)[self._visible(valid)]
            if visible.numel():
                q = values.new_tensor([self.adaptive_clip/100, 1-self.adaptive_clip/100])
                target = torch.quantile(visible, q)
                # A flat field must not divide by zero or amplify roundoff.
                gap = torch.maximum(target[0].abs()*1e-5, target.new_tensor(1e-4))
                target[1] = torch.maximum(target[1], target[0]+gap)
            else:
                target = self._adaptive_range
                if target is None:
                    target = values.new_tensor([1.5, 8.0])
            self._adaptive_target = target
            self._adaptive_target_key = key
        target = self._adaptive_target
        now = time.monotonic()
        if self._adaptive_range is None or self.adaptive_seconds == 0:
            self._adaptive_range = target.clone()
        else:
            dt = max(0., now-self._adaptive_time)
            amount = -math.expm1(-dt/self.adaptive_seconds)
            self._adaptive_range = torch.lerp(self._adaptive_range, target, amount)
        self._adaptive_time = now
        tolerance = (target[1]-target[0]).clamp_min(1e-4)*.002
        self.adaptive_pending = bool(((target-self._adaptive_range).abs().max() > tolerance).item())
        if not self.adaptive_pending:
            self._adaptive_range = target.clone()
        return self._adaptive_range

    def orbit_fields(self):
        """Compute once per sample grid; color sliders reuse both orbit fields."""
        if self._orbit_key != self.view:
            from deep_zoom import orbit_fields_kernel
            with localcontext() as context:
                context.prec = 50
                def split(value):
                    high = float(value)
                    return wp.vec2d(high, float(value-Decimal.from_float(high)))
                cx, cy = (split(Decimal(v)) for v in self.view.center_exact)
                step = split(Decimal(str(self.view.span))/self.sample_width)
            if self._orbit_fields is None:
                self._orbit_fields = torch.empty((*self.escape.shape, 2), device=self.device)
            deep = self.precision != "float64"
            wp.launch(orbit_fields_kernel, dim=self.escape.shape,
                      inputs=[cx, cy, step, self.view.z0.real, self.view.z0.imag,
                              self.view.iterations, deep, self._outputs[1]],
                      outputs=[wp.from_torch(self._orbit_fields, dtype=wp.vec2)],
                      device=self.warp_device,
                      stream=wp.stream_from_torch(torch.cuda.current_stream(self.device)))
            self._orbit_key = self.view
        return self._orbit_fields

    def _histogram(self, values, valid):
        visible = self._visible(values)[self._visible(valid)]
        if not visible.numel():
            return torch.zeros_like(values)
        low, high = visible.min(), visible.max()
        span = (high-low).clamp_min(1e-6)
        alpha = ((values-low)/span).clamp(0, 1)
        positions = ((visible-low)/span*511).clamp(0, 511)
        counts = torch.bincount(positions.long(), minlength=512).to(values.dtype)
        cdf = counts.cumsum(0)/counts.sum().clamp_min(1)
        before = torch.cat((cdf.new_zeros(1), cdf[:-1]))
        index = (alpha*511).long()
        equalized = before[index]+(cdf-before)[index]*(alpha*511-index)
        return torch.lerp(alpha, equalized, self.histogram_strength)

    def _color_samples(self, source):
        style = self.style
        valid = source.escape != -1
        if style == "cosine":
            wp.launch(palette_kernel, dim=source.escape.shape,
                      inputs=[self.period, self.phase, source._outputs[1]], outputs=[source._samples],
                      device=self.warp_device,
                      stream=wp.stream_from_torch(torch.cuda.current_stream(self.device)))
            return
        palette_name = "blue_gold"
        if style == "smooth":
            alpha = (source.escape.clamp_min(0).log1p()*self.smooth_scale+self.phase).remainder(1)
        elif style == "histogram":
            alpha = (self._histogram(source.escape.clamp_min(0).log1p(), valid)+self.phase).clamp(0, 1)
        elif style == "stripes":
            alpha = ((source.orbit_fields()[...,0]-.5)*self.stripe_contrast+.5+self.phase).clamp(0, 1)
        elif style == "orbit_trap":
            alpha = (-source.orbit_fields()[...,1].clamp_min(1e-30).log10()*self.trap_scale+self.phase).remainder(1)
            palette_name = "teal_copper"
        elif style == "3b1b_adaptive":
            values = source.styled_escape.clamp_min(0).sqrt()
            lower, upper = self._adaptive_limits(values, valid)
            alpha = ((values-lower)/(upper-lower)+self.phase).clamp(0, 1)
            palette_name = "3b1b"
        else:
            values = (source.styled_escape if style == "3b1b" else source.escape).clamp_min(0)
            if self.mapping == "sqrt":
                values = values.sqrt()
            elif self.mapping == "log":
                values = values.log1p()
            alpha = (values-self.lower)/(self.upper-self.lower)+self.phase
            alpha = alpha.remainder(1) if self.repeat else alpha.clamp(0, 1)
            palette_name = style
        palette = self.palettes[palette_name]
        position = alpha*(len(palette)-1)
        index = position.long().clamp(0, len(palette)-2)
        rgb = torch.lerp(palette[index], palette[index+1], (position-index)[...,None])
        source.samples.copy_(torch.where(valid[...,None], rgb, 0))

    def recolor(self, **updates):
        """Color cached samples, lazily adding orbit fields for stripes/traps."""
        self.set_colors(**updates)
        self.adaptive_pending = False
        source = self._grid if self.filter == "tent" and self._grid is not None else self
        self._color_samples(source)
        stream = wp.stream_from_torch(torch.cuda.current_stream(self.device))
        if source is not self:
            wp.launch(tent_resolve, dim=(self.height, self.width),
                      inputs=[source._samples, self.side, *self._grid_offset], outputs=[self._outputs[2]],
                      device=self.warp_device, stream=stream)
        elif self.side > 1:
            wp.launch(average_colors_kernel, dim=(self.height, self.width),
                      inputs=[self.side, self._samples], outputs=[self._outputs[2]],
                      device=self.warp_device, stream=stream)
        return self.image

    def escape_time_image(self):
        """Resolve smooth escape time and coverage without palette dependence."""
        grid = self._grid if self.filter == "tent" and self._grid is not None else self
        valid = (grid.escape != -1).float()
        packed = torch.stack((grid.escape.clamp_min(0)*valid, valid, torch.zeros_like(valid)), dim=-1)
        if grid is not self:
            resolved = torch.empty_like(self.image)
            wp.launch(tent_resolve, dim=(self.height,self.width),
                      inputs=[wp.from_torch(packed),self.side,*self._grid_offset],
                      outputs=[wp.from_torch(resolved)],device=self.warp_device,
                      stream=wp.stream_from_torch(torch.cuda.current_stream(self.device)))
        else:
            resolved = torch.nn.functional.avg_pool2d(packed.permute(2,0,1)[None],self.side)[0].permute(1,2,0)
        coverage = resolved[...,1]
        return resolved[...,0]/coverage.clamp_min(1e-8), coverage

    def save(self, path):
        import matplotlib.image as mpimg

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        mpimg.imsave(path, self.image.cpu().numpy())
        return path


class GrayscaleRenderer:
    """Reusable float32 [H,W] coverage: black unescaped, white escaped.

    This is a finite-budget approximation, not a proof of membership. Gray
    pixels average black/white subpixel samples; no escape-time palette is used.
    render() reuses the image buffer and retains the latest view for navigation.
    """

    def __init__(self, width=64, height=64, device="cuda", side=4, view=None):
        if type(side) is not int or not 1 <= side <= 16:
            raise ValueError("AA side must be an integer from 1 to 16")
        if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive integers")
        self.device, self.warp_device = renderer_devices(device)
        self.width, self.height = width, height
        self.view = View() if view is None else view
        self.side = side
        self.image = torch.empty((height, width), device=self.device)
        self._output = wp.from_torch(self.image)
        self.samples = torch.empty((height * side, width * side), device=self.device)
        self._samples = wp.from_torch(self.samples)

    def minimum_span(self, view):
        # Leave 32 representable coordinate steps per subpixel sample.
        extent = max(abs(view.center.real), abs(view.center.imag), view.span, 1.0)
        return 32 * math.ulp(extent) * self.width * self.side

    def validate_view(self, view):
        if view.span < self.minimum_span(view):
            raise ValueError("View exceeds the float64 precision margin; use a reference-orbit renderer for deeper zooms")

    def render(self, view=None):
        view = self.view if view is None else view
        self.validate_view(view)
        stream = wp.stream_from_torch(torch.cuda.current_stream(self.device))
        wp.launch(escape_mask_kernel, dim=(self.height*self.side, self.width*self.side),
                  inputs=[view.center.real, view.center.imag, view.span,
                          view.z0.real, view.z0.imag, view.iterations],
                  outputs=[self._samples], device=self.warp_device, stream=stream)
        wp.launch(average_samples_kernel, dim=(self.height, self.width),
                  inputs=[self.side, self._samples], outputs=[self._output],
                  device=self.warp_device, stream=stream)
        self.view = view
        return self.image

    def save(self, path):
        import matplotlib.image as mpimg
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        mpimg.imsave(path, self.image.cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--center", nargs=2, default=("-0.5", "0.0"), metavar=("REAL", "IMAG"))
    parser.add_argument("--span", type=float, default=3.5, help="Horizontal extent in the complex plane")
    parser.add_argument("--iterations", type=int, default=512)
    parser.add_argument("--z0", type=float, nargs=2, default=(0.0, 0.0), metavar=("REAL", "IMAG"))
    parser.add_argument("--device", default="cuda", help="cuda or cuda:N")
    parser.add_argument("--bw", action="store_true", help="Render antialiased black-and-white coverage")
    parser.add_argument("--palette", choices=tuple(ColorRenderer.styles), default="cosine")
    parser.add_argument("--aa-side", type=int, default=2, help="Subpixel grid side (2 gives four samples per pixel)")
    parser.add_argument("--filter", choices=("tent", "box"), default="tent",
                        help="Color reconstruction: anchored tent filtering or view-aligned box averaging")
    parser.add_argument("--headless", action="store_true", help="Render one PNG without opening a window")
    parser.add_argument("--watch", action="store_true", help="Restart the viewer after Python source edits")
    parser.add_argument("--output", default="output/mandelbrot.png")
    args = parser.parse_args()
    if args.watch:
        if args.headless:
            parser.error("--watch is for the interactive viewer")
        from viewer import watch
        watch()
        return
    try:
        real, imag = args.center
        view = View(f"({real}{'' if imag.startswith('-') else '+'}{imag}j)", args.span, args.iterations, complex(*args.z0))
        if args.bw:
            renderer = GrayscaleRenderer(args.width, args.height, args.device, args.aa_side, view)
        else:
            renderer = ColorRenderer(args.width, args.height, args.device, view, side=args.aa_side, filter=args.filter)
            renderer.style = args.palette
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    if args.headless:
        renderer.render()
        print(f"Saved {renderer.save(args.output)} ({renderer.device})")
    else:
        import signal
        def stop_viewer(_signum, _frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, stop_viewer)
        try:
            Viewer(renderer, args.output).show()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
