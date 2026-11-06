"""Warp Mandelbrot renderer with reusable PyTorch buffers and a small viewer."""

import argparse
from dataclasses import dataclass, replace
import math
from pathlib import Path
import time

import torch
import warp as wp


@wp.kernel(enable_backward=False)
def mandelbrot_kernel(
    center_x: wp.float64,
    center_y: wp.float64,
    span: wp.float64,
    z0_x: wp.float64,
    z0_y: wp.float64,
    max_iterations: int,
    counts: wp.array2d(dtype=wp.int32),
    escape: wp.array2d(dtype=wp.float32),
    image: wp.array3d(dtype=wp.float32),
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
    r = float(0.0)
    g = float(0.0)
    b = float(0.0)
    if radius_squared > wp.float64(4.0):
        # Continuous escape time: n + 1 - log2(log(|z|)).
        smooth = float(wp.float64(n + 1) - wp.log(wp.float64(0.5) * wp.log(radius_squared)) / wp.log(wp.float64(2.0)))
        phase = float(0.15) * smooth
        r = float(0.5) + float(0.5) * wp.cos(phase)
        g = float(0.5) + float(0.5) * wp.cos(phase + float(2.1))
        b = float(0.5) + float(0.5) * wp.cos(phase + float(4.2))
    escape[y, x] = smooth
    image[y, x, 0] = r
    image[y, x, 1] = g
    image[y, x, 2] = b


@dataclass(frozen=True)
class View:
    center: complex = -0.5 + 0j
    span: float = 3.5  # Horizontal extent in the complex plane.
    iterations: int = 512
    z0: complex = 0j

    def __post_init__(self):
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


class Mandelbrot:
    """Initialize once; render views into the same buffers on CPU or CUDA.

    render() returns float32 RGB [height, width, 3] in [0, 1]. The counts and
    escape attributes contain int32 iteration counts and float32 smooth escape
    times. Unescaped pixels have escape = -1 and RGB = 0. Buffers are reused:
    clone tensors when retaining multiple frames. This is a forward renderer.
    """

    def __init__(self, width=1024, height=768, device="auto", view=None):
        if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive integers")
        wp.init()
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() and wp.is_cuda_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Warp supports CPU or NVIDIA CUDA here; use --device cpu on a Mac")
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.warp_device = wp.device_from_torch(self.device)
        self.width, self.height = width, height
        self.view = view if view is not None else View()
        self.counts = torch.empty((height, width), dtype=torch.int32, device=self.device)
        self.escape = torch.empty((height, width), dtype=torch.float32, device=self.device)
        self.image = torch.empty((height, width, 3), dtype=torch.float32, device=self.device)
        # Wrap once: Warp and PyTorch share the same storage.
        self._outputs = [wp.from_torch(tensor) for tensor in (self.counts, self.escape, self.image)]

    def render(self, view=None):
        view = self.view if view is None else view
        stream = wp.stream_from_torch(torch.cuda.current_stream(self.device)) if self.device.type == "cuda" else None
        wp.launch(
            mandelbrot_kernel,
            dim=(self.height, self.width),
            inputs=[view.center.real, view.center.imag, view.span, view.z0.real, view.z0.imag, view.iterations],
            outputs=self._outputs,
            device=self.warp_device,
            stream=stream,
        )
        self.view = view
        return self.image

    def save(self, path):
        import matplotlib.image as mpimg

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        mpimg.imsave(path, self.image.cpu().numpy())
        return path


class Viewer:
    def __init__(self, renderer, output="output/mandelbrot.png"):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, TextBox

        self.renderer = renderer
        self.initial_view = renderer.view
        self.output = output
        self.drag = None
        with plt.rc_context({"toolbar": "None"}):
            self.figure, self.axes = plt.subplots(figsize=(11, 8))
        self.figure.canvas.manager.set_window_title("Mandelbrot · PyTorch + Warp")
        self.figure.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.22)
        self.axes.set_xlabel("Re(c)")
        self.axes.set_ylabel("Im(c)")
        self.artist = self.axes.imshow(
            torch.zeros((renderer.height, renderer.width, 3)).numpy(),
            origin="upper", interpolation="nearest", extent=renderer.view.extent(renderer.width, renderer.height),
        )
        self.status = self.figure.text(0.08, 0.025, "", fontsize=9)
        self.figure.text(0.08, 0.96, "Scroll to zoom · drag to pan · R reset · S save · Q quit", fontsize=11)
        self.fields = {}
        specs = (("center", "Center", 0.12, 0.22), ("span", "Span", 0.41, 0.12),
                 ("iterations", "Steps", 0.61, 0.10), ("z0", "z₀", 0.80, 0.16))
        for name, label, left, width in specs:
            field = TextBox(self.figure.add_axes((left, 0.125, width, 0.045)), label)
            field.on_submit(self.on_submit)
            self.fields[name] = field
        self.reset_button = Button(self.figure.add_axes((0.70, 0.055, 0.12, 0.04)), "Reset")
        self.save_button = Button(self.figure.add_axes((0.84, 0.055, 0.12, 0.04)), "Save PNG")
        self.reset_button.on_clicked(self.on_reset)
        self.save_button.on_clicked(self.on_save)
        for name, callback in (("scroll_event", self.on_scroll), ("button_press_event", self.on_press),
                               ("motion_notify_event", self.on_motion), ("button_release_event", self.on_release),
                               ("key_press_event", self.on_key)):
            self.figure.canvas.mpl_connect(name, callback)
        self.redraw()

    def redraw(self):
        start = time.perf_counter()
        image = self.renderer.render().cpu().numpy()
        elapsed = (time.perf_counter() - start) * 1000
        view = self.renderer.view
        extent = view.extent(self.renderer.width, self.renderer.height)
        self.artist.set_data(image)
        self.artist.set_extent(extent)
        self.axes.set_xlim(extent[:2])
        self.axes.set_ylim(extent[2:])
        self.axes.set_title(f"{self.renderer.device} · {self.renderer.width} × {self.renderer.height} · {elapsed:.1f} ms (render + copy)")
        for name, field in self.fields.items():
            field.eventson = False
            field.set_val(str(getattr(view, name)))
            field.eventson = True
        self.status.set_text(f"center={view.center}   span={view.span:.12g}   iterations={view.iterations}   z₀={view.z0}")
        self.figure.canvas.draw_idle()

    def on_submit(self, _text):
        try:
            view = View(center=complex(self.fields["center"].text), span=float(self.fields["span"].text),
                        iterations=int(self.fields["iterations"].text), z0=complex(self.fields["z0"].text))
        except ValueError as error:
            self.status.set_text(str(error))
            self.figure.canvas.draw_idle()
            return
        self.renderer.view = view
        self.redraw()

    def on_scroll(self, event):
        if event.inaxes != self.axes or self.drag is not None or self.figure.canvas.widgetlock.locked():
            return
        view = self.renderer.view
        anchor = complex(event.xdata, event.ydata)
        factor = 1.25 ** (-event.step)
        span = min(1e6, max(1e-14, view.span * factor))
        factor = span / view.span
        self.renderer.view = replace(view, center=anchor + (view.center - anchor) * factor, span=span)
        self.redraw()

    def on_press(self, event):
        if event.inaxes == self.axes and event.button == 1 and not self.figure.canvas.widgetlock.locked():
            self.drag = (event.x, event.y, self.renderer.view)

    def on_motion(self, event):
        if self.drag is None or event.inaxes != self.axes:
            return
        x, y, view = self.drag
        # Use display coordinates, which stay stable as the complex view moves.
        dx = (event.x - x) / self.axes.bbox.width * view.span
        dy = (event.y - y) / self.axes.bbox.height * view.span * self.renderer.height / self.renderer.width
        self.renderer.view = replace(view, center=view.center - complex(dx, dy))
        self.redraw()

    def on_release(self, _event):
        self.drag = None

    def on_reset(self, _event=None):
        self.drag = None
        self.renderer.view = self.initial_view
        self.redraw()

    def on_save(self, _event=None):
        path = self.renderer.save(self.output)
        self.status.set_text(f"Saved {path}")
        self.figure.canvas.draw_idle()

    def on_key(self, event):
        if any(field.capturekeystrokes for field in self.fields.values()):
            return
        if event.key == "r":
            self.on_reset()
        elif event.key == "s":
            self.on_save()
        elif event.key in ("q", "escape"):
            import matplotlib.pyplot as plt
            plt.close(self.figure)

    def show(self):
        import matplotlib.pyplot as plt
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--center", type=float, nargs=2, default=(-0.5, 0.0), metavar=("REAL", "IMAG"))
    parser.add_argument("--span", type=float, default=3.5, help="Horizontal extent in the complex plane")
    parser.add_argument("--iterations", type=int, default=512)
    parser.add_argument("--z0", type=float, nargs=2, default=(0.0, 0.0), metavar=("REAL", "IMAG"))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--headless", action="store_true", help="Render one PNG without opening a window")
    parser.add_argument("--output", default="output/mandelbrot.png")
    args = parser.parse_args()
    try:
        view = View(complex(*args.center), args.span, args.iterations, complex(*args.z0))
        renderer = Mandelbrot(args.width, args.height, args.device, view)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    if args.headless:
        renderer.render()
        print(f"Saved {renderer.save(args.output)} ({renderer.device})")
    else:
        import matplotlib
        matplotlib.use("QtAgg")
        Viewer(renderer, args.output).show()


if __name__ == "__main__":
    main()
