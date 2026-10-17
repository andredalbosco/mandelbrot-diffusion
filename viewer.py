"""Python Mandelbrot application using Warp's OpenGL viewer and ImGui controls."""

import ctypes
import hashlib
import json
import os
from pathlib import Path
import time

import torch
import warp as wp


def stop_process(process):
    """Close the whole viewer process group before starting its replacement."""
    import os
    import signal
    import subprocess

    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    except ProcessLookupError:
        process.wait()


def watch():
    """Watch our Python sources; own exactly one child and one watcher per display."""
    import fcntl
    import hashlib
    import os
    import signal
    import subprocess
    import sys
    import tempfile

    root = Path(__file__).resolve().parent
    identity = hashlib.sha256(f"{root}:{os.environ.get('DISPLAY', '')}".encode()).hexdigest()[:16]
    lock_path = Path(tempfile.gettempdir()) / f"mandelbrot-watch-{os.getuid()}-{identity}.lock"
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Mandelbrot is already being watched on this display.", flush=True)
            return
        paths = (root / "mandelbrot.py", root / "viewer.py", root / "deep_zoom.py", root / "mandelbrot_fast.py")
        command = [sys.executable, str(paths[0]), *(arg for arg in sys.argv[1:] if arg != "--watch")]
        stopping = False

        def stop(_signum, _frame):
            nonlocal stopping
            stopping = True

        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        previous = tuple(path.stat().st_mtime_ns for path in paths)
        process = subprocess.Popen(command, start_new_session=True)
        print("Watching Mandelbrot Python sources. Ctrl+C closes the viewer.", flush=True)
        try:
            while not stopping:
                time.sleep(.2)
                try:
                    current = tuple(path.stat().st_mtime_ns for path in paths)
                except FileNotFoundError:
                    continue  # Editors may briefly remove a file during atomic save.
                if current != previous:
                    stop_process(process)
                    if stopping:
                        break
                    process = subprocess.Popen(command, start_new_session=True)
                    previous = current
                    print("Source changed; replaced the previous viewer.", flush=True)
        finally:
            stop_process(process)


@wp.kernel(enable_backward=False)
def pack_pixels(image: wp.array3d(dtype=wp.float32), pixels: wp.array3d(dtype=wp.uint8)):
    y, x = wp.tid()
    for channel in range(3):
        pixels[y, x, channel] = wp.uint8(wp.clamp(image[y, x, channel], 0.0, 1.0) * 255.0 + 0.5)
    pixels[y, x, 3] = wp.uint8(255)


def zoom_view(view, anchor, factor, minimum_span, offset=None):
    span = min(1e6, max(minimum_span, view.span * factor))
    offset = anchor-view.center if offset is None else offset
    return view.shifted(offset*(1-span/view.span), span=span)


def pan_view(view, dx, dy, width):
    return view.shifted(-complex(dx, -dy) * view.span / width)


def escape_sobel(values, coverage, scale=500.):
    """Masked Sobel on smooth escape times; no artificial interior sentinel edge.

    Missing neighbors contribute the center value. A normalized binomial blur
    reduces single-pixel noise before differentiating. Manual scale is fixed;
    scale=None uses the 95th percentile for display contrast, not scoring.
    """
    from torch.nn import functional as F
    valid = (coverage > 1e-6).float()[None,None]
    x = values[None,None]
    blur = values.new_tensor([[1,2,1],[2,4,2],[1,2,1]])[None,None]/16
    weight = F.conv2d(F.pad(valid,(1,1,1,1),mode="replicate"),blur)
    smooth = F.conv2d(F.pad(x*valid,(1,1,1,1),mode="replicate"),blur)/weight.clamp_min(1e-8)
    neighbors = F.unfold(F.pad(smooth,(1,1,1,1),mode="replicate"),3).reshape(1,9,*values.shape)
    support = F.unfold(F.pad(valid,(1,1,1,1),mode="replicate"),3).reshape(1,9,*values.shape)
    delta = (neighbors-smooth)*support
    kernels = values.new_tensor([[-1,0,1,-2,0,2,-1,0,1],[-1,-2,-1,0,0,0,1,2,1]])/8
    gradients = torch.einsum("kc,bchw->bkhw",kernels,delta)
    magnitude = gradients.square().sum(1).sqrt()[0]*valid[0,0]
    if scale is None:
        escaped = magnitude[valid[0,0] > 0]
        scale = torch.quantile(escaped,.95).clamp_min(1e-3) if escaped.numel() else 1.
    brightness = -torch.expm1(-magnitude/scale)
    return brightness[...,None].expand(-1,-1,3).contiguous()


class ProgressiveRenderer:
    """One worker, one in-flight phase, and only the newest camera request.

    GPU work runs on a separate nonblocking CUDA stream. Cancellation is between
    phases; an already launched CUDA kernel must finish. OpenGL stays on the UI
    thread. Result images own their storage until presentation finishes.
    """
    def __init__(self, renderer):
        from concurrent.futures import ThreadPoolExecutor
        self.executor = ThreadPoolExecutor(max_workers=1,thread_name_prefix="mandelbrot")
        self.width,self.height,self.side=renderer.width,renderer.height,renderer.side
        self.device=renderer.device
        self.renderers={}
        self.future=None
        self.request=None
        self.generation=0
        self.phase=0
        self.profile={}
        self.adaptive_range = self.adaptive_time = None
        self.adaptive_style = None
        self.next_adaptation = 0.

    def submit(self, view, colors, gradient=False, scale=None):
        self.generation+=1
        self.request=(self.generation,view,dict(colors),gradient,scale)
        self.phase=0

    def _cached(self, renderer, view):
        from decimal import Decimal,localcontext
        if renderer._grid_key != (view.span,view.iterations,view.z0) or renderer._grid_origin is None:return False
        with localcontext() as ctx:
            ctx.prec=50
            dx=float((Decimal(view.center_exact[0])-Decimal(renderer._grid_anchor.center_exact[0]))/renderer._grid_step)-renderer._grid_origin[0]
            dy=-float((Decimal(view.center_exact[1])-Decimal(renderer._grid_anchor.center_exact[1]))/renderer._grid_step)-renderer._grid_origin[1]
        return abs(dx)<=(renderer._grid.width-renderer.sample_width)/2-renderer.side and abs(dy)<=(renderer._grid.height-renderer.sample_height)/2-renderer.side

    def _work(self, request, phase):
        from mandelbrot import ColorRenderer
        generation,view,colors,gradient,scale=request
        torch.cuda.set_device(self.device)
        if not hasattr(self,"stream"):self.stream=torch.cuda.Stream(device=self.device)
        started=time.perf_counter()
        with torch.cuda.stream(self.stream):
            cache=self.renderers.get(2)
            if phase==0 and cache is not None and self._cached(cache,view):phase=2
            if phase not in self.renderers:
                width=max(8,self.width//4) if phase==0 else self.width
                height=max(8,self.height//4) if phase==0 else self.height
                self.renderers[phase]=ColorRenderer(width,height,device=str(self.device),side=1 if phase==0 else self.side,filter="tent",pan_margin=phase==2)
            renderer=self.renderers[phase]
            renderer.set_colors(**colors)
            if renderer.style == "3b1b_adaptive" and self.adaptive_style == renderer.style:
                renderer._adaptive_range = self.adaptive_range
                renderer._adaptive_time = self.adaptive_time
            else:
                renderer._adaptive_range = renderer._adaptive_time = None
            self.adaptive_style = renderer.style
            previous_backend=getattr(renderer._grid,"_tiled_backend",None)
            image=renderer.render(view)
            self.adaptive_range, self.adaptive_time = renderer._adaptive_range, renderer._adaptive_time
            if gradient:image=escape_sobel(*renderer.escape_time_image(),scale=scale)
            image=torch.nn.functional.interpolate(image.permute(2,0,1)[None],size=(self.height,self.width),mode="nearest")[0].permute(1,2,0).contiguous().clone()
            self.stream.synchronize()
            backend=getattr(renderer._grid,"_tiled_backend",None)
            profile={"phase":("preview","visible","cache")[phase],"milliseconds":(time.perf_counter()-started)*1000,"precision":renderer.precision}
            profile["adaptive_pending"] = renderer.adaptive_pending and not gradient
            if renderer.style == "3b1b_adaptive":
                profile["color_range"] = renderer._adaptive_range.tolist()
            profile["cache_hit"] = backend is not None and backend is previous_backend
            if backend is not None and backend is not previous_backend:
                profile.update(backend.diagnostics())
        return generation,view,phase,image,profile

    def poll(self):
        result=None
        if self.future is not None and self.future.done():
            try:
                value=self.future.result()
            except Exception:
                self.future=None
                self.request=None
                raise
            self.future=None
            if value[0]==self.generation:
                result=value
                self.profile=value[4]
                self.phase=value[2]+1
        if self.future is None and self.request is not None:
            if self.phase < 3:
                self.future=self.executor.submit(self._work,self.request,self.phase)
            elif self.profile.get("adaptive_pending") and time.monotonic() >= self.next_adaptation:
                # Ease the mapping after the camera stops, using cached samples.
                self.next_adaptation = time.monotonic()+1/30
                self.future=self.executor.submit(self._work,self.request,2)
        return result

    def close(self):
        self.executor.shutdown(wait=True,cancel_futures=True)


class Viewer:
    def __init__(self, renderer, output="output/mandelbrot.png"):
        import imgui
        from imgui.integrations.pyglet import PygletProgrammablePipelineRenderer
        from pyglet import gl
        from pyglet.display import get_display
        from warp.render import OpenGLRenderer

        self.imgui, self.gl = imgui, gl
        self.renderer, self.output = renderer, output
        self.initial_view = renderer.view
        # Allocate the drawable at desktop size before restoring/maximizing.
        # X11 PRIME may retain a smaller initial backing extent after growth.
        screen = get_display().get_default_screen()
        self.window_renderer = OpenGLRenderer(
            title="Mandelbrot · Warp OpenGL", screen_width=screen.width, screen_height=screen.height,
            draw_grid=False, draw_sky=False, draw_axis=False, show_info=False,
            enable_mouse_interaction=False, enable_keyboard_interaction=False,
            vsync=True, device=str(renderer.device),
        )
        self.context = imgui.create_context()
        imgui.style_colors_dark()
        imgui.get_io().ini_file_name = None
        self.ui_scale = self.window_renderer.window.get_pixel_ratio()
        window = self.window_renderer.window
        font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
        if font.exists():
            imgui.get_io().fonts.add_font_from_file_ttf(str(font), 16*self.ui_scale)
        else:
            imgui.get_io().font_global_scale = 1.25*self.ui_scale
        style = imgui.get_style()
        for name in ("window_padding", "frame_padding", "item_spacing", "item_inner_spacing"):
            value = getattr(style, name)
            setattr(style, name, (value.x*self.ui_scale, value.y*self.ui_scale))
        for name in ("indent_spacing", "scrollbar_size", "grab_min_size"):
            setattr(style, name, getattr(style, name)*self.ui_scale)
        self.ui = PygletProgrammablePipelineRenderer(self.window_renderer.window)
        self.window_renderer.render_2d_callbacks.append(self.draw_ui)
        self.window_renderer.window.push_handlers(on_close=self.request_close, on_key_press=self.on_key_press)
        self.texture = gl.GLuint()
        gl.glGenTextures(1, ctypes.byref(self.texture))
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_BORDER)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_BORDER)
        gl.glTexParameterfv(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_BORDER_COLOR, (gl.GLfloat*4)(.08,.10,.12,1.))
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, renderer.width, renderer.height,
                        0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
        self.pixel_buffer = gl.GLuint()
        gl.glGenBuffers(1, ctypes.byref(self.pixel_buffer))
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, self.pixel_buffer)
        gl.glBufferData(gl.GL_PIXEL_UNPACK_BUFFER, renderer.width * renderer.height * 4, None, gl.GL_DYNAMIC_DRAW)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
        self.interop = wp.RegisteredGLBuffer(self.pixel_buffer.value, device=str(renderer.device),
                                            flags=wp.RegisteredGLBuffer.WRITE_DISCARD,
                                            fallback_to_copy=False)
        vendor = gl.gl_info.get_renderer()
        print(f"OpenGL: {vendor}; CUDA–OpenGL image transfer enabled", flush=True)
        self.render_needed, self.recolor_needed = True, False
        self.dragging = False
        self.render_ms = 0.0
        self.last_frame = time.perf_counter()
        self.status = "Scroll to zoom · drag to pan"
        self.fields = {}
        self.gradient_mode = False
        self.gradient_scale = 500.
        self.gradient_auto_contrast = True
        identity = hashlib.sha256(f"{Path(__file__).resolve()}:{os.environ.get('DISPLAY', '')}".encode()).hexdigest()[:16]
        self.state_path = Path(os.environ.get("XDG_STATE_HOME", Path.home()/".local/state")) / "mandelbrot-diffusion" / f"viewer-{identity}.json"
        self.load_state()
        self.sync_fields()
        self.window_renderer.window.maximize()
        self.progressive = ProgressiveRenderer(renderer) if hasattr(renderer,"escape_time_image") else None
        self.display_view = renderer.view
        self.complete_frame = False
        self.phase_status = "Waiting for preview"

    def load_state(self):
        if not self.state_path.exists():
            return
        initial = self.renderer.view
        try:
            state = json.loads(self.state_path.read_text())
            display = state.get("display", {})
            self.gradient_mode = bool(display.get("escape_gradient", False))
            self.gradient_auto_contrast = bool(display.get("auto_contrast", True))
            self.gradient_scale = max(.01, float(display.get("gradient_scale", 500.)))
            values = state["view"]
            view = type(initial)(center=values["center"], span=values["span"],
                                 iterations=values["iterations"], z0=complex(values["z0"]))
            # Populate the orbit buffers before restoring cached color controls.
            self.renderer.render(view)
            if self.renderer.image.ndim == 3:
                self.renderer.recolor(**state.get("color", {}))
            window = self.window_renderer.window
            width, height = state["window"]["size"]
            window.set_size(max(640, min(width, window.screen.width)),
                            max(480, min(height, window.screen.height)))
            x, y = state["window"]["position"]
            window.set_location(x, y)
            self.status = "Restored previous view and colors"
        except (ValueError, TypeError, KeyError, OSError) as error:
            self.renderer.view = initial
            self.status = f"Could not restore saved settings: {error}"

    def save_state(self):
        view = self.renderer.view
        window = self.window_renderer.window
        state = {"view": {"center": view.center_text, "span": view.span,
                          "iterations": view.iterations, "z0": str(view.z0)},
                 "window": {"position": window.get_location(), "size": window.get_size()}}
        state["display"] = {"escape_gradient": self.gradient_mode, "gradient_scale": self.gradient_scale, "auto_contrast": self.gradient_auto_contrast}
        if self.renderer.image.ndim == 3:
            state["color"] = self.renderer.color_settings()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(self.state_path)

    def sync_fields(self):
        view = self.renderer.view
        self.fields = {"center": view.center_text, "span": f"{view.span:.16g}",
                       "iterations": str(view.iterations), "z0": str(view.z0)}

    def set_view(self, view):
        self.renderer.view = view
        self.sync_fields()
        self.render_needed = True

    def reset_parameters(self):
        if self.renderer.image.ndim == 3:
            self.renderer.set_colors(**self.renderer.color_defaults)
        self.gradient_mode = False
        self.gradient_auto_contrast = True
        self.gradient_scale = 500.
        self.set_view(type(self.renderer.view)())
        self.status = "View, coloring, and gradient parameters reset"

    def update_image(self):
        start = time.perf_counter()
        if self.progressive is not None:
            if self.render_needed or self.recolor_needed:
                colors=self.renderer.color_settings()
                self.progressive.submit(self.renderer.view,colors,self.gradient_mode,None if self.gradient_auto_contrast else self.gradient_scale)
                self.render_needed=self.recolor_needed=False
                self.complete_frame=False
                self.phase_status="Rendering preview…"
            try:result=self.progressive.poll()
            except Exception as error:
                self.status=f"Render failed: {error}"
                return
            if result is None:return
            _,self.display_view,phase,image,profile=result
            self.renderer.image.copy_(image)
            self.renderer.precision=profile["precision"]
            self.complete_frame=phase>=1
            self.phase_status=f"{profile['phase']} · {profile['milliseconds']:.1f} ms"
            self.render_ms=profile["milliseconds"]
            print("Render profile: "+json.dumps(profile),flush=True)
        else:
            if not self.render_needed and not self.recolor_needed:return
            image = self.renderer.render() if self.render_needed else self.renderer.recolor()
        if image.ndim == 2:
            image = image[:, :, None].expand(-1, -1, 3)
        # Match the PyTorch stream used by the renderer before GL maps the buffer.
        torch.cuda.current_stream(self.renderer.device).synchronize()
        pixels = self.interop.map(wp.uint8, (self.renderer.height, self.renderer.width, 4))
        try:
            wp.launch(pack_pixels, dim=(self.renderer.height, self.renderer.width),
                      inputs=[wp.from_torch(image)], outputs=[pixels], device=str(self.renderer.device))
            torch.cuda.current_stream(self.renderer.device).synchronize()
        finally:
            self.interop.unmap()
        gl = self.gl
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, self.pixel_buffer)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
        gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, self.renderer.width, self.renderer.height,
                          gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
        if self.progressive is None:self.render_ms = (time.perf_counter()-start) * 1000
        self.render_needed = self.recolor_needed = False

    def on_key_press(self, symbol, modifiers):
        from pyglet.window import key
        if symbol == key.G and not self.imgui.get_io().want_text_input and hasattr(self.renderer,"escape_time_image"):
            self.gradient_mode = not self.gradient_mode
            self.recolor_needed = True
            self.status = "Escape-time Sobel (G)" if self.gradient_mode else "Palette view (G)"

    def draw_image(self):
        imgui = self.imgui
        available = imgui.get_content_region_available()
        width = max(1, min(available.x, available.y * self.renderer.width / self.renderer.height))
        height = width * self.renderer.height / self.renderer.width
        # Reproject the last complete camera while a fresh result is pending.
        from decimal import Decimal,localcontext
        old=self.display_view
        new=self.renderer.view
        with localcontext() as ctx:
            ctx.prec=50
            dx=float((Decimal(new.center_exact[0])-Decimal(old.center_exact[0]))/Decimal(str(old.span)))
            dy=-float((Decimal(new.center_exact[1])-Decimal(old.center_exact[1]))/Decimal(str(old.span)))*self.renderer.width/self.renderer.height
        ratio=new.span/old.span
        imgui.image(self.texture.value,width,height,uv0=(.5+dx-ratio/2,.5+dy-ratio/2),uv1=(.5+dx+ratio/2,.5+dy+ratio/2))
        origin = imgui.get_item_rect_min()
        io = imgui.get_io()
        if imgui.is_item_hovered():
            if io.mouse_wheel:
                view = self.renderer.view
                x = (io.mouse_pos.x-origin.x) / width - .5
                y = .5 - (io.mouse_pos.y-origin.y) / height
                offset = complex(x, y*self.renderer.height/self.renderer.width)*view.span
                anchor = view.center + offset
                minimum = self.renderer.minimum_span(view)
                self.set_view(zoom_view(view, anchor, 1.25 ** (-max(-20, min(20, io.mouse_wheel))), minimum, offset))
            if imgui.is_mouse_clicked(0):
                self.dragging = True
        if self.dragging and imgui.is_mouse_down(0):
            if io.mouse_delta.x or io.mouse_delta.y:
                self.set_view(pan_view(self.renderer.view, io.mouse_delta.x, io.mouse_delta.y, width))
        else:
            self.dragging = False

    def draw_controls(self):
        imgui = self.imgui
        imgui.text("MANDELBROT")
        imgui.text_disabled("Warp · CUDA · OpenGL")
        if hasattr(self.renderer,"escape_time_image"):
            changed, enabled = imgui.checkbox("Escape-time Sobel [G]", self.gradient_mode)
            if changed:
                self.gradient_mode = enabled
                self.recolor_needed = True
            if self.gradient_mode:
                changed, automatic = imgui.checkbox("Auto gradient contrast", self.gradient_auto_contrast)
                if changed:
                    self.gradient_auto_contrast = automatic
                    self.recolor_needed = True
                changed, value = imgui.input_float("Gradient scale", self.gradient_scale, 1.,10.)
                if changed:
                    self.gradient_scale = max(.01,value)
                    self.recolor_needed = True
                imgui.text_wrapped("White: stronger escape-time variation. Black: low variation or unescaped.")
        imgui.separator()
        for name, label in (("center", "Center"), ("span", "Span"),
                            ("iterations", "Iterations"), ("z0", "Initial z")):
            _, self.fields[name] = imgui.input_text(label, self.fields[name], 128)
        if imgui.button("Apply view"):
            try:
                view = type(self.renderer.view)(self.fields["center"], float(self.fields["span"]),
                                                int(self.fields["iterations"]), complex(self.fields["z0"]))
                if hasattr(self.renderer, "validate_view"):
                    self.renderer.validate_view(view)
                self.set_view(view)
                self.status = "View updated"
            except ValueError as error:
                self.status = str(error)
        imgui.same_line()
        if imgui.button("Reset view"):
            self.set_view(self.initial_view)
        if self.renderer.image.ndim == 3:
            self.draw_palette()
        else:
            imgui.text(f"Grayscale · {self.renderer.side**2} samples/pixel")
        imgui.separator()
        if imgui.button("Reset all parameters"):
            self.reset_parameters()
        if imgui.button("Save PNG"):
            if self.progressive is not None and not self.complete_frame:
                self.status="Wait for the visible render before saving"
            else:
                self.renderer.save(self.output)
                self.status = f"Saved {self.output}"
        imgui.text_wrapped(self.phase_status)
        imgui.text(f"Image update: {self.render_ms:.1f} ms")
        imgui.text(f"UI: {imgui.get_io().framerate:.0f} fps")
        imgui.text(f"{self.renderer.width} x {self.renderer.height} · {getattr(self.renderer, 'side', 1)**2} samples/pixel")
        if hasattr(self.renderer, "filter"):
            imgui.text(f"Filter: {self.renderer.filter}")
        imgui.text_wrapped(f"Precision: {getattr(self.renderer, 'precision', 'float64')}")
        imgui.text_wrapped(self.status)

    def draw_palette(self):
        imgui, renderer = self.imgui, self.renderer
        imgui.separator()
        imgui.text("Coloring scheme")
        imgui.push_item_width(-1)
        updates = {}
        names = list(renderer.styles)
        changed, index = imgui.combo("##scheme", names.index(renderer.style), list(renderer.styles.values()))
        if changed:
            updates["style"] = names[index]
        style = updates.get("style", renderer.style)

        def slider(label, name, low, high):
            imgui.text(label)
            changed, value = imgui.slider_float("##"+name, getattr(renderer, name), low, high)
            if changed:
                updates[name] = value

        slider("Phase", "phase", 0., 1.)
        if style == "cosine":
            slider("Period", "period", 2., 200.)
        elif style == "smooth":
            slider("Color cycles / log escape", "smooth_scale", .05, 2.)
        elif style == "histogram":
            slider("Equalization strength", "histogram_strength", 0., 1.)
            imgui.text_wrapped("Spreads the visible escape times across the palette.")
        elif style == "stripes":
            slider("Stripe contrast", "stripe_contrast", .1, 4.)
            imgui.text_wrapped("Five post-escape stripe octaves. Fine stripes may need more samples.")
        elif style == "orbit_trap":
            slider("Color cycles / distance decade", "trap_scale", .02, 1.)
            imgui.text_wrapped("Closest orbit approach to the real or imaginary axis.")
        elif style == "3b1b_adaptive":
            slider("Clip each tail (%)", "adaptive_clip", 0., 20.)
            slider("Adaptation time (seconds)", "adaptive_seconds", 0., 2.)
            imgui.text_wrapped("3b1b colors fit to visible escape times. Zero seconds adapts immediately.")
        else:
            imgui.text("Escape mapping")
            choices = ["linear", "sqrt", "log"]
            changed, index = imgui.combo("##mapping", choices.index(renderer.mapping), choices)
            if changed:
                updates["mapping"] = choices[index]
            for label, name in (("Range min", "lower"), ("Range max", "upper")):
                imgui.text(label)
                changed, value = imgui.input_float("##"+name, getattr(renderer, name), .1, 1.)
                if changed:
                    updates[name] = value
            changed, value = imgui.checkbox("Repeat palette", renderer.repeat)
            if changed:
                updates["repeat"] = value
        imgui.pop_item_width()
        if updates:
            try:
                renderer.set_colors(**updates)
                self.recolor_needed = True
                self.status = "Palette updated"
            except ValueError as error:
                self.status = str(error)
        if imgui.button("Reset coloring"):
            renderer.reset_colors()
            self.recolor_needed = True
            self.status = "Selected coloring scheme reset"

    def draw_ui(self):
        imgui = self.imgui
        self.update_image()
        io = imgui.get_io()
        io.display_size = self.window_renderer.window.get_size()
        framebuffer = self.window_renderer.window.get_framebuffer_size()
        # Pyglet's DPI ratio can differ from actual framebuffer/window sizes on X11.
        io.display_fb_scale = (framebuffer[0]/io.display_size.x, framebuffer[1]/io.display_size.y)
        now = time.perf_counter()
        io.delta_time = max(1e-6, now-self.last_frame)
        self.last_frame = now
        imgui.new_frame()
        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(*io.display_size)
        flags = imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_SAVED_SETTINGS
        imgui.begin("Mandelbrot", flags=flags)
        imgui.columns(2)
        imgui.set_column_width(0, max(100, io.display_size.x - 340*self.ui_scale))
        self.draw_image()
        imgui.next_column()
        imgui.begin_child("Controls", 0, 0)
        self.draw_controls()
        imgui.end_child()
        imgui.columns(1)
        imgui.end()
        imgui.render()
        self.ui.render(imgui.get_draw_data())

    def show(self):
        try:
            while self.window_renderer.is_running():
                started = time.perf_counter()
                self.window_renderer.begin_frame()
                self.window_renderer.end_frame()
                time.sleep(max(0, 1/60 - (time.perf_counter()-started)))
        finally:
            self.close()

    def request_close(self):
        from pyglet.event import EVENT_HANDLED
        self.window_renderer.app.event_loop.exit()
        return EVENT_HANDLED

    def close(self):
        self.save_state()
        if self.progressive is not None:self.progressive.close()
        self.window_renderer.render_2d_callbacks.remove(self.draw_ui)
        del self.interop
        self.gl.glDeleteBuffers(1, ctypes.byref(self.pixel_buffer))
        self.gl.glDeleteTextures(1, ctypes.byref(self.texture))
        self.ui.shutdown()
        self.imgui.destroy_context(self.context)
        self.window_renderer.close()
