"""Batched Mandelbrot renderer for generation experiments, Python + Warp.

Default cosine coloring, optional orbit fields and sample recoloring, z0=0,
and tent reconstruction with a one-pixel halo. GPU output buffers are reused.
The direct path deliberately rejects zooms beyond its coordinate precision.
"""

from decimal import Decimal, localcontext
from typing import Any

import torch
import warp as wp

from deep_zoom import dd_add, dd_mul


@wp.func
def iterate(cx: Any, cy: Any, limit: int):
    zx, zy = type(cx)(0.0), type(cx)(0.0)
    xx, yy = type(cx)(0.0), type(cx)(0.0)
    n = int(0)
    dx = cx-type(cx)(0.25)
    q = dx*dx+cy*cy
    if q*(q+dx) <= type(cx)(0.25)*cy*cy or (cx+type(cx)(1.0))*(cx+type(cx)(1.0))+cy*cy <= type(cx)(0.0625):
        n = limit
    while n < limit and xx+yy <= type(cx)(4.0):
        zy = type(cx)(2.0)*zx*zy+cy
        zx = xx-yy+cx
        xx, yy = zx*zx, zy*zy
        n += 1
    smooth = float(-1.0)
    if xx+yy > type(cx)(4.0):
        refined = n
        for _ in range(3):
            if xx+yy < type(cx)(1e15):
                zy = type(cx)(2.0)*zx*zy+cy
                zx = xx-yy+cx
                xx, yy = zx*zx, zy*zy
                refined += 1
        smooth = float(type(cx)(refined+1)-wp.log(type(cx)(0.5)*wp.log(xx+yy))/type(cx)(0.6931471805599453))
    return wp.vec2(float(n), smooth)


@wp.func
def color(smooth: float):
    value = wp.vec3(0.0)
    if smooth != -1.0:
        phase = smooth*0.15
        value = wp.vec3(0.5+0.5*wp.cos(phase), 0.5+0.5*wp.cos(phase+2.1), 0.5+0.5*wp.cos(phase+4.2))
    return value


@wp.kernel(enable_backward=False)
def direct_kernel(views: wp.array2d(dtype=Any), limits: wp.array(dtype=int),
                  side: int, width: int, counts: wp.array3d(dtype=int),
                  samples: wp.array3d(dtype=wp.vec3), escape: wp.array3d(dtype=float)):
    b, y, x = wp.tid()
    step = views[b, 2]/type(views[b, 0])(width*side)
    cx = views[b, 0]+(type(step)(x)+type(step)(0.5)-type(step)(counts.shape[2])*type(step)(0.5))*step
    cy = views[b, 1]+(type(step)(counts.shape[1])*type(step)(0.5)-type(step)(y)-type(step)(0.5))*step
    result = iterate(cx, cy, limits[b])
    counts[b, y, x] = int(result[0])
    samples[b, y, x] = color(result[1])
    escape[b, y, x] = result[1]


@wp.kernel(enable_backward=False)
def recolor_kernel(escape: wp.array3d(dtype=float), palette: wp.array(dtype=wp.vec3),
                   coordinates: wp.array3d(dtype=float), samples: wp.array3d(dtype=wp.vec3)):
    b, y, x = wp.tid()
    smooth = escape[b, y, x]
    rgb = wp.vec3(0.0)
    if smooth != -1.0:
        alpha = wp.clamp(coordinates[b, y, x], 0.0, 1.0)
        position = alpha * float(palette.shape[0] - 1)
        index = wp.min(int(position), palette.shape[0] - 2)
        rgb = wp.lerp(palette[index], palette[index + 1], position - float(index))
    samples[b, y, x] = rgb


@wp.kernel(enable_backward=False)
def orbit_coloring_kernel(views: wp.array2d(dtype=wp.float64), side: int, width: int,
                          counts: wp.array3d(dtype=int), escape: wp.array3d(dtype=float),
                          fields: wp.array3d(dtype=wp.vec2)):
    """Cross trap over z1..escape; mathr's five-octave post-escape stripes."""
    b, y, x = wp.tid()
    fields[b, y, x] = wp.vec2(0.0)
    if escape[b, y, x] == -1.0:
        return
    step = views[b, 2] / wp.float64(width * side)
    cx = views[b, 0] + (wp.float64(x) + wp.float64(.5) - wp.float64(counts.shape[2]) * wp.float64(.5)) * step
    cy = views[b, 1] + (wp.float64(counts.shape[1]) * wp.float64(.5) - wp.float64(y) - wp.float64(.5)) * step
    zx, zy = wp.float64(0.0), wp.float64(0.0)
    xx, yy = wp.float64(0.0), wp.float64(0.0)
    trap = wp.float64(1e30)
    for _ in range(counts[b, y, x]):
        zy = wp.float64(2.0)*zx*zy+cy
        zx = xx-yy+cx
        xx, yy = zx*zx, zy*zy
        trap = wp.min(trap, wp.min(wp.abs(zx), wp.abs(zy)))
    for _ in range(3):
        if xx+yy < wp.float64(1e15):
            zy = wp.float64(2.0)*zx*zy+cy
            zx = xx-yy+cx
            xx, yy = zx*zx, zy*zy
    radius = wp.sqrt(zx*zx+zy*zy)
    zx, zy = zx/radius, zy/radius
    f = wp.float64(escape[b, y, x] - wp.floor(escape[b, y, x]))
    weight = wp.float64(1.0) - f
    total = wp.float64(0.0)
    for _ in range(5):
        total += weight*zy
        weight = wp.float64(1.0)
        zx, zy = zx*zx-zy*zy, wp.float64(2.0)*zx*zy
    total = (total+f*zy)/wp.float64(5.0)
    fields[b, y, x] = wp.vec2(wp.clamp(float(wp.float64(.5)+wp.float64(.5)*total), 0.0, 1.0), float(trap))


@wp.kernel(enable_backward=False)
def resolve_kernel(side: int, samples: wp.array3d(dtype=wp.vec3), image: wp.array3d(dtype=wp.vec3)):
    b, y, x = wp.tid()
    sx = float(side)+(float(x)+0.5)*float(side)-0.5
    sy = float(side)+(float(y)+0.5)*float(side)-0.5
    ix, iy = int(wp.floor(sx)), int(wp.floor(sy))
    total = wp.vec3(0.0)
    weight_sum = float(0.0)
    for j in range(iy-side+1, iy+side+1):
        wy = wp.max(0.0, 1.0-wp.abs(float(j)-sy)/float(side))
        for i in range(ix-side+1, ix+side+1):
            weight = wy*wp.max(0.0, 1.0-wp.abs(float(i)-sx)/float(side))
            total += weight*samples[b, j, i]
            weight_sum += weight
    image[b, y, x] = total/weight_sum


class FastRenderer:
    """A reusable batch; render() returns CUDA float RGB [B,H,W,3].

    Direct and perturbation modes fill raw escape fields; perturbation also
    fills styled_escape. Hybrid mode does not expose either field. A caller
    owning reconstruction can disable the halo and use render(resolve=False),
    which returns raw colored samples instead of the resolved image.
    """

    def __init__(self, views, width=128, height=128, side=2, precision="float64", tile=32, guard="strict", compact=True, halo=True, device="cuda"):
        if precision not in ("float32", "float64", "hybrid32", "perturb32", "perturb64"):
            raise ValueError("Unknown precision")
        if not views or any(type(v) is not int for v in (width,height,side)) or min(width, height, side) < 1 or side > 16:
            raise ValueError("Need nonempty views and positive dimensions")
        self.width, self.height, self.side = width, height, side
        self.precision = precision
        if guard not in ("strict", "bailout"):
            raise ValueError("Unknown perturbation guard")
        self.guard = guard
        self.compact = compact
        device = torch.device(device)
        self.device = str(torch.device("cuda", torch.cuda.current_device()) if device.index is None else device)
        self.halo = bool(halo)
        wp.init()
        eps = 2**(-23 if precision == "float32" else -52)
        for view in views:
            if view.iterations > 65536:
                raise ValueError("Experimental renderer supports at most 65536 iterations")
            minimum = 32*(2**-100 if precision.startswith("perturb") else eps)*max(1., abs(view.center))*width*side
            if view.z0 != 0j or view.span < minimum:
                raise ValueError(f"View is outside the {precision} direct-coordinate margin")
        dtype = torch.float32 if precision == "float32" else torch.float64
        self.views = torch.tensor([(v.center.real, v.center.imag, v.span) for v in views], device=self.device, dtype=dtype)
        self.limits = torch.tensor([v.iterations for v in views], device=self.device, dtype=torch.int32)
        batch = len(views)
        self.counts = torch.empty((batch, height*side+2*side*int(self.halo), width*side+2*side*int(self.halo)), device=self.device, dtype=torch.int32)
        self.samples = torch.empty((*self.counts.shape, 3), device=self.device)
        self.escape = torch.empty_like(self.counts, dtype=torch.float32)
        self.styled_escape = torch.empty_like(self.escape)
        self._escape = wp.from_torch(self.escape)
        self._styled_escape = wp.from_torch(self.styled_escape)
        self.image = torch.empty((batch, height, width, 3), device=self.device)
        self._views = wp.from_torch(self.views)
        self._limits = wp.from_torch(self.limits)
        self._counts = wp.from_torch(self.counts)
        self._samples = wp.from_torch(self.samples, dtype=wp.vec3)
        self._image = wp.from_torch(self.image, dtype=wp.vec3)
        if precision.startswith("perturb") or precision == "hybrid32":
            self.reasons = wp.zeros(5,dtype=wp.int64,device=self.device)
            self.fallbacks = wp.zeros(batch, dtype=int, device=self.device)
            self.queue = wp.empty(self.counts.numel(), dtype=int, device=self.device)
            self.queue_length = wp.zeros(1, dtype=int, device=self.device)
        if precision.startswith("perturb"):
            if type(tile) is not int or tile < 1:
                raise ValueError("Tile side must be positive")
            self.tile = tile
            self.tiles_x = (self.counts.shape[2]+tile-1)//tile
            tiles_y = (self.counts.shape[1]+tile-1)//tile
            self.reference_limit = min(max(v.iterations for v in views), 65536)
            self.references = wp.empty((batch, tiles_y*self.tiles_x, self.reference_limit+1), dtype=wp.vec4d, device=self.device)
            self.lengths = wp.empty((batch, tiles_y*self.tiles_x), dtype=int, device=self.device)
            centers, steps = [], []
            with localcontext() as context:
                context.prec = 50
                def split(value):
                    high = float(value)
                    return (high, float(value-Decimal.from_float(high)))
                for view in views:
                    centers.append((*split(Decimal(view.center_exact[0])), *split(Decimal(view.center_exact[1]))))
                    steps.append(split(Decimal(str(view.span))/Decimal(width*side)))
            self.centers = wp.array(centers, dtype=wp.vec4d, device=self.device)
            self.steps = wp.array(steps, dtype=wp.vec2d, device=self.device)

    def render(self, resolve=True):
        if resolve and not self.halo:
            raise ValueError("Rendering without a filter halo requires resolve=False")
        stream = wp.stream_from_torch(torch.cuda.current_stream())
        # Memsets for counters must be ordered with the kernels on Torch's stream.
        with wp.ScopedStream(stream):
            return self._render(stream, resolve)

    def _render(self, stream, resolve=True):
        if self.precision.startswith("perturb"):
            self._timing_events = []
            def mark():
                event=torch.cuda.Event(enable_timing=True)
                event.record(torch.cuda.current_stream(self.device))
                self._timing_events.append(event)
            self.reasons.zero_()
            mark()
            wp.launch(tiled_reference_kernel, dim=self.lengths.shape,
                      inputs=[self.centers, self.steps, self._limits, self.tile, self.tiles_x,
                              self.counts.shape[2], self.counts.shape[1], self.reference_limit],
                      outputs=[self.references, self.lengths], device=self.device, stream=stream)
            mark()
            self.fallbacks.zero_()
            self.queue_length.zero_()
            marker = wp.float32(1) if self.precision == "perturb32" else wp.float64(0)
            wp.launch(tiled_perturbation_kernel, dim=self.counts.shape,
                      inputs=[self.references, self.lengths, self.steps, self._limits,
                              self.tile, self.tiles_x, marker, self.guard == "strict"],
                      outputs=[self._counts, self._samples, self._escape, self._styled_escape, self.fallbacks, self.queue, self.queue_length, self.reasons], device=self.device, stream=stream)
            mark()
            wp.launch(batched_dd_fallback, dim=self.counts.numel(),
                      inputs=[self.centers, self.steps, self._limits, self.queue, self.queue_length, self.compact], outputs=[self._counts, self._samples, self._escape, self._styled_escape],
                      device=self.device, stream=stream)
            mark()
        elif self.precision == "hybrid32":
            self.fallbacks.zero_()
            self.queue_length.zero_()
            wp.launch(hybrid_kernel, dim=self.counts.shape,
                      inputs=[self._views, self._limits, self.side, self.width],
                      outputs=[self._counts, self._samples, self.fallbacks, self.queue, self.queue_length],
                      device=self.device, stream=stream)
            wp.launch(fp64_fallback, dim=self.counts.numel(),
                      inputs=[self._views, self._limits, self.side, self.width, self.queue, self.queue_length],
                      outputs=[self._counts, self._samples], device=self.device, stream=stream)
        else:
            wp.launch(direct_kernel, dim=self.counts.shape,
                      inputs=[self._views, self._limits, self.side, self.width],
                      outputs=[self._counts, self._samples, self._escape], device=self.device, stream=stream)
        if resolve:
            wp.launch(resolve_kernel, dim=self.image.shape[:3], inputs=[self.side, self._samples],
                      outputs=[self._image], device=self.device, stream=stream)
        return self.image if resolve else self.samples

    def orbit_coloring(self):
        """Return per-sample (post-escape stripe, cross-trap distance) fields."""
        if self.precision != 'float64':
            raise ValueError('Orbit coloring currently requires direct float64 rendering')
        fields = torch.empty((*self.counts.shape, 2), device=self.device)
        wp.launch(orbit_coloring_kernel, dim=self.counts.shape,
                  inputs=[self._views, self.side, self.width, self._counts, self._escape],
                  outputs=[wp.from_torch(fields, dtype=wp.vec2)], device=self.device,
                  stream=wp.stream_from_torch(torch.cuda.current_stream(self.device)))
        return fields

    def recolor(self, palette, coordinates):
        """Apply an RGB lookup table to existing samples, then tent-filter RGB."""
        if self.precision == "hybrid32" or not self.halo:
            raise ValueError("Recoloring requires direct/perturbation samples and a filter halo")
        palette = torch.as_tensor(palette, dtype=torch.float32, device=self.device).contiguous()
        if palette.ndim != 2 or palette.shape[1] != 3 or len(palette) < 2:
            raise ValueError("Palette must be [N,3], with at least two colors")
        if not torch.isfinite(palette).all() or palette.min() < 0 or palette.max() > 1:
            raise ValueError("Palette colors must be finite and in [0,1]")
        coordinates = torch.as_tensor(coordinates, dtype=torch.float32, device=self.device).contiguous()
        if coordinates.shape != self.escape.shape or not torch.isfinite(coordinates).all():
            raise ValueError('Need one finite palette coordinate per spatial sample')
        stream = wp.stream_from_torch(torch.cuda.current_stream(self.device))
        wp.launch(recolor_kernel, dim=self.counts.shape,
                  inputs=[self._escape, wp.from_torch(palette, dtype=wp.vec3), wp.from_torch(coordinates)],
                  outputs=[self._samples], device=self.device, stream=stream)
        wp.launch(resolve_kernel, dim=self.image.shape[:3], inputs=[self.side, self._samples],
                  outputs=[self._image], device=self.device, stream=stream)
        return self.image


    def diagnostics(self):
        """Read completed stage events/counters; call outside the UI thread."""
        if not self.precision.startswith("perturb"):return {}
        events=self._timing_events
        reasons=self.reasons.numpy().tolist()
        failed=sum(reasons[:4])
        return {"gpu_ms":{name:events[i].elapsed_time(events[i+1]) for i,name in enumerate(("references","perturbation","dd_fallback"))},
                "fallback_reasons":dict(zip(("bailout_uncertainty","cancellation","roundoff_guard","reference_exhausted"),reasons[:4])),
                "fallback_fraction":failed/self.counts.numel(),
                "mean_iterations_before_fallback":reasons[4]/failed if failed else 0.}


@wp.kernel(enable_backward=False)
def hybrid_kernel(views: wp.array2d(dtype=wp.float64), limits: wp.array(dtype=int), side: int, width: int,
                  counts: wp.array3d(dtype=int), samples: wp.array3d(dtype=wp.vec3),
                  fallbacks: wp.array(dtype=int), queue: wp.array(dtype=int), queue_length: wp.array(dtype=int)):
    b, y, x = wp.tid()
    step = views[b, 2]/wp.float64(width*side)
    cx = views[b, 0]+(wp.float64(x)+wp.float64(.5)-wp.float64(counts.shape[2])*wp.float64(.5))*step
    cy = views[b, 1]+(wp.float64(counts.shape[1])*wp.float64(.5)-wp.float64(y)-wp.float64(.5))*step
    dx = cx-wp.float64(.25)
    q = dx*dx+cy*cy
    interior = q*(q+dx) < wp.float64(.25)*cy*cy or (cx+wp.float64(1))*(cx+wp.float64(1))+cy*cy < wp.float64(.0625)
    n = int(0)
    if interior:
        n = limits[b]
    fx, fy = float(cx), float(cy)
    coordinate_error = float(wp.abs(cx-wp.float64(fx))+wp.abs(cy-wp.float64(fy)))
    zx, zy = float(0.0), float(0.0)
    xx, yy = float(0.0), float(0.0)
    error = float(0.0)
    failed = bool(False)
    while n < limits[b]:
        r2 = xx+yy
        radius_error = (2.0*(wp.abs(zx)+wp.abs(zy))+2.0*error)*error+1e-6*(r2+1.0)
        if wp.abs(r2-4.0) <= radius_error or error > 1e-4*(wp.abs(zx)+wp.abs(zy)+1.0):
            failed = True
            break
        if r2 > 4.0:
            break
        error = (2.000002*wp.sqrt(r2)+error)*error+coordinate_error+1e-6*(r2+wp.abs(fx)+wp.abs(fy))
        zy = 2.0*zx*zy+fy
        zx = xx-yy+fx
        xx, yy = zx*zx, zy*zy
        n += 1
    # Check the final state too: the last iteration can be uncertain or escape.
    r2 = xx+yy
    radius_error = (2.0*(wp.abs(zx)+wp.abs(zy))+2.0*error)*error+1e-6*(r2+1.0)
    if not interior and (wp.abs(r2-4.0) <= radius_error or error > 1e-4*(wp.abs(zx)+wp.abs(zy)+1.0)):
        failed = True
    counts[b, y, x] = n
    if failed:
        counts[b, y, x] = -1
        wp.atomic_add(fallbacks, b, 1)
        index = wp.atomic_add(queue_length, 0, 1)
        queue[index] = (b*counts.shape[1]+y)*counts.shape[2]+x
    else:
        smooth = float(-1.0)
        if r2 > 4.0:
            rx, ry = wp.float64(zx), wp.float64(zy)
            radius = rx*rx+ry*ry
            refined = n
            for _ in range(3):
                if radius < wp.float64(1e100):
                    next_x = rx*rx-ry*ry+cx
                    ry = wp.float64(2)*rx*ry+cy
                    rx = next_x
                    radius = rx*rx+ry*ry
                    refined += 1
            smooth = float(wp.float64(refined+1)-wp.log(wp.float64(.5)*wp.log(radius))/wp.float64(.6931471805599453))
        samples[b, y, x] = color(smooth)


@wp.kernel(enable_backward=False)
def fp64_fallback(views: wp.array2d(dtype=wp.float64), limits: wp.array(dtype=int), side: int, width: int,
                  queue: wp.array(dtype=int), queue_length: wp.array(dtype=int),
                  counts: wp.array3d(dtype=int), samples: wp.array3d(dtype=wp.vec3)):
    index = wp.tid()
    if index >= queue_length[0]:
        return
    index = queue[index]
    x = index % counts.shape[2]
    y = (index//counts.shape[2]) % counts.shape[1]
    b = index//(counts.shape[2]*counts.shape[1])
    step = views[b, 2]/wp.float64(width*side)
    cx = views[b, 0]+(wp.float64(x)+wp.float64(.5)-wp.float64(counts.shape[2])*wp.float64(.5))*step
    cy = views[b, 1]+(wp.float64(counts.shape[1])*wp.float64(.5)-wp.float64(y)-wp.float64(.5))*step
    result = iterate(cx, cy, limits[b])
    counts[b, y, x] = int(result[0])
    samples[b, y, x] = color(result[1])


@wp.kernel(enable_backward=False, module="unique", module_options={"fuse_fp": False})
def tiled_reference_kernel(centers: wp.array(dtype=wp.vec4d), steps: wp.array(dtype=wp.vec2d),
                           limits: wp.array(dtype=int), tile: int, tiles_x: int,
                           width: int, height: int, cap: int,
                           orbit: wp.array3d(dtype=wp.vec4d), lengths: wp.array2d(dtype=int)):
    b, t = wp.tid()
    center = centers[b]
    ox = wp.float64((t % tiles_x)*tile)+wp.float64(tile)*wp.float64(.5)-wp.float64(width)*wp.float64(.5)
    oy = wp.float64(height)*wp.float64(.5)-wp.float64((t // tiles_x)*tile)-wp.float64(tile)*wp.float64(.5)
    cx = dd_add(wp.vec2d(center[0], center[1]), dd_mul(steps[b], wp.vec2d(ox, wp.float64(0.0))))
    cy = dd_add(wp.vec2d(center[2], center[3]), dd_mul(steps[b], wp.vec2d(oy, wp.float64(0.0))))
    zx, zy = wp.vec2d(0.0, 0.0), wp.vec2d(0.0, 0.0)
    xx, yy = dd_mul(zx, zx), dd_mul(zy, zy)
    n = int(0)
    orbit[b, t, 0] = wp.vec4d(0.0, 0.0, 0.0, 0.0)
    while n < wp.min(limits[b], cap) and dd_add(xx, yy)[0] < wp.float64(1e40):
        next_x = dd_add(dd_add(xx, -yy), cx)
        zy = dd_add(dd_mul(zx, zy)*wp.float64(2), cy)
        zx = next_x
        xx, yy = dd_mul(zx, zx), dd_mul(zy, zy)
        n += 1
        orbit[b, t, n] = wp.vec4d(zx[0], zx[1], zy[0], zy[1])
    lengths[b, t] = n


@wp.kernel(enable_backward=False)
def tiled_perturbation_kernel(orbit: wp.array3d(dtype=wp.vec4d), lengths: wp.array2d(dtype=int),
                              steps: wp.array(dtype=wp.vec2d), limits: wp.array(dtype=int),
                              tile: int, tiles_x: int, marker: Any, strict: bool,
                              counts: wp.array3d(dtype=int), samples: wp.array3d(dtype=wp.vec3),
                              escape: wp.array3d(dtype=float), styled_escape: wp.array3d(dtype=float),
                              fallbacks: wp.array(dtype=int), queue: wp.array(dtype=int), queue_length: wp.array(dtype=int), reasons: wp.array(dtype=wp.int64)):
    b, y, x = wp.tid()
    tx, ty = x//tile, y//tile
    t = ty*tiles_x+tx
    step = type(marker)(steps[b][0])
    dcx = (type(marker)(x % tile)+type(marker)(.5)-type(marker)(tile)*type(marker)(.5))*step
    dcy = (type(marker)(tile)*type(marker)(.5)-type(marker)(y % tile)-type(marker)(.5))*step
    dx, dy, error = type(marker)(0.0), type(marker)(0.0), type(marker)(0.0)
    eps, threshold = type(marker)(8e-15), type(marker)(1e-10)
    if marker > type(marker)(0.5):
        eps, threshold = type(marker)(1e-6), type(marker)(1e-5)
    n = int(0)
    failed = bool(False)
    reason = int(0)
    rx, ry, r2 = type(marker)(0.0), type(marker)(0.0), type(marker)(0.0)
    while n <= limits[b]:
        ref = orbit[b, t, n]
        ax = type(marker)(ref[0])+type(marker)(ref[1])
        ay = type(marker)(ref[2])+type(marker)(ref[3])
        rx, ry = ax+dx, ay+dy
        r2 = rx*rx+ry*ry
        norm = wp.abs(rx)+wp.abs(ry)
        tolerance = type(marker)(32.0)*(norm+type(marker)(1.0))*(error+eps)
        if wp.abs(r2-type(marker)(4.0)) <= tolerance:
            failed = True
            reason = 0
            break
        if r2 < type(marker)(1e-8)*(ax*ax+ay*ay):
            failed = True
            reason = 1
            break
        if strict and error > threshold*(norm+type(marker)(1e-12)):
            failed = True
            reason = 2
            break
        if r2 > type(marker)(4.0) or n == limits[b]:
            break
        if n >= lengths[b, t]:
            reason = 3
            failed = True
            break
        delta_norm = wp.abs(dx)+wp.abs(dy)
        growth = type(marker)(2.000002)*type(marker)(wp.sqrt(float(r2)))
        error = (growth+type(marker)(2.0)*error)*error + eps*(
            type(marker)(2.0)*(wp.abs(ax)+wp.abs(ay))*delta_norm+delta_norm*delta_norm+wp.abs(dcx)+wp.abs(dcy))
        next_dx = type(marker)(2.0)*(ax*dx-ay*dy)+dx*dx-dy*dy+dcx
        next_dy = type(marker)(2.0)*(ax*dy+ay*dx+dx*dy)+dcy
        dx, dy = next_dx, next_dy
        n += 1
    counts[b, y, x] = n
    if failed:
        wp.atomic_add(reasons,reason,wp.int64(1))
        wp.atomic_add(reasons,4,wp.int64(n))
        counts[b, y, x] = -n-1
        wp.atomic_add(fallbacks, b, 1)
        index = wp.atomic_add(queue_length, 0, 1)
        queue[index] = (b*counts.shape[1]+y)*counts.shape[2]+x
    else:
        smooth = float(-1.0)
        styled = float(-1.0)
        if r2 > type(marker)(4.0):
            # Once escaped, orbit continuation for smooth coloring can use
            # ordinary double coordinates. The deep offset is negligible here.
            ref = orbit[b, t, 1]
            cx, cy = ref[0]+ref[1]+wp.float64(dcx), ref[2]+ref[3]+wp.float64(dcy)
            styled = float(wp.float64(n-1)+wp.log(wp.float64(2.0))/(wp.float64(.5)*wp.log(wp.float64(r2)))+wp.float64(.5)*wp.sqrt(cx*cx+cy*cy))
            zx, zy = wp.float64(rx), wp.float64(ry)
            radius = zx*zx+zy*zy
            refined = n
            for _ in range(3):
                if radius < wp.float64(1e100):
                    next_x = zx*zx-zy*zy+cx
                    zy = wp.float64(2.0)*zx*zy+cy
                    zx = next_x
                    radius = zx*zx+zy*zy
                    refined += 1
            smooth = float(wp.float64(refined+1)-wp.log(wp.float64(.5)*wp.log(radius))/wp.float64(.6931471805599453))
        samples[b, y, x] = color(smooth)
        escape[b,y,x] = smooth
        styled_escape[b,y,x] = styled


@wp.kernel(enable_backward=False, module="unique", module_options={"fuse_fp": False})
def batched_dd_fallback(centers: wp.array(dtype=wp.vec4d), steps: wp.array(dtype=wp.vec2d),
                        limits: wp.array(dtype=int), queue: wp.array(dtype=int), queue_length: wp.array(dtype=int),
                        compact: bool, counts: wp.array3d(dtype=int),
                        samples: wp.array3d(dtype=wp.vec3), escape: wp.array3d(dtype=float),
                        styled_escape: wp.array3d(dtype=float)):
    index = wp.tid()
    if compact:
        if index >= queue_length[0]:
            return
        index = queue[index]
    x = index % counts.shape[2]
    y = (index // counts.shape[2]) % counts.shape[1]
    b = index // (counts.shape[2]*counts.shape[1])
    if counts[b, y, x] >= 0:
        return
    center = centers[b]
    ox = wp.float64(x)+wp.float64(.5)-wp.float64(counts.shape[2])*wp.float64(.5)
    oy = wp.float64(counts.shape[1])*wp.float64(.5)-wp.float64(y)-wp.float64(.5)
    cx = dd_add(wp.vec2d(center[0], center[1]), dd_mul(steps[b], wp.vec2d(ox, wp.float64(0.0))))
    cy = dd_add(wp.vec2d(center[2], center[3]), dd_mul(steps[b], wp.vec2d(oy, wp.float64(0.0))))
    zx, zy = wp.vec2d(0.0, 0.0), wp.vec2d(0.0, 0.0)
    xx, yy = dd_mul(zx, zx), dd_mul(zy, zy)
    radius = dd_add(xx, yy)
    n = int(0)
    while n < limits[b]:
        if radius[0] > wp.float64(4.0) or (radius[0] == wp.float64(4.0) and radius[1] > wp.float64(0.0)):
            break
        next_x = dd_add(dd_add(xx, -yy), cx)
        zy = dd_add(dd_mul(zx, zy)*wp.float64(2.0), cy)
        zx = next_x
        xx, yy = dd_mul(zx, zx), dd_mul(zy, zy)
        radius = dd_add(xx, yy)
        n += 1
    counts[b, y, x] = n
    smooth = float(-1.0)
    styled = float(-1.0)
    if radius[0] > wp.float64(4.0) or (radius[0] == wp.float64(4.0) and radius[1] > wp.float64(0.0)):
        rx, ry = zx[0]+zx[1], zy[0]+zy[1]
        r2 = rx*rx+ry*ry
        styled = float(wp.float64(n-1)+wp.log(wp.float64(2.0))/(wp.float64(.5)*wp.log(r2))+wp.float64(.5)*wp.sqrt(cx[0]*cx[0]+cy[0]*cy[0]))
        refined = n
        for _ in range(3):
            if r2 < wp.float64(1e100):
                next_rx = rx*rx-ry*ry+cx[0]
                ry = wp.float64(2.0)*rx*ry+cy[0]
                rx = next_rx
                r2 = rx*rx+ry*ry
                refined += 1
        smooth = float(wp.float64(refined+1)-wp.log(wp.float64(.5)*wp.log(r2))/wp.float64(.6931471805599453))
    samples[b, y, x] = color(smooth)
    escape[b,y,x] = smooth
    styled_escape[b,y,x] = styled


class DatasetRenderer:
    """Select a measured backend per scale; check every batch against FP64.

    Comparisons are finite-budget numerical checks, not fractal convergence
    proofs. Selection includes allocation and device-to-host image transfer.
    """

    def __init__(self, side=2, mode="auto", palette="cosine"):
        if palette not in ('cosine', 'smooth') or (palette == 'smooth' and mode == 'hybrid32'):
            raise ValueError("Smooth coloring requires a backend with escape samples")
        self.palette = palette
        self.side, self.mode = side, mode
        self.selected = {}
        self.report = {"selection": {}, "checks": [], "render_seconds": 0.,
                       "calibration_seconds": 0., "rgb_mae_limit": .0005,
                       "count_mismatch_limit": .001}

    @staticmethod
    def bucket(view):
        import math
        return round(math.log(3.5 / view.span, 4))

    def _make(self, views, mode):
        return FastRenderer(views, side=self.side, precision=mode, tile=64, guard="bailout")

    def _render(self, renderer):
        image = renderer.render()
        if self.palette == 'smooth':
            table = torch.tensor([[0,7,80],[20,70,180],[80,170,230],[235,250,255],
                                  [255,185,45],[130,45,10],[0,7,80]], device=renderer.escape.device)/255
            image = renderer.recolor(table, (renderer.escape.clamp_min(0).log1p()*.7).remainder(1))
        return image

    def _difference(self, renderer, expected_images, expected_counts):
        delta = (renderer.image - expected_images).abs().flatten(1)
        mismatch = (renderer.counts != expected_counts).flatten(1).float().mean(1)
        return {"rgb_mae": delta.mean(1).max().item(), "rgb_max": delta.max().item(),
                "count_mismatch_fraction": mismatch.max().item()}

    def _acceptable(self, error):
        return error["rgb_mae"] <= self.report["rgb_mae_limit"] and error["count_mismatch_fraction"] <= self.report["count_mismatch_limit"]

    def calibrate(self, views):
        import time
        import statistics
        started = time.perf_counter()
        groups = {}
        for view in views:
            groups.setdefault(self.bucket(view), []).append(view)
        for key, entries in sorted(groups.items()):
            # Spread checks across selected locations, parents and targets.
            entries = [entries[i] for i in torch.linspace(0, len(entries)-1, min(8, len(entries))).long().tolist()]
            reference = self._make(entries, "float64")
            expected_images = self._render(reference).clone()
            expected_counts = reference.counts.clone()
            del reference
            candidates = []
            modes = ("float64", "hybrid32", "perturb32", "perturb64") if self.mode == "auto" else tuple(dict.fromkeys(("float64", self.mode)))
            if self.palette == 'smooth':
                modes = tuple(m for m in modes if m != 'hybrid32')
            for mode in modes:
                renderer = self._make(entries, mode)
                self._render(renderer)
                error = self._difference(renderer, expected_images, expected_counts)
                del renderer
                times = []
                for _ in range(3):
                    torch.cuda.synchronize()
                    before = time.perf_counter()
                    renderer = self._make(entries, mode)
                    cpu = self._render(renderer).cpu()
                    times.append(time.perf_counter() - before)
                    del renderer, cpu
                candidates.append({"mode": mode, "seconds": statistics.median(times),
                                   "accepted": self._acceptable(error), **error})
            accepted = [r for r in candidates if r["accepted"] and (self.mode == "auto" or r["mode"] == self.mode)]
            chosen = min(accepted, key=lambda r: r["seconds"])["mode"] if accepted else "float64"
            self.selected[key] = chosen
            self.report["selection"][str(key)] = {"mode": chosen, "checked_views": len(entries), "candidates": candidates}
            print(f"Scale {key}: selected {chosen}", flush=True)
        self.report["calibration_seconds"] = time.perf_counter() - started

    def render(self, views):
        import time
        started = time.perf_counter()
        output = torch.empty((len(views), 128, 128, 3))
        used = [None] * len(views)
        groups = {}
        for i, view in enumerate(views):
            groups.setdefault(self.selected[self.bucket(view)], []).append(i)
        for mode, indices in groups.items():
            key = sorted({self.bucket(views[i]) for i in indices})
            renderer = self._make([views[i] for i in indices], mode)
            self._render(renderer)
            if mode != "float64":
                # A rotating representative checks every production batch.
                check = len(self.report["checks"]) % len(indices)
                ref = self._make([views[indices[check]]], "float64")
                expected = self._render(ref)
                delta = (renderer.image[check:check+1]-expected).abs()
                error = {"rgb_mae": delta.mean().item(), "rgb_max": delta.max().item(),
                         "count_mismatch_fraction": (renderer.counts[check:check+1] != ref.counts).float().mean().item()}
                accepted = self._acceptable(error)
                self.report["checks"].append({"scale": key, "mode": mode, "accepted": accepted, **error})
                del ref
                if not accepted:
                    del renderer
                    mode = "float64"
                    renderer = self._make([views[i] for i in indices], mode)
                    self._render(renderer)
            output[indices] = renderer.image.cpu()
            for i in indices:
                used[i] = mode
            del renderer
        self.report["render_seconds"] += time.perf_counter() - started
        return output, used
