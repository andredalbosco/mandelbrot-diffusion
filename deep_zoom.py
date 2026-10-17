"""Double-double Mandelbrot arithmetic: about 31 significant decimal digits."""

import warp as wp


@wp.func
def dd_add(a: wp.vec2d, b: wp.vec2d):
    total = a[0] + b[0]
    v = total - a[0]
    error = (a[0] - (total - v)) + (b[0] - v) + a[1] + b[1]
    high = total + error
    return wp.vec2d(high, error - (high - total))


@wp.func
def dd_mul(a: wp.vec2d, b: wp.vec2d):
    product = a[0] * b[0]
    split_a = wp.float64(134217729.0) * a[0]
    split_b = wp.float64(134217729.0) * b[0]
    ah = split_a - (split_a - a[0])
    bh = split_b - (split_b - b[0])
    al, bl = a[0] - ah, b[0] - bh
    error = ((ah * bh - product) + ah * bl + al * bh) + al * bl
    error += a[0] * b[1] + a[1] * b[0] + a[1] * b[1]
    high = product + error
    return wp.vec2d(high, error - (high - product))


@wp.kernel(enable_backward=False, module="unique", module_options={"fuse_fp": False})
def orbit_fields_kernel(
    center_x: wp.vec2d, center_y: wp.vec2d, step: wp.vec2d,
    z0_x: wp.float64, z0_y: wp.float64, iterations: int, deep: bool,
    escape: wp.array2d(dtype=float), fields: wp.array2d(dtype=wp.vec2),
):
    """Lazy orbit replay for cross traps and mathr's post-escape stripes.

    Only escaped samples need replay. DD coordinates keep the same deep-zoom
    precision as the viewer; ordinary views use float64 orbit arithmetic.
    """
    y, x = wp.tid()
    fields[y, x] = wp.vec2(0.0)
    if escape[y, x] == -1.0:
        return
    dx = wp.float64(x) + wp.float64(.5) - wp.float64(escape.shape[1])*wp.float64(.5)
    dy = wp.float64(escape.shape[0])*wp.float64(.5) - wp.float64(y) - wp.float64(.5)
    cx = dd_add(center_x, dd_mul(wp.vec2d(dx, wp.float64(0.0)), step))
    cy = dd_add(center_y, dd_mul(wp.vec2d(dy, wp.float64(0.0)), step))
    zx, zy = z0_x, z0_y
    hx, hy = wp.vec2d(zx, wp.float64(0.0)), wp.vec2d(zy, wp.float64(0.0))
    trap = wp.float64(1e30)
    n = int(0)
    r2 = zx*zx + zy*zy
    escaped = r2 > wp.float64(4.0)
    while n < iterations and not escaped:
        if deep:
            next_dd = dd_add(dd_add(dd_mul(hx, hx), -dd_mul(hy, hy)), cx)
            hy = dd_add(wp.float64(2.0)*dd_mul(hx, hy), cy)
            hx = next_dd
            zx, zy = hx[0]+hx[1], hy[0]+hy[1]
        else:
            nx = zx*zx - zy*zy + cx[0]
            zy = wp.float64(2.0)*zx*zy + cy[0]
            zx = nx
        r2 = zx*zx + zy*zy
        escaped = r2 > wp.float64(4.0)
        if deep:
            radius_dd = dd_add(dd_mul(hx, hx), dd_mul(hy, hy))
            escaped = radius_dd[0] > wp.float64(4.0) or (radius_dd[0] == wp.float64(4.0) and radius_dd[1] > wp.float64(0.0))
        trap = wp.min(trap, wp.min(wp.abs(zx), wp.abs(zy)))
        n += 1
    if n == 0:
        trap = wp.min(wp.abs(zx), wp.abs(zy))
    fields[y, x] = wp.vec2(0.5, float(trap))
    if not escaped:
        return
    for _ in range(3):
        if r2 < wp.float64(1e15):
            nx = zx*zx - zy*zy + cx[0]
            zy = wp.float64(2.0)*zx*zy + cy[0]
            zx = nx
            r2 = zx*zx + zy*zy
            n += 1
    smooth = wp.float64(n+1) - wp.log(wp.float64(.5)*wp.log(r2))/wp.log(wp.float64(2.0))
    f = smooth - wp.floor(smooth)
    radius = wp.sqrt(r2)
    zx, zy = zx/radius, zy/radius
    total = wp.float64(0.0)
    weight = wp.float64(1.0)-f
    for _ in range(5):
        total += weight*zy
        weight = wp.float64(1.0)
        zx, zy = zx*zx-zy*zy, wp.float64(2.0)*zx*zy
    stripe = wp.float64(.5) + wp.float64(.5)*(total+f*zy)/wp.float64(5.0)
    fields[y, x] = wp.vec2(wp.clamp(float(stripe), 0.0, 1.0), float(trap))


# Error-free transforms above require separate multiply/add operations.
@wp.kernel(enable_backward=False, module="unique", module_options={"fuse_fp": False})
def escape_time_dd_kernel(
    center_x: wp.vec2d, center_y: wp.vec2d, step: wp.vec2d,
    z0_x: wp.float64, z0_y: wp.float64, max_iterations: int,
    unresolved_only: bool,
    counts: wp.array2d(dtype=wp.int32), escape: wp.array2d(dtype=wp.float32),
    styled_escape: wp.array2d(dtype=wp.float32),
):
    y, x = wp.tid()
    if unresolved_only and counts[y, x] >= 0:
        return
    height, width = counts.shape[0], counts.shape[1]
    dx = wp.vec2d(wp.float64(x) + wp.float64(.5) - wp.float64(width)*wp.float64(.5), wp.float64(0.0))
    dy = wp.vec2d(wp.float64(height)*wp.float64(.5) - wp.float64(y) - wp.float64(.5), wp.float64(0.0))
    cx, cy = dd_add(center_x, dd_mul(dx, step)), dd_add(center_y, dd_mul(dy, step))
    zx, zy = wp.vec2d(z0_x, wp.float64(0.0)), wp.vec2d(z0_y, wp.float64(0.0))
    xx, yy = dd_mul(zx, zx), dd_mul(zy, zy)
    radius = dd_add(xx, yy)
    n = int(0)
    while n < max_iterations:
        if radius[0] > wp.float64(4.0) or (radius[0] == wp.float64(4.0) and radius[1] > wp.float64(0.0)):
            break
        # Reuse the squares from the bailout test. Multiplication by two is
        # exact for these bounded orbit values, including both DD components.
        next_x = dd_add(dd_add(xx, -yy), cx)
        zy = dd_add(dd_mul(zx, zy) * wp.float64(2.0), cy)
        zx = next_x
        xx, yy = dd_mul(zx, zx), dd_mul(zy, zy)
        radius = dd_add(xx, yy)
        n += 1
    counts[y, x] = n
    smooth, styled = float(-1.0), float(-1.0)
    if radius[0] > wp.float64(4.0) or (radius[0] == wp.float64(4.0) and radius[1] > wp.float64(0.0)):
        # Color calculations need only ordinary double precision after escape.
        rx, ry = zx[0] + zx[1], zy[0] + zy[1]
        r2 = rx*rx + ry*ry
        styled = float(wp.float64(n-1) + wp.log(wp.float64(2.0))/(wp.float64(.5)*wp.log(r2))
                       + wp.float64(.5)*wp.sqrt(cx[0]*cx[0] + cy[0]*cy[0]))
        refined_n = n
        for _ in range(3):
            if r2 < wp.float64(1.0e100):
                next_rx = rx*rx - ry*ry + cx[0]
                ry = wp.float64(2.0)*rx*ry + cy[0]
                rx = next_rx
                r2 = rx*rx + ry*ry
                refined_n += 1
        smooth = float(wp.float64(refined_n+1) - wp.log(wp.float64(.5)*wp.log(r2))/wp.log(wp.float64(2.0)))
    escape[y, x] = smooth
    styled_escape[y, x] = styled


@wp.kernel(enable_backward=False, module="unique", module_options={"fuse_fp": False})
def reference_orbit_kernel(
    cx: wp.vec2d, cy: wp.vec2d, z0_x: wp.float64, z0_y: wp.float64,
    limit: int, orbit: wp.array(dtype=wp.vec4d), length: wp.array(dtype=wp.int32),
):
    """Compute one double-double orbit shared by all pixels in this viewport."""
    zx, zy = wp.vec2d(z0_x, wp.float64(0.0)), wp.vec2d(z0_y, wp.float64(0.0))
    xx, yy = dd_mul(zx, zx), dd_mul(zy, zy)
    n = int(0)
    orbit[0] = wp.vec4d(zx[0], zx[1], zy[0], zy[1])
    while n < limit:
        radius = dd_add(xx, yy)
        # Continue beyond escape so nearby pixels can escape a little later
        # without exhausting the reference. Stay far from overflow.
        if radius[0] > wp.float64(1e40):
            break
        next_x = dd_add(dd_add(xx, -yy), cx)
        zy = dd_add(dd_mul(zx, zy) * wp.float64(2.0), cy)
        zx = next_x
        xx, yy = dd_mul(zx, zx), dd_mul(zy, zy)
        n += 1
        orbit[n] = wp.vec4d(zx[0], zx[1], zy[0], zy[1])
    length[0] = n


@wp.kernel(enable_backward=False)
def perturbation_kernel(
    center_x: wp.float64, center_y: wp.float64, step: wp.float64,
    max_iterations: int, orbit: wp.array(dtype=wp.vec4d), length: wp.array(dtype=wp.int32),
    counts: wp.array2d(dtype=wp.int32), escape: wp.array2d(dtype=wp.float32),
    styled_escape: wp.array2d(dtype=wp.float32),
):
    """Evolve pixel offsets, with conservative checks and a direct-DD fallback.

    delta_next = 2 * reference * delta + delta**2 + delta_c.
    The accumulated roundoff estimate is a safety heuristic, not interval
    arithmetic. Uncertain bailout, cancellation, or an exhausted reference
    leaves count=-1 so the following kernel recomputes that pixel directly.
    """
    y, x = wp.tid()
    height, width = counts.shape[0], counts.shape[1]
    dcx = (wp.float64(x)+wp.float64(.5)-wp.float64(width)*wp.float64(.5))*step
    dcy = (wp.float64(height)*wp.float64(.5)-wp.float64(y)-wp.float64(.5))*step
    dx, dy, error = wp.float64(0.0), wp.float64(0.0), wp.float64(0.0)
    n = int(0)
    failed = bool(False)
    rx, ry, r2 = wp.float64(0.0), wp.float64(0.0), wp.float64(0.0)
    while n <= max_iterations:
        ref = orbit[n]
        rx, ry = ref[0] + (ref[1] + dx), ref[2] + (ref[3] + dy)
        r2 = rx*rx + ry*ry
        ref_norm = wp.abs(ref[0]) + wp.abs(ref[2])
        delta_norm = wp.abs(dx) + wp.abs(dy)
        # Bailout uncertainty includes reconstruction rounding. Detect severe
        # cancellation and accumulated error before trusting an escape count.
        tolerance = wp.float64(32.0)*(wp.abs(rx)+wp.abs(ry)+wp.float64(1.0))*(error+wp.float64(1e-15))
        if (wp.abs(r2-wp.float64(4.0)) <= tolerance
                or r2 < wp.float64(1e-8)*(ref[0]*ref[0]+ref[2]*ref[2])
                or error > wp.float64(1e-10)*(wp.abs(rx)+wp.abs(ry)+wp.float64(1e-12))):
            failed = True
            break
        if r2 > wp.float64(4.0) or n == max_iterations:
            break
        if n >= length[0]:
            failed = True
            break
        # The complex derivative has magnitude 2|z|. An L1 bound here grows
        # spuriously on every iteration and rejects otherwise stable orbits.
        growth = wp.float64(2.000002)*wp.float64(wp.sqrt(float(r2)))
        error = (growth+wp.float64(2.0)*error)*error + wp.float64(8e-15)*(
            wp.float64(2.0)*ref_norm*delta_norm + delta_norm*delta_norm + wp.abs(dcx)+wp.abs(dcy))
        next_dx = wp.float64(2.0)*(ref[0]*dx-ref[2]*dy) + dx*dx-dy*dy + dcx
        next_dy = wp.float64(2.0)*(ref[0]*dy+ref[2]*dx+dx*dy) + dcy
        # Preserve the small reference components in the perturbation too.
        next_dx += wp.float64(2.0)*(ref[1]*dx-ref[3]*dy)
        next_dy += wp.float64(2.0)*(ref[1]*dy+ref[3]*dx)
        dx, dy = next_dx, next_dy
        n += 1
    counts[y, x] = n
    if failed:
        counts[y, x] = -1
    else:
        smooth, styled = float(-1.0), float(-1.0)
        if r2 > wp.float64(4.0):
            cx, cy = center_x+dcx, center_y+dcy
            styled = float(wp.float64(n-1)+wp.log(wp.float64(2.0))/(wp.float64(.5)*wp.log(r2))
                           + wp.float64(.5)*wp.sqrt(cx*cx+cy*cy))
            refined_n = n
            for _ in range(3):
                if r2 < wp.float64(1e100):
                    next_rx = rx*rx-ry*ry+cx
                    ry = wp.float64(2.0)*rx*ry+cy
                    rx = next_rx
                    r2 = rx*rx+ry*ry
                    refined_n += 1
            smooth = float(wp.float64(refined_n+1)-wp.log(wp.float64(.5)*wp.log(r2))/wp.log(wp.float64(2.0)))
        escape[y, x] = smooth
        styled_escape[y, x] = styled
