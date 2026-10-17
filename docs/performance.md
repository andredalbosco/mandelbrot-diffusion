# Viewer performance, 2026-10-16

The original deep renderer ran double-double arithmetic independently for every
pixel. At 1024×768 this was substantially more expensive than ordinary float64
rendering. Frame profiling on Moonlight's `:101` display found roughly 105 ms per
moving frame at the deep boundary viewport, versus 7.5 ms when idle. Window
presentation accounted for about 6.5 ms. Bypassing Warp's empty 3D pass did not
improve the result, so the existing Warp viewer was retained.

Deep rendering now computes one double-double reference orbit and evolves small
float64 offsets around it. Pixels with cancellation, uncertain bailout, excessive
estimated roundoff, or an exhausted reference are recomputed directly with
double-double arithmetic. Direct arithmetic also reuses orbit squares and uses
exact multiplication by two. These changes preserve high-precision viewport
coordinates; neither iteration budgets nor the requested sample count are reduced.

The perturbation error estimate is a conservative numerical heuristic, not a
rigorous interval-arithmetic proof. The direct fallback also has finite precision.
A difficult viewport can fall back heavily and will not achieve the same speedup
as a stable reference orbit.

## Measurements

RTX 4090, CUDA, cosine palette. Medians of warmed render calls followed by CUDA
synchronization, including coloring and sample resolve, excluding OpenGL and
Moonlight. The script below records exact centers, spans, iteration budgets and
min/max times. GPU contention changes absolute timing; these are observations,
not frame-rate guarantees.

At the unchanged 1024×768, one-sample workload:

| View | Original direct DD | Optimized direct DD | Reference perturbation |
| --- | ---: | ---: | ---: |
| `deep_512` | 114 ms | 79 ms | 34–37 ms |
| `deep_2048` | 456 ms | 314 ms | 147–154 ms |

The current requested output is **128×128 with four distinct spatial samples per
pixel**, arranged in a 2×2 grid. Every sample is colored before the four colors
are averaged. This evaluates 65,536 subpixels, compared with the previous 786,432
single-sample pixels: twelve times fewer orbit evaluations.

| View | 128×128, 4 samples/pixel |
| --- | ---: |
| Main view, 512 iterations | 0.14 ms |
| Shallow boundary, 512 iterations | 0.48 ms |
| Deep boundary, 512 iterations | 9.17 ms |
| `deep_512` | 3.38 ms |
| `deep_2048` | 14.50 ms |

The deep boundary case required direct fallback for about 74% of subpixels.
The two `deep_*` reference cases remain unescaped at their specified budgets;
the boundary case includes both escaped and unescaped samples. They measure
different workloads and should not be presented as an all-view speedup.

Reproduce current measurements:

```sh
uv run python docs/benchmark_render.py
uv run python docs/benchmark_render.py --width 1024 --height 768 --aa-side 1
```

The original and intermediate timings were measured before replacing those
implementations; the script runs the current implementation.

## Numerical validation

`tests/test_deep_zoom.py` compares escape counts with an independent 70-digit
Decimal implementation across eight cases, including the detailed deep boundary,
near-radius-two bailout, nonzero initial orbits, and immediate escape. Separate
checks exercise both the fast path and the direct fallback, and exhaustion of
the capped reference buffer. These are numerical correctness checks, not a
requirement that colors reproduce the old renderer byte for byte.


## Tiled backend in the OpenGL viewer

Deep views with z0=0 and up to 65,536 iterations now use the same tiled DD
references, FP32 perturbation, and compact direct-DD fallback as the generation
backend. The viewer retains its world-anchored pan cache, all palettes, smooth
escape-time fields, and G gradient mode. Other initial z/iteration configurations
retain the previous compatible path. Numerical error guards remain heuristic.

Matched RTX 4090 test at center
`(0.3063437178089269082657835032384019-0.5672430783502638215578630647942312j)`,
span `4.995366924470893e-13`, 8,192 iterations, output128×128, four samples/pixel,
tent filtering, cached sample extent384×384. Warmed medians over five repeats:

| Work | Previous single reference | Tiled references + compact fallback |
|---|---:|---:|
| Fresh cache + RGB + escape resolve + Sobel/auto contrast | 413.0 ms | 194.9 ms |
| Cached view + RGB + escape resolve + Sobel/auto contrast | — | 0.54 ms |

This is a 2.12× improvement for this location. Timings exclude OpenGL presentation,
Moonlight and compilation. Roughly52.5% of samples still require direct DD, so the
fresh render remains costly. There is no claim of a fixed speedup across views.
A second sampled location at span7.00649e-25 took67.0ms for the new fresh path;
no matched legacy timing was recorded there.

The entire384×384 sample cache was compared against independently launched direct
DD: zero count differences; smooth escape MAE2.32e-8 (max4.88e-4); styled escape
MAE9.93e-9; displayed gradient MAE0. Small independent70-digit Decimal tests cover
counts, and integration tests cover raw fields, palettes, and tent reconstruction.
Details: `output/benchmarks/viewer-tiled-update.json` and
`output/benchmarks/viewer-tiled-current-location.json`.
