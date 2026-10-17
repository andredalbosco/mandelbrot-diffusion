"""Headless viewer workload benchmark; timings exclude OpenGL and Moonlight.

Run: uv run python docs/benchmark_render.py
For the former resolution: --width 1024 --height 768 --aa-side 1
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from mandelbrot import ColorRenderer, View


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--height', type=int, default=128)
    parser.add_argument('--aa-side', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=20)
    args = parser.parse_args()
    renderer = ColorRenderer(args.width, args.height, side=args.aa_side)
    deep_center = '(-0.7436438870371510000000001+0.1318259042053300000000002j)'
    views = {
        'main': View(),
        'shallow_boundary': View(center='(-0.74583666281102334781938374733147424375+0.135001050939548007382205118563122323j)', span=6.372367644529897e-7),
        'deep_boundary': View(center='(-0.92844118559550850366513868844580424143709174+0.262617371450439604741270010095574502976117479j)', span=4.118046071574424e-20),
        'deep_512': View(center=deep_center, span=1e-16),
        'deep_2048': View(center=deep_center, span=1e-16, iterations=2048),
    }
    result = {'gpu': torch.cuda.get_device_name(), 'width': args.width, 'height': args.height,
              'samples_per_pixel': args.aa_side**2, 'repeats': args.repeats, 'views': {}}
    for name, view in views.items():
        for _ in range(3):
            renderer.render(view)
        torch.cuda.synchronize()
        timings = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            renderer.render(view)
            torch.cuda.synchronize()
            timings.append((time.perf_counter()-start)*1000)
        result['views'][name] = {'center': view.center_text, 'span': view.span, 'iterations': view.iterations,
                                'precision': renderer.precision, 'median_ms': statistics.median(timings),
                                'min_ms': min(timings), 'max_ms': max(timings)}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
