"""Measure rendering, paired-flow training, and inference on the local CUDA GPU.

uv run python docs/benchmark_pipeline.py
Disposable model weights: this measures execution, not learning or image quality.
"""

import argparse
import copy
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F

from mandelbrot import ColorRenderer, View
from model import RectifiedFlow, build_model
from train import batch_loss, validation_loss


def measure(function, warmup, repeats):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    wall, gpu = [], []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        before = time.perf_counter()
        start.record()
        function()
        end.record()
        end.synchronize()
        wall.append((time.perf_counter()-before)*1000)
        gpu.append(start.elapsed_time(end))
    return {"median_ms": statistics.median(wall), "mean_ms": statistics.mean(wall),
            "p90_ms": sorted(wall)[math.ceil(.9*len(wall))-1],
            "gpu_median_ms": statistics.median(gpu), "repeats": repeats}


def benchmark_render(repeats):
    deep = "(-0.7436438870371510000000001+0.1318259042053300000000002j)"
    views = {
        "main": View(),
        "shallow_boundary": View(center=-.7458366628110233+.135001050939548j, span=6.372367644529897e-7),
        "deep_boundary": View(center="(-0.92844118559550850366513868844580424143709174+0.262617371450439604741270010095574502976117479j)", span=4.118046071574424e-20),
        "deep_2048": View(center=deep, span=1e-16, iterations=2048),
    }
    states = list((Path.home()/".local/state/mandelbrot-diffusion").glob("viewer-*.json"))
    if states:
        state = json.loads(max(states, key=lambda p: p.stat().st_mtime).read_text())["view"]
        views["saved_view"] = View(state["center"], state["span"], state["iterations"], complex(state["z0"]))
    results, pairs = {}, []
    for side in (2, 4, 8):
        for name, view in views.items():
            parent = ColorRenderer(128, 128, side=side, filter="tent")
            child = ColorRenderer(128, 128, side=side, filter="tent")
            target_view = view.shifted(0j, span=view.span/4)

            def fresh_image():
                # Force orbit evaluation: repeated cached pans are not data production.
                parent._grid_origin = None
                return parent.render(view)

            def pair():
                parent._grid_origin = child._grid_origin = None
                coarse = parent.render(view)
                target = child.render(target_view)
                crop = coarse[48:80, 48:80].permute(2, 0, 1)[None]
                condition = F.interpolate(crop, size=(128, 128), mode="bilinear", align_corners=False)[0]
                return torch.stack((condition, target.permute(2, 0, 1)))

            image_stats = measure(fresh_image, 2, repeats)
            pair_stats = measure(pair, 2, repeats)
            # Separate cost of a cached camera update from fresh rendering.
            pan_stats = measure(lambda: parent.render(view.shifted(complex(view.span/500, 0))), 2, repeats)
            record = {"view": {"center": view.center_text, "span": view.span, "iterations": view.iterations},
                      "side": side, "image": image_stats, "pair": pair_stats, "cached_pan": pan_stats,
                      "precision": parent.precision}
            results[f"{name}_{side**2}spp"] = record
            if side == 2:
                pairs.append(pair())
            print(f"render {name} {side**2}spp: image {image_stats['median_ms']:.2f}ms; pair {pair_stats['median_ms']:.2f}ms", flush=True)
            del parent, child
    pool = torch.stack(pairs).repeat(16, 1, 1, 1, 1).contiguous()*2-1
    return results, pool


def benchmark_model(config, pool, steps, warmup):
    torch.manual_seed(config["seed"])
    model = build_model(config).cuda()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0)
    flow = RectifiedFlow()
    batch_size = config["batch_size"]

    def train_step():
        indices = torch.randint(len(pool), (batch_size,), device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = batch_loss(model, flow, pool[indices])
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite benchmark loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError("Nonfinite benchmark gradient")
        optimizer.step()
        with torch.no_grad():
            for target, source in zip(ema.parameters(), model.parameters()):
                target.lerp_(source, 1-config["ema_decay"])
        # Match train.py's scalar logging synchronization.
        return loss.item()

    for _ in range(warmup):
        train_step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    training = measure(train_step, 0, steps)
    training["pairs_per_second"] = batch_size*1000/training["mean_ms"]
    training["peak_allocated_gib"] = torch.cuda.max_memory_allocated()/2**30
    training["peak_reserved_gib"] = torch.cuda.max_memory_reserved()/2**30
    training["projected_20000_steps_minutes"] = training["mean_ms"]*20000/60000
    print(f"{config['architecture']} train: {training['pairs_per_second']:.1f} pairs/s, {training['median_ms']:.1f}ms/step, {training['peak_allocated_gib']:.2f}GiB peak", flush=True)

    validation = benchmark_validation(model, ema, flow, pool, config)

    # Measure checkpoint writes separately; no benchmark checkpoint is retained.
    import tempfile
    with tempfile.TemporaryDirectory(prefix="mandelbrot-benchmark-") as directory:
        path = Path(directory)/"checkpoint.pt"
        state = {"config": config, "model": model.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict()}
        checkpoint = measure(lambda: torch.save(state, path), 0, 2)
        checkpoint["bytes"] = path.stat().st_size

    # Free optimizer/EMA so inference memory is measured independently.
    del optimizer, ema
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    inference = {}
    for count in (1, batch_size):
        condition = pool[:count, 0]
        def sample():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return flow.sample(model, condition, steps=16)
        torch.cuda.reset_peak_memory_stats()
        stats = measure(sample, 2, 3)
        stats["images_per_second"] = count*1000/stats["mean_ms"]
        stats["peak_allocated_gib"] = torch.cuda.max_memory_allocated()/2**30
        inference[str(count)] = stats
        print(f"{config['architecture']} 16-step sample batch {count}: {stats['median_ms']:.1f}ms, {stats['images_per_second']:.1f} images/s", flush=True)
    return {"config": config, "parameters": sum(p.numel() for p in model.parameters()),
            "training": training, "validation_1000_pairs_model_and_ema": validation,
            "checkpoint": checkpoint, "inference_16_steps": inference}


def benchmark_validation(model, ema, flow, pool, config):
    validation = pool.repeat((1000+len(pool)-1)//len(pool), 1, 1, 1, 1)[:1000]
    def evaluate():
        for network in (model, ema):
            validation_loss(network, flow, validation, config["batch_size"], config["seed"]+1)
    return measure(evaluate, 1, 3)


def benchmark_storage(pool):
    import tempfile
    from PIL import Image
    transfer = measure(lambda: pool.cpu(), 2, 10)
    host = pool.cpu()
    with tempfile.TemporaryDirectory(prefix="mandelbrot-benchmark-io-") as directory:
        path = Path(directory)/"pairs.pt"
        save = measure(lambda: torch.save(host, path), 1, 5)
        save["bytes"] = path.stat().st_size
        save["mib_per_second"] = save["bytes"]/2**20/(save["mean_ms"]/1000)
        rgb = ((host[0, 1]+1)*127.5).clamp(0,255).byte().permute(1,2,0).numpy()
        png = measure(lambda: Image.fromarray(rgb).save(Path(directory)/"image.png"), 1, 10)
    return {"pool_gpu_to_cpu": transfer, "pool_buffered_pt_save": save,
            "one_128px_png_cpu_encoding": png, "note": "Writes use OS page cache; no fsync/physical-disk throughput claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--render-repeats", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path("output/benchmarks")/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")/"results.json"
    if output.exists():
        parser.error("Output exists; choose a fresh path")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {"utc": datetime.now(timezone.utc).isoformat(), "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
              "gpu_initial": subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,power.draw", "--format=csv"], text=True),
              "notes": ["Eager BF16 autocast; FP32 parameters; no torch.compile or production changes.",
                        "Train loop includes indexing, loss checks, backward, gradient clipping, AdamW, EMA, scalar logging; excludes scheduled validation/checkpoints.",
                        "Rendered benchmark pairs repeated in GPU-resident pool; throughput only, no learning claim.",
                        "Rendering includes CUDA work and CPU launch overhead, excludes region search/rejection, PNG encoding and GUI.",
                        "Tent cache misses include overscan; four samples/pixel is sample density, not total cache work.",
                        "Desktop/viewer remains running; unrelated GPU load is not controlled."], "models": {}}
    result["source_sha256"] = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                               for name in ("model.py", "train.py", "mandelbrot.py", "deep_zoom.py", "docs/benchmark_pipeline.py")}
    def save():
        output.write_text(json.dumps(result, indent=2)+"\n")
    print(f"Writing {output}", flush=True)
    result["rendering"], pool = benchmark_render(args.render_repeats)
    save()
    result["pool_shape"] = list(pool.shape)
    result["pool_gib"] = pool.numel()*pool.element_size()/2**30
    result["storage"] = benchmark_storage(pool)
    for architecture in ("dit", "unet"):
        gc.collect()
        torch.cuda.empty_cache()
        config = json.loads(Path(f"configs/zoom4x-128-{architecture}.json").read_text())
        try:
            result["models"][architecture] = benchmark_model(config, pool, args.steps, args.warmup)
        except torch.cuda.OutOfMemoryError:
            result["models"][architecture] = {"error": "CUDA out of memory", "config": config}
        save()
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
