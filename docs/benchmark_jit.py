"""Disposable conditional JiT timing experiment; no quality or convergence claims.

Install einops in the project environment, then:
  .venv/bin/python docs/benchmark_jit.py
Upstream MIT sources are downloaded at a pinned revision into the output folder.
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
import types
import urllib.request

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import DiT, JiT, UNet
from mandelbrot import ColorRenderer, View

REVISION = "cbc743a2ada5e9762697da2c83f8c4f8379e8c17"


def upstream(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("model_jit.py", "util/model_util.py", "LICENSE"):
        path = directory / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(f"https://raw.githubusercontent.com/LTH14/JiT/{REVISION}/{name}", path)
    sys.path.insert(0, str(directory))
    module = types.ModuleType("benchmark_upstream_jit")
    module.__file__ = str(directory / "model_jit.py")
    sys.modules[module.__name__] = module
    # Control compilation at the whole-model level for an honest eager baseline.
    source = (directory / "model_jit.py").read_text().replace("    @torch.compile\n", "")
    exec(compile(source, str(directory / "model_jit.py"), "exec"), module.__dict__)
    return module


def make_jit(module, patch, width=256, depth=8, heads=8):
    class ConditionalJiT(module.JiT):
        def __init__(self):
            super().__init__(input_size=128, patch_size=patch, hidden_size=width,
                             depth=depth, num_heads=heads, bottleneck_dim=128,
                             num_classes=1, in_context_len=0, in_context_start=depth)
            # RGB state + RGB crop condition in; clean RGB prediction out.
            self.x_embedder.proj1 = nn.Conv2d(6, 128, patch, stride=patch, bias=False)
            nn.init.xavier_uniform_(self.x_embedder.proj1.weight.flatten(1))
            del self.y_embedder

        def forward(self, x, t, condition):
            c = self.t_embedder(t)
            tokens = self.x_embedder(torch.cat((x, condition), dim=1)) + self.pos_embed
            for block in self.blocks:
                tokens = block(tokens, c, self.feat_rope)
            return self.unpatchify(self.final_layer(tokens, c), self.patch_size)

    return ConditionalJiT()


def measure(fn, warmup=10, repeats=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    wall, gpu = [], []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        before = time.perf_counter()
        start.record()
        fn()
        end.record()
        end.synchronize()
        wall.append(1000 * (time.perf_counter() - before))
        gpu.append(start.elapsed_time(end))
    return {"median_ms": statistics.median(wall), "mean_ms": statistics.mean(wall),
            "p90_ms": sorted(wall)[math.ceil(.9 * repeats) - 1],
            "gpu_median_ms": statistics.median(gpu), "repeats": repeats}


def run_model(name, model, prediction, condition, clean, args):
    model = model.cuda()
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0)

    def update():
        t = torch.rand(len(clean), device="cuda")
        state = condition.lerp(clean, t[:, None, None, None])
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(state, t, condition)
            if prediction == "x":
                scale = (1 - t[:, None, None, None]).clamp_min(.05)
                loss = ((output.float() - clean) / scale).square().mean()
            else:
                loss = F.mse_loss(output, clean - condition)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        with torch.no_grad():
            for target, source in zip(ema.parameters(), model.parameters()):
                target.lerp_(source, .001)
        if not torch.isfinite(norm).item() or not torch.isfinite(loss).item():
            raise RuntimeError("Nonfinite training smoke")
        return loss.item()

    model.train()
    for _ in range(5):
        update()
    torch.cuda.reset_peak_memory_stats()
    training = measure(update, warmup=0, repeats=args.train_repeats)
    training["pairs_per_second"] = len(clean) * 1000 / training["mean_ms"]
    training["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    del optimizer, ema
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    model.eval()
    result = {"parameters": parameters, "prediction": prediction, "training_eager": training,
              "inference": {}}
    print(name, "training", training, flush=True)

    for batch in (1, len(clean)):
        y = condition[:batch].contiguous()
        x = clean[:batch].contiguous()
        t = torch.full((batch,), .5, device="cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            expected = model(x, t, y).float().clone()
            if not torch.isfinite(expected).all():
                raise RuntimeError("Nonfinite inference")
            eager = measure(lambda: model(x, t, y), repeats=args.repeats)
            record = {"eager": eager}
            if not args.no_compile:
                torch._dynamo.reset()
                compiled = torch.compile(model, mode="reduce-overhead", fullgraph=True)
                before = time.perf_counter()
                actual = compiled(x, t, y).float().clone()
                torch.cuda.synchronize()
                record["compile_first_call_seconds"] = time.perf_counter() - before
                difference = (actual - expected).abs()
                record["compiled_difference"] = {"max": difference.max().item(), "mean": difference.mean().item()}
                torch.testing.assert_close(actual, expected, atol=.03, rtol=.03)
                record["compiled"] = measure(lambda: compiled(x, t, y), repeats=args.repeats)

            def sample(network):
                state = y.clone()
                for i in range(16):
                    ts = torch.full((batch,), i / 16, device="cuda")
                    out = network(state, ts, y)
                    velocity = (out - state) / (1 - i / 16) if prediction == "x" else out
                    state = state + velocity / 16
                return state

            record["sample16_eager"] = measure(lambda: sample(model), warmup=2, repeats=5)
            if not args.no_compile:
                record["sample16_compiled"] = measure(lambda: sample(compiled), warmup=2, repeats=5)
                del compiled
            for mode in ("eager", "compiled"):
                if mode in record:
                    record[mode]["images_per_second"] = batch * 1000 / record[mode]["mean_ms"]
            result["inference"][str(batch)] = record
            print(name, "batch", batch, json.dumps(record), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--train-repeats", type=int, default=30)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "output/benchmarks" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-jit") / "results.json"
    if output.exists():
        parser.error("Output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    module = upstream(ROOT / "output/benchmarks/jit-source")
    # Same mathematical operation, using PyTorch's fused SDPA dispatch.
    module.scaled_dot_product_attention = F.scaled_dot_product_attention
    torch.manual_seed(20261016)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    renderer = ColorRenderer(128, 128, side=2, filter="tent")
    parent = renderer.render(View()).permute(2, 0, 1)[None].clone()
    clean = renderer.render(View(span=3.5/4)).permute(2, 0, 1)[None].clone()
    condition = F.interpolate(parent[:, :, 48:80, 48:80], size=(128, 128), mode="bilinear", align_corners=False)
    clean, condition = [(v.repeat(32, 1, 1, 1) * 2 - 1).contiguous() for v in (clean, condition)]
    del renderer
    configs = {
        "dit_p4": (lambda: DiT(size=128, channels=3, condition_channels=3, hidden_size=256, depth=8, num_heads=8, patch_size=4), "v"),
        "dit_p16": (lambda: DiT(size=128, channels=3, condition_channels=3, hidden_size=256, depth=8, num_heads=8, patch_size=16), "v"),
        "unet": (lambda: UNet(), "v"),
        "jit_small_p8": (lambda: make_jit(module, 8), "x"),
        "jit_small_p16": (lambda: make_jit(module, 16), "x"),
        "jit_b_p16": (lambda: make_jit(module, 16, width=768, depth=12, heads=12), "x"),
        "jit_1m": (lambda: JiT(size=128, patch_size=8, hidden_size=128, depth=3, num_heads=4, bottleneck_dim=64), "x"),
        "jit_10m": (lambda: JiT(size=128, patch_size=8, hidden_size=256, depth=8, num_heads=8, bottleneck_dim=128), "x"),
    }
    result = {"utc": datetime.now(timezone.utc).isoformat(), "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name(), "upstream_revision": REVISION,
              "source_sha256": hashlib.sha256((ROOT / "output/benchmarks/jit-source/model_jit.py").read_bytes()).hexdigest(),
              "settings": vars(args) | {"output": str(output)},
              "gpu_before": subprocess.check_output(["nvidia-smi"], text=True),
              "notes": "128 RGB, BF16 autocast, FP32 parameters, GPU resident repeated main-view pair; eager AdamW+clip+EMA training. JiT: six input channels, three outputs, no class/context tokens, SDPA. jit_1m/jit_10m use model.py; other JiTs use upstream with decorators removed. No convergence or quality evidence.",
              "models": {}}
    for name, (factory, prediction) in configs.items():
        if args.only and name not in args.only:
            continue
        torch.manual_seed(20261016)
        result["models"][name] = run_model(name, factory(), prediction, condition, clean, args)
        output.write_text(json.dumps(result, indent=2))
    print("Saved", output, flush=True)


if __name__ == "__main__":
    main()
