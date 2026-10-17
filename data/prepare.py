"""Generate diverse 4x zoom pairs, with smooth blue/gold colors."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace
import torch
from torch.nn import functional as F
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train import save_grid, reconstruct_conditioning

def zoom_modes(count):
    names = ('interior', 'edge', 'corner', 'unconditional')
    expected = [count*p for p in (.60, .15, .15, .10)]
    quotas = [int(n) for n in expected]
    for i in sorted(range(4), key=lambda i: expected[i]-quotas[i], reverse=True)[:count-sum(quotas)]:
        quotas[i] += 1
    return [name for name, n in zip(names, quotas) for _ in range(n)]


class ViewDiversity:
    """Reject cameras within 3% of a view width and 3% in scale.

    This filters overlapping cameras, not all visually similar fractal copies.
    Spatial cells accelerate exact neighbor checks across adjacent scale bins.
    """
    def __init__(self, tolerance=.03):
        self.tolerance = tolerance
        self.bins = {}

    def add(self, center, span):
        x, y = center
        level = math.floor(math.log(span)/self.tolerance)
        for neighbor in range(level-1, level+2):
            cell = math.exp(neighbor*self.tolerance)*self.tolerance
            ix, iy = math.floor(x/cell), math.floor(y/cell)
            grid = self.bins.get(neighbor, {})
            for dx in (-2, -1, 0, 1, 2):
                for dy in (-2, -1, 0, 1, 2):
                    for ax, ay, old_span in grid.get((ix+dx, iy+dy), ()):
                        if abs(math.log(span/old_span)) < self.tolerance and math.hypot(x-ax,y-ay) < self.tolerance*min(span,old_span):
                            return False
        cell = math.exp(level*self.tolerance)*self.tolerance
        key = (math.floor(x/cell), math.floor(y/cell))
        self.bins.setdefault(level, {}).setdefault(key, []).append((x,y,span))
        return True


def zoom_depth_quotas(count, max_depth, overview_count=None):
    if overview_count is None:
        return [count//(max_depth+1)+(d<count%(max_depth+1)) for d in range(max_depth+1)]
    if not 0 <= overview_count <= count or max_depth < 1:
        raise ValueError("Overview quota requires 0 <= count <= split count and max-depth >= 1")
    rest = count-overview_count
    return [overview_count]+[rest//max_depth+(d<rest%max_depth) for d in range(max_depth)]


def prepare_zoom_dataset(args, sampling_state=None):
    """RGB evidence/coverage/target data shared by zoom and generation tasks."""
    from mandelbrot import GrayscaleRenderer, View
    from mandelbrot import FastRenderer, DatasetRenderer

    if min(args.train_count, args.val_count, args.iterations, args.aa_side, args.batch_size) < 1:
        raise ValueError("Counts and iteration budget must be positive")
    if not 0 <= args.max_depth <= 16:
        raise ValueError("max-depth must be between 0 and 16 for this FP64 dataset search")
    if args.output.exists():
        raise ValueError("Output already exists; choose a new directory")
    total_started = time.perf_counter()
    rng = random.Random(args.seed)
    state = {} if sampling_state is None else sampling_state
    if "groups" not in state:
        state["groups"] = boundary_anchors("cuda", args.seed)
    groups = state["groups"]
    seen = state.setdefault("seen", set())
    diversity = state.setdefault("diversity", ViewDiversity()) if getattr(args, "diverse", False) else None
    rejected = Counter()

    def pilot_masks(views):
        masks = []
        for start in range(0, len(views), 128):
            renderer = FastRenderer(views[start:start+128], width=24, height=24, side=1, precision="float64")
            renderer.render()
            limits = torch.tensor([v.iterations for v in views[start:start+128]], device="cuda")
            masks.extend((renderer.counts[:, 1:25, 1:25] >= limits[:, None, None]).cpu())
        return masks

    def choose(count, split):
        # Search breadth-first: each GPU launch evaluates many independent paths.
        selected = []
        names = list(groups[split])
        quotas = zoom_depth_quotas(count, args.max_depth, getattr(args, f"{split}_overview_count", None))
        for depth, quota in enumerate(quotas):
            accepted = []
            attempts = 0
            while len(accepted) < quota:
                batch = min(128, max(8, quota-len(accepted)))
                paths = []
                for _ in range(batch):
                    group = rng.choice(names)
                    ax, ay = rng.choice(groups[split][group])
                    cx, cy = ((-.5+rng.uniform(-.08,.08), rng.uniform(-.08,.08)) if depth == 0 else (ax+rng.uniform(-1.5/512,1.5/512),ay+rng.uniform(-1.5/512,1.5/512)))
                    span = 3.5 if depth == 0 else 3.5/4
                    if diversity is not None:
                        span *= rng.uniform(.85, 1.15) if depth == 0 else 4**rng.uniform(-.5, .5)
                        cx,cy = ((-.5+rng.uniform(-.3,.3),rng.uniform(-.3,.3)) if depth==0
                                 else (ax+rng.uniform(-.4,.4)*span,ay+rng.uniform(-.4,.4)*span))
                    paths.append({"center": [cx,cy], "span": span, "region": group})
                valid = [True] * batch
                for level in range(1, depth):
                    masks = pilot_masks([View(complex(*r["center"]),r["span"],args.iterations) for r in paths])
                    for i,(record,black) in enumerate(zip(paths,masks)):
                        edge = torch.zeros_like(black)
                        edge[:,1:] |= black[:,1:] != black[:,:-1]
                        edge[1:] |= black[1:] != black[:-1]
                        points = edge.nonzero().tolist()
                        if not points:
                            valid[i] = False
                            continue
                        y,x = rng.choice(points)
                        record["center"][0] += (x+.5-12)*record["span"]/24
                        record["center"][1] += (12-y-.5)*record["span"]/24
                        record["span"] /= 4
                masks = pilot_masks([View(complex(*r["center"]),r["span"],args.iterations) for r in paths])
                for record,black,ok in zip(paths,masks,valid):
                    fraction = black.float().mean().item()
                    key = (*record["center"],record["span"])
                    if ok and .1 <= fraction <= .9 and key not in seen and (diversity is None or diversity.add(record["center"], record["span"])):
                        seen.add(key)
                        record.update(zoom_depth=depth,magnification=3.5/record["span"],
                                      pilot_black_fraction=fraction,iterations=args.iterations)
                        accepted.append(record)
                        if len(accepted) == quota:
                            break
                    else:
                        rejected["uninformative_or_duplicate"] += 1
                attempts += batch
                if attempts > max(1,quota)*2000:
                    raise RuntimeError(f"Could not select depth {depth} for {split}")
            selected.extend(accepted)
            print(f"{split}: depth {depth}, selected {len(accepted)}",flush=True)
        rng.shuffle(selected)
        return selected

    search_started = time.perf_counter()
    validation = choose(args.val_count, "val")
    training = choose(args.train_count, "train")
    search_seconds = time.perf_counter() - search_started
    args.output.mkdir(parents=True)
    backend = DatasetRenderer(side=args.aa_side, mode=args.render_mode, palette=args.palette)
    baseline_seconds = 0.
    io_seconds = 0.
    report = {"config": {**vars(args), "output": str(args.output)},
              "rendering": f"128 RGB; calibrated backend {args.render_mode}; z0=0; {args.palette} palette; {args.aa_side**2} samples/pixel; tent filter",
              "diversity": {"enabled": diversity is not None, "camera_tolerance": diversity.tolerance if diversity else None, "overview_counts": {split: getattr(args, f"{split}_overview_count", None) for split in ("train", "val")}},
              "distribution": ("Capped overviews; balanced remaining depth bins; continuous scales and wider anchor offsets; reject nearby cameras. " if diversity else "") + f"Nominal depth bins across {args.max_depth+1} depths: span=3.5/4**depth, depth 0..{args.max_depth}. Recursively follow the local boundary; pilot black fraction .1–.9.",
              "split_policy": "Disjoint initial boundary anchor groups for zoom trajectories. Overview views and crop footprints can overlap across splits; no unseen-geography claim. Depth quotas match within one example per bin.",
              "mode_proportions": {"interior": .60, "edge": .15, "corner": .15, "unconditional": .10},
              "masks": "Interpolation support in [0,1]; 1 means coarse RGB evidence available, not an exact pixel to preserve. Missing RGB is neutral .5.",
              "training_note": "Targets/images share storage. Full/partial/no-evidence modes require mask-aware training; use the conditional-jit-x-pred trainer.",
              "source_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                for name in ("data/prepare.py", "mandelbrot.py")},
              "rejected": dict(rejected), "splits": {}}
    for split, records in (("train", training), ("val", validation)):
        plan = zoom_modes(len(records))
        rng.shuffle(plan)
        for record, mode in zip(records, plan):
            left, top = rng.randrange(97), rng.randrange(97)
            if mode in ("edge", "corner"):
                outside = lambda: rng.choice((rng.randint(-16, -1), rng.randint(97, 112)))
                if mode == "corner":
                    left, top = outside(), outside()
                elif rng.random() < .5:
                    left = outside()
                else:
                    top = outside()
            parent_span = record["span"] * 4
            cx, cy = record["center"]
            parent_center = [cx-(left+16-64)*parent_span/128,
                             cy-(64-top-16)*parent_span/128]
            record.update(mode=mode, crop_left=left, crop_top=top, crop_size=32,
                          parent_center=parent_center, parent_span=parent_span, zoom_factor=4)
        all_views = [View(complex(*r["parent_center"]), r["parent_span"], args.iterations) for r in records]
        all_views += [View(complex(*r["center"]), r["span"], args.iterations) for r in records]
        if args.render_mode == "float64":
            backend.selected.update({backend.bucket(v): "float64" for v in all_views})
        elif not backend.selected:
            backend.calibrate(all_views)
        # Preallocate once: a 100k dataset must not hold both batch lists and
        # their concatenated copies in RAM.
        inputs = torch.empty(len(records), 3, 32, 32)
        targets = torch.empty(len(records), 3, 128, 128)
        masks = torch.empty(len(records), 1, 32, 32)
        for start in range(0, len(records), args.batch_size):
            chunk = records[start:start+args.batch_size]
            views = [View(complex(*r["parent_center"]), r["parent_span"], args.iterations) for r in chunk]
            views += [View(complex(*r["center"]), r["span"], args.iterations) for r in chunk]
            images, used = backend.render(views)
            for i, record in enumerate(chunk):
                record["parent_render_mode"], record["target_render_mode"] = used[i], used[i+len(chunk)]
            if args.benchmark_baseline:
                torch.cuda.synchronize()
                before = time.perf_counter()
                reference = FastRenderer(views, side=args.aa_side, precision="float64")
                reference.render().cpu()
                baseline_seconds += time.perf_counter() - before
                del reference
            images = images.permute(0, 3, 1, 2)
            parent, target = images.chunk(2)
            evidence, coverage = zoom_evidence(parent, [r["crop_left"] for r in chunk], [r["crop_top"] for r in chunk])
            for i, record in enumerate(chunk):
                if record["mode"] == "unconditional":
                    coverage[i].zero_()
                    evidence[i].fill_(.5)
                record["observed_fraction"] = coverage[i].mean().item()
            end = start+len(chunk)
            inputs[start:end] = evidence.cpu()
            targets[start:end] = target.cpu()
            masks[start:end] = coverage.cpu()
            print(f"{split}: rendered {min(start+args.batch_size,len(records))}/{len(records)}", flush=True)
        target = targets
        payload = {"inputs": inputs, "targets": target, "images": target,
                   "masks": masks, "records": records}
        for key in ("inputs", "targets", "masks"):
            x = payload[key]
            if not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
                raise RuntimeError(f"Invalid {key} values")
        io_started = time.perf_counter()
        payload["encoding"] = "uint16-patch32"
        for key in ("inputs", "targets", "masks"):
            payload[key] = (payload[key]*65535).round().to(torch.uint16)
        payload["images"] = payload["targets"]
        torch.save(payload, args.output/f"{split}.pt")
        _, expanded, expanded_mask = reconstruct_conditioning(target[:16], inputs[:16]*2-1, masks[:16])
        preview_data = {"inputs": (expanded+1)/2, "masks": expanded_mask, "targets": target[:16]}
        save_grid(target[:64], args.output/f"{split}-targets.png")
        preview_ids = list(range(min(16, len(records))))
        preview = torch.stack([preview_data[k][i].expand(3,-1,-1) for i in preview_ids
                               for k in ("inputs", "masks", "targets")])
        save_grid(preview, args.output/f"{split}-pairs.png", columns=3)
        io_seconds += time.perf_counter() - io_started
        report["splits"][split] = {"count": len(records), "mode_counts": dict(Counter(plan)),
                                   "span_min": min(r["span"] for r in records),
                                   "span_max": max(r["span"] for r in records),
                                   "depth_counts": dict(Counter(r["zoom_depth"] for r in records))}
        del payload, inputs, targets, masks, target, x
    report["performance"] = {"search_seconds": search_seconds, "io_seconds": io_seconds,
                             "total_seconds": time.perf_counter()-total_started,
                             "fp64_baseline_render_cpu_seconds": baseline_seconds if args.benchmark_baseline else None,
                             **backend.report}
    (args.output/"manifest.json").write_text(json.dumps(report, indent=2)+"\n")


def boundary_anchors(device, seed):
    from mandelbrot import GrayscaleRenderer, View

    renderer = GrayscaleRenderer(512, 512, device, side=1)
    mask = renderer.render(View(center=-0.7+0j, span=3, iterations=4096)).cpu()
    black = mask < 0.5
    edge = torch.zeros_like(black)
    edge[:, 1:] |= black[:, 1:] != black[:, :-1]
    edge[:, :-1] |= black[:, 1:] != black[:, :-1]
    edge[1:] |= black[1:] != black[:-1]
    edge[:-1] |= black[1:] != black[:-1]
    groups = {}
    for y, x in edge.nonzero().tolist():
        cx = -.7 + (x+.5-256)*3/512
        cy = (256-y-.5)*3/512
        # Mirror anchors stay together. Large image footprints may still overlap.
        group = f"x{math.floor((cx+2.2)/.2):02d}y{math.floor(abs(cy)/.2):02d}"
        groups.setdefault(group, []).append((cx, cy))
    names = sorted(groups)
    random.Random(seed).shuffle(names)
    val_count = max(1, len(names)//5)
    return {"train": {name: groups[name] for name in names[val_count:]},
            "val": {name: groups[name] for name in names[:val_count]}}


def zoom_evidence(parents, left, top):
    n = len(parents)
    rgb = torch.full((n, 3, 32, 32), .5, device=parents.device)
    mask = torch.zeros((n, 1, 32, 32), device=parents.device)
    for i, (x, y) in enumerate(zip(left, top)):
        x0, x1, y0, y1 = max(x, 0), min(x+32, 128), max(y, 0), min(y+32, 128)
        if x1 > x0 and y1 > y0:
            rgb[i, :, y0-y:y1-y, x0-x:x1-x] = parents[i, :, y0:y1, x0:x1]
            mask[i, :, y0-y:y1-y, x0-x:x1-x] = 1
    return rgb, mask

def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=1_000_000)
    parser.add_argument('--shard-size', type=int, default=10000)
    parser.add_argument('--overviews', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20261018)
    args = parser.parse_args()
    if args.count % args.shard_size or args.shard_size % 10 or not 0 <= args.overviews <= args.count:
        parser.error('Count must divide into shards; shard size must divide by 10; invalid overview quota')
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    config = dict(count=args.count, shard_size=args.shard_size, overviews=args.overviews, seed=args.seed,
                  palette='smooth', size=128, samples_per_pixel=4, iterations=8192,
                  camera_tolerance=.01, train_fraction=.9, max_depth=12, storage='uint16-patch32',
                  mode_proportions=dict(interior=.60, edge=.15, corner=.15, unconditional=.10))
    manifest_path = args.output/'manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous['config'] != config:
            raise ValueError('Resume configuration differs from existing dataset')
        if previous.get('complete'):
            if not all((args.output / item['path']).is_file() for shard in previous['shards'] for item in shard['files'].values()):
                raise ValueError('Completed dataset has missing shards')
            print('Already complete:', args.output)
            return
    state={'diversity': ViewDiversity(.01)}
    # Fix the anchor split once for all shards, including resumed shards.
    state['groups']=boundary_anchors('cuda',args.seed)
    shards=[]
    started=time.time()
    source_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                   for name in ('data/prepare.py','mandelbrot.py')}
    n_shards=args.count//args.shard_size
    overview_train=round(args.overviews*.9)
    overview_val=args.overviews-overview_train
    for index in range(n_shards):
        final=args.output/f'shard-{index:04d}'
        reused=final.exists()
        if not reused:
            pending=args.output/f'.shard-{index:04d}.partial'
            if pending.exists():
                pending.rename(args.output/f'.shard-{index:04d}.interrupted-{time.time_ns()}')
            opts=SimpleNamespace(output=pending,train_count=args.shard_size*9//10,
                val_count=args.shard_size//10,max_depth=12,iterations=8192,
                seed=args.seed+index,render_mode='float64',palette='smooth',
                aa_side=2,batch_size=128,benchmark_baseline=False,diverse=True,
                train_overview_count=overview_train//n_shards+(index<overview_train%n_shards),
                val_overview_count=overview_val//n_shards+(index<overview_val%n_shards))
            print(f'SHARD {index+1}/{n_shards} starting',flush=True)
            prepare_zoom_dataset(opts,sampling_state=state)
            pending.rename(final)
        report=json.loads((final/'manifest.json').read_text())
        files={}
        for split in ('train','val'):
            file=final/f'{split}.pt'
            payload=torch.load(file,weights_only=True,mmap=True,map_location='cpu')
            expected=args.shard_size*(9 if split=='train' else 1)//10
            assert len(payload['records'])==len(payload['targets'])==expected
            if reused:
                for r in payload['records']:
                    if not state['diversity'].add(r['center'],r['span']):
                        raise RuntimeError('Duplicate camera in completed shards')
                    state.setdefault('seen',set()).add((*r['center'],r['span']))
            with file.open('rb') as stream:
                digest=hashlib.file_digest(stream,'sha256').hexdigest()
            files[split]=dict(path=str(file.relative_to(args.output)),count=expected,sha256=digest)
            del payload
        shards.append(dict(index=index,files=files,distribution=report['splits']))
        summary=dict(config=config,completed_count=(index+1)*args.shard_size,
                     complete=index+1==n_shards,shards=shards,source_sha256=source_hashes,
                     updated_unix=time.time(),session_seconds=time.time()-started,
                     split_policy=report['split_policy'],
                     deduplication='Reject cameras within 1% of view width and 1% of scale across all shards and splits. This is not a guarantee of perceptually unique fractal structures.')
        atomic_json(manifest_path,summary)
        print(f'COMMITTED {summary["completed_count"]}/{args.count}',flush=True)
    print('COMPLETE',args.output,flush=True)


if __name__ == "__main__":
    main()
