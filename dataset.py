"""Prepare, load, preview, and export Mandelbrot image datasets."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import time
import zipfile

import torch
from torch.nn import functional as F


def load_images(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    images = payload["images"]
    if images.ndim != 4 or images.shape[1] not in (1, 3) or len(images) == 0:
        raise ValueError("Dataset must be nonempty [N,C,H,W] with C=1 or C=3")
    if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError("Dataset values must be finite in [0,1]")
    return images.float() * 2 - 1


def load_zoom_pairs(path):
    """Load aligned upsampled inputs and targets, stored as floats in [0,1]."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if "masks" in payload and not (payload["masks"] == 1).all():
        raise ValueError("This dataset has partial/no-evidence samples; use mask-aware training instead of the legacy paired loader")
    inputs, targets = payload["inputs"], payload["targets"]
    if inputs.shape != targets.shape or inputs.ndim != 4 or len(inputs) == 0 or inputs.shape[1] != 3:
        raise ValueError("Zoom pairs must be matching nonempty [N,3,H,W] tensors")
    for images in (inputs, targets):
        if not images.is_floating_point() or not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
            raise ValueError("Zoom pairs must be finite floating-point tensors in [0,1]")
    # One tensor keeps random indices aligned through training and validation.
    return torch.stack((inputs, targets), dim=1).float() * 2 - 1


def save_grid(images, path, columns=8):
    from PIL import Image
    images = images.detach().cpu().clamp(0, 1)
    if images.ndim == 3:
        images = images[:, None]
    count, channels, height, width = images.shape
    columns = min(count, columns)
    rows = math.ceil(count / columns)
    canvas = torch.ones((channels, rows * height, columns * width))
    for index, image in enumerate(images):
        row, col = divmod(index, columns)
        canvas[:, row*height:(row+1)*height, col*width:(col+1)*width] = image
    pixels = (canvas * 255).round().to(torch.uint8)
    pixels = pixels[0] if channels == 1 else pixels.permute(1, 2, 0)
    Image.fromarray(pixels.numpy()).save(path)


def conditioning_patch_mask(records):
    """Coverage of parent pixels retained in the 32x32 conditioning crop."""
    if any(r.get('zoom_factor') != 4 or r.get('crop_size') != 32 or
           type(r.get('crop_left')) is not int or type(r.get('crop_top')) is not int
           for r in records):
        raise ValueError('Patch32 requires integer 32-pixel crops and 4x zoom')
    x = torch.tensor([r['crop_left'] for r in records])[:, None]+torch.arange(32)
    y = torch.tensor([r['crop_top'] for r in records])[:, None]+torch.arange(32)
    present = torch.tensor([r['mode'] != 'unconditional' for r in records])
    return (((y[:, :, None] >= 0) & (y[:, :, None] < 128) &
             (x[:, None, :] >= 0) & (x[:, None, :] < 128)) & present[:, None, None])[:, None]


def reconstruct_conditioning(target, evidence, mask):
    """Expand normalized compact evidence with clamped crop edges on the GPU."""
    if evidence.shape[-2:] == target.shape[-2:]:
        return target, evidence, mask
    size = target.shape[-2:]
    weights = F.interpolate(mask, size=size, mode='bilinear', align_corners=False)
    rgb = F.interpolate(evidence*mask, size=size, mode='bilinear', align_corners=False)
    rgb = torch.where(weights > 1e-6, rgb/weights.clamp_min(1e-6), 0.)
    return target, rgb, weights


class ZoomDataset:
    """Memory-map mixed zoom pairs; transfer only selected batches to CUDA."""

    def __init__(self, path):
        self.data = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        self.inputs, self.targets = self.data['inputs'], self.data['targets']
        self.records = self.data["records"]
        self.encoding = self.data.get('encoding', 'float32')
        if self.encoding == 'uint16-patch32':
            self.masks = conditioning_patch_mask(self.records).to(torch.int32).mul_(65535).to(torch.uint16)
        else:
            self.masks = self.data['masks']
        if self.encoding not in ('float32', 'uint16-unorm', 'uint16-patch32'):
            raise ValueError('Unknown zoom tensor encoding')
        if self.encoding.startswith('uint16-') and any(v.dtype != torch.uint16 for v in (self.inputs, self.targets, self.masks)):
            raise ValueError('Quantized zoom data requires uint16 tensors')
        n, c, h, w = self.targets.shape
        input_shape = (n, 3, 32, 32) if self.encoding == 'uint16-patch32' else self.targets.shape
        mask_shape = (n, 1, *input_shape[-2:])
        if self.encoding == 'uint16-patch32' and (h, w) != (128, 128):
            raise ValueError('Patch32 requires 128x128 targets')
        if not n or c != 3 or self.inputs.shape != input_shape or self.masks.shape != mask_shape or len(self.records) != n:
            raise ValueError("Invalid mixed zoom dataset shapes")

    def __len__(self):
        return len(self.targets)

    def batch(self, indices, device="cuda"):
        values = [x[indices].to(device=device, dtype=torch.float32) for x in (self.targets, self.inputs, self.masks)]
        if self.encoding.startswith('uint16-'):
            values = [v.div_(65535) for v in values]
        if any(not x.isfinite().all() or x.min() < 0 or x.max() > 1 for x in values):
            raise ValueError("Dataset values must be finite and in [0,1]")
        target, evidence, mask = values
        return reconstruct_conditioning(target*2-1, evidence*2-1, mask)


class ShardedZoomDataset:
    """Manifest-backed metadata and sparse reads for held-out evaluation."""

    def __init__(self, root, split):
        self.root = Path(root)
        manifest = json.loads((self.root/'manifest.json').read_text())
        if not manifest.get('complete'):
            raise ValueError('Training requires a complete dataset manifest')
        self.files = [s['files'][split] for s in manifest['shards']]
        if not self.files or any(item['count'] <= 0 for item in self.files):
            raise ValueError('Dataset must have nonempty shards')
        self.ends = []
        total = 0
        for item in self.files:
            total += item['count']
            self.ends.append(total)
        self._records = None

    def __len__(self):
        return self.ends[-1]

    @property
    def records(self):
        if self._records is None:
            self._records = []
            for item in self.files:
                data = ZoomDataset(self.root/item['path'])
                if len(data) != item['count']:
                    raise ValueError('Shard count differs from manifest')
                self._records.extend(data.records)
        return self._records

    def batch(self, indices, device='cuda'):
        import bisect
        groups = {}
        for position, index in enumerate(indices):
            index = int(index)
            if not 0 <= index < len(self):
                raise IndexError(index)
            shard = bisect.bisect_right(self.ends, index)
            start = self.ends[shard-1] if shard else 0
            groups.setdefault(shard, []).append((position, index-start))
        output = None
        for shard, pairs in groups.items():
            positions, local = zip(*pairs)
            values = ZoomDataset(self.root/self.files[shard]['path']).batch(list(local), device)
            if output is None:
                output = [v.new_empty((len(indices), *v.shape[1:])) for v in values]
            for target, value in zip(output, values):
                target[list(positions)] = value
        return tuple(output)


def zoom_modes(count):
    names = ('interior', 'edge', 'corner', 'unconditional')
    expected = [count*p for p in (.60, .15, .15, .10)]
    quotas = [int(n) for n in expected]
    for i in sorted(range(4), key=lambda i: expected[i]-quotas[i], reverse=True)[:count-sum(quotas)]:
        quotas[i] += 1
    return [name for name, n in zip(names, quotas) for _ in range(n)]


def training_order(records, seed, unconditional_fraction=None):
    """Shuffle a shard; optionally thin modes to a desired unconditional share."""
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(records), generator=generator)
    if unconditional_fraction is None:
        return order
    fraction = unconditional_fraction
    if not 0 <= fraction <= 1:
        raise ValueError('Unconditional fraction must be in [0,1]')
    unconditional = torch.tensor([r['mode'] == 'unconditional' for r in records])
    u, c = order[unconditional[order]], order[~unconditional[order]]
    if fraction == 0:
        return c
    if fraction == 1:
        return u
    if not len(u) or not len(c):
        raise ValueError('Requested mix requires both conditional and unconditional examples')
    if len(u)/len(records) > fraction:
        u = u[:round(len(c)*fraction/(1-fraction))]
    else:
        c = c[:round(len(u)*(1-fraction)/fraction)]
    selected = torch.cat((c, u))
    return selected[torch.randperm(len(selected), generator=generator)]


class ShardBatches:
    """Shuffle shards and their rows; prefetch one shard while CUDA trains.

    The current and prefetched normalized float32 shards live on the GPU. Pinned CPU staging retains
    exact source values; no quantization or dataset rewriting. Batches crossing
    shard/epoch boundaries are joined, so no examples are dropped. State refers
    to consumed rows, never prefetched rows, making checkpoint resume exact.
    """

    def __init__(self, root, batch_size, seed, device='cuda', state=None, unconditional_fraction=None,
                 select_rows=False, staging_rows=0):
        from concurrent.futures import ThreadPoolExecutor
        self.dataset = ShardedZoomDataset(root, 'train')
        if unconditional_fraction is not None and not 0 <= unconditional_fraction <= 1:
            raise ValueError('Unconditional fraction must be in [0,1]')
        self.unconditional_fraction = unconditional_fraction
        self.select_rows = select_rows
        if type(staging_rows) is not int or staging_rows < 0:
            raise ValueError("Staging rows must be nonnegative")
        self.staging_rows = staging_rows
        self.staging_events = [None, None]
        if batch_size <= 0:
            raise ValueError('Batch size must be positive')
        self.batch_size, self.seed, self.device = batch_size, seed, torch.device(device)
        self.epoch, self.shard_cursor, self.offset = 0, 0, 0
        if state:
            self.epoch, self.shard_cursor, self.offset = (state[k] for k in ('epoch', 'shard_cursor', 'offset'))
        if self.epoch < 0 or not 0 <= self.shard_cursor < len(self.dataset.files):
            raise ValueError('Invalid sampler cursor')
        count = self.dataset.files[self._key()[1]]['count']
        if not 0 <= self.offset < count:
            raise ValueError('Invalid sampler offset')
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='zoom-prefetch')
        self.values = None
        self.loaded_key = None
        self.validated = set()
        self.staging = [None, None]
        self.read_index = 0
        self.read_timings = []
        self.copy_stream = torch.cuda.Stream(device=self.device) if self.device.type == 'cuda' else None
        self.future = self.pool.submit(self._read, self._key())
        self.future_key = self._key()
        self.load_seconds = 0.

    def __len__(self):
        return len(self.dataset)

    def _order(self, epoch):
        generator = torch.Generator().manual_seed(self.seed+epoch)
        return torch.randperm(len(self.dataset.files), generator=generator).tolist()

    def _key(self):
        return self.epoch, self._order(self.epoch)[self.shard_cursor]

    def _next_key(self):
        cursor, epoch = self.shard_cursor+1, self.epoch
        if cursor == len(self.dataset.files):
            cursor, epoch = 0, epoch+1
        return epoch, self._order(epoch)[cursor]

    def _read(self, key):
        shard = key[1]
        item = self.dataset.files[shard]
        data = ZoomDataset(self.dataset.root/item['path'])
        if len(data) != item['count']:
            raise ValueError('Shard count differs from manifest')
        start = time.perf_counter()
        source = (data.targets, data.inputs, data.masks)
        order = training_order(data.records, self.seed+1000003*(key[0]+1)+key[1],
                               self.unconditional_fraction)
        if not len(order):
            raise ValueError('Training mixture selected no rows')
        selected = None
        if self.select_rows and len(order) < len(data):
            # Read selected rows in file order, retaining the original shuffle.
            selected, permutation = order.sort()
            order = permutation.argsort()
        shapes = [(len(selected) if selected is not None else len(v), *v.shape[1:]) for v in source]
        from contextlib import nullcontext
        context = torch.cuda.stream(self.copy_stream) if self.copy_stream is not None else nullcontext()
        with context:
            if self.staging_rows:
                gpu = self._stage_chunks(source, selected, shapes)
            else:
                slot = self.read_index % 2
                self.read_index += 1
                values = self.staging[slot]
                if values is None or any(a.shape != shape or a.dtype != v.dtype for a, shape, v in zip(values, shapes, source)):
                    values = [torch.empty(shape, dtype=v.dtype, pin_memory=self.device.type == 'cuda') for v, shape in zip(source, shapes)]
                    self.staging[slot] = values
                for destination, value in zip(values, source):
                    if selected is None:
                        destination.copy_(value)
                    else:
                        if value.dtype == torch.uint16:
                            torch.index_select(value.view(torch.int16), 0, selected, out=destination.view(torch.int16))
                        else:
                            torch.index_select(value, 0, selected, out=destination)
                gpu = tuple(v.to(self.device, non_blocking=True) if self.device.type == 'cuda' else v.clone() for v in values)
            if data.encoding.startswith('uint16-'):
                gpu = tuple(v.float().div_(65535) for v in gpu)
            copied = time.perf_counter()
            self.read_timings.append(dict(shard=shard, staging_enqueue=copied-start))
            if selected is not None or shard not in self.validated:
                for value in gpu:
                    low, high = torch.aminmax(value)
                    if not (low >= 0 and high <= 1):
                        raise ValueError('Dataset values must be finite and in [0,1]')
                self.validated.add(shard)
            for value in gpu[:2]:
                value.mul_(2).sub_(1)
        if self.copy_stream is not None:
            self.copy_stream.synchronize()
        self.read_timings[-1]['transfer_validate_normalize'] = time.perf_counter()-copied
        return gpu, order

    def _stage_chunks(self, source, selected, shapes):
        """Overlap sequential CPU reads with H2D using two event-protected slots.

        Full GPU shards retain random row access; only host staging is chunked.
        Events guard pinned source reuse, not consumer tensor lifetime.
        """
        cuda = self.device.type == 'cuda'
        gpu = tuple(torch.empty(shape, dtype=v.dtype, device=self.device) for v, shape in zip(source, shapes))
        for begin in range(0, shapes[0][0], self.staging_rows):
            end = min(begin+self.staging_rows, shapes[0][0])
            slot = self.read_index % 2
            self.read_index += 1
            if self.staging_events[slot] is not None:
                self.staging_events[slot].synchronize()
            needed = [(self.staging_rows, *v.shape[1:]) for v in source]
            buffers = self.staging[slot]
            if buffers is None or any(buf.shape != shape or buf.dtype != value.dtype for buf, shape, value in zip(buffers, needed, source)):
                buffers = [torch.empty(shape, dtype=v.dtype, pin_memory=cuda) for v, shape in zip(source, needed)]
                self.staging[slot] = buffers
            for target, buf, value in zip(gpu, buffers, source):
                host = buf[:end-begin]
                if selected is None:
                    host.copy_(value[begin:end])
                else:
                    if value.dtype == torch.uint16:
                        torch.index_select(value.view(torch.int16), 0, selected[begin:end], out=host.view(torch.int16))
                    else:
                        torch.index_select(value, 0, selected[begin:end], out=host)
                target[begin:end].copy_(host, non_blocking=cuda)
            if cuda:
                if self.staging_events[slot] is None:
                    self.staging_events[slot] = torch.cuda.Event()
                self.staging_events[slot].record(self.copy_stream)
        return gpu

    def _load(self):
        key = self._key()
        if self.loaded_key == key:
            return
        start = time.perf_counter()
        if self.future_key != key:
            raise RuntimeError('Prefetch cursor disagrees with consumed data')
        self.values, order = self.future.result()
        if self.device.type == 'cuda':
            for value in self.values:
                value.record_stream(torch.cuda.current_stream(self.device))
        self.indices = order.to(self.device)
        if self.offset >= len(self.indices):
            raise ValueError('Sampler offset exceeds selected shard rows')
        self.loaded_key = key
        self.future_key = self._next_key()
        self.future = self.pool.submit(self._read, self.future_key)
        self.load_seconds += time.perf_counter()-start

    def __next__(self):
        remaining, pieces = self.batch_size, []
        while remaining:
            self._load()
            count = min(remaining, len(self.indices)-self.offset)
            ids = self.indices[self.offset:self.offset+count]
            pieces.append(reconstruct_conditioning(*(v[ids] for v in self.values)))
            self.offset += count
            remaining -= count
            if self.offset == len(self.indices):
                self.offset = 0
                self.shard_cursor += 1
                if self.shard_cursor == len(self.dataset.files):
                    self.shard_cursor = 0
                    self.epoch += 1
        if len(pieces) == 1:
            return pieces[0]
        return tuple(torch.cat([piece[i] for piece in pieces]) for i in range(3))

    def state_dict(self):
        return dict(epoch=self.epoch, shard_cursor=self.shard_cursor, offset=self.offset)

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.future = None
        self.values = None
        self.staging = [None, None]
        self.staging_events = [None, None]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def zoom_evidence(parents, left, top, factor=4):
    """Reproject a possibly out-of-frame crop; return RGB and soft coverage.

    left/top are pixel-boundary coordinates. Normalize partial interpolation
    support so zero padding does not create black fringes in observed evidence.
    """
    n, _, height, width = parents.shape
    left = torch.as_tensor(left, device=parents.device, dtype=torch.float32).reshape(n, 1, 1)
    top = torch.as_tensor(top, device=parents.device, dtype=torch.float32).reshape(n, 1, 1)
    yy, xx = torch.meshgrid(torch.arange(height, device=parents.device),
                            torch.arange(width, device=parents.device), indexing="ij")
    gx = 2 * (left + (xx + .5) / factor) / width - 1
    gy = 2 * (top + (yy + .5) / factor) / height - 1
    grid = torch.stack((gx, gy), dim=-1)
    weights = F.grid_sample(torch.ones_like(parents[:, :1]), grid, align_corners=False)
    samples = F.grid_sample(parents, grid, align_corners=False)
    evidence = torch.where(weights > 1e-6, samples / weights.clamp_min(1e-6), .5)
    return evidence, weights


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
    from mandelbrot_fast import FastRenderer, DatasetRenderer

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
              "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                for name in ("dataset.py", "mandelbrot.py", "mandelbrot_fast.py")},
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
        inputs = torch.empty(len(records), 3, 128, 128)
        targets = torch.empty_like(inputs)
        masks = torch.empty(len(records), 1, 128, 128)
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
        torch.save(payload, args.output/f"{split}.pt")
        save_grid(target[:64], args.output/f"{split}-targets.png")
        preview_ids = list(range(min(16, len(records))))
        preview = torch.stack([payload[k][i].expand(3,-1,-1) for i in preview_ids
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


def sampling_plan(count, rng):
    # Exact marginal quotas, with randomized pairing and ordering.
    coverage_bins = [index % 6 for index in range(count)]
    proportions = (.10, .25, .35, .30)
    scale_bins = []
    for index, proportion in enumerate(proportions):
        amount = round(count*proportion) if index < 3 else count-len(scale_bins)
        scale_bins.extend([index]*amount)
    rng.shuffle(coverage_bins)
    rng.shuffle(scale_bins)
    return list(zip(coverage_bins, scale_bins))


def generate_balanced_split(count, groups, renderer, rng, iterations, tolerance,
                            max_iterations, seen):
    from mandelbrot import GrayscaleRenderer, View

    spans = ((.05, .12), (.12, .35), (.35, 1.0), (1.0, 2.5))
    pilot = GrayscaleRenderer(24, 24, str(renderer.device), side=1)
    group_names = list(groups)
    images, records = [], []
    rejected = Counter()
    attempts = 0
    for coverage_bin, scale_bin in sampling_plan(count, rng):
        low_fraction = .2 + coverage_bin*.1
        high_fraction = low_fraction + .1
        for trial in range(4000):
            attempts += 1
            group = rng.choice(group_names)
            ax, ay = rng.choice(groups[group])
            span = math.exp(rng.uniform(*map(math.log, spans[scale_bin])))
            cx = ax + rng.uniform(-.45, .45)*span
            cy = ay + rng.uniform(-.45, .45)*span
            view = View(complex(cx, cy), span, iterations)
            approx = 1-pilot.render(view).mean().item()
            if not low_fraction-.04 <= approx <= high_fraction+.04:
                rejected["pilot_coverage"] += 1
                continue
            low = renderer.render(view).clone()
            fraction = 1-low.mean().item()
            if not low_fraction <= fraction < high_fraction:
                rejected["coverage"] += 1
                continue
            budget = iterations
            while budget*2 <= max_iterations:
                budget *= 2
                high = renderer.render(View(view.center, span, budget)).clone()
                delta = (high-low).abs()
                change = delta.mean().item()
                if change <= tolerance:
                    break
                low = high
            if change > tolerance:
                rejected["iteration_change"] += 1
                continue
            fraction = 1-high.mean().item()
            if not low_fraction <= fraction < high_fraction:
                rejected["coverage_after_refinement"] += 1
                continue
            cpu_image = high.cpu().unsqueeze(0)
            key = hashlib.sha256((cpu_image*255).round().byte().numpy().tobytes()).hexdigest()
            if key in seen:
                rejected["duplicate"] += 1
                continue
            seen.add(key)
            images.append(cpu_image)
            records.append({"region": group, "center": [cx, cy], "span": span,
                            "anchor": [ax, ay], "iterations": budget,
                            "aa_side": renderer.side, "black_fraction": fraction,
                            "coverage_bin": coverage_bin, "scale_bin": scale_bin,
                            "iteration_delta_mean": change,
                            "iteration_changed_pixels": (delta > 0).float().mean().item()})
            break
        else:
            raise RuntimeError(f"Could not fill coverage bin {coverage_bin}, scale bin {scale_bin}")
        if len(images) % 64 == 0:
            print(f"accepted {len(images)}/{count}; attempts={attempts}", flush=True)
    fractions = [r["black_fraction"] for r in records]
    quality = {"attempts": attempts, "rejected": dict(rejected),
               "black_fraction_min": min(fractions), "black_fraction_max": max(fractions),
               "black_fraction_mean": sum(fractions)/len(fractions),
               "coverage_bin_counts": dict(Counter(r["coverage_bin"] for r in records)),
               "scale_bin_counts": dict(Counter(r["scale_bin"] for r in records)),
               "iteration_counts": dict(Counter(r["iterations"] for r in records))}
    return torch.stack(images), records, quality


def prepare_dataset(args):
    from mandelbrot import GrayscaleRenderer

    if min(args.train_count, args.val_count, args.size, args.iterations) < 1 or args.tolerance < 0:
        raise ValueError("Counts, size and iterations must be positive; tolerance must be nonnegative")
    if args.max_iterations < args.iterations*2:
        raise ValueError("Maximum iteration budget must permit at least one doubling")
    if args.output.exists():
        raise ValueError("Output already exists; choose a new directory")
    rng = random.Random(args.seed)
    groups = boundary_anchors(args.device, args.seed)
    splits = {name: [{"id": group, "anchor_count": len(points)} for group, points in collection.items()]
              for name, collection in groups.items()}
    renderer = GrayscaleRenderer(args.size, args.size, args.device, args.aa_side)
    args.output.mkdir(parents=True)
    report = {"config": {**vars(args), "output": str(args.output)},
              "target": "white escaped fraction; black unescaped at adaptively checked iteration budget",
              "sampling": "boundary anchors, six equal black-coverage bins from .2 to .8, four zoom ranges",
              "scale_ranges": [[.05,.12],[.12,.35],[.35,1.0],[1.0,2.5]],
              "scale_proportions": [.10,.25,.35,.30],
              "split_policy": "disjoint anchor groups with conjugate pairs grouped together; crop footprints can overlap",
              "regions": splits, "quality": {},
              "source_sha256": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                                for name in ("dataset.py", "mandelbrot.py")}}
    seen = set()
    for name, count in (("train", args.train_count), ("val", args.val_count)):
        images, records, quality = generate_balanced_split(count, groups[name], renderer, rng,
                                                          args.iterations, args.tolerance,
                                                          args.max_iterations, seen)
        torch.save({"images": images, "records": records}, args.output / f"{name}.pt")
        save_grid(images[:64], args.output / f"{name}-preview.png")
        report["quality"][name] = quality
        print(f"{name}: {tuple(images.shape)}, {quality}", flush=True)
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")


def contact_sheet(images, labels, path, columns, scale=1):
    from PIL import Image, ImageDraw

    width, height = images[0].size
    gap, label_height = 4, 14
    tile_width, tile_height = width*scale+gap, height*scale+label_height+gap
    rows = (len(images)+columns-1)//columns
    canvas = Image.new("L", (columns*tile_width+gap, rows*tile_height+gap), 210)
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(zip(images, labels)):
        row, col = divmod(index, columns)
        x, y = gap+col*tile_width, gap+row*tile_height
        draw.text((x, y), label, fill=0)
        enlarged = image.resize((width*scale, height*scale), Image.Resampling.NEAREST)
        canvas.paste(enlarged, (x, y+label_height))
    canvas.save(path)


def export_dataset(dataset):
    from PIL import Image

    image_root = dataset / "images"
    archive = dataset.with_suffix(".zip")
    if image_root.exists() or archive.exists():
        raise ValueError("Export already exists; refusing to overwrite")
    manifest = json.loads((dataset / "manifest.json").read_text())
    groups = {split: {r["id"] for r in manifest["regions"][split]} for split in ("train", "val")}
    assert not groups["train"] & groups["val"]
    images, labels, records, hashes = [], [], [], []
    counts = {}
    for split in ("train", "val"):
        payload = torch.load(dataset / f"{split}.pt", map_location="cpu", weights_only=True)
        tensors = payload["images"]
        assert tensors.ndim == 4 and tensors.shape[1] == 1
        assert torch.isfinite(tensors).all() and tensors.min() >= 0 and tensors.max() <= 1
        assert len(tensors) == len(payload["records"])
        counts[split] = len(tensors)
        folder = image_root / split
        folder.mkdir(parents=True)
        for index, (tensor, record) in enumerate(zip(tensors, payload["records"])):
            assert record["region"] in groups[split]
            assert record["iteration_delta_mean"] <= manifest["config"]["tolerance"]
            filename = f"{split}-{index:04d}.png"
            pixels = (tensor[0]*255).round().to(torch.uint8).numpy()
            image = Image.fromarray(pixels)
            image.save(folder / filename)
            images.append(image)
            labels.append(f"{split} {index:04d}")
            records.append({"file": f"images/{split}/{filename}", **record})
            hashes.append(hashlib.sha256(pixels.tobytes()).hexdigest())
    assert sum(counts.values()) == manifest["config"]["train_count"]+manifest["config"]["val_count"]
    (dataset / "images.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
    selected = random.Random(20261015).sample(range(len(images)), min(64, len(images)))
    contact_sheet([images[i] for i in selected], [labels[i] for i in selected],
                  dataset / "preview-64.png", columns=8, scale=2)
    contact_sheet(images, labels, dataset / "all-images.png", columns=25)
    summary = {"counts": counts, "total": len(images), "size": list(images[0].size),
               "unique_quantized_images": len(set(hashes)), "png_mode": "L",
               "preview_policy": "64 deterministic random examples, nearest-neighbor enlargement for viewing",
               "gray_pixels": "antialiased coverage; tensor data retain full float32 precision"}
    (dataset / "export.json").write_text(json.dumps(summary, indent=2)+"\n")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(dataset.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(dataset.parent))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved {archive}; {archive.stat().st_size:,} bytes", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Generate balanced Mandelbrot crops")
    prepare.add_argument("--output", type=Path, default=Path("data/bw64"))
    prepare.add_argument("--train-count", type=int, default=512)
    prepare.add_argument("--val-count", type=int, default=64)
    prepare.add_argument("--size", type=int, default=64)
    prepare.add_argument("--aa-side", type=int, default=4)
    prepare.add_argument("--iterations", type=int, default=1024)
    prepare.add_argument("--max-iterations", type=int, default=16384)
    prepare.add_argument("--tolerance", type=float, default=0.0005)
    prepare.add_argument("--seed", type=int, default=20261014)
    prepare.add_argument("--device", default="cuda")
    zoom = commands.add_parser("prepare-zoom", help="Generate RGB 4x zoom evidence, coverage masks and targets")
    zoom.add_argument("--output", type=Path, default=Path("data/zoom4x-fast-128"))
    zoom.add_argument("--train-count", type=int, default=900)
    zoom.add_argument("--val-count", type=int, default=100)
    zoom.add_argument("--max-depth", type=int, default=12, help="Absolute zoom depths 0..N; magnification is 4**depth")
    zoom.add_argument("--iterations", type=int, default=8192)
    zoom.add_argument("--seed", type=int, default=20261016)
    zoom.add_argument("--render-mode", choices=("auto", "float64", "hybrid32", "perturb32", "perturb64"), default="auto")
    zoom.add_argument("--aa-side", type=int, default=2)
    zoom.add_argument("--batch-size", type=int, default=32)
    zoom.add_argument("--benchmark-baseline", action="store_true")
    zoom.add_argument("--palette", choices=("cosine", "smooth"), default="cosine")
    export = commands.add_parser("export", help="Export dataset PNGs, previews, and ZIP")
    export.add_argument("dataset", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            prepare_dataset(args)
        elif args.command == "prepare-zoom":
            prepare_zoom_dataset(args)
        else:
            export_dataset(args.dataset)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
