"""Train JiT on full, partial, and absent 4x zoom evidence."""
import argparse
import copy
import csv
import hashlib
import json
import math
import runpy
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from model import build_model

MODES = ("interior", "edge", "corner", "unconditional")

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


def select_views(records, per_mode=64, seed=17):
    generator = torch.Generator().manual_seed(seed)
    result = []
    for mode in MODES:
        buckets = {}
        for i, record in enumerate(records):
            if record['mode'] == mode:
                buckets.setdefault(record['zoom_depth'], []).append(i)
        for depth, ids in buckets.items():
            buckets[depth] = [ids[i] for i in torch.randperm(len(ids), generator=generator).tolist()]
        chosen = []
        while len(chosen) < per_mode and any(buckets.values()):
            for depth in sorted(buckets):
                if buckets[depth] and len(chosen) < per_mode:
                    chosen.append(buckets[depth].pop())
        result.extend(chosen)
    return result


def train_masked(config, resume=None, max_seconds=None):
    """Train one clean-image JiT on full, partial, and absent evidence."""
    from model import ConditionalFlow

    torch.set_num_threads(4)
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(config['device'])
    root = Path(config['data'])
    sharded = (root/'manifest.json').exists()
    stream = None
    if sharded:
        train = ShardedZoomDataset(root, 'train')
        val = ShardedZoomDataset(root, 'val')
        # Manifest includes generator-recorded per-file SHA256 values. Do not
        # rescan 460 GB at every startup; validate shard contents as they load.
        hashes = {'manifest': hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()}
    else:
        if config.get('unconditional_fraction') is not None:
            raise ValueError('Training mixture selection currently requires a sharded dataset')
        train = ZoomDataset(root/'train.pt')
        val = ZoomDataset(root/'val.pt')
        if train.targets.shape[1:] != (3, config['size'], config['size']) or val.targets.shape[1:] != train.targets.shape[1:]:
            raise ValueError('Dataset shape disagrees with config')
        hashes = {}
        for split in ('train', 'val'):
            with (root/f'{split}.pt').open('rb') as file:
                hashes[split] = hashlib.file_digest(file, 'sha256').hexdigest()
    if config['architecture'] != 'jit' or config['model'].get('mask_channels') != 1:
        raise ValueError('Conditional flow requires mask-aware JiT')
    model = build_model(config).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    flow = ConditionalFlow(**config.get('flow', {}))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config.get('weight_decay', .01),
                                  betas=tuple(config.get('betas', (.9, .999))), fused=True)
    start = 0
    if resume:
        saved = torch.load(resume, map_location=device, weights_only=True)
        for key in ('architecture','model','size','formulation','learning_rate','ema_decay','batch_size','seed','flow','warmup_steps','compile','compile_mode','betas','weight_decay','unconditional_fraction'):
            if saved['config'].get(key) != config.get(key):
                raise ValueError(f'Resume config mismatch: {key}')
        if saved['dataset_sha256'] != hashes:
            raise ValueError('Resume dataset differs from checkpoint')
        model.load_state_dict(saved['model']); ema.load_state_dict(saved['ema'])
        optimizer.load_state_dict(saved['optimizer'])
        start = saved['step']
        torch.set_rng_state(saved['cpu_rng'].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in saved['cuda_rng']])
    if sharded:
        stream = ShardBatches(root, config['batch_size'], config['seed'], device,
                              saved.get('sampler') if resume else None,
                              unconditional_fraction=config.get('unconditional_fraction'),
                              select_rows=config.get('select_rows', False),
                              staging_rows=config.get('staging_rows', 0))
        if resume and 'sampler' not in saved:
            stream.close()
            raise ValueError('Sharded resume requires sampler state')
    def training_loss(clean, evidence, mask):
        return flow.loss(model, clean, evidence, mask)
    if config.get('compile', False):
        training_loss = torch.compile(training_loss, fullgraph=True, mode=config.get('compile_mode', 'default'))
    try:
        if start >= config['steps']:
            raise ValueError('Total steps must exceed checkpoint step')
        output = Path(config['output']); output.mkdir(parents=True, exist_ok=True)
        (output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
        parameters = sum(p.numel() for p in model.parameters())
        # Fixed evaluation selection, optionally balanced over both mode and depth.
        modes = ('interior', 'edge', 'corner', 'unconditional')
        if config.get('stratify_depth', False):
            validation_ids = select_views(val.records, per_mode=64, seed=config['seed']+1)
            preview_ids = []
            for mode in modes:
                chosen = [i for i in validation_ids if val.records[i]['mode'] == mode]
                preview_ids.extend(chosen[::max(1, len(chosen)//4)][:4])
        else:
            rng = torch.Generator().manual_seed(config['seed']+1)
            validation_ids, preview_ids = [], []
            for mode in modes:
                ids = [i for i,r in enumerate(val.records) if r['mode']==mode]
                order = torch.randperm(len(ids), generator=rng).tolist()
                chosen = [ids[i] for i in order[:64]]
                validation_ids.extend(chosen); preview_ids.extend(chosen[:4])
        (output/'validation-selection.json').write_text(json.dumps({
            'ids': validation_ids, 'records': [val.records[i] for i in validation_ids]}, indent=2)+'\n')
        if not validation_ids:
            raise ValueError('No validation samples in supported modes')

        # Reuse held-out CUDA tensors instead of repeating sparse disk reads.
        validation_batch = val.batch(validation_ids, device)
        preview_batch = val.batch(preview_ids, device)
        if validation_batch[0].shape[1:] != (3, config['size'], config['size']):
            raise ValueError('Dataset shape disagrees with config')

        @torch.no_grad()
        def evaluate(network):
            was_training = network.training; network.eval()
            generator = torch.Generator(device=device).manual_seed(config['seed']+2)
            total = 0.
            try:
                for offset in range(0,len(validation_ids),config['batch_size']):
                    ids=validation_ids[offset:offset+config['batch_size']]
                    clean,evidence,mask=(v[offset:offset+len(ids)] for v in validation_batch)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        total += flow.loss(network,clean,evidence,mask,generator)*len(ids)
                return (total/len(validation_ids)).item()
            finally:
                network.train(was_training)

        @torch.no_grad()
        def preview(step):
            clean,evidence,mask=preview_batch
            with torch.autocast('cuda',dtype=torch.bfloat16):
                one=flow.sample(ema,evidence,mask,steps=1,seed=config['seed']+3)
                multi=flow.sample(ema,evidence,mask,steps=50,solver="heun",seed=config['seed']+3)
            target=(clean+1)/2
            panels=torch.stack([(evidence[i]+1)/2 if col==0 else mask[i].expand(3,-1,-1) if col==1
                                else one[i] if col==2 else multi[i] if col==3 else target[i]
                                for i in range(len(preview_ids)) for col in range(5)])
            save_grid(panels,output/f'preview-{step:06d}.png',columns=5)
            latest = output/'preview-latest.tmp'
            latest.write_bytes((output/f'preview-{step:06d}.png').read_bytes())
            latest.replace(output/'preview-latest.png')
            scores={}
            for mode in modes:
                ids = [i for i, row in enumerate(preview_ids) if val.records[row]['mode'] == mode]
                if ids:
                    scores[mode] = {'one_step_mse': (one[ids]-target[ids]).square().mean().item(),
                                    'heun50_mse': (multi[ids]-target[ids]).square().mean().item()}
            (output/f'preview-{step:06d}.json').write_text(json.dumps({'columns':['input','mask','one step','50-step Heun','target'],
                     'validation_ids':preview_ids,'scores':scores,'note':'Fixed examples; pixel MSE is not a distribution-quality metric for unconditional generation.'},indent=2)+'\n')
            return scores

        startup_val=evaluate(model)
        initial_val=saved.get('initial_val_mse', startup_val) if resume else startup_val
        print(f'architecture=jit; parameters={parameters:,}; train={len(train)}; val={len(val)}; batch={config["batch_size"]}; startup_val={startup_val:.6f}; start_step={start}',flush=True)
        preview(start)
        started=time.perf_counter()
        print('TRAINING_STARTED',time.time(),flush=True)
        log=output/'loss.csv';existing=log.exists()
        with log.open('a',newline='') as file:
            writer=csv.writer(file)
            if not existing:writer.writerow(('step','samples_seen','train_mse','val_mse','ema_val_mse','elapsed_seconds','steps_per_second','gpu_allocated_gib'))
            for step in range(start+1,config['steps']+1):
                if config.get('compile', False):
                    torch.compiler.cudagraph_mark_step_begin()
                if stream is None:
                    indices=torch.randint(len(train),(config['batch_size'],))
                    clean,evidence,mask=train.batch(indices,device)
                else:
                    clean,evidence,mask=next(stream)
                warmup=config.get('warmup_steps',500)
                for group in optimizer.param_groups:
                    group['lr']=config['learning_rate']*min(1.,step/max(1,warmup))
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss=training_loss(clean,evidence,mask)
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=True)
                # One host synchronization; reject invalid loss or gradients before
                # either optimizer or EMA can mutate model state.
                if not (torch.isfinite(loss.detach()) & torch.isfinite(norm)):
                    raise RuntimeError(f'Nonfinite loss or gradients at {step}')
                optimizer.step()
                with torch.no_grad():
                    torch._foreach_lerp_(list(ema.parameters()),list(model.parameters()),1-config['ema_decay'])
                stop_for_time = max_seconds is not None and time.perf_counter()-started >= max_seconds
                final_step = step == config['steps'] or stop_for_time
                val_loss=ema_loss=''
                if step%config['eval_every']==0 or final_step:
                    val_loss=evaluate(model);ema_loss=evaluate(ema)
                    preview(step)
                    print(f'step={step} train={loss.item():.6f} val={val_loss:.6f} ema_val={ema_loss:.6f}',flush=True)
                elapsed=time.perf_counter()-started
                rate=(step-start)/elapsed
                memory=torch.cuda.memory_allocated(device)/2**30
                if step%config.get('log_every', 1)==0 or step==1 or val_loss!='':
                    writer.writerow((step,step*config['batch_size'],loss.item(),val_loss,ema_loss,elapsed,rate,memory));file.flush()
                if step%100==0 or step==1 or final_step:
                    status=dict(step=step,total_steps=config['steps'],samples_seen=step*config['batch_size'],
                                train_mse=loss.item(),elapsed_seconds=elapsed,steps_per_second=rate,gpu_allocated_gib=memory,
                                initial_val_mse=initial_val,updated_unix=time.time(),
                                images_per_second=rate*config['batch_size'],
                                peak_gpu_gib=torch.cuda.max_memory_allocated(device)/2**30,
                                shard_wait_seconds=stream.load_seconds if stream else 0)
                    tmp=output/'status.tmp';tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(output/'status.json')
                    print(f'progress step={step}/{config["steps"]} samples={step*config["batch_size"]} steps/s={rate:.1f} gpu={memory:.2f}GiB',flush=True)
                if step%config['checkpoint_every']==0 or final_step:
                    checkpoint={'config':config,'step':step,'model':model.state_dict(),'ema':ema.state_dict(),
                                'optimizer':optimizer.state_dict(),'cpu_rng':torch.get_rng_state(),
                                'cuda_rng':torch.cuda.get_rng_state_all(),'dataset_sha256':hashes,
                                'parameters':parameters,'initial_val_mse':initial_val}
                    if stream is not None:
                        checkpoint['sampler'] = stream.state_dict()
                    tmp=output/'checkpoint.tmp';torch.save(checkpoint,tmp);tmp.replace(output/'checkpoint.pt')
                if stop_for_time:
                    print('Training time limit reached; final evaluation and checkpoint saved.', flush=True)
                    break
        print(f'COMPLETE {output} at step {step}',flush=True)

    finally:
        if stream is not None:
            stream.close()


def parse_training_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Defaults to saved configuration when resuming")
    parser.add_argument("--output", type=Path)
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--steps", type=int, help="Total optimizer steps, including resumed steps")
    budget.add_argument("--additional-steps", type=int, help="Extra updates after the checkpoint")
    parser.add_argument("--resume", type=Path, help="Checkpoint file or run directory")
    parser.add_argument("--max-seconds", type=float, help="Limit this invocation training loop; save before stopping (excludes startup and final save)")
    args = parser.parse_args(argv)
    if args.max_seconds is not None and not 0 < args.max_seconds < float("inf"):
        parser.error("--max-seconds must be finite and positive")
    if args.additional_steps is not None and (args.resume is None or args.additional_steps < 1):
        parser.error("--additional-steps requires --resume and a positive value")
    saved = None
    if args.resume is not None:
        if args.resume.is_dir():
            args.resume = args.resume / "checkpoint.pt"
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
    if args.config is not None:
        config = runpy.run_path(str(args.config))["config"] if args.config.suffix == ".py" else json.loads(args.config.read_text())
    elif saved is not None:
        config = copy.deepcopy(saved["config"])
        config["output"] = str(args.resume.parent)
    else:
        config = runpy.run_path("config/small.py")["config"]
    if args.output is not None:
        config["output"] = str(args.output)
    if args.steps is not None:
        config["steps"] = args.steps
    if args.additional_steps is not None:
        config["steps"] = saved["step"] + args.additional_steps
    if saved is not None and config["steps"] <= saved["step"]:
        parser.error("Run has reached its budget; use --additional-steps N or a larger --steps total")
    if min(config["steps"], config["batch_size"], config["eval_every"], config["checkpoint_every"]) < 1:
        parser.error("Steps, batch size and intervals must be positive")
    output = Path(config["output"])
    if output.exists() and args.resume is None:
        parser.error("Output already exists; resume or choose a new directory")
    if args.max_seconds is not None and config["formulation"] != "conditional-jit-x-pred":
        parser.error("--max-seconds currently supports conditional JiT training")
    return args, config


if __name__ == "__main__":
    args, config = parse_training_args()
    train_masked(config, args.resume, max_seconds=args.max_seconds)
