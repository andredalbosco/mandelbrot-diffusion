"""Build an explicitly quantized benchmark cache; preserve source datasets."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import ZoomDataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--shards', type=int, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Output exists; choose a new cache directory')
    torch.set_num_threads(4)
    source = args.source.resolve()
    original = json.loads((source/'manifest.json').read_text())
    if not original.get('complete') or not 0 < args.shards <= len(original['shards']):
        raise ValueError('Need a complete source and valid shard count')
    args.output.mkdir(parents=True)
    compact, reference = [], []
    errors = []
    started = time.perf_counter()
    for i, entry in enumerate(original['shards'][:args.shards]):
        ref = copy.deepcopy(entry)
        for item in ref['files'].values():
            item['path'] = str(source/item['path'])
        reference.append(ref)
        item = copy.deepcopy(ref)
        data = ZoomDataset(ref['files']['train']['path'])
        payload = dict(records=data.records, encoding='uint16-unorm')
        shard_errors = {}
        for key in ('inputs', 'targets', 'masks'):
            value = data.data[key]
            encoded = torch.empty(value.shape, dtype=torch.uint16)
            maximum, squared, count = 0., 0., 0
            for begin in range(0, len(value), 128):
                chunk = value[begin:begin+128]
                if not torch.isfinite(chunk).all() or chunk.min() < 0 or chunk.max() > 1:
                    raise ValueError('Invalid source values')
                quantized = (chunk*65535).round().to(torch.uint16)
                encoded[begin:begin+len(chunk)].copy_(quantized)
                delta = quantized.float()/65535-chunk
                maximum = max(maximum, delta.abs().max().item())
                squared += delta.double().square().sum().item()
                count += delta.numel()
            payload[key] = encoded
            shard_errors[key] = dict(max_abs=maximum, mse=squared/count)
        payload['images'] = payload['targets']
        path = args.output/f'train-{i:04d}.pt'
        torch.save(payload, path)
        with path.open('rb') as f:
            os.fsync(f.fileno())
            digest = hashlib.file_digest(f, 'sha256').hexdigest()
        item['files']['train'].update(path=path.name, sha256=digest)
        compact.append(item)
        errors.append(shard_errors)
        print(f'{i+1}/{args.shards}: {path.stat().st_size/1e9:.3f} GB', flush=True)
        del payload, encoded, data
    for name, shards in [('manifest.json', compact), ('reference.json', reference)]:
        manifest = dict(complete=True, shards=shards, parent=str(source),
                        purpose='Subset benchmark; uint16-unorm is lossy, not original precision')
        (args.output/name).write_text(json.dumps(manifest, indent=2)+'\n')
    baseline = args.output/'reference'
    baseline.mkdir()
    (baseline/'manifest.json').write_bytes((args.output/'reference.json').read_bytes())
    (args.output/'quantization.json').write_text(json.dumps(dict(errors=errors,seconds=time.perf_counter()-started),indent=2)+'\n')


if __name__ == '__main__':
    main()
