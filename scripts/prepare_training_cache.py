"""Convert original zoom pairs to RGB16 targets and 32x32 conditioning patches."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import ZoomDataset, conditioning_patch_mask


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def convert(source, output):
    data = ZoomDataset(source)
    if data.encoding != 'float32' or data.targets.shape[1:] != (3, 128, 128):
        raise ValueError('Conversion requires original float32 128x128 zoom pairs')
    # Invert the old separable bilinear map, then retain only the 32x32 core.
    axis = torch.arange(128, dtype=torch.float64)/4+.625
    weights = (1-(axis[:, None]-torch.arange(34, dtype=torch.float64)).abs()).clamp_min(0)
    inverse = torch.linalg.pinv(weights)
    targets = torch.empty(data.targets.shape, dtype=torch.uint16)
    inputs = torch.empty((len(data), 3, 32, 32), dtype=torch.uint16)
    support = conditioning_patch_mask(data.records)
    maximum = 0.
    for begin in range(0, len(data), 128):
        end = min(begin+128, len(data))
        x, mask, target = data.inputs[begin:end], data.masks[begin:end], data.targets[begin:end]
        for value in (x, mask, target):
            if not torch.isfinite(value).all() or value.min() < 0 or value.max() > 1:
                raise ValueError('Source values must be finite and in [0,1]')
        weighted = x.double()*mask.double()
        patch = inverse @ weighted @ inverse.T
        error = (weights @ patch @ weights.T-weighted).abs().max().item()
        maximum = max(maximum, error)
        if error > 1e-5:
            raise ValueError('Evidence is incompatible with the expected bilinear crop')
        patch = patch[:, :, 1:-1, 1:-1].clamp(0, 1)
        patch = torch.where(support[begin:end], patch, .5)
        inputs[begin:end] = (patch*65535).round().to(torch.uint16)
        targets[begin:end] = (target*65535).round().to(torch.uint16)
    payload = dict(encoding='uint16-patch32', inputs=inputs, targets=targets,
                   images=targets, records=data.records)
    temporary = output.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(output)
    with output.open('rb') as file:
        digest = hashlib.file_digest(file, 'sha256').hexdigest()
    return dict(path=output.name, count=len(data), sha256=digest, bytes=output.stat().st_size,
                recovered_evidence_max_error=maximum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shards', type=int, help='Subset for validation; default all shards')
    args = parser.parse_args()
    torch.set_num_threads(4)
    source = args.source.resolve()
    original = json.loads((source/'manifest.json').read_text())
    if not original.get('complete'):
        raise ValueError('Source dataset must be complete')
    count = args.shards if args.shards is not None else len(original['shards'])
    if not 0 < count <= len(original['shards']):
        raise ValueError('Invalid shard count')
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output/'manifest.json'
    source_hash = hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest()
    manifest = dict(complete=False, encoding='uint16-patch32', source=str(source),
                    source_manifest_sha256=source_hash, requested_shards=count,
                    conditioning='32x32 core; bilinear upsampling with clamped crop edges; normalized coverage',
                    shards=[])
    if path.exists():
        manifest = json.loads(path.read_text())
        if (manifest.get('source_manifest_sha256'), manifest.get('requested_shards'), manifest.get('encoding')) != (source_hash, count, 'uint16-patch32'):
            raise ValueError('Existing cache has a different source or recipe')
        for shard in manifest['shards']:
            for item in shard['files'].values():
                file = args.output/item['path']
                if not file.exists() or file.stat().st_size != item['bytes']:
                    raise ValueError('Existing cache file missing or truncated')
    atomic_json(path, manifest)
    started = time.perf_counter()
    for index in range(len(manifest['shards']), count):
        shard = copy.deepcopy(original['shards'][index])
        for split in ('train', 'val'):
            shard['files'][split] = convert(source/shard['files'][split]['path'],
                                            args.output/f'{split}-{index:04d}.pt')
        manifest['shards'].append(shard)
        atomic_json(path, manifest)
        print(f'{index+1}/{count} shards; elapsed {time.perf_counter()-started:.1f}s', flush=True)
    manifest['complete'] = True
    manifest['counts'] = {split: sum(s['files'][split]['count'] for s in manifest['shards']) for split in ('train', 'val')}
    atomic_json(path, manifest)
    print(json.dumps(manifest['counts']), flush=True)


if __name__ == '__main__':
    main()
