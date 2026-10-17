"""Generate resumable mixed zoom shards with global camera deduplication."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from dataset import ViewDiversity, prepare_zoom_dataset


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
                  camera_tolerance=.01, train_fraction=.9, max_depth=12, storage='float32',
                  mode_proportions=dict(interior=.60, edge=.15, corner=.15, unconditional=.10))
    manifest_path = args.output/'manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous['config'] != config:
            raise ValueError('Resume configuration differs from existing dataset')
    state={'diversity': ViewDiversity(.01)}
    # Fix the anchor split once for all shards, including resumed shards.
    from dataset import boundary_anchors
    state['groups']=boundary_anchors('cuda',args.seed)
    shards=[]
    started=time.time()
    source_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                   for name in ('dataset.py','mandelbrot_fast.py','scripts/generate_zoom_shards.py')}
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


if __name__=='__main__':
    main()
