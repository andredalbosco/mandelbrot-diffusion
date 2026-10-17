"""CPU-only checks before starting or resuming a sharded conditional JiT run."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shlex
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import ShardedZoomDataset, ZoomDataset, training_order
from model import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = json.loads(args.config.read_text())
    saved = torch.load(args.resume, map_location='cpu', weights_only=True) if args.resume else None
    if not saved and Path(config['output']).exists():
        raise ValueError('Fresh-run output already exists')
    keys = ('architecture', 'model', 'size', 'formulation', 'learning_rate',
            'ema_decay', 'batch_size', 'seed', 'flow', 'warmup_steps', 'compile',
            'compile_mode', 'betas', 'weight_decay', 'unconditional_fraction')
    if saved:
        for key in keys:
            if config.get(key) != saved['config'].get(key):
                raise ValueError(f'Resume config mismatch: {key}')
        if not 0 <= saved['step'] < config['steps']:
            raise ValueError('Checkpoint must precede requested total steps')
    root = Path(config['data'])
    manifest_path = root/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if saved and saved['dataset_sha256'] != {'manifest': digest}:
        raise ValueError('Dataset manifest changed')
    counts, modes, depths = {}, {}, {}
    selected_count = 0
    encodings = set()
    for split in ('train', 'val'):
        dataset = ShardedZoomDataset(root, split)
        mode_counts, depth_counts = Counter(), Counter()
        for shard, item in enumerate(dataset.files):
            data = ZoomDataset(root/item['path'])
            encodings.add(data.encoding)
            if len(data) != item['count'] or data.targets.shape[1:] != (3, config['size'], config['size']):
                raise ValueError(f'Invalid shape/count: {item["path"]}')
            mode_counts.update(r['mode'] for r in data.records)
            depth_counts.update(r.get('zoom_depth', 'missing') for r in data.records)
            if split == 'train':
                selected_count += len(training_order(data.records, config['seed']+1000003+shard,
                                                     config.get('unconditional_fraction')))
        counts[split] = len(dataset)
        modes[split], depths[split] = dict(mode_counts), dict(depth_counts)
    model = build_model(config)
    if len(encodings) != 1:
        raise ValueError('Train and validation must use the same cache encoding')
    sampler = None
    if saved:
        for label in ('model', 'ema'):
            model.load_state_dict(saved[label], strict=True)
            if any(not torch.isfinite(v).all() for v in saved[label].values()):
                raise ValueError(f'Nonfinite {label} checkpoint tensor')
        for state in saved['optimizer']['state'].values():
            if any(isinstance(v, torch.Tensor) and not torch.isfinite(v).all() for v in state.values()):
                raise ValueError('Nonfinite optimizer state')
        for key in ('cpu_rng', 'cuda_rng', 'sampler'):
            if key not in saved:
                raise ValueError(f'Missing resume state: {key}')
        sampler = saved['sampler']
        files = ShardedZoomDataset(root, 'train').files
        order = torch.randperm(len(files), generator=torch.Generator().manual_seed(config['seed']+sampler['epoch']))
        shard = int(order[sampler['shard_cursor']])
        data = ZoomDataset(root/files[shard]['path'])
        selected = training_order(data.records, config['seed']+1000003*(sampler['epoch']+1)+shard,
                                  config.get('unconditional_fraction'))
        if not 0 <= sampler['offset'] < len(selected):
            raise ValueError('Invalid selected-row sampler offset')
        selection = json.loads((Path(config['output'])/'validation-selection.json').read_text())
        ids = selection['ids']
    else:
        from scripts.evaluate_training import select_views
        ids = select_views(ShardedZoomDataset(root, 'val').records, per_mode=64, seed=config['seed']+1)
    if len(ids) != len(set(ids)) or not ids or min(ids) < 0 or max(ids) >= counts['val']:
        raise ValueError('Invalid fixed validation selection')
    command_args = [str(Path(sys.executable)), 'train.py', '--config', str(args.config)]
    if args.resume:
        command_args += ['--resume', str(args.resume)]
    command = shlex.join(command_args)
    report = dict(passed=True, checkpoint_step=saved['step'] if saved else 0,
                  remaining_steps=config['steps']-(saved['step'] if saved else 0),
                  parameters=sum(p.numel() for p in model.parameters()), encoding=next(iter(encodings)),
                  counts=counts, modes=modes, depths=depths, selected_training_rows_per_pass=selected_count,
                  validation_examples=len(ids), sampler=sampler, manifest_sha256=digest,
                  checkpoint_sha256=hashlib.sha256(args.resume.read_bytes()).hexdigest() if args.resume else None,
                  source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                 for p in ('train.py', 'dataset.py', 'model.py', 'scripts/evaluate_training.py',
                                           'scripts/prepare_training_cache.py')},
                  torch_version=torch.__version__, command=command,
                  limitations=['Metadata and checkpoint validation only; no full pixel scan or shard rehash.',
                               'Does not start training or stop the paused trainer.'])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
