"""Fixed depth/mode galleries and reconstruction diagnostics for training runs."""
import json
from pathlib import Path
import torch
from dataset import save_grid

MODES = ('interior', 'edge', 'corner', 'unconditional')


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


@torch.no_grad()
def quality_report(model, flow, data, output, label, step, seed=17, ids=None):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if ids is None:
        ids = select_views(data.records, per_mode=13, seed=seed)
    clean, evidence, mask = data.batch(ids, 'cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        prediction = flow.sample(model, evidence, mask, steps=50, solver='heun', seed=seed)
    target, input_rgb = (clean+1)/2, (evidence+1)/2
    mse = (prediction-target).square().mean((1, 2, 3))
    input_mse = (input_rgb-target).square().mean((1, 2, 3))
    rows = []
    for j, i in enumerate(ids):
        record = data.records[i]
        rows.append(dict(id=i, mode=record['mode'], depth=record['zoom_depth'],
                         prediction_mse=mse[j].item() if record['mode'] != 'unconditional' else None,
                         input_mse=input_mse[j].item() if record['mode'] != 'unconditional' else None))
    conditional = [r for r in rows if r['prediction_mse'] is not None]
    report = dict(label=label, step=step, solver='heun', steps=50, seed=seed, rows=rows,
                  conditional_mse=sum(r['prediction_mse'] for r in conditional)/len(conditional),
                  input_mse=sum(r['input_mse'] for r in conditional)/len(conditional))
    panels = torch.stack([v[j] for j in range(len(ids)) for v in (input_rgb, prediction, target)])
    name = f'{label}-{step:06d}'
    save_grid(panels, output/f'{name}.png', columns=3)
    (output/f'{name}.json').write_text(json.dumps(report, indent=2)+'\n')
    report['image'] = name+'.png'
    html = ['<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>JiT experiment</title>',
            '<style>body{background:#17191d;color:#eee;font:16px system-ui;max-width:1100px;margin:24px auto;padding:12px}img{width:100%;image-rendering:pixelated}a{color:#9cd}p{line-height:1.5}</style>',
            f'<h1>{label}: {step:,} updates</h1><p>Heun 50, fixed noise. Columns: input · prediction · target. Unconditional targets are references, not paired correctness scores.</p>',
            f'<p>Conditional MSE: {report["conditional_mse"]:.6f}; input baseline: {report["input_mse"]:.6f}.</p>']
    # Individual rows crop a single contact sheet without duplicating PNG assets.
    for j, row in enumerate(rows):
        html.append(f'<p>{row["mode"]} · depth {row["depth"]} · sample {row["id"]}</p>')
        html.append(f'<div style="aspect-ratio:3; background:url({name}.png) 0 {100*j/max(1,len(rows)-1):.8f}%/100% {100*len(rows)}%"></div>')
    (output/f'{name}.html').write_text('\n'.join(html))
    return report
