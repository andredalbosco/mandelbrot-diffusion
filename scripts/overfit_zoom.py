"""Fresh-model, 32-view memorization check; uses training data only."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from dataset import ZoomDataset
from model import build_model, ConditionalFlow
from scripts.evaluate_training import select_views, quality_report, MODES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=10000)
    args = parser.parse_args()
    torch.set_num_threads(4); torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    config = json.loads(Path('configs/zoom4x-smooth-1m-jit-1m-fast.json').read_text())
    output = Path('runs/jit-overfit-32'); output.mkdir(exist_ok=False)
    gallery = Path('output/dataset-browser/training-experiments')
    data = ZoomDataset(Path(config['data'])/'shard-0000/train.pt')
    candidates = select_views(data.records, per_mode=26, seed=42)
    ids = []
    for mode, count in zip(MODES, (20, 5, 4, 3)):
        pool = [i for i in candidates if data.records[i]['mode'] == mode]
        ids.extend(pool[j] for j in torch.linspace(0, len(pool)-1, count).round().int().tolist())
    assert len(ids) == len(set(ids)) == 32
    clean, evidence, mask = data.batch(ids)
    model = build_model(config).cuda()
    flow = ConditionalFlow(**config['flow'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                 betas=tuple(config['betas']), weight_decay=config['weight_decay'], fused=True)
    # Exactly 10% unconditional in expectation despite 3/32 unique examples.
    weights = torch.tensor([.1/3 if data.records[i]['mode']=='unconditional' else .9/29 for i in ids], device='cuda')
    def loss_fn(x, y, m): return flow.loss(model, x, y, m)
    compiled = torch.compile(loss_fn, fullgraph=True, mode='reduce-overhead')
    before = quality_report(model, flow, data, gallery, 'overfit', 0, ids=ids)
    history = []
    start = time.perf_counter()
    for step in range(1, args.steps+1):
        torch.compiler.cudagraph_mark_step_begin()
        index = torch.multinomial(weights, 64, replacement=True)
        optimizer.param_groups[0]['lr'] = config['learning_rate']*min(1, step/config['warmup_steps'])
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = compiled(clean[index], evidence[index], mask[index])
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., foreach=True)
        if not torch.isfinite(norm): raise RuntimeError('Nonfinite gradients')
        optimizer.step()
        if step%1000 == 0:
            entry = dict(step=step, loss=loss.item(), seconds=time.perf_counter()-start)
            history.append(entry); print(json.dumps(entry), flush=True)
    after = quality_report(model, flow, data, gallery, 'overfit', args.steps, ids=ids)
    report = dict(purpose='Memorization check, not held-out evaluation', training_ids=ids,
                  before=before, after=after, history=history, seconds=time.perf_counter()-start)
    torch.save(dict(config=config, model=model.state_dict(), diagnostic=report), output/'checkpoint.pt')
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'conditional_mse':after['conditional_mse'], 'input_mse':after['input_mse'], 'seconds':report['seconds']}), flush=True)

if __name__ == '__main__': main()
