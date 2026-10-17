"""Measure steady-state JiT training throughput; does not touch training runs."""
import argparse
from contextlib import nullcontext
import copy
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from model import build_model, ConditionalFlow
from dataset import ZoomDataset, ShardBatches


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--config', type=Path, default=Path('configs/zoom4x-smooth-1m-jit-1m-fast.json'))
    p.add_argument('--mlp-hidden-size', type=int, help='Override SwiGLU width for alignment experiments')
    p.add_argument('--compile', action='store_true')
    p.add_argument('--compile-mode', default='default')
    p.add_argument('--real-data', action='store_true')
    p.add_argument('--sharded', action='store_true')
    p.add_argument('--data', help='Override the configured dataset')
    p.add_argument('--checked', action='store_true')
    p.add_argument('--unconditional-fraction', type=float)
    p.add_argument('--select-rows', action='store_true')
    p.add_argument('--staging-rows', type=int, default=0)
    p.add_argument('--combined-check', action='store_true')
    p.add_argument('--attention', choices=['auto', 'flash', 'cudnn', 'efficient', 'math'], default='auto')
    p.add_argument('--liger-modulation', action='store_true', help='Experimental fused block normalization; requires liger-kernel')
    p.add_argument('--profile', action='store_true')
    p.add_argument('--graph-update', action='store_true')
    p.add_argument('--state-output', type=Path)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.attention != 'auto':
        # A persisted compiled graph can retain the previous SDPA choice.
        # Backend experiments must retrace and verify actual dispatched kernels.
        torch.compiler.config.force_disable_caches = True
    from torch.nn.attention import sdpa_kernel, SDPBackend
    backends = dict(flash=SDPBackend.FLASH_ATTENTION, cudnn=SDPBackend.CUDNN_ATTENTION,
                    efficient=SDPBackend.EFFICIENT_ATTENTION, math=SDPBackend.MATH)
    attention_context = nullcontext() if args.attention == 'auto' else sdpa_kernel(backends[args.attention])
    # Keep selection active through compilation, forward, and backward.
    with attention_context:
        benchmark(args)


def io_counts():
    return {k: int(v) for k, v in (line.split(":") for line in Path("/proc/self/io").read_text().splitlines())}


def benchmark(args):
    initial_io = io_counts()
    torch.set_num_threads(4)
    torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32 = True
    config = json.loads(args.config.read_text())
    if args.data is None:
        args.data = config['data']
    if args.mlp_hidden_size is not None:
        config['model']['mlp_hidden_size'] = args.mlp_hidden_size
    model = build_model(config).cuda()
    parity = None
    if args.liger_modulation:
        from types import MethodType
        from liger_kernel.ops import LigerModulatedRMSNormFunction
        reference = copy.deepcopy(model)
        def fused_forward(self, tokens, condition):
            sa, ca, ga, sm, cm, gm = self.condition(condition).chunk(6, dim=-1)
            def norm(x, layer, scale, shift):
                return LigerModulatedRMSNormFunction.apply(x, layer.weight, scale, shift,
                                                         layer.eps, 0., 'llama', False)
            tokens = tokens + ga[:, None] * self.attention(norm(tokens, self.norm_attention, ca, sa))
            return tokens + gm[:, None] * self.mlp(norm(tokens, self.norm_mlp, cm, sm))
        for block in model.blocks:
            block.forward = MethodType(fused_forward, block)
        # Exercise nonzero gates/output: zero-init would conceal block errors.
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if 'condition.1' in name or 'final_projection' in name:
                    parameter.normal_(0, .02)
        reference.load_state_dict(model.state_dict())
        x = torch.randn(2, 3, 128, 128, device='cuda')
        t = torch.full((2,), .4, device='cuda')
        m = torch.ones(2, 1, 128, 128, device='cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            a, b = reference(x, t, x, m), model(x, t, x, m)
        a.float().square().mean().backward(); b.float().square().mean().backward()
        gradient_error = max((p.grad-q.grad).abs().max().item() for p,q in zip(reference.parameters(), model.parameters()) if p.grad is not None)
        parity = dict(output_max_abs=(a-b).abs().max().item(), gradient_max_abs=gradient_error)
        torch.testing.assert_close(a, b, atol=.01, rtol=.02)
        for p,q in zip(reference.parameters(), model.parameters()):
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, atol=.002, rtol=.05)
        del reference, a, b
        # Restore the identical initialization and RNG for timed runs.
        torch.manual_seed(123)
        fresh = build_model(config).cuda()
        model.load_state_dict(fresh.state_dict())
        del fresh
        model.zero_grad(set_to_none=True)
    ema = copy.deepcopy(model).requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=.01, fused=True,
                            capturable=args.graph_update)
    update_graph = None
    update_count = 0
    static_grads = None
    parameters, averages = list(model.parameters()), list(ema.parameters())
    def update():
        nonlocal update_graph, update_count, static_grads
        def apply():
            opt.step()
            with torch.no_grad():
                torch._foreach_lerp_(averages, parameters, .001)
        if not args.graph_update or update_count == 0:
            apply()  # Allocate Adam state outside capture.
        else:
            if static_grads is None:
                static_grads = [torch.empty_like(p.grad) for p in parameters]
            torch._foreach_copy_(static_grads, [p.grad for p in parameters])
            for p, grad in zip(parameters, static_grads):
                p.grad = grad
            if update_graph is None:
                update_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(update_graph):
                    apply()
            update_graph.replay()
        update_count += 1
    flow = ConditionalFlow(**config['flow'])
    def loss_fn(clean, evidence, mask):
        return flow.loss(model, clean, evidence, mask)
    loss_fn = torch.compile(loss_fn, fullgraph=True, mode=args.compile_mode) if args.compile else loss_fn
    clean = torch.rand(args.batch, 3, 128, 128, device='cuda')*2-1
    evidence = clean.clone()
    mask = torch.ones(args.batch, 1, 128, 128, device='cuda')
    data = ZoomDataset('data/zoom4x-smooth-blue-gold-1m-128/shard-0000/train.pt') if args.real_data else None
    stream = ShardBatches(args.data, args.batch, 123,
                         unconditional_fraction=args.unconditional_fraction, select_rows=args.select_rows,
                         staging_rows=args.staging_rows) if args.sharded else None
    def step():
        torch.compiler.cudagraph_mark_step_begin()
        if stream is not None:
            batch = next(stream)
        elif data is not None:
            batch = data.batch(torch.randint(len(data), (args.batch,)))
        else:
            batch = clean, evidence, mask
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = loss_fn(*batch)
        if args.checked and not args.combined_check and not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., foreach=True)
        finite = torch.isfinite(norm)
        if args.combined_check:
            finite = finite & torch.isfinite(loss.detach())
        if args.checked and not finite:
            raise RuntimeError('Nonfinite gradient')
        update()
        return loss.detach(), norm.detach()
    start = time.perf_counter()
    for _ in range(15):
        loss, norm = step()
    torch.cuda.synchronize()
    setup = time.perf_counter()-start
    torch.cuda.reset_peak_memory_stats()
    timed_io = io_counts()
    start = time.perf_counter()
    for _ in range(args.steps):
        loss, norm = step()
    torch.cuda.synchronize()
    seconds = time.perf_counter()-start
    final_io = io_counts()
    assert torch.isfinite(loss) and torch.isfinite(norm)
    result = dict(model=config["model"], parameters=sum(p.numel() for p in model.parameters()), batch=args.batch, compiled=args.compile, compile_mode=args.compile_mode, real_data=args.real_data,
                  attention=args.attention,
                  unconditional_fraction=args.unconditional_fraction,
                  select_rows=args.select_rows, combined_check=args.combined_check,
                  staging_rows=args.staging_rows,
                  data=args.data,
                  graph_update=args.graph_update,
                  compile_caches_disabled=torch.compiler.config.force_disable_caches,
                  liger_modulation=args.liger_modulation, parity=parity,
                  sharded=args.sharded, checked=args.checked, shard_wait_seconds=stream.load_seconds if stream else 0,
                  shard_reads=stream.read_timings if stream else [],
                  steps=args.steps, seconds=seconds, images_per_second=args.batch*args.steps/seconds,
                  ms_per_step=1000*seconds/args.steps, warmup_seconds=setup,
                  peak_gib=torch.cuda.max_memory_allocated()/2**30, loss=loss.item(), torch=torch.__version__)
    result['io_timed'] = {k: final_io[k]-timed_io[k] for k in final_io}
    result['io_including_setup'] = {k: final_io[k]-initial_io[k] for k in final_io}
    if stream is not None:
        result['host_staging_bytes'] = sum(v.numel()*v.element_size()
                                          for slot in stream.staging if slot is not None for v in slot)
    if args.profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
            step()
            torch.cuda.synchronize()
        result['profile_operations'] = sorted({e.name for e in prof.events() if any(s in e.name.lower() for s in ('attention', 'flash', 'cudnn', 'fmha'))})
    if stream is not None:
        stream.close()
    if args.state_output:
        torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), optimizer=opt.state_dict()), args.state_output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)

if __name__ == '__main__':
    main()
