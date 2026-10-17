"""Compare fresh batched rendering with the full viewer renderer.

uv run python docs/benchmark_fast.py
All output stays on CUDA; timings exclude allocation, compilation and image IO.
"""

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import warp as wp

from mandelbrot import ColorRenderer, View
from mandelbrot_fast import FastRenderer, resolve_kernel
from deep_zoom import escape_time_dd_kernel
from docs.benchmark_pipeline import measure


def end_to_end(view, mode, side=2, batch=32, repeats=5):
    """Fresh metadata/buffers through CPU-resident pairs, without disk writes."""
    from torch.nn import functional as F
    sequence = [0]
    def produce():
        sequence[0] += 1
        parents = [view.shifted(complex((i%8)*view.span*.03+sequence[0]*view.span*.001,
                                       (i//8)*view.span*.02)) for i in range(batch)]
        views = [v for parent in parents for v in (parent,parent.shifted(0j,span=parent.span/4))]
        renderer = FastRenderer(views,side=side,precision=mode,tile=64,guard='bailout')
        images = renderer.render().view(batch,2,128,128,3)
        coarse = F.interpolate(images[:,0,48:80,48:80].permute(0,3,1,2),size=(128,128),mode='bilinear',align_corners=False)
        return torch.stack((coarse,images[:,1].permute(0,3,1,2)),dim=1).cpu()
    stats = measure(produce,2,repeats)
    stats.update(precision=mode,guard='bailout',batch=batch,pairs_per_second=batch*1000/stats['mean_ms'])
    return stats


def reference(view, side):
    # Same sample positions as the fast renderer, including its minimal halo.
    size = 128*side+2*side
    sampled = ColorRenderer(size, size)
    sampled.render(view.shifted(0j, span=view.span*size/(128*side)))
    if view.span < 1e-10:
        from decimal import Decimal, localcontext
        with localcontext() as context:
            context.prec = 50
            def split(value):
                high = float(value)
                return wp.vec2d(high, float(value-Decimal.from_float(high)))
            cx, cy = (split(Decimal(v)) for v in view.center_exact)
            step = split(Decimal(str(view.span))/Decimal(128*side))
        wp.launch(escape_time_dd_kernel, dim=(size, size),
                  inputs=[cx, cy, step, 0., 0., view.iterations, False],
                  outputs=[*sampled._outputs[:2], sampled._styled_escape], device="cuda",
                  stream=wp.stream_from_torch(torch.cuda.current_stream()))
        sampled.recolor()
    image = torch.empty((1, 128, 128, 3), device="cuda")
    wp.launch(resolve_kernel, dim=(1,128,128), inputs=[side, wp.from_torch(sampled.image[None],dtype=wp.vec3)],
              outputs=[wp.from_torch(image, dtype=wp.vec3)], device="cuda",
              stream=wp.stream_from_torch(torch.cuda.current_stream()))
    return sampled.counts.clone(), image.clone()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--side", type=int, default=2)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--tiles", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--guard", choices=("strict", "bailout"), default="strict")
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--modes", nargs="+")
    parser.add_argument("--pairs", action="store_true", help="Render parent+target batches and construct 4x zoom pairs")
    parser.add_argument("--end-to-end", action="store_true", help="Also measure fresh batch construction and transfer to CPU")
    args = parser.parse_args()
    output = args.output or Path('output/benchmarks')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')/'fast.json'
    if output.exists():
        parser.error("Output exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    views = {
        "main": View(),
        "medium": View(center=-.75+.1j, span=.05),
        "shallow_boundary": View(center=-.7458366628110233+.135001050939548j, span=6.372367644529897e-7),
        "deep_boundary": View(center="(-0.92844118559550850366513868844580424143709174+0.262617371450439604741270010095574502976117479j)",span=4.118046071574424e-20),
        "deep_2048": View(center="(-0.7436438870371510000000001+0.1318259042053300000000002j)",span=1e-16,iterations=2048),
    }
    states=list((Path.home()/'.local/state/mandelbrot-diffusion').glob('viewer-*.json'))
    if states:
        s=json.loads(max(states,key=lambda p:p.stat().st_mtime).read_text())['view']
        views['saved_view']=View(s['center'],s['span'],s['iterations'],complex(s['z0']))
    if args.cases:
        views={key:views[key] for key in args.cases}
    result = {"gpu":torch.cuda.get_device_name(),"side":args.side,"pairs":args.pairs,"baseline":{},"fast":[],
              "notes":["128x128, cosine palette, z0=0, fresh samples; same tent filter with smaller halo.",
                       "Batch entries use different centers; first entry matches the scalar baseline.",
                       "Deep count reference forces direct double-double, bypassing perturbation.",
                       "Perturbation error guards are heuristic, not certified interval bounds.",
                       "No viewer changes; background desktop remains active."]}
    result['source_sha256'] = {name:hashlib.sha256(Path(name).read_bytes()).hexdigest()
                               for name in ('mandelbrot.py','deep_zoom.py','mandelbrot_fast.py','docs/benchmark_fast.py')}
    print('Writing',output,flush=True)
    def save():
        output.write_text(json.dumps(result,indent=2)+'\n')
    for name,view in views.items():
        baseline = ColorRenderer(128,128,side=args.side,filter="tent")
        def fresh():
            baseline._grid_origin = None
            return baseline.render(view)
        result['baseline'][name]=measure(fresh,2,args.repeats)
        ref_counts,ref_image=reference(view,args.side)
        modes = ['float32','float64','hybrid32'] if name in ('main','medium') else ['float64','hybrid32'] if name=='shallow_boundary' else ['perturb32','perturb64']
        if args.modes:
            modes=[mode for mode in modes if mode in args.modes]
        for mode in modes:
            tiles = args.tiles if mode.startswith('perturb') else [32]
            for tile in tiles:
                for batch in args.batches:
                    # Perturb all entries in a reproducible pattern, not copies
                    # that could benefit from unusually coherent workloads.
                    batch_views=[view.shifted(complex(((i%8)-3)*view.span*.03, (i//8)*view.span*.02)) for i in range(batch)]
                    batch_views[0]=view
                    if args.pairs:
                        batch_views=[v for parent in batch_views for v in (parent,parent.shifted(0j,span=parent.span/4))]
                    renderer=FastRenderer(batch_views,side=args.side,precision=mode,tile=tile,guard=args.guard)
                    def generate():
                        images=renderer.render()
                        if args.pairs:
                            from torch.nn import functional as F
                            images=images.view(batch,2,128,128,3)
                            coarse=F.interpolate(images[:,0,48:80,48:80].permute(0,3,1,2),size=(128,128),mode='bilinear',align_corners=False)
                            return torch.stack((coarse,images[:,1].permute(0,3,1,2)),dim=1)
                        return images
                    stats=measure(generate,2,args.repeats)
                    difference=(renderer.counts[0]-ref_counts).abs()
                    rgb=(renderer.image[:1]-ref_image).abs()
                    row={"view":name,"center":view.center_text,"span":view.span,"iterations":view.iterations,
                         "precision":mode,"tile":tile,"batch":batch,"guard":args.guard,**stats,
                         "images_per_second":batch*1000/stats['mean_ms'],
                         "ms_per_image":stats['mean_ms']/batch,
                         "count_mismatch_fraction":(difference!=0).float().mean().item(),
                         "max_count_difference":difference.max().item(),
                         "rgb_mae":rgb.mean().item(),"rgb_max_error":rgb.max().item()}
                    if args.pairs:
                        row['render_batch']=2*batch
                        row['pairs_per_second']=batch*1000/stats['mean_ms']
                        row['ms_per_pair']=stats['mean_ms']/batch
                        row['images_per_second']*=2
                        row['ms_per_image']/=2
                    # Check several batch entries, including children for pairs.
                    validation=[]
                    indices=sorted({0,len(batch_views)//3,2*len(batch_views)//3,len(batch_views)-1})
                    for i in indices:
                        expected_counts,expected_image=(ref_counts,ref_image) if i==0 else reference(batch_views[i],args.side)
                        delta=(renderer.counts[i]-expected_counts).abs()
                        rgb_delta=(renderer.image[i:i+1]-expected_image).abs()
                        validation.append({'index':i,'count_mismatch_fraction':(delta!=0).float().mean().item(),
                                           'max_count_difference':delta.max().item(),'rgb_mae':rgb_delta.mean().item(),
                                           'rgb_max_error':rgb_delta.max().item()})
                    row['validated_batch_entries']=validation
                    if hasattr(renderer,'fallbacks'):
                        row['fallback_fraction']=float(renderer.fallbacks.numpy().sum()/renderer.counts.numel())
                    result['fast'].append(row)
                    print(f"{name} {mode} tile{tile} B{batch}: {row['ms_per_image']:.3f}ms/img, {row['images_per_second']:.0f}/s; count mismatch {row['count_mismatch_fraction']:.3g}; RGB MAE {row['rgb_mae']:.3g}; fallback {row.get('fallback_fraction',0):.1%}",flush=True)
                    save()
                    del renderer
                    gc.collect()
                    torch.cuda.empty_cache()
        del baseline,ref_counts,ref_image
        if args.end_to_end:
            preferred={'main':'hybrid32','medium':'float64','shallow_boundary':'float64',
                       'deep_boundary':'perturb64' if args.side==2 else 'perturb32'}.get(name,'perturb32')
            stats=end_to_end(view,preferred,args.side,batch=max(args.batches))
            stats['view']=name
            result.setdefault('end_to_end_cpu_pairs',[]).append(stats)
            print(name,'end-to-end CPU pairs:',stats,flush=True)
            save()
    save()
    print('Saved',output,flush=True)


if __name__=='__main__':
    main()
