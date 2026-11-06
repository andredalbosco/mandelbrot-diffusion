"""Compare EMA checkpoints on identical recursive 4x zoom paths."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from PIL import Image
from model import build_model, ConditionalFlow
from mandelbrot import View
from mandelbrot_fast import FastRenderer, DatasetRenderer

OUT=Path('output/dataset-browser/zoom-comparison-8h')
POINTS=[('Seahorse',complex(-.743643887,.131825904)),('Spiral',complex(-.7463,.1102)),('Miniature',complex(-1.768,0))]
CHECKPOINTS=[('before','runs/jit-10m-smooth-1m-patch32-pilot/checkpoint.pt'),('after','runs/jit-10m-smooth-1m-8h/checkpoint.pt')]

def save(x,name):
    Image.fromarray((x.clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype('uint8')).save(OUT/name)

@torch.inference_mode()
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    color=DatasetRenderer(side=2,mode='float64',palette='smooth')
    cameras=[]; offsets=[]; refs=[]
    center=[complex(-.5,0)]*len(POINTS)
    span=3.5
    for depth in range(15):
        if depth:
            delta=[complex(max(-.375,min(.375,(p-c).real/span)),max(-.375,min(.375,(p-c).imag/span))) for (_,p),c in zip(POINTS,center)]
            center=[c+d*span for c,d in zip(center,delta)]
            offsets.append(delta);span/=4
        cameras.append([{'real':c.real,'imag':c.imag,'span':span} for c in center])
        renderer=FastRenderer([View(c,span,8192) for c in center],side=2,precision='float64',tile=64)
        ref=color._render(renderer).permute(0,3,1,2).clone()
        refs.append(ref)
        for i in range(len(POINTS)): save(ref[i],f'p{i}-d{depth}-reference.png')
        print('reference',depth,flush=True)
    rows=[];checkpoints={}
    for label,path in CHECKPOINTS:
        ck=torch.load(path,map_location='cpu',weights_only=True)
        model=build_model(ck['config']).cuda().eval();model.load_state_dict(ck['ema'])
        flow=ConditionalFlow(**ck['config']['flow']);checkpoints[label]=ck['step'];del ck
        for steps in (5,50):
            previous=refs[0]*2-1
            for depth in range(1,15):
                axis=(torch.arange(32,device='cuda')+.5)/32-.5
                yy,xx=torch.meshgrid(axis,axis,indexing='ij')
                grid=torch.stack([torch.stack((2*(d.real+xx/4),2*(-d.imag+yy/4)),-1) for d in offsets[depth-1]])
                patch=F.grid_sample(previous,grid,align_corners=False)
                condition=F.interpolate(patch,size=(128,128),mode='bilinear',align_corners=False)
                mask=torch.ones_like(condition[:,:1])
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    pred=flow.sample(model,condition,mask,steps=steps,solver='heun',seed=1234+depth)
                for i in range(len(POINTS)):
                    save(pred[i],f'p{i}-d{depth}-{label}-{steps}.png')
                    rows.append(dict(model=label,steps=steps,path=i,depth=depth,mse=(pred[i]-refs[depth][i]).square().mean().item()))
                previous=pred*2-1
                print(label,steps,depth,flush=True)
        del model
    (OUT/'results.json').write_text(json.dumps(dict(checkpoints=checkpoints,cameras=cameras,metrics=rows),indent=2))
    html='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Recursive zoom comparison</title><style>body{background:#14171c;color:#eee;font:16px system-ui;margin:24px}img{width:100%;image-rendering:pixelated}.row{display:grid;grid-template-columns:repeat(5,minmax(100px,1fr));gap:10px;max-width:1450px}h2{margin-top:40px}p{max-width:1000px;line-height:1.5}.labels{position:sticky;top:0;background:#14171c;padding:12px 0}</style><h1>Recursive zoom: before vs eight more hours</h1><p>Left: mathematical reference. Each model feeds back its own full-precision output. Same cameras and per-depth noise seeds for both checkpoints. EMA weights; Heun 5 or 50 steps. Only depth 0 is a rendered seed. Reference: 128², four samples/pixel, 8,192 iterations. Depths 13–14 exceed the previous explorer's nominal training range. These fixed paths can enter uniform interiors/exteriors; they do not select interesting points adaptively.</p>'''
    html+=f'<p>Before: step {checkpoints["before"]:,}. After: step {checkpoints["after"]:,}.</p>'
    html+='<div class="row labels"><b>Reference</b><b>Before · 5</b><b>After · 5</b><b>Before · 50</b><b>After · 50</b></div>'
    for i,(name,_) in enumerate(POINTS):
        html+=f'<h2>{name}</h2><details><summary>Rendered initial seed</summary><img style="width:256px" src="p{i}-d0-reference.png"></details>'
        for d in range(1,15):
            html+=f'<h3>Zoom {d} · {4**d:,}×</h3><div class="row">'
            for suffix in ('reference','before-5','after-5','before-50','after-50'):
                html+=f'<img loading="lazy" alt="{name} zoom {d} {suffix}" src="p{i}-d{d}-{suffix}.png">'
            html+='</div>'
    (OUT/'index.html').write_text(html)

if __name__=='__main__': main()
