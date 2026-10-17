"""Browser explorer for a trained conditional JiT; Python server owns CUDA inference."""
import argparse
import base64
from decimal import Decimal, localcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import math
from pathlib import Path
import time
import threading

import torch
from PIL import Image
from dataset import reconstruct_conditioning
from mandelbrot import View
from mandelbrot_fast import FastRenderer, DatasetRenderer
from model import build_model, ConditionalFlow

PAGE = '''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mandelbrot · learned zoom</title><style>
body{margin:0;background:#11151b;color:#e9edf4;font:16px system-ui}main{max-width:1500px;margin:auto;padding:24px}h1{font-size:26px;margin-bottom:8px}p{color:#aebacc;line-height:1.5}button,input{font:inherit;background:#242e3b;color:white;border:1px solid #47566b;border-radius:6px;padding:9px}button{cursor:pointer}button:disabled{opacity:.4}nav{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}img{display:block;width:100%;aspect-ratio:1;background:#080b10;image-rendering:pixelated;cursor:crosshair}label{display:block;margin:8px 0}#status{min-height:24px}input{width:130px}small{color:#aebacc}@media(max-width:700px){.grid{grid-template-columns:1fr}}
</style><main><h1>Mandelbrot · learned 4× zoom</h1><p>Click any image to zoom 4× into that point. Pan with the arrows. Only the initial view is rendered as a seed. Further views use the previous model output; the reference is comparison-only.</p>
<nav><button id="back">Back</button><button id="out">Zoom out 4×</button><button id="reset">Reset</button><button data-pan="-1,0">←</button><button data-pan="0,1">↑</button><button data-pan="0,-1">↓</button><button data-pan="1,0">→</button><button id="new">New noise</button></nav>
<nav><label>Real <input id="real" value="-0.5"></label><label>Imaginary <input id="imag" value="0"></label><label>Span <input id="span" value="3.5"></label><button id="go">Go</button></nav>
<div id="status" role="status">Connecting…</div><div class="grid"><section><h3>Upsampled input</h3><img id="input" alt="Bilinear conditioning input"></section><section><h3>10M JiT · Heun 50</h3><img id="prediction" alt="Model prediction"></section><section><h3>Reference render</h3><img id="target" alt="Mandelbrot reference"></section></div><p id="details"></p><small>128×128 · smooth blue & gold · reference: 8,192 iterations, 4 samples/pixel. Deep zoom is supported down to span 1e-24; views beyond the training range are marked. Similar-looking generated detail is not guaranteed to match the exact fractal.</small></main>
<script>
let current={real:-.5,imag:0,span:3.5,seed:1234},history=[],busy=false,last=null;
async function show(next,remember=true,reset=false){if(busy)return;busy=true;document.querySelectorAll('button').forEach(b=>b.disabled=true);document.getElementById('status').textContent='Rendering and sampling…';
try{let response=await fetch('/view',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...next,...(last&&!reset?{parent_image:last.prediction,parent_camera:current}:{})})});let data=await response.json();if(!response.ok)throw Error(data.error);if(remember)history.push(last);last=data;current=data.camera;localStorage.setItem("mandelbrot-camera",JSON.stringify(current));for(let key of ['input','prediction','target'])document.getElementById(key).src='data:image/png;base64,'+data[key];for(let key of ['real','imag','span'])document.getElementById(key).value=current[key];document.getElementById('status').textContent=`Model ${data.model_seconds.toFixed(2)}s · renderer ${data.render_seconds.toFixed(2)}s`;document.getElementById('details').textContent=`Checkpoint ${data.step.toLocaleString()} · zoom ${(3.5/current.span).toPrecision(4)}× · ${data.source} · seed ${current.seed}${data.outside_training ? " · beyond training zoom range" : ""}`;
}catch(e){document.getElementById('status').textContent=e.message;}finally{busy=false;document.querySelectorAll('button').forEach(b=>b.disabled=false);}}
document.querySelectorAll('img').forEach(im=>im.onclick=e=>{let r=im.getBoundingClientRect();show({...current,offset_real:((e.clientX-r.left)/r.width-.5)*current.span,offset_imag:(.5-(e.clientY-r.top)/r.height)*current.span,span:current.span/4});});
document.getElementById('back').onclick=()=>{if(!history.length||busy)return;let data=history.pop();if(!data)return;last=data;current=data.camera;for(let key of ['input','prediction','target'])document.getElementById(key).src='data:image/png;base64,'+data[key];for(let key of ['real','imag','span'])document.getElementById(key).value=current[key];document.getElementById('status').textContent='Restored previous view';document.getElementById('details').textContent=data.source;};
document.getElementById('out').onclick=()=>show({...current,span:Math.min(3.5,current.span*4)});
document.getElementById('reset').onclick=()=>show({real:-.5,imag:0,span:3.5,seed:1234},true,true);
document.getElementById('new').onclick=()=>show({...current,seed:current.seed+1});
document.querySelectorAll('[data-pan]').forEach(b=>b.onclick=()=>{let [x,y]=b.dataset.pan.split(',').map(Number);show({...current,offset_real:x*current.span/4,offset_imag:y*current.span/4});});
document.getElementById('go').onclick=()=>show({...current,...Object.fromEntries(['real','imag','span'].map(k=>[k,k==="span"?Number(document.getElementById(k).value):document.getElementById(k).value]))});try{current=JSON.parse(localStorage.getItem("mandelbrot-camera"))||current;}catch{}show(current,false);
</script>'''


def png(tensor):
    pixels=(tensor.detach().clamp(0,1)*255).round().byte().permute(1,2,0).cpu().numpy()
    stream=BytesIO();Image.fromarray(pixels).save(stream,format='PNG')
    return base64.b64encode(stream.getvalue()).decode()


class Explorer:
    def __init__(self, checkpoint):
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=True
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        self.step=saved['step']; config=saved['config']
        self.model=build_model(config).cuda().eval()
        self.model.load_state_dict(saved['ema'])
        self.flow=ConditionalFlow(**config.get('flow',{}))
        self.color=DatasetRenderer(side=2,mode='float64',palette='smooth')

    @torch.inference_mode()
    def view(self, request):
        with localcontext() as context:
            context.prec=50
            real_exact=Decimal(str(request['real']))+Decimal(str(request.get('offset_real',0)))
            imag_exact=Decimal(str(request['imag']))+Decimal(str(request.get('offset_imag',0)))
        real,imag,span=float(real_exact),float(imag_exact),float(request['span'])
        seed=int(request.get('seed',1234))
        if not all(math.isfinite(x) for x in (real,imag,span)) or abs(real)>4 or abs(imag)>4:
            raise ValueError('Choose a finite center within [-4,4].')
        if not 1e-24 <= span <= 3.5:
            raise ValueError('Reference precision limit reached: span must be between 1e-24 and 3.5.')
        if not 0 <= seed < 2**63: raise ValueError('Invalid noise seed.')
        start=time.perf_counter()
        exact=(str(real_exact),str(imag_exact))
        precision='perturb64' if span < 1e-9 else 'float64'
        renderer=FastRenderer([View(complex(real,imag),span*factor,8192,center_exact=exact) for factor in (1,)],
                              side=2,precision=precision,tile=64)
        images=self.color._render(renderer).permute(0,3,1,2)
        target=images[:1]*2-1
        parent=request.get('parent_image')
        if parent is None:
            # A rendered starting image is the only ground-truth seed.
            evidence=target
            prediction=images[:1]
            source='Rendered initial seed'
            torch.cuda.synchronize(); rendered=time.perf_counter()
        else:
            image=Image.open(BytesIO(base64.b64decode(parent,validate=True)))
            if image.size != (128,128) or image.mode != 'RGB':
                raise ValueError('Parent must be a 128x128 RGB image')
            values=torch.tensor(list(image.getdata()),device='cuda',dtype=torch.float32)
            previous=values.reshape(128,128,3).permute(2,0,1)[None]/127.5-1
            camera=request['parent_camera']
            parent_span=float(camera['span'])
            if not math.isfinite(parent_span) or parent_span<=0:raise ValueError('Invalid parent span')
            with localcontext() as context:
                context.prec=50
                dx=float((real_exact-Decimal(str(camera['real'])))/Decimal(str(parent_span)))
                dy=float((imag_exact-Decimal(str(camera['imag'])))/Decimal(str(parent_span)))
            if not all(math.isfinite(v) for v in (dx,dy)):raise ValueError('Invalid camera offset')
            axis=(torch.arange(32,device='cuda',dtype=torch.float32)+.5)/32-.5
            yy,xx=torch.meshgrid(axis,axis,indexing='ij')
            grid=torch.stack((2*(dx+xx*span/parent_span),2*(-dy+yy*span/parent_span)),dim=-1)[None]
            weights=torch.nn.functional.grid_sample(torch.ones_like(previous[:,:1]),grid,align_corners=False)
            patch=torch.nn.functional.grid_sample(previous,grid,align_corners=False)
            patch=torch.where(weights>1e-6,patch/weights.clamp_min(1e-6),0.)
            _,evidence,mask=reconstruct_conditioning(torch.empty_like(previous),patch,weights)
            torch.cuda.synchronize(); rendered=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                prediction=self.flow.sample(self.model,evidence,mask,steps=50,solver='heun',seed=seed)
            source='Recursive model output'
        torch.cuda.synchronize();sampled=time.perf_counter()
        return dict(input=png((evidence[0]+1)/2),prediction=png(prediction[0]),target=png(images[0]),step=self.step,source=source,
                    render_seconds=rendered-start,model_seconds=sampled-rendered,
                    camera=dict(real=str(real_exact),imag=str(imag_exact),span=span,seed=seed),
                    outside_training=span<3.5/4**12,precision=precision)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=Path('runs/jit-10m-smooth-1m-patch32-pilot/checkpoint.pt'))
    parser.add_argument('--host',default='127.0.0.1');parser.add_argument('--port',type=int,default=8766)
    args=parser.parse_args();explorer=Explorer(args.checkpoint)
    inference_lock=threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def send(self,code,body,kind):
            self.send_response(code);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def do_GET(self):
            if self.path != '/':self.send(404,b'Not found','text/plain');return
            self.send(200,PAGE.encode(),'text/html; charset=utf-8')
        def do_POST(self):
            if self.path != '/view':self.send(404,b'Not found','text/plain');return
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=200000:raise ValueError('Invalid request length')
                request=json.loads(self.rfile.read(size))
                with inference_lock:
                    result=explorer.view(request)
            except (ValueError,KeyError,TypeError,OverflowError) as error:
                self.send(400,json.dumps({'error':str(error)}).encode(),'application/json');return
            self.send(200,json.dumps(result).encode(),'application/json')
    print(f'Explorer checkpoint {explorer.step}: http://{args.host}:{args.port}',flush=True)
    ThreadingHTTPServer((args.host,args.port),Handler).serve_forever()


if __name__=='__main__':main()
