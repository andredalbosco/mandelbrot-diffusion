"""Export camera metadata only to a static, interactive sampling heat map."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import torch


def export(dataset, output):
    manifest_path=dataset/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    output.mkdir(parents=True,exist_ok=True)
    counts=Counter();depths=Counter();total=0
    modes={'interior':0,'edge':1,'corner':2,'unconditional':3}
    temporary=output/'sample-centers.tmp'
    with temporary.open('wb') as stream:
        for shard in manifest['shards']:
            for split,flag in [('train',0),('val',1)]:
                payload=torch.load(dataset/shard['files'][split]['path'],mmap=True,weights_only=True,map_location='cpu')
                records=payload['records']
                assert len(records)==shard['files'][split]['count']
                for index,r in enumerate(records):
                    # 32 bytes: float64 x,y,span; uint8 depth,split,mode,padding; uint16 shard,index.
                    stream.write(struct.pack('<dddBBBBHH',*r['center'],r['span'],r['zoom_depth'],flag,modes[r['mode']],0,shard['index'],index))
                    counts[split]+=1;depths[r['zoom_depth']]+=1;total+=1
                del payload
            print(f'Exported {total:,} centers',flush=True)
    assert total==manifest['completed_count']
    temporary.replace(output/'sample-centers.bin')
    metadata=dict(dataset=dataset.name,count=total,complete=manifest['complete'],counts=counts,
                  depth_counts=depths,record_bytes=32,source_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    (output/'sample-centers.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (output/'sample-heatmap.html').write_text(PAGE)


PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mandelbrot · Sample locations</title><link rel="icon" href="data:,">
<style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#10151e;color:#e7edf5;font:15px/1.5 system-ui,sans-serif}main{max-width:1450px;margin:auto;padding:24px}h1{font-size:28px;margin:0 0 6px}p{color:#aebed0;margin:8px 0 18px}.controls{display:flex;flex-wrap:wrap;gap:14px;align-items:end;padding:16px;background:#192330;border:1px solid #2e4056;border-radius:12px}label{display:flex;gap:6px;flex-direction:column;font-size:13px;color:#c9d5e4}label.check{flex-direction:row;align-items:center;padding:8px}select,button{background:#25374b;color:white;border:1px solid #49627b;border-radius:6px;padding:8px;font:inherit}button{cursor:pointer}button:hover{background:#35516d}.stats{display:flex;gap:24px;flex-wrap:wrap;margin:18px 0}.stats strong{font-size:22px;display:block}.stats span{color:#aebed0;font-size:13px}#wrap{position:relative;border:1px solid #354960;border-radius:10px;overflow:hidden}canvas#map{display:block;width:100%;height:min(70vh,850px);min-height:380px;touch-action:none;cursor:crosshair}#tip{position:absolute;pointer-events:none;background:#111e2eee;border:1px solid #65819c;padding:8px 12px;border-radius:6px;white-space:pre;font:12px/1.6 monospace;display:none}.legend{display:flex;align-items:center;gap:12px;margin-top:12px;flex-wrap:wrap}.ramp{width:240px;height:12px;background:linear-gradient(90deg,#173c7b,#13a7ba,#fee580,#ff6c31,#fff4df)}.notes{font-size:13px}a{color:#8fcfff}#selection{font:13px/1.7 monospace;white-space:pre-wrap;margin-top:16px;padding:12px;background:#192330;border-radius:8px;display:none}#status{margin:14px 0;color:#ffe09a}small{color:#aebed0}
</style></head><body><main>
<h1>Where the Mandelbrot samples were generated</h1>
<p>Target-center density across the complex plane. Scroll to zoom, drag to pan, and click to inspect the nearest visible sample. The map uses all exported samples, not a thumbnail subset.</p>
<div class="controls">
<label>Split<select id="split"><option value="-1">Train + validation</option><option value="0">Training</option><option value="1">Validation</option></select></label>
<label>Minimum depth<select id="lo"></select></label><label>Maximum depth<select id="hi"></select></label>
<label>Input mode<select id="mode"><option value="-1">All modes</option><option value="0">Interior</option><option value="1">Edge</option><option value="2">Corner</option><option value="3">Unconditional</option></select></label>
<label>Heat-map bins<select id="bins"><option>128</option><option selected>256</option><option>512</option></select></label>
<label>Color scale<select id="scale"><option value="log">Logarithmic</option><option value="linear">Linear</option></select></label>
<label class="check"><input id="reference" type="checkbox" checked>Mandelbrot backdrop</label>
<button id="focus" disabled>Zoom to selected sample</button><button id="reset">Reset view</button><button id="fit">Fit filtered samples</button>
</div>
<div id="status">Loading sample locations…</div>
<div class="stats"><div><strong id="total">—</strong><span>Exported samples</span></div><div><strong id="filtered">—</strong><span>Match filters</span></div><div><strong id="visible">—</strong><span>Centers in view</span></div><div><strong id="peak">—</strong><span>Peak count per bin</span></div></div>
<div id="wrap"><canvas id="map"></canvas><div id="tip"></div></div>
<div class="legend"><span>Empty bins are transparent</span><div class="ramp"></div><span id="legend">1 → peak</span></div>
<p id="extent" class="notes"></p><div id="selection"></div>
<p class="notes">Colors represent center counts per bin, not the area covered by image footprints. Colors rescale to the visible peak after each change. Depth is the generator’s nominal bin; actual scales vary within each bin. The backdrop is an approximate 128-iteration Mandelbrot reference. Deep zooms may not show its boundary accurately.</p>
<p class="notes">The generator partitions initial anchor groups between splits, but crop footprints can overlap. This map does not establish geographic independence. <a href="dataset-1m.html">Dataset previews and generation status</a></p>
</main><script>
const $=id=>document.getElementById(id), canvas=$('map'),ctx=canvas.getContext('2d');
let raw, doubles, bytes, n=0, ready=false, hist, bw=256,bh=256, geometry,selected=null;
let view={x:-.65,y:0,span:3.3},drag=null,queued=false,showFit=false;
const palette=[[23,60,123],[19,167,186],[254,229,128],[255,108,49],[255,244,223]];
for(let i=0;i<=12;i++){ $('lo').add(new Option(i===0?'0 · overview':String(i),String(i)));$('hi').add(new Option(String(i),String(i))); }$('hi').value='12';
function filters(){return {split:+$('split').value,lo:+$('lo').value,hi:+$('hi').value,mode:+$('mode').value};}
function matches(i,f){const o=i*32;return bytes[o+24]>=f.lo&&bytes[o+24]<=f.hi&&(f.split<0||bytes[o+25]===f.split)&&(f.mode<0||bytes[o+26]===f.mode);}
function bounds(){const w=canvas.clientWidth,h=canvas.clientHeight,sy=view.span*h/w;return {left:view.x-view.span/2,right:view.x+view.span/2,bottom:view.y-sy/2,top:view.y+sy/2,sy,w,h};}
function locate(event){const r=canvas.getBoundingClientRect(),b=bounds(),px=event.clientX-r.left,py=event.clientY-r.top;return {px,py,x:b.left+px/b.w*view.span,y:b.top-py/b.h*b.sy};}
function schedule(){if(!ready||queued)return;queued=true;requestAnimationFrame(()=>{queued=false;render();});}
function backdrop(b){const w=300,h=Math.max(1,Math.round(w*b.h/b.w)),c=document.createElement('canvas');c.width=w;c.height=h;const cc=c.getContext('2d'),im=cc.createImageData(w,h);
for(let py=0;py<h;py++)for(let px=0;px<w;px++){let cr=b.left+(px+.5)/w*view.span,ci=b.top-(py+.5)/h*b.sy,zr=0,zi=0,k=0;while(k<128&&zr*zr+zi*zi<=4){let next=zr*zr-zi*zi+cr;zi=2*zr*zi+ci;zr=next;k++;}const o=4*(py*w+px),v=k===128?48:Math.min(38,10+Math.log1p(k)*4);im.data[o]=v;im.data[o+1]=v+7;im.data[o+2]=v+15;im.data[o+3]=255;}cc.putImageData(im,0,0);ctx.drawImage(c,0,0,b.w,b.h);}
function render(){const dpr=Math.min(devicePixelRatio||1,2),b=bounds();canvas.width=Math.round(b.w*dpr);canvas.height=Math.round(b.h*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);ctx.fillStyle='#10151e';ctx.fillRect(0,0,b.w,b.h);if($('reference').checked)backdrop(b);
bw=+$('bins').value;bh=Math.max(1,Math.round(bw*b.h/b.w));hist=new Uint32Array(bw*bh);const f=filters();let count=0,inView=0,minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity;
for(let i=0;i<n;i++){if(!matches(i,f))continue;count++;const x=doubles[i*4],y=doubles[i*4+1];minX=Math.min(minX,x);maxX=Math.max(maxX,x);minY=Math.min(minY,y);maxY=Math.max(maxY,y);const ix=Math.floor((x-b.left)/view.span*bw),iy=Math.floor((b.top-y)/b.sy*bh);if(ix>=0&&ix<bw&&iy>=0&&iy<bh){hist[iy*bw+ix]++;inView++;}}
if(showFit){showFit=false;if(count){view.x=(minX+maxX)/2;view.y=(minY+maxY)/2;view.span=Math.max(maxX-minX,(maxY-minY)*b.w/b.h,1e-9)*1.15;schedule();return;}}
let peak=0;for(const v of hist)peak=Math.max(peak,v);const hc=document.createElement('canvas');hc.width=bw;hc.height=bh;const hctx=hc.getContext('2d'),image=hctx.createImageData(bw,bh);const log=$('scale').value==='log';
for(let i=0;i<hist.length;i++){if(!hist[i])continue;const t=peak<=1?1:(log?Math.log(hist[i])/Math.log(peak):(hist[i]-1)/(peak-1));const a=Math.min(3,Math.floor(t*4)),u=t*4-a;for(let j=0;j<3;j++)image.data[i*4+j]=palette[a][j]*(1-u)+palette[a+1][j]*u;image.data[i*4+3]=235;}hctx.putImageData(image,0,0);ctx.imageSmoothingEnabled=false;ctx.drawImage(hc,0,0,b.w,b.h);
ctx.strokeStyle='#b0c5e540';ctx.lineWidth=1;ctx.beginPath();const zx=(0-b.left)/view.span*b.w,zy=b.top/b.sy*b.h;ctx.moveTo(zx,0);ctx.lineTo(zx,b.h);ctx.moveTo(0,zy);ctx.lineTo(b.w,zy);ctx.stroke();
if(selected!==null){const x=doubles[selected*4],y=doubles[selected*4+1],span=doubles[selected*4+2];ctx.strokeStyle='#fff';ctx.lineWidth=1.5;ctx.strokeRect((x-span/2-b.left)/view.span*b.w,(b.top-y-span/2)/b.sy*b.h,span/view.span*b.w,span/b.sy*b.h);}
$('filtered').textContent=count.toLocaleString();$('visible').textContent=inView.toLocaleString();$('peak').textContent=peak.toLocaleString();$('legend').textContent=`1 → ${peak.toLocaleString()} samples / bin (${$('scale').value})`;$('extent').textContent=`Real: ${b.left.toPrecision(9)} … ${b.right.toPrecision(9)} · Imaginary: ${b.bottom.toPrecision(9)} … ${b.top.toPrecision(9)} · Bin width: ${(view.span/bw).toPrecision(4)}`;geometry=b;window.heatmapState={total:n,filtered:count,visible:inView,peak,sum:hist.reduce((a,v)=>a+v,0)};}
canvas.addEventListener('wheel',e=>{e.preventDefault();if(!ready)return;const p=locate(e),factor=Math.exp(Math.max(-1,Math.min(1,e.deltaY*.002)));const span=Math.max(1e-12,Math.min(20,view.span*factor)),ratio=span/view.span;view.x=p.x+(view.x-p.x)*ratio;view.y=p.y+(view.y-p.y)*ratio;view.span=span;schedule();},{passive:false});
canvas.addEventListener('pointerdown',e=>{if(!ready)return;drag={px:e.clientX,py:e.clientY,x:view.x,y:view.y,moved:0};canvas.setPointerCapture(e.pointerId);});
canvas.addEventListener('pointermove',e=>{if(!ready)return;const p=locate(e);if(drag){const dx=e.clientX-drag.px,dy=e.clientY-drag.py;drag.moved=Math.max(drag.moved,Math.hypot(dx,dy));view.x=drag.x-dx/canvas.clientWidth*view.span;view.y=drag.y+dy/canvas.clientWidth*view.span;$('tip').style.display='none';schedule();return;}if(!hist)return;const ix=Math.floor(p.px/canvas.clientWidth*bw),iy=Math.floor(p.py/canvas.clientHeight*bh);const tip=$('tip');tip.textContent=`Re ${p.x.toPrecision(10)}\nIm ${p.y.toPrecision(10)}\n${hist[iy*bw+ix]||0} sample centers in bin`;tip.style.display='block';tip.style.left=Math.min(p.px+15,canvas.clientWidth-245)+'px';tip.style.top=Math.max(0,p.py-80)+'px';});
canvas.addEventListener('pointerup',e=>{if(!drag)return;const moved=drag.moved;drag=null;if(moved>4)return;const p=locate(e),f=filters();let best=-1,distance=Infinity;for(let i=0;i<n;i++){if(!matches(i,f))continue;const dx=(doubles[i*4]-p.x)/view.span*canvas.clientWidth,dy=(doubles[i*4+1]-p.y)/view.span*canvas.clientWidth,dist=dx*dx+dy*dy;if(dist<distance){distance=dist;best=i;}}if(best>=0&&distance<900){selected=best;$('focus').disabled=false;const v=new DataView(raw),o=best*32;const split=bytes[o+25]?'val':'train';$('selection').style.display='block';$('selection').textContent=`Center: (${doubles[best*4].toPrecision(17)}, ${doubles[best*4+1].toPrecision(17)})\nSpan: ${doubles[best*4+2].toPrecision(10)} · depth ${bytes[o+24]} · ${split} · ${['interior','edge','corner','unconditional'][bytes[o+26]]}\nShard ${v.getUint16(o+28,true)}, sample ${v.getUint16(o+30,true)}. White square = this target’s footprint.`;schedule();}});
canvas.addEventListener('pointercancel',()=>drag=null);canvas.addEventListener('pointerleave',()=>$('tip').style.display='none');
for(const id of ['split','lo','hi','mode','bins','scale','reference'])$(id).addEventListener('change',()=>{if(+$('lo').value>+$('hi').value){if(id==='lo')$('hi').value=$('lo').value;else $('lo').value=$('hi').value;}selected=null;$('focus').disabled=true;$('selection').style.display='none';schedule();});
$('focus').onclick=()=>{if(selected!==null){view={x:doubles[selected*4],y:doubles[selected*4+1],span:doubles[selected*4+2]*4};schedule();}};
$('reset').onclick=()=>{view={x:-.65,y:0,span:Math.max(3.3,2.6*canvas.clientWidth/canvas.clientHeight)};selected=null;schedule();};$('fit').onclick=()=>{showFit=true;schedule();};window.addEventListener('resize',schedule);
(async()=>{try{const metadata=await fetch('sample-centers.json').then(r=>{if(!r.ok)throw Error(r.status);return r.json();});const r=await fetch('sample-centers.bin');if(!r.ok)throw Error(r.status);raw=await r.arrayBuffer();if(raw.byteLength!==metadata.count*32)throw Error('Metadata and binary size differ');doubles=new Float64Array(raw);bytes=new Uint8Array(raw);n=metadata.count;ready=true;showFit=true;$('total').textContent=n.toLocaleString();$('status').textContent=metadata.dataset+' · '+(metadata.complete?'generation complete':'partial export');schedule();}catch(e){$('status').textContent='Could not load sample locations: '+e.message;}})();
</script></body></html>'''

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=Path('data/zoom4x-smooth-blue-gold-1m-128'))
    parser.add_argument('--output',type=Path,default=Path('output/dataset-browser'))
    args=parser.parse_args()
    export(args.dataset,args.output)
