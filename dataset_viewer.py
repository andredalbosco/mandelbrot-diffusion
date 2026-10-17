"""Export local tensor datasets to a static HTML browser (no server required)."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time

import torch
from PIL import Image


PALETTES = {"smooth": "Smooth · blue & gold", "histogram": "Histogram equalized",
            "stripes": "Orbit stripes", "orbit_trap": "Orbit trap · cross"}


def histogram_coordinates(escape):
    """One smooth CDF per parent/child pair, excluding unescaped samples."""
    pairs = len(escape) // 2
    fields = torch.stack(escape.chunk(2), dim=1)
    valid = fields != -1
    values = fields.clamp_min(0).log1p()
    low = values.masked_fill(~valid, float('inf')).flatten(1).amin(1)
    high = values.masked_fill(~valid, -float('inf')).flatten(1).amax(1)
    low = torch.where(torch.isfinite(low), low, 0.)
    high = torch.where(torch.isfinite(high), high, low + 1.)
    position = ((values-low[:,None,None,None])/(high-low).clamp_min(1e-6)[:,None,None,None]*511).clamp(0,511)
    index = position.long().flatten(1)
    counts = values.new_zeros((pairs,512)).scatter_add_(1,index,valid.float().flatten(1))
    cdf = counts.cumsum(1) / counts.sum(1,keepdim=True).clamp_min(1)
    before = torch.cat((cdf.new_zeros((pairs,1)),cdf[:,:-1]),dim=1)
    alpha = before.gather(1,index)+(cdf-before).gather(1,index)*(position.flatten(1)-index)
    alpha = alpha.reshape_as(fields)
    return torch.cat((alpha[:,0],alpha[:,1]))


def coloring_coordinates(escape, orbit):
    """Four Mandelbrot-specific color indices, evaluated before reconstruction."""
    return {
        'smooth': (escape.clamp_min(0).log1p()*.7).remainder(1),
        'histogram': histogram_coordinates(escape),
        'stripes': orbit[...,0],
        'orbit_trap': (-torch.log10(orbit[...,1].clamp_min(1e-12))*.2).remainder(1),
    }


def save_sheet(images, path):
    """Pack up to 64 RGB thumbnails into a lossless 8-by-8 sprite sheet."""
    sheet = Image.new('RGB', (1024, 1024), (24, 29, 33))
    for j, image in enumerate(images):
        pixels = (image.clamp(0, 1) * 255).round().byte()
        pixels = pixels.expand(3, -1, -1).permute(1, 2, 0).contiguous().numpy()
        thumb = Image.fromarray(pixels).resize((128, 128), Image.Resampling.NEAREST)
        sheet.paste(thumb, ((j % 8) * 128, (j // 8) * 128))
    sheet.save(path)


def add_palette_catalog(entry, data, output):
    assets = data / 'dataset-viewer'
    metadata = assets / f'{entry["name"]}-palettes.json'
    if not metadata.exists():
        return
    preview = json.loads(metadata.read_text())
    manifest = data / entry['name'] / 'manifest.json'
    if preview['manifest_sha256'] != hashlib.sha256(manifest.read_bytes()).hexdigest():
        return
    entry['palettes'] = preview['palettes']
    for split, info in entry['splits'].items():
        info['palettes'] = {
            name: {key: [Path(os.path.relpath(assets / filename, output.parent)).as_posix()
                         for filename in pages] for key, pages in sheets.items()}
            for name, sheets in preview['splits'][split].items()}


def write_viewer(catalog, output):
    encoded = json.dumps(catalog, separators=(',', ':')).replace('<', '\\u003c')
    temporary = output.with_suffix('.tmp')
    temporary.write_text(PAGE.replace('__CATALOG__', encoded))
    temporary.replace(output)


def export_palette_previews(data, output, names):
    """Rerender numeric samples once per page, then reuse them for every palette.

    Only preview PNGs and the HTML catalog change; dataset tensors stay intact.
    Includes smooth escape, paired-histogram, post-escape stripe and cross-trap
    coloring. Each pair shares its color mapping, including the histogram CDF.
    """
    from dataset import zoom_evidence
    from mandelbrot import View
    from mandelbrot_fast import FastRenderer

    torch.set_num_threads(4)
    if not output.exists():
        export_viewer(data, output)
    catalog = json.loads(re.search(r'<script id="catalog" type="application/json">(.*?)</script>',
                                  output.read_text(), re.S).group(1))
    # Closed blue/white/gold and teal/copper gradients, shared across locations.
    blue_gold = torch.tensor([[0,7,80],[20,70,180],[80,170,230],[235,250,255],
                              [255,185,45],[130,45,10],[0,7,80]],device='cuda')/255
    teal_copper = torch.tensor([[5,20,28],[20,125,140],[210,240,225],[250,180,80],
                                [150,45,20],[30,12,30],[5,20,28]],device='cuda')/255
    tables = {p:teal_copper if p=='orbit_trap' else blue_gold for p in PALETTES}
    assets = data / 'dataset-viewer'
    assets.mkdir(exist_ok=True)
    for name in names:
        entry = next((d for d in catalog if d['name'] == name), None)
        if entry is None:
            raise ValueError(f'Dataset not in viewer: {name}')
        directory = data / name
        manifest = json.loads((directory / 'manifest.json').read_text())
        started = time.perf_counter()
        metadata = {'palettes': PALETTES, 'splits': {},
                    'manifest_sha256': hashlib.sha256((directory / 'manifest.json').read_bytes()).hexdigest(),
                    'mapping': 'Smooth: 0.7*log1p(escape) cycles. Histogram: interpolated 512-bin log-escape CDF shared per parent/target pair. Stripes: mathr post-escape variant, five octaves. Cross trap: -0.2*log10(min(|Re z|,|Im z|)) cycles over z1..escape. Interior black; color before tent reconstruction.',
                    'references': ['https://www.ultrafractal.com/help/coloring/standard/smooth.html',
                                   'https://www.mathr.co.uk/web/stripe-colouring.html',
                                   'https://www.ultrafractal.com/help/coloring/standard/orbittraps.html'],
                    'reference_max_error': 0.0}
        for split, info in entry['splits'].items():
            payload = torch.load(directory / f'{split}.pt', weights_only=True, mmap=True, map_location='cpu')
            if info['shape'] != [3, 128, 128] or 'targets' not in payload:
                raise ValueError('Palette previews require an RGB 128x128 zoom dataset')
            records = payload['records']
            metadata['splits'][split] = {p: {'inputs': [], 'targets': []} for p in PALETTES}
            for start in range(0, len(records), 64):
                chunk = records[start:start+64]
                views = [View(complex(*r['parent_center']), r['parent_span'], r['iterations']) for r in chunk]
                views += [View(complex(*r['center']), r['span'], r['iterations']) for r in chunk]
                renderer = FastRenderer(views, side=manifest['config']['aa_side'], precision='float64')
                original = renderer.render().permute(0, 3, 1, 2).cpu()
                error = (original[len(chunk):] - payload['targets'][start:start+len(chunk)]).abs().max().item()
                metadata['reference_max_error'] = max(metadata['reference_max_error'], error)
                if error > 1e-4:
                    raise ValueError(f'Rerender disagrees with stored targets: {name}/{split}/{start}: {error}')
                coordinates = coloring_coordinates(renderer.escape, renderer.orbit_coloring())
                for palette, table in tables.items():
                    images = renderer.recolor(table, coordinates[palette]).permute(0, 3, 1, 2)
                    parent, target = images.chunk(2)
                    evidence, _ = zoom_evidence(parent, [r['crop_left'] for r in chunk], [r['crop_top'] for r in chunk])
                    for i, record in enumerate(chunk):
                        if record['mode'] == 'unconditional':
                            evidence[i].fill_(.5)
                    for key, tensor in [('targets', target), ('inputs', evidence)]:
                        filename = f'{name}-{split}-palette-{palette}-{key}-{start//64}.png'
                        save_sheet(tensor.cpu(), assets / filename)
                        metadata['splits'][split][palette][key].append(filename)
                del renderer
                print(f'{name}/{split}: palettes {min(start+64,len(records))}/{len(records)}', flush=True)
        metadata['seconds'] = time.perf_counter() - started
        (assets / f'{name}-palettes.json').write_text(json.dumps(metadata, indent=2) + '\n')
        add_palette_catalog(entry, data, output)
        write_viewer(catalog, output)
        print(f'Published {name}: {len(PALETTES)} palettes, {metadata["seconds"]:.1f}s', flush=True)


def sobel_gradients(images):
    """Mean-score source: luminance Sobel magnitude after one binomial blur.

    All images are evaluated at 128x128. Kernels are normalized (Sobel /8),
    reflect padding avoids a false frame edge, and maps share one fixed scale.
    """
    from torch.nn import functional as F
    x = images.float()
    if x.ndim != 4 or x.shape[1] not in (1,3):
        raise ValueError("Expected [N,1|3,H,W]")
    if x.shape[1] == 3:
        x = (x * x.new_tensor([.2126,.7152,.0722])[None,:,None,None]).sum(1,keepdim=True)
    x = F.interpolate(x,size=(128,128),mode="bilinear",align_corners=False)
    blur = x.new_tensor([[1,2,1],[2,4,2],[1,2,1]])[None,None]/16
    x = F.conv2d(F.pad(x,(1,1,1,1),mode="reflect"),blur)
    kernels = x.new_tensor([[[-1,0,1],[-2,0,2],[-1,0,1]],
                            [[-1,-2,-1],[0,0,0],[1,2,1]]])[:,None]/8
    g = F.conv2d(F.pad(x,(1,1,1,1),mode="reflect"),kernels)
    return g.square().sum(1,keepdim=True).sqrt()


PAGE = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mandelbrot · Datasets</title>
<style>
:root{color-scheme:dark;font:15px system-ui,sans-serif;background:#101316;color:#e8ebee}*{box-sizing:border-box}body{margin:0}header{padding:25px 32px 18px;border-bottom:1px solid #30383e;display:flex;justify-content:space-between;align-items:center;gap:20px}h1{font-size:25px;letter-spacing:-.7px;margin:0 0 4px}.muted{color:#9da9b2;font-size:13px}main{max-width:1680px;margin:auto;padding:22px 32px}label{font-size:12px;color:#a9b6bf;display:grid;gap:7px}select,button,input{font:inherit;color:inherit;background:#20272d;border:1px solid #3a464e;border-radius:7px;padding:9px 12px}button{cursor:pointer}button:hover{background:#303e47}button:disabled{opacity:.4;cursor:default}.filters{display:flex;gap:14px;flex-wrap:wrap;align-items:end;margin-bottom:18px}#dataset{min-width:270px}.stats{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0 20px}.pill{font-size:12px;background:#202a30;border-radius:5px;padding:6px 9px;color:#b9d5d7}.panels{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}figure{margin:0;background:#1a2025;border:1px solid #30383e;border-radius:9px;overflow:hidden}figcaption{padding:11px 13px;color:#b8c6cf;font-size:13px}canvas{image-rendering:pixelated;display:block;width:100%;aspect-ratio:1;background:#242a2f}.palette-row{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin:0 0 18px}.comparison{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:24px}.comparison[hidden]{display:none}.empty{display:grid;place-items:center;aspect-ratio:1;color:#81909a;font-size:13px}.navigation{display:flex;align-items:center;gap:10px;margin:18px 0}.navigation input{width:100px}#position{margin-left:auto}.gallery{display:grid;grid-template-columns:repeat(8,minmax(0,1fr));gap:10px}.thumb{padding:0;overflow:hidden;text-align:left;background:#1a2025;min-width:0}.thumb.active{outline:2px solid #67c8c5}.thumb span{display:block;padding:7px 9px;font-size:11px;color:#b8c6cf;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.sectionhead{display:flex;justify-content:space-between;align-items:center;margin:26px 0 13px}.sectionhead h2{font-size:16px;margin:0}.sectionhead button{padding:5px 12px}details{margin:18px 0;border-top:1px solid #30383e;padding-top:14px}summary{cursor:pointer;color:#b9c8d0}pre{white-space:pre-wrap;word-break:break-word;font-size:12px;line-height:1.6;color:#aabcc8}#notice{background:#252b24;border-left:3px solid #91a674;padding:12px 14px;line-height:1.5;color:#c3cdbb;font-size:13px;margin-bottom:18px}a{color:#7bd1cd}#error{color:#ffb6a6}@media(max-width:1000px){.gallery{grid-template-columns:repeat(4,minmax(0,1fr))}main,header{padding:18px}.filters{gap:10px}}@media(max-width:650px){.panels{grid-template-columns:1fr}.gallery{grid-template-columns:repeat(3,minmax(0,1fr))}header{align-items:start;flex-direction:column}.navigation{flex-wrap:wrap}#position{margin-left:0}}
</style>
<header><div><h1>Mandelbrot datasets</h1><div class="muted">Rendered samples · 4× zoom evidence · coverage masks</div></div><div class="muted">Local, offline browser · nearest-neighbor display</div></header>
<main><div class="filters"><label>Dataset<select id="dataset"></select></label><label>Split<select id="split"></select></label><label>Absolute zoom<select id="depth"></select></label><label>Evidence mode<select id="mode"></select></label><label>Target display<select id="display"><option value="rgb">Rendered RGB</option><option value="sobel">Sobel gradient</option><option value="overlay">Sobel overlay</option></select></label><label>Gallery order<select id="sort"><option value="index">Sample index</option><option value="high">Highest Sobel score</option><option value="low">Lowest Sobel score</option></select></label><label>Gradient gain<select id="gain"><option value="1">1×</option><option value="2">2×</option><option value="4" selected>4×</option><option value="8">8×</option></select></label><button id="random">Random sample</button></div>
<div class="palette-row"><label>Coloring scheme<select id="palette"><option value="original">Original dataset RGB</option></select></label><button id="compare" aria-pressed="false">Compare schemes</button><span id="paletteNote" class="muted"></span></div><div id="comparison" class="comparison" hidden></div><div id="stats" class="stats"></div><div id="notice"></div><div id="error"></div>
<div class="panels"><figure><figcaption>Input · upsampled coarse evidence</figcaption><div id="input"></div></figure><figure><figcaption id="targetLabel">Target · mathematical render</figcaption><div id="target"></div></figure><figure><figcaption>Mask · coarse evidence coverage</figcaption><div id="mask"></div></figure></div>
<div class="navigation"><button id="previous">← Previous</button><label>Sample index<input id="index" type="number" min="0"></label><button id="next">Next →</button><span id="position" class="muted"></span></div>
<details><summary>Sample camera and rendering settings</summary><pre id="record"></pre></details><details><summary>Dataset manifest and generator measurements</summary><pre id="manifest"></pre></details>
<div class="sectionhead"><h2>Target gallery</h2><div><button id="pagePrev">←</button> <span id="pageLabel" class="muted"></span> <button id="pageNext">→</button></div></div><div id="gallery" class="gallery"></div>
<p class="muted">Mandelbrot coloring: <a href="https://www.ultrafractal.com/help/coloring/standard/smooth.html" target="_blank" rel="noopener">smooth escape-time</a> · <a href="https://www.mathr.co.uk/web/stripe-colouring.html" target="_blank" rel="noopener">post-escape orbit stripes</a> · <a href="https://www.ultrafractal.com/help/coloring/standard/orbittraps.html" target="_blank" rel="noopener">orbit traps</a>. Histogram coloring shares one distribution per input/target pair. All schemes color spatial samples before tent filtering. Training tensors are unchanged.</p><p class="muted">Sobel views and scores always use the original stored RGB, regardless of the palette preview. Sobel score = mean luminance gradient at 128×128 after a 3×3 binomial blur. Fixed heatmap scale across images; gain changes display only. Bright regions have stronger gradients. Palette transitions and remaining aliasing can also score highly. Sorting ranks the current filters; use a depth filter to compare one scale.</p><p class="muted">Masks describe available coarse observations; white pixels are not exact target pixels to preserve. Broad views and crop footprints may overlap across splits. Preview PNGs are 8-bit; training tensors retain floating-point values.</p>
</main><script id="catalog" type="application/json">__CATALOG__</script>
<script>
const datasets=JSON.parse(document.getElementById('catalog').textContent),$=id=>document.getElementById(id);
const cache=new Map();let selected=0,page=0,filtered=[],token=0,comparing=false,firstBuild=true;
function options(el,values,value){el.replaceChildren(...values.map(([v,t])=>{const o=document.createElement('option');o.value=v;o.textContent=t;return o}));if(value!==undefined)el.value=value}
function state(){const d=datasets.find(x=>x.name===$('dataset').value);return[d,d.splits[$('split').value]]}
function image(url){if(!cache.has(url))cache.set(url,new Promise((resolve,reject)=>{const im=new Image();im.onload=()=>resolve(im);im.onerror=()=>reject(new Error('Missing preview: '+url+'. Rebuild with python dataset_viewer.py.'));im.src=url}));if(cache.size>48)cache.delete(cache.keys().next().value);return cache.get(url)}
function drawTile(canvas,im,index){canvas.getContext('2d').drawImage(im,(index%8)*128,Math.floor((index%64)/8)*128,128,128,0,0,128,128)}
function heat(v){const stops=[[10,9,32],[59,19,89],[148,38,103],[224,91,51],[252,200,83],[249,253,181]],p=Math.min(.999999,Math.max(0,v))*(stops.length-1),i=Math.floor(p),t=p-i;return stops[i].map((c,k)=>Math.round(c*(1-t)+stops[i+1][k]*t))}
async function tile(host,key,index,version,palette=$('palette').value){const[,s]=state();const sheets=$('display').value==='rgb'&&(key==='inputs'||key==='targets')&&palette!=='original'?s.palettes?.[palette]:s.sheets;host.replaceChildren();if(!sheets?.[key]){host.className='empty';host.textContent='Not stored in this dataset';return}host.className='';const canvas=document.createElement('canvas');canvas.width=canvas.height=128;host.append(canvas);try{const im=await image(sheets[key][Math.floor(index/64)]);if(version!==token)return;drawTile(canvas,im,index);
if(key==='targets'&&$('display').value!=='rgb'){
 const gradient=await image(s.sheets.sobel[Math.floor(index/64)]);if(version!==token)return;
 const temp=document.createElement('canvas');temp.width=temp.height=128;drawTile(temp,gradient,index);
 const g=temp.getContext('2d').getImageData(0,0,128,128).data,ctx=canvas.getContext('2d'),pixels=ctx.getImageData(0,0,128,128),gain=Number($('gain').value),overlay=$('display').value==='overlay';
 for(let j=0;j<g.length;j+=4){const v=Math.min(1,g[j]/255*gain),color=heat(v),alpha=overlay?Math.min(.85,v*.85):1;for(let k=0;k<3;k++)pixels.data[j+k]=pixels.data[j+k]*(1-alpha)+color[k]*alpha;pixels.data[j+3]=255}ctx.putImageData(pixels,0,0)
}}catch(e){$('error').textContent=e.message}}

function rebuild(){const[d,s]=state();const previous=firstBuild?(query.get('scheme')||'original'):$('palette').value;firstBuild=false;options($('palette'),[['original','Original dataset RGB'],...Object.entries(d.palettes||{})],previous);if(!$('palette').value)$('palette').value='original';const depths=[...new Set(s.records.map(r=>r.zoom_depth).filter(v=>v!==undefined))].sort((a,b)=>a-b),modes=[...new Set(s.records.map(r=>r.mode).filter(Boolean))].sort();options($('depth'),[['all','All zoom depths'],...depths.map(d=>[String(d),(4**d).toLocaleString()+'×'])]);options($('mode'),[['all','All evidence modes'],...modes.map(m=>[m,m])]);$('manifest').textContent=JSON.stringify({...d.manifest,quality:d.quality},null,2);$('notice').textContent=d.manifest.split_policy||'Legacy dataset. Check stored camera metadata and manifest before comparing validation results.';const perf=d.manifest.performance;if(perf)$('notice').textContent+=' Generator: '+perf.total_seconds.toFixed(1)+' s total; '+perf.search_seconds.toFixed(1)+' s search; '+perf.calibration_seconds.toFixed(1)+' s calibration; '+perf.render_seconds.toFixed(1)+' s render/check/CPU transfer.'+(perf.fp64_baseline_render_cpu_seconds?' Total includes an extra FP64 benchmark pass.':'');filter()}
function changeDataset(){const d=datasets.find(x=>x.name===$('dataset').value);options($('split'),Object.keys(d.splits).map(s=>[s,s]),'train');rebuild()}
function filter(){const[d,s]=state();filtered=s.records.map((r,i)=>i).filter(i=>($('depth').value==='all'||String(s.records[i].zoom_depth)===$('depth').value)&&($('mode').value==='all'||s.records[i].mode===$('mode').value));if($('sort').value!=='index')filtered.sort((a,b)=>($('sort').value==='high'?-1:1)*(s.records[a].sobel_score-s.records[b].sobel_score)||a-b);selected=filtered[0]??0;page=0;$('stats').replaceChildren(...[s.count.toLocaleString()+' samples',s.shape.join(' × '),'Showing '+filtered.length.toLocaleString(),d.name.includes('fast')?'Calibrated renderer':'Stored dataset'].map(t=>{const e=document.createElement('span');e.className='pill';e.textContent=t;return e}));render()}
function render(){token++;const version=token,[d,s]=state();const available=Object.keys(d.palettes||{}).length>0,rgb=$('display').value==='rgb';$('palette').disabled=!available||!rgb;$('compare').disabled=!available||!rgb;$('compare').setAttribute('aria-pressed',String(comparing));$('paletteNote').textContent=!available?'Coloring previews have not been exported for this dataset.':!rgb?'Sobel uses the original dataset colors.':'Same location and samples; both input and target use the selected coloring.';const comparison=$('comparison');comparison.replaceChildren();comparison.hidden=!(comparing&&available&&rgb&&filtered.length);if(!comparison.hidden){for(const[name,label]of[['original','Original'],...Object.entries(d.palettes)]){const figure=document.createElement('figure'),caption=document.createElement('figcaption'),host=document.createElement('div');caption.textContent=label;figure.append(caption,host);comparison.append(figure);tile(host,'targets',selected,version,name)}}$('error').textContent='';$('index').max=s.count-1;$('index').value=selected;$('index').disabled=!filtered.length;const pos=filtered.indexOf(selected);$('position').textContent=filtered.length?`${pos+1} / ${filtered.length} matching samples`:'No matching samples';$('previous').disabled=pos<=0;$('next').disabled=pos<0||pos>=filtered.length-1;$('random').disabled=!filtered.length;$('record').textContent=JSON.stringify({index:selected,...s.records[selected]},null,2);$('targetLabel').textContent=($('display').value==='rgb'?'Target · mathematical render':$('display').value==='sobel'?'Target · Sobel gradient':'Target · Sobel overlay')+(filtered.length?' · score '+s.records[selected].sobel_score.toFixed(4):'');for(const[k,id]of[['inputs','input'],['targets','target'],['masks','mask']]){if(filtered.length)tile($(id),k,selected,version);else $(id).replaceChildren()}
$('gallery').replaceChildren();const ids=filtered.slice(page*24,page*24+24);for(const i of ids){const b=document.createElement('button');b.className='thumb'+(i===selected?' active':'');b.setAttribute('aria-label','Select sample '+i);const c=document.createElement('div'),label=document.createElement('span'),r=s.records[i];label.textContent='#'+i+(r.magnification?' · '+r.magnification.toLocaleString()+'×':'')+(r.mode?' · '+r.mode:'')+' · S '+r.sobel_score.toFixed(4);label.title='Mean Sobel gradient: '+r.sobel_score.toFixed(6);b.append(c,label);b.onclick=()=>{selected=i;render()};$('gallery').append(b);tile(c,'targets',i,version)}$('pageLabel').textContent=(page+1)+' / '+Math.max(1,Math.ceil(filtered.length/24));$('pagePrev').disabled=page===0;$('pageNext').disabled=(page+1)*24>=filtered.length}
$('palette').onchange=render;$('compare').onclick=()=>{comparing=!comparing;render()};$('dataset').onchange=changeDataset;$('split').onchange=rebuild;$('depth').onchange=filter;$('mode').onchange=filter;$('sort').onchange=filter;$('display').onchange=render;$('gain').onchange=render;$('random').onclick=()=>{selected=filtered[Math.floor(Math.random()*filtered.length)];page=Math.floor(filtered.indexOf(selected)/24);render()};function move(d){const p=filtered.indexOf(selected)+d;if(p>=0&&p<filtered.length){selected=filtered[p];page=Math.floor(p/24);render()}}$('previous').onclick=()=>move(-1);$('next').onclick=()=>move(1);$('index').onchange=()=>{const n=Number($('index').value);if(filtered.includes(n)){selected=n;page=Math.floor(filtered.indexOf(n)/24);render()}else{$('index').value=selected;$('error').textContent='That index is outside the current filters.'}};$('pagePrev').onclick=()=>{page--;render()};$('pageNext').onclick=()=>{page++;render()};document.addEventListener('keydown',e=>{if(['INPUT','SELECT'].includes(document.activeElement.tagName))return;if(e.key==='ArrowRight')move(1);if(e.key==='ArrowLeft')move(-1)});const query=new URLSearchParams(location.search);options($('dataset'),datasets.map(d=>[d.name,d.name]),(datasets.find(d=>d.name===query.get('dataset'))||datasets.find(d=>d.name==='zoom4x-10k-128')||datasets.find(d=>d.name==='zoom4x-deep-128')||datasets.find(d=>d.name==='zoom4x-fast-128'))?.name||datasets[0].name);for(const[id,param]of[['display','view'],['sort','sort']]){const v=query.get(param);if([...$(id).options].some(o=>o.value===v))$(id).value=v}changeDataset();
</script></html>'''


def export_viewer(data, output):
    """Export all available splits; PNG sprite sheets preserve pixel previews."""
    assets = data / 'dataset-viewer'
    assets.mkdir(exist_ok=True)
    catalog = []
    torch.set_num_threads(4)
    for directory in sorted(data.iterdir()):
        if not (directory / 'train.pt').exists():
            continue
        manifest = directory / 'manifest.json'
        entry = {"name": directory.name, "manifest": json.loads(manifest.read_text()) if manifest.exists() else {}, "splits": {}}
        quality = directory / 'quality.json'
        entry['quality'] = json.loads(quality.read_text()) if quality.exists() else None
        for split in ('train', 'val'):
            path = directory / f'{split}.pt'
            if not path.exists():
                continue
            payload = torch.load(path, weights_only=True, mmap=True, map_location='cpu')
            targets = payload.get('targets', payload.get('images'))
            count = len(targets)
            records = [dict(r) for r in payload.get('records', [{} for _ in range(count)])]
            gradients = torch.cat([sobel_gradients(targets[start:start+64]) for start in range(0,count,64)])
            for record,score in zip(records,gradients.mean((1,2,3)).tolist()):
                record['sobel_score'] = score
            if len(records) != count:
                raise ValueError(f'Record count mismatch: {path}')
            info = {"count": count, "shape": list(targets.shape[1:]), "records": records, "sheets": {}}
            for key in ('targets', 'inputs', 'masks', 'sobel'):
                tensor = targets if key == 'targets' else gradients if key == 'sobel' else payload.get(key)
                if tensor is None:
                    continue
                pages = []
                for start in range(0, count, 64):
                    dest = assets / f'{directory.name}-{split}-{key}-{start//64}.png'
                    save_sheet(tensor[start:start+64], dest)
                    pages.append(Path(os.path.relpath(dest, output.parent)).as_posix())
                info['sheets'][key] = pages
            entry['splits'][split] = info
        add_palette_catalog(entry, data, output)
        catalog.append(entry)
        print('Exported',directory.name,flush=True)
    if not catalog:
        raise ValueError('No datasets found')
    write_viewer(catalog, output)
    print(f'Saved {output.resolve()} ({len(catalog)} datasets)',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('data'))
    parser.add_argument('--output',type=Path,default=Path('dataset_viewer.html'))
    parser.add_argument('--palettes', nargs='+', metavar='DATASET', help='Add palette previews for named RGB zoom datasets')
    args=parser.parse_args()
    if args.palettes:
        export_palette_previews(args.data, args.output, args.palettes)
    else:
        export_viewer(args.data,args.output)
