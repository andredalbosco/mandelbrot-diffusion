const $=id=>document.getElementById(id);
const initialCamera={x:-.5,y:0,span:3.5};
let current,reference,seedImage,seedReference,camera={...initialCamera},history=[],condition=null,depth=0,busy=false;
let worker,ready=false,loading=null,loadResolve,loadReject,sampleResolve,sampleReject;
const referenceWorker=new Worker(new URL('./reference.js',import.meta.url),{type:'module'});
let referenceID=0;const referenceRequests=new Map();
referenceWorker.onmessage=({data})=>{const request=referenceRequests.get(data.id);if(!request)return;referenceRequests.delete(data.id);data.error?request.reject(Error(data.error)):request.resolve(data);};
referenceWorker.onerror=e=>{for(const request of referenceRequests.values())request.reject(Error(e.message));referenceRequests.clear();};
function renderReference(view){return new Promise((resolve,reject)=>{const id=++referenceID;referenceRequests.set(id,{resolve,reject});referenceWorker.postMessage({id,camera:view});});}
function draw(id,pixels){
 const image=new ImageData(128,128);
 for(let i=0;i<16384;i++){for(let c=0;c<3;c++)image.data[i*4+c]=Math.round((pixels[c*16384+i]+1)*127.5);image.data[i*4+3]=255;}
 $(id).getContext('2d').putImageData(image,0,0);
}
function show(){
 draw('output',current);
 if(reference.pixels)draw('input',reference.pixels);
 else{const ctx=$('input').getContext('2d');ctx.fillStyle='#10151d';ctx.fillRect(0,0,128,128);ctx.fillStyle='#edf1f7';ctx.font='8px system-ui';ctx.textAlign='center';ctx.fillText('Reference precision limit',64,64);}
 $('reference-status').textContent=reference.limited?'Float64 precision limit reached; reference unavailable.':`${reference.backend} · ${reference.seconds.toFixed(2)}s · 8192 iterations · 4 samples/pixel`;
 $('depth').textContent=`· depth ${depth}`;
 $('camera').textContent=`Center ${camera.x.toPrecision(15)}, ${camera.y.toPrecision(15)} · span ${camera.span.toExponential(4)}`;
 controls();
}
function controls(){
 $('load').disabled=ready||!!loading;
 $('reset').disabled=busy||!current;
 $('back').disabled=busy||!history.length;
 $('sample').disabled=busy||!condition;
 for(const id of ['input','output'])$(id).setAttribute('aria-busy',String(busy||!current));
}
function fail(error){$('status').textContent=error.message;controls();}
function rejectWorker(error){
 ready=false;loading=null;loadReject?.(error);sampleReject?.(error);
 loadResolve=loadReject=sampleResolve=sampleReject=null;controls();
}
function loadModel(){
 if(ready)return Promise.resolve();if(loading)return loading;
 if(!isSecureContext||!navigator.gpu)return Promise.reject(Error('WebGPU requires HTTPS or localhost and a supported browser.'));
 $('status').textContent='Loading model…';
 loading=new Promise((resolve,reject)=>{loadResolve=resolve;loadReject=reject;});
 worker?.terminate();worker=new Worker(new URL('./worker.js',import.meta.url),{type:'module'});
 worker.onerror=e=>rejectWorker(Error(e.message));
 worker.onmessage=({data})=>{
  if(data.type==='ready'){
   ready=true;loading=null;$('device').textContent='Local model device: '+data.device;
   $('status').textContent='Ready. Click either image to zoom.';loadResolve?.();loadResolve=loadReject=null;controls();
   if(location.search.includes('test'))worker.postMessage({type:'test'});
  }else if(data.type==='progress'){
   $('progress').max=data.steps;$('progress').value=data.step;$('status').textContent=`Generating · ${data.step}/${data.steps}`;
  }else if(data.type==='result'){
   sampleResolve?.(data);sampleResolve=sampleReject=null;
  }else if(data.type==='error')rejectWorker(Error(data.message));
  else if(data.type==='test'){window.parityResult=data;}
 };
 worker.postMessage({type:'load'});controls();return loading;
}
function crop(pixels,left,top){const patch=new Float32Array(3*1024),coverage=new Float32Array(1024);for(let y=0;y<32;y++)for(let x=0;x<32;x++){const px=left+x,py=top+y,k=y*32+x;if(px>=0&&px<128&&py>=0&&py<128){coverage[k]=1;for(let c=0;c<3;c++)patch[c*1024+k]=pixels[c*16384+py*128+px];}}
 const evidence=new Float32Array(3*16384),mask=new Float32Array(16384);for(let y=0;y<128;y++)for(let x=0;x<128;x++){const sx=Math.max(0,Math.min(31,(x+.5)/4-.5)),sy=Math.max(0,Math.min(31,(y+.5)/4-.5)),ix=Math.floor(sx),iy=Math.floor(sy),fx=sx-ix,fy=sy-iy,k=y*128+x;
 for(let dy=0;dy<2;dy++)for(let dx=0;dx<2;dx++){let w=(dx?fx:1-fx)*(dy?fy:1-fy),p=Math.min(31,iy+dy)*32+Math.min(31,ix+dx);mask[k]+=w*coverage[p];for(let c=0;c<3;c++)evidence[c*16384+k]+=w*patch[c*1024+p];}if(mask[k]>1e-6)for(let c=0;c<3;c++)evidence[c*16384+k]/=mask[k];}
 return {evidence,mask};}
async function generate(next,nextCamera,nextDepth){
 if(busy)return;
 busy=true;controls();$('progress').value=0;
 try{
  // Reference pixels never enter the model. Keep the previous pair until both finish.
  const referenceResult=renderReference(nextCamera);
  const prediction=(async()=>{
   await loadModel();
   return new Promise((resolve,reject)=>{
    sampleResolve=resolve;sampleReject=reject;
    worker.postMessage({type:'sample',...next,seed:Number($('seed').value)>>>0});
   });
  })();
  const outcomes=await Promise.allSettled([referenceResult,prediction]);
  const failure=outcomes.find(item=>item.status==='rejected');if(failure)throw failure.reason;
  const [rendered,result]=outcomes.map(item=>item.value);
  history.push({pixels:current,reference,condition,camera,depth});
  current=result.pixels;reference=rendered;condition=next;camera=nextCamera;depth=nextDepth;
  show();$('status').textContent=`Diffusion ${result.seconds.toFixed(2)}s · both views computed locally`;
 }catch(error){fail(error);}finally{busy=false;controls();}
}
function zoom(e){
 if(busy||!current)return;
 const r=e.currentTarget.getBoundingClientRect();
 const left=Math.round((e.clientX-r.left)/r.width*128-16),top=Math.round((e.clientY-r.top)/r.height*128-16);
 const nextCamera={x:camera.x+(left+16-64)*camera.span/128,y:camera.y+(64-top-16)*camera.span/128,span:camera.span/4};
 generate(crop(current,left,top),nextCamera,depth+1);
}
$('input').onclick=zoom;$('output').onclick=zoom;
$('load').onclick=()=>loadModel().catch(fail);
document.addEventListener('click',()=>{if(!ready&&!loading)loadModel().catch(fail);});
$('sample').onclick=()=>{if(condition&&!busy){$('seed').value=(Number($('seed').value)+1)>>>0;generate(condition,{...camera},depth);}};
$('back').onclick=()=>{
 if(busy||!history.length)return;
 const old=history.pop();current=old.pixels;reference=old.reference;condition=old.condition;camera=old.camera;depth=old.depth;show();
 $('status').textContent='Previous view restored.';
};
$('reset').onclick=()=>{
 if(busy||!seedImage)return;
 current=seedImage.slice();reference=seedReference;condition=null;camera={...initialCamera};history=[];depth=0;show();
 $('status').textContent='Initial view restored.';
};
try{
 $('status').textContent='Rendering initial view…';controls();
 reference=await renderReference(camera);seedReference=reference;
 seedImage=reference.pixels.slice();current=seedImage.slice();show();
 $('status').textContent='Click either image to zoom. The first click loads the model.';
}catch(error){fail(error);}
