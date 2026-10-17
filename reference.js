// Local Mandelbrot reference: WebGPU for shallow views, float64 for deeper views.
// Matches the training renderer: 8192 iterations, 2x2 samples and tent filtering.
const W=128, S=260, LIMIT=8192;
const palette=[[0,7,80],[20,70,180],[80,170,230],[235,250,255],[255,185,45],[130,45,10],[0,7,80]];
let gpuPromise;
const shader=`
struct Camera { x:f32, y:f32, span:f32, pad:f32 }
@group(0) @binding(0) var<uniform> camera:Camera;
@group(0) @binding(1) var<storage,read_write> samples:array<vec4f>;
const colors=array<vec3f,7>(vec3f(0,7,80),vec3f(20,70,180),vec3f(80,170,230),vec3f(235,250,255),vec3f(255,185,45),vec3f(130,45,10),vec3f(0,7,80));
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) id:vec3u) {
 if(id.x>=260u || id.y>=260u){return;}
 let cx=camera.x+(f32(id.x)+0.5-130.0)*camera.span/256.0;
 let cy=camera.y+(130.0-f32(id.y)-0.5)*camera.span/256.0;
 let dx=cx-0.25;let q=dx*dx+cy*cy;
 var n=0u;var x=0.0;var y=0.0;var xx=0.0;var yy=0.0;
 if(q*(q+dx)<=0.25*cy*cy || (cx+1.0)*(cx+1.0)+cy*cy<=0.0625){n=8192u;}
 loop {
  if(n>=8192u || xx+yy>4.0){break;}
  y=2.0*x*y+cy;x=xx-yy+cx;xx=x*x;yy=y*y;n++;
 }
 var rgb=vec3f(0);
 if(xx+yy>4.0){
  for(var j=0;j<3;j++){
   if(xx+yy<1e15){y=2.0*x*y+cy;x=xx-yy+cx;xx=x*x;yy=y*y;n++;}
  }
  let escapeValue=max(0.0,f32(n+1u)-log(0.5*log(xx+yy))/log(2.0));
  let p=fract(log(1.0+escapeValue)*0.7)*6.0;let i=min(u32(p),5u);
  rgb=mix(colors[i],colors[i+1u],p-f32(i))/255.0;
 }
 samples[id.y*260u+id.x]=vec4f(rgb,1);
}`;
async function setupGPU(){
 if(!navigator.gpu)return null;
 try{
  const adapter=await navigator.gpu.requestAdapter();if(!adapter)return null;
  const device=await adapter.requestDevice();
  const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module:device.createShaderModule({code:shader}),entryPoint:'main'}});
  const camera=device.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  const size=S*S*16;
  const samples=device.createBuffer({size,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const read=device.createBuffer({size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:camera}},{binding:1,resource:{buffer:samples}}]});
  return {device,pipeline,camera,samples,read,group,size};
 }catch{return null;}
}
async function gpuSamples(camera){
 gpuPromise??=setupGPU();const g=await gpuPromise;if(!g)return null;
 g.device.queue.writeBuffer(g.camera,0,new Float32Array([camera.x,camera.y,camera.span,0]));
 const encoder=g.device.createCommandEncoder(),pass=encoder.beginComputePass();
 pass.setPipeline(g.pipeline);pass.setBindGroup(0,g.group);pass.dispatchWorkgroups(Math.ceil(S/8),Math.ceil(S/8));pass.end();
 encoder.copyBufferToBuffer(g.samples,0,g.read,0,g.size);g.device.queue.submit([encoder.finish()]);
 await g.read.mapAsync(GPUMapMode.READ);const result=new Float32Array(g.read.getMappedRange().slice(0));g.read.unmap();return result;
}
function float64Samples(camera){
 const samples=new Float32Array(S*S*4),step=camera.span/256;
 for(let py=0;py<S;py++)for(let px=0;px<S;px++){
  const cx=camera.x+(px+.5-S/2)*step,cy=camera.y+(S/2-py-.5)*step;
  const dx=cx-.25,q=dx*dx+cy*cy;
  if(q*(q+dx)<=.25*cy*cy || (cx+1)**2+cy*cy<=.0625)continue;
  let x=0,y=0,xx=0,yy=0,n=0;
  while(n<LIMIT && xx+yy<=4){y=2*x*y+cy;x=xx-yy+cx;xx=x*x;yy=y*y;n++;}
  if(xx+yy<=4)continue;
  for(let j=0;j<3;j++)if(xx+yy<1e15){y=2*x*y+cy;x=xx-yy+cx;xx=x*x;yy=y*y;n++;}
  const escapeValue=Math.max(0,n+1-Math.log(.5*Math.log(xx+yy))/Math.LN2);
  const phase=Math.log1p(escapeValue)*.7,p=(phase-Math.floor(phase))*6,i=Math.min(5,Math.floor(p));
  for(let c=0;c<3;c++)samples[(py*S+px)*4+c]=(palette[i][c]*(1-p+i)+palette[i+1][c]*(p-i))/255;
 }
 return samples;
}
function resolve(samples){
 const pixels=new Float32Array(3*W*W),weights=[.25,.75,.75,.25];
 for(let y=0;y<W;y++)for(let x=0;x<W;x++)for(let c=0;c<3;c++){
  let sum=0;
  for(let j=0;j<4;j++)for(let i=0;i<4;i++)sum+=weights[i]*weights[j]*samples[((2*y+1+j)*S+2*x+1+i)*4+c];
  pixels[c*W*W+y*W+x]=sum/2-1;
 }
 return pixels;
}
self.onmessage=async({data})=>{
 const {id,camera}=data,start=performance.now();
 try{
  if(![camera.x,camera.y,camera.span].every(Number.isFinite)||camera.span<=0)throw Error('Invalid reference camera');
  // Retain at least 32 coordinate ULPs per subpixel; orbit errors may occur earlier.
  if(camera.span<32*Number.EPSILON*Math.max(1,Math.abs(camera.x),Math.abs(camera.y))*256){postMessage({id,limited:true});return;}
  let samples=null,backend='float64 worker';
  if(camera.span>=.03*Math.max(1,Math.abs(camera.x),Math.abs(camera.y))&&!data.forceFloat64){
   try{samples=await gpuSamples(camera);if(samples)backend='WebGPU';}catch{samples=null;}
  }
  samples??=float64Samples(camera);
  const pixels=resolve(samples);
  postMessage({id,pixels,backend,seconds:(performance.now()-start)/1000},[pixels.buffer]);
 }catch(e){postMessage({id,error:e.message});}
};
