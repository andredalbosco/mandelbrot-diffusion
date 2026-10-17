import Model from './assets/model.js';
let run,device;
function noise(seed,n){let a=seed>>>0;function uniform(){a+=0x6D2B79F5;let t=a;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return ((t^t>>>14)>>>0)/4294967296;}const z=new Float32Array(n);for(let i=0;i<n;i+=2){let r=Math.sqrt(-2*Math.log(Math.max(1e-10,uniform()))),theta=2*Math.PI*uniform();z[i]=r*Math.cos(theta);z[i+1]=r*Math.sin(theta);}return z;}
self.onmessage=async({data})=>{try{
 if(data.type==='load'){
  if(!navigator.gpu)throw Error('WebGPU unavailable. Use a WebGPU-capable browser over HTTPS or localhost.');
  const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});if(!adapter)throw Error('No WebGPU adapter available.');
  device=await adapter.requestDevice();device.lost.then(info=>postMessage({type:'error',message:'GPU connection lost: '+info.message}));
  device.addEventListener('uncapturederror',e=>postMessage({type:'error',message:e.error.message}));
  run=await Model.load(device,new URL('./assets/weights.safetensors',import.meta.url));postMessage({type:'ready',device:adapter.info?.description||adapter.info?.architecture||'WebGPU adapter'});
 }else if(data.type==='test'){
  const load=async name=>new Float32Array(await(await fetch(new URL('./assets/'+name,import.meta.url))).arrayBuffer());
  const inputs=await Promise.all([0,1,2,3].map(i=>load(`test-input-${i}.bin`)));const expected=await load('test-output.bin');
  const start=performance.now(),[actual]=await run(...inputs);let max=0,sum=0;for(let i=0;i<actual.length;i++){let d=Math.abs(actual[i]-expected[i]);max=Math.max(max,d);sum+=d;}
  postMessage({type:'test',passed:max<.003&&actual.every(Number.isFinite),maxError:max,meanError:sum/actual.length,seconds:(performance.now()-start)/1000,finite:actual.every(Number.isFinite)});
 }else if(data.type==='sample'){
  const start=performance.now(),steps=5;let state=noise(data.seed,3*128*128);const evidence=data.evidence,mask=data.mask;
  for(let i=0;i<steps;i++){
   const t=i/steps,dt=1/steps,[clean]=await run(state,new Float32Array([t]),evidence,mask),v=new Float32Array(state.length),proposal=new Float32Array(state.length);
   for(let j=0;j<state.length;j++){v[j]=(clean[j]-state[j])/Math.max(1-t,.05);proposal[j]=state[j]+dt*v[j];}
   if(i<steps-1){const [next]=await run(proposal,new Float32Array([t+dt]),evidence,mask);for(let j=0;j<state.length;j++)state[j]+=dt*.5*(v[j]+(next[j]-proposal[j])/Math.max(1-t-dt,.05));}else state=proposal;
   postMessage({type:'progress',step:i+1,steps});
  }
  if(!state.every(Number.isFinite))throw Error('Inference produced nonfinite pixels.');
  for(let j=0;j<state.length;j++)state[j]=Math.max(-1,Math.min(1,state[j]));
  postMessage({type:'result',pixels:state,seconds:(performance.now()-start)/1000},[state.buffer]);
 }
}catch(e){postMessage({type:'error',message:e.message+'\n'+(e.stack||'')});}};
