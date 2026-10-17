"""Export checkpoint EMA through tinygrad to static browser WebGPU kernels."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from tinygrad import Tensor
from tinygrad.nn.state import safe_save
from extra.export_model import export_model
import extra.export_model as exporter
import inspect

# This pinned exporter keys WGSL by shape-based names, which collide for
# different transformer kernels. Give each dispatch its own shader identifier.
source=inspect.getsource(exporter.compile_net)
source=source.replace('functions[prg.src[0].arg.function_name] = prg.src[2].arg',
    'name = prg.src[0].arg.function_name + "_dispatch_" + str(len(statements))\n    functions[name] = prg.src[2].arg.replace(prg.src[0].arg.function_name, name)')
source=source.replace('statements.append((prg.src[0].arg.function_name,', 'statements.append((name,')
exec(compile(source, '<unique-webgpu-kernels>', 'exec'), exporter.__dict__)

from model import JiT

p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,default=Path('runs/jit-10m-smooth-1m-patch32-pilot/checkpoint.pt'));p.add_argument('--check-only',action='store_true');args=p.parse_args()
torch.set_num_threads(4)
saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
c=saved['config'];model=JiT(c['model'],saved['ema'])
rng=np.random.default_rng(17)
arrays=[rng.standard_normal((1,3,128,128)).astype(np.float32),np.array([.4],np.float32),rng.uniform(-1,1,(1,3,128,128)).astype(np.float32),np.ones((1,1,128,128),np.float32)]
arrays[3][:,:,:,:32]=0
inputs=[Tensor(a).realize() for a in arrays]
actual=model.forward(*inputs).numpy()
# Import the PyTorch model without shadowing this directory's tinygrad port.
import importlib.util
spec=importlib.util.spec_from_file_location('torch_model',Path(__file__).resolve().parents[1]/'model.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
ref=module.build_model(c).eval();ref.load_state_dict(saved['ema'])
with torch.no_grad():expected=ref(torch.from_numpy(arrays[0]),torch.from_numpy(arrays[1]),torch.from_numpy(arrays[2]),torch.from_numpy(arrays[3])).numpy()
error=float(np.max(np.abs(actual-expected)));print('max error',error,flush=True)
np.testing.assert_allclose(actual,expected,atol=3e-3,rtol=3e-3)
if args.check_only:sys.exit()
code,ins,outs,state=export_model(model,'webgpu',*inputs,model_name='mandelbrot')
out=Path(__file__).parent/'assets';out.mkdir(exist_ok=True)
(out/'model.js').write_text(code);safe_save(state,str(out/'weights.safetensors'))
for i,a in enumerate(arrays):a.tofile(out/f'test-input-{i}.bin')
expected.tofile(out/'test-output.bin')
(out/'manifest.json').write_text(json.dumps(dict(step=saved['step'],parameters=saved['parameters'],max_abs_error=error,inputs=ins,outputs=outs,checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()),indent=2))
print('Exported',out,flush=True)
