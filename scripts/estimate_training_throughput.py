"""CPU-only arithmetic estimate for the current JiT; no torch/CUDA imports.

Counts multiply-add as two FLOPs. This is an optimistic matrix-arithmetic bound,
not a prediction of achievable throughput. Elementwise operations, optimizer,
launches, cache behavior and actual kernel padding need separate profiling.
"""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
c = json.loads((root/'configs/zoom4x-smooth-1m-jit-1m-fast.json').read_text())
m = c['model']
n = (c['size']//m['patch_size'])**2
d, layers, r, p = m['hidden_size'], m['depth'], m['bottleneck_dim'], m['patch_size']
h = int(int(d*m['mlp_ratio'])*2/3)
input_channels = 2*m['channels']+m['mask_channels']
parts = dict(
    patch_embedding=2*n*input_channels*p*p*r+2*n*r*d,
    qkv_and_output=layers*8*n*d*d,
    attention_qk_av=layers*4*n*n*d,
    swiglu_linears=layers*6*n*d*h,
    time_and_modulation=2*256*d+2*d*d+layers*12*d*d+4*d*d,
    image_projection=2*n*d*m['channels']*p*p,
)
forward = sum(parts.values())
# First patch/time projections do not need input gradients.
training = 3*forward-2*n*input_channels*p*p*r-2*256*d
recompute = layers*2*n*n*d
peak = 165.2e12  # NVIDIA Ada whitepaper: dense BF16 with FP32 accumulation.
bytes_per_row = (2*m['channels']+m['mask_channels'])*c['size']**2*4
result = dict(
    architecture=dict(tokens=n, width=d, depth=layers, mlp_width=h, batch=c['batch_size']),
    forward_flops=parts, forward_total=forward,
    training_matrix_flops=training,
    training_with_one_attention_score_recompute=training+recompute,
    dense_bf16_fp32_accum_peak_flops=peak,
    arithmetic_ceiling_training_images_per_second=peak/training,
    arithmetic_ceiling_with_attention_recompute=peak/(training+recompute),
    arithmetic_ceiling_forward_images_per_second=peak/forward,
    batch64_arithmetic_floor_ms=64*training/peak*1000,
    stored_bytes_per_row=bytes_per_row,
    stored_bytes_per_selected_training_row=bytes_per_row*9000/7500,
    gpu_memory_bandwidth_bytes_per_second=1008e9,
    ridge_flops_per_byte=peak/1008e9,
    gemm_forward_intensity_no_interkernel_cache_bf16={
        name: k*out/(k+out) for name,k,out in (
            ('qkv', d,3*d), ('attention_output',d,d),
            ('swiglu_up',d,2*h), ('swiglu_down',h,d))},
    sources=['https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf'],
)
result['measured'] = {}
for name in ('compiled-64', 'compiled-128', 'prefetch-gpu-64'):
    path=root/f'output/training-speed/{name}.json'
    if path.exists():
        measurement=json.loads(path.read_text())
        rate=measurement['images_per_second']
        result['measured'][name]=dict(images_per_second=rate,
                                     useful_model_tflops=training*rate/1e12,
                                     model_flop_utilization=training*rate/peak)
status_path=root/'runs/jit-1m-smooth-1m-fast/status.json'
if status_path.exists():
    status=json.loads(status_path.read_text()); rate=status['images_per_second']
    result['measured']['baseline_last_logged']=dict(step=status['step'], images_per_second=rate,
        useful_model_tflops=training*rate/1e12, model_flop_utilization=training*rate/peak,
        shard_wait_fraction=status['shard_wait_seconds']/status['elapsed_seconds'])
print(json.dumps(result, indent=2))
