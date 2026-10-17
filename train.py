"""Train pixel-space image flows on Mandelbrot images or paired 4x zoom views."""

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import time

import torch

from dataset import load_images, load_zoom_pairs
from model import RectifiedFlow, XPredictionFlow, build_model


def batch_loss(model, diffusion, batch, generator=None):
    if isinstance(diffusion, RectifiedFlow):
        t = torch.rand(len(batch), device=batch.device, generator=generator)
        return diffusion.loss(model, batch[:, 1], batch[:, 0], t)
    t = diffusion.times(len(batch), batch.device, generator)
    noise = torch.randn(batch.shape, device=batch.device, generator=generator)
    return diffusion.loss(model, batch, t, noise)


@torch.no_grad()
def validation_loss(model, diffusion, images, batch_size, seed):
    model.eval()
    generator = torch.Generator(device=images.device).manual_seed(seed)
    total = 0.0
    for clean in images.split(batch_size):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=isinstance(diffusion, RectifiedFlow)):
            total += batch_loss(model, diffusion, clean, generator).item() * len(clean)
    model.train()
    return total / len(images)


def train_masked(config, resume=None, max_seconds=None):
    """Train one clean-image JiT on full, partial, and absent evidence."""
    from dataset import ZoomDataset, ShardedZoomDataset, ShardBatches, save_grid
    from model import ConditionalFlow

    torch.set_num_threads(4)
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(config['device'])
    root = Path(config['data'])
    sharded = (root/'manifest.json').exists()
    stream = None
    if sharded:
        train = ShardedZoomDataset(root, 'train')
        val = ShardedZoomDataset(root, 'val')
        # Manifest includes generator-recorded per-file SHA256 values. Do not
        # rescan 460 GB at every startup; validate shard contents as they load.
        hashes = {'manifest': hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()}
    else:
        if config.get('unconditional_fraction') is not None:
            raise ValueError('Training mixture selection currently requires a sharded dataset')
        train = ZoomDataset(root/'train.pt')
        val = ZoomDataset(root/'val.pt')
        if train.targets.shape[1:] != (3, config['size'], config['size']) or val.targets.shape[1:] != train.targets.shape[1:]:
            raise ValueError('Dataset shape disagrees with config')
        hashes = {}
        for split in ('train', 'val'):
            with (root/f'{split}.pt').open('rb') as file:
                hashes[split] = hashlib.file_digest(file, 'sha256').hexdigest()
    if config['architecture'] != 'jit' or config['model'].get('mask_channels') != 1:
        raise ValueError('Conditional flow requires mask-aware JiT')
    model = build_model(config).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    flow = ConditionalFlow(**config.get('flow', {}))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config.get('weight_decay', .01),
                                  betas=tuple(config.get('betas', (.9, .999))), fused=True)
    start = 0
    if resume:
        saved = torch.load(resume, map_location=device, weights_only=True)
        for key in ('architecture','model','size','formulation','learning_rate','ema_decay','batch_size','seed','flow','warmup_steps','compile','compile_mode','betas','weight_decay','unconditional_fraction'):
            if saved['config'].get(key) != config.get(key):
                raise ValueError(f'Resume config mismatch: {key}')
        if saved['dataset_sha256'] != hashes:
            raise ValueError('Resume dataset differs from checkpoint')
        model.load_state_dict(saved['model']); ema.load_state_dict(saved['ema'])
        optimizer.load_state_dict(saved['optimizer'])
        start = saved['step']
        torch.set_rng_state(saved['cpu_rng'].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in saved['cuda_rng']])
    if sharded:
        stream = ShardBatches(root, config['batch_size'], config['seed'], device,
                              saved.get('sampler') if resume else None,
                              unconditional_fraction=config.get('unconditional_fraction'),
                              select_rows=config.get('select_rows', False),
                              staging_rows=config.get('staging_rows', 0))
        if resume and 'sampler' not in saved:
            stream.close()
            raise ValueError('Sharded resume requires sampler state')
    def training_loss(clean, evidence, mask):
        return flow.loss(model, clean, evidence, mask)
    if config.get('compile', False):
        training_loss = torch.compile(training_loss, fullgraph=True, mode=config.get('compile_mode', 'default'))
    try:
        if start >= config['steps']:
            raise ValueError('Total steps must exceed checkpoint step')
        output = Path(config['output']); output.mkdir(parents=True, exist_ok=True)
        (output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
        parameters = sum(p.numel() for p in model.parameters())
        # Fixed evaluation selection, optionally balanced over both mode and depth.
        modes = ('interior', 'edge', 'corner', 'unconditional')
        if config.get('stratify_depth', False):
            from scripts.evaluate_training import select_views
            validation_ids = select_views(val.records, per_mode=64, seed=config['seed']+1)
            preview_ids = []
            for mode in modes:
                chosen = [i for i in validation_ids if val.records[i]['mode'] == mode]
                preview_ids.extend(chosen[j] for j in (0, 16, 32, 48))
        else:
            rng = torch.Generator().manual_seed(config['seed']+1)
            validation_ids, preview_ids = [], []
            for mode in modes:
                ids = [i for i,r in enumerate(val.records) if r['mode']==mode]
                order = torch.randperm(len(ids), generator=rng).tolist()
                chosen = [ids[i] for i in order[:64]]
                validation_ids.extend(chosen); preview_ids.extend(chosen[:4])
        (output/'validation-selection.json').write_text(json.dumps({
            'ids': validation_ids, 'records': [val.records[i] for i in validation_ids]}, indent=2)+'\n')
        if not validation_ids:
            raise ValueError('No validation samples in supported modes')

        # Reuse held-out CUDA tensors instead of repeating sparse disk reads.
        validation_batch = val.batch(validation_ids, device)
        preview_batch = val.batch(preview_ids, device)
        if validation_batch[0].shape[1:] != (3, config['size'], config['size']):
            raise ValueError('Dataset shape disagrees with config')

        @torch.no_grad()
        def evaluate(network):
            was_training = network.training; network.eval()
            generator = torch.Generator(device=device).manual_seed(config['seed']+2)
            total = 0.
            try:
                for offset in range(0,len(validation_ids),config['batch_size']):
                    ids=validation_ids[offset:offset+config['batch_size']]
                    clean,evidence,mask=(v[offset:offset+len(ids)] for v in validation_batch)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        total += flow.loss(network,clean,evidence,mask,generator)*len(ids)
                return (total/len(validation_ids)).item()
            finally:
                network.train(was_training)

        @torch.no_grad()
        def preview(step):
            clean,evidence,mask=preview_batch
            with torch.autocast('cuda',dtype=torch.bfloat16):
                one=flow.sample(ema,evidence,mask,steps=1,seed=config['seed']+3)
                multi=flow.sample(ema,evidence,mask,steps=16,seed=config['seed']+3)
            target=(clean+1)/2
            panels=torch.stack([(evidence[i]+1)/2 if col==0 else mask[i].expand(3,-1,-1) if col==1
                                else one[i] if col==2 else multi[i] if col==3 else target[i]
                                for i in range(len(preview_ids)) for col in range(5)])
            save_grid(panels,output/f'preview-{step:06d}.png',columns=5)
            latest = output/'preview-latest.tmp'
            latest.write_bytes((output/f'preview-{step:06d}.png').read_bytes())
            latest.replace(output/'preview-latest.png')
            scores={}
            for i,mode in enumerate(modes):
                begin=i*4;end=begin+4
                if end<=len(target):
                    scores[mode]={'one_step_mse':(one[begin:end]-target[begin:end]).square().mean().item(),
                                  'sixteen_step_mse':(multi[begin:end]-target[begin:end]).square().mean().item()}
            (output/f'preview-{step:06d}.json').write_text(json.dumps({'columns':['input','mask','one step','16 steps','target'],
                     'validation_ids':preview_ids,'scores':scores,'note':'Fixed examples; pixel MSE is not a distribution-quality metric for unconditional generation.'},indent=2)+'\n')
            return scores

        startup_val=evaluate(model)
        initial_val=saved.get('initial_val_mse', startup_val) if resume else startup_val
        print(f'architecture=jit; parameters={parameters:,}; train={len(train)}; val={len(val)}; batch={config["batch_size"]}; startup_val={startup_val:.6f}; start_step={start}',flush=True)
        preview(start)
        started=time.perf_counter()
        print('TRAINING_STARTED',time.time(),flush=True)
        log=output/'loss.csv';existing=log.exists()
        with log.open('a',newline='') as file:
            writer=csv.writer(file)
            if not existing:writer.writerow(('step','samples_seen','train_mse','val_mse','ema_val_mse','elapsed_seconds','steps_per_second','gpu_allocated_gib'))
            for step in range(start+1,config['steps']+1):
                if config.get('compile', False):
                    torch.compiler.cudagraph_mark_step_begin()
                if stream is None:
                    indices=torch.randint(len(train),(config['batch_size'],))
                    clean,evidence,mask=train.batch(indices,device)
                else:
                    clean,evidence,mask=next(stream)
                warmup=config.get('warmup_steps',500)
                for group in optimizer.param_groups:
                    group['lr']=config['learning_rate']*min(1.,step/max(1,warmup))
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss=training_loss(clean,evidence,mask)
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=True)
                # One host synchronization; reject invalid loss or gradients before
                # either optimizer or EMA can mutate model state.
                if not (torch.isfinite(loss.detach()) & torch.isfinite(norm)):
                    raise RuntimeError(f'Nonfinite loss or gradients at {step}')
                optimizer.step()
                with torch.no_grad():
                    torch._foreach_lerp_(list(ema.parameters()),list(model.parameters()),1-config['ema_decay'])
                stop_for_time = max_seconds is not None and time.perf_counter()-started >= max_seconds
                final_step = step == config['steps'] or stop_for_time
                val_loss=ema_loss=''
                if step%config['eval_every']==0 or final_step:
                    val_loss=evaluate(model);ema_loss=evaluate(ema)
                    preview(step)
                    print(f'step={step} train={loss.item():.6f} val={val_loss:.6f} ema_val={ema_loss:.6f}',flush=True)
                if step in config.get('quality_steps', []):
                    from scripts.evaluate_training import quality_report
                    gallery = Path(config['quality_output'])
                    quality_report(ema, flow, val, gallery, 'validation', step, seed=config['seed']+7)
                    training_views = ZoomDataset(root/train.files[0]['path']) if sharded else train
                    quality_report(ema, flow, training_views, gallery, 'training', step, seed=config['seed']+7)
                    del training_views
                elapsed=time.perf_counter()-started
                rate=(step-start)/elapsed
                memory=torch.cuda.memory_allocated(device)/2**30
                if step%config.get('log_every', 1)==0 or step==1 or val_loss!='':
                    writer.writerow((step,step*config['batch_size'],loss.item(),val_loss,ema_loss,elapsed,rate,memory));file.flush()
                if step%100==0 or step==1 or final_step:
                    status=dict(step=step,total_steps=config['steps'],samples_seen=step*config['batch_size'],
                                train_mse=loss.item(),elapsed_seconds=elapsed,steps_per_second=rate,gpu_allocated_gib=memory,
                                initial_val_mse=initial_val,updated_unix=time.time(),
                                images_per_second=rate*config['batch_size'],
                                peak_gpu_gib=torch.cuda.max_memory_allocated(device)/2**30,
                                shard_wait_seconds=stream.load_seconds if stream else 0)
                    tmp=output/'status.tmp';tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(output/'status.json')
                    print(f'progress step={step}/{config["steps"]} samples={step*config["batch_size"]} steps/s={rate:.1f} gpu={memory:.2f}GiB',flush=True)
                if step%config['checkpoint_every']==0 or final_step:
                    checkpoint={'config':config,'step':step,'model':model.state_dict(),'ema':ema.state_dict(),
                                'optimizer':optimizer.state_dict(),'cpu_rng':torch.get_rng_state(),
                                'cuda_rng':torch.cuda.get_rng_state_all(),'dataset_sha256':hashes,
                                'parameters':parameters,'initial_val_mse':initial_val}
                    if stream is not None:
                        checkpoint['sampler'] = stream.state_dict()
                    tmp=output/'checkpoint.tmp';torch.save(checkpoint,tmp);tmp.replace(output/'checkpoint.pt')
                    if step in config.get('quality_steps', []):
                        import shutil
                        shutil.copyfile(output/'checkpoint.pt', output/f'checkpoint-{step:06d}.pt')
                if stop_for_time:
                    print('Training time limit reached; final evaluation and checkpoint saved.', flush=True)
                    break
        print(f'COMPLETE {output} at step {step}',flush=True)

    finally:
        if stream is not None:
            stream.close()


def parse_training_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Defaults to saved configuration when resuming")
    parser.add_argument("--output", type=Path)
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--steps", type=int, help="Total optimizer steps, including resumed steps")
    budget.add_argument("--additional-steps", type=int, help="Extra updates after the checkpoint")
    parser.add_argument("--resume", type=Path, help="Checkpoint file or run directory")
    parser.add_argument("--max-seconds", type=float, help="Limit this invocation training loop; save before stopping (excludes startup and final save)")
    args = parser.parse_args(argv)
    if args.max_seconds is not None and not 0 < args.max_seconds < float("inf"):
        parser.error("--max-seconds must be finite and positive")
    if args.additional_steps is not None and (args.resume is None or args.additional_steps < 1):
        parser.error("--additional-steps requires --resume and a positive value")
    saved = None
    if args.resume is not None:
        if args.resume.is_dir():
            args.resume = args.resume / "checkpoint.pt"
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
    if args.config is not None:
        config = json.loads(args.config.read_text())
    elif saved is not None:
        config = copy.deepcopy(saved["config"])
        config["output"] = str(args.resume.parent)
    else:
        config = json.loads(Path("configs/flow64.json").read_text())
    if args.output is not None:
        config["output"] = str(args.output)
    if args.steps is not None:
        config["steps"] = args.steps
    if args.additional_steps is not None:
        config["steps"] = saved["step"] + args.additional_steps
    if saved is not None and config["steps"] <= saved["step"]:
        parser.error("Run has reached its budget; use --additional-steps N or a larger --steps total")
    if min(config["steps"], config["batch_size"], config["eval_every"], config["checkpoint_every"]) < 1:
        parser.error("Steps, batch size and intervals must be positive")
    output = Path(config["output"])
    if output.exists() and args.resume is None:
        parser.error("Output already exists; resume or choose a new directory")
    if args.max_seconds is not None and config["formulation"] != "conditional-jit-x-pred":
        parser.error("--max-seconds currently supports conditional JiT training")
    return args, config


def main():
    args, config = parse_training_args()
    output = Path(config["output"])
    if config["formulation"] == "conditional-jit-x-pred":
        train_masked(config,args.resume,max_seconds=args.max_seconds)
        return
    paired = config["formulation"] == "rectified-flow-4x-zoom"
    if not paired and config["formulation"] != "modality-forcing-x-pred-single-image":
        raise ValueError("Unsupported formulation")
    device = torch.device(config["device"])
    torch.manual_seed(config["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
    train_path = Path(config["data"]) / "train.pt"
    val_path = Path(config["data"]) / "val.pt"
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
              for name, path in (("train", train_path), ("val", val_path))}
    loader = load_zoom_pairs if paired else load_images
    train = loader(train_path).to(device)
    val = loader(val_path).to(device)
    expected = (config["model"].get("channels", 1), config["size"], config["size"])
    if paired:
        expected = (2, *expected)
    if train.shape[1:] != expected or val.shape[1:] != expected:
        raise ValueError("Dataset shape disagrees with config")
    if paired and config["architecture"] == "dit" and config["model"].get("condition_channels") != 3:
        raise ValueError("Zoom DiT requires three conditioning channels")
    if not paired and config["architecture"] != "dit":
        raise ValueError("Unconditional training requires DiT")
    model = build_model(config).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    diffusion = RectifiedFlow() if paired else XPredictionFlow(config["minimum_time"], config["time_shift"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0)
    start = 0
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=True)
        if any(saved["config"].get(key) != config[key] for key in ("architecture", "size", "model")):
            raise ValueError("Resume architecture mismatch; start a new run when switching models")
        for key in ("size", "formulation", "minimum_time", "time_shift", "learning_rate", "ema_decay", "batch_size", "seed"):
            if saved["config"].get(key) != config.get(key):
                raise ValueError(f"Resume config mismatch: {key}")
        if saved["dataset_sha256"] != hashes:
            raise ValueError("Resume dataset differs from checkpoint")
        model.load_state_dict(saved["model"])
        ema.load_state_dict(saved["ema"])
        optimizer.load_state_dict(saved["optimizer"])
        start = saved["step"]
        torch.set_rng_state(saved["cpu_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng"]])
    if config["steps"] <= start:
        parser.error("Total steps must exceed checkpoint step")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"architecture={config['architecture']}; device={device}; parameters={parameter_count:,}; train={len(train)}; val={len(val)}", flush=True)
    initial_val = validation_loss(model, diffusion, val, config["batch_size"], config["seed"] + 1)
    print(f"initial fixed-randomness validation MSE={initial_val:.6f}", flush=True)
    started = time.perf_counter()
    log_path = output / "loss.csv"
    has_log = log_path.exists()
    with log_path.open("a", newline="") as logfile:
        writer = csv.writer(logfile)
        if not has_log:
            writer.writerow(("step", "train_mse", "val_mse", "ema_val_mse", "elapsed_seconds"))
        for step in range(start + 1, config["steps"] + 1):
            indices = torch.randint(len(train), (config["batch_size"],), device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=paired):
                loss = batch_loss(model, diffusion, train[indices])
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"Nonfinite gradients at step {step}")
            optimizer.step()
            with torch.no_grad():
                for target, source in zip(ema.parameters(), model.parameters()):
                    target.lerp_(source, 1 - config["ema_decay"])
            val_loss, ema_loss = "", ""
            if step % config["eval_every"] == 0 or step == config["steps"]:
                val_loss = validation_loss(model, diffusion, val, config["batch_size"], config["seed"] + 1)
                ema_loss = validation_loss(ema, diffusion, val, config["batch_size"], config["seed"] + 1)
                ema.eval()
                print(f"step={step} train={loss.item():.6f} val={val_loss:.6f} ema_val={ema_loss:.6f}", flush=True)
            writer.writerow((step, loss.item(), val_loss, ema_loss, time.perf_counter() - started))
            logfile.flush()
            if step % config["checkpoint_every"] == 0 or step == config["steps"]:
                checkpoint = {"config": config, "step": step, "model": model.state_dict(),
                              "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
                              "cpu_rng": torch.get_rng_state(),
                              "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                              "dataset_sha256": hashes, "parameters": parameter_count,
                              "initial_val_mse": initial_val}
                temporary = output / "checkpoint.tmp"
                torch.save(checkpoint, temporary)
                temporary.replace(output / "checkpoint.pt")
    print(f"Saved {output / 'checkpoint.pt'}; this run performed {config['steps'] - start} optimizer steps", flush=True)


if __name__ == "__main__":
    main()
