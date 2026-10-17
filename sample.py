"""Sample an unconditional or paired zoom flow from an EMA checkpoint."""

import argparse
from pathlib import Path

import torch

from dataset import load_images, save_grid
from model import ConditionalFlow, RectifiedFlow, XPredictionFlow, build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("output/flow-samples.png"))
    parser.add_argument("--steps", type=int, help="Euler steps: 16 for conditional JiT and paired zoom; 50 for legacy unconditional")
    parser.add_argument("--solver", choices=("euler", "heun"), default="euler", help="Conditional JiT solver; Heun defaults to 50 steps")
    parser.add_argument("--input", type=Path, help="Zoom condition: .pt file containing float images [N,3,H,W] in [0,1]")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("Count must be positive")
    if args.output.exists() or args.output.with_suffix(".pt").exists():
        parser.error("Output already exists; choose a new path")
    saved = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    config = saved["config"]
    model = build_model(config).to(args.device)
    model.load_state_dict(saved["ema"])
    if config['formulation'] == 'conditional-jit-x-pred':
        if args.input:
            payload = torch.load(args.input, weights_only=True, mmap=True, map_location='cpu')
            evidence = payload.get('inputs', payload.get('images'))
            if evidence is None:
                parser.error('Condition file requires inputs or images')
            evidence = evidence[:args.count].to(args.device)
            mask = payload.get('masks', torch.ones(len(evidence), 1, config['size'], config['size']))[:len(evidence)].to(args.device)
            if evidence.shape[1:] != (3, config['size'], config['size']) or mask.shape != (len(evidence), 1, config['size'], config['size']):
                parser.error('Condition image or mask shape mismatch')
            if any(not x.isfinite().all() or x.min()<0 or x.max()>1 for x in (evidence,mask)):
                parser.error('Conditions and masks must be finite and in [0,1]')
            evidence = evidence*2-1
        else:
            evidence = torch.zeros(args.count, 3, config['size'], config['size'], device=args.device)
            mask = torch.zeros_like(evidence[:, :1])
        args.steps = (50 if args.solver == "heun" else 16) if args.steps is None else args.steps
        with torch.autocast('cuda', dtype=torch.bfloat16):
            images = ConditionalFlow(**config.get('flow', {})).sample(model, evidence, mask, steps=args.steps, seed=args.seed, solver=args.solver)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_grid(images,args.output)
        torch.save({'images':images.cpu(),'seed':args.seed,'checkpoint_step':saved['step'],
                    'integration_steps':args.steps,'solver':args.solver,'formulation':config['formulation']},args.output.with_suffix('.pt'))
        print(f'Saved {args.output}: {args.steps} step(s), clean-image prediction, conditional={args.input is not None}')
        return
    if args.solver != "euler":
        parser.error("--solver heun is supported for conditional JiT checkpoints")
    paired = config["formulation"] == "rectified-flow-4x-zoom"
    if args.steps is None:
        args.steps = 16 if paired else 50
    if paired:
        if args.input is None:
            parser.error("Zoom checkpoints require --input")
        condition = load_images(args.input).to(args.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            images = RectifiedFlow().sample(model, condition, steps=args.steps)
    else:
        if args.input is not None:
            parser.error("--input requires a zoom checkpoint")
        diffusion = XPredictionFlow(config["minimum_time"], config["time_shift"])
        images = diffusion.sample(model, args.count, config["size"], steps=args.steps, seed=args.seed, channels=model.channels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_grid(images, args.output)
    torch.save({"images": images.cpu(), "seed": None if paired else args.seed, "checkpoint_step": saved["step"],
                "integration_steps": args.steps, "formulation": config["formulation"]}, args.output.with_suffix(".pt"))
    direction = "0 to 1" if paired else "1 to 0"
    print(f"Saved {args.output} from optimizer step {saved['step']}; integrated {args.steps} Euler steps from t={direction}")


if __name__ == "__main__":
    main()
