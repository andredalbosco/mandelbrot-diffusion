"""Sample an unconditional or paired zoom flow from an EMA checkpoint."""

import argparse
from pathlib import Path

import torch

from train import save_grid
from model import ConditionalFlow, build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("output/flow-samples.png"))
    parser.add_argument("--steps", type=int, help="Number of integration steps")
    parser.add_argument("--solver", choices=("euler", "heun"), default="heun", help="Heun defaults to 50 steps")
    parser.add_argument("--input", type=Path, help="Prepared zoom shard (.pt); uses its evidence and coverage")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("Count must be positive")
    if args.output.exists() or args.output.with_suffix(".pt").exists():
        parser.error("Output already exists; choose a new path")
    saved = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    config = saved["config"]
    model = build_model(config).to(args.device)
    model.load_state_dict(saved["ema"])
    if config["formulation"] != "conditional-jit-x-pred":
        parser.error("This branch supports conditional JiT checkpoints only")
    if config['formulation'] == 'conditional-jit-x-pred':
        if args.input:
            from train import ZoomDataset
            dataset = ZoomDataset(args.input)
            _, evidence, mask = dataset.batch(list(range(min(args.count, len(dataset)))), args.device)
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


if __name__ == "__main__":
    main()
