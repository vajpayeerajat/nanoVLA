"""Offline eval of a checkpoint on a split: flow-matching loss and L1 (radians) of sampled action chunks.

    uv run eval.py runs/nanovla/best.pt [--split=test] [--val_batches=100]
"""

import sys

import torch

from nanovla.config import parse_args
from nanovla.data import make_loader
from nanovla.policy import Policy
from train import evaluate


def main():
    ckpt, args = sys.argv[1], sys.argv[2:]
    split = next((a.split("=", 1)[1] for a in args if a.startswith("--split=")), "test")
    overrides = [a for a in args if not a.startswith("--split=")]
    policy = Policy.from_checkpoint(ckpt)
    cfg = policy.cfg
    for k, v in vars(parse_args(overrides)).items():
        if any(a.startswith(f"--{k}=") for a in overrides):
            setattr(cfg, k, v)
    torch.manual_seed(0)
    loader = make_loader(cfg, split, policy.normalizer, policy.tokenizer, train=False)
    metrics = evaluate(policy.model, loader, policy.normalizer, cfg, "cuda")
    print(f"{split}: " + " ".join(f"{k} {v:.4f}" for k, v in metrics.items()))


if __name__ == "__main__":
    main()
