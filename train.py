"""Train nanoVLA on the G1 + Dex3 dataset.

    uv run train.py                                   # defaults from nanovla/config.py
    uv run train.py --run_name=smoke --max_steps=200 --batch_size=8
"""

import dataclasses
import json
import math
import os
import random
import shutil
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer

from nanovla.config import parse_args
from nanovla.data import Normalizer, make_loader
from nanovla.model import NanoVLA


def lr_at(step, cfg):
    if step < cfg.warmup_steps:
        return (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))  # cosine down to 10%


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, normalizer, cfg, device):
    """Flow-matching loss and the L1 error (radians) of sampled action chunks on a few val batches."""
    model.eval()
    losses, l1s = [], []
    for i, batch in enumerate(loader):
        if i >= cfg.val_batches:
            break
        batch = to_device(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses.append(model(batch).item())
            pred = model.sample_actions(batch)
        err = (normalizer.unnormalize(pred, "action") - normalizer.unnormalize(batch["actions"], "action")).abs()
        l1s.append(err.mean((0, 1)).cpu())
    model.train()
    l1 = torch.stack(l1s).mean(0)  # [28], same in delta or absolute space
    groups = {"left_arm": l1[0:7], "left_hand": l1[7:14], "right_arm": l1[14:21], "right_hand": l1[21:28]}
    return {"loss": float(np.mean(losses)), "l1": l1.mean().item(), **{f"l1_{k}": v.mean().item() for k, v in groups.items()}}


def save(path, model, optimizer, step, cfg):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "config": dataclasses.asdict(cfg)}, path)


def main():
    cfg = parse_args()
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda"

    run_dir = os.path.join(cfg.out_dir, cfg.run_name)  # logs
    ckpt_dir = os.path.join(cfg.save_dir, cfg.run_name)  # checkpoints, one folder per experiment
    for d in (run_dir, ckpt_dir):
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(dataclasses.asdict(cfg), f, indent=1)
        # Policy.from_checkpoint reads the norm stats next to the checkpoint.
        shutil.copy(cfg.norm_stats, os.path.join(d, "norm_stats.json"))
    print(f"checkpoints -> {ckpt_dir}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.lm_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    normalizer = Normalizer.load(cfg.norm_stats)
    train_loader = make_loader(cfg, "train", normalizer, tokenizer, train=True, seed=cfg.seed)
    val_loader = make_loader(cfg, "val", normalizer, tokenizer, train=False, seed=cfg.seed)
    print(f"train frames: {len(train_loader.dataset)}  val frames: {len(val_loader.dataset)}")

    model = NanoVLA(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params: {n_params / 1e6:.1f}M total, {n_train / 1e6:.1f}M trainable")

    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone_parameters(), "lr_scale": cfg.backbone_lr_scale},
            {"params": model.head_parameters(), "lr_scale": 1.0},
        ],
        lr=cfg.lr,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay,
        fused=True,
    )
    step = 0
    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"]
        print(f"resumed from {cfg.resume} at step {step}")
    train_model = torch.compile(model) if cfg.compile else model

    writer = SummaryWriter(os.path.join(run_dir, "tb"))
    # Train and val loss overlaid in one chart (TensorBoard "Custom Scalars" tab) to spot overfitting.
    writer.add_custom_scalars({"loss": {"train vs val": ["Multiline", ["loss/train", "loss/val"]]}})
    best_val = float("inf")
    loss_sum, loss_n = 0.0, 0
    t0 = time.time()
    data_iter = iter(train_loader)
    while step < cfg.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        batch = to_device(batch, device)

        for g in optimizer.param_groups:
            g["lr"] = cfg.lr * g["lr_scale"] * lr_at(step, cfg)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = train_model(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        step += 1
    
        lr = optimizer.param_groups[1]["lr"]
        loss_sum, loss_n = loss_sum + loss.item(), loss_n + 1
        print(f"step {step:6d}/{cfg.max_steps}  | loss {loss.item():.4f} | grad {grad_norm.item():.2f} | lr {lr:.2e} ", end = '\r')
        if step % cfg.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            # Mean over the interval, so it is as smooth as the val loss (mean over val_batches).
            writer.add_scalar("loss/train", loss_sum / loss_n, step)
            loss_sum, loss_n = 0.0, 0
            writer.add_scalar("train/grad_norm", grad_norm.item(), step)
            writer.add_scalar("train/lr", lr, step)
            writer.add_scalar("train/steps_per_sec", cfg.log_interval / dt, step)

        if step % cfg.val_interval == 0 or step == cfg.max_steps:
            metrics = evaluate(model, val_loader, normalizer, cfg, device)
            print(f"\n Val step {step:6d}/{cfg.max_steps} | val " + " ".join(f"{k} {v:.4f}" for k, v in metrics.items()))
            for k, v in metrics.items():
                writer.add_scalar("loss/val" if k == "loss" else f"val/{k}", v, step)
            if metrics["loss"] < best_val:
                best_val = metrics["loss"]
                save(os.path.join(ckpt_dir, "best.pt"), model, optimizer, step, cfg)
            t0 = time.time()

        if step % cfg.save_interval == 0 or step == cfg.max_steps:
            save(os.path.join(ckpt_dir, f"{step}.pt"), model, optimizer, step, cfg)

    writer.close()


if __name__ == "__main__":
    main()
