"""Inference wrapper: raw robot observation -> absolute 28-dim joint targets for the next action chunk.

    policy = Policy.from_checkpoint("runs/nanovla/best.pt")
    actions = policy.infer({
        "ego_view": hwc_uint8, "ego_left": hwc_uint8, "ego_right": hwc_uint8,
        "state": np.float32[28],  # or the 43-dim whole-body vector
        "prompt": "Pick up the pink octopus and place it inside the orange basket.",
    })  # -> np.float32[action_horizon, 28]
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from nanovla.config import Config
from nanovla.data import ARM_MASK, Normalizer
from nanovla.model import NanoVLA


def resize_with_pad(img: torch.Tensor, size: int) -> torch.Tensor:
    """[3, H, W] uint8 -> [3, size, size], aspect kept and padded with black (matches prepare_data.py)."""
    h, w = img.shape[-2:]
    scale = size / max(h, w)
    nh, nw = round(h * scale), round(w * scale)
    img = F.interpolate(img[None].float(), size=(nh, nw), mode="bilinear", antialias=True, align_corners=False)[0]
    top, left = (size - nh) // 2, (size - nw) // 2
    img = F.pad(img, (left, size - nw - left, top, size - nh - top))
    return img.round().clamp(0, 255).to(torch.uint8)


class Policy:
    def __init__(self, model: NanoVLA, tokenizer, normalizer: Normalizer, cfg: Config, device="cuda"):
        self.model, self.tokenizer, self.normalizer, self.cfg, self.device = model.eval(), tokenizer, normalizer, cfg, device

    @classmethod
    def from_checkpoint(cls, path, device="cuda"):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        saved = {"dino_model": ""} | ckpt["config"]  # checkpoints from before DINO support are SigLIP only
        cfg = Config(**{k: tuple(v) if isinstance(v, list) else v for k, v in saved.items()})
        model = NanoVLA(cfg)
        model.load_state_dict(ckpt["model"])
        tokenizer = AutoTokenizer.from_pretrained(cfg.lm_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        stats_path = os.path.join(os.path.dirname(path), "norm_stats.json")
        normalizer = Normalizer.load(stats_path if os.path.exists(stats_path) else cfg.norm_stats)
        return cls(model.to(device), tokenizer, normalizer, cfg, device)

    @torch.no_grad()
    def infer(self, obs: dict) -> np.ndarray:
        state = np.asarray(obs["state"], dtype=np.float32)
        if state.shape[-1] == 43:  # whole-body vector from the robot -> upper body
            state = state[15:43]
        images = []
        for cam in self.cfg.cameras:
            img = torch.as_tensor(np.asarray(obs[cam])).permute(2, 0, 1)
            images.append(resize_with_pad(img, self.cfg.image_size))
        tok = self.tokenizer([obs["prompt"]], return_tensors="pt", truncation=True, max_length=self.cfg.max_text_len)
        state_t = torch.from_numpy(state).to(self.device)
        batch = {
            "images": torch.stack(images)[None].to(self.device),
            "state": self.normalizer.normalize(state_t, "state")[None],
            "input_ids": tok["input_ids"].to(self.device),
            "attention_mask": tok["attention_mask"].to(self.device),
        }
        with torch.autocast("cuda", dtype=torch.bfloat16):
            actions = self.model.sample_actions(batch)[0]
        actions = self.normalizer.unnormalize(actions.float(), "action")
        if self.cfg.delta_arms:
            mask = torch.from_numpy(ARM_MASK).to(self.device)
            actions[:, mask] += state_t[mask]
        return actions.cpu().numpy()
