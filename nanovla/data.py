"""LeRobot v2.1 reader for the G1 + Dex3 dataset (splitted_lerobot_and_school).

Layout per split (train / val / test):
  meta/episodes.jsonl, meta/tasks.jsonl, meta/info.json
  data/chunk-XXX/episode_XXXXXX.parquet     28-dim state and action per frame
  videos/chunk-XXX/observation.images.<cam>/episode_XXXXXX.mp4

Low-dim data for a whole split is loaded into memory once (~125 MB for train). Frames are read from the
224x224 re-encoded videos in `video_root` (see scripts/prepare_data.py), which have a short GOP so random
access only decodes a few frames.
"""

import json
import os
import random

import av
import numpy as np
import pyarrow.parquet as pq
import torch
import torchvision.transforms.v2 as T

# Dims interleave as [L_arm 7, L_hand 7, R_arm 7, R_hand 7].
ARM_MASK = np.array(([True] * 7 + [False] * 7) * 2)


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_split(root: str, split: str) -> dict:
    """Returns concatenated state/action arrays plus per-episode offsets for one split."""
    split_dir = os.path.join(root, split)
    info = json.load(open(os.path.join(split_dir, "meta/info.json")))
    tasks = {t["task_index"]: t["task"] for t in _read_jsonl(os.path.join(split_dir, "meta/tasks.jsonl"))}
    episodes = sorted(e["episode_index"] for e in _read_jsonl(os.path.join(split_dir, "meta/episodes.jsonl")))

    states, actions, task_idx, starts, lengths = [], [], [], [], []
    offset = 0
    for ep in episodes:
        path = os.path.join(split_dir, info["data_path"].format(episode_chunk=ep // info["chunks_size"], episode_index=ep))
        table = pq.read_table(path, columns=["observation.state", "action", "task_index", "frame_index"])
        frame_index = table["frame_index"].to_numpy()
        assert (frame_index == np.arange(len(frame_index))).all(), f"non-contiguous frames in {path}"
        states.append(np.stack(table["observation.state"].to_numpy(zero_copy_only=False)).astype(np.float32))
        actions.append(np.stack(table["action"].to_numpy(zero_copy_only=False)).astype(np.float32))
        task_idx.append(table["task_index"].to_numpy())
        starts.append(offset)
        lengths.append(len(frame_index))
        offset += len(frame_index)

    return {
        "info": info,
        "tasks": tasks,
        "episodes": np.array(episodes),
        "starts": np.array(starts),
        "lengths": np.array(lengths),
        "state": np.concatenate(states),
        "action": np.concatenate(actions),
        "task_index": np.concatenate(task_idx),
    }


def action_chunk(split: dict, ep_i: int, t: int, horizon: int, delta_arms: bool):
    """Actions t..t+horizon-1 of episode `ep_i` (last action repeated past the end), and the state at t."""
    start, length = split["starts"][ep_i], split["lengths"][ep_i]
    idx = start + np.minimum(np.arange(t, t + horizon), length - 1)
    state = split["state"][start + t]
    actions = split["action"][idx].copy()
    if delta_arms:
        actions[:, ARM_MASK] -= state[ARM_MASK]
    return state, actions


class Normalizer:
    """Mean/std normalization for state and actions.

    Reads either scripts/prepare_data.py output ({"state", "action"}) or an openpi norm_stats.json
    ({"norm_stats": {"state", "actions"}}); both hold arm-delta / hand-absolute action stats.
    """

    def __init__(self, stats: dict):
        stats = stats.get("norm_stats", stats)
        if "actions" in stats:
            stats["action"] = stats.pop("actions")
        self.stats = {k: {s: torch.tensor(stats[k][s], dtype=torch.float32) for s in ("mean", "std")} for k in ("state", "action")}

    @classmethod
    def load(cls, path):
        return cls(json.load(open(path)))

    def normalize(self, x, key):
        s = self.stats[key]
        return (x - s["mean"].to(x.device)) / (s["std"].to(x.device) + 1e-6)

    def unnormalize(self, x, key):
        s = self.stats[key]
        return x * (s["std"].to(x.device) + 1e-6) + s["mean"].to(x.device)


def read_frame(path: str, index: int, fps: int) -> np.ndarray:
    """Decodes one frame (HWC uint8) by seeking to the nearest keyframe before it."""
    with av.open(path) as container:
        stream = container.streams.video[0]
        target = int(round(index / fps / stream.time_base))
        container.seek(target, stream=stream, backward=True, any_frame=False)
        frame = None
        for frame in container.decode(stream):
            if frame.pts >= target:
                break
        return frame.to_ndarray(format="rgb24")


class G1Dataset(torch.utils.data.Dataset):
    """One sample = 3 camera images at frame t, state at t, action chunk t..t+H-1, and the task prompt."""

    def __init__(self, cfg, split: str, normalizer: Normalizer, train: bool = True):
        self.cfg = cfg
        self.split_name = split
        self.data = load_split(cfg.data_root, split)
        self.normalizer = normalizer
        self.augment = cfg.augment and train
        # Flat index over (episode, frame).
        self.index = np.concatenate([np.stack([np.full(n, i), np.arange(n)], 1) for i, n in enumerate(self.data["lengths"])])
        self.color = T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)
        self.crop = T.RandomResizedCrop(cfg.image_size, scale=(0.9, 1.0), ratio=(1.0, 1.0), antialias=True)

    def __len__(self):
        return len(self.index)

    def video_path(self, ep: int, cam: str) -> str:
        info = self.data["info"]
        rel = info["video_path"].format(episode_chunk=ep // info["chunks_size"], video_key=f"observation.images.{cam}", episode_index=ep)
        return os.path.join(self.cfg.video_root, self.split_name, rel)

    def __getitem__(self, i):
        ep_i, t = self.index[i]
        ep = int(self.data["episodes"][ep_i])
        state, actions = action_chunk(self.data, ep_i, t, self.cfg.action_horizon, self.cfg.delta_arms)

        images = []
        for cam in self.cfg.cameras:
            img = torch.from_numpy(read_frame(self.video_path(ep, cam), int(t), self.data["info"]["fps"])).permute(2, 0, 1)
            if self.augment:
                img = self.crop(img)
                # Color jitter on the head camera only, as in openpi (wrist views see mostly the hand).
                if cam == "ego_view":
                    img = self.color(img)
            images.append(img)

        return {
            "images": torch.stack(images),  # [n_cams, 3, H, W] uint8
            "state": self.normalizer.normalize(torch.from_numpy(state), "state"),
            "actions": self.normalizer.normalize(torch.from_numpy(actions), "action"),
            "prompt": self.data["tasks"][int(self.data["task_index"][self.data["starts"][ep_i] + t])],
        }


def make_collate(tokenizer, max_len: int):
    def collate(batch):
        out = {k: torch.stack([b[k] for b in batch]) for k in ("images", "state", "actions")}
        tok = tokenizer([b["prompt"] for b in batch], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        out["input_ids"], out["attention_mask"] = tok["input_ids"], tok["attention_mask"]
        return out

    return collate


def make_loader(cfg, split, normalizer, tokenizer, train=True, seed=0):
    ds = G1Dataset(cfg, split, normalizer, train=train)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,  # val is shuffled too so a few batches cover many episodes
        num_workers=cfg.num_workers,
        collate_fn=make_collate(tokenizer, cfg.max_text_len),
        drop_last=train,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
        worker_init_fn=lambda w: random.seed(seed * 1000 + w),
        generator=torch.Generator().manual_seed(seed),
    )
