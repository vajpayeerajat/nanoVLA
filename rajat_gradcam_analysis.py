"""Grad-CAM + causal input analysis of a nanoVLA checkpoint on a held-out split.

Port of ../openpi-implementation/rajat_gradcam_analysis.py (pi05, JAX) to nanoVLA (PyTorch). Same measurements,
verdicts, figures, report and `--compare`, so reports from both repos read the same way. It runs on real samples
(3 cameras, real state, real action chunk) through the training pipeline of nanovla/data.py (224x224 letterboxed
frames from the re-encoded videos, delta arms / absolute hands, mean/std norm), so every number is measured against
the demonstrated action.

Differences from the pi05 version, all due to the architecture:
  - The state is not text in the prompt; it enters the action expert as its own token. "state" attribution is
    |grad . activation| of that token, and the "-state" ablation replaces the state with the train-mean state.
  - The prompt is the bare instruction (no "Task: ..., State: ...; Action:" template), so prompt:template is 0
    unless --prompt_prefix is set.
  - Camera removal drops the camera's tokens from attention (image_mask = False), as pi05 does.
  - With a depth branch (cfg.depth_model), "-depth" switches the depth embedding off and keeps the RGB tokens.

What is measured, per sample (frames sampled across episodes of every task):

  1. Action accuracy   sampled 50-step chunk (fixed noise) vs the ground-truth chunk, per joint group, compared with a
                       "hold still" baseline (repeat the current state). skill = MAE(model) / MAE(hold); < 1 is better.
  2. Grad-CAM          score = -flow-matching loss on the GT chunk (averaged over a fixed set of noise/time draws, i.e.
                       the training objective). Gradients w.r.t. the LM input tokens (projected vision tokens for image
                       patches, token embeddings for the prompt) and the expert's state token. Per camera:
                       cam = ReLU(sum_c alpha_c * A_c), alpha = mean gradient over that camera's patches.
                       Per token: |grad . activation| -> share of attribution per input group (cameras, task words,
                       state, template). Computed for all joints and separately for right arm / right hand / left arm.
  3. Causal ablations  relative change of the same loss when an input is removed: each camera masked out, state set to
                       the train mean, and the instruction swapped (pink<->blue task; two-object tasks swap the order).
                       Gradients say where the model is sensitive; ablations say what it actually needs.
  4. Object grounding  octopus segmented by colour on the ego view; CAM lift = (CAM mass on object) / (object area
                       fraction). > 1 means the map concentrates on the object. On two-object tasks the instructed
                       object is compared against the other one.

Output (out_dir): report.md (verdicts + recommended actions), summary.json, per_sample.csv, fig_*.png, samples/*.png.
`--compare dirA dirB ...` writes a side-by-side compare.md from existing summary.json files (also pi05 ones).

Usage:
  uv run rajat_gradcam_analysis.py --checkpoint runs/baseline_siglip/best.pt
Compare:
  uv run rajat_gradcam_analysis.py --compare plots/gradcam/<run1> plots/gradcam/<run2>
"""

import argparse
import json
import math
import os
from pathlib import Path
import re
import time as _time

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch

FPS = 20
STOPWORDS = {"a", "an", "the", "and", "or", "in", "on", "at", "to", "it", "is", "of", "up", "with", "for"}

CAMERA_KEYS = ("ego_view", "ego_left", "ego_right")  # nanoVLA cfg.cameras, same order as in the prefix
CAMERA_NAMES = ("ego_view", "left_wrist", "right_wrist")
ACTION_GROUPS = {"left_arm": slice(0, 7), "left_hand": slice(7, 14), "right_arm": slice(14, 21),
                 "right_hand": slice(21, 28)}
REAL_DIM = 28
CAM_TARGETS = ("all", "right_arm", "right_hand", "left_arm")  # Grad-CAM score variants (left hand never moves)
INPUT_GROUPS = ("ego_view", "left_wrist", "right_wrist", "prompt:task", "state", "prompt:template")
SHORT_HORIZON = 10  # steps (0.5 s at 20 fps): roughly what is executed before the next inference; --short_horizon

# Verdict thresholds (relative loss increase when an input is removed / swapped)
NEGLIGIBLE = 0.02
LANG_IGNORED, LANG_USED = 0.05, 0.25
SKILL_GOOD, SKILL_USELESS = 0.5, 0.9
LIFT_GOOD, LIFT_WEAK = 1.5, 0.8
MIN_OBJ_AREA, MAX_OBJ_AREA = 0.003, 0.25
IDLE_HAND_RANGE = 0.05  # episodes whose right-hand action never moves more than this are idle (no demonstration)


# ============================================================================
# Val data
# ============================================================================


def _read_jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def _hand_phases(actions):
    """approach / holding / open-after-grasp per frame, from the GT right-hand trajectory (deviation from the start)."""
    c = actions[:, ACTION_GROUPS["right_hand"]].mean(1)
    dev = np.abs(c - c[0])
    if dev.max() < 1e-3:
        return np.array(["approach"] * len(c))
    closed = dev / dev.max() > 0.5
    first = int(np.argmax(closed))
    return np.where(closed, "holding", np.where(np.arange(len(c)) < first, "approach", "open (after grasp)"))


def is_idle(actions):
    return float(np.ptp(actions[:, ACTION_GROUPS["right_hand"]].mean(1))) < IDLE_HAND_RANGE


def idle_episodes(data_root, split):
    """Episodes where the right hand never grasps: recorded with a pick-up prompt but the robot does nothing."""
    from nanovla.data import load_split

    split_dir = Path(data_root) / split
    if not (split_dir / "meta" / "episodes.jsonl").exists():
        return None
    d = load_split(data_root, split)
    first_task = {e["episode_index"]: e["tasks"][0] for e in _read_jsonl(split_dir / "meta" / "episodes.jsonl")}
    idle, idle_frames, by_task = [], 0, {}
    for ep, start, n in zip(d["episodes"], d["starts"], d["lengths"]):
        if is_idle(d["action"][start:start + n]):
            idle.append(int(ep))
            idle_frames += int(n)
            by_task[first_task[int(ep)]] = by_task.get(first_task[int(ep)], 0) + 1
    return {"split": str(split_dir), "n_episodes": len(d["episodes"]), "idle_episodes": idle,
            "n_idle": len(idle), "idle_frame_frac": idle_frames / max(int(d["lengths"].sum()), 1), "idle_by_task": by_task}


def load_val_samples(cfg, split, episodes_per_task, frames_per_episode, horizon, seed, skip_idle=()):
    """Frames from the 224x224 re-encoded videos (exactly what training sees, minus augmentation)."""
    from nanovla.data import load_split, read_frame

    d = load_split(cfg.data_root, split)
    info, tasks = d["info"], d["tasks"]
    ep_row = {int(ep): i for i, ep in enumerate(d["episodes"])}
    by_task = {}
    for ep in _read_jsonl(Path(cfg.data_root) / split / "meta" / "episodes.jsonl"):
        if ep["episode_index"] not in set(skip_idle):
            by_task.setdefault(ep["tasks"][0], []).append(ep["episode_index"])
    rng = np.random.default_rng(seed)
    samples = []
    for task_index, task in sorted(tasks.items()):
        eps = sorted(by_task.get(task, []))
        for ep in sorted(rng.choice(eps, size=min(episodes_per_task, len(eps)), replace=False).tolist()):
            i = ep_row[ep]
            start, n = d["starts"][i], d["lengths"][i]
            state, action = d["state"][start:start + n], d["action"][start:start + n]
            frames = np.unique(np.linspace(0.03 * n, 0.93 * n, frames_per_episode).astype(int))
            phases = _hand_phases(action)
            for t in frames:
                idx = np.minimum(np.arange(t, t + horizon), n - 1)
                images = {}
                for cam in CAMERA_KEYS:
                    rel = info["video_path"].format(episode_chunk=ep // info["chunks_size"],
                                                    video_key=f"observation.images.{cam}", episode_index=ep)
                    images[cam] = read_frame(os.path.join(cfg.video_root, split, rel), int(t), info["fps"])
                samples.append({
                    "episode": ep, "frame": int(t), "progress": float(t / n), "task_index": task_index,
                    "task": task, "phase": str(phases[t]), "state": state[t], "gt_actions": action[idx],
                    "images": images,
                })
        print(f"[data] task {task_index}: {len([s for s in samples if s['task_index'] == task_index])} samples")
    return samples, tasks


# Task indices differ between splits (test/val/train each number their tasks.jsonl differently), so everything below is
# derived from the instruction text, never from task_index.
_SWAP_WORDS = {"pink": "blue", "blue": "pink", "orange": "brown", "brown": "orange"}  # octopus tasks: blue->brown, pink->orange
_SWAP_BASKET = {"beige": "brown", "brown": "beige"}  # school tasks: one object, keep it, change the basket


def _sub_words(text, mapping):
    """Case-insensitive word substitution that keeps the capitalisation of the replaced word."""
    def repl(m):
        new = mapping[m.group(1).lower()]
        return new.capitalize() if m.group(1)[0].isupper() else new
    return re.sub(r"\b(" + "|".join(mapping) + r")\b", repl, text, flags=re.IGNORECASE)


def swapped_prompt(task):
    """Counterpart instruction that asks for a different action in the same scene:
    two-object tasks reverse the order, single-octopus tasks swap pink<->blue (and their baskets),
    other single-object tasks keep the object and swap the basket (beige<->brown).
    Returns the task unchanged if no rule applies (swap_prompt then equals base for that frame)."""
    if ", then " in task and len(set(re.findall(r"\b(pink|blue|yellow)\b", task, re.IGNORECASE))) > 1:
        a, b = task.rstrip(".").split(", then ", 1)
        return f"{b[0].upper()}{b[1:]}, then {a[0].lower()}{a[1:]}."
    if "octopus" in task.lower():
        return _sub_words(task, _SWAP_WORDS)
    return _sub_words(task, _SWAP_BASKET)


def target_colour(task):
    """Colour of the object the instruction asks to pick first, or None if it names none."""
    m = re.search(r"\b(pink|blue|yellow)\b", task, re.IGNORECASE)
    return m.group(1).lower() if m else None


# Prepended to every instruction the model sees (base and swapped). "" = exactly the training prompts; --prompt_prefix.
PROMPT_PREFIX = ""


def noise_and_times(n, horizon, dim, seed):
    """Fixed noise draws and stratified times from the training distribution Beta(1.5, 1) * 0.999 + 0.001."""
    noise = np.random.default_rng(seed).standard_normal((n, horizon, dim)).astype(np.float32)
    t = (((np.arange(n) + 0.5) / n) ** (2.0 / 3.0) * 0.999 + 0.001).astype(np.float32)
    return noise, t


def group_matrix(action_dim):
    g = np.zeros((len(CAM_TARGETS), action_dim), np.float32)
    for i, name in enumerate(CAM_TARGETS):
        g[i, slice(0, REAL_DIM) if name == "all" else ACTION_GROUPS[name]] = 1.0
    return g


def token_groups(cleaned, full, offsets):
    """Char offsets per token -> (word_map {word: [token idx]}, {"task", "other": [token idx]})."""
    def tokens_in(lo, hi):
        return [i for i, (s, e) in enumerate(offsets) if s >= 0 and s < hi and e > lo]

    t0 = full.index(cleaned)
    task_idx = tokens_in(t0, t0 + len(cleaned))
    other = [i for i in range(len(offsets)) if i not in set(task_idx)]
    word_map = {}
    for m in re.finditer(r"[A-Za-z]+", cleaned):
        w = m.group().lower()
        idx = tokens_in(t0 + m.start(), t0 + m.end())
        if w in STOPWORDS or len(w) < 2 or not idx:
            continue
        key, k = w, 2
        while key in word_map:
            key, k = f"{w}#{k}", k + 1
        word_map[key] = idx
    return word_map, {"task": task_idx, "other": other}


# ============================================================================
# Backend (nanoVLA). Exposes:
#   layout: cams -> prefix indices [tpc], grid, text_start
#   item(sample, prompt, state_n) -> one model input; batch(items) -> model batch
#   losses(batch, actions, noise, t) -> [B, D] per-dim flow-matching MSE (mean over draws and horizon)
#   gradcam(batch1, actions1, noise, t, gmat) -> {target: (act [P,E], grad [P,E], state attribution)}
#   sample(batch, noise1) -> [B, H, D] normalized actions;  to_robot(actions_n, state) -> absolute joint targets
# ============================================================================


class NanoBackend:
    kind = "nanovla"

    def __init__(self, checkpoint, num_steps, device="cuda"):
        from nanovla.policy import Policy

        print(f"[nanovla] loading {checkpoint}")
        self.policy = Policy.from_checkpoint(checkpoint, device=device)
        self.model, self.cfg, self.tokenizer, self.norm = (self.policy.model, self.policy.cfg, self.policy.tokenizer,
                                                           self.policy.normalizer)
        if tuple(self.cfg.cameras) != CAMERA_KEYS:
            raise ValueError(f"expected cameras {CAMERA_KEYS}, checkpoint has {self.cfg.cameras}")
        self.device, self.num_steps = device, num_steps
        self.has_depth = getattr(self.model.vision, "depth", None) is not None
        vision = self.cfg.vision_model.split("/")[-1] + (f" + depth ({self.cfg.depth_model.split('/')[-1]})" if self.has_depth else "")
        self.label = (f"nanoVLA ({vision} + {self.cfg.lm_model.split('/')[-1]}, "
                      f"{self.cfg.lm_num_layers or 'all'} LM layers, PyTorch)")
        tpc = self.model.vision.tokens_per_cam
        self.grid = math.isqrt(tpc)
        self.cam_index = {c: np.arange(i * tpc, (i + 1) * tpc) for i, c in enumerate(CAMERA_NAMES)}
        self.text_start = 3 * tpc
        self.mean_state_n = np.zeros(self.cfg.state_dim, np.float32)  # train-mean state in normalized space

    # --- transforms (nanovla/data.py pipeline, no augmentation)
    def item(self, s, *, prompt=None, state_n=None):
        from nanovla.data import ARM_MASK

        actions = s["gt_actions"].copy()
        if self.cfg.delta_arms:
            actions[:, ARM_MASK] -= s["state"][ARM_MASK]
        return {
            "images": torch.stack([torch.from_numpy(s["images"][k]).permute(2, 0, 1) for k in CAMERA_KEYS]),
            "image_mask": torch.ones(len(CAMERA_KEYS), dtype=torch.bool),
            "state": self.norm.normalize(torch.from_numpy(s["state"]), "state") if state_n is None else torch.from_numpy(state_n),
            "actions": self.norm.normalize(torch.from_numpy(actions), "action"),
            "prompt": PROMPT_PREFIX + (s["task"] if prompt is None else prompt),
        }

    def batch(self, items):
        tok = self.tokenizer([it["prompt"] for it in items], padding=True, truncation=True, max_length=self.cfg.max_text_len,
                             return_tensors="pt")
        out = {k: torch.stack([it[k] for it in items]).to(self.device) for k in ("images", "image_mask", "state", "actions")}
        out["input_ids"], out["attention_mask"] = tok["input_ids"].to(self.device), tok["attention_mask"].to(self.device)
        return out

    def to_robot(self, actions_n, state):
        from nanovla.data import ARM_MASK

        a = self.norm.unnormalize(torch.as_tensor(actions_n, dtype=torch.float32), "action").numpy()
        if self.cfg.delta_arms:
            a[:, ARM_MASK] += state[ARM_MASK]
        return a

    # --- model
    def _sqerr(self, ctx, pad, state, actions, noise, t):
        """Per-element squared error of the predicted velocity, every sample repeated over the R noise/time draws."""
        b, r = actions.shape[0], noise.shape[0]
        rep = lambda x: x.repeat_interleave(r, 0)  # noqa: E731
        a = rep(actions)
        nz, tt = noise.repeat(b, 1, 1), t.repeat(b)
        x_t = tt[:, None, None] * nz + (1 - tt[:, None, None]) * a
        v = self.model.expert(x_t, rep(state), tt, rep(ctx), rep(pad))
        return (v.float() - (nz - a)).square().unflatten(0, (b, r))  # [B, R, H, D]

    @torch.no_grad()
    def losses(self, batch, noise, t, use_depth=True):
        self.model.vision.use_depth = use_depth
        try:
            return self._losses(batch, noise, t)
        finally:
            self.model.vision.use_depth = True

    def _losses(self, batch, noise, t):
        ctx, pad = self.model.encode(batch["images"], batch["input_ids"], batch["attention_mask"], batch["image_mask"])
        se = self._sqerr(ctx, pad, batch["state"], batch["actions"], torch.from_numpy(noise).to(self.device),
                         torch.from_numpy(t).to(self.device))
        return se.mean((1, 2)).cpu().numpy()

    def gradcam(self, batch, noise, t, gmat):
        m = self.model
        with torch.no_grad():
            embeds, mask = m.embed_prefix(batch["images"], batch["input_ids"], batch["attention_mask"], batch["image_mask"])
        embeds = embeds.detach().requires_grad_(True)
        captured = {}
        hook = m.expert.state_in.register_forward_hook(lambda mod, inp, out: captured.__setitem__("state", out))
        try:
            ctx, pad = m.encode_embeds(embeds, mask)
            se = self._sqerr(ctx, pad, batch["state"], batch["actions"], torch.from_numpy(noise).to(self.device),
                             torch.from_numpy(t).to(self.device))
        finally:
            hook.remove()
        per_dim = se[0].mean((0, 1))  # [D]
        g = torch.from_numpy(gmat).to(self.device)
        group_losses = (g @ per_dim) / g.sum(1)
        act, s_act = embeds[0].detach().float(), captured["state"].detach().float()
        out = {}
        for i, target in enumerate(CAM_TARGETS):
            ge, gs = torch.autograd.grad(group_losses[i], [embeds, captured["state"]], retain_graph=i < len(CAM_TARGETS) - 1)
            # score = -loss, so its gradient is minus the loss gradient
            state_attr = float((-gs.float() * s_act).sum().abs())
            out[target] = (act.cpu().numpy(), -ge[0].float().cpu().numpy(), state_attr)
        return out

    @torch.no_grad()
    def sample(self, batch, noise1):
        b = batch["state"].shape[0]
        noise = torch.from_numpy(noise1).to(self.device)[None].repeat(b, 1, 1)
        return self.model.sample_actions(batch, num_steps=self.num_steps, noise=noise).cpu().numpy()

    def text_groups(self, prompt):
        tok = self.tokenizer(prompt, return_offsets_mapping=True, truncation=True, max_length=self.cfg.max_text_len)
        offsets = [(s, e) if e > s else (-1, -1) for s, e in tok["offset_mapping"]]
        cleaned = prompt[len(PROMPT_PREFIX):] if PROMPT_PREFIX and prompt.startswith(PROMPT_PREFIX) else prompt
        return (np.asarray(tok["input_ids"]), *token_groups(cleaned, prompt, offsets))


# ============================================================================
# Attribution helpers
# ============================================================================


def token_attribution(act, grad):
    """|grad . activation| per prefix token."""
    return np.abs((grad * act).sum(-1))


def token_valid(img, grid):
    """[grid*grid] bool: patch tokens that are mostly real image (not the letterbox band)."""
    m = content_mask(img).astype(np.float32)
    h, w = m.shape
    cells = m[: h // grid * grid, : w // grid * grid].reshape(grid, h // grid, grid, w // grid).mean((1, 3))
    return cells.ravel() > 0.5


def gradcam_maps(act, grad, backend, valid):
    """Classic Grad-CAM per camera: alpha = mean gradient over the camera's real-image patches,
    cam = ReLU(sum_c alpha_c * A_c); letterbox patches are left out of alpha and zeroed in the map."""
    maps = {}
    for cam in CAMERA_NAMES:
        idx, v = backend.cam_index[cam], valid[cam]
        a, g = act[idx], grad[idx]
        m = a @ g[v].mean(0)
        maps[cam] = (np.maximum(m, 0.0) * v).reshape(backend.grid, backend.grid)
    return maps


def input_group_shares(r, state_attr, backend, text_groups):
    shares = {c: r[backend.cam_index[c]].sum() for c in CAMERA_NAMES}
    for g, label in [("task", "prompt:task"), ("other", "prompt:template")]:
        shares[label] = r[backend.text_start + np.asarray(text_groups[g], int)].sum() if text_groups[g] else 0.0
    shares["state"] = state_attr
    total = sum(shares[k] for k in INPUT_GROUPS) + 1e-12
    return {k: float(shares[k] / total) for k in INPUT_GROUPS}


def word_shares(r, backend, word_map):
    raw = {}
    for w, idx in word_map.items():
        raw[w.split("#")[0]] = raw.get(w.split("#")[0], 0.0) + r[backend.text_start + np.asarray(idx)].sum()
    total = sum(raw.values()) + 1e-12
    return {w: float(v / total) for w, v in raw.items()}


def octopus_masks(img):
    import scipy.ndimage as ndi

    hsv = mcolors.rgb_to_hsv(img.astype(np.float32) / 255.0)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    def largest(m):
        lab, n = ndi.label(ndi.binary_opening(m, iterations=2))
        if n == 0:
            return np.zeros_like(m)
        return lab == 1 + int(np.argmax(ndi.sum(m, lab, range(1, n + 1))))

    return {"pink": largest((h > 0.80) & (h < 0.97) & (s > 0.35) & (v > 0.35)),
            "blue": largest((h > 0.45) & (h < 0.54) & (s > 0.38) & (v > 0.5))}


def content_mask(img):
    """False on the black letterbox bands added by the 224x224 resize-with-pad. A small threshold, because the
    re-encoded (lossy) video leaves the bands near-black rather than exactly 0."""
    rows, cols = img.max(axis=(1, 2)) > 16, img.max(axis=(0, 2)) > 16
    return rows[:, None] & cols[None, :]


def upsample(cam, size):
    return np.asarray(Image.fromarray(cam.astype(np.float32), mode="F").resize((size[1], size[0]), Image.BICUBIC)).clip(0)


def cam_lift(cam_up, mask, valid):
    cam_up = cam_up * valid
    area = mask[valid].mean()
    if not (MIN_OBJ_AREA <= area <= MAX_OBJ_AREA) or cam_up.sum() <= 0:
        return np.nan
    return float((cam_up[mask].sum() / cam_up.sum()) / area)


# ============================================================================
# Main analysis
# ============================================================================


def analyse(args):
    ckpt = Path(args.checkpoint)
    run, step = ckpt.parent.name, ckpt.stem
    out_dir = Path(args.out_dir or f"./plots/gradcam/{run}__{step}")
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    print(f"[out] {out_dir}")

    torch.backends.cuda.matmul.allow_tf32 = False  # exact fp32 gradients; the model is small
    backend = NanoBackend(args.checkpoint, args.num_steps)
    cfg = backend.cfg
    if args.data_root:
        cfg.data_root = args.data_root
    if args.video_root:
        cfg.video_root = args.video_root
    horizon, adim = cfg.action_horizon, cfg.action_dim
    data_check = {"val": idle_episodes(cfg.data_root, args.split),
                  "train": None if args.skip_train_check else idle_episodes(cfg.data_root, "train")}
    for k, d in data_check.items():
        if d:
            print(f"[data] {k}: {d['n_idle']}/{d['n_episodes']} idle episodes ({100 * d['idle_frame_frac']:.1f}% of frames)")
    skip = [] if args.include_idle else data_check["val"]["idle_episodes"]
    samples, tasks = load_val_samples(cfg, args.split, args.episodes_per_task, args.frames_per_episode, horizon, args.seed,
                                      skip_idle=skip)
    noise, times = noise_and_times(args.n_noise, horizon, adim, args.seed)
    gmat = group_matrix(adim)
    sample_noise = np.random.default_rng(args.seed + 1).standard_normal((horizon, adim)).astype(np.float32)

    variants = ["base", "-ego_view", "-left_wrist", "-right_wrist", "-all_cameras", "scene_swap", "-state", "swap_prompt"]
    rows, cams_to_plot = [], []
    t0 = _time.time()
    for si, s in enumerate(samples):
        base = backend.item(s)
        # Scene swap: all 3 cameras from a sample of another episode/task (in-distribution images, wrong scene).
        other = backend.item(samples[(si + len(samples) // 2) % len(samples)])
        items = []
        for v in variants:
            it = {**base, "image_mask": base["image_mask"].clone()}
            if v.startswith("-") and (v[1:] in CAMERA_NAMES or v == "-all_cameras"):
                for c in CAMERA_NAMES if v == "-all_cameras" else [v[1:]]:
                    ci = CAMERA_NAMES.index(c)
                    it["image_mask"][ci] = False
                    it["images"] = it["images"].clone()
                    it["images"][ci] = 0
            elif v == "scene_swap":
                it["images"] = other["images"]
            elif v == "-state":
                it["state"] = torch.from_numpy(backend.mean_state_n)
            elif v == "swap_prompt":
                it["prompt"] = PROMPT_PREFIX + swapped_prompt(s["task"])
            items.append(it)
        batch = backend.batch(items)

        # 1) losses of every variant, per action dim -> per joint group
        per_dim = backend.losses(batch, noise, times)  # [V, D]
        row = {k: s[k] for k in ("episode", "frame", "progress", "task_index", "phase")}
        row["target"] = target_colour(s["task"])
        for vi, v in enumerate(variants):
            row[f"loss[{v}]"] = float(per_dim[vi, :REAL_DIM].mean())
            for g, sl in ACTION_GROUPS.items():
                row[f"loss[{v}]_{g}"] = float(per_dim[vi, sl].mean())
        if backend.has_depth:  # depth embedding off, RGB and everything else as in base
            no_depth = backend.losses(backend.batch([items[0]]), noise, times, use_depth=False)[0]
            row["loss[-depth]"] = float(no_depth[:REAL_DIM].mean())
            for g, sl in ACTION_GROUPS.items():
                row[f"loss[-depth]_{g}"] = float(no_depth[sl].mean())

        # 2) sampled actions (base + swapped prompt) vs GT and the hold-still baseline, in robot units
        sampled = backend.sample(backend.batch([items[0], items[-1]]), sample_noise)
        abs_actions = [backend.to_robot(a, s["state"]) for a in sampled]
        gt, hold = s["gt_actions"], np.repeat(s["state"][None], horizon, 0)
        for g, sl in ACTION_GROUPS.items():
            for hname, hz in [("short", SHORT_HORIZON), ("full", horizon)]:
                row[f"mae_{hname}_{g}"] = float(np.abs(abs_actions[0][:hz, sl] - gt[:hz, sl]).mean())
                row[f"hold_{hname}_{g}"] = float(np.abs(hold[:hz, sl] - gt[:hz, sl]).mean())
            row[f"swap_dact_short_{g}"] = float(np.abs(abs_actions[1][:SHORT_HORIZON, sl] - abs_actions[0][:SHORT_HORIZON, sl]).mean())

        # 3) Grad-CAM on the base sample
        one = backend.batch([items[0]])
        ids, word_map, tgroups = backend.text_groups(items[0]["prompt"])
        if not np.array_equal(ids, one["input_ids"][0].cpu().numpy()):
            raise RuntimeError("Prompt token reconstruction differs from the model batch")
        gc = backend.gradcam(one, noise, times, gmat)
        images = {c: s["images"][k] for c, k in zip(CAMERA_NAMES, CAMERA_KEYS)}
        ego_img = images["ego_view"]
        masks = octopus_masks(ego_img)
        valid = content_mask(ego_img)
        tok_valid = {c: token_valid(images[c], backend.grid) for c in CAMERA_NAMES}
        sample_maps = {}
        for target, (act, grad, state_attr) in gc.items():
            r = token_attribution(act, grad)
            maps = gradcam_maps(act, grad, backend, tok_valid)
            sample_maps[target] = maps
            for k, v in input_group_shares(r, state_attr, backend, tgroups).items():
                row[f"share[{target}]_{k}"] = v
            if target == "all":
                for w, v in word_shares(r, backend, word_map).items():
                    row[f"word_{w}"] = v
            up = upsample(maps["ego_view"], ego_img.shape[:2])
            for color, m in masks.items():
                row[f"lift[{target}]_{color}"] = cam_lift(up, m, valid)
        for color, m in masks.items():
            row[f"area_{color}"] = float(m.mean())
        rows.append(row)
        cams_to_plot.append({"sample": s, "images": images, "maps": sample_maps, "row": row})
        el = _time.time() - t0
        print(f"[{si + 1}/{len(samples)}] task {s['task_index']} ep {s['episode']} f {s['frame']} ({s['phase']}) "
              f"loss {row['loss[base]']:.4f}  skill(r_arm) {row['mae_short_right_arm'] / max(row['hold_short_right_arm'], 1e-9):.2f}"
              f"  eta {el / (si + 1) * (len(samples) - si - 1) / 60:.1f} min")

    df = pd.DataFrame(rows)
    df[[c for c in df.columns if c.startswith("word_")]] = df[[c for c in df.columns if c.startswith("word_")]].fillna(0.0)
    df.to_csv(out_dir / "per_sample.csv", index=False)
    meta = {"checkpoint": str(args.checkpoint), "config": run, "model": backend.label, "kind": backend.kind,
            "horizon": horizon, "short_horizon": SHORT_HORIZON, "prompt_prefix": PROMPT_PREFIX,
            "n_samples": len(df), "n_noise": args.n_noise, "num_steps": args.num_steps,
            "val_dir": str(Path(cfg.data_root) / args.split),
            "grid": backend.grid, "include_idle": bool(args.include_idle), "data_check": data_check}
    summary = summarize(df, meta)
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)
    make_figures(df, cams_to_plot, out_dir, meta)
    save_cams(cams_to_plot, out_dir)
    write_report(summary, df, out_dir)
    print(f"\nReport: {out_dir / 'report.md'}")


# ============================================================================
# Summary + verdicts
# ============================================================================


def _rel(df, v, sub=None, group=None):
    d = df if sub is None else df[sub]
    if len(d) == 0:
        return float("nan")
    col = f"loss[{v}]" + (f"_{group}" if group else "")
    base = "loss[base]" + (f"_{group}" if group else "")
    return float(d[col].mean() / d[base].mean() - 1.0)


def summarize(df, meta):
    s = {"meta": meta, "val_loss": float(df["loss[base]"].mean())}
    s["val_loss_by_group"] = {g: float(df[f"loss[base]_{g}"].mean()) for g in ACTION_GROUPS}
    acc = {}
    for g in ACTION_GROUPS:
        for h in ("short", "full"):
            m, b = df[f"mae_{h}_{g}"].mean(), df[f"hold_{h}_{g}"].mean()
            acc[f"{g}_{h}"] = {"mae": float(m), "hold_mae": float(b), "skill": float(m / max(b, 1e-9))}
    s["accuracy"] = acc
    s["accuracy_by_phase"] = {
        p: {g: float(d[f"mae_short_{g}"].mean() / max(d[f"hold_short_{g}"].mean(), 1e-9)) for g in ("right_arm", "right_hand")}
        for p, d in df.groupby("phase")}
    s["ablation"] = {v: {"all": _rel(df, v), **{g: _rel(df, v, group=g) for g in ("right_arm", "right_hand")}}
                     for v in ("-ego_view", "-left_wrist", "-right_wrist", "-all_cameras", "scene_swap", "-state",
                               *(["-depth"] if "loss[-depth]" in df.columns else []))}
    # Instruction matters only where it disambiguates: frames with both octopi visible on the ego view.
    two = (df["area_pink"] >= MIN_OBJ_AREA) & (df["area_blue"] >= MIN_OBJ_AREA)
    one = ~two
    s["language"] = {
        "swap_two_object": _rel(df, "swap_prompt", two), "swap_single_object": _rel(df, "swap_prompt", one),
        "n_two_object": int(two.sum()),
        "swap_two_object_approach": _rel(df, "swap_prompt", two & (df["phase"] == "approach")),
        "swap_dact_short_right_arm_two_object": float(df.loc[two, "swap_dact_short_right_arm"].mean()) if two.any() else float("nan"),
        "gt_motion_short_right_arm": float(df["hold_short_right_arm"].mean()),
    }
    s["grad_shares"] = {t: {k: float(df[f"share[{t}]_{k}"].mean()) for k in INPUT_GROUPS} for t in CAM_TARGETS}
    s["grad_shares_by_phase"] = {p: {k: float(d[f"share[all]_{k}"].mean()) for k in INPUT_GROUPS} for p, d in df.groupby("phase")}
    s["word_shares"] = {c[5:]: float(df[c].mean()) for c in df.columns if c.startswith("word_")}

    # grounding: target object lift (single object: the visible one; two-object: instructed vs other, approach only)
    g = {}
    for t in ("all", "right_arm"):
        lifts, tgt, oth = [], [], []
        for _, r in df.iterrows():
            obj = r["target"]
            other = "blue" if obj == "pink" else "pink"
            if r["phase"] == "approach" and obj in ("pink", "blue"):  # octopus_masks only segments pink/blue
                lifts.append(r[f"lift[{t}]_{obj}"])
                if r["area_pink"] >= MIN_OBJ_AREA and r["area_blue"] >= MIN_OBJ_AREA:
                    tgt.append(r[f"lift[{t}]_{obj}"])
                    oth.append(r[f"lift[{t}]_{other}"])
        lifts = np.array(lifts, float)
        pair = np.array([(a, b) for a, b in zip(tgt, oth) if np.isfinite(a) and np.isfinite(b)])
        g[t] = {"approach_target_lift_median": float(np.nanmedian(lifts)) if np.isfinite(lifts).any() else float("nan"),
                "approach_frac_lift_gt1": float(np.nanmean(lifts[np.isfinite(lifts)] > 1)) if np.isfinite(lifts).any() else float("nan"),
                "n": int(np.isfinite(lifts).sum()),
                "two_object_target_vs_other": [float(np.median(pair[:, 0])), float(np.median(pair[:, 1]))] if len(pair) else None,
                "two_object_frac_target_gt_other": float(np.mean(pair[:, 0] > pair[:, 1])) if len(pair) else float("nan"),
                "n_pairs": int(len(pair))}
    s["grounding"] = g
    s["verdicts"], s["actions"] = verdicts(s)
    return s


def _fmt(x, pct=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{100 * x:+.1f}%" if pct else f"{x:.3f}"


def verdicts(s):
    """Rule-based conclusions. Each verdict: (area, status OK/WARN/FAIL/INFO, text). Plus recommended actions."""
    v, acts = [], []

    # --- action accuracy
    for g in ("right_arm", "right_hand"):
        sk = s["accuracy"][f"{g}_short"]["skill"]
        st = "OK" if sk < SKILL_GOOD else ("WARN" if sk < SKILL_USELESS else "FAIL")
        v.append(("accuracy", st, f"{g}: first {SHORT_HORIZON} steps MAE {s['accuracy'][f'{g}_short']['mae']:.4f} vs "
                                  f"hold-still {s['accuracy'][f'{g}_short']['hold_mae']:.4f} (skill {sk:.2f}; <{SKILL_GOOD} good, "
                                  f">{SKILL_USELESS} no better than not moving)"))
    worst_phase = max(s["accuracy_by_phase"].items(), key=lambda kv: kv[1]["right_arm"])
    v.append(("accuracy", "INFO", f"Hardest phase: '{worst_phase[0]}' (right-arm skill {worst_phase[1]['right_arm']:.2f}). "
                                  + ", ".join(f"{p}: {d['right_arm']:.2f}" for p, d in s["accuracy_by_phase"].items())))
    if s["accuracy"]["right_hand_short"]["skill"] >= SKILL_USELESS:
        acts.append("Right-hand (grasp) predictions are no better than holding still: check grasp timing in the data, "
                    "and weight/over-sample the grasp/release frames.")
    if worst_phase[1]["right_arm"] > SKILL_GOOD:
        acts.append(f"Errors concentrate in the '{worst_phase[0]}' phase: inspect samples/ for that phase and "
                    "consider more demonstrations or longer training on it.")

    # --- what is used (causal)
    abl = s["ablation"]
    cams = {c: abl[f"-{c}"]["all"] for c in CAMERA_NAMES}
    vision = max(abl["-all_cameras"]["all"], abl["scene_swap"]["all"])
    sd = abl["-state"]["all"]
    st = "FAIL" if vision < NEGLIGIBLE else ("WARN" if vision < 0.1 else "OK")
    v.append(("inputs", st, f"Vision: removing all cameras {_fmt(abl['-all_cameras']['all'], True)}, showing another "
                            f"episode's scene {_fmt(abl['scene_swap']['all'], True)} "
                            f"({'ignored' if st == 'FAIL' else 'weakly used' if st == 'WARN' else 'used'})."))
    v.append(("inputs", "INFO", "Single camera removed: " + ", ".join(f"{c} {_fmt(d, True)}" for c, d in cams.items())
              + (" -> the cameras are redundant; none is individually essential." if max(cams.values()) < NEGLIGIBLE <= vision
                 else "")))
    v.append(("inputs", "INFO", f"Replacing the state with the train-mean state: loss {_fmt(sd, True)}"))
    if "-depth" in abl:
        dd = abl["-depth"]
        st = "INFO" if dd["all"] >= NEGLIGIBLE else "WARN"
        v.append(("inputs", st, f"Depth map switched off (RGB kept): loss {_fmt(dd['all'], True)} (right arm "
                                f"{_fmt(dd['right_arm'], True)}, right hand {_fmt(dd['right_hand'], True)})"
                                + (" -> the depth branch is barely used." if st == "WARN" else ".")))
    if vision < NEGLIGIBLE:
        acts.append("Vision is not used. Add image augmentation and object-position diversity; consider a "
                    "higher-resolution ego view or less image-token pooling (image_pool).")
    if sd > 3 * max(vision, NEGLIGIBLE):
        v.append(("inputs", "WARN", f"Proprioception dominates: the state matters {sd / max(vision, NEGLIGIBLE):.1f}x more than "
                                    "all cameras together. The policy largely continues its own trajectory, and reacts "
                                    "to the scene less than to where its arm is."))
        acts.append("Reduce state shortcutting: state dropout / noise on the state input during training, and "
                    "test with objects placed at positions not seen in training.")
    if cams["left_wrist"] >= max(cams["right_wrist"], NEGLIGIBLE):
        v.append(("inputs", "WARN", "The left wrist camera matters as much as or more than the right one, although only the "
                                    "right arm/hand do the task (left hand never moves): likely a spurious cue."))
        acts.append("Try dropping or randomly masking the left wrist camera during training.")

    # --- language
    lang = s["language"]
    sw = lang["swap_two_object"]
    if np.isfinite(sw):
        st = "FAIL" if sw < LANG_IGNORED else ("WARN" if sw < LANG_USED else "OK")
        word = "IGNORED" if st == "FAIL" else ("WEAK" if st == "WARN" else "USED")
        v.append(("language", st, f"Instruction {word} where both octopi are visible ({int(s['language']['n_two_object'])} frames): "
                                  "swapping the instruction raises loss "
                                  f"{_fmt(sw, True)} (approach phase {_fmt(lang['swap_two_object_approach'], True)}); "
                                  f"first-{SHORT_HORIZON}-step right-arm action changes by {lang['swap_dact_short_right_arm_two_object']:.4f} "
                                  f"(typical GT motion {lang['gt_motion_short_right_arm']:.4f})."))
        if st != "OK":
            acts.append("The policy picks the object from the scene, not from the instruction. More two-object "
                        "episodes with both orders, and/or prompt-swap negatives, are needed before trusting it on "
                        "multi-object scenes.")
    v.append(("language", "INFO", f"Frames with only one octopus visible: swapping the instruction changes loss "
                                  f"{_fmt(lang['swap_single_object'], True)} (the scene alone determines the action there)."))

    # --- grounding
    gr = s["grounding"]["all"]
    lift = gr["approach_target_lift_median"]
    if np.isfinite(lift):
        st = "OK" if lift >= LIFT_GOOD else ("WARN" if lift >= LIFT_WEAK else "FAIL")
        v.append(("grounding", st, f"Before the grasp, Grad-CAM on the ego view puts {lift:.2f}x the uniform density on the "
                                   f"target octopus (median of {gr['n']} frames; {100 * gr['approach_frac_lift_gt1']:.0f}% of frames > 1)."))
        if st == "FAIL":
            acts.append("Grad-CAM does not concentrate on the object to pick: inspect samples/*approach*.png. If it "
                        "lights up the robot's own arm or the background, the model is tracking itself, not the object.")
    if gr["two_object_target_vs_other"]:
        a, b = gr["two_object_target_vs_other"]
        st = "OK" if gr["two_object_frac_target_gt_other"] > 0.6 else "WARN"
        v.append(("grounding", st, f"Both octopi visible, approach: lift on instructed object {a:.2f} vs other {b:.2f} "
                                   f"(instructed > other in {100 * gr['two_object_frac_target_gt_other']:.0f}% of {gr['n_pairs']} frames)."))

    # --- gradient vs ablation agreement
    gs = s["grad_shares"]["all"]
    top_grad = max(INPUT_GROUPS, key=lambda k: gs[k])
    v.append(("inputs", "INFO", f"Largest gradient attribution: {top_grad} ({100 * gs[top_grad]:.0f}%). Shares: "
                                + ", ".join(f"{k} {100 * gs[k]:.0f}%" for k in INPUT_GROUPS)))

    # --- data
    tr = s["meta"]["data_check"].get("train")
    if tr and tr["idle_frame_frac"] > 0.05:
        v.insert(0, ("data", "FAIL", f"{tr['n_idle']}/{tr['n_episodes']} train episodes ({100 * tr['idle_frame_frac']:.1f}% of "
                                     "frames) are idle: the right hand never grasps, the robot barely moves (empty table or "
                                     "a person handling the object), yet they carry a 'Pick up ...' prompt. They teach "
                                     "'stay still' under the same instruction and image distribution as real demos."))
        acts.insert(0, f"Remove the {tr['n_idle']} idle train episodes (and the val ones, which also make the val loss look "
                       f"better than it is) and retrain/fine-tune; list in summary.json -> meta.data_check. This is the "
                       "cheapest fix and most likely contributor to the state reliance below.")
    if not acts:
        acts.append("No blocking issue found on these val samples. Next: closed-loop evaluation on the robot.")
    return v, acts


# ============================================================================
# Figures
# ============================================================================

PALETTE = {"ego_view": "#4C72B0", "left_wrist": "#A0A0A0", "right_wrist": "#55A868", "prompt:task": "#C44E52",
           "state": "#DD8452", "prompt:template": "#D9D9D9"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
                     "axes.titlesize": 10, "figure.dpi": 100})


def _bbox(img):
    m = content_mask(img)
    rows, cols = np.where(m.any(1))[0], np.where(m.any(0))[0]
    return slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)


def _overlay(img, cam):
    """Heat fades in with intensity (cold regions keep the image), scaled to the 99.5th percentile, letterbox cropped."""
    valid = content_mask(img)
    up = upsample(cam, img.shape[:2]) * valid
    lo, hi = np.percentile(up[valid], [5, 99.5])
    u = np.clip((up - lo) / max(hi - lo, 1e-12), 0, 1) * valid
    a = (0.8 * u ** 0.7)[..., None]
    out = img.astype(np.float32) / 255.0 * (1 - a) + plt.get_cmap("turbo")(u)[..., :3] * a
    r, c = _bbox(img)
    return out[r, c]


def _show(ax, img, cam=None, title=None, contour=None):
    r, c = _bbox(img)
    ax.imshow(_overlay(img, cam) if cam is not None else img[r, c])
    if contour is not None and contour[r, c].any():
        ax.contour(contour[r, c], levels=[0.5], colors="white", linewidths=1.3)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if title:
        ax.set_title(title, fontsize=9, fontweight="normal")


def _short(meta):
    if meta.get("kind") == "nanovla":
        return f"nanoVLA {meta['config']} @ {Path(meta['checkpoint']).stem}"
    names = {"pi05_g1_pickplace": "pi05", "pi05_g1_dex3": "pi05 G1 Dex3", "pi05_g1_dex3_lora": "pi05 G1 Dex3 (LoRA)",
             "pi05_g1_dex3_debug": "pi05 G1 Dex3 (debug)"}
    return f"{names.get(meta['config'], meta['config'])} @ {Path(meta['checkpoint']).name}"


def _rel_row(r, v):
    return 100 * (r[f"loss[{v}]"] / r["loss[base]"] - 1)


def sample_figure(c, path, tag):
    s, r, imgs, maps = c["sample"], c["row"], c["images"], c["maps"]
    obj = target_colour(s["task"])
    contour = octopus_masks(imgs["ego_view"]).get(obj)
    fig = plt.figure(figsize=(17.5, 7.6))
    gs = fig.add_gridspec(2, 6, width_ratios=[1, 1, 1, 0.05, 0.62, 0.95], wspace=0.06, hspace=0.18)
    _show(fig.add_subplot(gs[0, 0]), imgs["ego_view"], title=f"ego_view input ({obj} octopus outlined)" if contour is not None else "ego_view input", contour=contour)
    _show(fig.add_subplot(gs[0, 1]), imgs["ego_view"], maps["all"]["ego_view"], "ego_view · all joints", contour)
    _show(fig.add_subplot(gs[0, 2]), imgs["right_wrist"], maps["all"]["right_wrist"], "right_wrist · all joints")
    _show(fig.add_subplot(gs[1, 0]), imgs["ego_view"], maps["right_arm"]["ego_view"], "ego_view · right arm", contour)
    _show(fig.add_subplot(gs[1, 1]), imgs["ego_view"], maps["right_hand"]["ego_view"], "ego_view · right hand (grasp)", contour)
    _show(fig.add_subplot(gs[1, 2]), imgs["left_wrist"], maps["all"]["left_wrist"], "left_wrist · all joints")
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap="turbo", norm=mcolors.Normalize(0, 1)), cax=fig.add_subplot(gs[:, 3]))
    cb.set_label("Grad-CAM (relative, per image)")

    ax = fig.add_subplot(gs[0, 5])
    vals = [r[f"share[all]_{k}"] for k in INPUT_GROUPS]
    ax.barh(INPUT_GROUPS[::-1], vals[::-1], color=[PALETTE[k] for k in INPUT_GROUPS[::-1]])
    for i, v in enumerate(vals[::-1]):
        ax.text(v + 0.01, i, f"{100 * v:.0f}%", va="center", fontsize=8)
    ax.set_xlim(0, 1.15)
    ax.set_title("Gradient share by input")
    ax = fig.add_subplot(gs[1, 5])
    names = ["-all_cameras", "scene_swap", "-state", "swap_prompt"]
    labels = ["no cameras", "other scene", "mean state", "other instruction"]
    d = [_rel_row(r, v) for v in names]
    ax.barh(labels[::-1], d[::-1], color=["#C44E52", "#DD8452", "#55A868", "#4C72B0"])
    ax.axvline(0, color="k", lw=0.8)
    ax.axvspan(-100 * NEGLIGIBLE, 100 * NEGLIGIBLE, color="0.9", zorder=0)
    for i, v in enumerate(d[::-1]):
        ax.text(v, i, f" {v:+.0f}%", va="center", ha="left" if v >= 0 else "right", fontsize=8)
    ax.set_title("Loss change when input removed")
    ax.set_xlabel("%")
    skill = r["mae_short_right_arm"] / max(r["hold_short_right_arm"], 1e-9)
    fig.suptitle(f"{tag}   |   task {s['task_index']} · ep {s['episode']} · frame {s['frame']} ({100 * s['progress']:.0f}% of episode) · "
                 f"{s['phase']}\n\"{s['task']}\"   loss {r['loss[base]']:.4f} · right-arm skill {skill:.2f} (vs hold still)",
                 fontsize=10)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def make_figures(df, cams_to_plot, out_dir, meta):
    tag = _short(meta)
    phases = [p for p in ("approach", "holding", "open (after grasp)") if p in set(df["phase"])]

    # --- dashboard
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5))
    ax = axes[0, 0]
    left = np.zeros(3)
    for k in INPUT_GROUPS:
        vals = np.array([df[f"share[{t}]_{k}"].mean() for t in ("all", "right_arm", "right_hand")])
        ax.barh(["all joints", "right arm", "right hand"], vals, left=left, color=PALETTE[k], label=k, edgecolor="white")
        for i, (l, v) in enumerate(zip(left, vals)):
            if v > 0.06:
                ax.text(l + v / 2, i, f"{100 * v:.0f}%", ha="center", va="center", fontsize=8)
        left += vals
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_title("Gradient attribution: what the loss is sensitive to")
    ax.legend(ncol=3, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False)

    ax = axes[0, 1]
    abl = [("-ego_view", "no ego_view"), ("-left_wrist", "no left_wrist"), ("-right_wrist", "no right_wrist"),
           ("-all_cameras", "no cameras"), ("scene_swap", "other episode's scene"), ("-state", "mean state"),
           ("swap_prompt", "other instruction")]
    colors = ["#4C72B0"] * 5 + ["#DD8452", "#C44E52"]
    if "loss[-depth]" in df.columns:
        abl.insert(5, ("-depth", "no depth (RGB kept)"))
        colors.insert(5, "#8172B3")
    d = np.array([100 * _rel(df, v) for v, _ in abl])
    ax.barh([lab for _, lab in abl], d, color=colors)
    ax.axvspan(-100 * NEGLIGIBLE, 100 * NEGLIGIBLE, color="0.92", zorder=0, label=f"negligible (±{100 * NEGLIGIBLE:.0f}%)")
    ax.axvline(0, color="k", lw=0.8)
    for i, v in enumerate(d):
        ax.text(v, i, f" {v:+.1f}% ", va="center", ha="left" if v >= 0 else "right", fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("flow-matching loss change (%)")
    ax.set_title("Causal test: what the model actually needs")
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    ax.set_xlim(min(d.min() * 1.3, -10), max(d.max() * 1.25, 10))

    ax = axes[0, 2]
    groups = ["right_arm", "right_hand", "left_arm"]
    x = np.arange(len(groups))
    for j, (h, lab, col) in enumerate([("short", f"first {SHORT_HORIZON} steps ({SHORT_HORIZON / FPS:.2f} s)", "#4C72B0"), ("full", f"full {meta['horizon']}-step chunk", "#8DA0CB")]):
        vals = [df[f"mae_{h}_{g}"].mean() / max(df[f"hold_{h}_{g}"].mean(), 1e-9) for g in groups]
        b = ax.bar(x + (j - 0.5) * 0.36, vals, 0.36, label=lab, color=col)
        ax.bar_label(b, fmt="%.2f", fontsize=8)
    ax.axhline(1, ls="--", color="#C44E52", lw=1, label="hold still (=1)")
    ax.axhspan(0, SKILL_GOOD, color="#55A868", alpha=0.08, label=f"good (<{SKILL_GOOD})")
    ax.set_xticks(x, groups)
    ax.set_ylabel("skill = MAE(model) / MAE(hold still)")
    ax.set_title("Action accuracy vs not moving (lower is better)")
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1, 0]
    x = np.arange(len(phases))
    for j, (g, col) in enumerate([("right_arm", "#4C72B0"), ("right_hand", "#55A868")]):
        vals = [df[df.phase == p][f"mae_short_{g}"].mean() / max(df[df.phase == p][f"hold_short_{g}"].mean(), 1e-9) for p in phases]
        b = ax.bar(x + (j - 0.5) * 0.36, vals, 0.36, label=g, color=col)
        ax.bar_label(b, fmt="%.2f", fontsize=8)
    ax.axhline(1, ls="--", color="#C44E52", lw=1)
    ax.set_xticks(x, [f"{p}\n(n={int((df.phase == p).sum())})" for p in phases])
    ax.set_ylabel(f"skill (first {SHORT_HORIZON} steps)")
    ax.set_title("Accuracy by task phase")
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1, 1]
    bottom = np.zeros(len(phases))
    for k in INPUT_GROUPS:
        vals = np.array([df[df.phase == p][f"share[all]_{k}"].mean() for p in phases])
        ax.bar(phases, vals, bottom=bottom, color=PALETTE[k], label=k, edgecolor="white", width=0.6)
        for i, (b0, v) in enumerate(zip(bottom, vals)):
            if v > 0.06:
                ax.text(i, b0 + v / 2, f"{100 * v:.0f}%", ha="center", va="center", fontsize=8)
        bottom += vals
    ax.set_ylim(0, 1)
    ax.set_title("Gradient attribution by task phase")

    ax = axes[1, 2]
    words = [c for c in df.columns if c.startswith("word_")]
    order = sorted(words, key=lambda w: -df[w].mean())
    tcols = ["#C44E52", "#4C72B0", "#DD8452", "#55A868"]
    for j, (ti, dd) in enumerate(df.groupby("task_index")):
        ax.bar(np.arange(len(order)) + (j - 1.5) * 0.2, [dd[w].mean() for w in order], 0.2, label=f"task {ti}", color=tcols[j % 4])
    ax.set_xticks(np.arange(len(order)), [w[5:] for w in order], rotation=30, ha="right")
    ax.set_ylabel("share among task words")
    ax.set_title("Which instruction words carry gradient")

    fig.suptitle(f"{tag}  ·  {meta['n_samples']} val frames  ·  {meta['model']}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_summary.png", dpi=130)
    plt.close(fig)

    # --- per-sample panels
    for c in cams_to_plot:
        s = c["sample"]
        sample_figure(c, out_dir / "samples" / f"task{s['task_index']}_ep{s['episode']:03d}_f{s['frame']:04d}_{s['phase'].split()[0]}.png", tag)

    # --- overview: one episode per task, ego view + right wrist over time
    first_eps = {}
    for c in cams_to_plot:
        first_eps.setdefault(c["sample"]["task_index"], c["sample"]["episode"])
    rows_ = [[c for c in cams_to_plot if c["sample"]["episode"] == ep] for _, ep in sorted(first_eps.items())]
    ncol = max(len(r) for r in rows_)
    fig, axes = plt.subplots(2 * len(rows_), ncol, figsize=(2.5 * ncol, 2.1 * 2 * len(rows_)), squeeze=False,
                             gridspec_kw={"wspace": 0.04, "hspace": 0.28})
    for i, rr in enumerate(rows_):
        for j in range(ncol):
            for k, cam in enumerate(("ego_view", "right_wrist")):
                ax = axes[2 * i + k, j]
                ax.axis("off")
                if j < len(rr):
                    c = rr[j]
                    obj = target_colour(c["sample"]["task"])
                    contour = octopus_masks(c["images"]["ego_view"]).get(obj) if cam == "ego_view" else None
                    _show(ax, c["images"][cam], c["maps"]["all"][cam],
                          f"{100 * c['sample']['progress']:.0f}% · {c['sample']['phase']}" if k == 0 else None, contour)
            axes[2 * i, 0].axis("on")
            axes[2 * i, 0].set_ylabel(f"task {rr[0]['sample']['task_index']}\nego_view", fontsize=9)
            axes[2 * i + 1, 0].axis("on")
            axes[2 * i + 1, 0].set_ylabel("right_wrist", fontsize=9)
            for a in (axes[2 * i, 0], axes[2 * i + 1, 0]):
                a.set_xticks([])
                a.set_yticks([])
                for sp in a.spines.values():
                    sp.set_visible(False)
    fig.suptitle(f"Grad-CAM through one episode per task (score: all joints; outline = instructed octopus)  ·  {tag}",
                 fontsize=11, fontweight="bold")
    fig.savefig(out_dir / "fig_gradcam_overview.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def save_cams(cams_to_plot, out_dir):
    """Images + maps for `--compare` (side-by-side Grad-CAM of several checkpoints on the same frames)."""
    arrays, index = {}, []
    for i, c in enumerate(cams_to_plot):
        s = c["sample"]
        index.append({k: s[k] for k in ("episode", "frame", "progress", "task_index", "phase", "task")})
        for cam in CAMERA_NAMES:
            arrays[f"img_{i}_{cam}"] = c["images"][cam]
            for t in ("all", "right_arm", "right_hand"):
                arrays[f"map_{i}_{t}_{cam}"] = c["maps"][t][cam].astype(np.float32)
    np.savez_compressed(out_dir / "cams.npz", index=json.dumps(index), **arrays)


# ============================================================================
# Report
# ============================================================================


def write_report(s, df, out_dir):
    m = s["meta"]
    L = [f"# Grad-CAM / input analysis — {_short(m)}", "",
         f"- Checkpoint: `{m['checkpoint']}`", f"- Model: {m['model']}",
         *([f"- **Prompt prefix added to every instruction: \"{m['prompt_prefix']}\"** (not the training prompts)"]
           if m.get("prompt_prefix") else []),
         f"- Samples ({m['val_dir']}): {m['n_samples']} frames ({df['episode'].nunique()} episodes, tasks "
         f"{sorted(df['task_index'].unique().tolist())}), {m['n_noise']} noise/time draws per loss, {m['num_steps']} denoising steps",
         f"- Flow-matching loss on these samples: **{s['val_loss']:.4f}** (right arm {s['val_loss_by_group']['right_arm']:.4f}, "
         f"right hand {s['val_loss_by_group']['right_hand']:.4f})", "",
         "## Verdicts", ""]
    for area, st, text in s["verdicts"]:
        L.append(f"- **[{st}] {area}** — {text}")
    L += ["", "## Recommended actions", ""] + [f"{i + 1}. {a}" for i, a in enumerate(s["actions"])]
    L += ["", "## Data check (idle episodes: right hand never moves)", ""]
    for k, d in m["data_check"].items():
        if d:
            L.append(f"- {k}: {d['n_idle']}/{d['n_episodes']} episodes, {100 * d['idle_frame_frac']:.1f}% of frames; by task: "
                     + ", ".join(f"'{t[:40]}…' {n}" for t, n in d["idle_by_task"].items()))
    L.append(f"- Idle episodes of the analysed split are {'included in' if m['include_idle'] else 'excluded from'} the analysis below.")
    L += ["", "## Action accuracy (robot units, rad)", "",
          f"| joint group | MAE first {SHORT_HORIZON} | hold-still | skill | MAE full {m['horizon']} | hold-still | skill |", "|---|---|---|---|---|---|---|"]
    for g in ACTION_GROUPS:
        a, b = s["accuracy"][f"{g}_short"], s["accuracy"][f"{g}_full"]
        L.append(f"| {g} | {a['mae']:.4f} | {a['hold_mae']:.4f} | {a['skill']:.2f} | {b['mae']:.4f} | {b['hold_mae']:.4f} | {b['skill']:.2f} |")
    L += ["", "Skill = MAE(model) / MAE(hold still). The left hand never moves in this data, so its skill is not meaningful.", "",
          "## What the model relies on", "",
          "| input | gradient share (all) | gradient share (right arm) | gradient share (right hand) | loss change when removed |",
          "|---|---|---|---|---|"]
    abl_name = {"ego_view": "-ego_view", "left_wrist": "-left_wrist", "right_wrist": "-right_wrist", "state": "-state",
                "prompt:task": "swap_prompt"}
    for k in INPUT_GROUPS:
        ab = s["ablation"].get(abl_name.get(k, ""), {}).get("all") if abl_name.get(k) != "swap_prompt" else \
            _rel(df, "swap_prompt")
        L.append(f"| {k} | {100 * s['grad_shares']['all'][k]:.1f}% | {100 * s['grad_shares']['right_arm'][k]:.1f}% | "
                 f"{100 * s['grad_shares']['right_hand'][k]:.1f}% | {_fmt(ab, True) if abl_name.get(k) else '—'} |")
    L += ["", f"All three cameras removed: **{_fmt(s['ablation']['-all_cameras']['all'], True)}**; cameras showing "
          f"another episode's scene: **{_fmt(s['ablation']['scene_swap']['all'], True)}** (right arm "
          f"{_fmt(s['ablation']['scene_swap']['right_arm'], True)}, right hand {_fmt(s['ablation']['scene_swap']['right_hand'], True)}).",
          *([f"", f"Depth map switched off, RGB kept: **{_fmt(s['ablation']['-depth']['all'], True)}** (right arm "
             f"{_fmt(s['ablation']['-depth']['right_arm'], True)}, right hand {_fmt(s['ablation']['-depth']['right_hand'], True)})."]
            if "-depth" in s["ablation"] else []),
          "", "Prompt:task 'removed' = instruction swapped for the other task; state 'removed' = train-mean state. Gradient "
          "shares show sensitivity; the loss change shows what the model actually needs. Trust a conclusion when both agree.", "",
          "### By phase (gradient share, all joints)", "", "| phase | " + " | ".join(INPUT_GROUPS) + " |",
          "|---|" + "---|" * len(INPUT_GROUPS)]
    for p, d in s["grad_shares_by_phase"].items():
        L.append(f"| {p} | " + " | ".join(f"{100 * d[k]:.0f}%" for k in INPUT_GROUPS) + " |")
    L += ["", "### Task words (share of gradient among task words)", "",
          ", ".join(f"{w}: {100 * v:.0f}%" for w, v in sorted(s["word_shares"].items(), key=lambda kv: -kv[1])), "",
          "## Figures", "", "![summary](fig_summary.png)", "", "![overview](fig_gradcam_overview.png)", "",
          "- `samples/` — every analysed frame: Grad-CAM on all 3 cameras, right-arm / right-hand scores, input reliance",
          "- `per_sample.csv`, `summary.json`, `cams.npz` — raw numbers and maps (`--compare` uses them)", "",
          "## Method and caveats", "",
          "- Score = −flow-matching loss on the ground-truth chunk (the training objective), averaged over fixed noise draws "
          "and stratified times t ~ Beta(1.5, 1). Grad-CAM is taken at the LM input (projected vision tokens + camera "
          "embedding), the closest analogue of the last conv layer. Map resolution: "
          f"{m['grid']}x{m['grid']} per camera.",
          "- State attribution is |grad . activation| of the action expert's state token (the state is not in the prompt).",
          "- Camera removal masks the camera's tokens (never seen in training, so it is an upper bound on reliance); "
          "state removal feeds the train-mean state; the GT target is unchanged in all variants.",
          "- Object masks come from colour thresholds (pink / cyan octopus) on the ego view; frames where the object is "
          f"not visible (area < {MIN_OBJ_AREA:.1%}) are skipped. Baskets are not segmented (their colours overlap the table).",
          "- Open-loop, single-frame analysis. It explains what the checkpoint uses; closed-loop success still needs a robot run."]
    (out_dir / "report.md").write_text("\n".join(L) + "\n")


def compare(dirs, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    runs = [json.load(open(Path(d) / "summary.json")) for d in dirs]
    names = [_short(r["meta"]) for r in runs]
    L = ["# Checkpoint comparison", "", "| metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(runs)]

    def row(label, f):
        vals = []
        for r in runs:
            try:
                vals.append(f(r))
            except (KeyError, TypeError):
                vals.append("n/a")
        L.append(f"| {label} | " + " | ".join(vals) + " |")

    def share(r, k):
        # pi05 reports put the state in the prompt ("prompt:state"); nanoVLA has a separate "state" input.
        g = r["grad_shares"]["all"]
        return g.get(k, g.get("prompt:state") if k == "state" else None)

    row("val loss", lambda r: f"{r['val_loss']:.4f}")
    for g in ("right_arm", "right_hand"):
        row(f"skill {g} (first {SHORT_HORIZON}, lower better)", lambda r, g=g: f"{r['accuracy'][f'{g}_short']['skill']:.2f}")
    for v in ("-all_cameras", "scene_swap", "-state", "-depth"):
        row(f"loss change {v}", lambda r, v=v: _fmt(r["ablation"][v]["all"], True))
    row("instruction swap, both octopi visible", lambda r: _fmt(r["language"]["swap_two_object"], True))
    row("target-object CAM lift (approach)", lambda r: _fmt(r["grounding"]["all"]["approach_target_lift_median"]))
    for k in INPUT_GROUPS:
        row(f"grad share {k}", lambda r, k=k: f"{100 * share(r, k):.0f}%")
    L += ["", "![metrics](compare_metrics.png)", "", "![gradcam](compare_gradcam.png)", ""]
    L += ["## Verdicts per checkpoint", ""]
    for n, r in zip(names, runs):
        L += [f"### {n}", ""] + [f"- [{st}] {a}: {t}" for a, st, t in r["verdicts"]] + [""]
    out.write_text("\n".join(L) + "\n")

    # --- metrics figure
    rc = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), gridspec_kw={"width_ratios": [0.8, 1.3, 1]})
    n = len(runs)
    w = 0.8 / n
    cats = [("right_arm", "right arm"), ("right_hand", "right hand")]
    for j, r in enumerate(runs):
        b = axes[0].bar(np.arange(2) + (j - (n - 1) / 2) * w, [r["accuracy"][f"{g}_short"]["skill"] for g, _ in cats], w,
                        color=rc[j % 5], label=names[j])
        axes[0].bar_label(b, fmt="%.2f", fontsize=7)
    axes[0].axhline(1, ls="--", color="k", lw=0.8)
    axes[0].set_xticks(np.arange(2), [c for _, c in cats])
    axes[0].set_title("Skill vs hold-still (lower better)")
    abl = [("-all_cameras", "no cameras"), ("scene_swap", "other scene"), ("-state", "no / mean state")]
    for j, r in enumerate(runs):
        vals = [100 * r["ablation"][v]["all"] for v, _ in abl] + [100 * r["language"]["swap_two_object"]]
        b = axes[1].bar(np.arange(4) + (j - (n - 1) / 2) * w, vals, w, color=rc[j % 5], label=names[j])
        axes[1].bar_label(b, fmt="%+.0f%%", fontsize=7)
    axes[1].axhspan(-100 * NEGLIGIBLE, 100 * NEGLIGIBLE, color="0.92", zorder=0)
    axes[1].set_xticks(np.arange(4), [lab for _, lab in abl] + ["other instruction\n(both octopi visible)"])
    axes[1].set_ylabel("loss change (%)")
    axes[1].set_title("Causal reliance on inputs")
    axes[1].legend(fontsize=8, frameon=False)
    left = np.zeros(n)
    for k in INPUT_GROUPS:
        vals = np.array([share(r, k) or 0.0 for r in runs])
        axes[2].barh(names, vals, left=left, color=PALETTE[k], label=k, edgecolor="white")
        for i, (l, v) in enumerate(zip(left, vals)):
            if v > 0.06:
                axes[2].text(l + v / 2, i, f"{100 * v:.0f}%", ha="center", va="center", fontsize=8)
        left += vals
    axes[2].invert_yaxis()
    axes[2].set_xlim(0, 1)
    axes[2].set_title("Gradient attribution share")
    axes[2].legend(ncol=3, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.tight_layout()
    fig.savefig(out.parent / "compare_metrics.png", dpi=130)
    plt.close(fig)

    # --- Grad-CAM side by side on the same val frames
    cams = []
    for d in dirs:
        z = np.load(Path(d) / "cams.npz")
        cams.append((json.loads(str(z["index"])), z))
    keys = [{(e["episode"], e["frame"]) for e in idx} for idx, _ in cams]
    common = set.intersection(*keys)
    picks = []
    for t in sorted({e["task_index"] for e in cams[0][0]}):
        cands = [e for e in cams[0][0] if e["task_index"] == t and (e["episode"], e["frame"]) in common]
        for ph in ("approach", "holding"):
            hit = [e for e in cands if e["phase"] == ph]
            if hit:
                picks.append(hit[len(hit) // 2])
    if not picks:
        print("No common frames between runs; skipping compare_gradcam.png")
    else:
        fig, axes = plt.subplots(len(picks), 2 * n, figsize=(2.6 * 2 * n, 2.2 * len(picks)), squeeze=False,
                                 gridspec_kw={"wspace": 0.03, "hspace": 0.12})
        for i, e in enumerate(picks):
            obj = target_colour(e["task"])
            for j, (idx, z) in enumerate(cams):
                k = next(q for q, x in enumerate(idx) if (x["episode"], x["frame"]) == (e["episode"], e["frame"]))
                for m, cam in enumerate(("ego_view", "right_wrist")):
                    img = z[f"img_{k}_{cam}"]
                    contour = octopus_masks(img).get(obj) if cam == "ego_view" else None
                    ax = axes[i, 2 * j + m]
                    _show(ax, img, z[f"map_{k}_all_{cam}"], f"{names[j]}\n{cam}" if i == 0 else None, contour)
            axes[i, 0].set_ylabel(f"task {e['task_index']}\n{e['phase']}\nep {e['episode']} f{e['frame']}", fontsize=8)
        fig.suptitle("Grad-CAM on identical val frames (score: all joints; outline = instructed octopus)",
                     fontsize=11, fontweight="bold")
        fig.savefig(out.parent / "compare_gradcam.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
    print(f"Wrote {out} (+ compare_metrics.png, compare_gradcam.png)")


def main():
    global PROMPT_PREFIX, SHORT_HORIZON  # noqa: PLW0603
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", help="nanoVLA checkpoint (.pt), e.g. runs/baseline_siglip/best.pt")
    p.add_argument("--split", default="test", help="split of data_root to analyse (test / val)")
    p.add_argument("--data_root", default='/mnt/nas/rajat_ws/splitted_lerobot_and_school/', help="override the checkpoint config's data_root (parquet + meta)")
    p.add_argument("--video_root", default=None, help="override the checkpoint config's video_root (224x224 videos)")
    p.add_argument("--skip_train_check", action="store_true", help="skip the idle-episode scan of the train split")
    p.add_argument("--prompt_prefix", default="", help="text prepended to every instruction ('' = training prompts)")
    p.add_argument("--short_horizon", type=int, default=SHORT_HORIZON,
                   help="steps scored as the executed part of the chunk (accuracy 'first N steps')")
    p.add_argument("--include_idle", action="store_true", help="also analyse idle (no-grasp) episodes")
    p.add_argument("--episodes_per_task", type=int, default=3)
    p.add_argument("--frames_per_episode", type=int, default=8)
    p.add_argument("--n_noise", type=int, default=8, help="noise/time draws averaged in every loss")
    p.add_argument("--num_steps", type=int, default=10, help="denoising steps when sampling actions")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--compare", nargs="+", default=None, help="analysis dirs with summary.json to compare")
    p.add_argument("--compare_out", default="plots/gradcam/compare/compare.md")
    args = p.parse_args()
    PROMPT_PREFIX, SHORT_HORIZON = args.prompt_prefix, args.short_horizon
    if args.compare:
        compare(args.compare, args.compare_out)
    elif args.checkpoint:
        analyse(args)
    else:
        p.error("--checkpoint or --compare is required")


if __name__ == "__main__":
    main()
