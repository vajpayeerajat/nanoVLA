"""One-time data preparation.

1. Re-encodes every camera video to 224x224 (aspect kept, padded with black like openpi's resize_with_pad)
   with a keyframe every 10 frames, so the training loader can seek to any frame cheaply. The source videos
   have a 250-frame GOP at 640x480, which makes random access ~50x slower.
2. Computes mean/std of the state and of the (delta-arm) action chunks on the train split -> cfg.norm_stats,
   unless that file already exists (e.g. the openpi norm_stats.json, which holds the same quantities).

Usage: uv run scripts/prepare_data.py [--skip_videos=1] [--jobs=14] plus any Config override.
"""

import glob
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nanovla.config import parse_args
from nanovla.data import action_chunk, load_split


def reencode(args):
    src, dst, size = args
    if os.path.exists(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = f"{dst}.{os.getpid()}.tmp.mp4"  # unique per process, so concurrent runs cannot clobber each other
    vf = f"scale={size}:{size}:force_original_aspect_ratio=decrease,pad={size}:{size}:(ow-iw)/2:(oh-ih)/2"
    cmd = ["ffmpeg", "-v", "error", "-y", "-threads", "1", "-i", src, "-vf", vf, "-c:v", "libx264",
           "-preset", "fast", "-crf", "18", "-g", "10", "-bf", "0", "-pix_fmt", "yuv420p", "-an", tmp]  # fmt: skip
    subprocess.run(cmd, check=True)
    os.replace(tmp, dst)


def main():
    extra = {}
    argv = []
    for a in sys.argv[1:]:
        key = a[2:].split("=", 1)[0]
        if key in ("skip_videos", "jobs"):
            extra[key] = int(a.split("=", 1)[1])
        else:
            argv.append(a)
    cfg = parse_args(argv)

    if not extra.get("skip_videos"):
        srcs = sorted(glob.glob(os.path.join(cfg.data_root, "*/videos/chunk-*/*/*.mp4")))
        jobs = [(s, os.path.join(cfg.video_root, os.path.relpath(s, cfg.data_root)), cfg.image_size) for s in srcs]
        print(f"re-encoding {len(jobs)} videos -> {cfg.video_root}")
        with ProcessPoolExecutor(extra.get("jobs", os.cpu_count() - 2)) as pool:
            list(tqdm(pool.map(reencode, jobs, chunksize=4), total=len(jobs)))

    if os.path.exists(cfg.norm_stats):
        print(f"{cfg.norm_stats} exists, not recomputing norm stats")
        return
    split = load_split(cfg.data_root, "train")
    # Every 5th frame is plenty for the action stats (chunks of neighbouring frames overlap heavily anyway).
    chunks = [
        action_chunk(split, ep_i, t, cfg.action_horizon, cfg.delta_arms)[1]
        for ep_i, n in enumerate(split["lengths"])
        for t in range(0, n, 5)
    ]
    actions = np.concatenate(chunks)
    stats = {
        "state": {"mean": split["state"].mean(0).tolist(), "std": split["state"].std(0).tolist()},
        "action": {"mean": actions.mean(0).tolist(), "std": actions.std(0).tolist()},
        "_meta": {"action_horizon": cfg.action_horizon, "delta_arms": cfg.delta_arms, "frames": int(len(split["state"]))},
    }
    os.makedirs(os.path.dirname(cfg.norm_stats) or ".", exist_ok=True)
    with open(cfg.norm_stats, "w") as f:
        json.dump(stats, f, indent=1)
    print(f"wrote {cfg.norm_stats}")


if __name__ == "__main__":
    main()
