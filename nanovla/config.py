"""All knobs in one place. Override any field from the command line, e.g. `--batch_size=16 --lr=3e-4`."""

import dataclasses
import sys


@dataclasses.dataclass
class Config:
    # --- data ---
    data_root: str = "/mnt/nas/rajat_ws/splitted_lerobot_and_school/"
    # Videos re-encoded to 224x224 by scripts/prepare_data.py (fast random access). Same layout as data_root.
    video_root: str = "/home/rajat/work/Datasets/splitted_lerobot_and_school_224"
    norm_stats: str = "assets/norm_stats.json"
    cameras: tuple = ("ego_view", "ego_left", "ego_right")
    image_size: int = 224
    action_horizon: int = 50  # 2.5 s at 20 fps, same as the openpi pi05_g1_dex3 config.
    state_dim: int = 28
    action_dim: int = 28
    # Arms are predicted as deltas from the current state, hands as absolute targets.
    delta_arms: bool = True
    max_text_len: int = 256
    augment: bool = True

    # --- model ---
    vision_model: str = "google/siglip-base-patch16-224"
    lm_model: str = "HuggingFaceTB/SmolLM2-135M"
    lm_num_layers: int = 16  # keep only the first N LM layers (SmolVLA-style); 0 = all
    image_pool: int = 2  # average-pool the 14x14 patch grid by this factor -> 49 tokens per camera
    freeze_vision: bool = False  # freezes SigLIP only (fusion layers stay trainable)
    # Second vision encoder fused with SigLIP patch by patch ("" = SigLIP only). DINOv3 B/16 at 224 gives the same
    # 14x14 grid as SigLIP B/16, so patch i of both encoders covers the same image region.
    dino_model: str = ""
    freeze_dino: bool = True
    expert_dim: int = 512
    expert_layers: int = 6
    expert_heads: int = 8
    num_inference_steps: int = 10

    # --- training ---
    run_name: str = "nanovla"
    out_dir: str = "runs"
    batch_size: int = 96
    num_workers: int = 12
    lr: float = 3e-4
    backbone_lr_scale: float = 0.1  # pretrained vision/LM weights train at lr * scale
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    max_steps: int = 20_000
    grad_clip: float = 1.0
    log_interval: int = 50
    val_interval: int = 250
    val_batches: int = 20
    save_interval: int = 500
    save_dir: str = "/mnt/nas/rajat_ws/checkpoints/nanoVLA"  # checkpoints go to <save_dir>/<run_name>/
    compile: bool = False
    seed: int = 0
    resume: str = ""  # path to a checkpoint to resume from


def parse_args(argv=None) -> Config:
    """nanoGPT-style overrides: `--field=value`. Values are parsed with the type of the default."""
    cfg = Config()
    for arg in sys.argv[1:] if argv is None else argv:
        assert arg.startswith("--") and "=" in arg, f"expected --key=value, got {arg}"
        key, val = arg[2:].split("=", 1)
        assert hasattr(cfg, key), f"unknown config key: {key}"
        default = getattr(cfg, key)
        if isinstance(default, bool):
            val = val.lower() in ("1", "true", "yes")
        elif isinstance(default, tuple):
            val = tuple(v for v in val.split(",") if v)
        else:
            val = type(default)(val)
        setattr(cfg, key, val)
    return cfg
