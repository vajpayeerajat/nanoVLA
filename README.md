# nanoVLA

> **Branch `f3_early_fusion`** — F3 early fusion: **learned gate**. Each patch decides how much DINOv3 to add: `siglip + sigmoid(gate([siglip ; dino])) * proj(dino)`. The projection starts at zero (training starts from the SigLIP-only model) and the gate at 0.5. The mean gate is logged as `train/dino_gate`. DINOv3 ViT-B/16 (frozen) runs at 224 px on every camera, giving the same 14×14 patch grid as SigLIP. Its weights are gated on Hugging Face: accept the license on the model page and run `huggingface-cli login` once.

A small vision-language-action model for the Unitree G1 + Dex3 dataset (`splitted_lerobot_and_school`, 28-dim upper body, 3 cameras).

SigLIP-base (3 cameras × 49 tokens) + SmolLM2-135M (first 16 layers) → small transformer action expert trained with flow matching (pi0-style). It predicts 50-step action chunks: arm joints as deltas from the current state, hand joints as absolute targets. All images become tokens in `VisionEncoder` ([nanovla/model.py](nanovla/model.py)), so other encoders (e.g. DINOv2) and fusion methods plug in there.

## Setup

```bash
uv sync
```

Set `data_root` and `video_root` in [nanovla/config.py](nanovla/config.py), or pass them as `--data_root=... --video_root=...`.

## 1. Prepare data (once)

```bash
uv run scripts/prepare_data.py --jobs=14
```

Re-encodes every video to 224×224 with a keyframe every 10 frames, so training can read any frame quickly. It writes them to `video_root`. It also computes `assets/norm_stats.json`, unless that file already exists. openpi's `norm_stats.json` also works there.

## 2. Train

```bash
uv run train.py --run_name=baseline_siglip
uv run tensorboard --logdir runs
```

Quick smoke test first: `uv run train.py --run_name=smoke --max_steps=200 --val_interval=100 --val_batches=5 --log_interval=10`.

You can override any field in `nanovla/config.py` with `--key=value`. Checkpoints go to `runs/<run_name>/{best,last}.pt`.

## 3. Evaluate

```bash
# flow-matching loss + L1 error (rad) per joint group on the test split
uv run eval.py runs/baseline_siglip/best.pt --split=test --val_batches=100

# Grad-CAM + input-ablation report -> plots/gradcam/baseline_siglip__best/report.md
uv run rajat_gradcam_analysis.py --checkpoint runs/baseline_siglip/best.pt
uv run rajat_gradcam_analysis.py --compare plots/gradcam/<runA> plots/gradcam/<runB>
```

## Inference

```python
from nanovla.policy import Policy
policy = Policy.from_checkpoint("runs/baseline_siglip/best.pt")
actions = policy.infer({"ego_view": img, "ego_left": img, "ego_right": img,   # HxWx3 uint8, any size
                        "state": state28,                                     # or 43-dim whole body
                        "prompt": "Pick up the pink octopus and place it inside the orange basket."})
# -> [50, 28] absolute joint targets
```
