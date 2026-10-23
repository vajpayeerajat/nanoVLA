"""nanoVLA: a pocket-sized pi0-style VLA.

    cameras ──► VisionEncoder (SigLIP) ──► pool ──► proj ─┐
                                                          ├─► SmolLM2 (first N layers) ──► context tokens
    prompt  ──► tokenizer ──► LM embeddings ──────────────┘                                     │
                                                                                         cross-attention
    state, noisy action chunk, flow time t ──► ActionExpert (small transformer) ◄───────────────┘
                                                     └──► velocity  (flow matching, as in pi0)

`VisionEncoder` is the single place that turns images into tokens, so other encoders (e.g. DINOv2) and
fusion schemes plug in there without touching the rest of the model.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForDepthEstimation, SiglipVisionModel


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class VisionEncoder(nn.Module):
    """[B, n_cams, 3, H, W] uint8 -> [B, n_cams * tokens_per_cam, out_dim].

    SigLIP patch tokens, plus (if `cfg.depth_model`) an embedding of the depth map of the same patch from a frozen
    monocular depth model, then average-pooled.
    """

    def __init__(self, cfg):
        super().__init__()
        self.siglip = SiglipVisionModel.from_pretrained(cfg.vision_model, dtype=torch.float32)
        self.pool = cfg.image_pool
        self.out_dim = self.siglip.config.hidden_size
        patch = self.siglip.config.patch_size
        grid = cfg.image_size // patch
        self.tokens_per_cam = (grid // self.pool) ** 2

        self.depth = None
        self.use_depth = True  # switched off by the Grad-CAM "-depth" ablation
        if cfg.depth_model:
            self.depth = AutoModelForDepthEstimation.from_pretrained(cfg.depth_model, dtype=torch.float32)
            self.depth.requires_grad_(False)
            self.depth_cams = [i for i, c in enumerate(cfg.cameras) if c in cfg.depth_cameras]
            assert self.depth_cams, f"depth_cameras {cfg.depth_cameras} not in cameras {cfg.cameras}"
            # One token per SigLIP patch: a patch x patch block of the depth map -> out_dim. Zero-initialised, so
            # training starts exactly from the SigLIP-only model.
            self.depth_embed = nn.Conv2d(1, self.out_dim, kernel_size=patch, stride=patch)
            nn.init.zeros_(self.depth_embed.weight)
            nn.init.zeros_(self.depth_embed.bias)
            self.register_buffer("depth_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
            self.register_buffer("depth_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def pretrained_modules(self):
        """Pretrained weights that train at lr * backbone_lr_scale (the depth model is frozen)."""
        return [self.siglip]

    def train(self, mode=True):
        super().train(mode)
        if self.depth is not None:
            self.depth.eval()  # frozen: always in inference mode
        return self

    @torch.no_grad()
    def depth_map(self, x):
        """x: [N, 3, H, W] in [0, 1] -> [N, 1, H, W] relative inverse depth, standardised per image over the real
        image pixels (Depth Anything's scale differs from image to image); 0 on the black letterbox bands."""
        d = self.depth(pixel_values=(x - self.depth_mean) / self.depth_std).predicted_depth.float()  # [N, H, W]
        if d.shape[-2:] != x.shape[-2:]:
            d = F.interpolate(d[:, None], size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        thr = 16 / 255  # letterbox bands are near-black after lossy re-encoding
        valid = (x.amax(dim=(1, 3)) > thr)[:, :, None] & (x.amax(dim=(1, 2)) > thr)[:, None, :]  # [N, H, W]
        n = valid.sum((1, 2), keepdim=True).clamp(min=1)
        mean = (d * valid).sum((1, 2), keepdim=True) / n
        std = (((d - mean) * valid) ** 2).sum((1, 2), keepdim=True).div(n).sqrt()
        return (((d - mean) / (std + 1e-6)) * valid)[:, None]

    def forward(self, images):
        b, n = images.shape[:2]
        x = images.flatten(0, 1).float() / 255.0
        tokens = self.siglip(pixel_values=(x - 0.5) / 0.5).last_hidden_state  # [B*n, grid*grid, D]
        if self.depth is not None and self.use_depth:
            cams = self.depth_cams
            d = self.depth_map(x.unflatten(0, (b, n))[:, cams].flatten(0, 1))  # [B*k, 1, H, W]
            emb = self.depth_embed(d).flatten(2).transpose(1, 2)  # [B*k, grid*grid, D]
            add = torch.zeros_like(tokens.unflatten(0, (b, n)))
            add[:, cams] = emb.unflatten(0, (b, len(cams))).to(add.dtype)
            tokens = tokens + add.flatten(0, 1)
        if self.pool > 1:
            g = int(math.isqrt(tokens.shape[1]))
            tokens = tokens.transpose(1, 2).unflatten(2, (g, g))
            tokens = F.avg_pool2d(tokens, self.pool).flatten(2).transpose(1, 2)
        return tokens.unflatten(0, (b, n)).flatten(1, 2)


def time_embedding(t, dim, max_period=4.0, min_period=4e-3):
    """Sine-cosine embedding of the flow time t in [0, 1] (same periods as pi0)."""
    frac = torch.linspace(0, 1, dim // 2, device=t.device)
    period = min_period * (max_period / min_period) ** frac
    angle = t[:, None] * (2 * math.pi / period[None])
    return torch.cat([angle.sin(), angle.cos()], -1)


class ExpertBlock(nn.Module):
    def __init__(self, dim, heads, ctx_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, kdim=ctx_dim, vdim=ctx_dim, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(approximate="tanh"), nn.Linear(4 * dim, dim))

    def forward(self, x, ctx, ctx_pad):
        h = self.norm1(x)
        x = x + self.self_attn(h, h, h, need_weights=False)[0]
        x = x + self.cross_attn(self.norm2(x), ctx, ctx, key_padding_mask=ctx_pad, need_weights=False)[0]
        return x + self.mlp(self.norm3(x))


class ActionExpert(nn.Module):
    """Tokens = [state, a_1..a_H]; the time embedding is mixed into every action token (pi0 style)."""

    def __init__(self, cfg, ctx_dim):
        super().__init__()
        d = cfg.expert_dim
        self.state_in = nn.Linear(cfg.state_dim, d)
        self.action_in = nn.Linear(cfg.action_dim, d)
        self.time_mlp = nn.Sequential(nn.Linear(2 * d, d), nn.SiLU(), nn.Linear(d, d))
        self.pos = nn.Parameter(torch.randn(1, cfg.action_horizon + 1, d) * 0.02)
        self.blocks = nn.ModuleList([ExpertBlock(d, cfg.expert_heads, ctx_dim) for _ in range(cfg.expert_layers)])
        self.norm = nn.LayerNorm(d)
        self.action_out = nn.Linear(d, cfg.action_dim)

    def forward(self, noisy_actions, state, t, ctx, ctx_pad):
        a = self.action_in(noisy_actions)
        te = time_embedding(t, a.shape[-1]).to(a.dtype)[:, None].expand_as(a)
        a = self.time_mlp(torch.cat([a, te], -1))
        x = torch.cat([self.state_in(state)[:, None], a], 1) + self.pos
        for block in self.blocks:
            x = block(x, ctx, ctx_pad)
        return self.action_out(self.norm(x[:, 1:]))


class NanoVLA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.vision = VisionEncoder(cfg)
        self.lm = AutoModel.from_pretrained(cfg.lm_model, dtype=torch.float32)
        if cfg.lm_num_layers:
            self.lm.layers = self.lm.layers[: cfg.lm_num_layers]
            self.lm.config.num_hidden_layers = cfg.lm_num_layers
        lm_dim = self.lm.config.hidden_size
        self.vision_proj = nn.Sequential(nn.Linear(self.vision.out_dim, lm_dim), nn.GELU(), nn.Linear(lm_dim, lm_dim))
        self.cam_embed = nn.Parameter(torch.zeros(len(cfg.cameras), 1, lm_dim))
        self.expert = ActionExpert(cfg, ctx_dim=lm_dim)
        if cfg.freeze_vision:
            self.vision.siglip.requires_grad_(False)

    def backbone_parameters(self):
        """Pretrained weights (trained at lr * backbone_lr_scale). New fusion layers are head parameters."""
        return [p for m in (*self.vision.pretrained_modules(), self.lm) for p in m.parameters() if p.requires_grad]

    def head_parameters(self):
        backbone = {id(p) for p in self.backbone_parameters()}
        return [p for p in self.parameters() if p.requires_grad and id(p) not in backbone]

    def embed_prefix(self, images, input_ids, attention_mask, image_mask=None):
        """LM input embeddings [B, L, D] = [camera tokens, prompt tokens] and their attention mask [B, L].

        image_mask: optional [B, n_cams] bool; False drops that camera's tokens from attention.
        """
        b, n = images.shape[:2]
        img = self.vision_proj(self.vision(images))
        img = (img.unflatten(1, (n, -1)) + self.cam_embed).flatten(1, 2)
        txt = self.lm.get_input_embeddings()(input_ids)
        img_mask = torch.ones(b, n, dtype=attention_mask.dtype, device=img.device) if image_mask is None else image_mask.to(attention_mask.dtype)
        img_mask = img_mask.repeat_interleave(img.shape[1] // n, dim=1)
        return torch.cat([img, txt], 1), torch.cat([img_mask, attention_mask], 1)

    def encode_embeds(self, embeds, mask):
        """Runs the LM. Returns context tokens [B, L, D] and a key padding mask [B, L] (True = ignore)."""
        ctx = self.lm(inputs_embeds=embeds, attention_mask=mask).last_hidden_state
        return ctx, mask == 0

    def encode(self, images, input_ids, attention_mask, image_mask=None):
        return self.encode_embeds(*self.embed_prefix(images, input_ids, attention_mask, image_mask))

    def forward(self, batch):
        """Flow-matching loss. x_t = t * noise + (1 - t) * actions, target velocity = noise - actions."""
        actions = batch["actions"]
        ctx, ctx_pad = self.encode(batch["images"], batch["input_ids"], batch["attention_mask"], batch.get("image_mask"))
        b = actions.shape[0]
        t = torch.distributions.Beta(1.5, 1.0).sample((b,)).to(actions.device) * 0.999 + 0.001
        noise = torch.randn_like(actions)
        x_t = t[:, None, None] * noise + (1 - t[:, None, None]) * actions
        v = self.expert(x_t, batch["state"], t, ctx, ctx_pad)
        return F.mse_loss(v.float(), (noise - actions).float())

    @torch.no_grad()
    def sample_actions(self, batch, num_steps=None, noise=None):
        """Euler-integrates the flow from noise (t=1) to actions (t=0). Returns normalized actions [B, H, A]."""
        num_steps = num_steps or self.cfg.num_inference_steps
        ctx, ctx_pad = self.encode(batch["images"], batch["input_ids"], batch["attention_mask"], batch.get("image_mask"))
        b = batch["state"].shape[0]
        x = torch.randn(b, self.cfg.action_horizon, self.cfg.action_dim, device=ctx.device) if noise is None else noise
        dt = -1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((b,), 1.0 + i * dt, device=ctx.device)
            x = x + dt * self.expert(x, batch["state"], t, ctx, ctx_pad).float()
        return x
