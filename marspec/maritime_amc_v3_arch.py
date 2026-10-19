#!/usr/bin/env python3
"""
MaritimeAMC v3.1 — Multi-Modal AMC with Context-Aware Equalization + MoE
========================================================================

Improvements over the first v3 (driven by an architecture review + Paper 1's
finding that random global phase rotation from 2-ray multipath is the dominant
AMC degradation):

  1. Context-conditioned equalizer (FiLM): the maritime context [dist, bft,
     sst, speed] modulates the equalizer's feature maps, so it cleans harder
     under storms than calm seas. Residual scale is now LEARNABLE (was 0.1).
  2. FiLM context-gating fusion (replaces a degenerate 1-token cross-attention
     whose softmax over a single key was always 1.0 -> effectively q+v). Context
     now produces a feature-wise affine (gamma,beta) over the signal embedding.
  3. MCformer attention pooling over ALL patch tokens (was CLS-token only) ->
     richer global feature.
  4. Squeeze-Excitation channel attention in the CNN branches.
  5. MoE channel routing (unchanged) + load-balance loss in the trainer.

PhaseAug (global random phase rotation) is applied in the TRAINING channel
(train_maritime_v3.py / notebook), directly teaching phase-invariance — the
single most important robustness lever per Paper 1.

External interface is UNCHANGED so the trainer/notebook need no edits:
    forward(iq[B,2,1024], context[B,4]) -> (logits[B,24], iq_eq[B,2,1024], gate[B,E])

Run standalone for the parameter breakdown + forward test:
    python marspec_poc/maritime_amc_v3_arch.py
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Default v3 config (user-scaled to ~24M)
# ---------------------------------------------------------------------------
CFG_V3 = dict(
    # v3.2 — scaled for MARITIME ROBUSTNESS (clean was already 80%; the gap is
    # robustness, retention 59%). Capacity shifted to the TRANSFORMER, which
    # Paper 1 found most resilient to the dominant phase-rotation degradation.
    # ChannelEqualizer — strong (undoes 2-ray multipath) but no longer dominant
    eq_filters     = 384,
    eq_layers      = 16,
    # CNN multi-scale feature extractor (wider)
    cnn_channels   = [96, 192, 384],
    cnn_pool       = 16,
    # MCformer — BIG (head dim = 256/8 = 32; deep) for phase-robust features
    mcf_d_model    = 256,
    mcf_nhead      = 16,
    mcf_nlayers    = 15,
    mcf_patch_len  = 32,
    # Context encoder — modest (input is only 4 numbers)
    ctx_hidden     = 512,
    ctx_mlp_hidden = 512,
    ctx_layers     = 3,
    # Fusion + MoE
    fusion_d       = 384,
    n_experts      = 12,
    expert_hidden  = 1024,
    moe_dropout    = 0.1,
    drop_path      = 0.1,
    grad_checkpoint = True,    # bigger model -> save VRAM
    num_classes    = 24,
)


def validate_config(cfg: Dict) -> None:
    """Fail loudly on edits that break an architectural constraint."""
    nhead, d_model = cfg["mcf_nhead"], cfg["mcf_d_model"]
    fusion, patch = cfg["fusion_d"], cfg["mcf_patch_len"]
    errs = []
    if d_model % nhead != 0:
        errs.append(f"mcf_d_model ({d_model}) must be divisible by mcf_nhead ({nhead}).")
    if fusion % 2 != 0:
        errs.append(f"fusion_d ({fusion}) must be even (split into two halves).")
    if 1024 % patch != 0:
        errs.append(f"mcf_patch_len ({patch}) must divide 1024 (e.g. 8,16,32,64).")
    if errs:
        raise ValueError("Invalid CFG_V3:\n  - " + "\n  - ".join(errs))


# ---------------------------------------------------------------------------
# Squeeze-Excitation channel attention (1D)
# ---------------------------------------------------------------------------
class SE1d(nn.Module):
    """Recalibrates channel importance: [B,C,L] -> [B,C,L]."""

    def __init__(self, c: int, r: int = 8):
        super().__init__()
        h = max(c // r, 4)
        self.fc = nn.Sequential(nn.Linear(c, h), nn.GELU(), nn.Linear(h, c), nn.Sigmoid())

    def forward(self, x):
        s = self.fc(x.mean(-1)).unsqueeze(-1)   # [B,C,1]
        return x * s


# ---------------------------------------------------------------------------
# 1. Context-conditioned Channel Equalizer (FiLM + learnable residual)
# ---------------------------------------------------------------------------
class ChannelEqualizer(nn.Module):
    """
    Residual dilated CNN mapping distorted IQ -> cleaned IQ. The maritime
    context modulates the mid-stack feature maps (FiLM), so correction strength
    adapts to the channel regime. [B,2,1024] -> [B,2,1024].
    """

    def __init__(self, n_filters: int = 64, n_layers: int = 8, ctx_dim: int = 0):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            dilation = 2 ** (i % 6)
            self.layers.append(nn.Sequential(
                nn.Conv1d(2 if i == 0 else n_filters, n_filters,
                          kernel_size=3, padding=dilation, dilation=dilation),
                nn.BatchNorm1d(n_filters),
                nn.GELU(),
            ))
        self.out_proj = nn.Conv1d(n_filters, 2, kernel_size=1)
        self.res_scale = nn.Parameter(torch.tensor(0.1))   # learnable (was fixed 0.1)
        self.film_at = n_layers // 2
        self.film = nn.Linear(ctx_dim, 2 * n_filters) if ctx_dim > 0 else None
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, ctx_feat=None):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if self.film is not None and ctx_feat is not None and i == self.film_at:
                gamma, beta = self.film(ctx_feat).chunk(2, dim=-1)
                h = h * (1 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)   # FiLM
        return x + self.res_scale * self.out_proj(h)


# ---------------------------------------------------------------------------
# 2. CNN1D Multi-Scale Extractor (+ Squeeze-Excitation)
# ---------------------------------------------------------------------------
class CNN1DExtractor(nn.Module):
    """Multi-scale parallel CNN branches (k=3,7,11) with SE. [B,2,1024]->[B,out_dim]."""

    def __init__(self, channels=(64, 128, 256), out_dim=256, pool=4):
        super().__init__()
        self.branch3 = self._branch(channels, 3)
        self.branch7 = self._branch(channels, 7)
        self.branch11 = self._branch(channels, 11)
        self.pool = nn.AdaptiveAvgPool1d(pool)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1] * 3 * pool, out_dim),
            nn.LayerNorm(out_dim), nn.GELU(),
        )

    @staticmethod
    def _branch(channels, ksize):
        layers, in_c = [], 2
        for out_c in channels:
            layers += [
                nn.Conv1d(in_c, out_c, ksize, padding=ksize // 2),
                nn.BatchNorm1d(out_c), nn.GELU(),
                SE1d(out_c),                 # channel attention
                nn.MaxPool1d(2),
            ]
            in_c = out_c
        return nn.Sequential(*layers)

    def forward(self, x):
        f = torch.cat([self.branch3(x), self.branch7(x), self.branch11(x)], dim=1)
        return self.fc(self.pool(f))


# ---------------------------------------------------------------------------
# 3. MCformer Encoder (attention pooling over all patch tokens)
# ---------------------------------------------------------------------------
class MCformerEncoder(nn.Module):
    """Patch Transformer over IQ with attention pooling. [B,2,1024]->[B,out_dim]."""

    def __init__(self, d_model=128, nhead=8, nlayers=8, patch_len=16, out_dim=256,
                 grad_checkpoint=False, drop_path=0.0):
        super().__init__()
        n_patches = 1024 // patch_len
        self.patch_len = patch_len
        self.grad_checkpoint = grad_checkpoint
        self.drop_path = drop_path        # LayerDrop probability (stochastic depth)
        self.patch_embed = nn.Linear(2 * patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model * 4,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc, nlayers)
        # Attention pooling: a learnable query attends over all patch tokens.
        self.pool_q = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, out_dim)

    def forward(self, x):
        B = x.size(0)
        p = self.patch_len
        patches = x.reshape(B, 2, -1, p).permute(0, 2, 1, 3).reshape(B, -1, 2 * p)
        tokens = self.patch_embed(patches) + self.pos_embed
        ckpt = None
        if self.grad_checkpoint and self.training:
            from torch.utils.checkpoint import checkpoint as ckpt
        for layer in self.transformer.layers:
            # LayerDrop (stochastic depth): randomly skip a layer during training.
            # Each layer is residual, so skipping == identity.
            if self.training and self.drop_path > 0 and torch.rand(()).item() < self.drop_path:
                continue
            tokens = ckpt(layer, tokens, use_reentrant=False) if ckpt else layer(tokens)
        q = self.pool_q.expand(B, -1, -1)
        pooled, _ = self.pool_attn(q, tokens, tokens)        # [B,1,d]
        return self.out_proj(self.norm(pooled.squeeze(1)))


# ---------------------------------------------------------------------------
# 4. Context Encoder
# ---------------------------------------------------------------------------
class ContextEncoder(nn.Module):
    """
    Deep residual MLP for the [dist, bft, sst, speed] context.
    `mlp_hidden`/`n_layers` control its size (intentionally large per request);
    `out_dim` is the downstream context dim used by equalizer/fusion/MoE.
    Residual blocks keep a big MLP trainable despite the tiny 4-dim input.
    """

    def __init__(self, in_dim=4, mlp_hidden=512, out_dim=256, n_layers=6):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(in_dim, mlp_hidden),
                                 nn.LayerNorm(mlp_hidden), nn.GELU())
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(mlp_hidden, mlp_hidden),
                          nn.LayerNorm(mlp_hidden), nn.GELU())
            for _ in range(n_layers)
        ])
        self.out = nn.Sequential(nn.Linear(mlp_hidden, out_dim), nn.LayerNorm(out_dim))

    def forward(self, ctx):
        h = self.inp(ctx)
        for blk in self.blocks:
            h = h + blk(h)              # residual — large MLP stays trainable
        return self.out(h)


# ---------------------------------------------------------------------------
# 5. Context-Gated Fusion (FiLM) — replaces degenerate 1-token cross-attention
# ---------------------------------------------------------------------------
class ContextGatedFusion(nn.Module):
    """
    Context produces a feature-wise affine (gamma, beta) over the signal
    embedding — true conditioning, unlike the old single-token attention whose
    softmax over one key was always 1.0.
    """

    def __init__(self, sig_dim=512, ctx_dim=128, fusion_dim=512):
        super().__init__()
        self.sig_proj = nn.Linear(sig_dim, fusion_dim)
        self.film = nn.Linear(ctx_dim, 2 * fusion_dim)
        self.norm1 = nn.LayerNorm(fusion_dim)
        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2), nn.GELU(),
            nn.Linear(fusion_dim * 2, fusion_dim),
        )
        self.norm2 = nn.LayerNorm(fusion_dim)

    def forward(self, sig_feat, ctx_feat):
        s = self.sig_proj(sig_feat)
        gamma, beta = self.film(ctx_feat).chunk(2, dim=-1)
        s = self.norm1(s * (1 + gamma) + beta)
        return self.norm2(s + self.ffn(s))


# ---------------------------------------------------------------------------
# 6. Mixture-of-Experts Channel Routing
# ---------------------------------------------------------------------------
class MoEChannelRouter(nn.Module):
    """Soft mixture of experts routed by maritime context."""

    def __init__(self, dim=512, ctx_dim=128, n_experts=4, hidden=512, dropout=0.1):
        super().__init__()
        self.n_experts = n_experts
        self.gate = nn.Sequential(
            nn.Linear(ctx_dim, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, n_experts),
        )
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden, dim),
            ) for _ in range(n_experts)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, ctx_feat):
        gate_w = F.softmax(self.gate(ctx_feat), dim=-1)
        expert_out = torch.stack([e(x) for e in self.experts], 1)
        mixed = (gate_w.unsqueeze(-1) * expert_out).sum(1)
        return self.norm(x + mixed), gate_w


# ---------------------------------------------------------------------------
# 7. Full MaritimeAMC v3.1
# ---------------------------------------------------------------------------
class MaritimeAMCv3(nn.Module):
    """
    forward(iq[B,2,1024], context[B,4]) ->
        logits[B,num_classes], iq_eq[B,2,1024], gate_weights[B,n_experts]
    """

    def __init__(self, cfg: Dict = CFG_V3):
        super().__init__()
        validate_config(cfg)
        fusion_d = cfg["fusion_d"]
        half_d = fusion_d // 2
        ctx_h = cfg["ctx_hidden"]

        self.ctx_encoder = ContextEncoder(
            4, mlp_hidden=cfg.get("ctx_mlp_hidden", 512),
            out_dim=ctx_h, n_layers=cfg.get("ctx_layers", 6))
        self.equalizer = ChannelEqualizer(cfg["eq_filters"], cfg["eq_layers"], ctx_dim=ctx_h)
        self.cnn_extractor = CNN1DExtractor(tuple(cfg["cnn_channels"]), out_dim=half_d,
                                            pool=cfg.get("cnn_pool", 4))
        self.mcf_encoder = MCformerEncoder(
            d_model=cfg["mcf_d_model"], nhead=cfg["mcf_nhead"],
            nlayers=cfg["mcf_nlayers"], patch_len=cfg["mcf_patch_len"], out_dim=half_d,
            grad_checkpoint=cfg.get("grad_checkpoint", False),
            drop_path=cfg.get("drop_path", 0.0),
        )
        self.fusion = ContextGatedFusion(sig_dim=fusion_d, ctx_dim=ctx_h, fusion_dim=fusion_d)
        self.moe = MoEChannelRouter(
            dim=fusion_d, ctx_dim=ctx_h, n_experts=cfg["n_experts"],
            hidden=cfg["expert_hidden"], dropout=cfg["moe_dropout"],
        )
        self.classifier = nn.Sequential(
            nn.Linear(fusion_d, fusion_d), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(fusion_d, cfg["num_classes"]),
        )

    def forward(self, iq, context):
        f_ctx = self.ctx_encoder(context)              # encode context first
        iq_eq = self.equalizer(iq, f_ctx)              # context-conditioned cleaning
        f_sig = torch.cat([self.cnn_extractor(iq_eq), self.mcf_encoder(iq_eq)], dim=-1)
        fused = self.fusion(f_sig, f_ctx)
        routed, gate_w = self.moe(fused, f_ctx)
        return self.classifier(routed), iq_eq, gate_w


# ---------------------------------------------------------------------------
# Standalone: parameter breakdown + forward test
# ---------------------------------------------------------------------------
def _count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def main() -> None:
    import argparse, copy
    ap = argparse.ArgumentParser(
        description="MaritimeAMC v3.1 parameter breakdown. Override any config knob.")
    for k, v in CFG_V3.items():
        if k == "cnn_channels":
            ap.add_argument("--cnn_channels", type=int, nargs="+", default=None)
        elif isinstance(v, float):
            ap.add_argument(f"--{k}", type=float, default=None)
        else:
            ap.add_argument(f"--{k}", type=int, default=None)
    args = ap.parse_args()

    cfg = copy.deepcopy(CFG_V3)
    for k in cfg:
        val = getattr(args, k, None)
        if val is not None:
            cfg[k] = val

    model = MaritimeAMCv3(cfg)
    parts = {
        "Context encoder": model.ctx_encoder,
        "ChannelEqualizer": model.equalizer,
        "CNN1D + SE": model.cnn_extractor,
        "MCformer + attnpool": model.mcf_encoder,
        "FiLM fusion": model.fusion,
        "MoE router": model.moe,
        "Classifier": model.classifier,
    }
    total = _count(model)
    print("=" * 58)
    print("  MaritimeAMC v3.1 — Parameter Breakdown")
    print("=" * 58)
    for name, mod in parts.items():
        n = _count(mod)
        print(f"  {name:<22} {n:>10,}  ({n/total*100:4.1f}%)")
    print("-" * 58)
    print(f"  {'TOTAL':<22} {total:>10,}")
    print(f"  {'v2 baseline':<22} {1_530_138:>10,}  ({total/1_530_138:.2f}x)")
    print("=" * 58)

    model.eval()
    with torch.no_grad():
        logits, iq_eq, gate = model(torch.randn(4, 2, 1024), torch.rand(4, 4))
    print(f"\n  Forward OK: logits={tuple(logits.shape)}, "
          f"iq_eq={tuple(iq_eq.shape)}, gate={tuple(gate.shape)}")
    print(f"  Gate sums to 1: {torch.allclose(gate.sum(-1), torch.ones(4))}")
    print(f"  Learnable eq res_scale init: {model.equalizer.res_scale.item():.3f}")


if __name__ == "__main__":
    main()
