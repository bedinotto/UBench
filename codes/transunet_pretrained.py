"""Pretrained R50+ViT-B/16 hybrid encoder + upsampling decoder (M5; UB-16).

``transunet.py`` trains a ~100M-parameter ViT-B from scratch on ~1.8k images
(UB-16) — a known-degenerate setup that invalidates any fairness claim. This
model instead uses timm's ImageNet-pretrained **R50+ViT-B/16 hybrid**
(``vit_base_r50_s16_224``, a ResNet-50 stem feeding ViT-B blocks) as the
encoder, with a progressive-upsampling convolutional decoder over the 1/16
token map.

Shipped construction: **option (a)** from the T3.2 plan — the hybrid encoder's
``forward_features`` token map (no explicit CNN skip connections; the decoder
upsamples the single 1/16 stage). This is the minimum honest pretrained
variant; a skip-connected variant (option b) is deferred.

Single-channel thermal input uses timm's ``in_chans=1`` stem adaptation, which
sums the pretrained RGB kernels (M5 — see ``codes/pretrained_stem.py``).
Weight download is gated by ``pretrained`` / ``UBENCH_PRETRAINED``; tests and
the CPU smoke build the identical architecture with random weights.
"""

from __future__ import annotations

from typing import Optional

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from codes.model_registry import register_model
from codes.pretrained_stem import resolve_pretrained

# timm hybrid model id (ResNet-50 stem + ViT-B/16). Verified available at
# session start via timm.list_models('*vit_base_r50*', pretrained=True).
VIT_TIMM_MODEL = "vit_base_r50_s16_224.orig_in21k"


class _DoubleConv(nn.Module):
    """Conv-BN-ReLU ×2."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _UpConv(nn.Module):
    """Bilinear-upsample ×2 then DoubleConv (no skip — option (a))."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = _DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv(x)


@register_model("transunet_pretrained")
class TransUNetPretrained(nn.Module):
    """ImageNet-pretrained R50+ViT-B/16 hybrid encoder + upsampling decoder."""

    def __init__(self, in_channels: int = 1, num_classes: int = 10,
                 img_size: int = 256, pretrained: Optional[bool] = None) -> None:
        super().__init__()
        self.img_size = img_size
        self.pretrained = resolve_pretrained(pretrained)
        self.encoder = timm.create_model(
            VIT_TIMM_MODEL,
            pretrained=self.pretrained,
            in_chans=in_channels,
            img_size=img_size,
            num_classes=0,
            global_pool="",
        )
        self.num_prefix = self.encoder.num_prefix_tokens
        embed = self.encoder.embed_dim

        # Token map is at stride 16 (16x16 for 256 input). Upsample 16 -> 256.
        self.proj = nn.Conv2d(embed, 256, kernel_size=1)
        self.up1 = _UpConv(256, 128)   # 16 -> 32
        self.up2 = _UpConv(128, 64)    # 32 -> 64
        self.up3 = _UpConv(64, 32)     # 64 -> 128
        self.up4 = _UpConv(32, 16)     # 128 -> 256
        self.head = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder.forward_features(x)      # (B, num_prefix + N, C)
        tokens = tokens[:, self.num_prefix:, :]        # drop cls/prefix -> (B, N, C)
        b, n, c = tokens.shape
        side = int(round(n ** 0.5))
        if side * side != n:
            raise ValueError(f"non-square token grid: {n} tokens")
        # (B, N, C) -> (B, C, side, side)
        grid = tokens.transpose(1, 2).reshape(b, c, side, side)
        d = self.proj(grid)
        d = self.up1(d)
        d = self.up2(d)
        d = self.up3(d)
        d = self.up4(d)
        return self.head(d)


if __name__ == "__main__":  # pragma: no cover - offline package self-test (UB-21/T3.6)
    # Run as: python -m codes.transunet_pretrained — builds with RANDOM weights
    # (pretrained=False, no network) and prints the forward-pass output shape.
    import os

    os.environ.setdefault("UBENCH_PRETRAINED", "0")
    from codes.model_registry import create_model

    _m = create_model("transunet_pretrained", in_channels=1, num_classes=10, img_size=256, pretrained=False)
    print("transunet_pretrained:", tuple(_m(torch.zeros(2, 1, 256, 256)).shape))
