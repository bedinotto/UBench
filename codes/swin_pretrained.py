"""Pretrained SwinV2 encoder + UNet decoder (M5; supersedes UB-17).

The hand-rolled shifted-window attention in ``swin_unet_plus_plus`` is broken
(no attention mask, no relative position bias — the shift is a no-op, UB-17).
Rather than repair hand-written attention (§11 forbids re-implementing what
timm provides correctly), this model uses timm's ImageNet-pretrained
**SwinV2-tiny** encoder (``swinv2_tiny_window8_256``, native 256×256 input,
window 8) behind a light U-Net-style convolutional decoder.

Single-channel thermal input is handled by timm's ``in_chans=1`` stem
adaptation, which sums the pretrained RGB kernels (M5 — see
``codes/pretrained_stem.py``). Downloading the ImageNet weights needs network
access, so construction is gated by a ``pretrained`` flag / ``UBENCH_PRETRAINED``
env var; tests and the CPU smoke build the identical architecture with random
weights (``pretrained=False``) and never hit the network.
"""

from __future__ import annotations

from typing import Optional

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from codes.model_registry import register_model
from codes.pretrained_stem import resolve_pretrained

# timm model id (native 256 input, window 8, ImageNet-1k). Verified available
# at session start via timm.list_models('swinv2*256*', pretrained=True).
SWIN_TIMM_MODEL = "swinv2_tiny_window8_256.ms_in1k"


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


class _Up(nn.Module):
    """Bilinear-upsample ×2, concatenate the encoder skip, then DoubleConv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = _DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


@register_model("swin_pretrained")
class SwinV2UNet(nn.Module):
    """ImageNet-pretrained SwinV2-tiny encoder + U-Net-style conv decoder."""

    def __init__(self, in_channels: int = 1, num_classes: int = 10,
                 img_size: int = 256, pretrained: Optional[bool] = None) -> None:
        super().__init__()
        if img_size != 256:
            raise ValueError(
                f"swin_pretrained uses swinv2_*_window8_256 (fixed 256 input); "
                f"got img_size={img_size}."
            )
        self.pretrained = resolve_pretrained(pretrained)
        self.encoder = timm.create_model(
            SWIN_TIMM_MODEL,
            pretrained=self.pretrained,
            in_chans=in_channels,
            features_only=True,
        )
        # Encoder feature-map channels at strides [4, 8, 16, 32].
        c0, c1, c2, c3 = self.encoder.feature_info.channels()
        self.up3 = _Up(c3, c2, c2)   # 8x8  -> 16x16 (+ skip f2)
        self.up2 = _Up(c2, c1, c1)   # 16x16 -> 32x32 (+ skip f1)
        self.up1 = _Up(c1, c0, c0)   # 32x32 -> 64x64 (+ skip f0)
        self.up_128 = _DoubleConv(c0, c0 // 2)      # after upsample to 128
        self.up_256 = _DoubleConv(c0 // 2, c0 // 4)  # after upsample to 256
        self.head = nn.Conv2d(c0 // 4, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # timm swin features_only yields NHWC maps; convert to NCHW.
        f0, f1, f2, f3 = (
            f.permute(0, 3, 1, 2).contiguous() for f in self.encoder(x)
        )
        d = self.up3(f3, f2)   # 16x16
        d = self.up2(d, f1)    # 32x32
        d = self.up1(d, f0)    # 64x64
        d = F.interpolate(d, scale_factor=2, mode="bilinear", align_corners=False)  # 128
        d = self.up_128(d)
        d = F.interpolate(d, scale_factor=2, mode="bilinear", align_corners=False)  # 256
        d = self.up_256(d)
        return self.head(d)


if __name__ == "__main__":  # pragma: no cover - offline package self-test (UB-21/T3.6)
    # Run as: python -m codes.swin_pretrained — builds with RANDOM weights
    # (pretrained=False, no network) and prints the forward-pass output shape.
    import os

    os.environ.setdefault("UBENCH_PRETRAINED", "0")
    from codes.model_registry import create_model

    _m = create_model("swin_pretrained", in_channels=1, num_classes=10, img_size=256, pretrained=False)
    print("swin_pretrained:", tuple(_m(torch.zeros(2, 1, 256, 256)).shape))
