"""Shared 1-channel stem adaptation for pretrained (RGB) encoders (M5).

Pretrained ImageNet encoders expect 3-channel RGB input; thermal frames are
single-channel. The M5 policy adapts the pretrained stem by **summing the RGB
input kernels into one channel** rather than re-initialising it, so the
pretrained filters keep responding (a grayscale input has R=G=B, and summing
the three kernels reproduces that filter's activation). timm applies exactly
this when a model is built with ``in_chans=1`` (its ``adapt_input_conv``); this
module exposes the operation on its own so it can be unit-tested and referenced
from the pretrained-load verification.
"""

from __future__ import annotations

import os
from typing import Optional

import torch


def resolve_pretrained(pretrained: Optional[bool]) -> bool:
    """Resolve the pretrained flag shared by the pretrained encoders (M5).

    An explicit ``pretrained`` value wins; otherwise the ``UBENCH_PRETRAINED``
    env var decides. Default is ``True`` (real runs load ImageNet weights);
    tests and the CPU smoke pass ``pretrained=False`` or set
    ``UBENCH_PRETRAINED=0`` to build the identical architecture with random
    weights and never touch the network.
    """
    if pretrained is not None:
        return bool(pretrained)
    return os.environ.get("UBENCH_PRETRAINED", "1").strip() not in ("0", "false", "False", "")


def sum_rgb_kernels(weight: torch.Tensor) -> torch.Tensor:
    """Sum a pretrained conv's 3 RGB input-channel kernels into 1 channel.

    Args:
        weight: convolution weight of shape ``(out_ch, 3, kh, kw)``.

    Returns:
        A weight of shape ``(out_ch, 1, kh, kw)`` — the per-output-channel sum
        over the three input channels.

    Raises:
        ValueError: if ``weight`` is not a 4-D tensor with 3 input channels.
    """
    if weight.dim() != 4 or weight.shape[1] != 3:
        raise ValueError(
            f"expected an (out_ch, 3, kh, kw) RGB conv weight, "
            f"got shape {tuple(weight.shape)}"
        )
    return weight.sum(dim=1, keepdim=True)
