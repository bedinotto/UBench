"""
UB-05: batch-size dicts keyed by canonical registry names, hard lookup.

The hardware profile's ``batch_sizes`` must use the model registry's keys
(``unet``, ``transunet``, ``swin_unet_plus_plus``) in every tier — CPU
included — so that training's per-model lookup can be a hard ``[key]``
access that raises ``KeyError`` instead of silently defaulting (R4/R5).

The parity test makes UB-05 structurally unrepeatable: registering a new
model (T3.2) forces a batch-size entry for it in every tier.
"""

from __future__ import annotations

import pytest

from codes.hardware_detector import HardwareProfile
from codes.model_registry import get_registered_models

# Importing the model modules runs their @register_model decorators,
# exactly as main_pipeline.py does.
import codes.unet_v2  # noqa: F401
import codes.transunet  # noqa: F401
import codes.swin_unet_plus_plus  # noqa: F401


def _profile(gpu_memory_gb: float, device: str = "cuda") -> HardwareProfile:
    """Simulate a hardware tier without any GPU present (pure Python)."""
    return HardwareProfile(
        gpu_name="Simulated GPU",
        gpu_memory_gb=gpu_memory_gb,
        cpu_count=8,
        ram_gb=16.0,
        os_type="Linux",
        device=device,
    )


# One representative VRAM value per branch of _calculate_batch_sizes.
TIERS = {
    "cpu": (0.0, "cpu"),
    "below-min-4gb": (4.0, "cuda"),
    "6gb-gtx1660ti": (6.0, "cuda"),
    "8gb": (8.0, "cuda"),
    "12gb": (12.0, "cuda"),
    "16gb": (16.0, "cuda"),
    "24gb-plus": (24.0, "cuda"),
}


def test_six_gb_tier_swin_batch_size():
    """Ledger reference value: the 6 GB (GTX 1660 Ti) tier gives Swin 6."""
    profile = _profile(6.0)
    assert profile.batch_sizes["swin_unet_plus_plus"] == 6


def test_below_min_tier_swin_batch_size():
    """Ledger reference value: the <5.5 GB tier gives Swin 3."""
    profile = _profile(4.0)
    assert profile.batch_sizes["swin_unet_plus_plus"] == 3


@pytest.mark.parametrize("tier", TIERS, ids=TIERS.keys())
def test_batch_size_keys_match_registry(tier):
    """Every tier's keys == the registered model keys (parity invariant)."""
    gpu_memory_gb, device = TIERS[tier]
    profile = _profile(gpu_memory_gb, device=device)
    assert set(profile.batch_sizes) == set(get_registered_models())


def test_unknown_key_raises_keyerror():
    """The retired short key must raise, never silently default (R4)."""
    profile = _profile(6.0)
    with pytest.raises(KeyError):
        profile.batch_sizes["swin"]
