"""
UB-23: CPU execution profile behind the explicit UBENCH_ALLOW_CPU=1 opt-in.

Without the opt-in, hardware detection must keep its current behavior
(hard exit when CUDA is unavailable) — the CPU profile is never a hidden
fallback (R4).
"""

from __future__ import annotations

import pytest

from codes.hardware_detector import HardwareDetector


def _force_no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make torch report no CUDA regardless of the host machine."""
    monkeypatch.setattr(
        "codes.hardware_detector.torch.cuda.is_available", lambda: False
    )


def test_detect_returns_cpu_profile_when_opted_in(monkeypatch):
    """UBENCH_ALLOW_CPU=1 + no CUDA → complete CPU profile, no exit."""
    monkeypatch.setenv("UBENCH_ALLOW_CPU", "1")
    _force_no_cuda(monkeypatch)

    profile = HardwareDetector().detect()

    assert profile.device == "cpu"
    assert profile.num_workers == 0
    # Existing key scheme stays until T1.3 (UB-05) — do not rename here.
    assert set(profile.batch_sizes) == {"unet", "transunet", "swin"}
    assert all(v >= 1 for v in profile.batch_sizes.values())
    # Fields read elsewhere in the pipeline must all be populated.
    assert profile.gpu_name
    assert profile.cpu_count >= 1
    assert profile.ram_gb > 0
    # to_dict feeds hardware_profile.json — must not raise.
    assert profile.to_dict()["device"] == "cpu"


def test_detect_exits_without_optin(monkeypatch):
    """No opt-in + no CUDA → current behavior stands: sys.exit(1)."""
    monkeypatch.delenv("UBENCH_ALLOW_CPU", raising=False)
    _force_no_cuda(monkeypatch)

    with pytest.raises(SystemExit) as excinfo:
        HardwareDetector().detect()
    assert excinfo.value.code == 1
