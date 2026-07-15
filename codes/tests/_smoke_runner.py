"""
Test-only subprocess entry point for the smoke test.

Patches ``detect_and_optimize`` to return a CPU-compatible ``HardwareProfile``
so that the pipeline can proceed past hardware detection on CPU-only machines
(CI, dev workstations without NVIDIA GPUs).  The first real failure is then
UB-01 (metadata.csv missing), which is what the smoke test verifies.

This module is TEST INFRASTRUCTURE ONLY — it must never be imported by
any production pipeline code.  It is invoked as a subprocess by
``test_pipeline_smoke.run_pipeline_subprocess()``.

Usage (set by the smoke test):
    python -m codes.tests._smoke_runner --models unet transunet swin
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

# Ensure the repo root is on sys.path so ``codes.*`` imports resolve.
_REPO_ROOT = Path(__file__).parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from codes.hardware_detector import HardwareProfile  # noqa: E402


def _cpu_profile() -> HardwareProfile:
    """Return a minimal CPU-compatible HardwareProfile.

    gpu_memory_gb=8 puts us in the "8 GB" tier so all batch-size keys
    (unet/transunet/swin) resolve to non-trivial values.  The profile is
    never used for real GPU operations — it only satisfies Pipeline.__init__.
    """
    import os
    import psutil

    cpu_count = os.cpu_count() or 4
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    return HardwareProfile(
        gpu_name="CPU (test mode)",
        gpu_memory_gb=8.0,
        cpu_count=cpu_count,
        ram_gb=ram_gb,
    )


if __name__ == "__main__":
    # Patch detect_and_optimize so Pipeline.__init__ skips GPU probing,
    # then hand control to the real pipeline main().
    with patch("codes.hardware_detector.detect_and_optimize", return_value=_cpu_profile()):
        from codes.main_pipeline import main  # noqa: E402 (inside guard)
        main()
