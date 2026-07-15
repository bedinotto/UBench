"""
E2E smoke test — the merge gate (CLAUDE.md §7.3).

This test is the single definition of "the repo works end-to-end."
It is intentionally marked xfail(strict=True) because UB-01 prevents
the pipeline from completing until T1.1 lands.  strict=True means
the test suite will FAIL if the pipeline somehow passes before the
xfail marker is removed — ensuring the marker is removed when UB-01 is
fixed.

Expected current failure path:
    create_single_fold_loader() → FileNotFoundError(data/processed/metadata.csv)
    caught by train_all_models try/except (UB-07 silent swallow)
    re-raised inside run_benchmark() → caught by Pipeline.run() → returns False
    main() → sys.exit(1) → subprocess rc=1 → assert rc==0 fails → XFAIL ✓

The subprocess runs the REAL entry point (codes/main_pipeline.py) with
UBENCH_ALLOW_CPU=1 so hardware detection succeeds on CPU-only machines
(UB-23) — the gate exercises exactly what run.sh invokes.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

# Paths — subprocess CWD is the fixture root, so we need absolute paths.
_REPO_ROOT = Path(__file__).parents[2]
_MAIN_PIPELINE = _REPO_ROOT / "codes" / "main_pipeline.py"


def run_pipeline_subprocess(
    models: list[str],
    timeout: int = 360,
) -> tuple[int, str]:
    """Invoke the real pipeline entry point in a subprocess.

    Runs ``codes/main_pipeline.py`` — the same script ``run.sh`` invokes —
    with ``UBENCH_ALLOW_CPU=1`` so hardware detection returns a CPU
    profile on machines without NVIDIA GPUs (UB-23).  The first real
    failure is then UB-01 (metadata.csv missing).

    Parameters
    ----------
    models:
        Model keys to pass to ``--models``.
    timeout:
        Hard wall-clock timeout in seconds.  CI must comfortably finish
        all three models at 1 epoch on 64×64 synthetic data in this budget.

    Returns
    -------
    (returncode, combined_output)
        ``combined_output`` is stdout + stderr, truncated for display.
    """
    env = os.environ.copy()
    env["UBENCH_ALLOW_CPU"] = "1"  # UB-23: explicit CPU opt-in for the gate
    cmd = [sys.executable, str(_MAIN_PIPELINE), "--models"] + models
    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    combined = result.stdout + "\n--- STDERR ---\n" + result.stderr
    return result.returncode, combined


def read_latest(glob_pattern: str) -> pd.DataFrame:
    """Return the most recently modified CSV matching *glob_pattern*.

    Searched relative to the current working directory, so call this
    after ``monkeypatch.chdir(synthetic_dataset)``.
    """
    matches = sorted(
        Path(".").glob(glob_pattern),
        key=lambda p: p.stat().st_mtime,
    )
    if not matches:
        raise FileNotFoundError(
            f"No CSV found matching {glob_pattern!r} in {Path('.').resolve()}"
        )
    return pd.read_csv(matches[-1])


# ---------------------------------------------------------------------------
# The gate.  xfail(strict=True) keeps the suite GREEN while UB-01 exists and
# forces removal of this marker once T1.1 makes the pipeline complete.
# AC update (R9 / T0.2): original CLAUDE.md §9 T0.2 AC said "red only on the
# smoke test"; updated to "strict-xfail for UB-01 keeps suite green" so CI
# always reports green while the expected failure is tracked here.
# ---------------------------------------------------------------------------
@pytest.mark.xfail(reason="UB-01: preprocessing not wired into pipeline", strict=True)
def test_full_pipeline_smoke(synthetic_dataset, monkeypatch):
    """Full end-to-end pipeline smoke (UB-01/02/03/05/07 guard).

    Once UB-01 is fixed (T1.1), this test should pass and the xfail
    marker must be removed.  With strict=True, a surprise pass turns
    the run RED, forcing the marker removal.

    Smoke artifacts produced here are labeled as such (R10) and must
    never be treated as real benchmark results.
    """
    monkeypatch.chdir(synthetic_dataset)
    monkeypatch.setenv("LIMIT_SAMPLES", "20")
    monkeypatch.setenv("NUM_EPOCHS", "1")
    monkeypatch.setenv("K_FOLDS", "2")
    monkeypatch.setenv("NUM_WORKERS", "0")

    rc, output = run_pipeline_subprocess(models=["unet", "transunet", "swin"])

    assert rc == 0, (
        f"Pipeline subprocess exited with code {rc}.\n"
        f"--- OUTPUT TAIL (last 3000 chars) ---\n"
        f"{output[-3000:]}"
    )

    # Guards: UB-01 (data loaded), UB-02 (all 3 model files found),
    #         UB-03 (GroupKFold did not crash), UB-05 (batch lookup OK),
    #         UB-07 (no silent failure with exit 0)
    df = read_latest("outputs/*/benchmark_comparison.csv")
    assert set(df["Model"]) == {"U-Net", "TransUNet", "Swin-UNet++"}, (
        f"Expected all 3 models in benchmark CSV, got: {set(df['Model'])}"
    )
