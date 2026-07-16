"""
E2E smoke test — the merge gate (CLAUDE.md §7.3).

This test is the single definition of "the repo works end-to-end":
preprocessing runs automatically (T1.1), all 3 models train on the
synthetic fixture, checkpoints are saved and found through the single
naming authority (codes/naming.py, T1.2), and the benchmark CSV
contains all three models.

Scope note: green here means smoke-level E2E on synthetic CPU data.
Since T1.6 (UB-07) rc == 0 means every model × fold actually trained:
per-model failures are recorded, summarized, and fatal, so this test
also asserts the absence of a failure summary.  Partial-corpus fold
reduction (UB-03) is covered by test_splits.py.

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

# Paths — subprocess CWD is the fixture root, so we need absolute paths.
_REPO_ROOT = Path(__file__).parents[2]
_MAIN_PIPELINE = _REPO_ROOT / "codes" / "main_pipeline.py"


def run_pipeline_subprocess(
    models: list[str],
    timeout: int = 360,
    extra_args: list[str] | None = None,
) -> tuple[int, str]:
    """Invoke the real pipeline entry point in a subprocess.

    Runs ``codes/main_pipeline.py`` — the same script ``run.sh`` invokes —
    with ``UBENCH_ALLOW_CPU=1`` so hardware detection returns a CPU
    profile on machines without NVIDIA GPUs (UB-23).

    Parameters
    ----------
    models:
        Model keys to pass to ``--models``.
    timeout:
        Hard wall-clock timeout in seconds.  CI must comfortably finish
        all three models at 1 epoch on 64×64 synthetic data in this budget.
    extra_args:
        Additional CLI flags appended after ``--models`` (e.g.
        ``["--skip-benchmark"]``).

    Returns
    -------
    (returncode, combined_output)
        ``combined_output`` is stdout + stderr, truncated for display.
    """
    env = os.environ.copy()
    env["UBENCH_ALLOW_CPU"] = "1"  # UB-23: explicit CPU opt-in for the gate
    cmd = [sys.executable, str(_MAIN_PIPELINE), "--models"] + models + (extra_args or [])
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


def test_full_pipeline_smoke(synthetic_dataset, monkeypatch):
    """Full end-to-end pipeline smoke (UB-01/02/03/05/07 guard).

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

    # UB-07: rc == 0 must mean zero recorded failures — belt and braces.
    assert "TRAINING FAILURE" not in output
    assert "training failed" not in output

    # Guards: UB-01 (data loaded), UB-02 (all 3 model files found),
    #         UB-03 (GroupKFold did not crash), UB-05 (batch lookup OK),
    #         UB-07 (failures are fatal, so exit 0 means all trained)
    df = read_latest("outputs/*/benchmark_comparison.csv")
    assert set(df["Model"]) == {"U-Net", "TransUNet", "Swin-UNet++"}, (
        f"Expected all 3 models in benchmark CSV, got: {set(df['Model'])}"
    )
