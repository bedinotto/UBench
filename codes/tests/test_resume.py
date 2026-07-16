"""
UB-06: --resume must reuse a previous run's directories and checkpoints.

Before T1.7, every ``main_pipeline.py`` invocation minted a fresh timestamp,
so ``UnifiedTrainer._find_latest_checkpoint()`` always scanned an empty
``outputs/<new-ts>/checkpoints`` directory — the trainer's complete resume
machinery (checkpoint save/load with full metric history) was dead across
restarts.

These tests drive the real entry point in a subprocess (same as the smoke
test) in two phases on the synthetic fixture (T1.7 AC):

* phase A trains 1 epoch and saves an epoch checkpoint;
* phase B with ``--resume <run_id>`` and ``NUM_EPOCHS=2`` reuses the same
  run dirs, resumes from the checkpoint, and **continues** the metric
  history to length 2 — without creating a second ``outputs/<ts>`` dir;
* ``outputs/latest`` resolves to the run dir;
* ``--resume`` with an unknown run id fails with an actionable error.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from codes.tests.test_pipeline_smoke import run_pipeline_subprocess


def _set_fast_env(monkeypatch, num_epochs: int) -> None:
    monkeypatch.setenv("LIMIT_SAMPLES", "20")
    monkeypatch.setenv("NUM_EPOCHS", str(num_epochs))
    monkeypatch.setenv("K_FOLDS", "2")
    monkeypatch.setenv("NUM_WORKERS", "0")


def _run_id_from(output: str) -> str:
    match = re.search(r"Run ID:\s+(\S+)", output)
    assert match, f"pipeline banner did not print a Run ID\n{output[:2000]}"
    return match.group(1)


def _run_dirs() -> set[str]:
    """Names of real run directories under outputs/ (symlinks excluded)."""
    return {
        p.name
        for p in Path("outputs").iterdir()
        if p.is_dir() and not p.is_symlink()
    }


def test_resume_continues_metric_history(synthetic_dataset, monkeypatch):
    """Epoch 1 run + --resume with NUM_EPOCHS=2 → history length 2 (T1.7 AC)."""
    monkeypatch.chdir(synthetic_dataset)

    # Phase A: one epoch, one model — leaves an epoch checkpoint behind.
    _set_fast_env(monkeypatch, num_epochs=1)
    rc, output = run_pipeline_subprocess(
        models=["unet"], extra_args=["--skip-benchmark"]
    )
    assert rc == 0, f"--- PHASE A OUTPUT TAIL ---\n{output[-3000:]}"
    run_id = _run_id_from(output)

    ckpt = Path("outputs") / run_id / "checkpoints" / "unet_fold_1_epoch_0000.pth"
    assert ckpt.exists(), f"phase A left no epoch checkpoint at {ckpt}"

    dirs_before = _run_dirs()

    # Phase B: same run id, two epochs — must resume, not restart.
    _set_fast_env(monkeypatch, num_epochs=2)
    rc, output = run_pipeline_subprocess(
        models=["unet"], extra_args=["--skip-benchmark", "--resume", run_id]
    )
    assert rc == 0, f"--- PHASE B OUTPUT TAIL ---\n{output[-3000:]}"

    # Resume went through the existing checkpoint discovery and said so.
    assert "Resuming from checkpoint" in output
    assert "unet_fold_1_epoch_0000.pth" in output

    # Same run identity: no second outputs/<ts> directory was minted.
    assert _run_id_from(output) == run_id
    assert _run_dirs() == dirs_before, (
        "resume created a new run dir instead of reusing the old one"
    )

    # The substance: metric history CONTINUED — epoch 1 (phase A) + epoch 2
    # (phase B) → length 2 in the saved metrics JSON, for both folds.
    for fold in (1, 2):
        # _safe_filename("unet_Fold-1") → "unet_fold_1"
        metrics_path = Path("logs") / run_id / f"unet_fold_{fold}_metrics.json"
        assert metrics_path.exists(), f"missing metrics JSON: {metrics_path}"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert len(metrics["train_losses"]) == 2, (
            f"fold {fold}: expected 2-epoch history after resume, got "
            f"{len(metrics['train_losses'])}: {metrics['train_losses']}"
        )
        assert len(metrics["val_losses"]) == 2

    # outputs/latest points at the (resumed) run dir.
    latest = Path("outputs") / "latest"
    assert latest.is_symlink(), "outputs/latest symlink was not created"
    assert latest.resolve() == (Path("outputs") / run_id).resolve()


def test_resume_unknown_run_id_fails_actionably(synthetic_dataset, monkeypatch):
    """--resume with a nonexistent run id → non-zero exit + actionable error."""
    monkeypatch.chdir(synthetic_dataset)
    _set_fast_env(monkeypatch, num_epochs=1)

    rc, output = run_pipeline_subprocess(
        models=["unet"],
        extra_args=["--skip-benchmark", "--resume", "no-such-run"],
    )

    assert rc != 0
    assert "no-such-run" in output, (
        f"error does not name the missing run id\n{output[-2000:]}"
    )
