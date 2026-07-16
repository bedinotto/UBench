"""
UB-13 + UB-14: the --epochs flag must always be honored, and one run
identity must exist per ./run.sh invocation.

Before T1.8:

* ``--epochs 100`` passed explicitly was silently dropped by the
  ``if args.epochs != 100`` guard in ``main()`` (UB-13);
* ``run.sh`` minted its own timestamp while ``main_pipeline.py`` minted a
  second one, so every ``./run.sh`` run produced two ``logs/<ts>`` dirs and
  the shell's final "Full console log" message pointed at the wrong one
  (UB-14).

The fix: ``--epochs`` uses ``default=None`` and is exported to
``NUM_EPOCHS`` whenever given; ``run.sh`` exports ``UBENCH_RUN_ID`` and
``Pipeline.__init__`` reuses it (``--resume`` takes precedence).  run.sh's
export is shell code, so it is asserted grep-level here and exercised for
real in the session report transcript.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from codes.tests.test_pipeline_smoke import run_pipeline_subprocess

_REPO_ROOT = Path(__file__).parents[2]


def _set_fast_env(monkeypatch, num_epochs: int | None) -> None:
    monkeypatch.setenv("LIMIT_SAMPLES", "20")
    monkeypatch.setenv("K_FOLDS", "2")
    monkeypatch.setenv("NUM_WORKERS", "0")
    if num_epochs is None:
        monkeypatch.delenv("NUM_EPOCHS", raising=False)
    else:
        monkeypatch.setenv("NUM_EPOCHS", str(num_epochs))


def _run_dirs() -> set[str]:
    """Names of real run directories under outputs/ (symlinks excluded)."""
    if not Path("outputs").is_dir():
        return set()
    return {
        p.name
        for p in Path("outputs").iterdir()
        if p.is_dir() and not p.is_symlink()
    }


def test_ubench_run_id_reused_and_resume_precedence(
    synthetic_dataset, monkeypatch
):
    """UBENCH_RUN_ID names the run's dirs; --resume overrides it (UB-14).

    Phase 2 documents the precedence contract: with both UBENCH_RUN_ID and
    --resume present, --resume wins and no dir is created for the env id.
    """
    monkeypatch.chdir(synthetic_dataset)
    _set_fast_env(monkeypatch, num_epochs=1)
    monkeypatch.setenv("UBENCH_RUN_ID", "fixed-test-id")

    dirs_before = _run_dirs()
    rc, output = run_pipeline_subprocess(
        models=["unet"], extra_args=["--skip-benchmark"]
    )
    assert rc == 0, f"--- OUTPUT TAIL ---\n{output[-3000:]}"

    assert re.search(r"Run ID:\s+fixed-test-id", output), (
        "pipeline did not adopt UBENCH_RUN_ID as its run id"
    )
    assert (Path("outputs") / "fixed-test-id").is_dir()
    assert (Path("logs") / "fixed-test-id").is_dir()
    # The UB-14 contract: the run id from the env is the ONLY run identity —
    # no second timestamp-named run dir is minted.  (The first pipeline run
    # also side-creates outputs/{models,plots,predictions} at top level via
    # preprocess_all_data()'s default Config() — see ledger UB-24 — so the
    # assertion targets timestamp-shaped dirs, not the whole delta.)
    new_dirs = _run_dirs() - dirs_before
    assert "fixed-test-id" in new_dirs
    ts_shaped = [
        d for d in new_dirs
        if re.match(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$", d)
    ]
    assert not ts_shaped, (
        f"pipeline minted its own timestamped run dir(s) {ts_shaped} "
        "despite UBENCH_RUN_ID being set"
    )

    # Phase 2: --resume takes precedence over UBENCH_RUN_ID.
    monkeypatch.setenv("UBENCH_RUN_ID", "other-id")
    rc, output = run_pipeline_subprocess(
        models=["unet"],
        extra_args=["--skip-benchmark", "--resume", "fixed-test-id"],
    )
    assert rc == 0, f"--- OUTPUT TAIL ---\n{output[-3000:]}"
    assert re.search(r"Run ID:\s+fixed-test-id", output)
    assert not (Path("outputs") / "other-id").exists(), (
        "--resume must take precedence over UBENCH_RUN_ID"
    )


def test_epochs_flag_honored_end_to_end(synthetic_dataset, monkeypatch):
    """--epochs 2 with NUM_EPOCHS unset → exactly 2 epochs trained (UB-13 AC)."""
    monkeypatch.chdir(synthetic_dataset)
    _set_fast_env(monkeypatch, num_epochs=None)
    monkeypatch.delenv("UBENCH_RUN_ID", raising=False)

    rc, output = run_pipeline_subprocess(
        models=["unet"], extra_args=["--skip-benchmark", "--epochs", "2"]
    )
    assert rc == 0, f"--- OUTPUT TAIL ---\n{output[-3000:]}"
    assert "Epoch 2/2" in output

    match = re.search(r"Run ID:\s+(\S+)", output)
    assert match, "pipeline banner did not print a Run ID"
    metrics_path = (
        Path("logs") / match.group(1) / "unet_fold_1_metrics.json"
    )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert len(metrics["train_losses"]) == 2


def test_epochs_100_sets_env(monkeypatch):
    """The exact value the old ``if args.epochs != 100`` guard dropped (UB-13)."""
    from codes.main_pipeline import apply_epochs_override

    monkeypatch.delenv("NUM_EPOCHS", raising=False)
    apply_epochs_override(100)
    import os
    assert os.environ["NUM_EPOCHS"] == "100"


def test_epochs_absent_leaves_env_untouched(monkeypatch):
    """No flag → no NUM_EPOCHS export; config stays the authority (UB-13)."""
    from codes.main_pipeline import apply_epochs_override

    monkeypatch.delenv("NUM_EPOCHS", raising=False)
    apply_epochs_override(None)
    import os
    assert "NUM_EPOCHS" not in os.environ


def test_run_sh_exports_run_id():
    """run.sh must export its timestamp as UBENCH_RUN_ID (UB-14, grep-level)."""
    content = (_REPO_ROOT / "run.sh").read_text(encoding="utf-8")
    assert "export UBENCH_RUN_ID" in content
