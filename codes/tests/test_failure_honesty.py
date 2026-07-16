"""
UB-07: training failures must be recorded, reported, and fatal.

Before T1.6, the per-model/fold ``try/except`` in ``train_all_models``
printed one line and continued: a SUCCESS banner and exit 0 were possible
with zero trained models, and ``main()``'s error-log writer was unreachable
because ``Pipeline()`` was constructed outside its ``try``.

These tests drive the real entry point in a subprocess (same as the smoke
test) with ``UBENCH_INJECT_FAIL=<model_key>`` — a test-only hook in
``Pipeline.train_model`` that raises for the named model and is inert
without the env var — and assert the honest behavior (T1.6 AC):

* injected failure → exit 1, a failure summary naming model + fold, and an
  ``error_log_*.txt`` with the traceback in the run's log dir;
* ``--fail-fast`` → the run aborts on the first failure (no later folds);
* a ``Pipeline()`` constructor crash → the FATAL-ERROR path writes the
  error log instead of dying as a bare traceback.
"""

from __future__ import annotations

import re
from pathlib import Path

from codes.tests.test_pipeline_smoke import run_pipeline_subprocess


def _set_fast_env(monkeypatch) -> None:
    monkeypatch.setenv("LIMIT_SAMPLES", "20")
    monkeypatch.setenv("NUM_EPOCHS", "1")
    monkeypatch.setenv("K_FOLDS", "2")
    monkeypatch.setenv("NUM_WORKERS", "0")


def _error_logs_in_run_dir(output: str) -> list[Path]:
    """error_log_*.txt inside the run's own log dir, parsed from the banner
    so error logs of other runs in the shared fixture can't interfere."""
    match = re.search(r"Log dir:\s+(\S+)", output)
    assert match, f"pipeline banner did not print a log dir\n{output[:2000]}"
    return sorted(Path(match.group(1)).glob("error_log_*.txt"))


def test_injected_failure_exits_nonzero_with_error_log(
    synthetic_dataset, monkeypatch
):
    """Injected per-model failure → exit 1 + named summary + error log.

    Without ``--fail-fast`` the loop still visits every fold (that behavior
    is kept), so both folds appear in the summary — but the run must no
    longer end in a SUCCESS banner with exit 0.
    """
    monkeypatch.chdir(synthetic_dataset)
    _set_fast_env(monkeypatch)
    monkeypatch.setenv("UBENCH_INJECT_FAIL", "unet")

    rc, output = run_pipeline_subprocess(
        models=["unet"], extra_args=["--skip-benchmark"]
    )

    assert rc == 1, f"--- OUTPUT TAIL ---\n{output[-3000:]}"
    assert "PIPELINE FINISHED WITH 2 TRAINING FAILURE(S)" in output
    assert "unet fold 1" in output and "unet fold 2" in output
    assert "STARTING FOLD 2/2" in output  # no fail-fast: loop continued
    assert "PIPELINE COMPLETED SUCCESSFULLY" not in output

    error_logs = _error_logs_in_run_dir(output)
    assert error_logs, "no error_log_*.txt written in the run's log dir"
    content = error_logs[-1].read_text(encoding="utf-8")
    assert "Injected failure for unet" in content
    assert "RuntimeError" in content


def test_fail_fast_aborts_on_first_failure(synthetic_dataset, monkeypatch):
    """--fail-fast: the first failure aborts the run — no later folds."""
    monkeypatch.chdir(synthetic_dataset)
    _set_fast_env(monkeypatch)
    monkeypatch.setenv("UBENCH_INJECT_FAIL", "unet")

    rc, output = run_pipeline_subprocess(
        models=["unet"], extra_args=["--skip-benchmark", "--fail-fast"]
    )

    assert rc == 1, f"--- OUTPUT TAIL ---\n{output[-3000:]}"
    assert "STARTING FOLD 2" not in output
    assert "aborting on first failure" in output
    assert "PIPELINE FINISHED WITH 1 TRAINING FAILURE(S)" in output
    assert "PIPELINE COMPLETED SUCCESSFULLY" not in output


def test_constructor_crash_writes_error_log(tmp_path, monkeypatch):
    """A Pipeline() constructor crash reaches main()'s error-log writer.

    An empty cwd has no ``data/`` directory, so ``Config`` raises inside
    ``Pipeline.__init__``; with the construction inside ``main()``'s try,
    the FATAL-ERROR path must write ``error_log_*.txt`` and exit non-zero
    instead of dying as a bare traceback.
    """
    monkeypatch.chdir(tmp_path)

    rc, output = run_pipeline_subprocess(
        models=["unet"], extra_args=["--skip-benchmark"]
    )

    assert rc != 0
    assert "FATAL ERROR" in output
    error_logs = list(tmp_path.glob("**/error_log_*.txt"))
    assert error_logs, "constructor crash did not produce an error log"
    assert "FileNotFoundError" in error_logs[-1].read_text(encoding="utf-8")
