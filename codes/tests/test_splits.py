"""
UB-03/UB-04: fold-count guard with leave-subjects-out semantics.

``GroupKFold(groups=df['dataset'])`` holds out whole subject datasets per
fold, so it can never produce more splits than there are subjects.
``resolve_fold_count`` is the single authority both loader factories and
the pipeline use: it reduces K to the subject count with a loud warning,
and raises an actionable error (not sklearn's cryptic ``ValueError``)
when fewer than 2 subjects are present.
"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from codes.unified_data import resolve_fold_count


def _metadata_frame(n_subjects: int, frames_per_subject: int = 4) -> pd.DataFrame:
    """Tiny stand-in for metadata.csv: sample_id + dataset columns only."""
    rows = [
        {"sample_id": f"S{s}/R{s}00{i}", "dataset": f"S{s}"}
        for s in range(1, n_subjects + 1)
        for i in range(1, frames_per_subject + 1)
    ]
    return pd.DataFrame(rows)


def test_three_subjects_reduce_k_with_warning():
    """3 subjects, K=5 → effective_k == 3, loud warning naming both."""
    df = _metadata_frame(3)
    with pytest.warns(UserWarning, match=r"K_FOLDS=5.*3.*subject"):
        effective_k = resolve_fold_count(5, df["dataset"])
    assert effective_k == 3


def test_two_subjects_reduce_k():
    """2 subjects, K=5 → effective_k == 2 (the minimum viable CV)."""
    df = _metadata_frame(2)
    with pytest.warns(UserWarning):
        effective_k = resolve_fold_count(5, df["dataset"])
    assert effective_k == 2


def test_one_subject_raises_actionable_error():
    """1 subject → clear leave-subjects-out error, not sklearn's."""
    df = _metadata_frame(1)
    with pytest.raises(ValueError, match=r"[Ll]eave-subjects-out.*2 subject"):
        resolve_fold_count(5, df["dataset"])


def test_k_within_subject_count_passes_through_silently():
    """K <= n_groups → requested K unchanged, no warning emitted."""
    df = _metadata_frame(5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert resolve_fold_count(5, df["dataset"]) == 5
        assert resolve_fold_count(2, df["dataset"]) == 2


@pytest.mark.parametrize("n_subjects", [2, 3, 5])
def test_subject_exclusivity_across_folds(n_subjects):
    """No subject appears in both train and val of any produced fold."""
    df = _metadata_frame(n_subjects)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # reduction covered above
        effective_k = resolve_fold_count(5, df["dataset"])
    gkf = GroupKFold(n_splits=effective_k)
    for train_idx, val_idx in gkf.split(df, groups=df["dataset"]):
        train_subjects = set(df.iloc[train_idx]["dataset"])
        val_subjects = set(df.iloc[val_idx]["dataset"])
        assert train_subjects & val_subjects == set()
        assert val_subjects  # every fold holds out at least one subject


def test_pipeline_runs_partial_corpus(synthetic_dataset, monkeypatch):
    """Integration: 3 subjects, default K=5 → pipeline completes with
    effective K=3 (the UB-03 partial-corpus crash scenario).

    LIMIT_SAMPLES=12 keeps exactly S1–S3 (metadata rows are grouped by
    subject in sorted discovery order, 4 frames each).  With UB-07 still
    open, rc == 0 alone is weak evidence, so the output must also show
    the reduction warning, fold 3 training, and no swallowed failures.
    """
    from codes.tests.test_pipeline_smoke import run_pipeline_subprocess

    monkeypatch.chdir(synthetic_dataset)
    monkeypatch.setenv("LIMIT_SAMPLES", "12")
    monkeypatch.setenv("NUM_EPOCHS", "1")
    monkeypatch.setenv("K_FOLDS", "5")
    monkeypatch.setenv("NUM_WORKERS", "0")

    rc, output = run_pipeline_subprocess(models=["unet"], extra_args=["--skip-benchmark"])

    assert rc == 0, f"--- OUTPUT TAIL ---\n{output[-3000:]}"
    assert "reduced to 3" in output
    assert "STARTING FOLD 3/3" in output
    assert "STARTING FOLD 4" not in output
    assert "training failed" not in output
