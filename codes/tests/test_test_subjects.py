"""Held-out test-subject tests (M1, T2.4).

Subjects listed in ``config.test_subjects`` are excluded from ALL CV folds and
evaluated separately. These tests are DataFrame/unit-level (no subprocess — the
smoke covers the end-to-end CV+TEST path); they fabricate a tiny ``metadata.csv``
and never read image bytes, so no fold loader is iterated.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from codes.unified_data import (
    create_kfold_data_loaders,
    create_test_loader,
    load_split_metadata,
    load_test_metadata,
    resolve_fold_count,
)


def _write_metadata(tmp_path, subjects, frames_per: int = 3) -> None:
    """Fabricate data/processed/metadata.csv for the given subjects."""
    proc = tmp_path / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in subjects:
        for i in range(frames_per):
            rows.append({
                "dataset": s,
                "sample_id": f"{s}/R{i}",
                "image_path": f"data/processed/images/{s}_{i}.npy",
                "mask_path": f"data/processed/masks/{s}_{i}.png",
            })
    pd.DataFrame(rows).to_csv(proc / "metadata.csv", index=False)
    # T3.4: the load guard requires a current-version manifest next to metadata.
    from codes.preprocess_manifest import write_preprocess_manifest
    write_preprocess_manifest(proc, image_size=(256, 256),
                              normalization="fixed_range",
                              fixed_range_celsius=(20.0, 40.0), num_samples=len(rows))


def _cfg(tmp_path, subjects, test_subjects=(), k_folds: int = 2):
    _write_metadata(tmp_path, subjects)
    from codes.config_schema import PreprocessingConfig
    return SimpleNamespace(
        PROCESSED_DIR=tmp_path / "processed",
        DATA_DIR=tmp_path / "data",
        TEST_SUBJECTS=list(test_subjects),
        K_FOLDS=k_folds,
        NUM_CLASSES=10,
        RANDOM_SEED=42,   # loaders build a seeded generator from this (T2.5)
        REGION_NAMES=["background", "Contorno inferior do Rosto",
                      "Sombrancelha esquerda", "Sombrancelha direita", "Nariz",
                      "Olho esquerdo", "Olho direito", "Boca", "Labios", "Testa"],
        PREPROCESSING=PreprocessingConfig(),
    )


def _subject_of(sample_id: str) -> str:
    return sample_id.split("/")[0]


def test_test_subjects_excluded_from_every_fold(tmp_path) -> None:
    """No fold's train or val split may contain a held-out test subject."""
    cfg = _cfg(tmp_path, ["S1", "S2", "S3", "S4"], test_subjects=["S4"])

    # CV pool excludes the test subject...
    cv_df = load_split_metadata(cfg)
    assert "S4" not in set(cv_df["dataset"])
    assert set(cv_df["dataset"]) == {"S1", "S2", "S3"}

    # ...and every produced fold is clean too (the real production split path).
    folds, _ = create_kfold_data_loaders(cfg, batch_size=2, num_workers=0)
    for fold in folds:
        subjects_in_fold = {_subject_of(s) for s in fold["train_ids"] + fold["val_ids"]}
        assert "S4" not in subjects_in_fold


def test_held_out_loader_is_exactly_the_test_subjects(tmp_path) -> None:
    """The test loader contains exactly the configured test subjects, nothing else."""
    cfg = _cfg(tmp_path, ["S1", "S2", "S3", "S4"], test_subjects=["S3", "S4"])

    test_df = load_test_metadata(cfg)
    assert set(test_df["dataset"]) == {"S3", "S4"}

    loader = create_test_loader(cfg, batch_size=2, num_workers=0)
    assert loader is not None
    assert set(loader.dataset.metadata["dataset"]) == {"S3", "S4"}


def test_too_few_cv_subjects_after_holdout_raises(tmp_path) -> None:
    """Reserving subjects that leave <2 in the CV pool raises the >=2 guard."""
    cfg = _cfg(tmp_path, ["S1", "S2", "S3"], test_subjects=["S2", "S3"])

    cv_df = load_split_metadata(cfg)
    assert set(cv_df["dataset"]) == {"S1"}          # only one CV subject remains
    with pytest.raises(ValueError, match=r">=2 subject"):
        resolve_fold_count(cfg.K_FOLDS, cv_df["dataset"])


def test_absent_test_subject_raises(tmp_path) -> None:
    """A test subject not present in the data is a hard error, not a silent no-op."""
    cfg = _cfg(tmp_path, ["S1", "S2", "S3"], test_subjects=["S9"])
    with pytest.raises(ValueError, match=r"not present in the discovered data"):
        load_split_metadata(cfg)


def test_empty_default_is_unchanged(tmp_path) -> None:
    """With no test subjects: full CV pool, empty test set, no test loader."""
    cfg = _cfg(tmp_path, ["S1", "S2", "S3"], test_subjects=[])

    cv_df = load_split_metadata(cfg)
    assert set(cv_df["dataset"]) == {"S1", "S2", "S3"}   # nothing withheld
    assert load_test_metadata(cfg).empty
    assert create_test_loader(cfg, batch_size=2, num_workers=0) is None
