"""Reproducibility tests (UB-20a, M6, T2.5).

Two seeded runs must produce identical randomness. The key subtlety (why a
NUM_WORKERS=0 test is not enough): `worker_init_fn` is only invoked when
`num_workers > 0`, so a 0-worker test would pass even if the per-worker seeding
were broken. Hence:

* a direct unit test of `seed_worker` (deterministic per worker id, distinct
  across ids), and
* a **num_workers=2** behavioral test showing the augmentation-style random
  stream is reproducible across two identically-seeded loaders *and* that the
  seeding is load-bearing (unseeded spawned workers diverge).
"""

from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import codes.main_pipeline as main_pipeline
from codes.unified_data import create_single_fold_loader, seed_worker


# --------------------------------------------------------------------------- #
# Direct unit test of seed_worker (reliable; the guarantee).
# --------------------------------------------------------------------------- #
def test_seed_worker_deterministic_and_distinct() -> None:
    """seed_worker gives the same seed for a given worker id, different across ids."""
    def draws_for(worker_id: int):
        torch.manual_seed(999)          # fix torch.initial_seed() base
        seed_worker(worker_id)
        return np.random.rand(), random.random()

    assert draws_for(0) == draws_for(0)     # deterministic for a worker id
    assert draws_for(0) != draws_for(1)     # distinct across worker ids


# --------------------------------------------------------------------------- #
# Behavioral test at num_workers=2 (exercises worker_init_fn).
# --------------------------------------------------------------------------- #
class _NoisyDataset(Dataset):
    """Each item draws from numpy — i.e. depends on the worker's numpy RNG."""

    def __len__(self) -> int:
        return 8

    def __getitem__(self, idx: int):
        return torch.tensor([np.random.rand()], dtype=torch.float64)


def _stream(*, generator, num_workers: int = 2):
    loader = DataLoader(
        _NoisyDataset(), batch_size=2, shuffle=True, num_workers=num_workers,
        generator=generator, worker_init_fn=seed_worker,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )
    return torch.cat(list(loader)).flatten().tolist()


def test_explicit_generator_makes_stream_global_rng_independent() -> None:
    """A per-loader seeded generator makes the 2-worker stream reproducible
    regardless of global RNG state — the robustness the fix adds.

    (Modern torch already auto-seeds a worker's numpy from the base seed, so the
    seeded ``generator`` — not merely ``seed_worker`` — is what pins the stream
    independent of whatever consumed the global RNG earlier in the run.)
    """
    def seeded(seed):
        g = torch.Generator()
        g.manual_seed(seed)
        return g

    # Explicit seeded generator: perturbing the global torch RNG has no effect.
    torch.manual_seed(1)
    a = _stream(generator=seeded(42))
    torch.manual_seed(2)
    b = _stream(generator=seeded(42))
    assert a == b

    # No explicit generator: shuffle draws its base seed from the global RNG, so
    # the same perturbation changes the stream (the pre-fix fragility).
    torch.manual_seed(1)
    c = _stream(generator=None)
    torch.manual_seed(2)
    d = _stream(generator=None)
    assert c != d


# --------------------------------------------------------------------------- #
# The real loader factory wires generator + worker_init_fn.
# --------------------------------------------------------------------------- #
def _cfg_with_metadata(tmp_path, subjects=("S1", "S2", "S3")):
    proc = tmp_path / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    rows = [
        {"dataset": s, "sample_id": f"{s}/R{i}",
         "image_path": f"data/processed/images/{s}_{i}.npy",
         "mask_path": f"data/processed/masks/{s}_{i}.png"}
        for s in subjects for i in range(3)
    ]
    pd.DataFrame(rows).to_csv(proc / "metadata.csv", index=False)
    # T3.4: the load guard requires a current-version manifest next to metadata.
    from codes.preprocess_manifest import write_preprocess_manifest
    write_preprocess_manifest(proc, image_size=(256, 256),
                              normalization="fixed_range",
                              fixed_range_celsius=(20.0, 40.0), num_samples=len(rows))
    return SimpleNamespace(
        PROCESSED_DIR=proc, DATA_DIR=tmp_path / "data",
        TEST_SUBJECTS=[], K_FOLDS=2, NUM_CLASSES=10, RANDOM_SEED=42,
    )


def test_real_loaders_carry_generator_and_worker_init(tmp_path) -> None:
    """create_single_fold_loader sets a seeded generator and seed_worker (no iteration)."""
    cfg = _cfg_with_metadata(tmp_path)
    loaders = create_single_fold_loader(cfg, fold_idx=0, batch_size=2, num_workers=0)
    for key in ("train_loader", "val_loader"):
        assert loaders[key].generator is not None
        assert loaders[key].worker_init_fn is seed_worker


# --------------------------------------------------------------------------- #
# Per-run metadata provenance.
# --------------------------------------------------------------------------- #
def test_run_metadata_has_provenance_keys() -> None:
    """collect_run_metadata records git/env/seed/config provenance (M6)."""
    from codes.config_schema import (
        OptimizerConfig,
        PreprocessingConfig,
        RecipesConfig,
        SchedulerConfig,
    )

    cfg = SimpleNamespace(
        RANDOM_SEED=42, DETERMINISTIC=True, NUM_EPOCHS=1, K_FOLDS=2,
        TEST_SUBJECTS=["S5"], IMAGE_SIZE=(256, 256), NUM_CLASSES=10,
        OPTIMIZER=OptimizerConfig(), SCHEDULER=SchedulerConfig(),
        RECIPES=RecipesConfig(), PREPROCESSING=PreprocessingConfig(),
    )
    md = main_pipeline.collect_run_metadata(cfg, "run-x", ["unet"])
    required = {
        "run_id", "git_sha", "git_dirty", "torch_version", "cuda_available",
        "random_seed", "deterministic", "test_subjects", "num_epochs",
        "k_folds", "lockfile_hash", "recipes",
    }
    assert required <= set(md)
    assert md["random_seed"] == 42
    assert md["deterministic"] is True
    assert md["test_subjects"] == ["S5"]
    # Effective recipe recorded per model (M9).
    assert md["recipes"]["unet"] == {"optimizer": "adam", "scheduler": "reduce_on_plateau"}
