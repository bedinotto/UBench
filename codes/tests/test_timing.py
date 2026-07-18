"""Timing honesty tests (UB-09, T2.1).

Two guarantees:

* ``benchmark_models.timed_inference`` discards warm-up batches with the tiny
  loader guard ``min(5, max(0, n_batches - 1))`` and synchronizes CUDA exactly
  when — and only when — the device is CUDA.
* ``UnifiedTrainer.validate`` no longer measures or returns per-epoch inference
  time (it was unsynced kernel-launch time that also included the loss).

The ``timed_inference`` tests use lightweight fakes so they need no GPU: the
sync decision keys off the ``device`` argument, not on tensor placement, so a
``device="cuda"`` call exercises the sync path on a CPU-only box.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from codes.benchmark_models import timed_inference
from codes.config_schema import LossConfig, OptimizerConfig, SchedulerConfig
from codes.unified_training import UnifiedTrainer


# --------------------------------------------------------------------------- #
# Fakes for timed_inference — only .to()/.size()/callable are exercised.
# --------------------------------------------------------------------------- #
class _FakeImages:
    """Stands in for a batch tensor; ``.to(cuda)`` is a no-op (no GPU needed)."""

    def __init__(self, n: int) -> None:
        self._n = n

    def to(self, *args, **kwargs):  # noqa: D401 - mimics Tensor.to
        return self

    def size(self, _dim: int) -> int:
        return self._n


class _FakeModel:
    """Callable model that counts forward passes; ``eval()`` returns self."""

    def __init__(self) -> None:
        self.calls = 0

    def eval(self):
        return self

    def __call__(self, x):
        self.calls += 1
        return x


def _fake_loader(n_batches: int, batch_size: int = 4):
    return [(_FakeImages(batch_size), None, None) for _ in range(n_batches)]


@pytest.mark.parametrize(
    "n_batches, exp_warmup, exp_measured",
    [(1, 0, 1), (3, 2, 1), (20, 5, 15)],
)
def test_warmup_arithmetic(n_batches: int, exp_warmup: int, exp_measured: int) -> None:
    """Warm-up = min(5, n_batches-1); >=1 measured batch even for tiny loaders."""
    model = _FakeModel()
    out = timed_inference(model, _fake_loader(n_batches), device="cpu", warmup=5)

    assert out["n_warmup"] == exp_warmup
    assert out["n_measured"] == exp_measured
    # Every batch runs a forward (warm-up + measured); none is skipped entirely.
    assert model.calls == n_batches


def test_sync_called_twice_on_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """One measured batch on a CUDA device → synchronize() before and after."""
    calls = {"n": 0}
    monkeypatch.setattr(
        torch.cuda, "synchronize",
        lambda *a, **k: calls.__setitem__("n", calls["n"] + 1),
    )

    # 1 batch → n_warmup=0, n_measured=1 → exactly one timed forward.
    out = timed_inference(_FakeModel(), _fake_loader(1), device="cuda", warmup=5)

    assert out["n_measured"] == 1
    assert calls["n"] == 2  # sync immediately before and after the timed pass


def test_sync_not_called_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """CPU device → synchronize() is never called (nothing to synchronize)."""
    calls = {"n": 0}
    monkeypatch.setattr(
        torch.cuda, "synchronize",
        lambda *a, **k: calls.__setitem__("n", calls["n"] + 1),
    )

    timed_inference(_FakeModel(), _fake_loader(3), device="cpu", warmup=5)

    assert calls["n"] == 0


# --------------------------------------------------------------------------- #
# validate() no longer emits a timing value.
# --------------------------------------------------------------------------- #
class _TinyDataset(Dataset):
    """4 tiny (1, 8, 8) frames with integer masks in ``[0, num_classes)``."""

    def __init__(self, n: int, num_classes: int, hw: int = 8) -> None:
        torch.manual_seed(0)
        self.images = torch.rand(n, 1, hw, hw)
        self.masks = torch.randint(0, num_classes, (n, hw, hw))
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        return self.images[i], self.masks[i], f"id{i}"


def test_validate_emits_no_timing(tmp_path) -> None:
    """validate() returns exactly (loss, mIoU, dice) and holds no timing state."""
    num_classes = 3
    loader = DataLoader(_TinyDataset(4, num_classes), batch_size=2)
    cfg = SimpleNamespace(
        DEVICE=torch.device("cpu"),
        NUM_CLASSES=num_classes,
        OUTPUT_DIR=tmp_path,
        # Loss/optimizer/scheduler recipes the trainer now reads from config
        # (T3.1/UB-12); schema defaults reproduce the old hardcoded values.
        LOSS=LossConfig(),
        OPTIMIZER=OptimizerConfig(),
        SCHEDULER=SchedulerConfig(),
    )
    model = nn.Conv2d(1, num_classes, kernel_size=1)  # (N,1,H,W) -> (N,C,H,W)

    trainer = UnifiedTrainer(
        model, "TinyModel", loader, loader, cfg,
        num_epochs=1, model_key="tiny", fold=1,
    )

    result = trainer.validate()

    assert len(result) == 3  # was a 4-tuple with avg_inference_time before T2.1
    assert all(isinstance(v, float) for v in result)
    assert not hasattr(trainer, "inference_times")
