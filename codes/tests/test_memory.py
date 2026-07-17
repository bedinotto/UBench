"""Fixed-batch VRAM probe tests (UB-10, T2.3).

The benchmark used to read peak VRAM during each model's evaluation pass, at
that model's hardware-selected batch size, so the numbers were not comparable
(M3). `benchmark_models.probe_peak_memory` replaces that with a single
inference forward on a synthetic batch of one FIXED size shared by every model.

These are unit-level (no subprocess). The CUDA path is exercised on a CPU-only
box by stripping the ``device`` kwarg from ``torch.zeros`` and counting the
monkeypatched ``torch.cuda`` calls — no GPU required.
"""

from __future__ import annotations

import pytest
import torch

from codes.benchmark_models import (
    MEMORY_PROBE_BATCH_SIZE,
    _format_vram,
    probe_peak_memory,
)


class _ShapeCapturingModel:
    """Records the shape of the tensor it is called with; ``eval()`` -> self."""

    def __init__(self) -> None:
        self.seen_shape: tuple | None = None

    def eval(self):
        return self

    def __call__(self, x):
        self.seen_shape = tuple(x.shape)
        return x


def _patch_cuda(monkeypatch: pytest.MonkeyPatch, mem_bytes: int = 100 * 1024 * 1024):
    """Make the CUDA branch runnable on a CPU box; return a call counter."""
    calls = {"reset": 0, "max": 0}
    monkeypatch.setattr(
        torch.cuda, "reset_peak_memory_stats",
        lambda *a, **k: calls.__setitem__("reset", calls["reset"] + 1),
    )

    def _max(*_a, **_k):
        calls["max"] += 1
        return mem_bytes

    monkeypatch.setattr(torch.cuda, "max_memory_allocated", _max)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda *a, **k: None)

    # Build the synthetic input on CPU regardless of the requested device.
    real_zeros = torch.zeros
    monkeypatch.setattr(
        torch, "zeros",
        lambda *a, **k: real_zeros(*a, **{kk: vv for kk, vv in k.items() if kk != "device"}),
    )
    return calls


def test_cpu_probe_returns_none_and_is_labeled() -> None:
    """On CPU there is nothing to measure: return None, render 'n/a (CPU)'."""
    model = _ShapeCapturingModel()
    assert probe_peak_memory(model, "cpu", (16, 16)) is None
    assert model.seen_shape is None          # no forward runs on the CPU path
    assert _format_vram(None) == "n/a (CPU)"
    assert _format_vram(123.4) == "123.40 MB"


@pytest.mark.parametrize("image_size", [(16, 16), (8, 32)])
def test_probe_uses_fixed_batch_and_synthetic_shape(
    monkeypatch: pytest.MonkeyPatch, image_size: tuple,
) -> None:
    """The forward sees a synthetic (FIXED_BATCH, 1, H, W) tensor."""
    _patch_cuda(monkeypatch)
    model = _ShapeCapturingModel()

    probe_peak_memory(model, "cuda", image_size, in_channels=1)

    assert model.seen_shape == (MEMORY_PROBE_BATCH_SIZE, 1, image_size[0], image_size[1])


def test_probe_batch_is_independent_of_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """The probe batch comes from its own argument, not the model's training batch."""
    _patch_cuda(monkeypatch)
    model = _ShapeCapturingModel()

    probe_peak_memory(model, "cuda", (16, 16), batch_size=7)

    assert model.seen_shape[0] == 7


def test_cuda_path_calls_memory_stats_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """reset_peak_memory_stats and max_memory_allocated each called exactly once."""
    calls = _patch_cuda(monkeypatch, mem_bytes=100 * 1024 * 1024)

    mb = probe_peak_memory(_ShapeCapturingModel(), "cuda", (16, 16))

    assert calls["reset"] == 1
    assert calls["max"] == 1
    assert mb == pytest.approx(100.0)   # 100 MiB reported in MB
