"""Fixed-batch, isolation-safe VRAM probe tests (UB-10 T2.3; UB-28 T5.2).

Two distinct defects, both about *comparability* of the peak-VRAM column:

* **UB-10 (fixed)** — the benchmark read peak VRAM during each model's
  evaluation pass, at that model's hardware-selected batch size, so the numbers
  were not comparable (M3).  ``probe_peak_memory`` replaced that with a single
  inference forward on a synthetic batch of one FIXED size shared by every model.
* **UB-28 (this fix)** — fixing the batch size was not sufficient.
  ``torch.cuda.max_memory_allocated()`` is a *device-wide* counter and
  ``reset_peak_memory_stats()`` re-seeds the peak to the currently-allocated
  total, not to zero.  ``run_benchmark`` holds every model instance in
  ``models_dict`` and ``load_model`` moves each onto the GPU without ever moving
  it off, so each successive probe was inflated by every previously-benchmarked
  model's weights.  On the reported run the inflation grew monotonically with
  probe order (+29 / +317 / +625 MB) and **inverted** the VRAM ordering of the
  three architectures.  The fix reports ``own parameters + this forward's
  working set`` — a delta, immune to whatever else is resident.

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

_MIB = 1024 * 1024


class _ShapeCapturingModel:
    """Records the shape of the tensor it is called with; ``eval()`` -> self.

    Carries real (CPU) parameter tensors so the probe can size the model the
    same way it sizes a genuine ``nn.Module`` — ``param_bytes`` is the exact
    total the probe must attribute to this model.
    """

    def __init__(self, param_bytes: int = 0) -> None:
        self.seen_shape: tuple | None = None
        self.n_calls = 0
        # float32 → 4 bytes/element; param_bytes must be a multiple of 4.
        self._params = [torch.zeros(param_bytes // 4)] if param_bytes else []

    def eval(self):
        return self

    def parameters(self):
        return iter(self._params)

    def buffers(self):
        return iter(())

    def __call__(self, x):
        self.seen_shape = tuple(x.shape)
        self.n_calls += 1
        return x


def _patch_cuda(
    monkeypatch: pytest.MonkeyPatch,
    activation_bytes: int = 100 * _MIB,
    resident_bytes: int = 0,
):
    """Make the CUDA branch runnable on a CPU box; return a call counter.

    Models a device holding ``resident_bytes`` of *other* tensors (previously
    benchmarked models) before the probe runs: ``memory_allocated`` reports that
    baseline and ``max_memory_allocated`` reports it plus this forward's
    ``activation_bytes``.  A correct probe must report a figure that does not
    depend on ``resident_bytes`` (UB-28).
    """
    calls = {"reset": 0, "max": 0, "current": 0}
    monkeypatch.setattr(
        torch.cuda, "reset_peak_memory_stats",
        lambda *a, **k: calls.__setitem__("reset", calls["reset"] + 1),
    )

    def _max(*_a, **_k):
        calls["max"] += 1
        return resident_bytes + activation_bytes

    def _current(*_a, **_k):
        calls["current"] += 1
        return resident_bytes

    monkeypatch.setattr(torch.cuda, "max_memory_allocated", _max)
    monkeypatch.setattr(torch.cuda, "memory_allocated", _current)
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
    calls = _patch_cuda(monkeypatch, activation_bytes=100 * _MIB)

    mb = probe_peak_memory(_ShapeCapturingModel(), "cuda", (16, 16))

    assert calls["reset"] == 1
    assert calls["max"] == 1
    assert mb == pytest.approx(100.0)   # 100 MiB reported in MB


@pytest.mark.parametrize("resident_bytes", [0, 400 * _MIB, 1024 * _MIB])
def test_probe_is_invariant_to_other_resident_models(
    monkeypatch: pytest.MonkeyPatch, resident_bytes: int,
) -> None:
    """UB-28: the figure must not depend on what else is resident on the device.

    ``run_benchmark`` keeps every model in ``models_dict`` on the GPU, so by the
    third probe two other models' weights are still allocated.  Reading
    ``max_memory_allocated`` directly charged those to whichever model happened
    to be probed last; this test pins the fix by sweeping the resident baseline
    and demanding a constant answer.
    """
    _patch_cuda(monkeypatch, activation_bytes=100 * _MIB, resident_bytes=resident_bytes)

    mb = probe_peak_memory(_ShapeCapturingModel(param_bytes=40 * _MIB), "cuda", (16, 16))

    # own parameters (40 MiB) + this forward's working set (100 MiB)
    assert mb == pytest.approx(140.0)


def test_probe_counts_own_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    """A model's own weights are attributed to it, not silently dropped.

    The reported figure is ``own parameters + working set``; with the working
    set held constant, a heavier model must report proportionally more.
    """
    _patch_cuda(monkeypatch, activation_bytes=10 * _MIB, resident_bytes=512 * _MIB)
    light = probe_peak_memory(_ShapeCapturingModel(param_bytes=8 * _MIB), "cuda", (16, 16))

    _patch_cuda(monkeypatch, activation_bytes=10 * _MIB, resident_bytes=0)
    heavy = probe_peak_memory(_ShapeCapturingModel(param_bytes=64 * _MIB), "cuda", (16, 16))

    assert light == pytest.approx(18.0)   # 8 + 10
    assert heavy == pytest.approx(74.0)   # 64 + 10
    assert heavy > light


def test_probe_reads_baseline_before_peak(monkeypatch: pytest.MonkeyPatch) -> None:
    """The current-allocation baseline is actually consulted (not assumed zero)."""
    calls = _patch_cuda(monkeypatch, activation_bytes=100 * _MIB, resident_bytes=256 * _MIB)

    probe_peak_memory(_ShapeCapturingModel(), "cuda", (16, 16))

    assert calls["current"] >= 1, "probe never read torch.cuda.memory_allocated()"


def test_probe_discards_a_warmup_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    """A warm-up forward runs before the measured one (UB-09's rationale).

    The first CUDA forward in a process allocates cuDNN/cuBLAS handles and
    workspaces; on the reference box that made the first reading ~7.8 MB higher
    than every subsequent one. Discarding one forward makes the probe
    reproducible, so the measured pass must be the *second* call.
    """
    _patch_cuda(monkeypatch)
    model = _ShapeCapturingModel()

    probe_peak_memory(model, "cuda", (16, 16))

    assert model.n_calls == 2, (
        f"expected 1 warm-up + 1 measured forward, got {model.n_calls}"
    )
