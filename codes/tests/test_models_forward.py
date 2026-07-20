"""Forward-shape, stem-adaptation, and train-loop integration for all models.

Covers T3.2 (UB-16/17, M5): the two new pretrained encoders (``swin_pretrained``,
``transunet_pretrained``) alongside the existing from-scratch variants. All
offline tests build with ``pretrained=False`` (random weights) so nothing here
touches the network; the real pretrained-load proof is a separate, network-gated
test skipped unless ``UBENCH_ALLOW_DOWNLOADS=1``.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn

from codes.model_registry import create_model, get_registered_models
from codes.pretrained_stem import resolve_pretrained, sum_rgb_kernels

# Importing the model modules runs their @register_model decorators.
import codes.unet_v2  # noqa: F401
import codes.transunet  # noqa: F401
import codes.swin_unet_plus_plus  # noqa: F401
import codes.swin_pretrained  # noqa: F401
import codes.transunet_pretrained  # noqa: F401

_IMG_SIZE_KEYS = {
    "transunet", "swin_unet_plus_plus", "swin_pretrained", "transunet_pretrained",
}
_PRETRAINED_KEYS = {"swin_pretrained", "transunet_pretrained"}


def build_model(key: str, num_classes: int = 10) -> nn.Module:
    """Construct a registered model offline (pretrained encoders get random weights)."""
    kwargs = {"in_channels": 1, "num_classes": num_classes}
    if key in _IMG_SIZE_KEYS:
        kwargs["img_size"] = 256
    if key in _PRETRAINED_KEYS:
        kwargs["pretrained"] = False
    return create_model(key, **kwargs)


# --------------------------------------------------------------------------- #
# Forward-shape: every registered model maps (2,1,256,256) -> (2,10,256,256).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", sorted(get_registered_models()))
def test_forward_shape(key: str) -> None:
    model = build_model(key).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 1, 256, 256))
    assert out.shape == (2, 10, 256, 256), f"{key}: got {tuple(out.shape)}"


# --------------------------------------------------------------------------- #
# Param counts (M9): logged, and sane ordering (pretrained ViT-B is heaviest).
# --------------------------------------------------------------------------- #
def test_param_counts_logged() -> None:
    counts = {}
    for key in sorted(get_registered_models()):
        n = sum(p.numel() for p in build_model(key).parameters())
        counts[key] = n
        print(f"PARAMS {key}: {n:,}")
    assert counts["unet"] < counts["transunet_pretrained"]
    assert counts["swin_pretrained"] < counts["transunet_pretrained"]
    assert all(n > 0 for n in counts.values())


# --------------------------------------------------------------------------- #
# 1-channel stem adaptation (M5): sum the RGB kernels.
# --------------------------------------------------------------------------- #
def test_stem_sum_offline() -> None:
    rgb = torch.randn(64, 3, 7, 7)
    gray = sum_rgb_kernels(rgb)
    assert gray.shape == (64, 1, 7, 7)
    assert torch.allclose(gray[:, 0], rgb.sum(dim=1))


def test_stem_sum_rejects_non_rgb() -> None:
    with pytest.raises(ValueError):
        sum_rgb_kernels(torch.randn(64, 1, 7, 7))


# --------------------------------------------------------------------------- #
# Network gating: resolve_pretrained honours flag then env, default True.
# --------------------------------------------------------------------------- #
def test_resolve_pretrained_explicit_wins(monkeypatch) -> None:
    monkeypatch.setenv("UBENCH_PRETRAINED", "1")
    assert resolve_pretrained(False) is False
    assert resolve_pretrained(True) is True


def test_resolve_pretrained_env(monkeypatch) -> None:
    monkeypatch.delenv("UBENCH_PRETRAINED", raising=False)
    assert resolve_pretrained(None) is True          # default: real runs pretrained
    monkeypatch.setenv("UBENCH_PRETRAINED", "0")
    assert resolve_pretrained(None) is False          # tests/smoke: offline


# --------------------------------------------------------------------------- #
# Train-loop integration: one optimizer step per new key through UnifiedTrainer.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", sorted(_PRETRAINED_KEYS))
def test_single_train_step(key: str, tmp_path) -> None:
    from types import SimpleNamespace

    from codes.config_schema import (
        LossConfig,
        OptimizerConfig,
        RecipesConfig,
        SchedulerConfig,
    )
    from codes.unified_training import UnifiedTrainer

    model = build_model(key)
    cfg = SimpleNamespace(
        DEVICE=torch.device("cpu"),
        NUM_CLASSES=10,
        OUTPUT_DIR=tmp_path,
        LOSS=LossConfig(),
        OPTIMIZER=OptimizerConfig(),
        SCHEDULER=SchedulerConfig(),
        RECIPES=RecipesConfig(),
    )
    x = torch.randn(1, 1, 256, 256)
    y = torch.randint(0, 10, (1, 256, 256))
    loader = [(x, y, ["a"])]
    trainer = UnifiedTrainer(
        model, key, loader, loader, cfg, num_epochs=1, model_key=key, fold=1,
    )
    before = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    trainer.optimizer.zero_grad()
    loss = trainer.criterion(model(x), y)
    loss.backward()
    trainer.optimizer.step()
    assert torch.isfinite(loss)
    after = [p for p in model.parameters() if p.requires_grad]
    assert any(not torch.equal(b, a) for b, a in zip(before, after)), "no param moved"


# --------------------------------------------------------------------------- #
# Real pretrained load (network) — proof, not absence-of-error (M5/M9/R1).
# Skipped unless UBENCH_ALLOW_DOWNLOADS=1 (CI/smoke must never download).
# --------------------------------------------------------------------------- #
@pytest.mark.pretrained
@pytest.mark.skipif(
    os.environ.get("UBENCH_ALLOW_DOWNLOADS") != "1",
    reason="needs network access to the HF hub; set UBENCH_ALLOW_DOWNLOADS=1",
)
@pytest.mark.parametrize(
    "key, timm_id",
    [
        ("swin_pretrained", "swinv2_tiny_window8_256.ms_in1k"),
        ("transunet_pretrained", "vit_base_r50_s16_224.orig_in21k"),
    ],
)
def test_pretrained_stem_matches_hub_sum(key: str, timm_id: str) -> None:
    """The adapted 1-ch stem equals the hub's RGB stem summed (M5)."""
    import timm

    kwargs = {"in_channels": 1, "num_classes": 10, "img_size": 256, "pretrained": True}
    model = create_model(key, **kwargs)

    def first_conv(state_dict):
        for name, w in state_dict.items():
            if name.endswith(".weight") and w.dim() == 4:
                return name, w
        raise AssertionError("no 4-D conv weight found")

    name1, w1 = first_conv(model.encoder.state_dict())
    # Reference: same encoder, 3-channel pretrained stem.
    ref = timm.create_model(timm_id, pretrained=True, in_chans=3, num_classes=0)
    _, w3 = first_conv(ref.state_dict())

    assert w1.shape[1] == 1 and w3.shape[1] == 3
    assert torch.allclose(w1, sum_rgb_kernels(w3), atol=1e-6), (
        f"{key}: adapted stem {name1} != summed RGB stem"
    )
    total = sum(p.numel() for p in model.parameters())
    print(f"PRETRAINED-LOAD {key}: stem={name1} verified; total params {total:,}")


# --------------------------------------------------------------------------- #
# Registration pin (UB-21 / T3.6 ▲D).
# --------------------------------------------------------------------------- #
def test_all_models_registered() -> None:
    """Importing the pipeline registers exactly the five expected models.

    ``@register_model`` fires only as an import side effect; after the UB-21
    import refactor a dropped or reordered registration import in
    ``main_pipeline`` would *silently* deregister a model and shrink the CLI
    choices (UB-26) — no error, wrong benchmark. Pin the set so any such
    regression fails loudly.
    """
    import codes.main_pipeline  # noqa: F401 — import triggers registration

    assert set(get_registered_models()) == {
        "unet",
        "transunet",
        "swin_unet_plus_plus",
        "swin_pretrained",
        "transunet_pretrained",
    }
