"""UB-12 (T3.1): the single validated config authority.

``codes/config.yaml`` is loaded through a pydantic schema
(``codes/config_schema.py``) so that unknown keys and wrong-typed values raise
at startup instead of being silently ignored by the old ``dict.get(key,
default)`` lookups. Every wired key must also demonstrably change behavior —
that is the other half of the UB-12 failure mode (dead keys) being closed.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import yaml

from codes.config_schema import (
    LossConfig,
    OptimizerConfig,
    RootConfig,
    SchedulerConfig,
    load_config,
)
from codes.unified_training import UnifiedTrainer, resolve_class_weights

# The single real config the pipeline loads (codes/config.yaml).
REPO_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


def _write_yaml(tmp_path: Path, data: dict) -> Path:
    """Write ``data`` as a config.yaml under ``tmp_path`` and return its path."""
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Schema gate: unknown keys / wrong types raise at load time (R4)
# ---------------------------------------------------------------------------

def test_shipped_config_validates():
    """The committed codes/config.yaml passes the schema."""
    cfg = load_config(REPO_CONFIG)
    assert isinstance(cfg, RootConfig)
    assert cfg.model.num_classes == 10
    assert cfg.training.num_epochs == 100


def test_unknown_nested_key_raises(tmp_path):
    """A typo'd key inside a section is a hard error, not a silent no-op."""
    p = _write_yaml(tmp_path, {"training": {"num_epochs": 1, "bogus_key": 3}})
    with pytest.raises(ValueError):
        load_config(p)


def test_unknown_top_level_key_raises(tmp_path):
    """An unknown top-level section raises."""
    p = _write_yaml(tmp_path, {"not_a_real_section": {"x": 1}})
    with pytest.raises(ValueError):
        load_config(p)


def test_wrong_type_raises(tmp_path):
    """A wrong-typed value raises instead of crashing deep in the pipeline."""
    p = _write_yaml(tmp_path, {"training": {"num_epochs": "not-an-int"}})
    with pytest.raises(ValueError):
        load_config(p)


def test_dead_batch_sizes_key_rejected(tmp_path):
    """training.batch_sizes was unconsumed (UB-12) and is now removed, so it
    is an unknown key — proving hardware_detector is the sole batch authority."""
    p = _write_yaml(
        tmp_path,
        {"training": {"batch_sizes": {"unet": 8, "transunet": 6, "swin": 6}}},
    )
    with pytest.raises(ValueError):
        load_config(p)


def test_missing_file_uses_defaults(tmp_path):
    """A missing config file yields all-default values (guards odd contexts)."""
    cfg = load_config(tmp_path / "does_not_exist.yaml")
    assert cfg.training.num_epochs == 100
    assert cfg.model.num_classes == 10


# ---------------------------------------------------------------------------
# The shipped recipe defaults reproduce the pre-T3.1 hardcoded values
# (no-numbers-moved guard, config level)
# ---------------------------------------------------------------------------

def test_shipped_recipe_defaults_match_hardcoded():
    cfg = load_config(REPO_CONFIG)
    assert (cfg.loss.ce_weight, cfg.loss.dice_weight) == (0.5, 0.5)
    assert cfg.loss.class_weights is None
    assert cfg.optimizer.name == "adam"
    assert cfg.optimizer.weight_decay == 0.0
    assert tuple(cfg.optimizer.betas) == (0.9, 0.999)
    assert cfg.scheduler.name == "reduce_on_plateau"
    assert cfg.scheduler.patience == 5
    assert cfg.scheduler.factor == 0.5


def test_bad_class_weights_string_rejected_at_load(tmp_path):
    """A non-'balanced' string for class_weights fails schema validation."""
    p = _write_yaml(tmp_path, {"loss": {"class_weights": "inverse"}})
    with pytest.raises(ValueError):
        load_config(p)


# ---------------------------------------------------------------------------
# Config -> trainer wiring: each key demonstrably changes the constructed object
# ---------------------------------------------------------------------------

def _make_trainer(tmp_path, *, loss=None, optimizer=None, scheduler=None):
    """Build a UnifiedTrainer on CPU with a lightweight stand-in config.

    Exercises exactly the loss/optimizer/scheduler wiring path without the full
    pipeline (no data, no training run).
    """
    cfg = SimpleNamespace(
        DEVICE=torch.device("cpu"),
        NUM_CLASSES=10,
        OUTPUT_DIR=tmp_path,
        LOSS=loss or LossConfig(),
        OPTIMIZER=optimizer or OptimizerConfig(),
        SCHEDULER=scheduler or SchedulerConfig(),
    )
    model = nn.Conv2d(1, 10, kernel_size=1)
    # len() is all __init__ needs from the loaders (class_weights default=None
    # so the loader is never iterated here).
    loaders = [(torch.zeros(1, 1, 4, 4), torch.zeros(1, 4, 4, dtype=torch.long), "id")]
    return UnifiedTrainer(
        model, "test", loaders, loaders, cfg, model_key="unet", fold=1
    )


def test_scheduler_keys_wired(tmp_path):
    t = _make_trainer(tmp_path, scheduler=SchedulerConfig(patience=9, factor=0.25))
    assert t.scheduler.patience == 9
    assert t.scheduler.factor == 0.25


def test_optimizer_keys_wired(tmp_path):
    t = _make_trainer(
        tmp_path, optimizer=OptimizerConfig(weight_decay=0.05, betas=(0.8, 0.9))
    )
    assert t.optimizer.defaults["weight_decay"] == 0.05
    assert tuple(t.optimizer.defaults["betas"]) == (0.8, 0.9)


def test_loss_weights_wired(tmp_path):
    t = _make_trainer(tmp_path, loss=LossConfig(ce_weight=0.7, dice_weight=0.3))
    assert t.criterion.ce_weight == 0.7
    assert t.criterion.dice_weight == 0.3


def test_unsupported_optimizer_raises(tmp_path):
    with pytest.raises(ValueError):
        _make_trainer(tmp_path, optimizer=OptimizerConfig(name="sgd"))


def test_unsupported_scheduler_raises(tmp_path):
    with pytest.raises(ValueError):
        _make_trainer(tmp_path, scheduler=SchedulerConfig(name="cosine"))


# ---------------------------------------------------------------------------
# class_weights resolution: null -> None; balanced -> non-uniform; list
# ---------------------------------------------------------------------------

def test_class_weights_null_is_none():
    assert resolve_class_weights(None, 10) is None


def test_class_weights_balanced_is_nonuniform():
    # Imbalanced tiny loader: class 0 fills the mask, class 1 is a single pixel.
    mask = torch.zeros(1, 4, 4, dtype=torch.long)
    mask[0, 0, 0] = 1
    loader = [(torch.zeros(1, 1, 4, 4), mask, "id")]
    w = resolve_class_weights(
        "balanced", 2, train_loader=loader, device=torch.device("cpu")
    )
    assert w is not None and w.shape == (2,)
    assert w[1] > w[0]  # the rarer class gets the larger weight


def test_class_weights_explicit_list():
    w = resolve_class_weights([1.0] * 10, 10)
    assert torch.allclose(w, torch.ones(10))


def test_class_weights_bad_string_raises():
    with pytest.raises(ValueError):
        resolve_class_weights("inverse", 10, train_loader=[])


def test_class_weights_wrong_length_raises():
    with pytest.raises(ValueError):
        resolve_class_weights([1.0, 2.0], 10)
