"""Per-family recipes + scheduler cadence (T3.3, UB-18, M4).

Covers recipe resolution (which model gets which optimizer/scheduler), the
degenerate-warmup arithmetic the smoke exercises, and the scheduler *stepping
contract* — the silent-corruption trap (▲A): plateau steps once per epoch WITH
the val metric; warmup_cosine steps once per optimizer step WITHOUT an argument
(a positional arg is read as an epoch index by stdlib schedulers).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from codes.config_schema import (
    FamilyRecipe,
    OptimizerConfig,
    RecipesConfig,
    SchedulerConfig,
    load_config,
    resolve_recipe,
)
from codes.unified_training import UnifiedTrainer, warmup_cosine_split

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"

_GLOBAL_OPT = OptimizerConfig(name="adam")
_GLOBAL_SCH = SchedulerConfig(name="reduce_on_plateau")


# --------------------------------------------------------------------------- #
# resolve_recipe
# --------------------------------------------------------------------------- #
def test_unmapped_key_uses_global_default():
    recipes = RecipesConfig()
    opt, sch = resolve_recipe(_GLOBAL_OPT, _GLOBAL_SCH, recipes, "unet")
    assert opt is _GLOBAL_OPT and sch is _GLOBAL_SCH


def test_family_override_applied():
    recipes = RecipesConfig(
        model_families={"transunet": "transformer"},
        families={"transformer": FamilyRecipe(
            optimizer=OptimizerConfig(name="adamw", weight_decay=0.05),
            scheduler=SchedulerConfig(name="warmup_cosine"),
        )},
    )
    opt, sch = resolve_recipe(_GLOBAL_OPT, _GLOBAL_SCH, recipes, "transunet")
    assert opt.name == "adamw" and opt.weight_decay == 0.05
    assert sch.name == "warmup_cosine"


def test_family_null_override_inherits_global():
    recipes = RecipesConfig(
        model_families={"unet": "cnn"},
        families={"cnn": FamilyRecipe()},  # both None -> inherit
    )
    opt, sch = resolve_recipe(_GLOBAL_OPT, _GLOBAL_SCH, recipes, "unet")
    assert opt is _GLOBAL_OPT and sch is _GLOBAL_SCH


def test_unknown_family_raises():
    recipes = RecipesConfig(model_families={"unet": "typo_family"}, families={})
    with pytest.raises(ValueError):
        resolve_recipe(_GLOBAL_OPT, _GLOBAL_SCH, recipes, "unet")


def test_shipped_config_family_assignment():
    """The committed config: unet -> adam/plateau; transformers -> adamw/cosine."""
    cfg = load_config(REPO_CONFIG)
    opt_u, sch_u = resolve_recipe(cfg.optimizer, cfg.scheduler, cfg.recipes, "unet")
    assert opt_u.name == "adam" and sch_u.name == "reduce_on_plateau"
    for key in ("transunet", "swin_unet_plus_plus",
                "swin_pretrained", "transunet_pretrained"):
        opt, sch = resolve_recipe(cfg.optimizer, cfg.scheduler, cfg.recipes, key)
        assert opt.name == "adamw", key
        assert sch.name == "warmup_cosine", key
        assert opt.weight_decay == pytest.approx(0.05), key


# --------------------------------------------------------------------------- #
# Degenerate-warmup arithmetic
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "total, frac, exp",
    [(2, 0.05, (1, 1)), (3, 0.05, (1, 2)), (1000, 0.05, (50, 950))],
)
def test_warmup_cosine_split(total, frac, exp):
    assert warmup_cosine_split(total, frac) == exp


def test_warmup_cosine_split_needs_two_steps():
    with pytest.raises(ValueError):
        warmup_cosine_split(1, 0.05)


# --------------------------------------------------------------------------- #
# Trainer construction from a resolved recipe.
# --------------------------------------------------------------------------- #
def _cfg(tmp_path, opt, sch):
    return SimpleNamespace(
        DEVICE=torch.device("cpu"),
        NUM_CLASSES=10,
        OUTPUT_DIR=tmp_path,
        LOSS=SimpleNamespace(ce_weight=0.5, dice_weight=0.5, class_weights=None),
        OPTIMIZER=opt,
        SCHEDULER=sch,
        RECIPES=RecipesConfig(),  # unmapped -> use OPTIMIZER/SCHEDULER above
    )


def _trainer(tmp_path, opt, sch, num_epochs=2, n_batches=3):
    loader = [(torch.zeros(2, 1, 16, 16),
               torch.zeros(2, 16, 16, dtype=torch.long), ["a", "b"])] * n_batches
    return UnifiedTrainer(
        nn.Conv2d(1, 10, 1), "m", loader, loader, _cfg(tmp_path, opt, sch),
        num_epochs=num_epochs, model_key="m", fold=1,
    )


def test_adamw_warmup_cosine_constructible(tmp_path):
    t = _trainer(tmp_path, OptimizerConfig(name="adamw", weight_decay=0.05),
                 SchedulerConfig(name="warmup_cosine"))
    assert isinstance(t.optimizer, torch.optim.AdamW)
    assert t.optimizer.defaults["weight_decay"] == 0.05
    assert isinstance(t.scheduler, torch.optim.lr_scheduler.SequentialLR)
    assert t._scheduler_per_batch is True


def test_adam_plateau_is_per_epoch(tmp_path):
    t = _trainer(tmp_path, OptimizerConfig(name="adam"),
                 SchedulerConfig(name="reduce_on_plateau"))
    assert isinstance(t.optimizer, torch.optim.Adam)
    assert t._scheduler_per_batch is False


def test_unknown_optimizer_and_scheduler_raise(tmp_path):
    with pytest.raises(ValueError):
        _trainer(tmp_path, OptimizerConfig(name="sgd"),
                 SchedulerConfig(name="reduce_on_plateau"))
    with pytest.raises(ValueError):
        _trainer(tmp_path, OptimizerConfig(name="adam"),
                 SchedulerConfig(name="cosine"))


# --------------------------------------------------------------------------- #
# Scheduler stepping-contract (▲A) — the silent-corruption guard.
# --------------------------------------------------------------------------- #
class _SpyScheduler:
    """Records every step() call and its args; stands in for the real scheduler."""

    def __init__(self):
        self.calls = []  # list of args tuples

    def step(self, *args):
        self.calls.append(args)

    def state_dict(self):
        return {}


def _run_one_epoch(tmp_path, sch_name):
    """Build a trainer, swap in a spy scheduler, run one epoch, return the spy."""
    opt = OptimizerConfig(name="adamw" if sch_name == "warmup_cosine" else "adam")
    t = _trainer(tmp_path, opt, SchedulerConfig(name=sch_name),
                 num_epochs=1, n_batches=3)
    spy = _SpyScheduler()
    t.scheduler = spy
    t.train()
    return spy


def test_plateau_receives_metric_per_epoch(tmp_path):
    spy = _run_one_epoch(tmp_path, "reduce_on_plateau")
    # Exactly one epoch-level step, carrying the val metric (one float arg).
    assert len(spy.calls) == 1
    assert len(spy.calls[0]) == 1 and isinstance(spy.calls[0][0], float)


def test_warmup_cosine_steps_per_batch_without_arg(tmp_path):
    spy = _run_one_epoch(tmp_path, "warmup_cosine")
    # One no-arg step per batch (3 batches), never the epoch-level metric step.
    assert len(spy.calls) == 3
    assert all(args == () for args in spy.calls)
