"""Typed schema for the single UBench config (``codes/config.yaml``).

UB-12 (T3.1): the config used to be read with unchecked ``dict.get(key,
default)`` lookups, so a typo'd key was silently ignored (the default won) and
a wrong-typed value only blew up deep inside the pipeline. This module puts a
pydantic v2 schema over the file, loaded by :class:`codes.unified_data.Config`
at import time. Two guarantees follow (R4 — no silent failure paths):

* **Unknown keys raise.** Every model sets ``extra="forbid"``; a typo is a
  hard :class:`ValueError` at startup, not a silent no-op.
* **Wrong types raise.** ``num_epochs: "ten"`` fails validation immediately.

The complementary rule (closing the UB-12 failure mode) lives in the pipeline:
every key defined here is consumed somewhere — there are no dead keys.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator


class _Strict(BaseModel):
    """Base model that rejects unknown keys so typos raise (R4)."""

    model_config = ConfigDict(extra="forbid")


class PathsConfig(_Strict):
    """Filesystem roots (``paths:`` section)."""

    data_dir: str = "data"
    processed_dir: str = "data/processed"
    output_dir: str = "outputs"
    log_dir: str = "logs"


class ModelConfig(_Strict):
    """Model-shape parameters (``model:`` section)."""

    image_size: Tuple[int, int] = (256, 256)
    num_classes: int = 10


class TrainingConfig(_Strict):
    """Training / cross-validation parameters (``training:`` section).

    Note: there is deliberately no ``batch_sizes`` key. Batch sizes come from
    ``hardware_detector`` (the single authority, UB-05); a config override
    would reintroduce the dual-authority that caused UB-05, so it is a separate
    feature, not part of this dead-key cleanup (UB-12).
    """

    learning_rate: float = 1e-4
    num_epochs: int = 100
    k_folds: int = 5
    random_seed: int = 42
    deterministic: bool = True
    test_subjects: List[str] = []


class LossConfig(_Strict):
    """Combined Cross-Entropy + Dice loss (``loss:`` section).

    ``class_weights`` controls the CrossEntropy per-class weighting:

    * ``null`` — uniform (the current default; numbers unchanged by T3.1).
    * ``"balanced"`` — inverse **train-fold** class frequency, recomputed per
      fold from the training split only (never val/test). Opt-in.
    * an explicit list of ``num_classes`` floats.
    """

    ce_weight: float = 0.5
    dice_weight: float = 0.5
    class_weights: Optional[Union[str, List[float]]] = None

    @field_validator("class_weights")
    @classmethod
    def _valid_class_weights(cls, value):
        if isinstance(value, str) and value != "balanced":
            raise ValueError(
                "loss.class_weights string must be 'balanced' "
                "(or null, or a list of per-class floats)"
            )
        return value


class OptimizerConfig(_Strict):
    """Optimizer recipe (``optimizer:`` section or a per-family override).

    ``name`` is one of ``adam`` / ``adamw`` (the trainer hard-raises on any
    other name, R4). ``grad_clip_norm`` is declared here (T3.3/M4) rather than
    hardcoded in the trainer so each family can set its own clip.
    """

    name: str = "adam"
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.999)
    grad_clip_norm: float = 1.0


class SchedulerConfig(_Strict):
    """LR scheduler recipe (``scheduler:`` section or a per-family override).

    ``name`` is one of ``reduce_on_plateau`` (uses ``patience``/``factor``,
    stepped once per epoch with the val metric) or ``warmup_cosine`` (uses
    ``warmup_frac``, stepped once per optimizer step). The trainer hard-raises
    on any other name (R4).
    """

    name: str = "reduce_on_plateau"
    patience: int = 5
    factor: float = 0.5
    warmup_frac: float = 0.05


class FamilyRecipe(_Strict):
    """Per-family optimizer/scheduler override (``recipes.families.<name>``).

    A ``None`` field inherits the global ``optimizer`` / ``scheduler`` recipe,
    so a family can override just one of the two.
    """

    optimizer: Optional[OptimizerConfig] = None
    scheduler: Optional[SchedulerConfig] = None


class RecipesConfig(_Strict):
    """Per-family recipe assignments (``recipes:`` section, T3.3/UB-18/M4).

    ``model_families`` maps a registered model key to a family name; a model
    not listed inherits the global ``optimizer``/``scheduler``. ``families``
    defines each family's override. The default (both empty) reproduces the
    single global recipe (pre-T3.3 behavior).
    """

    model_families: Dict[str, str] = {}
    families: Dict[str, FamilyRecipe] = {}


def resolve_recipe(default_optimizer: OptimizerConfig,
                   default_scheduler: SchedulerConfig,
                   recipes: RecipesConfig,
                   model_key: str) -> Tuple[OptimizerConfig, SchedulerConfig]:
    """Resolve the effective (optimizer, scheduler) recipe for a model (M4).

    An unmapped model key uses the global defaults (pre-T3.3 behavior). A key
    mapped to an unknown family is a hard error (R4 — catches typos) rather
    than a silent fallback.
    """
    family = recipes.model_families.get(model_key)
    if family is None:
        return default_optimizer, default_scheduler
    if family not in recipes.families:
        raise ValueError(
            f"model '{model_key}' is assigned to unknown family '{family}'; "
            f"known families: {sorted(recipes.families)}"
        )
    recipe = recipes.families[family]
    optimizer = recipe.optimizer if recipe.optimizer is not None else default_optimizer
    scheduler = recipe.scheduler if recipe.scheduler is not None else default_scheduler
    return optimizer, scheduler


_DEFAULT_REGIONS: List[str] = [
    "background",
    "Contorno inferior do Rosto",
    "Sombrancelha esquerda",
    "Sombrancelha direita",
    "Nariz",
    "Olho esquerdo",
    "Olho direito",
    "Boca",
    "Labios",
    "Testa",
]


class RootConfig(_Strict):
    """Top-level config document (the whole of ``codes/config.yaml``)."""

    # ``protected_namespaces=()`` silences pydantic's warning about the bare
    # ``model`` field (it does not actually collide with the ``model_`` prefix,
    # but disabling the check keeps startup output clean).
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    paths: PathsConfig = PathsConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    loss: LossConfig = LossConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    recipes: RecipesConfig = RecipesConfig()
    regions: List[str] = _DEFAULT_REGIONS


def load_config(path) -> RootConfig:
    """Load and validate ``codes/config.yaml`` into a :class:`RootConfig`.

    Args:
        path: Path to the YAML config file.

    Returns:
        The validated config. A missing file yields all-default values (the
        committed file always exists; this only guards odd import contexts).

    Raises:
        ValueError: if the file contains an unknown key or a wrong-typed value.
            The pydantic ``ValidationError`` (itself a ``ValueError``) is
            re-raised with the offending file path prefixed for a clear
            startup failure.
    """
    path = Path(path)
    if not path.exists():
        return RootConfig()
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    try:
        return RootConfig(**raw)
    except ValidationError as exc:
        raise ValueError(f"Invalid config at {path}:\n{exc}") from exc
