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
from typing import List, Optional, Tuple, Union

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
    """Optimizer recipe (``optimizer:`` section).

    Shaped so T3.3 can extend it with per-family recipes (e.g. AdamW + warmup
    for transformers); only ``adam`` is wired today and the trainer raises on
    any other ``name``.
    """

    name: str = "adam"
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.999)


class SchedulerConfig(_Strict):
    """LR scheduler recipe (``scheduler:`` section).

    Only ``reduce_on_plateau`` is wired today; cosine/warmup arrive with T3.3
    and the trainer raises on any other ``name``.
    """

    name: str = "reduce_on_plateau"
    patience: int = 5
    factor: float = 0.5


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
