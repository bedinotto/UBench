"""
Single authority for checkpoint file naming (UB-02, R5).

Registry keys (``unet``, ``transunet``, ``swin_unet_plus_plus``) are the
canonical ``model_key`` for **all** file I/O; display names ("U-Net",
"Swin-UNet++") are for logs, plots, and prose only.  UB-02 happened because
the benchmark derived filenames from display names via ``_safe_filename``
while the trainer saved under registry names — every ``.pth`` path must go
through this module so the contract cannot drift again.

Pure functions, no globals, no filesystem access.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

_VALID_KINDS = ("best", "epoch")


def _validate_model_key(model_key: str) -> None:
    """Reject anything that is not a lowercase snake_case registry key.

    Display names ("U-Net", "Swin-UNet++") contain uppercase letters,
    hyphens, pluses, or spaces — exactly the strings that caused UB-02 —
    so this guard turns the historical bug class into a hard error.

    Membership in ``model_registry`` is deliberately NOT checked here:
    the registry is populated only when the model modules are imported,
    so a membership check would silently pass or fail depending on import
    order (a conditional validation — R4-unfriendly).
    """
    if not model_key or not all(
        c.islower() or c.isdigit() or c == "_" for c in model_key
    ):
        raise ValueError(
            f"model_key must be a lowercase snake_case registry key "
            f"(e.g. 'swin_unet_plus_plus'), got {model_key!r}. "
            f"Display names must never reach file I/O (UB-02/R5)."
        )


def checkpoint_path(
    output_dir: Path,
    model_key: str,
    fold: int,
    kind: str = "best",
    *,
    epoch: Optional[int] = None,
) -> Path:
    """Return the canonical checkpoint path for (model_key, fold, kind).

    Args:
        output_dir: The run output directory (``outputs/<timestamp>``).
        model_key: Registry key — validated, display names raise.
        fold: 1-based fold number.
        kind: ``"best"`` (weights-only best model, under ``models/``) or
            ``"epoch"`` (full resume checkpoint, under ``checkpoints/``).
        epoch: Required for ``kind="epoch"``, forbidden otherwise.

    Returns:
        The path; no directories are created and no I/O happens here.
    """
    _validate_model_key(model_key)
    if fold < 1:
        raise ValueError(f"fold is 1-based, got {fold}")
    if kind == "best":
        if epoch is not None:
            raise ValueError("epoch is meaningless for kind='best'")
        return output_dir / "models" / f"best_{model_key}_fold_{fold}_model.pth"
    if kind == "epoch":
        if epoch is None:
            raise ValueError("epoch is required for kind='epoch'")
        return (
            output_dir
            / "checkpoints"
            / f"{model_key}_fold_{fold}_epoch_{epoch:04d}.pth"
        )
    raise ValueError(f"unknown checkpoint kind {kind!r}; expected {_VALID_KINDS}")


def epoch_checkpoint_glob(output_dir: Path, model_key: str, fold: int) -> str:
    """Glob pattern matching every epoch checkpoint of one (key, fold) series.

    Used by resume discovery and old-checkpoint pruning.  Scoped to a single
    model and fold so pruning one model's history never deletes another's.
    """
    _validate_model_key(model_key)
    if fold < 1:
        raise ValueError(f"fold is 1-based, got {fold}")
    return str(
        output_dir / "checkpoints" / f"{model_key}_fold_{fold}_epoch_*.pth"
    )
