"""UB-12 (T3.1): the single validated config authority.

``codes/config.yaml`` is loaded through a pydantic schema
(``codes/config_schema.py``) so that unknown keys and wrong-typed values raise
at startup instead of being silently ignored by the old ``dict.get(key,
default)`` lookups. Every wired key must also demonstrably change behavior —
that is the other half of the UB-12 failure mode (dead keys) being closed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from codes.config_schema import RootConfig, load_config

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
