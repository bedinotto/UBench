"""UB-26: --models CLI choices are derived from the model registry.

The argparse `choices` used to be a hand-maintained `['unet','transunet','swin']`
list that rejected the T3.2 pretrained keys (`swin_pretrained`,
`transunet_pretrained`) the README and the GPU-box checklist advertise — the
same second-authority bug class as UB-02/UB-05. Choices now come from the
registry (+ the explicit `swin` alias).
"""

from __future__ import annotations

import argparse

import pytest

from codes.main_pipeline import _model_choices, _resolve_model_key
from codes.model_registry import get_registered_models


def test_choices_cover_every_registered_model():
    choices = set(_model_choices())
    assert set(get_registered_models()) <= choices
    assert "swin" in choices  # short alias for swin_unet_plus_plus


def test_resolve_alias_and_passthrough():
    assert _resolve_model_key("swin") == "swin_unet_plus_plus"
    assert _resolve_model_key("unet") == "unet"
    assert _resolve_model_key("swin_pretrained") == "swin_pretrained"


@pytest.mark.parametrize("key", sorted(get_registered_models()) + ["swin"])
def test_argparse_accepts_valid_keys(key):
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=_model_choices())
    ns = parser.parse_args(["--models", key])
    assert ns.models == [key]


def test_argparse_rejects_unknown_key():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=_model_choices())
    with pytest.raises(SystemExit):
        parser.parse_args(["--models", "not_a_model"])
