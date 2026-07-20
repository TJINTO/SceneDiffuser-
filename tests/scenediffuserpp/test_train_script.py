from pathlib import Path
import importlib.util

import pytest


def _load_train_script():
    path = Path("scripts/scenediffuserpp/train.py")
    spec = importlib.util.spec_from_file_location("scenediffuserpp_train_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_training_loss_settings_default_to_paper_sparse_weighting():
    train_script = _load_train_script()

    settings = train_script._training_loss_settings({})

    assert settings == {"validity_transition_weight": 1.0}


def test_training_loss_settings_reads_validity_transition_weight():
    train_script = _load_train_script()

    settings = train_script._training_loss_settings(
        {"validity_transition_weight": "12.5"}
    )

    assert settings == {"validity_transition_weight": 12.5}


def test_training_loss_settings_rejects_invalid_validity_transition_weight():
    train_script = _load_train_script()

    with pytest.raises(ValueError, match="validity_transition_weight"):
        train_script._training_loss_settings({"validity_transition_weight": 0.0})


def test_checkpoint_every_runtime_override_is_validated():
    train_script = _load_train_script()

    assert train_script._resolve_checkpoint_every({"checkpoint_every": 50}) == 50
    assert (
        train_script._resolve_checkpoint_every(
            {"checkpoint_every": 50}, override=500
        )
        == 500
    )
    with pytest.raises(ValueError, match="checkpoint_every"):
        train_script._resolve_checkpoint_every({"checkpoint_every": 50}, override=0)
