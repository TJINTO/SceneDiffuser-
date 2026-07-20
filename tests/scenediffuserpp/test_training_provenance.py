import json
from pathlib import Path

import pytest

from scenediffuserpp.training_provenance import build_checkpoint_run_config
from scenediffuserpp.training_provenance import verify_resume_contract
from scenediffuserpp.training_provenance import verify_training_start


def test_fresh_training_rejects_a_nonempty_existing_log(tmp_path: Path):
    log_path = tmp_path / "train.jsonl"
    log_path.write_text('{"global_step": 1}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="already contains"):
        verify_training_start(log_path, checkpoint_step=None)


def test_resume_requires_log_to_end_at_checkpoint_step(tmp_path: Path):
    log_path = tmp_path / "train.jsonl"
    log_path.write_text(
        "".join(
            json.dumps({"global_step": step}) + "\n" for step in range(1, 4)
        ),
        encoding="utf-8",
    )

    verify_training_start(log_path, checkpoint_step=3)
    with pytest.raises(ValueError, match="checkpoint is step 2"):
        verify_training_start(log_path, checkpoint_step=2)


def test_resume_requires_existing_complete_log(tmp_path: Path):
    with pytest.raises(ValueError, match="requires an existing training log"):
        verify_training_start(tmp_path / "missing.jsonl", checkpoint_step=3)


def test_resume_contract_rejects_changed_mask_semantics():
    checkpoint = {
        "training": {
            "task": "mixed",
            "history_steps": 11,
            "scene_generation_probability": 0.5,
            "control_feature_probability": 0.5,
        },
        "model": {"hidden_dim": 512},
        "tensor_contract": {"agents": [32, 91, 12]},
    }
    current_training = {
        **checkpoint["training"],
        "control_feature_probability": 0.75,
    }

    with pytest.raises(ValueError, match="training configuration"):
        verify_resume_contract(
            checkpoint,
            training=current_training,
            model=checkpoint["model"],
            tensor_contract=checkpoint["tensor_contract"],
        )


def test_resume_contract_accepts_identical_semantics():
    checkpoint = {
        "training": {"task": "mixed", "history_steps": 11},
        "model": {"hidden_dim": 512},
        "tensor_contract": {"agents": [32, 91, 12]},
    }

    verify_resume_contract(
        checkpoint,
        training=checkpoint["training"],
        model=checkpoint["model"],
        tensor_contract=checkpoint["tensor_contract"],
    )


def test_checkpoint_run_config_preserves_optimizer_provenance():
    optimizer_provenance = {
        "implementation": "scenediffuserpp.trainer.PaperAdafactor",
        "decay_adam": 0.9999,
        "paper_decay_adam_0_9999_available": True,
    }

    run_config = build_checkpoint_run_config(
        training={"history_steps": 11},
        model={"validity_mode": "paper"},
        dataset="dataset-root",
        tensor_contract={"agents": [128, 91, 12]},
        incomplete_corpus_override=True,
        optimizer_provenance=optimizer_provenance,
        execution={"effective_max_steps": 2},
    )

    assert run_config["optimizer_provenance"] == optimizer_provenance
    assert run_config["training"] == {"history_steps": 11}
    assert run_config["model"] == {"validity_mode": "paper"}
    assert run_config["dataset"] == "dataset-root"
    assert run_config["tensor_contract"] == {"agents": [128, 91, 12]}
    assert run_config["incomplete_corpus_override"] is True
    assert run_config["execution"] == {"effective_max_steps": 2}
