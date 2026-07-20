import json

import pytest

from topoworld.scenediffuserpp.evaluation import aggregate_evaluation_records
from topoworld.scenediffuserpp.evaluation import evaluation_config_from_mapping
from topoworld.scenediffuserpp.evaluation import EvaluationRecord
from topoworld.scenediffuserpp.evaluation import select_evaluation_samples
from topoworld.scenediffuserpp.evaluation import training_log_through_checkpoint
from topoworld.scenediffuserpp.evaluation import verify_checkpoint_manifest
from topoworld.scenediffuserpp.evaluation import verify_training_log


def _row(sample_id, run_id, split, shard="shard_00000.h5", sample_index=0):
    return {
        "sample_id": sample_id,
        "run_id": run_id,
        "split": split,
        "shard": shard,
        "sample_index": sample_index,
    }


def _manifest():
    return {
        "samples": [
            _row("train-0", "train-run", "train", sample_index=0),
            _row("val-a-0", "val-a", "validation", sample_index=1),
            _row("val-a-1", "val-a", "validation", sample_index=2),
            _row("val-b-0", "val-b", "validation", sample_index=3),
            _row("test-0", "test-run", "test", sample_index=4),
        ],
        "shards": {"shard_00000.h5": "sha256"},
    }


def test_selection_keeps_complete_held_out_run_groups():
    selected = select_evaluation_samples(
        _manifest(), split="validation", max_runs=1, minimum_samples=2
    )

    assert [sample.sample_id for sample in selected] == ["val-a-0", "val-a-1"]
    assert {sample.run_id for sample in selected} == {"val-a"}
    assert [sample.dataset_index for sample in selected] == [1, 2]


def test_selection_rejects_training_split_without_explicit_override():
    with pytest.raises(ValueError, match="allow_train"):
        select_evaluation_samples(_manifest(), split="train")

    selected = select_evaluation_samples(
        _manifest(), split="train", allow_train=True, minimum_samples=1
    )
    assert [sample.sample_id for sample in selected] == ["train-0"]


def test_selection_rejects_empty_or_too_small_held_out_split():
    with pytest.raises(ValueError, match="no samples"):
        select_evaluation_samples(_manifest(), split="missing")
    with pytest.raises(ValueError, match="at least 2"):
        select_evaluation_samples(_manifest(), split="test", minimum_samples=2)


def test_selection_rejects_run_split_leakage():
    manifest = _manifest()
    manifest["samples"].append(
        _row("leaked", "val-a", "test", sample_index=5)
    )

    with pytest.raises(ValueError, match="multiple splits"):
        select_evaluation_samples(manifest, split="validation")


def test_aggregate_requires_multiple_seeds_for_every_sample():
    record = EvaluationRecord(
        sample_id="val-a-0",
        run_id="val-a",
        seed=7,
        report={"status": "passed", "failures": [], "loss_ratio": 0.1},
    )

    with pytest.raises(ValueError, match="at least 2 seeds"):
        aggregate_evaluation_records([record], provenance={})


def test_aggregate_reports_pass_rate_failures_and_provenance():
    records = []
    for sample_id in ("val-a-0", "val-a-1"):
        for seed in (7, 8):
            failed = sample_id == "val-a-1" and seed == 8
            records.append(
                EvaluationRecord(
                    sample_id=sample_id,
                    run_id="val-a",
                    seed=seed,
                    report={
                        "status": "blocked" if failed else "passed",
                        "failures": (
                            [{"code": "implausible_generated_acceleration"}]
                            if failed
                            else []
                        ),
                        "loss_ratio": 0.1,
                        "generated_agents": {
                            "birth_transitions": 0,
                            "removal_transitions": 1 if failed else 0,
                        },
                        "target_agents": {
                            "birth_transitions": 1,
                            "removal_transitions": 1,
                        },
                        "generated_lights": {
                            "birth_transitions": 0,
                            "removal_transitions": 0,
                            "state_change_transitions": 1,
                        },
                        "target_lights": {
                            "birth_transitions": 0,
                            "removal_transitions": 0,
                            "state_change_transitions": 2,
                        },
                    },
                )
            )

    report = aggregate_evaluation_records(
        records,
        provenance={
            "split": "validation",
            "weights": "ema",
            "checkpoint_sha256": "checkpoint-hash",
            "manifest_sha256": "manifest-hash",
        },
    )

    assert report["status"] == "blocked"
    assert report["record_count"] == 4
    assert report["sample_count"] == 2
    assert report["run_count"] == 1
    assert report["pass_rate"] == pytest.approx(0.75)
    assert report["failure_counts"] == {"implausible_generated_acceleration": 1}
    assert report["transition_totals_by_record"]["agents"] == {
        "generated_birth_transitions": 0,
        "target_birth_transitions": 4,
        "birth_transition_error": 4,
        "generated_removal_transitions": 1,
        "target_removal_transitions": 4,
        "removal_transition_error": 3,
    }
    assert report["transition_totals_by_record"]["lights"][
        "state_change_transition_error"
    ] == 4
    assert report["provenance"]["weights"] == "ema"


def test_aggregate_blocks_nonfinite_metric_even_when_record_claims_passed():
    records = [
        EvaluationRecord(
            sample_id="val-a-0",
            run_id="val-a",
            seed=seed,
            report={"status": "passed", "failures": [], "loss_ratio": float("nan")},
        )
        for seed in (7, 8)
    ]

    report = aggregate_evaluation_records(records, provenance={})

    assert report["status"] == "blocked"
    assert report["pass_rate"] == 0.0
    assert report["failure_counts"] == {"nonfinite_report": 2}
    assert report["records"][0]["report"]["loss_ratio"] is None
    json.dumps(report, allow_nan=False)


def test_evaluation_config_requires_distinct_multi_seed_sampling():
    values = {
        "evaluation": {
            "split": "validation",
            "minimum_samples": 3,
            "seeds": [7, 8],
            "weights": "ema",
            "sampling_steps": 24,
            "thresholds": {"maximum_p95_acceleration_mps2": 12.0},
        }
    }

    config = evaluation_config_from_mapping(values)

    assert config.split == "validation"
    assert config.minimum_samples == 3
    assert config.seeds == (7, 8)
    assert config.sampling_steps == 24
    assert config.thresholds.maximum_p95_acceleration_mps2 == 12.0
    with pytest.raises(ValueError, match="distinct seeds"):
        evaluation_config_from_mapping(
            {"evaluation": {**values["evaluation"], "seeds": [7, 7]}}
        )


def test_checkpoint_must_match_evaluated_manifest():
    verify_checkpoint_manifest(
        {"manifest_hash": "manifest-hash"}, manifest_hash="manifest-hash"
    )

    with pytest.raises(ValueError, match="different dataset manifest"):
        verify_checkpoint_manifest(
            {"manifest_hash": "training-hash"}, manifest_hash="evaluation-hash"
        )


def test_training_log_must_be_complete_through_checkpoint():
    verify_training_log(
        [{"global_step": step} for step in range(1, 21)], checkpoint_step=20
    )

    with pytest.raises(ValueError, match="ends at step 19"):
        verify_training_log(
            [{"global_step": 18}, {"global_step": 19}], checkpoint_step=20
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        verify_training_log(
            [{"global_step": 19}, {"global_step": 19}], checkpoint_step=19
        )
    with pytest.raises(ValueError, match="start at step 1"):
        verify_training_log(
            [{"global_step": 18}, {"global_step": 19}, {"global_step": 20}],
            checkpoint_step=20,
        )
    with pytest.raises(ValueError, match="contiguous"):
        verify_training_log(
            [{"global_step": 1}, {"global_step": 3}], checkpoint_step=3
        )
    with pytest.raises(ValueError, match="missing global_step"):
        verify_training_log([{"total_loss": 1.0}], checkpoint_step=1)


def test_training_log_checkpoint_prefix_allows_later_contiguous_rows():
    rows = [{"global_step": step, "loss": float(step)} for step in range(1, 6)]

    prefix = training_log_through_checkpoint(rows, checkpoint_step=3)

    assert [row["global_step"] for row in prefix] == [1, 2, 3]
    with pytest.raises(ValueError, match="does not reach"):
        training_log_through_checkpoint(rows, checkpoint_step=6)
    with pytest.raises(ValueError, match="contiguous"):
        training_log_through_checkpoint(
            [{"global_step": 1}, {"global_step": 2}, {"global_step": 4}],
            checkpoint_step=2,
        )
