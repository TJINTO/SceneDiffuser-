import pytest

from topoworld.scenediffuserpp.training_sampler import DeterministicEpochBatchSampler
from topoworld.scenediffuserpp.training_sampler import resolve_diagnostic_sample_ids
from topoworld.scenediffuserpp.training_sampler import select_training_indices


def test_epoch_sampler_shuffles_every_sample_once_and_is_resume_stable():
    indices = tuple(range(8))
    first = DeterministicEpochBatchSampler(indices, batch_size=2, seed=17)
    resumed = DeterministicEpochBatchSampler(indices, batch_size=2, seed=17)

    first_epoch = [index for step in range(4) for index in first.batch(step)]
    second_epoch = [index for step in range(4, 8) for index in first.batch(step)]

    assert sorted(first_epoch) == list(indices)
    assert sorted(second_epoch) == list(indices)
    assert first_epoch != list(indices)
    assert first_epoch != second_epoch
    assert resumed.batch(5) == first.batch(5)


def test_epoch_sampler_supports_batches_crossing_epoch_boundaries():
    sampler = DeterministicEpochBatchSampler((10, 11, 12), batch_size=2, seed=5)

    assert len(sampler.batch(1)) == 2
    assert sampler.batch(1)[0] in {10, 11, 12}
    assert sampler.batch(1)[1] in {10, 11, 12}


def test_training_indices_follow_shard_sample_order_not_manifest_row_order():
    manifest = {
        "shards": {"b.h5": "hash-b", "a.h5": "hash-a"},
        "samples": [
            _row("b0", "b.h5", 0),
            _row("a1", "a.h5", 1),
            _row("a0", "a.h5", 0),
        ],
    }

    assert select_training_indices(manifest) == [0, 1, 2]
    assert select_training_indices(manifest, sample_ids=("b0", "a1")) == [2, 1]


def test_training_sample_selection_rejects_unknown_nontrain_or_duplicate_ids():
    manifest = {
        "shards": {"a.h5": "hash-a"},
        "samples": [
            _row("train", "a.h5", 0),
            _row("validation", "a.h5", 1, split="validation"),
        ],
    }

    with pytest.raises(ValueError, match="unknown diagnostic sample_id"):
        select_training_indices(manifest, sample_ids=("missing",))
    with pytest.raises(ValueError, match="not in the training split"):
        select_training_indices(manifest, sample_ids=("validation",))
    with pytest.raises(ValueError, match="must be unique"):
        select_training_indices(manifest, sample_ids=("train", "train"))


def test_diagnostic_sample_ids_require_bp_at_fixed_diffusion_time():
    training = {
        "task": "behavior_prediction",
        "diagnostic_fixed_diffusion_time": 1.0,
        "diagnostic_sample_ids": ["sample-a"],
    }

    assert resolve_diagnostic_sample_ids(training) == ("sample-a",)

    with pytest.raises(ValueError, match="behavior_prediction"):
        resolve_diagnostic_sample_ids({**training, "task": "mixed"})
    with pytest.raises(ValueError, match="fixed diffusion time"):
        resolve_diagnostic_sample_ids(
            {key: value for key, value in training.items() if key != "diagnostic_fixed_diffusion_time"}
        )


@pytest.mark.parametrize("sample_ids", [[], "sample-a", [""], ["sample-a", 2]])
def test_diagnostic_sample_ids_reject_invalid_values(sample_ids):
    with pytest.raises(ValueError, match="diagnostic_sample_ids"):
        resolve_diagnostic_sample_ids(
            {
                "task": "behavior_prediction",
                "diagnostic_fixed_diffusion_time": 1.0,
                "diagnostic_sample_ids": sample_ids,
            }
        )


def _row(
    sample_id: str,
    shard: str,
    sample_index: int,
    *,
    split: str = "train",
) -> dict:
    return {
        "sample_id": sample_id,
        "shard": shard,
        "sample_index": sample_index,
        "split": split,
    }
