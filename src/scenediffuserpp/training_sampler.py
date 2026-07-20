from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


def resolve_diagnostic_sample_ids(
    training: Mapping[str, Any],
) -> tuple[str, ...] | None:
    sample_ids = training.get("diagnostic_sample_ids")
    if sample_ids is None:
        return None
    if (
        isinstance(sample_ids, (str, bytes))
        or not isinstance(sample_ids, Sequence)
        or not sample_ids
        or any(not isinstance(sample_id, str) or not sample_id for sample_id in sample_ids)
    ):
        raise ValueError("diagnostic_sample_ids must be a nonempty list of strings")
    if str(training.get("task", "")) != "behavior_prediction":
        raise ValueError("diagnostic_sample_ids require task=behavior_prediction")
    if training.get("diagnostic_fixed_diffusion_time") is None:
        raise ValueError("diagnostic_sample_ids require a fixed diffusion time")
    return tuple(sample_ids)


def select_training_indices(
    manifest: Mapping[str, Any],
    *,
    sample_ids: Sequence[str] | None = None,
) -> list[int]:
    shards = manifest.get("shards")
    if not isinstance(shards, Mapping) or not shards:
        raise ValueError("manifest shards must be a nonempty mapping")
    rows = manifest.get("samples")
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest samples must be a nonempty list")

    rows_by_shard: dict[str, list[tuple[int, Mapping[str, Any]]]] = {
        str(shard): [] for shard in shards
    }
    row_by_id: dict[str, tuple[int, Mapping[str, Any]]] = {}
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"manifest sample row {row_index} must be a mapping")
        sample_id = str(row.get("sample_id", ""))
        if not sample_id:
            raise ValueError(f"manifest sample row {row_index} has no sample_id")
        if sample_id in row_by_id:
            raise ValueError(f"manifest sample_id must be unique: {sample_id!r}")
        shard = str(row.get("shard", ""))
        if shard not in rows_by_shard:
            raise ValueError(f"manifest row references unknown shard: {shard!r}")
        row_by_id[sample_id] = (row_index, row)
        rows_by_shard[shard].append((row_index, row))

    dataset_index_by_row: dict[int, int] = {}
    dataset_index = 0
    for shard in sorted(rows_by_shard):
        shard_rows = sorted(
            rows_by_shard[shard], key=lambda item: int(item[1]["sample_index"])
        )
        sample_indices = [int(row["sample_index"]) for _, row in shard_rows]
        if sample_indices != list(range(len(sample_indices))):
            raise ValueError(f"shard {shard!r} sample indices are not contiguous")
        for row_index, _row in shard_rows:
            dataset_index_by_row[row_index] = dataset_index
            dataset_index += 1

    if sample_ids is None:
        return sorted(
            dataset_index_by_row[row_index]
            for row_index, row in enumerate(rows)
            if str(row.get("split", "")) == "train"
        )

    requested = tuple(str(sample_id) for sample_id in sample_ids)
    if not requested:
        raise ValueError("diagnostic sample_ids cannot be empty")
    if len(set(requested)) != len(requested):
        raise ValueError("diagnostic sample_ids must be unique")
    selected: list[int] = []
    for sample_id in requested:
        if sample_id not in row_by_id:
            raise ValueError(f"unknown diagnostic sample_id: {sample_id!r}")
        row_index, row = row_by_id[sample_id]
        if str(row.get("split", "")) != "train":
            raise ValueError(
                f"diagnostic sample_id {sample_id!r} is not in the training split"
            )
        selected.append(dataset_index_by_row[row_index])
    return selected


class DeterministicEpochBatchSampler:
    def __init__(self, indices: Sequence[int], *, batch_size: int, seed: int) -> None:
        self.indices = tuple(int(index) for index in indices)
        if not self.indices:
            raise ValueError("training indices cannot be empty")
        if len(set(self.indices)) != len(self.indices):
            raise ValueError("training indices must be unique")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self._cached_epoch: int | None = None
        self._cached_order: tuple[int, ...] = ()

    def batch(self, step: int) -> list[int]:
        if step < 0:
            raise ValueError("step must be nonnegative")
        start = int(step) * self.batch_size
        selected: list[int] = []
        for position in range(start, start + self.batch_size):
            epoch, offset = divmod(position, len(self.indices))
            selected.append(self._epoch_order(epoch)[offset])
        return selected

    def epoch_at_step(self, step: int) -> int:
        if step < 0:
            raise ValueError("step must be nonnegative")
        return int(step) * self.batch_size // len(self.indices)

    def _epoch_order(self, epoch: int) -> tuple[int, ...]:
        if epoch != self._cached_epoch:
            generator = torch.Generator().manual_seed(self.seed + epoch)
            order = torch.randperm(len(self.indices), generator=generator).tolist()
            self._cached_order = tuple(self.indices[index] for index in order)
            self._cached_epoch = epoch
        return self._cached_order
