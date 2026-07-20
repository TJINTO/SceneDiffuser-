from __future__ import annotations

import math
from collections import Counter
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from numbers import Real
from typing import Any
from typing import Mapping
from typing import Sequence

from scenediffuserpp.short_evaluation import ShortEvaluationThresholds


@dataclass(frozen=True)
class HeldoutEvaluationConfig:
    split: str = "validation"
    max_runs: int | None = None
    minimum_samples: int = 2
    seeds: tuple[int, ...] = (7, 8)
    weights: str = "ema"
    sampling_steps: int = 32
    allow_train: bool = False
    validity_mode: str | None = None
    max_speed_mps: float | None = None
    max_acceleration_mps2: float | None = None
    max_jerk_mps3: float | None = None
    speed_limit_margin: float = 1.0
    timestep_seconds: float = 0.1
    thresholds: ShortEvaluationThresholds = field(
        default_factory=ShortEvaluationThresholds
    )

    def __post_init__(self) -> None:
        if not self.split:
            raise ValueError("evaluation split cannot be empty")
        if self.max_runs is not None and self.max_runs <= 0:
            raise ValueError("max_runs must be positive")
        if self.minimum_samples <= 0:
            raise ValueError("minimum_samples must be positive")
        if len(self.seeds) < 2 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("evaluation requires at least two distinct seeds")
        if self.weights not in {"raw", "ema"}:
            raise ValueError("weights must be 'raw' or 'ema'")
        if self.sampling_steps <= 0:
            raise ValueError("sampling_steps must be positive")
        if self.validity_mode not in {None, "paper", "signed_stable"}:
            raise ValueError("unsupported validity_mode")
        if self.max_speed_mps is not None and self.max_speed_mps <= 0.0:
            raise ValueError("max_speed_mps must be positive")
        if (
            self.max_acceleration_mps2 is not None
            and self.max_acceleration_mps2 <= 0.0
        ):
            raise ValueError("max_acceleration_mps2 must be positive")
        if self.max_jerk_mps3 is not None and self.max_jerk_mps3 <= 0.0:
            raise ValueError("max_jerk_mps3 must be positive")
        if self.speed_limit_margin <= 0.0 or self.timestep_seconds <= 0.0:
            raise ValueError("speed-limit margin and timestep must be positive")


@dataclass(frozen=True)
class SelectedEvaluationSample:
    sample_id: str
    run_id: str
    split: str
    dataset_index: int
    manifest_row: Mapping[str, Any]


@dataclass(frozen=True)
class EvaluationRecord:
    sample_id: str
    run_id: str
    seed: int
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.sample_id or not self.run_id:
            raise ValueError("evaluation records require sample_id and run_id")


def evaluation_config_from_mapping(
    values: Mapping[str, Any],
) -> HeldoutEvaluationConfig:
    section = values.get("evaluation")
    if not isinstance(section, Mapping):
        raise ValueError("evaluation config must contain an 'evaluation' mapping")
    threshold_values = section.get("thresholds", {})
    if not isinstance(threshold_values, Mapping):
        raise ValueError("evaluation thresholds must be a mapping")
    try:
        thresholds = ShortEvaluationThresholds(**dict(threshold_values))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid evaluation thresholds: {exc}") from exc
    seeds = section.get("seeds", (7, 8))
    if not isinstance(seeds, (list, tuple)):
        raise ValueError("evaluation seeds must be a sequence")
    max_runs_value = section.get("max_runs")
    validity_mode_value = section.get("validity_mode")
    max_speed_value = section.get("max_speed_mps")
    max_acceleration_value = section.get("max_acceleration_mps2")
    max_jerk_value = section.get("max_jerk_mps3")
    return HeldoutEvaluationConfig(
        split=str(section.get("split", "validation")),
        max_runs=(None if max_runs_value is None else int(max_runs_value)),
        minimum_samples=int(section.get("minimum_samples", 2)),
        seeds=tuple(int(seed) for seed in seeds),
        weights=str(section.get("weights", "ema")),
        sampling_steps=int(section.get("sampling_steps", 32)),
        allow_train=bool(section.get("allow_train", False)),
        validity_mode=(
            None if validity_mode_value is None else str(validity_mode_value)
        ),
        max_speed_mps=(
            None if max_speed_value is None else float(max_speed_value)
        ),
        max_acceleration_mps2=(
            None
            if max_acceleration_value is None
            else float(max_acceleration_value)
        ),
        max_jerk_mps3=(
            None if max_jerk_value is None else float(max_jerk_value)
        ),
        speed_limit_margin=float(section.get("speed_limit_margin", 1.0)),
        timestep_seconds=float(section.get("timestep_seconds", 0.1)),
        thresholds=thresholds,
    )


def verify_checkpoint_manifest(
    checkpoint_metadata: Mapping[str, Any], *, manifest_hash: str
) -> None:
    checkpoint_hash = str(checkpoint_metadata.get("manifest_hash", ""))
    if not checkpoint_hash or checkpoint_hash != manifest_hash:
        raise ValueError("checkpoint was trained on a different dataset manifest")


def verify_training_log(
    rows: Sequence[Mapping[str, Any]], *, checkpoint_step: int
) -> None:
    if checkpoint_step <= 0:
        raise ValueError("checkpoint_step must be positive")
    if not rows:
        raise ValueError("training log is empty")
    steps: list[int] = []
    for index, row in enumerate(rows):
        value = row.get("global_step")
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"training log row {index} is missing global_step")
        steps.append(value)
    if any(current <= previous for previous, current in zip(steps, steps[1:])):
        raise ValueError("training log global_step values must be strictly increasing")
    if steps[-1] != checkpoint_step:
        raise ValueError(
            f"training log ends at step {steps[-1]}, but checkpoint is step "
            f"{checkpoint_step}"
        )
    if steps[0] != 1:
        raise ValueError("training log must start at step 1")
    if steps != list(range(1, checkpoint_step + 1)):
        raise ValueError("training log global_step values must be contiguous")


def training_log_through_checkpoint(
    rows: Sequence[Mapping[str, Any]], *, checkpoint_step: int
) -> tuple[Mapping[str, Any], ...]:
    if checkpoint_step <= 0:
        raise ValueError("checkpoint_step must be positive")
    if not rows:
        raise ValueError("training log is empty")
    final_step = rows[-1].get("global_step")
    if isinstance(final_step, bool) or not isinstance(final_step, int):
        raise ValueError("training log final row is missing global_step")
    verify_training_log(rows, checkpoint_step=final_step)
    if final_step < checkpoint_step:
        raise ValueError(
            f"training log does not reach checkpoint step {checkpoint_step}; "
            f"it ends at {final_step}"
        )
    prefix = tuple(rows[:checkpoint_step])
    verify_training_log(prefix, checkpoint_step=checkpoint_step)
    return prefix


def select_evaluation_samples(
    manifest: Mapping[str, Any],
    *,
    split: str,
    max_runs: int | None = None,
    minimum_samples: int = 2,
    allow_train: bool = False,
) -> tuple[SelectedEvaluationSample, ...]:
    if not split:
        raise ValueError("evaluation split cannot be empty")
    if split == "train" and not allow_train:
        raise ValueError("train evaluation requires allow_train=True")
    if max_runs is not None and max_runs <= 0:
        raise ValueError("max_runs must be positive")
    if minimum_samples <= 0:
        raise ValueError("minimum_samples must be positive")

    rows = manifest.get("samples")
    if not isinstance(rows, list):
        raise ValueError("manifest samples must be a list")
    dataset_indices = _dataset_indices(manifest, rows)
    run_splits: dict[str, set[str]] = defaultdict(set)
    grouped: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    sample_ids: set[str] = set()
    for row_index, row in enumerate(rows):
        sample_id = str(row.get("sample_id", ""))
        run_id = str(row.get("run_id", ""))
        row_split = str(row.get("split", ""))
        if not sample_id or not run_id or not row_split:
            raise ValueError(f"manifest row {row_index} has empty identity fields")
        if sample_id in sample_ids:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        sample_ids.add(sample_id)
        run_splits[run_id].add(row_split)
        if row_split == split:
            grouped[run_id].append((row_index, row))
    leaked = sorted(run_id for run_id, splits in run_splits.items() if len(splits) > 1)
    if leaked:
        raise ValueError(f"runs occur in multiple splits: {leaked}")
    if not grouped:
        raise ValueError(f"no samples found for split {split!r}")

    selected_runs = sorted(grouped)
    if max_runs is not None:
        selected_runs = selected_runs[:max_runs]
    selected: list[SelectedEvaluationSample] = []
    for run_id in selected_runs:
        for row_index, row in sorted(
            grouped[run_id], key=lambda item: str(item[1]["sample_id"])
        ):
            selected.append(
                SelectedEvaluationSample(
                    sample_id=str(row["sample_id"]),
                    run_id=run_id,
                    split=split,
                    dataset_index=dataset_indices[row_index],
                    manifest_row=row,
                )
            )
    if len(selected) < minimum_samples:
        raise ValueError(
            f"split {split!r} requires at least {minimum_samples} samples; "
            f"found {len(selected)}"
        )
    return tuple(selected)


def aggregate_evaluation_records(
    records: Sequence[EvaluationRecord],
    *,
    provenance: Mapping[str, Any],
    minimum_seeds_per_sample: int = 2,
) -> dict[str, Any]:
    if not records:
        raise ValueError("evaluation records cannot be empty")
    if minimum_seeds_per_sample <= 0:
        raise ValueError("minimum_seeds_per_sample must be positive")
    seeds_by_sample: dict[str, set[int]] = defaultdict(set)
    seen: set[tuple[str, int]] = set()
    for record in records:
        identity = (record.sample_id, int(record.seed))
        if identity in seen:
            raise ValueError(f"duplicate evaluation record: {identity}")
        seen.add(identity)
        seeds_by_sample[record.sample_id].add(int(record.seed))
    undersampled = sorted(
        sample_id
        for sample_id, seeds in seeds_by_sample.items()
        if len(seeds) < minimum_seeds_per_sample
    )
    if undersampled:
        raise ValueError(
            f"every sample requires at least {minimum_seeds_per_sample} seeds: "
            f"{undersampled}"
        )

    nonfinite_by_record = [_contains_nonfinite(record.report) for record in records]
    passed = sum(
        record.report.get("status") == "passed" and not contains_nonfinite
        for record, contains_nonfinite in zip(records, nonfinite_by_record)
    )
    failure_counts: Counter[str] = Counter()
    for record, contains_nonfinite in zip(records, nonfinite_by_record):
        if contains_nonfinite:
            failure_counts["nonfinite_report"] += 1
        for failure in record.report.get("failures", []):
            failure_counts[str(failure.get("code", "unknown_failure"))] += 1
    return {
        "status": "passed" if passed == len(records) else "blocked",
        "record_count": len(records),
        "sample_count": len(seeds_by_sample),
        "run_count": len({record.run_id for record in records}),
        "pass_rate": passed / len(records),
        "failure_counts": dict(sorted(failure_counts.items())),
        "transition_totals_by_record": _transition_totals(records),
        "provenance": _json_safe(dict(provenance)),
        "records": [
            {
                "sample_id": record.sample_id,
                "run_id": record.run_id,
                "seed": int(record.seed),
                "report": _json_safe(dict(record.report)),
            }
            for record in records
        ],
    }


def _transition_totals(records: Sequence[EvaluationRecord]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for kind in ("agents", "lights"):
        generated_key = f"generated_{kind}"
        target_key = f"target_{kind}"
        totals: dict[str, int] = {}
        for transition in (
            "birth_transitions",
            "removal_transitions",
            "state_change_transitions",
        ):
            generated = _sum_transition(records, generated_key, transition)
            target = _sum_transition(records, target_key, transition)
            if generated is None and target is None:
                continue
            generated = generated or 0
            target = target or 0
            short_name = transition.removesuffix("s")
            totals[f"generated_{transition}"] = generated
            totals[f"target_{transition}"] = target
            totals[f"{short_name}_error"] = abs(generated - target)
        result[kind] = totals
    return result


def _sum_transition(
    records: Sequence[EvaluationRecord], tensor_key: str, transition_key: str
) -> int | None:
    total = 0
    seen = False
    for record in records:
        tensor_report = record.report.get(tensor_key)
        if not isinstance(tensor_report, Mapping):
            continue
        value = tensor_report.get(transition_key)
        if isinstance(value, bool) or not isinstance(value, Real):
            continue
        if not math.isfinite(float(value)):
            continue
        total += int(value)
        seen = True
    return total if seen else None


def _contains_nonfinite(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, Real):
        return not math.isfinite(float(value))
    if isinstance(value, Mapping):
        return any(_contains_nonfinite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_nonfinite(item) for item in value)
    return False


def _json_safe(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, Real):
        if not math.isfinite(float(value)):
            return None
        return value if isinstance(value, (int, float)) else float(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _dataset_indices(
    manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[int, int]:
    shards = manifest.get("shards")
    if not isinstance(shards, Mapping) or not shards:
        raise ValueError("manifest shards must be a nonempty mapping")
    rows_by_shard: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        shard = str(row.get("shard", ""))
        if shard not in shards:
            raise ValueError(f"manifest row references unknown shard: {shard!r}")
        rows_by_shard[shard].append((row_index, row))

    result: dict[int, int] = {}
    dataset_index = 0
    for shard in sorted(shards):
        shard_rows = sorted(
            rows_by_shard.get(shard, []), key=lambda item: int(item[1]["sample_index"])
        )
        sample_indices = [int(row["sample_index"]) for _, row in shard_rows]
        if sample_indices != list(range(len(sample_indices))):
            raise ValueError(f"shard {shard!r} sample indices are not contiguous")
        for row_index, _row in shard_rows:
            result[row_index] = dataset_index
            dataset_index += 1
    if len(result) != len(rows):
        raise ValueError("could not map every manifest row to a dataset index")
    return result
