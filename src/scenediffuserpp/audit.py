from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

from scenediffuserpp.normalization import POSITION_SCALE
from scenediffuserpp.scene_builder import count_light_state_transitions
from scenediffuserpp.storage import SUPPORTED_SCHEMA_VERSIONS
from scenediffuserpp.storage import SceneDataset
from scenediffuserpp.storage import canonical_json_sha256
from scenediffuserpp.storage import file_sha256


DEFAULT_THRESHOLDS = {
    "max_implied_speed_mps": 80.0,
    "max_agent_truncation_fraction": 0.05,
    "max_light_truncation_fraction": 0.05,
    "max_map_truncation_fraction": 0.05,
    "max_map_lane_truncation_fraction": 0.05,
    "max_map_connection_truncation_fraction": 0.05,
    "min_samples_with_agent_birth_fraction": 0.0,
    "min_samples_with_agent_removal_fraction": 0.0,
    "min_agent_birth_transition_count": 0,
    "min_agent_removal_transition_count": 0,
    "min_samples_with_valid_light_fraction": 0.0,
    "min_samples_with_light_transition_fraction": 0.0,
    "min_light_state_transition_count": 0,
}
INCOMPLETE_CORPUS_FAILURE_CODES = frozenset(
    {
        "incomplete_configured_run_grid",
        "missing_validation_split",
        "missing_test_split",
        "insufficient_teacher_warmup",
        "insufficient_teacher_recording_duration",
    }
)


def has_only_incomplete_corpus_failures(report: dict[str, Any]) -> bool:
    codes = {
        str(issue.get("code", ""))
        for issue in report.get("failures", [])
        if isinstance(issue, dict)
    }
    return bool(codes) and codes <= INCOMPLETE_CORPUS_FAILURE_CODES


def audit_dataset(
    dataset_dir: str | Path,
    *,
    verify_hashes: bool = True,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    root = Path(dataset_dir)
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return _report(
            failures=[_issue("missing_manifest", f"missing {manifest_path}")],
            warnings=[],
            statistics={"sample_count": 0},
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
        failures.append(_issue("schema_version", "unsupported dataset schema"))
    rows = manifest.get("samples", [])
    shard_names = sorted(manifest.get("shards", {}))
    shard_paths = [root / name for name in shard_names]
    if not rows or not shard_paths:
        failures.append(_issue("empty_dataset", "dataset has no samples or shards"))
        return _report(failures, warnings, {"sample_count": 0})

    _check_manifest(rows, failures)
    _check_manifest_provenance(manifest, rows, failures)
    corpus_statistics = _check_configured_corpus(manifest, rows, failures)
    teacher_statistics = _check_teacher_manifests(
        manifest,
        root=root,
        verify_hashes=verify_hashes,
        failures=failures,
    )
    if verify_hashes:
        for name, expected in manifest.get("shards", {}).items():
            path = root / name
            if not path.is_file() or file_sha256(path) != expected:
                failures.append(
                    _issue("shard_hash_mismatch", f"shard hash mismatch: {name}")
                )

    try:
        dataset = SceneDataset(shard_paths)
    except Exception as exc:
        failures.append(_issue("unreadable_shard", str(exc)))
        return _report(failures, warnings, {"sample_count": 0})

    if len(dataset) != len(rows):
        failures.append(
            _issue(
                "manifest_sample_mismatch",
                f"manifest has {len(rows)} rows but shards contain {len(dataset)} samples",
            )
        )

    active_counts: list[int] = []
    entering = exiting = tracked_agents = 0
    selected_agents = selected_lights = selected_map = 0
    truncated_agents = truncated_lights = truncated_map = 0
    selected_map_lanes = selected_map_connections = 0
    truncated_map_lanes = truncated_map_connections = 0
    sample_hashes: dict[str, int] = {}
    network_paths: set[Path] = set()
    tls_sources: set[tuple[Path, int]] = set()
    samples_with_agent_birth = 0
    samples_with_agent_removal = 0
    agent_birth_transition_count = 0
    agent_removal_transition_count = 0
    agent_multi_segment_count = 0
    samples_with_valid_light = 0
    samples_with_light_transition = 0
    light_state_transition_count = 0
    effective_build_config = manifest.get("effective_build_config", {})
    minimum_transitions_per_sample = int(
        effective_build_config.get("minimum_light_state_transitions", 0)
    )

    for index in range(len(dataset)):
        sample = dataset[index]
        _check_sample(sample, index, failures, limits)
        digest = _sample_hash(sample)
        if digest in sample_hashes:
            failures.append(
                _issue(
                    "duplicate_sample",
                    f"samples {sample_hashes[digest]} and {index} have identical tensors",
                )
            )
        else:
            sample_hashes[digest] = index

        agent_valid = sample["agents"][..., -1] > 0.0
        light_valid = sample["lights"][..., -1] > 0.0
        sample_agent_births = int((~agent_valid[:, :-1] & agent_valid[:, 1:]).sum())
        sample_agent_removals = int((agent_valid[:, :-1] & ~agent_valid[:, 1:]).sum())
        samples_with_agent_birth += int(sample_agent_births > 0)
        samples_with_agent_removal += int(sample_agent_removals > 0)
        agent_birth_transition_count += sample_agent_births
        agent_removal_transition_count += sample_agent_removals
        sample_light_transitions = count_light_state_transitions(sample["lights"])
        samples_with_valid_light += int(light_valid.any())
        samples_with_light_transition += int(sample_light_transitions > 0)
        light_state_transition_count += sample_light_transitions
        if sample_light_transitions < minimum_transitions_per_sample:
            failures.append(
                _issue(
                    "insufficient_sample_light_state_transitions",
                    f"sample {index} has {sample_light_transitions} light-state transitions; expected at least {minimum_transitions_per_sample}",
                )
            )
        active_counts.extend(agent_valid.sum(axis=0).astype(int).tolist())
        for slot in range(agent_valid.shape[0]):
            states = agent_valid[slot]
            if not states.any():
                continue
            tracked_agents += 1
            entering += int((~states[:-1] & states[1:]).any())
            exiting += int((states[:-1] & ~states[1:]).any())
            agent_multi_segment_count += int(_valid_segment_count(states) > 1)
        selected_agents += sum(bool(value) for value in sample["agent_ids"])
        selected_lights += sum(bool(value) for value in sample["light_ids"])
        selected_map += int((~sample["map_padding_mask"]).sum())
        if "map_lane_padding_mask" in sample:
            selected_map_lanes += int((~sample["map_lane_padding_mask"]).sum())
            selected_map_connections += int(
                (~sample["map_successor_padding_mask"]).sum()
            )
            truncated_map_lanes += int(sample["truncated_map_lanes"])
            truncated_map_connections += int(
                sample["truncated_map_connections"]
            )
        truncated_agents += sample["truncated_agents"]
        truncated_lights += sample["truncated_lights"]
        truncated_map += sample["truncated_map_points"]
        net_file = sample["meta"].get("net_file")
        if net_file:
            network_paths.add(Path(net_file))
        tls_file = sample["meta"].get("tls_file")
        if tls_file:
            tls_sources.add(
                (Path(tls_file), int(sample["meta"].get("frequency_hz", 10)))
            )

    for net_file in sorted(network_paths):
        _check_network_coordinates(net_file, failures)
    for tls_file, frequency_hz in sorted(tls_sources):
        _check_tls_source(tls_file, frequency_hz, failures)
    if not network_paths:
        warnings.append(
            _issue("missing_network_provenance", "samples do not record net_file")
        )

    statistics = {
        "sample_count": len(dataset),
        **corpus_statistics,
        **teacher_statistics,
        "active_agents_p10": _percentile(active_counts, 10),
        "active_agents_p50": _percentile(active_counts, 50),
        "active_agents_p90": _percentile(active_counts, 90),
        "tracked_agent_count": tracked_agents,
        "entering_agent_fraction": entering / max(tracked_agents, 1),
        "exiting_agent_fraction": exiting / max(tracked_agents, 1),
        "samples_with_agent_birth_fraction": samples_with_agent_birth
        / max(len(dataset), 1),
        "samples_with_agent_removal_fraction": samples_with_agent_removal
        / max(len(dataset), 1),
        "agent_birth_transition_count": agent_birth_transition_count,
        "agent_removal_transition_count": agent_removal_transition_count,
        "agent_multi_segment_count": agent_multi_segment_count,
        "agent_truncation_fraction": _fraction(truncated_agents, selected_agents),
        "light_truncation_fraction": _fraction(truncated_lights, selected_lights),
        "map_truncation_fraction": _fraction(truncated_map, selected_map),
        "map_lane_truncation_fraction": _fraction(
            truncated_map_lanes, selected_map_lanes
        ),
        "map_connection_truncation_fraction": _fraction(
            truncated_map_connections, selected_map_connections
        ),
        "network_count": len(network_paths),
        "samples_with_valid_light_fraction": samples_with_valid_light
        / max(len(dataset), 1),
        "samples_with_light_transition_fraction": samples_with_light_transition
        / max(len(dataset), 1),
        "light_state_transition_count": light_state_transition_count,
    }
    for name in ("agent", "light", "map", "map_lane", "map_connection"):
        value = statistics[f"{name}_truncation_fraction"]
        maximum = limits[f"max_{name}_truncation_fraction"]
        if value > maximum:
            failures.append(
                _issue(
                    f"excessive_{name}_truncation",
                    f"{name} truncation {value:.3f} exceeds {maximum:.3f}",
                )
            )
    if statistics["active_agents_p50"] < 2.0:
        warnings.append(
            _issue("low_agent_density", "median active-agent count is below 2")
        )
    agent_lifecycle_requirements = (
        (
            "samples_with_agent_birth_fraction",
            "min_samples_with_agent_birth_fraction",
            "insufficient_agent_birth_coverage",
        ),
        (
            "samples_with_agent_removal_fraction",
            "min_samples_with_agent_removal_fraction",
            "insufficient_agent_removal_coverage",
        ),
        (
            "agent_birth_transition_count",
            "min_agent_birth_transition_count",
            "insufficient_agent_birth_transitions",
        ),
        (
            "agent_removal_transition_count",
            "min_agent_removal_transition_count",
            "insufficient_agent_removal_transitions",
        ),
    )
    for statistic, threshold, code in agent_lifecycle_requirements:
        if statistics[statistic] < limits[threshold]:
            failures.append(
                _issue(
                    code,
                    f"{statistic} {statistics[statistic]:.3f} is below {limits[threshold]:.3f}",
                )
            )
    light_requirements = (
        (
            "samples_with_valid_light_fraction",
            "min_samples_with_valid_light_fraction",
            "insufficient_valid_light_coverage",
        ),
        (
            "samples_with_light_transition_fraction",
            "min_samples_with_light_transition_fraction",
            "insufficient_light_transition_coverage",
        ),
        (
            "light_state_transition_count",
            "min_light_state_transition_count",
            "insufficient_light_state_transitions",
        ),
    )
    for statistic, threshold, code in light_requirements:
        if statistics[statistic] < limits[threshold]:
            failures.append(
                _issue(
                    code,
                    f"{statistic} {statistics[statistic]:.3f} is below {limits[threshold]:.3f}",
                )
            )
    return _report(failures, warnings, statistics)


def _valid_segment_count(states: np.ndarray) -> int:
    values = np.asarray(states, dtype=bool)
    if values.ndim != 1 or not values.any():
        return 0
    starts = values.copy()
    starts[1:] &= ~values[:-1]
    return int(starts.sum())


def _check_manifest(rows: list[dict[str, Any]], failures: list[dict[str, Any]]) -> None:
    run_splits: dict[str, set[str]] = {}
    sample_ids: set[str] = set()
    for row in rows:
        run_id = str(row.get("run_id", ""))
        split = str(row.get("split", ""))
        if split not in {"train", "validation", "test"}:
            failures.append(_issue("invalid_split", f"invalid split: {split!r}"))
        run_splits.setdefault(run_id, set()).add(split)
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in sample_ids:
            failures.append(
                _issue("duplicate_sample_id", f"duplicate or empty sample_id: {sample_id!r}")
            )
        sample_ids.add(sample_id)
    leaked = sorted(run_id for run_id, splits in run_splits.items() if len(splits) > 1)
    if leaked:
        failures.append(
            _issue("run_split_leakage", f"runs occur in multiple splits: {leaked}")
        )


def _check_manifest_provenance(
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> None:
    config = manifest.get("dataset_config")
    config_hash = manifest.get("dataset_config_sha256")
    if config is not None or config_hash is not None:
        if config is None or not config_hash:
            failures.append(
                _issue(
                    "incomplete_dataset_config_provenance",
                    "dataset config content and hash must both be present",
                )
            )
        elif canonical_json_sha256(config) != str(config_hash):
            failures.append(
                _issue(
                    "dataset_config_hash_mismatch",
                    "dataset config content differs from its canonical hash",
                )
            )

    declared_counts = manifest.get("split_counts")
    if declared_counts is not None:
        actual_counts = {
            split: sum(str(row.get("split", "")) == split for row in rows)
            for split in ("train", "validation", "test")
        }
        normalized_declared = {
            split: int(declared_counts.get(split, 0))
            for split in ("train", "validation", "test")
        }
        if normalized_declared != actual_counts:
            failures.append(
                _issue(
                    "split_count_mismatch",
                    f"declared splits {normalized_declared} differ from {actual_counts}",
                )
            )


def _check_configured_corpus(
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    run_ids_by_split = {
        split: {
            str(row.get("run_id", ""))
            for row in rows
            if str(row.get("split", "")) == split and str(row.get("run_id", ""))
        }
        for split in ("train", "validation", "test")
    }
    run_ids = set().union(*run_ids_by_split.values())
    configured_run_count = _configured_run_count(manifest.get("dataset_config"))
    statistics = {
        "configured_run_count": configured_run_count,
        "independent_run_count": len(run_ids),
        "split_counts": {
            split: len(run_ids_by_split[split])
            for split in ("train", "validation", "test")
        },
    }
    if configured_run_count is None:
        return statistics
    if len(run_ids) < configured_run_count:
        failures.append(
            _issue(
                "incomplete_configured_run_grid",
                f"dataset has {len(run_ids)} independent runs but configuration declares {configured_run_count}",
            )
        )
    if configured_run_count >= 3:
        for split in ("validation", "test"):
            if not run_ids_by_split[split]:
                failures.append(
                    _issue(
                        f"missing_{split}_split",
                        f"configured corpus has no independent {split} run",
                    )
                )
    return statistics


def _configured_run_count(config: Any) -> int | None:
    if not isinstance(config, dict):
        return None
    runs = config.get("runs")
    if not isinstance(runs, dict):
        return None
    dimensions = [len(values) for values in runs.values() if isinstance(values, list)]
    return math.prod(dimensions) if dimensions else None


def _check_teacher_manifests(
    manifest: dict[str, Any],
    *,
    root: Path,
    verify_hashes: bool,
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    sources = manifest.get("source_manifests")
    if sources is None:
        return {
            "teacher_run_count": 0,
            "minimum_teacher_recording_duration_s": None,
        }
    if not isinstance(sources, dict) or not sources:
        failures.append(
            _issue(
                "invalid_source_manifest_provenance",
                "source_manifests must be a nonempty mapping",
            )
        )
        return {
            "teacher_run_count": 0,
            "minimum_teacher_recording_duration_s": None,
        }

    dataset_config = manifest.get("dataset_config", {})
    dataset = (
        dataset_config.get("dataset", {})
        if isinstance(dataset_config, dict)
        else {}
    )
    frequency_hz = _optional_positive_float(dataset.get("frequency_hz"))
    expected_warmup = _optional_nonnegative_float(dataset.get("warmup_s"))
    expected_duration = _optional_positive_float(dataset.get("duration_s"))
    recording_durations: list[float] = []
    for source_name, expected_hash in sorted(sources.items()):
        source = Path(source_name)
        if not source.is_absolute():
            source = root / source
        if not source.is_file():
            failures.append(
                _issue("missing_source_manifest", f"missing teacher manifest: {source}")
            )
            continue
        if verify_hashes and file_sha256(source) != str(expected_hash):
            failures.append(
                _issue(
                    "source_manifest_hash_mismatch",
                    f"teacher manifest hash mismatch: {source}",
                )
            )
        try:
            teacher = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(
                _issue("invalid_source_manifest", f"{source}: {exc}")
            )
            continue
        if teacher.get("status") != "complete":
            failures.append(
                _issue("incomplete_teacher_run", f"teacher run is incomplete: {source}")
            )
        teacher_config = teacher.get("config")
        if not isinstance(teacher_config, dict):
            failures.append(
                _issue("invalid_source_manifest", f"teacher config is missing: {source}")
            )
            continue
        try:
            begin = float(teacher_config["begin_s"])
            end = float(teacher_config["end_s"])
            recording_value = teacher_config.get("recording_begin_s")
            recording_begin = (
                begin if recording_value is None else float(recording_value)
            )
            step_length = float(teacher["step_length_s"])
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(
                _issue("invalid_source_manifest", f"{source}: {exc}")
            )
            continue
        values = (begin, end, recording_begin, step_length)
        if (
            not all(math.isfinite(value) for value in values)
            or begin < 0.0
            or not begin <= recording_begin < end
            or step_length <= 0.0
        ):
            failures.append(
                _issue("invalid_source_manifest", f"invalid teacher timing: {source}")
            )
            continue
        if frequency_hz is not None and not math.isclose(
            step_length,
            1.0 / frequency_hz,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            failures.append(
                _issue(
                    "teacher_step_length_mismatch",
                    f"teacher step length {step_length:g}s differs from {frequency_hz:g} Hz: {source}",
                )
            )
        expected_steps = round((end - begin) / step_length)
        actual_steps = teacher.get("actual_step_count")
        if actual_steps is not None and int(actual_steps) != expected_steps:
            failures.append(
                _issue(
                    "teacher_step_count_mismatch",
                    f"teacher has {actual_steps} steps; expected {expected_steps}: {source}",
                )
            )
        warmup = recording_begin - begin
        recording_duration = end - recording_begin
        recording_durations.append(recording_duration)
        if expected_warmup is not None and warmup + 1e-9 < expected_warmup:
            failures.append(
                _issue(
                    "insufficient_teacher_warmup",
                    f"teacher warmup is {warmup:g}s; configured {expected_warmup:g}s: {source}",
                )
            )
        if (
            expected_duration is not None
            and recording_duration + 1e-9 < expected_duration
        ):
            failures.append(
                _issue(
                    "insufficient_teacher_recording_duration",
                    f"teacher records {recording_duration:g}s; configured {expected_duration:g}s: {source}",
                )
            )
    return {
        "teacher_run_count": len(sources),
        "minimum_teacher_recording_duration_s": (
            min(recording_durations) if recording_durations else None
        ),
    }


def _optional_positive_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) and result > 0.0 else None


def _optional_nonnegative_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) and result >= 0.0 else None


def _check_sample(
    sample: dict[str, Any],
    index: int,
    failures: list[dict[str, Any]],
    limits: dict[str, float],
) -> None:
    for name in ("agents", "lights", "map_points"):
        if not np.isfinite(sample[name]).all():
            failures.append(_issue(f"nonfinite_{name}", f"sample {index} contains NaN/Inf"))

    agents = sample["agents"]
    lights = sample["lights"]
    _check_encoded_rows(agents, 7, 11, "agents", index, failures)
    _check_encoded_rows(lights, 3, 12, "lights", index, failures)
    _check_ids(agents[..., -1] > 0.0, sample["agent_ids"], "agent", index, failures)
    _check_ids(lights[..., -1] > 0.0, sample["light_ids"], "light", index, failures)
    _check_tls_sequences(lights, sample["light_ids"], index, failures)

    padding = sample["map_padding_mask"]
    if padding.shape != sample["map_points"].shape[:1]:
        failures.append(_issue("map_padding_shape", f"sample {index} map mask shape differs"))
    elif not np.allclose(sample["map_points"][padding], 0.0):
        failures.append(
            _issue("nonzero_padded_map", f"sample {index} has nonzero padded map rows")
        )
    _check_map_topology(sample, index, failures)

    valid = agents[..., -1] > 0.0
    xy = agents[..., :2] * POSITION_SCALE
    consecutive = valid[:, :-1] & valid[:, 1:]
    implied = np.linalg.norm(np.diff(xy, axis=1), axis=-1) * 10.0
    if consecutive.any() and float(implied[consecutive].max()) > limits["max_implied_speed_mps"]:
        failures.append(
            _issue(
                "implausible_agent_motion",
                f"sample {index} exceeds {limits['max_implied_speed_mps']:.1f} m/s",
            )
        )


def _check_encoded_rows(
    values: np.ndarray,
    category_start: int,
    validity_index: int,
    name: str,
    sample_index: int,
    failures: list[dict[str, Any]],
) -> None:
    validity = values[..., validity_index]
    if not np.isin(validity, (-1.0, 1.0)).all():
        failures.append(_issue(f"invalid_{name}_validity", f"sample {sample_index}"))
    invalid = validity < 0.0
    if invalid.any() and not np.allclose(values[..., :validity_index][invalid], 0.0):
        failures.append(_issue(f"nonzero_invalid_{name}", f"sample {sample_index}"))
    valid = validity > 0.0
    categories = values[..., category_start:validity_index]
    if valid.any():
        rows = categories[valid]
        encoded = np.logical_or(np.isclose(rows, -1.0), np.isclose(rows, 1.0)).all()
        one_positive = np.isclose(rows, 1.0).sum(axis=-1) == 1
        if not encoded or not one_positive.all():
            failures.append(_issue(f"invalid_{name}_category", f"sample {sample_index}"))


def _check_ids(
    validity: np.ndarray,
    ids: tuple[str, ...],
    kind: str,
    sample_index: int,
    failures: list[dict[str, Any]],
) -> None:
    if len(ids) != validity.shape[0]:
        failures.append(_issue(f"{kind}_id_shape", f"sample {sample_index}"))
        return
    active = validity.any(axis=1)
    for slot, (identifier, is_active) in enumerate(zip(ids, active)):
        if bool(identifier) != bool(is_active):
            failures.append(
                _issue(
                    f"unstable_{kind}_ids",
                    f"sample {sample_index} slot {slot} ID does not match validity",
                )
            )
    nonempty = [identifier for identifier in ids if identifier]
    if len(nonempty) != len(set(nonempty)):
        failures.append(_issue(f"duplicate_{kind}_ids", f"sample {sample_index}"))


def _check_map_topology(
    sample: dict[str, Any],
    sample_index: int,
    failures: list[dict[str, Any]],
) -> None:
    names = (
        "map_point_lane_index",
        "map_lane_padding_mask",
        "map_successor_index",
        "map_successor_padding_mask",
        "map_lane_ids",
    )
    present = [name in sample for name in names]
    if not any(present):
        return
    problems: list[str] = []
    if not all(present):
        problems.append("topology tensors are incomplete")
    else:
        point_padding = sample["map_padding_mask"]
        point_lanes = sample["map_point_lane_index"]
        lane_padding = sample["map_lane_padding_mask"]
        successor = sample["map_successor_index"]
        successor_padding = sample["map_successor_padding_mask"]
        lane_ids = sample["map_lane_ids"]
        lane_count = len(lane_padding)
        if point_lanes.shape != point_padding.shape:
            problems.append("point lane indices do not match point padding")
        else:
            valid_point_lanes = point_lanes[~point_padding]
            if valid_point_lanes.size and (
                (valid_point_lanes < 0).any()
                or (valid_point_lanes >= lane_count).any()
            ):
                problems.append("a valid point references an invalid lane")
            elif valid_point_lanes.size and lane_padding[valid_point_lanes].any():
                problems.append("a valid point references a padded lane")
            if not np.all(point_lanes[point_padding] == -1):
                problems.append("a padded point has a lane index")
        if len(lane_ids) != lane_count:
            problems.append("lane IDs do not match lane padding")
        elif any(
            bool(identifier) == bool(is_padding)
            for identifier, is_padding in zip(lane_ids, lane_padding)
        ):
            problems.append("lane IDs do not match active lane slots")
        if successor.ndim != 2 or successor.shape[0] != 2:
            problems.append("successor index must have shape [2, edges]")
        elif successor_padding.shape != successor.shape[1:]:
            problems.append("successor padding shape differs")
        else:
            active_edges = successor[:, ~successor_padding]
            if active_edges.size and (
                (active_edges < 0).any() or (active_edges >= lane_count).any()
            ):
                problems.append("a successor references an invalid lane")
            elif active_edges.size and lane_padding[active_edges].any():
                problems.append("a successor references a padded lane")
            if not np.all(successor[:, successor_padding] == -1):
                problems.append("a padded successor slot is nonempty")
            edge_pairs = [tuple(edge) for edge in active_edges.T.tolist()]
            if len(edge_pairs) != len(set(edge_pairs)):
                problems.append("duplicate successor edges")
    if problems:
        failures.append(
            _issue(
                "invalid_map_topology",
                f"sample {sample_index}: " + "; ".join(problems),
            )
        )
def _check_tls_sequences(
    lights: np.ndarray,
    light_ids: tuple[str, ...],
    sample_index: int,
    failures: list[dict[str, Any]],
) -> None:
    legal = {
        0: set(range(9)),
        1: {0, 1, 2, 3},
        2: {0, 2, 3},
        3: {0, 1, 3},
        4: {0, 4, 5},
        5: {0, 5, 6},
        6: {0, 4, 6},
        7: set(range(9)),
        8: set(range(9)),
    }
    for slot, identifier in enumerate(light_ids):
        if not identifier:
            continue
        valid = lights[slot, :, -1] > 0.0
        valid_indices = np.flatnonzero(valid)
        if not len(valid_indices):
            continue
        states = np.argmax(lights[slot, :, 3:12], axis=-1)
        for frame, (source, target) in enumerate(zip(states[:-1], states[1:])):
            if not (valid[frame] and valid[frame + 1]):
                continue
            if int(target) not in legal[int(source)]:
                failures.append(
                    _issue(
                        "illegal_tls_transition",
                        f"sample {sample_index} light {identifier}: {source}->{target}",
                    )
                )
                break


def _check_tls_source(
    tls_file: Path,
    frequency_hz: int,
    failures: list[dict[str, Any]],
) -> None:
    if not tls_file.is_file():
        failures.append(_issue("missing_tls_file", f"missing TLS source: {tls_file}"))
        return
    steps_by_tls: dict[str, list[int]] = {}
    seen: set[tuple[str, int]] = set()
    with tls_file.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                tls_id = str(row["tls_id"])
                time_s = float(row["time_s"])
                state = str(row["state"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                failures.append(
                    _issue("invalid_tls_source", f"{tls_file}:{line_number}: {exc}")
                )
                return
            step = int(round(time_s * frequency_hz))
            if abs(step / frequency_hz - time_s) > 1e-6 or not state:
                failures.append(
                    _issue("invalid_tls_source", f"{tls_file}:{line_number} is misaligned")
                )
                return
            identity = (tls_id, step)
            if identity in seen:
                failures.append(
                    _issue("duplicate_tls_step", f"{tls_file}: duplicate {identity}")
                )
                return
            seen.add(identity)
            steps_by_tls.setdefault(tls_id, []).append(step)
    for tls_id, steps in steps_by_tls.items():
        ordered = sorted(steps)
        if any(second != first + 1 for first, second in zip(ordered[:-1], ordered[1:])):
            failures.append(
                _issue(
                    "incomplete_tls_steps",
                    f"{tls_file}: TLS {tls_id} has missing 0.1-second records",
                )
            )


def _check_network_coordinates(
    net_file: Path, failures: list[dict[str, Any]]
) -> None:
    if not net_file.is_file():
        failures.append(_issue("missing_network_file", f"missing network: {net_file}"))
        return
    try:
        location = ET.parse(net_file).getroot().find("location")
    except ET.ParseError as exc:
        failures.append(_issue("invalid_network_xml", f"{net_file}: {exc}"))
        return
    if location is None:
        failures.append(_issue("missing_network_location", str(net_file)))
        return
    conv = _boundary(location.attrib.get("convBoundary"))
    orig = _boundary(location.attrib.get("origBoundary"))
    if conv is None or orig is None:
        failures.append(_issue("invalid_network_boundary", str(net_file)))
        return
    conv_span = max(conv[2] - conv[0], conv[3] - conv[1])
    looks_geographic = all(-180.0 <= orig[i] <= 180.0 for i in (0, 2)) and all(
        -90.0 <= orig[i] <= 90.0 for i in (1, 3)
    )
    projection_disabled = location.attrib.get("projParameter", "!") == "!"
    if projection_disabled and looks_geographic and conv_span < 10.0:
        failures.append(
            _issue(
                "nonmetric_network_coordinates",
                f"{net_file} uses geographic degrees as SUMO metric coordinates",
            )
        )


def _boundary(value: str | None) -> tuple[float, float, float, float] | None:
    try:
        result = tuple(float(item) for item in (value or "").split(","))
    except ValueError:
        return None
    return result if len(result) == 4 else None


def _sample_hash(sample: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in ("agents", "lights", "map_points", "map_padding_mask"):
        digest.update(np.ascontiguousarray(sample[name]).tobytes())
    for name in (
        "map_point_lane_index",
        "map_lane_padding_mask",
        "map_successor_index",
        "map_successor_padding_mask",
    ):
        if name in sample:
            digest.update(np.ascontiguousarray(sample[name]).tobytes())
    return digest.hexdigest()


def _percentile(values: list[int], q: int) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def _fraction(truncated: int, selected: int) -> float:
    return float(truncated / max(truncated + selected, 1))


def _issue(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def _report(
    failures: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    statistics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": "blocked" if failures else "passed",
        "failures": failures,
        "warnings": warnings,
        "statistics": statistics,
    }
