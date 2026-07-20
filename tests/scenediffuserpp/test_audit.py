import json
from pathlib import Path

import h5py
import numpy as np

from topoworld.scenediffuserpp.audit import audit_dataset
from topoworld.scenediffuserpp.audit import has_only_incomplete_corpus_failures
from topoworld.scenediffuserpp.diagnostics import plot_scene
from topoworld.scenediffuserpp.roadgraph import LaneToken
from topoworld.scenediffuserpp.roadgraph import Roadgraph
from topoworld.scenediffuserpp.scene_builder import AgentState
from topoworld.scenediffuserpp.scene_builder import build_window
from topoworld.scenediffuserpp.schema import SceneSpec
from topoworld.scenediffuserpp.storage import SceneDataset
from topoworld.scenediffuserpp.storage import canonical_json_sha256
from topoworld.scenediffuserpp.storage import file_sha256
from topoworld.scenediffuserpp.storage import write_shard


def _write_net(path: Path, metric: bool = True) -> Path:
    if metric:
        location = (
            '<location netOffset="0.00,0.00" convBoundary="0.00,0.00,1000.00,800.00" '
            'origBoundary="118.70,32.00,118.71,32.01" projParameter="+proj=utm"/>'
        )
    else:
        location = (
            '<location netOffset="-118.72,-32.01" convBoundary="0.00,0.00,0.11,0.07" '
            'origBoundary="118.72,32.01,118.83,32.08" projParameter="!"/>'
        )
    path.write_text(f"<net>{location}</net>", encoding="utf-8")
    return path


def _scene(net_file: Path):
    tracks = {
        "ego": {
            step: AgentState(step, 0.0, 0.0, 0.0, 10.0, 4.5, 2.0, 1.75, "car")
            for step in range(91)
        },
        "entering": {
            step: AgentState(step + 8.0, 0.0, 0.0, 0.0, 10.0, 4.5, 2.0, 1.75, "car")
            for step in range(30, 91)
        },
    }
    lane_xy = np.array([[0.0, 0.0], [100.0, 0.0]], dtype=np.float32)
    roadgraph = Roadgraph(
        lane_tokens=(
            LaneToken(
                lane_id="e0_0",
                edge_id="e0",
                xy=lane_xy,
                tangent=np.tile([1.0, 0.0], (2, 1)).astype(np.float32),
                speed_limit_mps=13.9,
                lane_type="primary",
                signalized=False,
            ),
        ),
        successors={"e0_0": ()},
        light_tokens=(),
    )
    return build_window(
        tracks,
        av_id="ego",
        start_step=0,
        spec=SceneSpec.small(),
        roadgraph=roadgraph,
        meta={
            "sample_id": "sample_0",
            "run_id": "run_0",
            "scenario_id": "scene_0",
            "net_file": str(net_file.resolve()),
        },
    )


def _dataset(tmp_path: Path, metric: bool = True) -> Path:
    root = tmp_path / "dataset"
    root.mkdir()
    net_file = _write_net(tmp_path / "network.net.xml", metric=metric)
    shard = write_shard(root / "shard_00000.h5", [_scene(net_file)], max_map_points=16)
    manifest = {
        "schema_version": "scenediffuserpp-sumo-v1",
        "samples": [
            {
                "sample_id": "sample_0",
                "run_id": "run_0",
                "split": "train",
                "shard": shard.name,
                "sample_index": 0,
            }
        ],
        "shards": {shard.name: file_sha256(shard)},
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_audit_passes_valid_metric_dataset_and_reports_population_stats(tmp_path: Path):
    dataset = _dataset(tmp_path)

    report = audit_dataset(dataset)

    assert report["status"] == "passed"
    assert report["failures"] == []
    assert report["statistics"]["entering_agent_fraction"] > 0.0
    assert report["statistics"]["agent_truncation_fraction"] == 0.0
    assert report["statistics"]["map_lane_truncation_fraction"] == 0.0
    assert report["statistics"]["map_connection_truncation_fraction"] == 0.0
    assert report["statistics"]["active_agents_p50"] >= 1.0
    assert report["statistics"]["agent_birth_transition_count"] > 0
    assert report["statistics"]["samples_with_agent_birth_fraction"] == 1.0
    assert report["statistics"]["agent_removal_transition_count"] == 0
    assert report["statistics"]["samples_with_agent_removal_fraction"] == 0.0
    assert report["statistics"]["agent_multi_segment_count"] == 0


def test_audit_reports_agent_removal_transition_coverage(tmp_path: Path):
    dataset = _dataset(tmp_path)
    with h5py.File(dataset / "shard_00000.h5", "r+") as handle:
        rows = handle["samples/000000/agents"][1]
        rows[:50] = rows[30]
        rows[50:, :-1] = 0.0
        rows[50:, -1] = -1.0
        handle["samples/000000/agents"][1] = rows

    report = audit_dataset(dataset, verify_hashes=False)

    assert report["status"] == "passed"
    assert report["statistics"]["agent_removal_transition_count"] == 1
    assert report["statistics"]["samples_with_agent_removal_fraction"] == 1.0
    assert report["statistics"]["exiting_agent_fraction"] > 0.0


def test_audit_can_require_agent_removal_supervision(tmp_path: Path):
    dataset = _dataset(tmp_path)

    report = audit_dataset(
        dataset,
        thresholds={"min_agent_removal_transition_count": 1},
    )

    assert "insufficient_agent_removal_transitions" in {
        item["code"] for item in report["failures"]
    }


def test_audit_blocks_nonfinite_agent_value(tmp_path: Path):
    dataset = _dataset(tmp_path)
    with h5py.File(dataset / "shard_00000.h5", "r+") as handle:
        handle["samples/000000/agents"][0, 0, 0] = np.nan

    report = audit_dataset(dataset, verify_hashes=False)

    assert report["status"] == "blocked"
    assert has_only_incomplete_corpus_failures(report) is False
    assert "nonfinite_agents" in {item["code"] for item in report["failures"]}


def test_audit_blocks_invalid_point_lane_membership(tmp_path: Path):
    dataset = _dataset(tmp_path)
    with h5py.File(dataset / "shard_00000.h5", "r+") as handle:
        group = handle["samples/000000"]
        valid_point = int(np.flatnonzero(~group["map_padding_mask"][...])[0])
        group["map_point_lane_index"][valid_point] = 9999

    report = audit_dataset(dataset, verify_hashes=False)

    assert "invalid_map_topology" in {
        item["code"] for item in report["failures"]
    }


def test_audit_blocks_run_crossing_splits(tmp_path: Path):
    dataset = _dataset(tmp_path)
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["samples"].append(
        {
            **manifest["samples"][0],
            "sample_id": "sample_1",
            "split": "test",
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = audit_dataset(dataset)

    assert "run_split_leakage" in {item["code"] for item in report["failures"]}


def test_audit_blocks_tampered_dataset_config_or_split_counts(tmp_path: Path):
    dataset = _dataset(tmp_path)
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset_config"] = {"dataset": {"frequency_hz": 10}}
    manifest["dataset_config_sha256"] = canonical_json_sha256(
        manifest["dataset_config"]
    )
    manifest["split_counts"] = {"train": 0, "validation": 1, "test": 0}
    manifest["dataset_config"]["dataset"]["frequency_hz"] = 5
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = audit_dataset(dataset)

    failures = {item["code"] for item in report["failures"]}
    assert "dataset_config_hash_mismatch" in failures
    assert "split_count_mismatch" in failures


def test_audit_blocks_incomplete_configured_run_grid_and_missing_heldout_splits(
    tmp_path: Path,
):
    dataset = _dataset(tmp_path)
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset_config"] = {
        "dataset": {"frequency_hz": 10},
        "runs": {
            "departure_period_s": [0.5, 1.0, 2.0],
            "seeds": [0, 1],
        },
    }
    manifest["dataset_config_sha256"] = canonical_json_sha256(
        manifest["dataset_config"]
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = audit_dataset(dataset)

    failures = {item["code"] for item in report["failures"]}
    assert report["status"] == "blocked"
    assert report["statistics"]["configured_run_count"] == 6
    assert report["statistics"]["independent_run_count"] == 1
    assert report["statistics"]["split_counts"] == {
        "train": 1,
        "validation": 0,
        "test": 0,
    }
    assert "incomplete_configured_run_grid" in failures
    assert "missing_validation_split" in failures
    assert "missing_test_split" in failures
    assert has_only_incomplete_corpus_failures(report) is True


def test_audit_blocks_teacher_shorter_than_configured_recording_contract(
    tmp_path: Path,
):
    dataset = _dataset(tmp_path)
    teacher = tmp_path / "teacher_manifest.json"
    teacher.write_text(
        json.dumps(
            {
                "status": "complete",
                "step_length_s": 0.1,
                "actual_step_count": 1800,
                "config": {
                    "begin_s": 0.0,
                    "end_s": 180.0,
                    "recording_begin_s": None,
                },
            }
        ),
        encoding="utf-8",
    )
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset_config"] = {
        "dataset": {
            "frequency_hz": 10,
            "warmup_s": 60.0,
            "duration_s": 600.0,
        }
    }
    manifest["dataset_config_sha256"] = canonical_json_sha256(
        manifest["dataset_config"]
    )
    manifest["source_manifests"] = {str(teacher): file_sha256(teacher)}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = audit_dataset(dataset)

    failures = {item["code"] for item in report["failures"]}
    assert "insufficient_teacher_warmup" in failures
    assert "insufficient_teacher_recording_duration" in failures
    assert report["statistics"]["teacher_run_count"] == 1
    assert report["statistics"]["minimum_teacher_recording_duration_s"] == 180.0
    assert has_only_incomplete_corpus_failures(report) is True


def test_audit_rejects_teacher_hash_or_frequency_mismatch_as_structural(
    tmp_path: Path,
):
    dataset = _dataset(tmp_path)
    teacher = tmp_path / "teacher_manifest.json"
    teacher.write_text(
        json.dumps(
            {
                "status": "complete",
                "step_length_s": 0.2,
                "config": {
                    "begin_s": 0.0,
                    "end_s": 660.0,
                    "recording_begin_s": 60.0,
                },
            }
        ),
        encoding="utf-8",
    )
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset_config"] = {
        "dataset": {
            "frequency_hz": 10,
            "warmup_s": 60.0,
            "duration_s": 600.0,
        }
    }
    manifest["dataset_config_sha256"] = canonical_json_sha256(
        manifest["dataset_config"]
    )
    manifest["source_manifests"] = {str(teacher): "wrong-hash"}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = audit_dataset(dataset)

    failures = {item["code"] for item in report["failures"]}
    assert "source_manifest_hash_mismatch" in failures
    assert "teacher_step_length_mismatch" in failures
    assert has_only_incomplete_corpus_failures(report) is False


def test_audit_blocks_longitude_latitude_used_as_metric_coordinates(tmp_path: Path):
    dataset = _dataset(tmp_path, metric=False)

    report = audit_dataset(dataset)

    assert report["status"] == "blocked"
    assert "nonmetric_network_coordinates" in {
        item["code"] for item in report["failures"]
    }


def test_audit_allows_internal_light_visibility_gap(tmp_path: Path):
    dataset = _dataset(tmp_path)
    with h5py.File(dataset / "shard_00000.h5", "r+") as handle:
        group = handle["samples/000000"]
        group["light_ids"][0] = "tls0:0"
        rows = group["lights"][0]
        rows[:, 3:12] = -1.0
        rows[:, 7] = 1.0
        rows[:, -1] = 1.0
        rows[45, :-1] = 0.0
        rows[45, -1] = -1.0
        group["lights"][0] = rows

    report = audit_dataset(dataset, verify_hashes=False)

    assert "incomplete_tls_steps" not in {item["code"] for item in report["failures"]}


def test_audit_reports_light_coverage_and_can_require_state_transitions(
    tmp_path: Path,
):
    dataset = _dataset(tmp_path)
    with h5py.File(dataset / "shard_00000.h5", "r+") as handle:
        group = handle["samples/000000"]
        group["light_ids"][0] = "tls0:0"
        rows = group["lights"][0]
        rows[:, 3:12] = -1.0
        rows[:, 3] = 1.0
        rows[:, -1] = 1.0
        rows[50:, 3] = -1.0
        rows[50:, 4] = 1.0
        group["lights"][0] = rows

    report = audit_dataset(
        dataset,
        verify_hashes=False,
        thresholds={"min_light_state_transition_count": 1},
    )

    assert report["status"] == "passed"
    assert report["statistics"]["samples_with_valid_light_fraction"] == 1.0
    assert report["statistics"]["samples_with_light_transition_fraction"] == 1.0
    assert report["statistics"]["light_state_transition_count"] == 1


def test_audit_blocks_when_required_light_transition_supervision_is_absent(
    tmp_path: Path,
):
    dataset = _dataset(tmp_path)

    report = audit_dataset(
        dataset,
        thresholds={"min_light_state_transition_count": 1},
    )

    assert "insufficient_light_state_transitions" in {
        item["code"] for item in report["failures"]
    }


def test_audit_blocks_internal_gap_in_source_tls_records(tmp_path: Path):
    dataset = _dataset(tmp_path)
    tls_file = tmp_path / "tls_states.jsonl"
    tls_file.write_text(
        '\n'.join(
            (
                '{"time_s":0.0,"tls_id":"tls0","state":"r"}',
                '{"time_s":0.2,"tls_id":"tls0","state":"G"}',
            )
        )
        + "\n",
        encoding="utf-8",
    )
    with h5py.File(dataset / "shard_00000.h5", "r+") as handle:
        group = handle["samples/000000"]
        meta = json.loads(group.attrs["meta_json"])
        meta["tls_file"] = str(tls_file)
        group.attrs["meta_json"] = json.dumps(meta)

    report = audit_dataset(dataset, verify_hashes=False)

    assert "incomplete_tls_steps" in {item["code"] for item in report["failures"]}


def test_plot_scene_writes_nonempty_png(tmp_path: Path):
    dataset = _dataset(tmp_path)
    sample = SceneDataset([dataset / "shard_00000.h5"])[0]
    destination = tmp_path / "scene.png"

    plot_scene(sample, destination, history_steps=11)

    assert destination.stat().st_size > 1000
