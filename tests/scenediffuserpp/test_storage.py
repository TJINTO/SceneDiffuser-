from pathlib import Path
from dataclasses import replace

import h5py
import numpy as np

from scenediffuserpp.roadgraph import LaneToken
from scenediffuserpp.roadgraph import Roadgraph
from scenediffuserpp.normalization import POSITION_SCALE
from scenediffuserpp.scene_builder import AgentState
from scenediffuserpp.scene_builder import build_window
from scenediffuserpp.schema import SceneSpec
from scenediffuserpp.storage import SceneDataset
from scenediffuserpp.storage import assign_split
from scenediffuserpp.storage import canonical_json_sha256
from scenediffuserpp.storage import write_shard


def _scene_window():
    tracks = {
        "ego": {
            step: AgentState(step, 0.0, 0.0, 0.0, 10.0, 4.5, 2.0, 1.75, "car")
            for step in range(91)
        },
        "car": {
            step: AgentState(step, 10.0, 0.0, 0.0, 10.0, 4.5, 2.0, 1.75, "car")
            for step in range(91)
        },
    }
    xy = np.array([[0.0, 0.0], [100.0, 0.0]], dtype=np.float32)
    roadgraph = Roadgraph(
        lane_tokens=(
            LaneToken(
                lane_id="e0_0",
                edge_id="e0",
                xy=xy,
                tangent=np.tile([1.0, 0.0], (2, 1)).astype(np.float32),
                speed_limit_mps=13.9,
                lane_type="arterial",
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
        meta={"run_id": "run_17", "scenario_id": "scene_0"},
    )


def test_hdf5_round_trip_preserves_scene_exactly(tmp_path: Path):
    scene = _scene_window()
    path = write_shard(tmp_path / "shard_00000.h5", [scene], max_map_points=16)

    loaded = SceneDataset([path])[0]

    np.testing.assert_array_equal(loaded["agents"], scene.agents)
    np.testing.assert_array_equal(loaded["lights"], scene.lights)
    assert loaded["agent_ids"] == scene.agent_ids
    assert loaded["light_ids"] == scene.light_ids
    assert loaded["meta"]["run_id"] == "run_17"
    assert loaded["map_points"].shape == (16, 8)
    assert loaded["map_padding_mask"].sum() == 15


def test_shard_records_configured_map_capacity_and_radius(tmp_path: Path):
    near_path = write_shard(
        tmp_path / "near.h5",
        [_scene_window()],
        max_map_points=5,
        map_radius_m=20.0,
    )
    wide_path = write_shard(
        tmp_path / "wide.h5",
        [_scene_window()],
        max_map_points=7,
        map_radius_m=200.0,
    )

    near = SceneDataset([near_path])[0]
    wide = SceneDataset([wide_path])[0]
    assert near["map_points"].shape == (5, 8)
    assert wide["map_points"].shape == (7, 8)
    assert (~wide["map_padding_mask"]).sum() > (~near["map_padding_mask"]).sum()
    with h5py.File(wide_path, "r") as handle:
        assert handle.attrs["maximum_map_points"] == 7
        assert handle.attrs["map_radius_m"] == 200.0
        assert handle.attrs["position_scale_m"] == POSITION_SCALE


def test_shard_preserves_point_lane_membership_and_directed_successors(
    tmp_path: Path,
):
    scene = _scene_window()
    first = scene.roadgraph.lane_tokens[0]
    second_xy = np.array([[100.0, 0.0], [200.0, 0.0]], dtype=np.float32)
    second = LaneToken(
        lane_id="e1_0",
        edge_id="e1",
        xy=second_xy,
        tangent=np.tile([1.0, 0.0], (2, 1)).astype(np.float32),
        speed_limit_mps=13.9,
        lane_type="arterial",
        signalized=False,
    )
    scene = replace(
        scene,
        roadgraph=Roadgraph(
            lane_tokens=(first, second),
            successors={"e0_0": ("e1_0",), "e1_0": ()},
            light_tokens=(),
        ),
    )
    path = write_shard(
        tmp_path / "topology.h5",
        [scene],
        max_map_points=8,
        map_radius_m=250.0,
        max_map_lanes=4,
        max_map_connections=4,
    )

    loaded = SceneDataset([path])[0]
    valid_points = ~loaded["map_padding_mask"]
    valid_connections = ~loaded["map_successor_padding_mask"]
    assert loaded["map_lane_ids"][:2] == ("e0_0", "e1_0")
    assert set(loaded["map_point_lane_index"][valid_points]) == {0, 1}
    np.testing.assert_array_equal(
        loaded["map_successor_index"][:, valid_connections],
        np.array([[0], [1]], dtype=np.int64),
    )
    assert loaded["map_lane_padding_mask"].tolist() == [False, False, True, True]


def test_dataset_does_not_keep_hdf5_file_open(tmp_path: Path):
    path = write_shard(tmp_path / "shard_00000.h5", [_scene_window()])
    dataset = SceneDataset([path])

    _ = dataset[0]

    with h5py.File(path, "r+") as handle:
        handle.attrs["reopened"] = True


def test_shard_write_is_atomic_and_leaves_no_temporary_file(tmp_path: Path):
    path = write_shard(tmp_path / "shard_00000.h5", [_scene_window()])

    assert path.exists()
    assert not path.with_suffix(".h5.tmp").exists()


def test_split_is_stable_and_grouped_by_complete_run():
    first = assign_split("run_17", seed=123)
    rows = [{"run_id": "run_17", "av_id": value} for value in ("a", "b", "c")]

    assert first == assign_split("run_17", seed=123)
    assert {assign_split(row["run_id"], seed=123) for row in rows} == {first}
    assert first in {"train", "validation", "test"}


def test_split_seed_is_effective_without_splitting_a_run():
    assignments = {assign_split("run_17", seed=seed) for seed in range(100)}

    assert len(assignments) > 1


def test_canonical_config_hash_is_order_independent_and_value_sensitive():
    first = {"dataset": {"frequency_hz": 10, "map_radius_m": 1000.0}}
    reordered = {"dataset": {"map_radius_m": 1000.0, "frequency_hz": 10}}
    changed = {"dataset": {"frequency_hz": 10, "map_radius_m": 80.0}}

    assert canonical_json_sha256(first) == canonical_json_sha256(reordered)
    assert canonical_json_sha256(first) != canonical_json_sha256(changed)
