from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np

from scenediffuserpp.normalization import POSITION_SCALE
from scenediffuserpp.scene_builder import SceneWindow


SCHEMA_VERSION = "scenediffuserpp-sumo-v2"
SUPPORTED_SCHEMA_VERSIONS = frozenset(
    {"scenediffuserpp-sumo-v1", SCHEMA_VERSION}
)
MAP_CHANNELS = (
    "x",
    "y",
    "tangent_x",
    "tangent_y",
    "speed_limit",
    "signalized",
    "polyline_fraction",
    "road_class",
)


class SceneDataset:
    def __init__(self, shard_paths: Iterable[str | Path]):
        self.shard_paths = tuple(Path(path) for path in shard_paths)
        self._index: list[tuple[Path, str]] = []
        for path in self.shard_paths:
            if not path.is_file():
                raise FileNotFoundError(f"scene shard does not exist: {path}")
            with h5py.File(path, "r") as handle:
                if handle.attrs.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
                    raise ValueError(f"unsupported scene shard schema: {path}")
                self._index.extend(
                    (path, name) for name in sorted(handle["samples"].keys())
                )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> dict:
        path, group_name = self._index[index]
        with h5py.File(path, "r") as handle:
            group = handle["samples"][group_name]
            result = {
                "agents": group["agents"][...],
                "lights": group["lights"][...],
                "agent_ids": _decode_strings(group["agent_ids"][...]),
                "light_ids": _decode_strings(group["light_ids"][...]),
                "reference_world_pose": group["reference_world_pose"][...],
                "map_points": group["map_points"][...],
                "map_padding_mask": group["map_padding_mask"][...].astype(bool),
                "meta": json.loads(group.attrs["meta_json"]),
                "truncated_agents": int(group.attrs["truncated_agents"]),
                "truncated_lights": int(group.attrs["truncated_lights"]),
                "truncated_map_points": int(group.attrs["truncated_map_points"]),
            }
            if "map_point_lane_index" in group:
                result.update(
                    {
                        "map_point_lane_index": group[
                            "map_point_lane_index"
                        ][...].astype(np.int64),
                        "map_lane_padding_mask": group[
                            "map_lane_padding_mask"
                        ][...].astype(bool),
                        "map_successor_index": group[
                            "map_successor_index"
                        ][...].astype(np.int64),
                        "map_successor_padding_mask": group[
                            "map_successor_padding_mask"
                        ][...].astype(bool),
                        "map_lane_ids": _decode_strings(group["map_lane_ids"][...]),
                        "truncated_map_lanes": int(
                            group.attrs.get("truncated_map_lanes", 0)
                        ),
                        "truncated_map_connections": int(
                            group.attrs.get("truncated_map_connections", 0)
                        ),
                    }
                )
            return result


@dataclass(frozen=True)
class MapTensorBundle:
    points: np.ndarray
    point_padding_mask: np.ndarray
    point_lane_index: np.ndarray
    lane_padding_mask: np.ndarray
    successor_index: np.ndarray
    successor_padding_mask: np.ndarray
    lane_ids: tuple[str, ...]
    truncated_points: int
    truncated_lanes: int
    truncated_connections: int


def write_shard(
    path: str | Path,
    windows: Sequence[SceneWindow],
    max_map_points: int = 2048,
    map_radius_m: float = 80.0,
    max_map_lanes: int = 1024,
    max_map_connections: int = 4096,
) -> Path:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"scene shard already exists: {destination}")
    if not windows:
        raise ValueError("cannot write an empty scene shard")
    if max_map_points <= 0:
        raise ValueError("max_map_points must be positive")
    if max_map_lanes <= 0 or max_map_connections <= 0:
        raise ValueError("map topology capacities must be positive")
    if map_radius_m <= 0.0:
        raise ValueError("map_radius_m must be positive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    string_dtype = h5py.string_dtype(encoding="utf-8")
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs["schema_version"] = SCHEMA_VERSION
            handle.attrs["map_channels_json"] = json.dumps(MAP_CHANNELS)
            handle.attrs["maximum_map_points"] = int(max_map_points)
            handle.attrs["maximum_map_lanes"] = int(max_map_lanes)
            handle.attrs["maximum_map_connections"] = int(max_map_connections)
            handle.attrs["map_radius_m"] = float(map_radius_m)
            handle.attrs["position_scale_m"] = float(POSITION_SCALE)
            samples = handle.create_group("samples")
            for index, window in enumerate(windows):
                group = samples.create_group(f"{index:06d}")
                group.create_dataset("agents", data=window.agents, compression="gzip")
                group.create_dataset("lights", data=window.lights, compression="gzip")
                group.create_dataset(
                    "agent_ids",
                    data=np.asarray(window.agent_ids, dtype=object),
                    dtype=string_dtype,
                )
                group.create_dataset(
                    "light_ids",
                    data=np.asarray(window.light_ids, dtype=object),
                    dtype=string_dtype,
                )
                group.create_dataset(
                    "reference_world_pose",
                    data=np.array(
                        [
                            window.reference_world_pose.xy[0],
                            window.reference_world_pose.xy[1],
                            window.reference_world_pose.z,
                            window.reference_world_pose.heading,
                        ],
                        dtype=np.float64,
                    ),
                )
                map_bundle = _map_tensors(
                    window,
                    maximum_points=max_map_points,
                    maximum_lanes=max_map_lanes,
                    maximum_connections=max_map_connections,
                    radius=map_radius_m,
                )
                group.create_dataset(
                    "map_points", data=map_bundle.points, compression="gzip"
                )
                group.create_dataset(
                    "map_padding_mask", data=map_bundle.point_padding_mask
                )
                group.create_dataset(
                    "map_point_lane_index", data=map_bundle.point_lane_index
                )
                group.create_dataset(
                    "map_lane_padding_mask", data=map_bundle.lane_padding_mask
                )
                group.create_dataset(
                    "map_successor_index", data=map_bundle.successor_index
                )
                group.create_dataset(
                    "map_successor_padding_mask",
                    data=map_bundle.successor_padding_mask,
                )
                group.create_dataset(
                    "map_lane_ids",
                    data=np.asarray(map_bundle.lane_ids, dtype=object),
                    dtype=string_dtype,
                )
                group.attrs["meta_json"] = json.dumps(
                    window.meta, sort_keys=True, separators=(",", ":"), allow_nan=False
                )
                group.attrs["truncated_agents"] = int(window.truncated_agents)
                group.attrs["truncated_lights"] = int(window.truncated_lights)
                group.attrs["truncated_map_points"] = int(
                    map_bundle.truncated_points
                )
                group.attrs["truncated_map_lanes"] = int(
                    map_bundle.truncated_lanes
                )
                group.attrs["truncated_map_connections"] = int(
                    map_bundle.truncated_connections
                )
            handle.flush()
        with h5py.File(temporary, "r") as check:
            if len(check["samples"]) != len(windows):
                raise RuntimeError("HDF5 shard reopen validation found missing samples")
        temporary.replace(destination)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return destination


def assign_split(run_id: str, seed: int) -> str:
    if not run_id:
        raise ValueError("run_id cannot be empty")
    digest = hashlib.sha256(f"{int(seed)}:{run_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(values: object) -> str:
    encoded = json.dumps(
        values,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _map_tensors(
    window: SceneWindow,
    *,
    maximum_points: int,
    maximum_lanes: int,
    maximum_connections: int,
    radius: float,
) -> MapTensorBundle:
    rows: list[tuple[float, str, int, np.ndarray]] = []
    if window.roadgraph is not None:
        heading = window.reference_world_pose.heading
        cosine = np.cos(heading)
        sine = np.sin(heading)
        rotation = np.array([[cosine, sine], [-sine, cosine]], dtype=np.float64)
        for lane in window.roadgraph.lane_tokens:
            local_xy = (rotation @ (lane.xy - window.reference_world_pose.xy).T).T
            local_tangent = (rotation @ lane.tangent.T).T
            count = len(lane.xy)
            for point_index in range(count):
                distance = float(np.linalg.norm(local_xy[point_index]))
                if distance > radius:
                    continue
                fraction = point_index / max(count - 1, 1)
                feature = np.array(
                    [
                        local_xy[point_index, 0] / POSITION_SCALE,
                        local_xy[point_index, 1] / POSITION_SCALE,
                        local_tangent[point_index, 0],
                        local_tangent[point_index, 1],
                        lane.speed_limit_mps / 40.0,
                        float(lane.signalized),
                        fraction,
                        _road_class(lane.lane_type),
                    ],
                    dtype=np.float32,
                )
                rows.append((distance, lane.lane_id, point_index, feature))
    rows.sort(key=lambda row: (row[0], row[1], row[2]))
    point_limited = rows[:maximum_points]
    lane_minimum_distance: dict[str, float] = {}
    for distance, lane_id, _point_index, _feature in point_limited:
        lane_minimum_distance.setdefault(lane_id, distance)
    lane_rank = sorted(
        lane_minimum_distance.items(),
        key=lambda item: (item[1], item[0]),
    )
    selected_lane_ids = tuple(
        lane_id for lane_id, _distance in lane_rank[:maximum_lanes]
    )
    lane_to_index = {
        lane_id: index for index, lane_id in enumerate(selected_lane_ids)
    }
    kept = [row for row in point_limited if row[1] in lane_to_index]
    values = np.zeros((maximum_points, len(MAP_CHANNELS)), dtype=np.float32)
    point_padding = np.ones(maximum_points, dtype=bool)
    point_lane_index = np.full(maximum_points, -1, dtype=np.int64)
    for index, (_, lane_id, _point_index, feature) in enumerate(kept):
        values[index] = feature
        point_padding[index] = False
        point_lane_index[index] = lane_to_index[lane_id]

    lane_padding = np.ones(maximum_lanes, dtype=bool)
    lane_padding[: len(selected_lane_ids)] = False
    padded_lane_ids = selected_lane_ids + ("",) * (
        maximum_lanes - len(selected_lane_ids)
    )
    visible_lane_ids = {row[1] for row in rows}
    visible_connections = sorted(
        (source, target)
        for source in visible_lane_ids
        for target in (
            window.roadgraph.successors.get(source, ())
            if window.roadgraph is not None
            else ()
        )
        if target in visible_lane_ids
    )
    selected_connections = [
        (lane_to_index[source], lane_to_index[target])
        for source, target in visible_connections
        if source in lane_to_index and target in lane_to_index
    ][:maximum_connections]
    successor_index = np.full((2, maximum_connections), -1, dtype=np.int64)
    successor_padding = np.ones(maximum_connections, dtype=bool)
    for index, (source, target) in enumerate(selected_connections):
        successor_index[:, index] = (source, target)
        successor_padding[index] = False
    return MapTensorBundle(
        points=values,
        point_padding_mask=point_padding,
        point_lane_index=point_lane_index,
        lane_padding_mask=lane_padding,
        successor_index=successor_index,
        successor_padding_mask=successor_padding,
        lane_ids=padded_lane_ids,
        truncated_points=max(len(rows) - len(kept), 0),
        truncated_lanes=max(len(visible_lane_ids) - len(selected_lane_ids), 0),
        truncated_connections=max(
            len(visible_connections) - len(selected_connections), 0
        ),
    )


def _road_class(value: str) -> float:
    road_type = value.lower()
    ordered = (
        (("motorway", "trunk"), 1.0),
        (("primary", "arterial"), 0.8),
        (("secondary",), 0.6),
        (("tertiary", "collector"), 0.4),
        (("residential", "living_street", "service"), 0.2),
    )
    for names, encoded in ordered:
        if any(name in road_type for name in names):
            return encoded
    return 0.0


def _decode_strings(values: np.ndarray) -> tuple[str, ...]:
    return tuple(
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values.tolist()
    )
