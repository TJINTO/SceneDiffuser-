from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import sumolib

from topoworld.scenediffuserpp.schema import LightState


@dataclass(frozen=True)
class LaneToken:
    lane_id: str
    edge_id: str
    xy: np.ndarray
    tangent: np.ndarray
    speed_limit_mps: float
    lane_type: str
    signalized: bool


@dataclass(frozen=True)
class LightToken:
    tls_id: str
    link_index: int
    incoming_lane_id: str
    outgoing_lane_id: str
    stop_line_xy: np.ndarray
    turn_direction: str


@dataclass(frozen=True)
class Roadgraph:
    lane_tokens: tuple[LaneToken, ...]
    successors: dict[str, tuple[str, ...]]
    light_tokens: tuple[LightToken, ...]


def load_roadgraph(net_file: str | Path, point_spacing_m: float = 2.0) -> Roadgraph:
    path = Path(net_file)
    if not path.is_file():
        raise FileNotFoundError(f"SUMO net file does not exist: {path}")
    if point_spacing_m <= 0.0:
        raise ValueError("point_spacing_m must be positive")
    network = sumolib.net.readNet(str(path), withInternal=False)

    lane_tokens: list[LaneToken] = []
    successors: dict[str, tuple[str, ...]] = {}
    light_by_identity: dict[tuple[str, int], LightToken] = {}

    lanes = sorted(
        (
            lane
            for edge in network.getEdges()
            if not edge.getID().startswith(":")
            for lane in edge.getLanes()
        ),
        key=lambda lane: lane.getID(),
    )
    for lane in lanes:
        outgoing = sorted(
            lane.getOutgoing(),
            key=lambda connection: (
                connection.getToLane().getID(),
                connection.getTLSID(),
                connection.getTLLinkIndex(),
            ),
        )
        successor_ids = tuple(
            sorted(
                {
                    connection.getToLane().getID()
                    for connection in outgoing
                    if not connection.getToLane().getEdge().getID().startswith(":")
                }
            )
        )
        successors[lane.getID()] = successor_ids
        xy = _resample_polyline(lane.getShape(), point_spacing_m)
        signalized = any(connection.getTLSID() for connection in outgoing)
        lane_tokens.append(
            LaneToken(
                lane_id=lane.getID(),
                edge_id=lane.getEdge().getID(),
                xy=xy,
                tangent=_polyline_tangents(xy),
                speed_limit_mps=float(lane.getSpeed()),
                lane_type=str(lane.getEdge().getType() or "unknown"),
                signalized=signalized,
            )
        )
        for connection in outgoing:
            tls_id = str(connection.getTLSID())
            link_index = int(connection.getTLLinkIndex())
            if not tls_id or link_index < 0:
                continue
            identity = (tls_id, link_index)
            candidate = LightToken(
                tls_id=tls_id,
                link_index=link_index,
                incoming_lane_id=connection.getFromLane().getID(),
                outgoing_lane_id=connection.getToLane().getID(),
                stop_line_xy=np.asarray(lane.getShape()[-1][:2], dtype=np.float32),
                turn_direction=str(connection.getDirection()),
            )
            previous = light_by_identity.get(identity)
            if previous is None or (
                candidate.incoming_lane_id,
                candidate.outgoing_lane_id,
            ) < (previous.incoming_lane_id, previous.outgoing_lane_id):
                light_by_identity[identity] = candidate

    return Roadgraph(
        lane_tokens=tuple(lane_tokens),
        successors=successors,
        light_tokens=tuple(light_by_identity[key] for key in sorted(light_by_identity)),
    )


def map_sumo_signal(char: str, turn: str) -> LightState:
    if len(char) != 1:
        raise ValueError("SUMO signal state must be a single character")
    arrow = turn in {"l", "r", "L", "R", "t"}
    if char in {"G", "g"}:
        return LightState.GREEN_ARROW if arrow else LightState.GREEN
    if char in {"y", "Y"}:
        return LightState.YELLOW_ARROW if arrow else LightState.YELLOW
    if char in {"r", "R"}:
        return LightState.RED_ARROW if arrow else LightState.RED
    return LightState.UNKNOWN


def validate_tls_state(
    tls_id: str, state: str, light_tokens: Iterable[LightToken]
) -> None:
    indices = [token.link_index for token in light_tokens if token.tls_id == tls_id]
    if not indices:
        raise KeyError(f"roadgraph has no traffic-light tokens for {tls_id!r}")
    required = max(indices) + 1
    if len(state) < required:
        raise ValueError(
            f"TLS {tls_id!r} state length {len(state)} is shorter than required {required}"
        )


def _resample_polyline(
    shape: Iterable[tuple[float, float]], point_spacing_m: float
) -> np.ndarray:
    points = np.asarray([point[:2] for point in shape], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise ValueError("lane shape must contain at least one 2D point")
    if len(points) == 1:
        return points.astype(np.float32)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate(([True], segment_lengths > 1e-9))
    points = points[keep]
    if len(points) == 1:
        return points.astype(np.float32)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total = float(cumulative[-1])
    sample_distances = np.arange(0.0, total, point_spacing_m, dtype=np.float64)
    if len(sample_distances) == 0 or total - sample_distances[-1] > 1e-9:
        sample_distances = np.append(sample_distances, total)
    else:
        sample_distances[-1] = total
    result = np.column_stack(
        [np.interp(sample_distances, cumulative, points[:, axis]) for axis in range(2)]
    )
    return result.astype(np.float32)


def _polyline_tangents(xy: np.ndarray) -> np.ndarray:
    if len(xy) == 1:
        return np.zeros_like(xy, dtype=np.float32)
    differences = np.gradient(xy.astype(np.float64), axis=0)
    norms = np.linalg.norm(differences, axis=1, keepdims=True)
    tangents = np.divide(
        differences,
        norms,
        out=np.zeros_like(differences),
        where=norms > 1e-12,
    )
    return tangents.astype(np.float32)
