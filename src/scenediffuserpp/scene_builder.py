from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterable, Mapping
import xml.etree.ElementTree as ET

import numpy as np

from scenediffuserpp.normalization import AgentNormalizer
from scenediffuserpp.normalization import LightNormalizer
from scenediffuserpp.normalization import wrap_to_pi
from scenediffuserpp.roadgraph import LightToken
from scenediffuserpp.roadgraph import Roadgraph
from scenediffuserpp.roadgraph import map_sumo_signal
from scenediffuserpp.roadgraph import validate_tls_state
from scenediffuserpp.schema import AgentType
from scenediffuserpp.schema import SceneSpec


DEFAULT_LENGTH_M = 5.0
DEFAULT_WIDTH_M = 1.8
DEFAULT_HEIGHT_M = 1.5


@dataclass(frozen=True)
class AgentState:
    x: float
    y: float
    z: float
    heading: float
    speed: float
    length: float
    width: float
    height: float
    type_id: str


@dataclass(frozen=True)
class Pose2D:
    xy: np.ndarray
    z: float
    heading: float


@dataclass(frozen=True)
class SceneWindow:
    agents: np.ndarray
    lights: np.ndarray
    agent_ids: tuple[str, ...]
    light_ids: tuple[str, ...]
    reference_world_pose: Pose2D
    meta: dict = field(default_factory=dict)
    truncated_agents: int = 0
    truncated_lights: int = 0
    roadgraph: Roadgraph | None = None


def parse_fcd(
    fcd_file: str | Path,
    route_file: str | Path | None = None,
    frequency_hz: int = 10,
) -> dict[str, dict[int, AgentState]]:
    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be positive")
    dimensions = _parse_vehicle_types(route_file)
    tracks: dict[str, dict[int, AgentState]] = {}
    path = Path(fcd_file)
    if not path.is_file():
        raise FileNotFoundError(f"FCD file does not exist: {path}")
    for event, element in ET.iterparse(path, events=("end",)):
        if element.tag != "timestep":
            continue
        time_s = float(element.attrib["time"])
        step = int(round(time_s * frequency_hz))
        if abs(step / frequency_hz - time_s) > 1e-6:
            raise ValueError(f"FCD time {time_s} is not aligned to {frequency_hz} Hz")
        for vehicle in element.findall("vehicle"):
            vehicle_id = str(vehicle.attrib["id"])
            type_id = str(vehicle.attrib.get("type", "DEFAULT_VEHTYPE"))
            length, width, height = dimensions.get(
                type_id, (DEFAULT_LENGTH_M, DEFAULT_WIDTH_M, DEFAULT_HEIGHT_M)
            )
            state = AgentState(
                x=float(vehicle.attrib["x"]),
                y=float(vehicle.attrib["y"]),
                z=float(vehicle.attrib.get("z", 0.0)),
                heading=_sumo_bearing_to_heading(float(vehicle.attrib.get("angle", 0.0))),
                speed=float(vehicle.attrib.get("speed", 0.0)),
                length=length,
                width=width,
                height=height,
                type_id=type_id,
            )
            track = tracks.setdefault(vehicle_id, {})
            if step in track:
                raise ValueError(f"duplicate FCD state for vehicle {vehicle_id!r} at step {step}")
            track[step] = state
        element.clear()
    return tracks


def parse_tls_jsonl(
    tls_file: str | Path, frequency_hz: int = 10
) -> dict[int, dict[str, str]]:
    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be positive")
    path = Path(tls_file)
    if not path.is_file():
        raise FileNotFoundError(f"TLS state file does not exist: {path}")
    states: dict[int, dict[str, str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            time_s = float(row["time_s"])
            step = int(round(time_s * frequency_hz))
            if abs(step / frequency_hz - time_s) > 1e-6:
                raise ValueError(
                    f"TLS time {time_s} on line {line_number} is not aligned to {frequency_hz} Hz"
                )
            tls_id = str(row["tls_id"])
            step_states = states.setdefault(step, {})
            if tls_id in step_states:
                raise ValueError(f"duplicate TLS state for {tls_id!r} at step {step}")
            step_states[tls_id] = str(row["state"])
    return states


def build_window(
    tracks: Mapping[str, Mapping[int, AgentState]],
    av_id: str,
    start_step: int,
    spec: SceneSpec,
    observation_radius_m: float = 80.0,
    light_tokens: Iterable[LightToken] = (),
    tls_states: Mapping[int, Mapping[str, str]] | None = None,
    roadgraph: Roadgraph | None = None,
    meta: Mapping | None = None,
) -> SceneWindow:
    if observation_radius_m <= 0.0:
        raise ValueError("observation_radius_m must be positive")
    if av_id not in tracks:
        raise KeyError(f"AV track {av_id!r} does not exist")
    steps = tuple(range(start_step, start_step + spec.timesteps))
    reference_step = start_step + spec.history_steps - 1
    av_track = tracks[av_id]
    missing_av = [step for step in steps if step not in av_track]
    if missing_av:
        raise ValueError(
            f"AV track {av_id!r} is missing {len(missing_av)} required window steps"
        )
    reference_state = av_track[reference_step]
    reference_pose = Pose2D(
        xy=np.array([reference_state.x, reference_state.y], dtype=np.float64),
        z=float(reference_state.z),
        heading=float(reference_state.heading),
    )

    selected_ids, truncated_agents = _select_agents(
        tracks=tracks,
        av_id=av_id,
        steps=steps,
        av_track=av_track,
        radius=observation_radius_m,
        maximum=spec.max_agents,
    )
    agent_ids = tuple(selected_ids) + ("",) * (spec.max_agents - len(selected_ids))
    agents = np.zeros(
        (spec.max_agents, spec.timesteps, len(spec.agent_channels)), dtype=np.float32
    )
    agents[..., -1] = -1.0
    agent_normalizer = AgentNormalizer()
    for slot, agent_id in enumerate(selected_ids):
        track = tracks[agent_id]
        agent_type = AgentType.AV if agent_id == av_id else AgentType.CAR
        for frame, step in enumerate(steps):
            state = track.get(step)
            av_state = av_track[step]
            valid = state is not None and (
                agent_id == av_id
                or _distance_xy(state.x, state.y, av_state.x, av_state.y)
                <= observation_radius_m
            )
            if not valid:
                continue
            continuous = _state_in_reference_frame(state, reference_pose)
            agents[slot, frame] = agent_normalizer.encode_agent(
                continuous=continuous, type_index=agent_type, valid=True
            )

    tls_states = tls_states or {}
    tokens = tuple(light_tokens)
    selected_lights, truncated_lights = _select_lights(
        tokens=tokens,
        tls_states=tls_states,
        steps=steps,
        av_track=av_track,
        radius=observation_radius_m,
        maximum=spec.max_lights,
    )
    light_ids = tuple(
        f"{token.tls_id}:{token.link_index}" for token in selected_lights
    ) + ("",) * (spec.max_lights - len(selected_lights))
    lights = np.zeros(
        (spec.max_lights, spec.timesteps, len(spec.light_channels)), dtype=np.float32
    )
    lights[..., -1] = -1.0
    light_normalizer = LightNormalizer()
    for slot, token in enumerate(selected_lights):
        local_xy = _world_xy_to_local(token.stop_line_xy, reference_pose)
        xyz = np.array([local_xy[0], local_xy[1], -reference_pose.z], dtype=np.float32)
        for frame, step in enumerate(steps):
            state_string = tls_states.get(step, {}).get(token.tls_id)
            av_state = av_track[step]
            valid = state_string is not None and (
                _distance_xy(
                    token.stop_line_xy[0],
                    token.stop_line_xy[1],
                    av_state.x,
                    av_state.y,
                )
                <= observation_radius_m
            )
            if not valid:
                continue
            validate_tls_state(token.tls_id, state_string, tokens)
            state = map_sumo_signal(state_string[token.link_index], token.turn_direction)
            lights[slot, frame] = light_normalizer.encode_light(xyz, state, valid=True)

    result_meta = dict(meta or {})
    result_meta.update(
        {
            "av_id": av_id,
            "start_step": int(start_step),
            "reference_step": int(reference_step),
            "frequency_hz": int(spec.frequency_hz),
        }
    )
    return SceneWindow(
        agents=agents,
        lights=lights,
        agent_ids=agent_ids,
        light_ids=light_ids,
        reference_world_pose=reference_pose,
        meta=result_meta,
        truncated_agents=truncated_agents,
        truncated_lights=truncated_lights,
        roadgraph=roadgraph,
    )


def candidate_windows(
    tracks: Mapping[str, Mapping[int, AgentState]],
    spec: SceneSpec,
    *,
    stride: int,
    light_tokens: Iterable[LightToken] = (),
    min_reference_agents: int = 1,
    require_reference_light: bool = False,
    observation_radius_m: float = 80.0,
    minimum_travel_m: float = 20.0,
):
    if stride <= 0:
        raise ValueError("stride must be positive")
    if min_reference_agents <= 0:
        raise ValueError("min_reference_agents must be positive")
    if observation_radius_m <= 0.0 or minimum_travel_m < 0.0:
        raise ValueError("candidate radii/travel thresholds are invalid")
    states_by_step: dict[int, list[AgentState]] = {}
    for track in tracks.values():
        for step, state in track.items():
            states_by_step.setdefault(step, []).append(state)
    lights = tuple(light_tokens)
    for vehicle_id in sorted(tracks):
        track = tracks[vehicle_id]
        steps = sorted(track)
        if len(steps) < spec.timesteps:
            continue
        available = set(steps)
        for start in range(steps[0], steps[-1] - spec.timesteps + 2, stride):
            required = range(start, start + spec.timesteps)
            if not all(step in available for step in required):
                continue
            first = track[start]
            last = track[start + spec.timesteps - 1]
            if _distance_xy(first.x, first.y, last.x, last.y) < minimum_travel_m:
                continue
            reference_step = start + spec.history_steps - 1
            reference = track[reference_step]
            local_agents = sum(
                _distance_xy(reference.x, reference.y, state.x, state.y)
                <= observation_radius_m
                for state in states_by_step.get(reference_step, ())
            )
            if local_agents < min_reference_agents:
                continue
            if require_reference_light and not any(
                _distance_xy(
                    reference.x,
                    reference.y,
                    token.stop_line_xy[0],
                    token.stop_line_xy[1],
                )
                <= observation_radius_m
                for token in lights
            ):
                continue
            yield vehicle_id, start


def count_light_state_transitions(lights: np.ndarray) -> int:
    values = np.asarray(lights)
    if values.ndim != 3 or values.shape[-1] < 13:
        raise ValueError("lights must have shape [entities, time, channels>=13]")
    valid = values[..., -1] > 0.0
    states = np.argmax(values[..., 3:12], axis=-1)
    consecutive_valid = valid[:, :-1] & valid[:, 1:]
    return int(((states[:, :-1] != states[:, 1:]) & consecutive_valid).sum())


def _select_agents(
    tracks: Mapping[str, Mapping[int, AgentState]],
    av_id: str,
    steps: tuple[int, ...],
    av_track: Mapping[int, AgentState],
    radius: float,
    maximum: int,
) -> tuple[list[str], int]:
    rankings = []
    for agent_id, track in tracks.items():
        if agent_id == av_id:
            continue
        distances = [
            _distance_xy(track[step].x, track[step].y, av_track[step].x, av_track[step].y)
            for step in steps
            if step in track
        ]
        visible = [distance for distance in distances if distance <= radius]
        if visible:
            rankings.append((-len(visible), min(visible), str(agent_id)))
    rankings.sort()
    available = max(maximum - 1, 0)
    selected = [av_id] + [row[2] for row in rankings[:available]]
    return selected, max(len(rankings) - available, 0)


def _select_lights(
    tokens: tuple[LightToken, ...],
    tls_states: Mapping[int, Mapping[str, str]],
    steps: tuple[int, ...],
    av_track: Mapping[int, AgentState],
    radius: float,
    maximum: int,
) -> tuple[list[LightToken], int]:
    rankings = []
    for token in tokens:
        distances = [
            _distance_xy(
                token.stop_line_xy[0],
                token.stop_line_xy[1],
                av_track[step].x,
                av_track[step].y,
            )
            for step in steps
            if token.tls_id in tls_states.get(step, {})
        ]
        visible = [distance for distance in distances if distance <= radius]
        if visible:
            rankings.append(
                (-len(visible), min(visible), token.tls_id, token.link_index, token)
            )
    rankings.sort(key=lambda row: row[:4])
    return [row[4] for row in rankings[:maximum]], max(len(rankings) - maximum, 0)


def _state_in_reference_frame(state: AgentState, reference: Pose2D) -> np.ndarray:
    local_xy = _world_xy_to_local(np.array([state.x, state.y]), reference)
    return np.array(
        [
            local_xy[0],
            local_xy[1],
            state.z - reference.z,
            wrap_to_pi(state.heading - reference.heading),
            state.length,
            state.width,
            state.height,
        ],
        dtype=np.float32,
    )


def _world_xy_to_local(xy: np.ndarray, reference: Pose2D) -> np.ndarray:
    delta = np.asarray(xy, dtype=np.float64) - reference.xy
    cosine = np.cos(reference.heading)
    sine = np.sin(reference.heading)
    rotation = np.array([[cosine, sine], [-sine, cosine]], dtype=np.float64)
    return rotation @ delta


def _distance_xy(x0: float, y0: float, x1: float, y1: float) -> float:
    return float(np.hypot(x0 - x1, y0 - y1))


def _sumo_bearing_to_heading(angle_degrees: float) -> float:
    return wrap_to_pi(np.deg2rad(90.0 - angle_degrees))


def _parse_vehicle_types(
    route_file: str | Path | None,
) -> dict[str, tuple[float, float, float]]:
    if route_file is None:
        return {}
    path = Path(route_file)
    if not path.is_file():
        raise FileNotFoundError(f"SUMO route file does not exist: {path}")
    root = ET.parse(path).getroot()
    return {
        str(element.attrib["id"]): (
            float(element.attrib.get("length", DEFAULT_LENGTH_M)),
            float(element.attrib.get("width", DEFAULT_WIDTH_M)),
            float(element.attrib.get("height", DEFAULT_HEIGHT_M)),
        )
        for element in root.findall("vType")
    }
