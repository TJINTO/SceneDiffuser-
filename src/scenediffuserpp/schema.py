from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any
from typing import Mapping


class AgentType(IntEnum):
    AV = 0
    CAR = 1
    PEDESTRIAN = 2
    CYCLIST = 3


class LightState(IntEnum):
    UNKNOWN = 0
    GREEN_ARROW = 1
    YELLOW_ARROW = 2
    RED_ARROW = 3
    GREEN = 4
    YELLOW = 5
    RED = 6
    FLASHING_RED = 7
    FLASHING_YELLOW = 8


AGENT_CHANNELS = (
    "x",
    "y",
    "z",
    "heading",
    "length",
    "width",
    "height",
    "type_av",
    "type_car",
    "type_pedestrian",
    "type_cyclist",
    "validity",
)

LIGHT_CHANNELS = (
    "x",
    "y",
    "z",
    "state_unknown",
    "state_green_arrow",
    "state_yellow_arrow",
    "state_red_arrow",
    "state_green",
    "state_yellow",
    "state_red",
    "state_flashing_red",
    "state_flashing_yellow",
    "validity",
)


@dataclass(frozen=True)
class SceneSpec:
    frequency_hz: int
    timesteps: int
    history_steps: int
    future_steps: int
    max_agents: int
    max_lights: int
    latent_queries: int
    hidden_dim: int
    transformer_layers: int
    attention_heads: int
    sampling_steps: int
    agent_channels: tuple[str, ...] = AGENT_CHANNELS
    light_channels: tuple[str, ...] = LIGHT_CHANNELS

    def __post_init__(self) -> None:
        if self.history_steps + self.future_steps != self.timesteps:
            raise ValueError("history_steps plus future_steps must equal timesteps")
        positive = {
            "frequency_hz": self.frequency_hz,
            "timesteps": self.timesteps,
            "history_steps": self.history_steps,
            "future_steps": self.future_steps,
            "max_agents": self.max_agents,
            "max_lights": self.max_lights,
            "latent_queries": self.latent_queries,
            "hidden_dim": self.hidden_dim,
            "transformer_layers": self.transformer_layers,
            "attention_heads": self.attention_heads,
            "sampling_steps": self.sampling_steps,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"SceneSpec values must be positive: {', '.join(invalid)}")
        if self.hidden_dim % self.attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        if self.agent_channels[-1] != "validity":
            raise ValueError("agent validity must be the final channel")
        if self.light_channels[-1] != "validity":
            raise ValueError("light validity must be the final channel")

    @classmethod
    def small(cls) -> "SceneSpec":
        return cls(
            frequency_hz=10,
            timesteps=91,
            history_steps=11,
            future_steps=80,
            max_agents=32,
            max_lights=32,
            latent_queries=64,
            hidden_dim=128,
            transformer_layers=4,
            attention_heads=4,
            sampling_steps=32,
        )

    @classmethod
    def paper(cls) -> "SceneSpec":
        return cls(
            frequency_hz=10,
            timesteps=91,
            history_steps=11,
            future_steps=80,
            max_agents=128,
            max_lights=64,
            latent_queries=192,
            hidden_dim=512,
            transformer_layers=8,
            attention_heads=8,
            sampling_steps=32,
        )


@dataclass(frozen=True)
class DatasetBuildConfig:
    scene_spec: SceneSpec
    observation_radius_m: float = 80.0
    map_radius_m: float = 1000.0
    map_point_spacing_m: float = 10.0
    maximum_map_points: int = 12288
    maximum_map_lanes: int = 1024
    maximum_map_connections: int = 4096
    window_stride_steps: int = 20
    minimum_av_travel_m: float = 20.0
    minimum_reference_agents: int = 1
    require_reference_light: bool = False
    minimum_light_state_transitions: int = 0
    shard_size: int = 512
    split_seed: int = 20260720

    def __post_init__(self) -> None:
        if self.observation_radius_m <= 0.0:
            raise ValueError("observation radius must be positive")
        if self.map_radius_m < self.observation_radius_m:
            raise ValueError("map radius cannot be smaller than observation radius")
        if self.map_point_spacing_m <= 0.0:
            raise ValueError("map_point_spacing_m must be positive")
        if self.maximum_map_points <= 0:
            raise ValueError("maximum_map_points must be positive")
        if self.maximum_map_lanes <= 0 or self.maximum_map_connections <= 0:
            raise ValueError("map lane and connection capacities must be positive")
        if self.window_stride_steps <= 0:
            raise ValueError("window_stride_steps must be positive")
        if self.minimum_av_travel_m < 0.0:
            raise ValueError("minimum_av_travel_m must be nonnegative")
        if self.minimum_reference_agents <= 0:
            raise ValueError("minimum_reference_agents must be positive")
        if self.minimum_light_state_transitions < 0:
            raise ValueError("minimum_light_state_transitions must be nonnegative")
        if self.shard_size <= 0:
            raise ValueError("shard_size must be positive")


def dataset_build_config_from_mapping(
    values: Mapping[str, Any],
) -> DatasetBuildConfig:
    dataset = values.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("dataset config must contain a 'dataset' mapping")
    base = SceneSpec.small()
    spec = SceneSpec(
        frequency_hz=int(dataset.get("frequency_hz", base.frequency_hz)),
        timesteps=int(dataset.get("timesteps", base.timesteps)),
        history_steps=int(dataset.get("history_steps", base.history_steps)),
        future_steps=int(dataset.get("future_steps", base.future_steps)),
        max_agents=int(dataset.get("maximum_agents", base.max_agents)),
        max_lights=int(dataset.get("maximum_lights", base.max_lights)),
        latent_queries=base.latent_queries,
        hidden_dim=base.hidden_dim,
        transformer_layers=base.transformer_layers,
        attention_heads=base.attention_heads,
        sampling_steps=base.sampling_steps,
    )
    require_reference_light = dataset.get("require_reference_light", False)
    if not isinstance(require_reference_light, bool):
        raise ValueError("require_reference_light must be boolean")
    return DatasetBuildConfig(
        scene_spec=spec,
        observation_radius_m=float(dataset.get("observation_radius_m", 80.0)),
        map_radius_m=float(dataset.get("map_radius_m", 1000.0)),
        map_point_spacing_m=float(dataset.get("map_point_spacing_m", 10.0)),
        maximum_map_points=int(dataset.get("maximum_map_points", 12288)),
        maximum_map_lanes=int(dataset.get("maximum_map_lanes", 1024)),
        maximum_map_connections=int(
            dataset.get("maximum_map_connections", 4096)
        ),
        window_stride_steps=int(dataset.get("window_stride_steps", 20)),
        minimum_av_travel_m=float(dataset.get("minimum_av_travel_m", 20.0)),
        minimum_reference_agents=int(dataset.get("minimum_reference_agents", 1)),
        require_reference_light=require_reference_light,
        minimum_light_state_transitions=int(
            dataset.get("minimum_light_state_transitions", 0)
        ),
        shard_size=int(dataset.get("shard_size", 512)),
        split_seed=int(dataset.get("split_seed", 20260720)),
    )
