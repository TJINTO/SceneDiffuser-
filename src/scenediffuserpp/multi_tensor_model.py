from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from scenediffuserpp.axial_transformer import AxialTransformerBlock
from scenediffuserpp.context_encoder import RoadgraphEncoder


@dataclass(frozen=True)
class ModelConfig:
    agent_channels: int = 12
    light_channels: int = 13
    roadgraph_channels: int = 8
    hidden_dim: int = 128
    attention_heads: int = 4
    transformer_layers: int = 4
    latent_queries: int = 64
    max_timesteps: int = 91
    dropout: float = 0.0

    def __post_init__(self) -> None:
        positive = (
            self.agent_channels,
            self.light_channels,
            self.roadgraph_channels,
            self.hidden_dim,
            self.attention_heads,
            self.transformer_layers,
            self.latent_queries,
            self.max_timesteps,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("all model dimensions must be positive")
        if self.hidden_dim % self.attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout must be within [0, 1)")


@dataclass(frozen=True)
class DenoiserOutput:
    agent_v: torch.Tensor
    light_v: torch.Tensor


class MultiTensorDenoiser(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.agent_input = _input_projection(config.agent_channels, config.hidden_dim)
        self.light_input = _input_projection(config.light_channels, config.hidden_dim)
        self.agent_condition = _condition_projection(
            config.agent_channels, config.hidden_dim
        )
        self.light_condition = _condition_projection(
            config.light_channels, config.hidden_dim
        )
        self.noisy_local_projection = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.type_embedding = nn.Embedding(2, config.hidden_dim)
        self.diffusion_embedding = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
        )
        self.roadgraph_encoder = RoadgraphEncoder(
            input_dim=config.roadgraph_channels,
            hidden_dim=config.hidden_dim,
            attention_heads=config.attention_heads,
            latent_queries=config.latent_queries,
            dropout=config.dropout,
        )
        self.conditioning_fusion = nn.MultiheadAttention(
            config.hidden_dim,
            config.attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.conditioning_norm = nn.LayerNorm(config.hidden_dim)
        self.blocks = nn.ModuleList(
            AxialTransformerBlock(
                config.hidden_dim,
                config.attention_heads,
                dropout=config.dropout,
            )
            for _ in range(config.transformer_layers)
        )
        self.agent_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.agent_channels),
        )
        self.light_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.light_channels),
        )

    def forward(
        self,
        *,
        agent_z: torch.Tensor,
        light_z: torch.Tensor,
        agent_context: torch.Tensor,
        light_context: torch.Tensor,
        agent_inpaint_mask: torch.Tensor,
        light_inpaint_mask: torch.Tensor,
        roadgraph: torch.Tensor,
        roadgraph_padding_mask: torch.Tensor,
        diffusion_time: torch.Tensor,
        roadgraph_point_lane_index: torch.Tensor | None = None,
        roadgraph_lane_padding_mask: torch.Tensor | None = None,
        roadgraph_successor_index: torch.Tensor | None = None,
        roadgraph_successor_padding_mask: torch.Tensor | None = None,
    ) -> DenoiserOutput:
        self._validate_inputs(
            agent_z,
            light_z,
            agent_context,
            light_context,
            agent_inpaint_mask,
            light_inpaint_mask,
            roadgraph,
            roadgraph_padding_mask,
            diffusion_time,
            roadgraph_point_lane_index,
            roadgraph_lane_padding_mask,
            roadgraph_successor_index,
            roadgraph_successor_padding_mask,
        )
        batch, agent_count, timesteps, _ = agent_z.shape
        light_count = light_z.shape[1]
        agent_known = torch.where(
            agent_inpaint_mask, agent_context, torch.zeros_like(agent_context)
        )
        light_known = torch.where(
            light_inpaint_mask, light_context, torch.zeros_like(light_context)
        )
        agent_condition = torch.cat(
            (
                agent_known,
                agent_inpaint_mask.to(agent_z.dtype),
            ),
            dim=-1,
        )
        light_condition = torch.cat(
            (
                light_known,
                light_inpaint_mask.to(light_z.dtype),
            ),
            dim=-1,
        )
        agents = self.agent_input(agent_z)
        lights = self.light_input(light_z)
        agent_condition = self.agent_condition(agent_condition)
        light_condition = self.light_condition(light_condition)
        agents = agents + self.type_embedding.weight[0].view(1, 1, 1, -1)
        lights = lights + self.type_embedding.weight[1].view(1, 1, 1, -1)
        agent_condition = agent_condition + self.type_embedding.weight[0].view(
            1, 1, 1, -1
        )
        light_condition = light_condition + self.type_embedding.weight[1].view(
            1, 1, 1, -1
        )

        physical_time = torch.linspace(
            0.0,
            1.0,
            timesteps,
            device=agent_z.device,
            dtype=agent_z.dtype,
        )
        physical_embedding = sinusoidal_embedding(physical_time, self.config.hidden_dim)
        diffusion_embedding = self.diffusion_embedding(
            sinusoidal_embedding(diffusion_time, self.config.hidden_dim)
        )
        physical = physical_embedding.view(1, 1, timesteps, -1)
        agents = agents + physical
        lights = lights + physical
        agent_condition = agent_condition + physical
        light_condition = light_condition + physical

        values = torch.cat((agents, lights), dim=1)
        local_condition = torch.cat((agent_condition, light_condition), dim=1)
        noisy_local_condition = self.noisy_local_projection(
            torch.cat((values, local_condition), dim=-1)
        )
        global_context = self.roadgraph_encoder(
            roadgraph,
            roadgraph_padding_mask,
            point_lane_index=roadgraph_point_lane_index,
            lane_padding_mask=roadgraph_lane_padding_mask,
            successor_index=roadgraph_successor_index,
            successor_padding_mask=roadgraph_successor_padding_mask,
        )
        flat_condition = noisy_local_condition.reshape(
            batch, (agent_count + light_count) * timesteps, self.config.hidden_dim
        )
        fused, _ = self.conditioning_fusion(
            flat_condition,
            global_context,
            global_context,
            need_weights=False,
        )
        conditioning = self.conditioning_norm(flat_condition + fused).reshape_as(values)
        conditioning = conditioning + diffusion_embedding.view(batch, 1, 1, -1)
        for block in self.blocks:
            values = block(values, conditioning)
        return DenoiserOutput(
            agent_v=self.agent_head(values[:, :agent_count]),
            light_v=self.light_head(values[:, agent_count : agent_count + light_count]),
        )

    def _validate_inputs(
        self,
        agent_z: torch.Tensor,
        light_z: torch.Tensor,
        agent_context: torch.Tensor,
        light_context: torch.Tensor,
        agent_inpaint_mask: torch.Tensor,
        light_inpaint_mask: torch.Tensor,
        roadgraph: torch.Tensor,
        roadgraph_padding_mask: torch.Tensor,
        diffusion_time: torch.Tensor,
        roadgraph_point_lane_index: torch.Tensor | None,
        roadgraph_lane_padding_mask: torch.Tensor | None,
        roadgraph_successor_index: torch.Tensor | None,
        roadgraph_successor_padding_mask: torch.Tensor | None,
    ) -> None:
        if agent_z.ndim != 4 or agent_z.shape[-1] != self.config.agent_channels:
            raise ValueError("agent_z must have shape [B,A,T,agent_channels]")
        if light_z.ndim != 4 or light_z.shape[-1] != self.config.light_channels:
            raise ValueError("light_z must have shape [B,L,T,light_channels]")
        if agent_context.shape != agent_z.shape:
            raise ValueError("agent_context shape must match agent_z")
        if light_context.shape != light_z.shape:
            raise ValueError("light_context shape must match light_z")
        if agent_inpaint_mask.shape != agent_z.shape or agent_inpaint_mask.dtype != torch.bool:
            raise ValueError("agent_inpaint_mask must be bool and match agent_z")
        if light_inpaint_mask.shape != light_z.shape or light_inpaint_mask.dtype != torch.bool:
            raise ValueError("light_inpaint_mask must be bool and match light_z")
        if agent_z.shape[0] != light_z.shape[0] or agent_z.shape[2] != light_z.shape[2]:
            raise ValueError("agent and light batch/time dimensions must match")
        if agent_z.shape[2] > self.config.max_timesteps:
            raise ValueError("input exceeds configured maximum timesteps")
        if roadgraph.shape[:2] != roadgraph_padding_mask.shape:
            raise ValueError("roadgraph padding shape mismatch")
        if roadgraph.ndim != 3 or roadgraph.shape[-1] != self.config.roadgraph_channels:
            raise ValueError("roadgraph channel shape mismatch")
        if roadgraph.shape[0] != agent_z.shape[0]:
            raise ValueError("roadgraph batch must match agents")
        if roadgraph_padding_mask.dtype != torch.bool:
            raise ValueError("roadgraph_padding_mask must be boolean")
        if diffusion_time.shape != (agent_z.shape[0],):
            raise ValueError("diffusion_time must have shape [batch]")
        topology = (
            roadgraph_point_lane_index,
            roadgraph_lane_padding_mask,
            roadgraph_successor_index,
            roadgraph_successor_padding_mask,
        )
        if any(value is None for value in topology) and not all(
            value is None for value in topology
        ):
            raise ValueError("all roadgraph topology tensors must be provided together")
        if all(value is not None for value in topology) and any(
            value.device != agent_z.device for value in topology
        ):
            raise ValueError("roadgraph topology tensors must use the model device")
        tensors = (
            light_z,
            agent_context,
            light_context,
            roadgraph,
            diffusion_time,
        )
        if any(value.device != agent_z.device for value in tensors):
            raise ValueError("all floating model inputs must use one device")
        if any(value.dtype != agent_z.dtype for value in tensors):
            raise ValueError("all floating model inputs must use one dtype")


def sinusoidal_embedding(values: torch.Tensor, dimension: int) -> torch.Tensor:
    if values.ndim != 1:
        raise ValueError("sinusoidal inputs must be one-dimensional")
    half = dimension // 2
    if half == 0:
        return values.unsqueeze(-1)
    frequency = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=values.device, dtype=values.dtype)
        / max(half - 1, 1)
    )
    angles = values.unsqueeze(-1) * frequency.unsqueeze(0) * (2.0 * math.pi)
    embedding = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
    if embedding.shape[-1] < dimension:
        embedding = torch.nn.functional.pad(embedding, (0, dimension - embedding.shape[-1]))
    return embedding


def _input_projection(channels: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(channels, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
    )


def _condition_projection(channels: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(channels * 2, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
    )
