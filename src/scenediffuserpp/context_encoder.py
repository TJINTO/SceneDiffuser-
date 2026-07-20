from __future__ import annotations

import torch
from torch import nn


class RoadgraphEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        attention_heads: int,
        latent_queries: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.point_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.predecessor_message = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.topology_norm = nn.LayerNorm(hidden_dim)
        self.queries = nn.Parameter(torch.empty(latent_queries, hidden_dim))
        nn.init.normal_(self.queries, std=hidden_dim**-0.5)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim,
            attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        points: torch.Tensor,
        padding_mask: torch.Tensor,
        *,
        point_lane_index: torch.Tensor | None = None,
        lane_padding_mask: torch.Tensor | None = None,
        successor_index: torch.Tensor | None = None,
        successor_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if points.ndim != 3:
            raise ValueError("roadgraph must have shape [batch, points, channels]")
        if padding_mask.shape != points.shape[:2] or padding_mask.dtype != torch.bool:
            raise ValueError("roadgraph_padding_mask must be bool with shape [batch, points]")
        encoded = self.point_encoder(points)
        topology = (
            point_lane_index,
            lane_padding_mask,
            successor_index,
            successor_padding_mask,
        )
        if all(value is None for value in topology):
            tokens = encoded
            token_padding = padding_mask
        elif any(value is None for value in topology):
            raise ValueError("all roadgraph topology tensors must be provided together")
        else:
            tokens = self._encode_topology(
                encoded,
                padding_mask,
                point_lane_index=point_lane_index,
                lane_padding_mask=lane_padding_mask,
                successor_index=successor_index,
                successor_padding_mask=successor_padding_mask,
            )
            token_padding = lane_padding_mask
        safe_padding = token_padding.clone()
        all_padded = safe_padding.all(dim=1)
        if all_padded.any():
            safe_padding[all_padded, 0] = False
            tokens = tokens.clone()
            tokens[all_padded, 0] = 0.0
        queries = self.queries.unsqueeze(0).expand(points.shape[0], -1, -1)
        context, _ = self.cross_attention(
            queries,
            tokens,
            tokens,
            key_padding_mask=safe_padding,
            need_weights=False,
        )
        return self.output_norm(queries + context)

    def _encode_topology(
        self,
        encoded_points: torch.Tensor,
        point_padding_mask: torch.Tensor,
        *,
        point_lane_index: torch.Tensor,
        lane_padding_mask: torch.Tensor,
        successor_index: torch.Tensor,
        successor_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, point_count, hidden_dim = encoded_points.shape
        if point_lane_index.shape != (batch, point_count):
            raise ValueError("point_lane_index must have shape [batch, points]")
        if point_lane_index.dtype != torch.long:
            raise ValueError("point_lane_index must use torch.long")
        if lane_padding_mask.ndim != 2 or lane_padding_mask.shape[0] != batch:
            raise ValueError("lane_padding_mask must have shape [batch, lanes]")
        if lane_padding_mask.dtype != torch.bool:
            raise ValueError("lane_padding_mask must be boolean")
        lane_count = lane_padding_mask.shape[1]
        if lane_count == 0:
            raise ValueError("roadgraph topology requires at least one lane slot")
        if successor_index.ndim != 3 or successor_index.shape[:2] != (batch, 2):
            raise ValueError("successor_index must have shape [batch, 2, edges]")
        edge_count = successor_index.shape[2]
        if successor_index.dtype != torch.long:
            raise ValueError("successor_index must use torch.long")
        if successor_padding_mask.shape != (batch, edge_count):
            raise ValueError("successor padding must have shape [batch, edges]")
        if successor_padding_mask.dtype != torch.bool:
            raise ValueError("successor padding mask must be boolean")
        integer_tensors = (point_lane_index, successor_index)
        mask_tensors = (lane_padding_mask, successor_padding_mask)
        if any(value.device != encoded_points.device for value in integer_tensors + mask_tensors):
            raise ValueError("roadgraph topology tensors must share the point device")

        valid_points = ~point_padding_mask
        valid_point_lanes = point_lane_index[valid_points]
        if valid_point_lanes.numel() and (
            (valid_point_lanes < 0).any() or (valid_point_lanes >= lane_count).any()
        ):
            raise ValueError("valid points reference an invalid lane index")
        safe_point_lanes = point_lane_index.clamp(min=0, max=lane_count - 1)
        lane_sums = encoded_points.new_zeros(batch, lane_count, hidden_dim)
        lane_sums.scatter_add_(
            1,
            safe_point_lanes.unsqueeze(-1).expand(-1, -1, hidden_dim),
            encoded_points * valid_points.unsqueeze(-1),
        )
        lane_counts = encoded_points.new_zeros(batch, lane_count, 1)
        lane_counts.scatter_add_(
            1,
            safe_point_lanes.unsqueeze(-1),
            valid_points.unsqueeze(-1).to(encoded_points.dtype),
        )
        valid_lanes = ~lane_padding_mask
        if (valid_lanes & (lane_counts.squeeze(-1) == 0)).any():
            raise ValueError("an unpadded lane has no map points")
        lane_tokens = lane_sums / lane_counts.clamp_min(1.0)

        valid_edges = ~successor_padding_mask
        edge_values = successor_index.permute(0, 2, 1)
        active_edges = edge_values[valid_edges]
        if active_edges.numel() and (
            (active_edges < 0).any() or (active_edges >= lane_count).any()
        ):
            raise ValueError("a successor edge references an invalid lane index")
        safe_edges = edge_values.clamp(min=0, max=lane_count - 1)
        source = safe_edges[..., 0]
        target = safe_edges[..., 1]
        source_tokens = lane_tokens.gather(
            1, source.unsqueeze(-1).expand(-1, -1, hidden_dim)
        )
        source_messages = self.predecessor_message(source_tokens)
        source_messages = source_messages * valid_edges.unsqueeze(-1)
        messages = lane_tokens.new_zeros(batch, lane_count, hidden_dim)
        messages.scatter_add_(
            1,
            target.unsqueeze(-1).expand(-1, -1, hidden_dim),
            source_messages,
        )
        degrees = lane_tokens.new_zeros(batch, lane_count, 1)
        degrees.scatter_add_(
            1,
            target.unsqueeze(-1),
            valid_edges.unsqueeze(-1).to(lane_tokens.dtype),
        )
        lane_tokens = self.topology_norm(
            lane_tokens + messages / degrees.clamp_min(1.0)
        )
        return lane_tokens.masked_fill(lane_padding_mask.unsqueeze(-1), 0.0)
