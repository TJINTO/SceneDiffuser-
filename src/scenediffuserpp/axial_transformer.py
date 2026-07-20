from __future__ import annotations

import torch
from torch import nn


class AxialTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        attention_heads: int,
        dropout: float = 0.0,
        expansion: int = 4,
    ) -> None:
        super().__init__()
        self.entity_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.entity_attention = nn.MultiheadAttention(
            hidden_dim, attention_heads, dropout=dropout, batch_first=True
        )
        self.temporal_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.temporal_attention = nn.MultiheadAttention(
            hidden_dim, attention_heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * expansion, hidden_dim),
            nn.Dropout(dropout),
        )
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 9),
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, values: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        if values.ndim != 4:
            raise ValueError("axial values must have shape [batch, entities, time, hidden]")
        batch, entities, timesteps, hidden = values.shape
        if conditioning.shape != values.shape:
            raise ValueError("AdaLN conditioning must match axial values")
        (
            entity_shift,
            entity_scale,
            entity_gate,
            temporal_shift,
            temporal_scale,
            temporal_gate,
            ffn_shift,
            ffn_scale,
            ffn_gate,
        ) = self.modulation(conditioning).chunk(9, dim=-1)

        entity_input = _modulate(
            self.entity_norm(values), entity_shift, entity_scale
        )
        entity_input = entity_input.permute(0, 2, 1, 3).reshape(
            batch * timesteps, entities, hidden
        )
        entity_output, _ = self.entity_attention(
            entity_input, entity_input, entity_input, need_weights=False
        )
        entity_output = entity_output.reshape(batch, timesteps, entities, hidden).permute(
            0, 2, 1, 3
        )
        values = values + entity_gate * entity_output

        temporal_input = _modulate(
            self.temporal_norm(values), temporal_shift, temporal_scale
        ).reshape(
            batch * entities, timesteps, hidden
        )
        temporal_output, _ = self.temporal_attention(
            temporal_input, temporal_input, temporal_input, need_weights=False
        )
        values = values + temporal_gate * temporal_output.reshape(
            batch, entities, timesteps, hidden
        )
        ffn_input = _modulate(self.ffn_norm(values), ffn_shift, ffn_scale)
        return values + ffn_gate * self.ffn(ffn_input)


def _modulate(
    values: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return values * (1.0 + scale) + shift
