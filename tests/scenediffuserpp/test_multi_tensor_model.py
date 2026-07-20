import copy

import pytest
import torch

from scenediffuserpp.context_encoder import RoadgraphEncoder
from scenediffuserpp.multi_tensor_model import ModelConfig
from scenediffuserpp.multi_tensor_model import MultiTensorDenoiser


def _config() -> ModelConfig:
    return ModelConfig(
        hidden_dim=32,
        attention_heads=4,
        transformer_layers=2,
        latent_queries=8,
        max_timesteps=91,
        dropout=0.0,
    )


def _batch() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(123)
    agents = torch.randn(2, 4, 7, 12, generator=generator)
    lights = torch.randn(2, 3, 7, 13, generator=generator)
    roadgraph = torch.randn(2, 9, 8, generator=generator)
    map_padding = torch.tensor(
        [
            [False, False, False, False, False, True, True, True, True],
            [False, False, False, True, True, True, True, True, True],
        ]
    )
    point_lane_index = torch.tensor(
        [
            [0, 0, 1, 1, 2, -1, -1, -1, -1],
            [0, 1, 1, -1, -1, -1, -1, -1, -1],
        ]
    )
    lane_padding = torch.tensor(
        [
            [False, False, False, True],
            [False, False, True, True],
        ]
    )
    successor_index = torch.tensor(
        [
            [[0, 1, -1], [1, 2, -1]],
            [[0, -1, -1], [1, -1, -1]],
        ]
    )
    successor_padding = torch.tensor(
        [[False, False, True], [False, True, True]]
    )
    return {
        "agent_z": agents,
        "light_z": lights,
        "agent_context": torch.randn(2, 4, 7, 12, generator=generator),
        "light_context": torch.randn(2, 3, 7, 13, generator=generator),
        "agent_inpaint_mask": torch.rand(2, 4, 7, 12, generator=generator) > 0.7,
        "light_inpaint_mask": torch.rand(2, 3, 7, 13, generator=generator) > 0.7,
        "roadgraph": roadgraph,
        "roadgraph_padding_mask": map_padding,
        "roadgraph_point_lane_index": point_lane_index,
        "roadgraph_lane_padding_mask": lane_padding,
        "roadgraph_successor_index": successor_index,
        "roadgraph_successor_padding_mask": successor_padding,
        "diffusion_time": torch.tensor([0.2, 0.8]),
    }


def test_denoiser_returns_original_heterogeneous_shapes():
    batch = _batch()
    model = MultiTensorDenoiser(_config())

    output = model(**batch)

    assert output.agent_v.shape == batch["agent_z"].shape
    assert output.light_v.shape == batch["light_z"].shape


def test_all_parameter_groups_receive_finite_gradients():
    model = MultiTensorDenoiser(_config())
    output = model(**_batch())

    (output.agent_v.square().mean() + output.light_v.square().mean()).backward()

    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and (parameter.grad is None or not torch.isfinite(parameter.grad).all())
    ]
    assert missing == []


def test_masked_map_padding_cannot_change_output():
    model = MultiTensorDenoiser(_config()).eval()
    batch = _batch()
    changed = copy.deepcopy(batch)
    changed["roadgraph"][changed["roadgraph_padding_mask"]] = 9999.0

    with torch.no_grad():
        first = model(**batch)
        second = model(**changed)

    torch.testing.assert_close(first.agent_v, second.agent_v)
    torch.testing.assert_close(first.light_v, second.light_v)


def test_model_rejects_mismatched_agent_context_shape():
    model = MultiTensorDenoiser(_config())
    batch = _batch()
    batch["agent_context"] = batch["agent_context"][:, :, :-1]

    with pytest.raises(ValueError, match="agent_context"):
        model(**batch)


def test_axial_blocks_use_zero_initialized_adaln_conditioning():
    model = MultiTensorDenoiser(_config())

    for block in model.blocks:
        assert hasattr(block, "modulation")
        torch.testing.assert_close(
            block.modulation[-1].weight,
            torch.zeros_like(block.modulation[-1].weight),
        )
        torch.testing.assert_close(
            block.modulation[-1].bias,
            torch.zeros_like(block.modulation[-1].bias),
        )


def test_noisy_scene_tokens_enter_global_context_fusion_query():
    model = MultiTensorDenoiser(_config()).eval()
    batch = _batch()
    changed = copy.deepcopy(batch)
    changed["agent_z"] = changed["agent_z"] + 0.25
    captured_queries = []

    def capture_query(_module, inputs):
        captured_queries.append(inputs[0].detach().clone())

    handle = model.conditioning_fusion.register_forward_pre_hook(capture_query)
    try:
        with torch.no_grad():
            model(**batch)
            model(**changed)
    finally:
        handle.remove()

    assert len(captured_queries) == 2
    assert not torch.allclose(captured_queries[0], captured_queries[1])


def test_roadgraph_context_changes_when_only_legal_successors_change():
    torch.manual_seed(11)
    encoder = RoadgraphEncoder(
        input_dim=8,
        hidden_dim=16,
        attention_heads=4,
        latent_queries=4,
        dropout=0.0,
    ).eval()
    points = torch.zeros(1, 4, 8)
    points[0, :, 0] = torch.tensor([-0.8, -0.4, 0.4, 0.8])
    padding = torch.zeros(1, 4, dtype=torch.bool)
    point_lanes = torch.tensor([[0, 0, 1, 1]])
    lane_padding = torch.tensor([[False, False]])
    edge_padding = torch.tensor([[False]])

    forward = encoder(
        points,
        padding,
        point_lane_index=point_lanes,
        lane_padding_mask=lane_padding,
        successor_index=torch.tensor([[[0], [1]]]),
        successor_padding_mask=edge_padding,
    )
    reverse = encoder(
        points,
        padding,
        point_lane_index=point_lanes,
        lane_padding_mask=lane_padding,
        successor_index=torch.tensor([[[1], [0]]]),
        successor_padding_mask=edge_padding,
    )

    assert not torch.allclose(forward, reverse)


def test_topology_encoder_keeps_an_empty_graph_finite():
    encoder = RoadgraphEncoder(
        input_dim=8,
        hidden_dim=16,
        attention_heads=4,
        latent_queries=4,
    )

    encoded = encoder(
        torch.zeros(1, 3, 8),
        torch.ones(1, 3, dtype=torch.bool),
        point_lane_index=torch.full((1, 3), -1),
        lane_padding_mask=torch.ones(1, 2, dtype=torch.bool),
        successor_index=torch.full((1, 2, 2), -1),
        successor_padding_mask=torch.ones(1, 2, dtype=torch.bool),
    )

    assert torch.isfinite(encoded).all()
