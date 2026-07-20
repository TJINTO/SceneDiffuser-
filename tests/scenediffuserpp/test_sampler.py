from types import SimpleNamespace

import pytest
import torch
from torch import nn

from topoworld.scenediffuserpp.sampler import sample_scene
from topoworld.scenediffuserpp.sampler import project_sparse_agent_speed
from topoworld.scenediffuserpp.sampler import soft_clip_sparse
import topoworld.scenediffuserpp.sampler as sampler_module
from topoworld.scenediffuserpp.normalization import POSITION_SCALE


class ZeroVelocityDenoiser(nn.Module):
    def forward(self, **inputs):
        return SimpleNamespace(
            agent_v=torch.zeros_like(inputs["agent_z"]),
            light_v=torch.zeros_like(inputs["light_z"]),
        )


class TopologyCheckingDenoiser(ZeroVelocityDenoiser):
    def __init__(self):
        super().__init__()
        self.saw_topology = False

    def forward(self, **inputs):
        self.saw_topology = "roadgraph_successor_index" in inputs
        return super().forward(**inputs)


class RecordingDenoiser(ZeroVelocityDenoiser):
    def __init__(self):
        super().__init__()
        self.agent_inputs: list[torch.Tensor] = []

    def forward(self, **inputs):
        self.agent_inputs.append(inputs["agent_z"].detach().clone())
        return super().forward(**inputs)


def _inputs() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(81)
    agents = torch.randn(2, 3, 7, 12, generator=generator)
    lights = torch.randn(2, 2, 7, 13, generator=generator)
    agent_mask = torch.zeros_like(agents, dtype=torch.bool)
    light_mask = torch.zeros_like(lights, dtype=torch.bool)
    agent_mask[:, 0, :3] = True
    light_mask[:, 0, :2] = True
    return {
        "agent_context": agents,
        "light_context": lights,
        "agent_inpaint_mask": agent_mask,
        "light_inpaint_mask": light_mask,
        "roadgraph": torch.randn(2, 8, 8, generator=generator),
        "roadgraph_padding_mask": torch.tensor(
            [
                [False, False, False, False, True, True, True, True],
                [False, False, False, True, True, True, True, True],
            ]
        ),
    }


def test_soft_clip_scales_values_by_continuous_validity():
    values = torch.tensor([[[[2.0, 0.0]]]])

    clipped = soft_clip_sparse(values)

    assert clipped[..., 0].item() == 1.0
    assert clipped[..., -1].item() == 0.0


def test_paper_soft_clip_keeps_recursive_validity_in_signed_training_domain():
    values = torch.tensor([[[[2.0, -0.5]]]])

    clipped = soft_clip_sparse(values, validity_mode="paper")

    assert clipped[..., 0].item() == 0.5
    assert clipped[..., -1].item() == -0.5


def test_signed_soft_clip_keeps_recursive_validity_in_training_domain():
    values = torch.tensor([[[[2.0, -0.5]]]])

    clipped = soft_clip_sparse(values, validity_mode="signed_stable")

    assert clipped[..., 0].item() == 0.5
    assert clipped[..., -1].item() == -0.5


def test_soft_clip_rejects_unknown_validity_mode():
    with pytest.raises(ValueError, match="validity_mode"):
        soft_clip_sparse(torch.tensor([[[[2.0, -0.5]]]]), validity_mode="unknown")


def test_linear_sampling_time_grid_includes_published_noise_endpoints():
    schedule = sampler_module.sampling_time_grid(
        4, time_grid="linear", device=torch.device("cpu"), dtype=torch.float32
    )

    torch.testing.assert_close(
        schedule, torch.tensor([1.0, 0.75, 0.50, 0.25, 0.0])
    )


def test_paper_denoise_and_renoise_transition_matches_fixed_equations():
    noisy = torch.tensor([[[[0.8, -0.4]]]])
    velocity = torch.tensor([[[[0.2, 0.1]]]])
    noise = torch.tensor([[[[-0.3, 0.5]]]])
    current_time = torch.tensor([0.5])
    next_time = torch.tensor([0.25])

    clean = sampler_module.paper_denoise(
        noisy,
        velocity,
        current_time,
        validity_mode="paper",
    )
    renoised = sampler_module.paper_renoise(clean, noise, next_time)

    alpha_current = torch.cos(torch.tensor(torch.pi * 0.25))
    sigma_current = torch.sin(torch.tensor(torch.pi * 0.25))
    raw_clean = alpha_current * noisy - sigma_current * velocity
    validity_probability = raw_clean[..., -1].clamp(-1.0, 1.0) * 0.5 + 0.5
    expected_clean = torch.cat(
        (
            raw_clean[..., :-1] * validity_probability.unsqueeze(-1),
            (2.0 * validity_probability - 1.0).unsqueeze(-1),
        ),
        dim=-1,
    )
    alpha_next = torch.cos(torch.tensor(torch.pi * 0.125))
    sigma_next = torch.sin(torch.tensor(torch.pi * 0.125))
    expected_renoised = alpha_next * expected_clean + sigma_next * noise

    torch.testing.assert_close(clean, expected_clean)
    torch.testing.assert_close(renoised, expected_renoised)


def test_paper_soft_clip_does_not_turn_negative_validity_positive_recursively():
    values = torch.zeros(1, 1, 6, 2)
    values[..., -1] = -0.25

    clipped = values
    for _ in range(4):
        clipped = soft_clip_sparse(clipped, validity_mode="paper")

    assert (clipped[..., -1] < 0.0).all()


def test_inpainted_context_uses_independent_noise_at_each_paper_renoise_step():
    inputs = _inputs()
    model = RecordingDenoiser()

    sample_scene(model, **inputs, num_steps=3, seed=19, validity_mode="paper")

    context = inputs["agent_context"][0, 0, 0, 0]
    first_time = torch.tensor(1.0)
    second_time = torch.tensor(2.0 / 3.0)
    first_alpha = torch.cos(first_time * torch.pi / 2.0)
    first_sigma = torch.sin(first_time * torch.pi / 2.0)
    second_alpha = torch.cos(second_time * torch.pi / 2.0)
    second_sigma = torch.sin(second_time * torch.pi / 2.0)
    first_noise = (model.agent_inputs[0][0, 0, 0, 0] - first_alpha * context) / first_sigma
    second_noise = (
        model.agent_inputs[1][0, 0, 0, 0] - second_alpha * context
    ) / second_sigma

    assert not torch.isclose(first_noise, second_noise)


def test_sampler_rejects_unpublished_sampler_name():
    with pytest.raises(ValueError, match="sampler"):
        sample_scene(
            ZeroVelocityDenoiser(),
            **_inputs(),
            num_steps=4,
            seed=3,
            sampler="dpmpp_heun",
        )


def test_sampler_rejects_unknown_time_grid():
    with pytest.raises(ValueError, match="time_grid"):
        sample_scene(
            ZeroVelocityDenoiser(),
            **_inputs(),
            num_steps=4,
            seed=3,
            time_grid="quadratic",
        )


@pytest.mark.parametrize("validity_mode", ["paper", "signed_stable"])
def test_sampler_preserves_every_inpainted_value(validity_mode: str):
    inputs = _inputs()

    result = sample_scene(
        ZeroVelocityDenoiser(),
        **inputs,
        num_steps=4,
        seed=9,
        validity_mode=validity_mode,
    )

    torch.testing.assert_close(
        result.agents[inputs["agent_inpaint_mask"]],
        inputs["agent_context"][inputs["agent_inpaint_mask"]],
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        result.lights[inputs["light_inpaint_mask"]],
        inputs["light_context"][inputs["light_inpaint_mask"]],
        atol=1e-6,
        rtol=0,
    )


@pytest.mark.parametrize("validity_mode", ["paper", "signed_stable"])
def test_sampling_seed_is_repeatable_and_changes_uncontrolled_values(
    validity_mode: str,
):
    inputs = _inputs()
    model = ZeroVelocityDenoiser()

    first = sample_scene(
        model, **inputs, num_steps=4, seed=9, validity_mode=validity_mode
    )
    second = sample_scene(
        model, **inputs, num_steps=4, seed=9, validity_mode=validity_mode
    )
    third = sample_scene(
        model, **inputs, num_steps=4, seed=10, validity_mode=validity_mode
    )

    torch.testing.assert_close(first.agents, second.agents)
    assert not torch.equal(first.agents, third.agents)


def test_sampler_outputs_are_finite_for_sixteen_steps():
    result = sample_scene(ZeroVelocityDenoiser(), **_inputs(), num_steps=16, seed=3)

    assert torch.isfinite(result.agents).all()
    assert torch.isfinite(result.lights).all()


def test_sampler_trace_records_unknown_validity_at_every_denoising_step():
    result = sample_scene(
        ZeroVelocityDenoiser(),
        **_inputs(),
        num_steps=4,
        seed=3,
        validity_mode="paper",
        record_trace=True,
    )

    assert [row["step"] for row in result.trace] == [0, 1, 2, 3]
    assert result.trace[0]["diffusion_time"] == pytest.approx(1.0)
    assert result.trace[-1]["next_diffusion_time"] == pytest.approx(0.0)
    for row in result.trace:
        assert 0.0 <= row["agent_unknown_valid_probability_mean"] <= 1.0
        assert 0.0 <= row["agent_unknown_valid_probability_above_half"] <= 1.0
        assert 0.0 <= row["light_unknown_valid_probability_mean"] <= 1.0


def test_sampler_forwards_optional_roadgraph_topology():
    inputs = _inputs()
    inputs.update(
        {
            "roadgraph_point_lane_index": torch.tensor(
                [
                    [0, 0, 1, 1, -1, -1, -1, -1],
                    [0, 1, 1, -1, -1, -1, -1, -1],
                ]
            ),
            "roadgraph_lane_padding_mask": torch.tensor(
                [[False, False], [False, False]]
            ),
            "roadgraph_successor_index": torch.tensor(
                [[[0], [1]], [[0], [1]]]
            ),
            "roadgraph_successor_padding_mask": torch.zeros(
                2, 1, dtype=torch.bool
            ),
        }
    )
    model = TopologyCheckingDenoiser()

    sample_scene(model, **inputs, num_steps=1, seed=3)

    assert model.saw_topology is True


def test_speed_projection_uses_nearest_road_limit_and_propagates_forward():
    agents = torch.zeros(1, 1, 3, 12)
    agents[..., -1] = 1.0
    agents[0, 0, :, 0] = torch.tensor([0.0, 1.0, 2.0])
    inpaint = torch.zeros_like(agents, dtype=torch.bool)
    inpaint[:, :, 0] = True
    roadgraph = torch.zeros(1, 1, 8)
    roadgraph[..., 4] = 0.25  # 10 m/s after reversing the /40 normalization.

    projected = project_sparse_agent_speed(
        agents,
        agent_inpaint_mask=inpaint,
        roadgraph=roadgraph,
        roadgraph_padding_mask=torch.zeros(1, 1, dtype=torch.bool),
        timestep_seconds=0.1,
        fallback_max_speed_mps=40.0,
        speed_limit_margin=1.0,
    )

    step = 10.0 * 0.1 / POSITION_SCALE
    torch.testing.assert_close(
        projected[0, 0, :, 0], torch.tensor([0.0, step, 2.0 * step])
    )


def test_speed_projection_does_not_move_inpainted_or_newly_valid_steps():
    agents = torch.zeros(1, 1, 3, 12)
    agents[0, 0, :, 0] = torch.tensor([0.0, 0.8, 1.0])
    agents[0, 0, :, -1] = torch.tensor([0.0, 1.0, 1.0])
    inpaint = torch.zeros_like(agents, dtype=torch.bool)
    inpaint[:, :, 1] = True
    roadgraph = torch.zeros(1, 1, 8)
    roadgraph[..., 4] = 0.25

    projected = project_sparse_agent_speed(
        agents,
        agent_inpaint_mask=inpaint,
        roadgraph=roadgraph,
        roadgraph_padding_mask=torch.zeros(1, 1, dtype=torch.bool),
        timestep_seconds=0.1,
        fallback_max_speed_mps=40.0,
        speed_limit_margin=1.0,
    )

    torch.testing.assert_close(projected[0, 0, 1, 0], torch.tensor(0.8))
    torch.testing.assert_close(
        projected[0, 0, 2, 0], torch.tensor(0.8 + 10.0 * 0.1 / POSITION_SCALE)
    )


def test_speed_projection_interprets_signed_validity_at_zero_threshold():
    agents = torch.zeros(1, 1, 3, 12)
    agents[0, 0, :, 0] = torch.tensor([0.0, 1.0, 2.0])
    agents[0, 0, :, -1] = 0.25
    roadgraph = torch.zeros(1, 1, 8)
    roadgraph[..., 4] = 0.25

    projected = project_sparse_agent_speed(
        agents,
        agent_inpaint_mask=torch.zeros_like(agents, dtype=torch.bool),
        roadgraph=roadgraph,
        roadgraph_padding_mask=torch.zeros(1, 1, dtype=torch.bool),
        timestep_seconds=0.1,
        fallback_max_speed_mps=40.0,
        speed_limit_margin=1.0,
        validity_mode="signed_stable",
    )

    step = 10.0 * 0.1 / POSITION_SCALE
    torch.testing.assert_close(
        projected[0, 0, :, 0], torch.tensor([0.0, step, 2.0 * step])
    )


def test_kinematic_projection_limits_acceleration_and_jerk_after_speed_projection():
    agents = torch.zeros(1, 1, 8, 12)
    agents[..., -1] = 1.0
    step = 30.0 * 0.1 / POSITION_SCALE
    agents[0, 0, :, 0] = torch.tensor(
        [0.0, step, 0.0, step, 0.0, step, 0.0, step]
    )
    roadgraph = torch.zeros(1, 1, 8)
    roadgraph[..., 4] = 1.0

    projected = sampler_module.project_sparse_agent_kinematics(
        agents,
        agent_inpaint_mask=torch.zeros_like(agents, dtype=torch.bool),
        roadgraph=roadgraph,
        roadgraph_padding_mask=torch.zeros(1, 1, dtype=torch.bool),
        timestep_seconds=0.1,
        fallback_max_speed_mps=40.0,
        speed_limit_margin=1.0,
        max_acceleration_mps2=15.0,
        max_jerk_mps3=100.0,
        validity_mode="paper",
    )

    positions = projected[..., :2] * POSITION_SCALE
    velocities = torch.diff(positions, dim=2) * 10.0
    accelerations = torch.diff(velocities, dim=2) * 10.0
    jerks = torch.diff(accelerations, dim=2) * 10.0
    torch.testing.assert_close(
        torch.linalg.vector_norm(accelerations, dim=-1).max(),
        torch.tensor(15.0),
        atol=1e-4,
        rtol=0,
    )
    assert torch.linalg.vector_norm(jerks, dim=-1).max() <= 100.0 + 1e-4


def test_kinematic_projection_keeps_speed_as_final_hard_constraint():
    agents = torch.zeros(1, 1, 3, 12)
    agents[..., -1] = 1.0
    fast_history_step = 100.0 * 0.1 / POSITION_SCALE
    agents[0, 0, :, 0] = torch.tensor(
        [0.0, fast_history_step, fast_history_step + 1.0]
    )
    inpaint = torch.zeros_like(agents, dtype=torch.bool)
    inpaint[:, :, :2] = True
    roadgraph = torch.zeros(1, 1, 8)
    roadgraph[..., 4] = 1.0

    projected = sampler_module.project_sparse_agent_kinematics(
        agents,
        agent_inpaint_mask=inpaint,
        roadgraph=roadgraph,
        roadgraph_padding_mask=torch.zeros(1, 1, dtype=torch.bool),
        timestep_seconds=0.1,
        fallback_max_speed_mps=40.0,
        speed_limit_margin=1.0,
        max_acceleration_mps2=15.0,
        validity_mode="paper",
    )

    final_step = projected[0, 0, 2, 0] - projected[0, 0, 1, 0]
    assert final_step <= 40.0 * 0.1 / POSITION_SCALE + 1e-6
