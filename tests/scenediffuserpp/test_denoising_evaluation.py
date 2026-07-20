from types import SimpleNamespace

import pytest
import torch
from torch import nn

from topoworld.scenediffuserpp.denoising_evaluation import denoising_metrics
from topoworld.scenediffuserpp.denoising_evaluation import evaluate_denoising_levels
from topoworld.scenediffuserpp.diffusion import cosine_alpha_sigma
from topoworld.scenediffuserpp.normalization import POSITION_SCALE


class OracleDenoiser(nn.Module):
    def __init__(self, agents: torch.Tensor, lights: torch.Tensor):
        super().__init__()
        self.register_buffer("agents", agents)
        self.register_buffer("lights", lights)

    def forward(self, **inputs):
        alpha, sigma = cosine_alpha_sigma(inputs["diffusion_time"])
        shape = (inputs["agent_z"].shape[0], 1, 1, 1)
        alpha = alpha.reshape(shape)
        sigma = sigma.reshape(shape)
        return SimpleNamespace(
            agent_v=(alpha * inputs["agent_z"] - self.agents) / sigma,
            light_v=(alpha * inputs["light_z"] - self.lights) / sigma,
        )


def _targets() -> tuple[torch.Tensor, torch.Tensor]:
    agents = torch.zeros(1, 2, 2, 12)
    lights = torch.zeros(1, 2, 2, 13)
    agents[..., -1] = torch.tensor([[[1.0, 1.0], [-1.0, -1.0]]])
    lights[..., -1] = torch.tensor([[[1.0, 1.0], [-1.0, -1.0]]])
    lights[:, 0, :, 3:12] = -1.0
    lights[:, 0, :, 4] = 1.0
    return agents, lights


def test_denoising_metrics_exposes_majority_class_validity_collapse():
    agents, lights = _targets()
    predicted_agents = agents.clone()
    predicted_lights = lights.clone()
    predicted_agents[..., -1] = -1.0
    predicted_lights[..., -1] = -1.0
    mask_agents = torch.zeros_like(agents, dtype=torch.bool)
    mask_lights = torch.zeros_like(lights, dtype=torch.bool)

    report = denoising_metrics(
        predicted_agents,
        agents,
        predicted_lights,
        lights,
        agent_inpaint_mask=mask_agents,
        light_inpaint_mask=mask_lights,
    )

    assert report["agents"]["validity_accuracy"] == 0.5
    assert report["agents"]["validity_recall"] == 0.0
    assert report["agents"]["validity_specificity"] == 1.0
    assert report["agents"]["validity_balanced_accuracy"] == 0.5
    assert report["lights"]["validity_balanced_accuracy"] == 0.5


def test_oracle_denoiser_recovers_clean_scene_at_pure_noise_level():
    agents, lights = _targets()
    model = OracleDenoiser(agents, lights)
    agent_mask = torch.zeros_like(agents, dtype=torch.bool)
    light_mask = torch.zeros_like(lights, dtype=torch.bool)

    report = evaluate_denoising_levels(
        model,
        agent_context=agents,
        light_context=lights,
        agent_inpaint_mask=agent_mask,
        light_inpaint_mask=light_mask,
        roadgraph=torch.zeros(1, 2, 8),
        roadgraph_padding_mask=torch.zeros(1, 2, dtype=torch.bool),
        noise_levels=(1.0,),
        seeds=(7, 8),
    )

    level = report["levels"]["1.000000"]
    assert level["agents"]["validity_balanced_accuracy"] == 1.0
    assert level["agents"]["xy_rmse_m"] == pytest.approx(0.0, abs=1e-5)
    assert level["agents"]["heading_mae_deg"] == pytest.approx(0.0, abs=1e-5)
    assert level["agents"]["type_accuracy"] == 1.0
    assert level["lights"]["state_accuracy"] == 1.0
    assert report["seeds"] == [7, 8]


def test_denoising_heading_error_wraps_at_pi_boundary():
    agents, lights = _targets()
    predicted_agents = agents.clone()
    predicted_agents[:, 0, :, 3] = -0.99
    agents[:, 0, :, 3] = 0.99

    report = denoising_metrics(
        predicted_agents,
        agents,
        lights,
        lights,
        agent_inpaint_mask=torch.zeros_like(agents, dtype=torch.bool),
        light_inpaint_mask=torch.zeros_like(lights, dtype=torch.bool),
    )

    assert report["agents"]["heading_mae_deg"] == pytest.approx(3.6, abs=1e-3)
