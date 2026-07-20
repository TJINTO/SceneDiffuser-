from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

import torch
from torch import nn

from topoworld.scenediffuserpp.diffusion import forward_noise
from topoworld.scenediffuserpp.diffusion import recover_clean
from topoworld.scenediffuserpp.normalization import POSITION_SCALE


def denoising_metrics(
    predicted_agents: torch.Tensor,
    target_agents: torch.Tensor,
    predicted_lights: torch.Tensor,
    target_lights: torch.Tensor,
    *,
    agent_inpaint_mask: torch.Tensor,
    light_inpaint_mask: torch.Tensor,
) -> dict[str, Any]:
    _validate_pair(predicted_agents, target_agents, agent_inpaint_mask, "agents")
    _validate_pair(predicted_lights, target_lights, light_inpaint_mask, "lights")
    if predicted_agents.shape[-1] < 3 or predicted_lights.shape[-1] < 13:
        raise ValueError("scene tensors do not contain the required channels")

    agent_unknown = ~agent_inpaint_mask[..., -1]
    light_unknown = ~light_inpaint_mask[..., -1]
    agent_target_valid = target_agents[..., -1] > 0.0
    light_target_valid = target_lights[..., -1] > 0.0
    agent_report = _binary_metrics(
        predicted_agents[..., -1] > 0.0,
        agent_target_valid,
        agent_unknown,
    )
    light_report = _binary_metrics(
        predicted_lights[..., -1] > 0.0,
        light_target_valid,
        light_unknown,
    )

    xy_unknown = ~agent_inpaint_mask[..., :2].any(dim=-1)
    xy_mask = agent_target_valid & xy_unknown
    if xy_mask.any():
        xy_squared_error = (
            predicted_agents[..., :2] - target_agents[..., :2]
        ).square().sum(dim=-1)
        agent_report["xy_rmse_m"] = float(
            torch.sqrt(xy_squared_error[xy_mask].mean()).item() * POSITION_SCALE
        )
    else:
        agent_report["xy_rmse_m"] = None

    heading_mask = agent_target_valid & ~agent_inpaint_mask[..., 3]
    if heading_mask.any():
        heading_error = (
            (predicted_agents[..., 3] - target_agents[..., 3]) * math.pi
            + math.pi
        ).remainder(2.0 * math.pi) - math.pi
        agent_report["heading_mae_deg"] = float(
            torch.rad2deg(heading_error[heading_mask].abs()).mean().item()
        )
    else:
        agent_report["heading_mae_deg"] = None

    type_unknown = ~agent_inpaint_mask[..., 7:11].any(dim=-1)
    type_mask = agent_target_valid & type_unknown
    if type_mask.any():
        predicted_type = predicted_agents[..., 7:11].argmax(dim=-1)
        target_type = target_agents[..., 7:11].argmax(dim=-1)
        agent_report["type_accuracy"] = float(
            (predicted_type[type_mask] == target_type[type_mask]).float().mean().item()
        )
    else:
        agent_report["type_accuracy"] = None

    state_unknown = ~light_inpaint_mask[..., 3:12].any(dim=-1)
    state_mask = light_target_valid & state_unknown
    if state_mask.any():
        predicted_state = predicted_lights[..., 3:12].argmax(dim=-1)
        target_state = target_lights[..., 3:12].argmax(dim=-1)
        light_report["state_accuracy"] = float(
            (predicted_state[state_mask] == target_state[state_mask])
            .float()
            .mean()
            .item()
        )
    else:
        light_report["state_accuracy"] = None
    return {"agents": agent_report, "lights": light_report}


@torch.no_grad()
def evaluate_denoising_levels(
    model: nn.Module,
    *,
    agent_context: torch.Tensor,
    light_context: torch.Tensor,
    agent_inpaint_mask: torch.Tensor,
    light_inpaint_mask: torch.Tensor,
    roadgraph: torch.Tensor,
    roadgraph_padding_mask: torch.Tensor,
    noise_levels: Sequence[float],
    seeds: Sequence[int],
    roadgraph_point_lane_index: torch.Tensor | None = None,
    roadgraph_lane_padding_mask: torch.Tensor | None = None,
    roadgraph_successor_index: torch.Tensor | None = None,
    roadgraph_successor_padding_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    levels = tuple(float(level) for level in noise_levels)
    seed_values = tuple(int(seed) for seed in seeds)
    if not levels or any(level <= 0.0 or level > 1.0 for level in levels):
        raise ValueError("noise_levels must be nonempty and within (0, 1]")
    if not seed_values:
        raise ValueError("seeds must be nonempty")
    topology = {
        "roadgraph_point_lane_index": roadgraph_point_lane_index,
        "roadgraph_lane_padding_mask": roadgraph_lane_padding_mask,
        "roadgraph_successor_index": roadgraph_successor_index,
        "roadgraph_successor_padding_mask": roadgraph_successor_padding_mask,
    }
    if any(value is None for value in topology.values()) and not all(
        value is None for value in topology.values()
    ):
        raise ValueError("all roadgraph topology tensors must be provided together")
    topology_inputs = {
        name: value for name, value in topology.items() if value is not None
    }

    was_training = model.training
    model.eval()
    reports: dict[str, Any] = {}
    try:
        for level in levels:
            per_seed = []
            time = torch.full(
                (agent_context.shape[0],),
                level,
                dtype=agent_context.dtype,
                device=agent_context.device,
            )
            for seed in seed_values:
                generator = torch.Generator(device=agent_context.device).manual_seed(
                    seed
                )
                agent_noise = _randn_like(agent_context, generator)
                light_noise = _randn_like(light_context, generator)
                agent_z = forward_noise(agent_context, agent_noise, time)
                light_z = forward_noise(light_context, light_noise, time)
                output = model(
                    agent_z=agent_z,
                    light_z=light_z,
                    agent_context=agent_context,
                    light_context=light_context,
                    agent_inpaint_mask=agent_inpaint_mask,
                    light_inpaint_mask=light_inpaint_mask,
                    roadgraph=roadgraph,
                    roadgraph_padding_mask=roadgraph_padding_mask,
                    diffusion_time=time,
                    **topology_inputs,
                )
                per_seed.append(
                    denoising_metrics(
                        recover_clean(agent_z, output.agent_v, time),
                        agent_context,
                        recover_clean(light_z, output.light_v, time),
                        light_context,
                        agent_inpaint_mask=agent_inpaint_mask,
                        light_inpaint_mask=light_inpaint_mask,
                    )
                )
            reports[f"{level:.6f}"] = _mean_reports(per_seed)
    finally:
        model.train(was_training)
    return {"seeds": list(seed_values), "levels": reports}


def _binary_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, Any]:
    predicted_values = predicted[mask]
    target_values = target[mask]
    positives = target_values
    negatives = ~target_values
    positive_count = int(positives.sum().item())
    negative_count = int(negatives.sum().item())
    total = positive_count + negative_count
    accuracy = (
        float((predicted_values == target_values).float().mean().item())
        if total
        else None
    )
    recall = (
        float(predicted_values[positives].float().mean().item())
        if positive_count
        else None
    )
    specificity = (
        float((~predicted_values[negatives]).float().mean().item())
        if negative_count
        else None
    )
    balanced = (
        (recall + specificity) * 0.5
        if recall is not None and specificity is not None
        else None
    )
    return {
        "validity_accuracy": accuracy,
        "validity_recall": recall,
        "validity_specificity": specificity,
        "validity_balanced_accuracy": balanced,
        "predicted_valid_fraction": (
            float(predicted_values.float().mean().item()) if total else None
        ),
        "target_valid_fraction": (
            float(target_values.float().mean().item()) if total else None
        ),
        "positive_count": positive_count,
        "negative_count": negative_count,
    }


def _validate_pair(
    prediction: torch.Tensor,
    target: torch.Tensor,
    inpaint_mask: torch.Tensor,
    name: str,
) -> None:
    if prediction.shape != target.shape or inpaint_mask.shape != target.shape:
        raise ValueError(f"{name} prediction, target, and mask shapes must match")
    if inpaint_mask.dtype != torch.bool:
        raise ValueError(f"{name} inpaint mask must be boolean")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise FloatingPointError(f"{name} denoising tensors must be finite")


def _mean_reports(reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    keys = reports[0].keys()
    result: dict[str, Any] = {}
    for key in keys:
        values = [report[key] for report in reports]
        if isinstance(values[0], dict):
            result[key] = _mean_reports(values)
        elif values[0] is None:
            result[key] = None
        else:
            result[key] = sum(float(value) for value in values) / len(values)
    return result


def _randn_like(
    values: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    return torch.randn(
        values.shape,
        dtype=values.dtype,
        device=values.device,
        generator=generator,
    )
