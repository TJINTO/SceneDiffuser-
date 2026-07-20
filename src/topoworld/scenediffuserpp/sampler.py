from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from topoworld.scenediffuserpp.diffusion import forward_noise
from topoworld.scenediffuserpp.diffusion import recover_clean
from topoworld.scenediffuserpp.normalization import POSITION_SCALE


ROADGRAPH_SPEED_SCALE_MPS = 40.0
VALIDITY_MODES = frozenset({"paper", "signed_stable"})
SAMPLERS = frozenset({"paper_renoise"})
SAMPLING_TIME_GRIDS = frozenset({"linear"})


@dataclass(frozen=True)
class SamplingResult:
    agents: torch.Tensor
    lights: torch.Tensor
    trace: tuple[dict[str, float | int | None], ...] = ()


def soft_clip_sparse(
    values: torch.Tensor, *, validity_mode: str = "paper"
) -> torch.Tensor:
    if values.ndim < 1 or values.shape[-1] < 2:
        raise ValueError("sparse values need data channels plus final validity")
    _validate_validity_mode(validity_mode)
    raw_validity = values[..., -1].clamp(-1.0, 1.0)
    probability = raw_validity * 0.5 + 0.5
    return torch.cat(
        (
            values[..., :-1] * probability.unsqueeze(-1),
            raw_validity.unsqueeze(-1),
        ),
        dim=-1,
    )


def sampling_time_grid(
    num_steps: int,
    *,
    time_grid: str = "linear",
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    _validate_time_grid(time_grid)
    return torch.linspace(
        1.0,
        0.0,
        num_steps + 1,
        device=device,
        dtype=dtype,
    )


def paper_denoise(
    noisy: torch.Tensor,
    velocity: torch.Tensor,
    time: torch.Tensor,
    *,
    validity_mode: str = "paper",
) -> torch.Tensor:
    return soft_clip_sparse(
        recover_clean(noisy, velocity, time),
        validity_mode=validity_mode,
    )


def paper_renoise(
    clean: torch.Tensor,
    noise: torch.Tensor,
    time: torch.Tensor,
) -> torch.Tensor:
    return forward_noise(clean, noise, time)


def sample_scene(
    model: nn.Module,
    *,
    agent_context: torch.Tensor,
    light_context: torch.Tensor,
    agent_inpaint_mask: torch.Tensor,
    light_inpaint_mask: torch.Tensor,
    roadgraph: torch.Tensor,
    roadgraph_padding_mask: torch.Tensor,
    num_steps: int,
    seed: int,
    max_speed_mps: float | None = None,
    max_acceleration_mps2: float | None = None,
    max_jerk_mps3: float | None = None,
    timestep_seconds: float = 0.1,
    speed_limit_margin: float = 1.0,
    validity_mode: str = "paper",
    sampler: str = "paper_renoise",
    time_grid: str = "linear",
    record_trace: bool = False,
    roadgraph_point_lane_index: torch.Tensor | None = None,
    roadgraph_lane_padding_mask: torch.Tensor | None = None,
    roadgraph_successor_index: torch.Tensor | None = None,
    roadgraph_successor_padding_mask: torch.Tensor | None = None,
) -> SamplingResult:
    _validate_validity_mode(validity_mode)
    _validate_sampler(sampler)
    _validate_time_grid(time_grid)
    _validate_inputs(
        agent_context,
        light_context,
        agent_inpaint_mask,
        light_inpaint_mask,
        roadgraph,
        roadgraph_padding_mask,
        num_steps,
    )
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
    device = agent_context.device
    dtype = agent_context.dtype
    generator = torch.Generator(device=device).manual_seed(int(seed))
    agent_z = _randn_like(agent_context, generator)
    light_z = _randn_like(light_context, generator)
    schedule = sampling_time_grid(
        num_steps,
        time_grid=time_grid,
        device=device,
        dtype=dtype,
    )
    batch = agent_context.shape[0]
    trace: list[dict[str, float | int | None]] = []

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for step in range(num_steps):
                current_time = schedule[step].expand(batch)
                next_time = schedule[step + 1].expand(batch)
                output = model(
                    agent_z=agent_z,
                    light_z=light_z,
                    agent_context=agent_context,
                    light_context=light_context,
                    agent_inpaint_mask=agent_inpaint_mask,
                    light_inpaint_mask=light_inpaint_mask,
                    roadgraph=roadgraph,
                    roadgraph_padding_mask=roadgraph_padding_mask,
                    diffusion_time=current_time,
                    **topology_inputs,
                )
                agent_clean = paper_denoise(
                    agent_z,
                    output.agent_v,
                    current_time,
                    validity_mode=validity_mode,
                )
                light_clean = paper_denoise(
                    light_z,
                    output.light_v,
                    current_time,
                    validity_mode=validity_mode,
                )
                agent_clean = torch.where(
                    agent_inpaint_mask, agent_context, agent_clean
                )
                light_clean = torch.where(
                    light_inpaint_mask, light_context, light_clean
                )
                if (
                    max_speed_mps is not None
                    or max_acceleration_mps2 is not None
                    or max_jerk_mps3 is not None
                ):
                    agent_clean = project_sparse_agent_kinematics(
                        agent_clean,
                        agent_inpaint_mask=agent_inpaint_mask,
                        roadgraph=roadgraph,
                        roadgraph_padding_mask=roadgraph_padding_mask,
                        timestep_seconds=timestep_seconds,
                        fallback_max_speed_mps=max_speed_mps,
                        speed_limit_margin=speed_limit_margin,
                        max_acceleration_mps2=max_acceleration_mps2,
                        max_jerk_mps3=max_jerk_mps3,
                        validity_mode=validity_mode,
                    )
                if record_trace:
                    trace.append(
                        _sampling_trace_row(
                            step,
                            current_time,
                            next_time,
                            agent_clean,
                            light_clean,
                            agent_inpaint_mask,
                            light_inpaint_mask,
                            validity_mode,
                        )
                    )
                if step + 1 == num_steps:
                    agent_z = agent_clean
                    light_z = light_clean
                else:
                    agent_z = paper_renoise(
                        agent_clean, _randn_like(agent_clean, generator), next_time
                    )
                    light_z = paper_renoise(
                        light_clean, _randn_like(light_clean, generator), next_time
                    )
                if not torch.isfinite(agent_z).all() or not torch.isfinite(light_z).all():
                    raise FloatingPointError(f"nonfinite sample at denoising step {step}")
    finally:
        model.train(was_training)
    return SamplingResult(
        agents=_finalize_validity(agent_z, agent_inpaint_mask, validity_mode),
        lights=_finalize_validity(light_z, light_inpaint_mask, validity_mode),
        trace=tuple(trace),
    )


def project_sparse_agent_speed(
    agents: torch.Tensor,
    *,
    agent_inpaint_mask: torch.Tensor,
    roadgraph: torch.Tensor,
    roadgraph_padding_mask: torch.Tensor,
    timestep_seconds: float,
    fallback_max_speed_mps: float,
    speed_limit_margin: float = 1.0,
    validity_mode: str = "paper",
) -> torch.Tensor:
    _validate_validity_mode(validity_mode)
    if agents.ndim != 4 or agents.shape[-1] < 3:
        raise ValueError("agents must have shape [batch, entities, time, channels]")
    if agent_inpaint_mask.shape != agents.shape or agent_inpaint_mask.dtype != torch.bool:
        raise ValueError("agent_inpaint_mask must be bool and match agents")
    if roadgraph.ndim != 3 or roadgraph.shape[0] != agents.shape[0]:
        raise ValueError("roadgraph must have shape [batch, points, channels]")
    if roadgraph.shape[-1] < 5:
        raise ValueError("roadgraph must contain x, y, and speed-limit channels")
    if roadgraph_padding_mask.shape != roadgraph.shape[:2]:
        raise ValueError("roadgraph padding mask shape mismatch")
    if timestep_seconds <= 0.0 or fallback_max_speed_mps <= 0.0:
        raise ValueError("timestep and fallback speed must be positive")
    if speed_limit_margin <= 0.0:
        raise ValueError("speed_limit_margin must be positive")

    result = agents.clone()
    threshold = 0.0
    validity = result[..., -1] >= threshold
    for batch_index in range(result.shape[0]):
        map_valid = ~roadgraph_padding_mask[batch_index]
        map_xy = roadgraph[batch_index, map_valid, :2]
        map_speed = (
            roadgraph[batch_index, map_valid, 4]
            * ROADGRAPH_SPEED_SCALE_MPS
            * speed_limit_margin
        )
        for time_index in range(1, result.shape[2]):
            locked = agent_inpaint_mask[batch_index, :, time_index, :2].any(dim=-1)
            active = (
                validity[batch_index, :, time_index - 1]
                & validity[batch_index, :, time_index]
                & ~locked
            )
            if not active.any():
                continue
            entity_indices = torch.nonzero(active, as_tuple=False).flatten()
            previous = result[batch_index, entity_indices, time_index - 1, :2]
            current = result[batch_index, entity_indices, time_index, :2]
            speed_limits = torch.full(
                (entity_indices.numel(),),
                fallback_max_speed_mps,
                dtype=result.dtype,
                device=result.device,
            )
            if map_xy.numel():
                nearest = torch.cdist(previous, map_xy).argmin(dim=1)
                local_limits = map_speed[nearest]
                usable = torch.isfinite(local_limits) & (local_limits > 0.0)
                speed_limits = torch.where(
                    usable,
                    torch.minimum(local_limits, speed_limits),
                    speed_limits,
                )
            maximum_step = speed_limits * timestep_seconds / POSITION_SCALE
            displacement = current - previous
            distance = torch.linalg.vector_norm(displacement, dim=-1)
            scale = torch.minimum(
                torch.ones_like(distance),
                maximum_step / distance.clamp_min(torch.finfo(distance.dtype).eps),
            )
            result[batch_index, entity_indices, time_index, :2] = (
                previous + displacement * scale.unsqueeze(-1)
            )
    return result


def project_sparse_agent_kinematics(
    agents: torch.Tensor,
    *,
    agent_inpaint_mask: torch.Tensor,
    roadgraph: torch.Tensor,
    roadgraph_padding_mask: torch.Tensor,
    timestep_seconds: float,
    fallback_max_speed_mps: float | None = None,
    speed_limit_margin: float = 1.0,
    max_acceleration_mps2: float | None = None,
    max_jerk_mps3: float | None = None,
    validity_mode: str = "paper",
) -> torch.Tensor:
    if fallback_max_speed_mps is None and max_acceleration_mps2 is None and max_jerk_mps3 is None:
        return agents
    if max_acceleration_mps2 is None and max_jerk_mps3 is None:
        if fallback_max_speed_mps is None:
            return agents
        return project_sparse_agent_speed(
            agents,
            agent_inpaint_mask=agent_inpaint_mask,
            roadgraph=roadgraph,
            roadgraph_padding_mask=roadgraph_padding_mask,
            timestep_seconds=timestep_seconds,
            fallback_max_speed_mps=fallback_max_speed_mps,
            speed_limit_margin=speed_limit_margin,
            validity_mode=validity_mode,
        )
    return _project_sparse_agent_dynamics(
        agents,
        agent_inpaint_mask=agent_inpaint_mask,
        roadgraph=roadgraph,
        roadgraph_padding_mask=roadgraph_padding_mask,
        timestep_seconds=timestep_seconds,
        fallback_max_speed_mps=fallback_max_speed_mps,
        speed_limit_margin=speed_limit_margin,
        max_acceleration_mps2=max_acceleration_mps2,
        max_jerk_mps3=max_jerk_mps3,
        validity_mode=validity_mode,
    )


def _project_sparse_agent_dynamics(
    agents: torch.Tensor,
    *,
    agent_inpaint_mask: torch.Tensor,
    roadgraph: torch.Tensor,
    roadgraph_padding_mask: torch.Tensor,
    timestep_seconds: float,
    fallback_max_speed_mps: float | None,
    speed_limit_margin: float,
    max_acceleration_mps2: float | None,
    max_jerk_mps3: float | None,
    validity_mode: str,
) -> torch.Tensor:
    _validate_validity_mode(validity_mode)
    if agents.ndim != 4 or agents.shape[-1] < 3:
        raise ValueError("agents must have shape [batch, entities, time, channels]")
    if agent_inpaint_mask.shape != agents.shape or agent_inpaint_mask.dtype != torch.bool:
        raise ValueError("agent_inpaint_mask must be bool and match agents")
    if roadgraph.ndim != 3 or roadgraph.shape[0] != agents.shape[0]:
        raise ValueError("roadgraph must have shape [batch, points, channels]")
    if roadgraph.shape[-1] < 5:
        raise ValueError("roadgraph must contain x, y, and speed-limit channels")
    if roadgraph_padding_mask.shape != roadgraph.shape[:2]:
        raise ValueError("roadgraph padding mask shape mismatch")
    if timestep_seconds <= 0.0:
        raise ValueError("timestep must be positive")
    if fallback_max_speed_mps is not None and fallback_max_speed_mps <= 0.0:
        raise ValueError("fallback speed must be positive")
    if speed_limit_margin <= 0.0:
        raise ValueError("speed_limit_margin must be positive")
    if max_acceleration_mps2 is not None and max_acceleration_mps2 <= 0.0:
        raise ValueError("max_acceleration_mps2 must be positive")
    if max_jerk_mps3 is not None and max_jerk_mps3 <= 0.0:
        raise ValueError("max_jerk_mps3 must be positive")

    result = agents.clone()
    threshold = 0.0
    validity = result[..., -1] >= threshold
    acceleration_velocity_limit = (
        None
        if max_acceleration_mps2 is None
        else max_acceleration_mps2 * timestep_seconds / POSITION_SCALE
    )
    jerk_velocity_limit = (
        None
        if max_jerk_mps3 is None
        else max_jerk_mps3 * timestep_seconds * timestep_seconds / POSITION_SCALE
    )
    for batch_index in range(result.shape[0]):
        map_valid = ~roadgraph_padding_mask[batch_index]
        map_xy = roadgraph[batch_index, map_valid, :2]
        map_speed = (
            roadgraph[batch_index, map_valid, 4]
            * ROADGRAPH_SPEED_SCALE_MPS
            * speed_limit_margin
        )
        for time_index in range(1, result.shape[2]):
            locked = agent_inpaint_mask[batch_index, :, time_index, :2].any(dim=-1)
            active = (
                validity[batch_index, :, time_index - 1]
                & validity[batch_index, :, time_index]
                & ~locked
            )
            if not active.any():
                continue
            entity_indices = torch.nonzero(active, as_tuple=False).flatten()
            previous = result[batch_index, entity_indices, time_index - 1, :2]
            current = result[batch_index, entity_indices, time_index, :2]
            proposed_velocity = (current - previous) / timestep_seconds

            speed_limits = None
            if fallback_max_speed_mps is not None:
                speed_limits = torch.full(
                    (entity_indices.numel(),),
                    fallback_max_speed_mps / POSITION_SCALE,
                    dtype=result.dtype,
                    device=result.device,
                )
                if map_xy.numel():
                    nearest = torch.cdist(previous, map_xy).argmin(dim=1)
                    local_limits = map_speed[nearest] / POSITION_SCALE
                    usable = torch.isfinite(local_limits) & (local_limits > 0.0)
                    speed_limits = torch.where(
                        usable,
                        torch.minimum(local_limits, speed_limits),
                        speed_limits,
                    )

            acceleration_entities = None
            previous_velocity = None
            if acceleration_velocity_limit is not None and time_index >= 2:
                acceleration_active = validity[
                    batch_index, entity_indices, time_index - 2
                ]
                if acceleration_active.any():
                    acceleration_entities = torch.nonzero(
                        acceleration_active, as_tuple=False
                    ).flatten()
                    selected = entity_indices[acceleration_entities]
                    previous_previous = result[
                        batch_index, selected, time_index - 2, :2
                    ]
                    previous_selected = result[
                        batch_index, selected, time_index - 1, :2
                    ]
                    previous_velocity = (
                        previous_selected - previous_previous
                    ) / timestep_seconds

            jerk_entities = None
            jerk_center_velocity = None
            if jerk_velocity_limit is not None and time_index >= 3:
                jerk_active = (
                    validity[batch_index, entity_indices, time_index - 3]
                    & validity[batch_index, entity_indices, time_index - 2]
                )
                if jerk_active.any():
                    jerk_entities = torch.nonzero(jerk_active, as_tuple=False).flatten()
                    selected = entity_indices[jerk_entities]
                    before_previous = result[
                        batch_index, selected, time_index - 3, :2
                    ]
                    previous_previous_selected = result[
                        batch_index, selected, time_index - 2, :2
                    ]
                    previous_selected = result[
                        batch_index, selected, time_index - 1, :2
                    ]
                    previous_previous_velocity = (
                        previous_previous_selected - before_previous
                    ) / timestep_seconds
                    previous_velocity_selected = (
                        previous_selected - previous_previous_selected
                    ) / timestep_seconds
                    previous_acceleration = (
                        previous_velocity_selected - previous_previous_velocity
                    ) / timestep_seconds
                    jerk_center_velocity = (
                        previous_velocity_selected
                        + previous_acceleration * timestep_seconds
                    )

            for _ in range(12):
                if speed_limits is not None:
                    proposed_velocity = _limit_vector_norm(
                        proposed_velocity,
                        speed_limits,
                    )
                if acceleration_entities is not None and previous_velocity is not None:
                    proposed_velocity[acceleration_entities] = _limit_vector_delta(
                        proposed_velocity[acceleration_entities],
                        previous_velocity,
                        acceleration_velocity_limit,
                    )
                if jerk_entities is not None and jerk_center_velocity is not None:
                    proposed_velocity[jerk_entities] = _limit_vector_delta(
                        proposed_velocity[jerk_entities],
                        jerk_center_velocity,
                        jerk_velocity_limit,
                    )
            if speed_limits is not None:
                proposed_velocity = _limit_vector_norm(proposed_velocity, speed_limits)
            result[batch_index, entity_indices, time_index, :2] = (
                previous + proposed_velocity * timestep_seconds
            )
    return result


def _limit_vector_delta(
    proposed: torch.Tensor,
    reference: torch.Tensor,
    maximum_delta: float,
) -> torch.Tensor:
    delta = proposed - reference
    magnitude = torch.linalg.vector_norm(delta, dim=-1)
    scale = torch.minimum(
        torch.ones_like(magnitude),
        torch.as_tensor(
            maximum_delta,
            dtype=magnitude.dtype,
            device=magnitude.device,
        )
        / magnitude.clamp_min(torch.finfo(magnitude.dtype).eps),
    )
    return reference + delta * scale.unsqueeze(-1)


def _limit_vector_norm(values: torch.Tensor, maximum_norm: torch.Tensor) -> torch.Tensor:
    magnitude = torch.linalg.vector_norm(values, dim=-1)
    scale = torch.minimum(
        torch.ones_like(magnitude),
        maximum_norm / magnitude.clamp_min(torch.finfo(magnitude.dtype).eps),
    )
    return values * scale.unsqueeze(-1)


def _randn_like(values: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(
        values.shape,
        dtype=values.dtype,
        device=values.device,
        generator=generator,
    )


def _finalize_validity(
    values: torch.Tensor, inpaint_mask: torch.Tensor, validity_mode: str
) -> torch.Tensor:
    result = values.clone()
    signed = result[..., -1].clamp(-1.0, 1.0)
    result[..., -1] = torch.where(inpaint_mask[..., -1], result[..., -1], signed)
    return result


def _sampling_trace_row(
    step: int,
    current_time: torch.Tensor,
    next_time: torch.Tensor,
    agents: torch.Tensor,
    lights: torch.Tensor,
    agent_inpaint_mask: torch.Tensor,
    light_inpaint_mask: torch.Tensor,
    validity_mode: str,
) -> dict[str, float | int | None]:
    row: dict[str, float | int | None] = {
        "step": int(step),
        "diffusion_time": float(current_time[0].detach().cpu()),
        "next_diffusion_time": float(next_time[0].detach().cpu()),
    }
    for prefix, values, mask in (
        ("agent", agents, agent_inpaint_mask),
        ("light", lights, light_inpaint_mask),
    ):
        validity = values[..., -1]
        probability = validity.clamp(-1.0, 1.0) * 0.5 + 0.5
        unknown = ~mask[..., -1]
        selected = probability[unknown].float()
        row[f"{prefix}_unknown_count"] = int(selected.numel())
        if not selected.numel():
            for suffix in (
                "mean",
                "p10",
                "p50",
                "p90",
                "above_half",
            ):
                row[f"{prefix}_unknown_valid_probability_{suffix}"] = None
            continue
        row[f"{prefix}_unknown_valid_probability_mean"] = float(
            selected.mean().detach().cpu()
        )
        for suffix, quantile in (("p10", 0.1), ("p50", 0.5), ("p90", 0.9)):
            row[f"{prefix}_unknown_valid_probability_{suffix}"] = float(
                torch.quantile(selected, quantile).detach().cpu()
            )
        row[f"{prefix}_unknown_valid_probability_above_half"] = float(
            (selected >= 0.5).float().mean().detach().cpu()
        )
    return row


def _validate_validity_mode(validity_mode: str) -> None:
    if validity_mode not in VALIDITY_MODES:
        choices = ", ".join(sorted(VALIDITY_MODES))
        raise ValueError(
            f"unsupported validity_mode {validity_mode!r}; expected one of: {choices}"
        )


def _validate_sampler(sampler: str) -> None:
    if sampler not in SAMPLERS:
        choices = ", ".join(sorted(SAMPLERS))
        raise ValueError(f"unsupported sampler {sampler!r}; expected one of: {choices}")


def _validate_time_grid(time_grid: str) -> None:
    if time_grid not in SAMPLING_TIME_GRIDS:
        choices = ", ".join(sorted(SAMPLING_TIME_GRIDS))
        raise ValueError(
            f"unsupported time_grid {time_grid!r}; expected one of: {choices}"
        )


def _validate_inputs(
    agents: torch.Tensor,
    lights: torch.Tensor,
    agent_mask: torch.Tensor,
    light_mask: torch.Tensor,
    roadgraph: torch.Tensor,
    roadgraph_padding_mask: torch.Tensor,
    num_steps: int,
) -> None:
    if agents.ndim != 4 or lights.ndim != 4:
        raise ValueError("agent and light context must have shape [B,E,T,C]")
    if agents.shape[0] != lights.shape[0] or agents.shape[2] != lights.shape[2]:
        raise ValueError("agent and light batch/time dimensions must match")
    if agent_mask.shape != agents.shape or agent_mask.dtype != torch.bool:
        raise ValueError("agent_inpaint_mask must be bool and match agent context")
    if light_mask.shape != lights.shape or light_mask.dtype != torch.bool:
        raise ValueError("light_inpaint_mask must be bool and match light context")
    if roadgraph.ndim != 3 or roadgraph.shape[0] != agents.shape[0]:
        raise ValueError("roadgraph must have shape [batch, points, channels]")
    if roadgraph_padding_mask.shape != roadgraph.shape[:2]:
        raise ValueError("roadgraph padding mask shape mismatch")
    if roadgraph_padding_mask.dtype != torch.bool:
        raise ValueError("roadgraph padding mask must be boolean")
    floating = (lights, roadgraph)
    if any(value.device != agents.device or value.dtype != agents.dtype for value in floating):
        raise ValueError("all floating inputs must share device and dtype")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
