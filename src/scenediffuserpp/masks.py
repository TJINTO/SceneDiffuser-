from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MultiTensorTrainingMasks:
    agent_mask: torch.Tensor
    light_mask: torch.Tensor
    task_is_scene_generation: torch.Tensor


def behavior_prediction_mask(
    *,
    entities: int,
    timesteps: int,
    channels: int,
    history: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    _check_dimensions(entities, timesteps, channels)
    if history < 0 or history > timesteps:
        raise ValueError("history must be within [0, timesteps]")
    mask = torch.zeros((entities, timesteps, channels), dtype=torch.bool, device=device)
    mask[:, :history] = True
    return mask


def scene_generation_mask(
    *,
    entities: int,
    timesteps: int,
    channels: int,
    context_entities: int,
    generator: torch.Generator,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    _check_dimensions(entities, timesteps, channels)
    if generator is None:
        raise ValueError("an explicit torch.Generator is required")
    if context_entities < 0 or context_entities > entities:
        raise ValueError("context_entities must be within [0, entities]")
    mask = torch.zeros((entities, timesteps, channels), dtype=torch.bool, device=device)
    selected = torch.randperm(entities, generator=generator, device=device)[:context_entities]
    mask[selected] = True
    return mask


def random_control_mask(
    base_mask: torch.Tensor,
    *,
    keep_probability: float,
    generator: torch.Generator,
) -> torch.Tensor:
    if base_mask.dtype != torch.bool:
        raise TypeError("base_mask must be boolean")
    if generator is None:
        raise ValueError("an explicit torch.Generator is required")
    if keep_probability < 0.0 or keep_probability > 1.0:
        raise ValueError("keep_probability must be within [0, 1]")
    draws = torch.rand(
        base_mask.shape,
        generator=generator,
        device=base_mask.device,
        dtype=torch.float32,
    )
    return base_mask & (draws < keep_probability)


def mixed_multitensor_training_masks(
    *,
    agent_validity: torch.Tensor,
    light_validity: torch.Tensor,
    agent_channels: int,
    light_channels: int,
    history: int,
    scene_generation_probability: float,
    control_feature_probability: float,
    generator: torch.Generator,
) -> MultiTensorTrainingMasks:
    """Sample the BP/SceneGen and factorized control masks from SceneDiffuser."""
    _check_validity_pair(agent_validity, light_validity)
    if agent_channels <= 0 or light_channels <= 0:
        raise ValueError("channel counts must be positive")
    timesteps = agent_validity.shape[2]
    if history < 0 or history > timesteps:
        raise ValueError("history must be within [0, timesteps]")
    if not 0.0 <= scene_generation_probability <= 1.0:
        raise ValueError("scene_generation_probability must be within [0, 1]")
    if not 0.0 <= control_feature_probability <= 1.0:
        raise ValueError("control_feature_probability must be within [0, 1]")
    if generator is None:
        raise ValueError("an explicit torch.Generator is required")

    device = agent_validity.device
    batch = agent_validity.shape[0]
    task_is_scene_generation = torch.rand(
        batch, generator=generator, device=device
    ) < scene_generation_probability
    agent_mask = torch.zeros(
        (*agent_validity.shape, agent_channels), dtype=torch.bool, device=device
    )
    light_mask = torch.zeros(
        (*light_validity.shape, light_channels), dtype=torch.bool, device=device
    )
    for batch_index in range(batch):
        agent_valid_entities = agent_validity[batch_index].any(dim=1)
        light_valid_entities = light_validity[batch_index].any(dim=1)
        if bool(task_is_scene_generation[batch_index]):
            agent_base = _sample_scene_generation_base(
                agent_valid_entities, timesteps, agent_channels, generator
            )
            light_base = _sample_scene_generation_base(
                light_valid_entities, timesteps, light_channels, generator
            )
        else:
            agent_base = _behavior_prediction_base(
                agent_valid_entities, timesteps, agent_channels, history
            )
            light_base = _behavior_prediction_base(
                light_valid_entities, timesteps, light_channels, history
            )
        agent_mask[batch_index] = agent_base & _sample_factorized_control(
            agent_valid_entities,
            timesteps,
            agent_channels,
            control_feature_probability,
            generator,
        )
        light_mask[batch_index] = light_base & _sample_factorized_control(
            light_valid_entities,
            timesteps,
            light_channels,
            control_feature_probability,
            generator,
        )
    return MultiTensorTrainingMasks(
        agent_mask=agent_mask,
        light_mask=light_mask,
        task_is_scene_generation=task_is_scene_generation,
    )


def _behavior_prediction_base(
    valid_entities: torch.Tensor,
    timesteps: int,
    channels: int,
    history: int,
) -> torch.Tensor:
    result = torch.zeros(
        (valid_entities.shape[0], timesteps, channels),
        dtype=torch.bool,
        device=valid_entities.device,
    )
    result[:, :history] = valid_entities[:, None, None]
    return result


def _sample_scene_generation_base(
    valid_entities: torch.Tensor,
    timesteps: int,
    channels: int,
    generator: torch.Generator,
) -> torch.Tensor:
    selected = _sample_entity_subset(valid_entities, generator)
    return selected[:, None, None].expand(-1, timesteps, channels)


def _sample_factorized_control(
    valid_entities: torch.Tensor,
    timesteps: int,
    channels: int,
    feature_probability: float,
    generator: torch.Generator,
) -> torch.Tensor:
    entities = _sample_entity_subset(valid_entities, generator)
    time_count = int(
        torch.randint(
            0,
            timesteps + 1,
            (1,),
            generator=generator,
            device=valid_entities.device,
        ).item()
    )
    times = torch.zeros(timesteps, dtype=torch.bool, device=valid_entities.device)
    if time_count:
        indices = torch.randperm(
            timesteps, generator=generator, device=valid_entities.device
        )[:time_count]
        times[indices] = True
    features = torch.rand(
        channels, generator=generator, device=valid_entities.device
    ) < feature_probability
    return entities[:, None, None] & times[None, :, None] & features[None, None, :]


def _sample_entity_subset(
    valid_entities: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    valid_indices = torch.nonzero(valid_entities, as_tuple=False).flatten()
    result = torch.zeros_like(valid_entities)
    count = int(
        torch.randint(
            0,
            valid_indices.numel() + 1,
            (1,),
            generator=generator,
            device=valid_entities.device,
        ).item()
    )
    if count:
        order = torch.randperm(
            valid_indices.numel(), generator=generator, device=valid_entities.device
        )[:count]
        result[valid_indices[order]] = True
    return result


def _check_validity_pair(
    agent_validity: torch.Tensor, light_validity: torch.Tensor
) -> None:
    if agent_validity.dtype != torch.bool or light_validity.dtype != torch.bool:
        raise TypeError("validity tensors must be boolean")
    if agent_validity.ndim != 3 or light_validity.ndim != 3:
        raise ValueError("validity tensors must have shape [batch, entities, time]")
    if agent_validity.shape[0] != light_validity.shape[0]:
        raise ValueError("agent and light validity batches must match")
    if agent_validity.shape[2] != light_validity.shape[2]:
        raise ValueError("agent and light validity timesteps must match")
    if agent_validity.device != light_validity.device:
        raise ValueError("agent and light validity must share a device")


def _check_dimensions(entities: int, timesteps: int, channels: int) -> None:
    if entities <= 0 or timesteps <= 0 or channels <= 0:
        raise ValueError("entities, timesteps, and channels must be positive")
