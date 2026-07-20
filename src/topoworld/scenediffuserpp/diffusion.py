from __future__ import annotations

import math

import torch


def cosine_alpha_sigma(time: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not torch.is_floating_point(time):
        raise TypeError("diffusion time must be a floating-point tensor")
    if not torch.isfinite(time).all() or (time < 0.0).any() or (time > 1.0).any():
        raise ValueError("diffusion time must be finite and within [0, 1]")
    angle = time * (math.pi / 2.0)
    return torch.cos(angle), torch.sin(angle)


def forward_noise(clean: torch.Tensor, noise: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    _check_pair(clean, noise)
    alpha, sigma = _broadcast_schedule(time, clean)
    return alpha * clean + sigma * noise


def velocity_target(clean: torch.Tensor, noise: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    _check_pair(clean, noise)
    alpha, sigma = _broadcast_schedule(time, clean)
    return alpha * noise - sigma * clean


def recover_clean(noisy: torch.Tensor, velocity: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    _check_pair(noisy, velocity)
    alpha, sigma = _broadcast_schedule(time, noisy)
    return alpha * noisy - sigma * velocity


def recover_noise(noisy: torch.Tensor, velocity: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    _check_pair(noisy, velocity)
    alpha, sigma = _broadcast_schedule(time, noisy)
    return sigma * noisy + alpha * velocity


def sparse_v_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    validity: torch.Tensor,
    inpaint_mask: torch.Tensor | None,
    *,
    validity_transition_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int | float]]:
    _check_pair(prediction, target)
    if prediction.ndim < 2 or prediction.shape[-1] < 2:
        raise ValueError("sparse tensors need at least one value and one validity channel")
    if validity.shape != prediction.shape[:-1]:
        raise ValueError("validity shape must equal prediction shape without channels")
    if inpaint_mask is not None:
        if inpaint_mask.shape != prediction.shape or inpaint_mask.dtype != torch.bool:
            raise ValueError("inpaint_mask must be bool with prediction shape")
    if (
        not math.isfinite(float(validity_transition_weight))
        or float(validity_transition_weight) < 1.0
    ):
        raise ValueError("validity_transition_weight must be finite and at least 1.0")

    squared_error = (prediction - target).square()
    value_mask = (validity > 0.0).unsqueeze(-1).expand_as(prediction[..., :-1])
    validity_mask = torch.ones_like(validity, dtype=torch.bool)
    value_count = int(value_mask.sum().item())
    validity_count = int(validity_mask.sum().item())
    value_sum = squared_error[..., :-1][value_mask].sum()
    zero = prediction.sum() * 0.0
    transition_mask = _validity_transition_mask(validity)
    transition_count = int(transition_mask.sum().item())
    validity_squared_error = squared_error[..., -1]
    validity_weights = torch.ones_like(validity_squared_error)
    if float(validity_transition_weight) != 1.0 and transition_count:
        validity_weights = torch.where(
            transition_mask,
            validity_weights.new_full((), float(validity_transition_weight)),
            validity_weights,
        )
    validity_sum = (validity_squared_error * validity_weights)[validity_mask].sum()
    validity_weight_sum = validity_weights[validity_mask].sum()
    count = value_sum.new_tensor(float(value_count)) + validity_weight_sum
    total = (value_sum + validity_sum) / count.clamp_min(1.0)
    parts: dict[str, torch.Tensor | int | float] = {
        "value_loss": value_sum / value_count if value_count else zero,
        "validity_loss": (
            validity_sum / validity_weight_sum.clamp_min(1.0)
            if validity_count
            else zero
        ),
        "validity_transition_loss": (
            validity_squared_error[transition_mask].mean()
            if transition_count
            else zero
        ),
        "value_count": value_count,
        "validity_count": validity_count,
        "validity_transition_count": transition_count,
        "validity_weight_sum": float(validity_weight_sum.detach().cpu()),
    }
    return total, parts


def _validity_transition_mask(validity: torch.Tensor) -> torch.Tensor:
    transitions = torch.zeros_like(validity, dtype=torch.bool)
    if validity.shape[-1] <= 1:
        return transitions
    is_valid = validity > 0.0
    transitions[..., 1:] = is_valid[..., 1:] != is_valid[..., :-1]
    return transitions


def _broadcast_schedule(
    time: torch.Tensor, reference: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if time.ndim != 1:
        raise ValueError("diffusion time must have shape [batch]")
    if reference.ndim < 1 or time.shape[0] != reference.shape[0]:
        raise ValueError("diffusion time batch must match tensor batch")
    if time.device != reference.device:
        raise ValueError("diffusion time and values must use the same device")
    if time.dtype != reference.dtype:
        raise ValueError("diffusion time and values must use the same dtype")
    alpha, sigma = cosine_alpha_sigma(time)
    shape = (reference.shape[0],) + (1,) * (reference.ndim - 1)
    return alpha.reshape(shape), sigma.reshape(shape)


def _check_pair(first: torch.Tensor, second: torch.Tensor) -> None:
    if first.shape != second.shape:
        raise ValueError("paired diffusion tensors must have identical shapes")
    if first.dtype != second.dtype or first.device != second.device:
        raise ValueError("paired diffusion tensors must share dtype and device")
    if not torch.is_floating_point(first):
        raise TypeError("diffusion values must be floating-point tensors")
