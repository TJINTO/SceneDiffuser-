from __future__ import annotations

import math
from typing import Any

import torch

from topoworld.scenediffuserpp.normalization import POSITION_SCALE


DEFAULT_T1_THRESHOLDS = {
    "agent_xy_rmse_m": 2.0,
    "agent_heading_mae_deg": 5.0,
    "agent_type_accuracy": 0.99,
    "agent_validity_balanced_accuracy": 0.99,
    "light_state_accuracy": 0.99,
    "light_validity_balanced_accuracy": 0.99,
}


def behavior_prediction_baselines(
    target_agents: torch.Tensor,
    *,
    history_steps: int,
) -> dict[str, Any]:
    if target_agents.ndim != 4 or target_agents.shape[-1] < 12:
        raise ValueError("target_agents must have shape [batch, agents, time, >=12]")
    if history_steps < 1 or history_steps >= target_agents.shape[2]:
        raise ValueError("history_steps must leave at least one future timestep")
    if not torch.isfinite(target_agents).all():
        raise FloatingPointError("target_agents must be finite")

    target_points = int((target_agents[..., history_steps:, -1] > 0.0).sum().item())
    static_squared_errors: list[torch.Tensor] = []
    velocity_squared_errors: list[torch.Tensor] = []
    for batch_index in range(target_agents.shape[0]):
        for agent_index in range(target_agents.shape[1]):
            track = target_agents[batch_index, agent_index]
            history_valid = torch.nonzero(
                track[:history_steps, -1] > 0.0, as_tuple=False
            ).flatten()
            if history_valid.numel() == 0:
                continue
            last_index = int(history_valid[-1].item())
            last_xy = track[last_index, :2]
            velocity = torch.zeros_like(last_xy)
            if history_valid.numel() >= 2:
                previous_index = int(history_valid[-2].item())
                velocity = (last_xy - track[previous_index, :2]) / (
                    last_index - previous_index
                )
            for future_index in range(history_steps, target_agents.shape[2]):
                if track[future_index, -1] <= 0.0:
                    continue
                target_xy = track[future_index, :2]
                static_squared_errors.append((last_xy - target_xy).square().sum())
                predicted_xy = last_xy + velocity * (future_index - last_index)
                velocity_squared_errors.append(
                    (predicted_xy - target_xy).square().sum()
                )

    evaluated_points = len(static_squared_errors)
    future_only_target_points = target_points - evaluated_points
    return {
        "target_points": target_points,
        "evaluated_points": evaluated_points,
        "future_only_target_points": future_only_target_points,
        "pointwise_gate_eligible": (
            target_points > 0 and future_only_target_points == 0
        ),
        "coverage": evaluated_points / target_points if target_points else None,
        "static_xy_rmse_m": _rmse_m(static_squared_errors),
        "constant_velocity_xy_rmse_m": _rmse_m(velocity_squared_errors),
    }


def summarize_t1_overfit_gate(
    denoising_report: dict[str, Any],
    *,
    thresholds: dict[str, float] | None = None,
    future_only_target_points: int = 0,
) -> dict[str, Any]:
    if future_only_target_points < 0:
        raise ValueError("future_only_target_points must be nonnegative")
    limits = dict(DEFAULT_T1_THRESHOLDS)
    if thresholds is not None:
        unknown = set(thresholds) - set(limits)
        if unknown:
            raise ValueError(f"unknown t=1 overfit thresholds: {sorted(unknown)}")
        limits.update({key: float(value) for key, value in thresholds.items()})
    try:
        level = denoising_report["levels"]["1.000000"]
        metrics = {
            "agent_xy_rmse_m": level["agents"]["xy_rmse_m"],
            "agent_heading_mae_deg": level["agents"]["heading_mae_deg"],
            "agent_type_accuracy": level["agents"]["type_accuracy"],
            "agent_validity_balanced_accuracy": level["agents"][
                "validity_balanced_accuracy"
            ],
            "light_state_accuracy": level["lights"]["state_accuracy"],
            "light_validity_balanced_accuracy": level["lights"][
                "validity_balanced_accuracy"
            ],
        }
    except (KeyError, TypeError) as error:
        raise ValueError("denoising report has no complete t=1 metrics") from error

    lower_is_better = {"agent_xy_rmse_m", "agent_heading_mae_deg"}
    checks = {
        name: _passes(value, limit, lower_is_better=name in lower_is_better)
        for name, value in metrics.items()
        for limit in (limits[name],)
    }
    pointwise_gate_eligible = future_only_target_points == 0
    status = "passed" if all(checks.values()) else "failed"
    if not pointwise_gate_eligible:
        status = "inapplicable"
    return {
        "status": status,
        "eligibility": {
            "pointwise_gate_eligible": pointwise_gate_eligible,
            "future_only_target_points": int(future_only_target_points),
            "reason": (
                None
                if pointwise_gate_eligible
                else "future-only slots require permutation-invariant generation metrics"
            ),
        },
        "metrics": metrics,
        "thresholds": limits,
        "checks": checks,
        "seeds": denoising_report.get("seeds"),
    }


def _rmse_m(squared_errors: list[torch.Tensor]) -> float | None:
    if not squared_errors:
        return None
    return float(torch.stack(squared_errors).mean().sqrt().item() * POSITION_SCALE)


def _passes(value: Any, threshold: float, *, lower_is_better: bool) -> bool:
    if value is None:
        return False
    numeric = float(value)
    if not math.isfinite(numeric):
        return False
    return numeric <= threshold if lower_is_better else numeric >= threshold
