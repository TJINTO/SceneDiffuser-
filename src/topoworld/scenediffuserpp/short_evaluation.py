from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from typing import Any

import torch

from topoworld.scenediffuserpp.normalization import POSITION_SCALE


@dataclass(frozen=True)
class ShortEvaluationThresholds:
    max_loss_ratio: float = 0.70
    minimum_valid_fraction: float = 0.01
    maximum_valid_fraction: float = 0.99
    minimum_seed_difference: float = 1e-5
    maximum_validity_fraction_error: float = 0.05
    minimum_mean_speed_ratio: float = 0.5
    maximum_mean_speed_ratio: float = 2.0
    maximum_mean_speed_mps: float = 40.0
    maximum_p95_speed_mps: float = 45.0
    maximum_p95_speed_ratio: float = 2.0
    maximum_p95_acceleration_mps2: float = 15.0
    maximum_p95_acceleration_ratio: float = 2.0
    maximum_p95_jerk_mps3: float = 100.0
    maximum_p95_jerk_ratio: float = 2.0
    maximum_static_fraction_error: float = 0.15
    sparse_transition_absolute_tolerance: int = 2
    sparse_transition_relative_tolerance: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_valid_fraction < self.maximum_valid_fraction <= 1.0:
            raise ValueError("valid-fraction thresholds must be ordered within [0, 1]")
        if self.max_loss_ratio <= 0.0 or self.minimum_seed_difference < 0.0:
            raise ValueError("loss and diversity thresholds must be nonnegative")
        positive = (
            self.minimum_mean_speed_ratio,
            self.maximum_mean_speed_ratio,
            self.maximum_mean_speed_mps,
            self.maximum_p95_speed_mps,
            self.maximum_p95_speed_ratio,
            self.maximum_p95_acceleration_mps2,
            self.maximum_p95_acceleration_ratio,
            self.maximum_p95_jerk_mps3,
            self.maximum_p95_jerk_ratio,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("kinematic thresholds must be positive")
        if self.minimum_mean_speed_ratio > self.maximum_mean_speed_ratio:
            raise ValueError("mean-speed ratio thresholds are reversed")
        if self.maximum_validity_fraction_error < 0.0:
            raise ValueError("validity error threshold must be nonnegative")
        if self.maximum_static_fraction_error < 0.0:
            raise ValueError("static-fraction threshold must be nonnegative")
        if self.sparse_transition_absolute_tolerance < 0:
            raise ValueError("transition absolute tolerance must be nonnegative")
        if self.sparse_transition_relative_tolerance < 0.0:
            raise ValueError("transition relative tolerance must be nonnegative")


def summarize_short_evaluation(
    log_rows: list[dict[str, Any]],
    generated_agents: torch.Tensor,
    other_seed_agents: torch.Tensor,
    target_agents: torch.Tensor,
    generated_lights: torch.Tensor,
    other_seed_lights: torch.Tensor,
    target_lights: torch.Tensor,
    *,
    history_steps: int,
    frequency_hz: float = 10.0,
    thresholds: ShortEvaluationThresholds | None = None,
) -> dict[str, Any]:
    limits = thresholds or ShortEvaluationThresholds()
    if not log_rows:
        raise ValueError("training log is empty")
    if generated_agents.shape != target_agents.shape:
        raise ValueError("generated and target agent tensors must have equal shape")
    if other_seed_agents.shape != generated_agents.shape:
        raise ValueError("the second generated sample has a different shape")
    if generated_lights.shape != target_lights.shape:
        raise ValueError("generated and target light tensors must have equal shape")
    if other_seed_lights.shape != generated_lights.shape:
        raise ValueError("the second generated light sample has a different shape")
    window = min(10, max(len(log_rows) // 2, 1))
    initial_loss = sum(float(row["total_loss"]) for row in log_rows[:window]) / window
    final_loss = sum(float(row["total_loss"]) for row in log_rows[-window:]) / window
    loss_ratio = final_loss / max(initial_loss, 1e-12)
    generated_agent_stats = trajectory_statistics(
        generated_agents, history_steps=history_steps, frequency_hz=frequency_hz
    )
    target_agent_stats = trajectory_statistics(
        target_agents, history_steps=history_steps, frequency_hz=frequency_hz
    )
    generated_light_stats = light_statistics(
        generated_lights, history_steps=history_steps
    )
    target_light_stats = light_statistics(
        target_lights, history_steps=history_steps
    )
    seed_difference = float(
        (generated_agents - other_seed_agents).abs().mean().detach().cpu()
    )
    light_seed_difference = float(
        (generated_lights - other_seed_lights).abs().mean().detach().cpu()
    )
    failures: list[dict[str, str]] = []
    if loss_ratio >= limits.max_loss_ratio:
        failures.append(_issue("insufficient_loss_reduction", f"loss ratio is {loss_ratio:.3f}"))
    if not (
        limits.minimum_valid_fraction
        < generated_agent_stats["valid_fraction"]
        < limits.maximum_valid_fraction
    ):
        failures.append(
            _issue(
                "collapsed_agent_validity",
                "generated agent validity is all-off or all-on",
            )
        )
    if not _matches_target_boundary_validity(
        generated_light_stats["valid_fraction"],
        target_light_stats["valid_fraction"],
        limits,
    ) and not (
        limits.minimum_valid_fraction
        < generated_light_stats["valid_fraction"]
        < limits.maximum_valid_fraction
    ):
        failures.append(
            _issue(
                "collapsed_light_validity",
                "generated light validity is all-off or all-on",
            )
        )
    if seed_difference <= limits.minimum_seed_difference:
        failures.append(_issue("collapsed_sampling_diversity", "two seeds produce the same tensor"))
    target_speed = target_agent_stats["mean_speed_mps"]
    generated_speed = generated_agent_stats["mean_speed_mps"]
    speed_ratio = generated_speed / max(target_speed, 1e-6)
    if (
        generated_speed > limits.maximum_mean_speed_mps
        or generated_agent_stats["p95_speed_mps"] > limits.maximum_p95_speed_mps
        or (
            target_speed > 0.1
            and not limits.minimum_mean_speed_ratio
            <= speed_ratio
            <= limits.maximum_mean_speed_ratio
        )
        or (
            target_agent_stats["p95_speed_mps"] > 0.1
            and generated_agent_stats["p95_speed_mps"]
            / target_agent_stats["p95_speed_mps"]
            > limits.maximum_p95_speed_ratio
        )
    ):
        failures.append(
            _issue(
                "implausible_generated_speed",
                f"generated mean/P95 speed is {generated_speed:.2f}/{generated_agent_stats['p95_speed_mps']:.2f} m/s",
            )
        )
    if generated_agent_stats["static_step_fraction"] >= 0.99:
        failures.append(_issue("static_generation", "generated valid trajectories are static"))
    static_fraction_error = abs(
        generated_agent_stats["static_step_fraction"]
        - target_agent_stats["static_step_fraction"]
    )
    if static_fraction_error > limits.maximum_static_fraction_error:
        failures.append(
            _issue(
                "static_fraction_distribution_error",
                f"static-step fraction error is {static_fraction_error:.3f}",
            )
        )
    effective_acceleration_limit = max(
        limits.maximum_p95_acceleration_mps2,
        target_agent_stats["p95_acceleration_mps2"]
        * limits.maximum_p95_acceleration_ratio,
    )
    if _exceeds_limit(
        generated_agent_stats["p95_acceleration_mps2"],
        effective_acceleration_limit,
    ):
        failures.append(
            _issue(
                "implausible_generated_acceleration",
                "generated P95 acceleration is "
                f"{generated_agent_stats['p95_acceleration_mps2']:.2f} m/s^2 "
                f"(effective limit {effective_acceleration_limit:.2f})",
            )
        )
    effective_jerk_limit = max(
        limits.maximum_p95_jerk_mps3,
        target_agent_stats["p95_jerk_mps3"] * limits.maximum_p95_jerk_ratio,
    )
    if _exceeds_limit(
        generated_agent_stats["p95_jerk_mps3"],
        effective_jerk_limit,
    ):
        failures.append(
            _issue(
                "implausible_generated_jerk",
                f"generated P95 jerk is {generated_agent_stats['p95_jerk_mps3']:.2f} "
                f"m/s^3 (effective limit {effective_jerk_limit:.2f})",
            )
        )
    if (
        abs(
            generated_agent_stats["valid_fraction"]
            - target_agent_stats["valid_fraction"]
        )
        > limits.maximum_validity_fraction_error
    ):
        failures.append(
            _issue(
                "agent_validity_distribution_error",
                "agent valid fraction differs by over 0.05",
            )
        )
    if (
        abs(
            generated_light_stats["valid_fraction"]
            - target_light_stats["valid_fraction"]
        )
        > limits.maximum_validity_fraction_error
    ):
        failures.append(
            _issue(
                "light_validity_distribution_error",
                "light valid fraction differs by over 0.05",
            )
        )
    agent_transition_errors = _transition_errors(
        generated_agent_stats, target_agent_stats
    )
    light_transition_errors = _transition_errors(
        generated_light_stats, target_light_stats
    )
    _append_transition_failures(
        failures,
        "agent",
        agent_transition_errors,
        target_agent_stats,
        limits,
    )
    _append_transition_failures(
        failures,
        "light",
        light_transition_errors,
        target_light_stats,
        limits,
    )
    return {
        "status": "blocked" if failures else "passed",
        "failures": failures,
        "thresholds": asdict(limits),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_ratio": loss_ratio,
        "seed_mean_absolute_difference": seed_difference,
        "light_seed_mean_absolute_difference": light_seed_difference,
        "speed_ratio": speed_ratio,
        "effective_p95_acceleration_limit_mps2": effective_acceleration_limit,
        "effective_p95_jerk_limit_mps3": effective_jerk_limit,
        "static_fraction_error": static_fraction_error,
        "agent_transition_errors": agent_transition_errors,
        "light_transition_errors": light_transition_errors,
        "agent_valid_count_mae": _valid_count_mae(
            generated_agents, target_agents, history_steps
        ),
        "light_valid_count_mae": _valid_count_mae(
            generated_lights, target_lights, history_steps
        ),
        "generated_agents": generated_agent_stats,
        "target_agents": target_agent_stats,
        "generated_lights": generated_light_stats,
        "target_lights": target_light_stats,
    }


def sparse_validity_statistics(
    values: torch.Tensor, *, history_steps: int
) -> dict[str, float | int]:
    if values.ndim != 4 or values.shape[-1] < 1:
        raise ValueError("sparse values must have shape [batch, entities, time, channels]")
    if history_steps < 0 or history_steps >= values.shape[2]:
        raise ValueError("history_steps must leave at least one future frame")
    valid = values[..., -1] > 0.0
    future = valid[:, :, history_steps:]
    transition_start = max(history_steps - 1, 0)
    transitions = valid[:, :, transition_start:]
    previous = transitions[:, :, :-1]
    current = transitions[:, :, 1:]
    valid_counts = future.sum(dim=1).float()
    return {
        "valid_fraction": float(future.float().mean().detach().cpu()),
        "mean_valid_entities_per_frame": float(valid_counts.mean().detach().cpu()),
        "birth_transitions": int(((~previous) & current).sum().detach().cpu()),
        "removal_transitions": int((previous & ~current).sum().detach().cpu()),
        "consecutive_valid_steps": int((previous & current).sum().detach().cpu()),
    }


def light_statistics(
    values: torch.Tensor, *, history_steps: int
) -> dict[str, float | int]:
    if values.ndim != 4 or values.shape[-1] <= 4:
        raise ValueError("light values must contain xyz, state, and validity channels")
    stats = sparse_validity_statistics(values, history_steps=history_steps)
    valid = values[..., -1] > 0.0
    states = values[..., 3:-1].argmax(dim=-1)
    transition_start = max(history_steps - 1, 0)
    transition_valid = valid[:, :, transition_start:]
    transition_states = states[:, :, transition_start:]
    consecutive = transition_valid[:, :, :-1] & transition_valid[:, :, 1:]
    changes = consecutive & (
        transition_states[:, :, :-1] != transition_states[:, :, 1:]
    )
    return {
        **stats,
        "state_change_transitions": int(changes.sum().detach().cpu()),
    }


def trajectory_statistics(
    agents: torch.Tensor,
    *,
    history_steps: int,
    frequency_hz: float = 10.0,
) -> dict[str, float | int]:
    if agents.ndim != 4 or agents.shape[-1] < 3:
        raise ValueError("agents must have shape [batch, entities, time, channels]")
    if history_steps < 0 or history_steps >= agents.shape[2]:
        raise ValueError("history_steps must leave at least one future frame")
    motion_start = max(history_steps - 1, 0)
    motion = agents[:, :, motion_start:]
    valid = motion[..., -1] > 0.0
    positions = motion[..., :2] * POSITION_SCALE
    velocity_vectors = torch.diff(positions, dim=2) * frequency_hz
    consecutive = valid[:, :, 1:] & valid[:, :, :-1]
    speeds = torch.linalg.vector_norm(velocity_vectors, dim=-1)[consecutive]
    acceleration_vectors = torch.diff(velocity_vectors, dim=2) * frequency_hz
    acceleration_valid = valid[:, :, 2:] & valid[:, :, 1:-1] & valid[:, :, :-2]
    accelerations = torch.linalg.vector_norm(
        acceleration_vectors, dim=-1
    )[acceleration_valid]
    jerk_vectors = torch.diff(acceleration_vectors, dim=2) * frequency_hz
    jerk_valid = (
        valid[:, :, 3:]
        & valid[:, :, 2:-1]
        & valid[:, :, 1:-2]
        & valid[:, :, :-3]
    )
    jerks = torch.linalg.vector_norm(jerk_vectors, dim=-1)[jerk_valid]
    if speeds.numel():
        mean_speed = float(speeds.mean().detach().cpu())
        p95_speed = float(torch.quantile(speeds.float(), 0.95).detach().cpu())
        static_fraction = float((speeds < 0.5).float().mean().detach().cpu())
    else:
        mean_speed = p95_speed = 0.0
        static_fraction = 1.0
    return {
        **sparse_validity_statistics(agents, history_steps=history_steps),
        "mean_speed_mps": mean_speed,
        "p95_speed_mps": p95_speed,
        "p95_acceleration_mps2": _p95(accelerations),
        "p95_jerk_mps3": _p95(jerks),
        "static_step_fraction": static_fraction,
    }


def _p95(values: torch.Tensor) -> float:
    if not values.numel():
        return 0.0
    return float(torch.quantile(values.float(), 0.95).detach().cpu())


def _transition_errors(
    generated: dict[str, float | int], target: dict[str, float | int]
) -> dict[str, int]:
    result = {
        "births": abs(
            int(generated["birth_transitions"]) - int(target["birth_transitions"])
        ),
        "removals": abs(
            int(generated["removal_transitions"])
            - int(target["removal_transitions"])
        ),
    }
    if "state_change_transitions" in generated and "state_change_transitions" in target:
        result["state_changes"] = abs(
            int(generated["state_change_transitions"])
            - int(target["state_change_transitions"])
        )
    return result


def _append_transition_failures(
    failures: list[dict[str, str]],
    kind: str,
    errors: dict[str, int],
    target: dict[str, float | int],
    thresholds: ShortEvaluationThresholds,
) -> None:
    transitions = [
        ("births", "birth_transitions"),
        ("removals", "removal_transitions"),
    ]
    if "state_changes" in errors:
        transitions.append(("state_changes", "state_change_transitions"))
    for plural, target_key in transitions:
        target_count = int(target[target_key])
        tolerance = max(
            float(thresholds.sparse_transition_absolute_tolerance),
            target_count * thresholds.sparse_transition_relative_tolerance,
        )
        if errors[plural] > tolerance:
            singular = {
                "births": "birth",
                "removals": "removal",
                "state_changes": "state",
            }[plural]
            failures.append(
                _issue(
                    f"{kind}_{singular}_transition_error",
                    f"{kind} {plural} error {errors[plural]} exceeds tolerance {tolerance:g}",
                )
            )


def _valid_count_mae(
    generated: torch.Tensor, target: torch.Tensor, history_steps: int
) -> float:
    generated_counts = (generated[:, :, history_steps:, -1] > 0.0).sum(dim=1)
    target_counts = (target[:, :, history_steps:, -1] > 0.0).sum(dim=1)
    return float(
        (generated_counts - target_counts).abs().float().mean().detach().cpu()
    )


def _matches_target_boundary_validity(
    generated: float,
    target: float,
    thresholds: ShortEvaluationThresholds,
) -> bool:
    return (
        generated <= thresholds.minimum_valid_fraction
        and target <= thresholds.minimum_valid_fraction
    ) or (
        generated >= thresholds.maximum_valid_fraction
        and target >= thresholds.maximum_valid_fraction
    )


def _exceeds_limit(value: float, limit: float) -> bool:
    tolerance = max(abs(limit) * 1e-3, 1e-6)
    return value > limit + tolerance


def _issue(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}
