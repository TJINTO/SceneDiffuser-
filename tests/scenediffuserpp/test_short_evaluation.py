import pytest
import torch

from scenediffuserpp.normalization import POSITION_SCALE
from scenediffuserpp.short_evaluation import summarize_short_evaluation
from scenediffuserpp.short_evaluation import ShortEvaluationThresholds
from scenediffuserpp.short_evaluation import sparse_validity_statistics


def _trajectory(step_normalized: float) -> torch.Tensor:
    values = torch.zeros(1, 2, 91, 12)
    values[..., -1] = -1.0
    values[:, 0, :, -1] = 1.0
    values[:, 0, :, 7:11] = -1.0
    values[:, 0, :, 7] = 1.0
    values[:, 0, :, 0] = torch.arange(91) * step_normalized
    return values


def _log_rows():
    return [
        {"global_step": step, "total_loss": 1.0 if step <= 10 else 0.5}
        for step in range(1, 21)
    ]


def _lights() -> torch.Tensor:
    values = torch.zeros(1, 2, 91, 13)
    values[..., -1] = -1.0
    values[:, 0, :, -1] = 1.0
    return values


def test_sparse_validity_statistics_include_history_future_boundary():
    values = torch.zeros(1, 1, 4, 3)
    values[0, 0, :, -1] = torch.tensor([-1.0, 1.0, 1.0, -1.0])

    stats = sparse_validity_statistics(values, history_steps=1)

    assert stats["valid_fraction"] == pytest.approx(2 / 3)
    assert stats["mean_valid_entities_per_frame"] == pytest.approx(2 / 3)
    assert stats["birth_transitions"] == 1
    assert stats["removal_transitions"] == 1
    assert stats["consecutive_valid_steps"] == 1


def test_short_evaluation_passes_noncollapsed_physical_generation():
    target = _trajectory(10.0 / POSITION_SCALE / 10.0)
    generated = target.clone()
    other_seed = generated.clone()
    other_seed[:, 1, 11:, 0] = 0.02
    lights = _lights()

    report = summarize_short_evaluation(
        _log_rows(),
        generated,
        other_seed,
        target,
        lights,
        lights.clone(),
        lights,
        history_steps=11,
    )

    assert report["status"] == "passed"
    assert report["loss_ratio"] == 0.5
    assert report["generated_agents"]["mean_speed_mps"] == pytest.approx(10.0)
    assert report["generated_lights"]["valid_fraction"] == 0.5
    assert report["seed_mean_absolute_difference"] > 0.0


def test_short_evaluation_blocks_implausible_generated_speed():
    target = _trajectory(10.0 / POSITION_SCALE / 10.0)
    generated = _trajectory(160.0 / POSITION_SCALE / 10.0)
    lights = _lights()

    report = summarize_short_evaluation(
        _log_rows(),
        generated,
        generated + 0.01,
        target,
        lights,
        lights,
        lights,
        history_steps=11,
    )

    assert report["status"] == "blocked"
    assert "implausible_generated_speed" in {
        failure["code"] for failure in report["failures"]
    }


def test_short_evaluation_reports_agent_and_light_collapse_separately():
    target_agents = _trajectory(10.0 / POSITION_SCALE / 10.0)
    target_lights = _lights()
    generated_agents = target_agents.clone()
    generated_agents[..., -1] = 1.0
    generated_lights = target_lights.clone()
    generated_lights[..., -1] = -1.0

    report = summarize_short_evaluation(
        _log_rows(),
        generated_agents,
        generated_agents + 0.01,
        target_agents,
        generated_lights,
        generated_lights,
        target_lights,
        history_steps=11,
    )

    failures = {failure["code"] for failure in report["failures"]}
    assert "collapsed_agent_validity" in failures
    assert "collapsed_light_validity" in failures
    assert "agent_validity_distribution_error" in failures
    assert "light_validity_distribution_error" in failures


def test_short_evaluation_accepts_all_invalid_lights_when_teacher_has_no_lights():
    agents = _trajectory(10.0 / POSITION_SCALE / 10.0)
    other_seed_agents = agents.clone()
    other_seed_agents[0, 1, 11:, 0] = 0.01
    target_lights = torch.zeros(1, 2, 91, 13)
    target_lights[..., -1] = -1.0
    generated_lights = target_lights.clone()

    report = summarize_short_evaluation(
        _log_rows(),
        agents,
        other_seed_agents,
        agents,
        generated_lights,
        generated_lights.clone(),
        target_lights,
        history_steps=11,
    )

    failures = {failure["code"] for failure in report["failures"]}
    assert "collapsed_light_validity" not in failures
    assert "light_validity_distribution_error" not in failures
    assert report["generated_lights"]["valid_fraction"] == 0.0
    assert report["target_lights"]["valid_fraction"] == 0.0
    assert report["status"] == "passed"


def test_short_evaluation_blocks_high_acceleration_with_plausible_mean_speed():
    target = _trajectory(10.0 / POSITION_SCALE / 10.0)
    generated = target.clone()
    increments = torch.zeros(80)
    increments[::2] = 20.0 / POSITION_SCALE / 10.0
    generated[0, 0, 11:, 0] = generated[0, 0, 10, 0] + torch.cumsum(
        increments, dim=0
    )
    other_seed = generated.clone()
    other_seed[0, 1, 11:, 0] = 0.01
    lights = _lights()

    report = summarize_short_evaluation(
        _log_rows(),
        generated,
        other_seed,
        target,
        lights,
        lights.clone(),
        lights,
        history_steps=11,
    )

    failures = {failure["code"] for failure in report["failures"]}
    assert report["generated_agents"]["p95_acceleration_mps2"] > 100.0
    assert "implausible_generated_acceleration" in failures


def test_short_evaluation_allows_tiny_acceleration_roundoff_at_threshold():
    target = _trajectory(10.0 / POSITION_SCALE / 10.0)
    generated = target.clone()
    velocities_mps = 10.0 + torch.arange(80) * 1.5001
    generated[0, 0, 11:, 0] = generated[0, 0, 10, 0] + torch.cumsum(
        velocities_mps / POSITION_SCALE / 10.0,
        dim=0,
    )
    other_seed = generated.clone()
    other_seed[0, 1, 11:, 0] = 0.01
    lights = _lights()
    thresholds = ShortEvaluationThresholds(
        maximum_mean_speed_mps=10000.0,
        maximum_mean_speed_ratio=10000.0,
        maximum_p95_speed_mps=10000.0,
        maximum_p95_speed_ratio=10000.0,
        maximum_p95_jerk_mps3=100000.0,
        maximum_p95_jerk_ratio=100000.0,
        maximum_static_fraction_error=1.0,
    )

    report = summarize_short_evaluation(
        _log_rows(),
        generated,
        other_seed,
        target,
        lights,
        lights.clone(),
        lights,
        history_steps=11,
        thresholds=thresholds,
    )

    failures = {failure["code"] for failure in report["failures"]}
    assert report["generated_agents"]["p95_acceleration_mps2"] == pytest.approx(
        15.001,
        abs=0.01,
    )
    assert "implausible_generated_acceleration" not in failures


def test_short_evaluation_does_not_reject_teacher_matched_high_jerk():
    target = _trajectory(0.0)
    increments = torch.zeros(80)
    increments[::2] = 1.0 / POSITION_SCALE / 10.0
    target[0, 0, 11:, 0] = target[0, 0, 10, 0] + torch.cumsum(
        increments, dim=0
    )
    generated = target.clone()
    other_seed = generated.clone()
    other_seed[0, 1, 11:, 0] = 0.01
    lights = _lights()

    report = summarize_short_evaluation(
        _log_rows(),
        generated,
        other_seed,
        target,
        lights,
        lights.clone(),
        lights,
        history_steps=11,
    )

    failures = {failure["code"] for failure in report["failures"]}
    assert report["target_agents"]["p95_jerk_mps3"] > 100.0
    assert "implausible_generated_jerk" not in failures
    assert report["status"] == "passed"


def test_short_evaluation_blocks_stopped_fraction_distribution_error():
    target = _trajectory(0.0)
    target_increments = torch.zeros(80)
    target_increments[40:] = 10.0 / POSITION_SCALE / 10.0
    target[0, 0, 11:, 0] = torch.cumsum(target_increments, dim=0)
    generated = _trajectory(5.0 / POSITION_SCALE / 10.0)
    other_seed = generated.clone()
    other_seed[0, 1, 11:, 0] = 0.01
    lights = _lights()

    report = summarize_short_evaluation(
        _log_rows(),
        generated,
        other_seed,
        target,
        lights,
        lights.clone(),
        lights,
        history_steps=11,
    )

    failures = {failure["code"] for failure in report["failures"]}
    assert report["target_agents"]["static_step_fraction"] > 0.45
    assert report["generated_agents"]["static_step_fraction"] == 0.0
    assert "static_fraction_distribution_error" in failures


def test_short_evaluation_blocks_missing_light_removals():
    agents = _trajectory(10.0 / POSITION_SCALE / 10.0)
    target_lights = _lights()
    target_lights[0, 0, 31:51, -1] = -1.0
    target_lights[0, 0, 71:, -1] = -1.0
    generated_lights = _lights()
    thresholds = ShortEvaluationThresholds(
        sparse_transition_absolute_tolerance=0,
        sparse_transition_relative_tolerance=0.0,
    )

    report = summarize_short_evaluation(
        _log_rows(),
        agents,
        agents + 0.01,
        agents,
        generated_lights,
        generated_lights.clone(),
        target_lights,
        history_steps=11,
        thresholds=thresholds,
    )

    failures = {failure["code"] for failure in report["failures"]}
    assert report["light_transition_errors"]["removals"] == 2
    assert "light_removal_transition_error" in failures


def test_short_evaluation_blocks_missing_light_state_changes():
    agents = _trajectory(10.0 / POSITION_SCALE / 10.0)
    generated_lights = _lights()
    target_lights = _lights()
    target_lights[0, 0, :, 3:12] = -1.0
    target_lights[0, 0, :41, 4] = 1.0
    target_lights[0, 0, 41:, 6] = 1.0
    generated_lights[0, 0, :, 3:12] = -1.0
    generated_lights[0, 0, :, 4] = 1.0
    thresholds = ShortEvaluationThresholds(
        sparse_transition_absolute_tolerance=0,
        sparse_transition_relative_tolerance=0.0,
    )

    report = summarize_short_evaluation(
        _log_rows(),
        agents,
        agents + 0.01,
        agents,
        generated_lights,
        generated_lights.clone(),
        target_lights,
        history_steps=11,
        thresholds=thresholds,
    )

    failures = {failure["code"] for failure in report["failures"]}
    assert report["generated_lights"]["state_change_transitions"] == 0
    assert report["target_lights"]["state_change_transitions"] == 1
    assert report["light_transition_errors"]["state_changes"] == 1
    assert "light_state_transition_error" in failures
