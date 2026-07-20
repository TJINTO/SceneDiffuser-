import pytest
import torch

from scenediffuserpp.normalization import POSITION_SCALE
from scenediffuserpp.overfit_evaluation import behavior_prediction_baselines
from scenediffuserpp.overfit_evaluation import summarize_t1_overfit_gate


def test_constant_velocity_baseline_extrapolates_history():
    agents = torch.zeros(1, 1, 5, 12)
    agents[..., -1] = 1.0
    agents[0, 0, :, 0] = torch.arange(5) / POSITION_SCALE

    report = behavior_prediction_baselines(agents, history_steps=2)

    assert report["evaluated_points"] == 3
    assert report["coverage"] == 1.0
    assert report["constant_velocity_xy_rmse_m"] == pytest.approx(0.0, abs=1e-6)
    assert report["static_xy_rmse_m"] == pytest.approx((14 / 3) ** 0.5)


def test_baseline_reports_future_only_targets_as_pointwise_ineligible():
    agents = torch.zeros(1, 2, 5, 12)
    agents[0, 0, :, -1] = 1.0
    agents[0, 1, :2, -1] = -1.0
    agents[0, 1, 2:, -1] = 1.0

    report = behavior_prediction_baselines(agents, history_steps=2)

    assert report["target_points"] == 6
    assert report["evaluated_points"] == 3
    assert report["future_only_target_points"] == 3
    assert report["pointwise_gate_eligible"] is False


def test_t1_overfit_gate_requires_geometry_and_classification_metrics():
    report = {
        "seeds": [7, 8],
        "levels": {
            "1.000000": {
                "agents": {
                    "xy_rmse_m": 1.5,
                    "heading_mae_deg": 4.0,
                    "type_accuracy": 1.0,
                    "validity_balanced_accuracy": 0.995,
                },
                "lights": {
                    "state_accuracy": 1.0,
                    "validity_balanced_accuracy": 0.995,
                },
            }
        },
    }

    passed = summarize_t1_overfit_gate(report)
    failed = summarize_t1_overfit_gate(
        {
            **report,
            "levels": {
                "1.000000": {
                    **report["levels"]["1.000000"],
                    "agents": {
                        **report["levels"]["1.000000"]["agents"],
                        "xy_rmse_m": 2.1,
                    },
                }
            },
        }
    )

    assert passed["status"] == "passed"
    assert all(passed["checks"].values())
    assert failed["status"] == "failed"
    assert failed["checks"]["agent_xy_rmse_m"] is False


def test_t1_overfit_gate_is_inapplicable_to_future_only_agent_slots():
    report = {
        "seeds": [7, 8],
        "levels": {
            "1.000000": {
                "agents": {
                    "xy_rmse_m": 1.0,
                    "heading_mae_deg": 1.0,
                    "type_accuracy": 1.0,
                    "validity_balanced_accuracy": 1.0,
                },
                "lights": {
                    "state_accuracy": 1.0,
                    "validity_balanced_accuracy": 1.0,
                },
            }
        },
    }

    result = summarize_t1_overfit_gate(report, future_only_target_points=3)

    assert result["status"] == "inapplicable"
    assert result["eligibility"]["pointwise_gate_eligible"] is False
