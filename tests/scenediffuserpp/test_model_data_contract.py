import numpy as np
import pytest

from topoworld.scenediffuserpp.model_data_contract import validate_scene_model_contract
from topoworld.scenediffuserpp.model_data_contract import validate_training_model_contract


def _sample(*, agents: int = 32, lights: int = 32, timesteps: int = 91) -> dict:
    return {
        "agents": np.zeros((agents, timesteps, 12), dtype=np.float32),
        "lights": np.zeros((lights, timesteps, 13), dtype=np.float32),
    }


def _model(*, agents: int = 32, lights: int = 32, timesteps: int = 91) -> dict:
    return {
        "maximum_agents": agents,
        "maximum_lights": lights,
        "timesteps": timesteps,
        "agent_channels": 12,
        "light_channels": 13,
    }


def test_scene_model_contract_accepts_exact_tensor_capacities():
    assert validate_scene_model_contract(_sample(), _model()) == {
        "agents": [32, 91, 12],
        "lights": [32, 91, 13],
    }


def test_scene_model_contract_rejects_paper_label_on_reduced_data():
    with pytest.raises(
        ValueError,
        match=r"agents shape mismatch: expected \[128, 91, 12\], got \[32, 91, 12\]",
    ):
        validate_scene_model_contract(_sample(), _model(agents=128, lights=64))


def test_scene_model_contract_requires_declared_capacity_fields():
    with pytest.raises(ValueError, match="missing required fields: maximum_lights"):
        validate_scene_model_contract(_sample(), {"maximum_agents": 32, "timesteps": 91})


def test_training_model_contract_requires_the_same_history_boundary():
    validate_training_model_contract(
        {"history_steps": 11}, {"history_steps": 11}
    )

    with pytest.raises(
        ValueError, match="history_steps mismatch: model declares 11, training uses 10"
    ):
        validate_training_model_contract(
            {"history_steps": 11}, {"history_steps": 10}
        )
    with pytest.raises(ValueError, match="model config missing history_steps"):
        validate_training_model_contract({}, {"history_steps": 11})
