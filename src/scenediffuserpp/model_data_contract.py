from __future__ import annotations

from typing import Any
from typing import Mapping


def validate_training_model_contract(
    model_values: Mapping[str, Any], training_values: Mapping[str, Any]
) -> None:
    if "history_steps" not in model_values:
        raise ValueError("model config missing history_steps")
    if "history_steps" not in training_values:
        raise ValueError("training config missing history_steps")
    model_history = int(model_values["history_steps"])
    training_history = int(training_values["history_steps"])
    if model_history != training_history:
        raise ValueError(
            "history_steps mismatch: model declares "
            f"{model_history}, training uses {training_history}"
        )


def validate_scene_model_contract(
    sample: Mapping[str, Any], model_values: Mapping[str, Any]
) -> dict[str, list[int]]:
    required = ("maximum_agents", "maximum_lights", "timesteps")
    missing = [name for name in required if name not in model_values]
    if missing:
        raise ValueError("model config missing required fields: " + ", ".join(missing))

    expected = {
        "agents": (
            int(model_values["maximum_agents"]),
            int(model_values["timesteps"]),
            int(model_values.get("agent_channels", 12)),
        ),
        "lights": (
            int(model_values["maximum_lights"]),
            int(model_values["timesteps"]),
            int(model_values.get("light_channels", 13)),
        ),
    }
    actual: dict[str, list[int]] = {}
    for name, expected_shape in expected.items():
        if name not in sample or not hasattr(sample[name], "shape"):
            raise ValueError(f"scene sample is missing shaped tensor {name!r}")
        shape = tuple(int(value) for value in sample[name].shape)
        if shape != expected_shape:
            raise ValueError(
                f"{name} shape mismatch: expected {list(expected_shape)}, "
                f"got {list(shape)}"
            )
        actual[name] = list(shape)
    return actual
