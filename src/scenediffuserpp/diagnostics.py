from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

from scenediffuserpp.normalization import AgentNormalizer
from scenediffuserpp.normalization import LightNormalizer
from scenediffuserpp.normalization import POSITION_SCALE


def plot_scene(
    sample: dict,
    path: str | Path,
    *,
    history_steps: int = 11,
    observation_radius_m: float = 80.0,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 8), dpi=140)

    map_valid = ~sample["map_padding_mask"]
    map_xy = sample["map_points"][map_valid, :2] * POSITION_SCALE
    if len(map_xy):
        axis.scatter(map_xy[:, 0], map_xy[:, 1], s=2, c="#9ca3af", alpha=0.65)

    normalizer = AgentNormalizer()
    for slot, identifier in enumerate(sample["agent_ids"]):
        if not identifier:
            continue
        states = []
        frames = []
        for frame, row in enumerate(sample["agents"][slot]):
            continuous, _agent_type, valid = normalizer.decode_agent(row)
            if valid:
                states.append(continuous)
                frames.append(frame)
        if not states:
            continue
        values = np.asarray(states)
        frames_array = np.asarray(frames)
        is_av = slot == 0
        color = "#dc2626" if is_av else "#2563eb"
        history = frames_array < history_steps
        future = ~history
        axis.plot(values[history, 0], values[history, 1], color=color, linewidth=2.0)
        axis.plot(
            values[future, 0],
            values[future, 1],
            color=color,
            linewidth=1.2,
            alpha=0.55,
        )
        state = values[-1]
        box = Rectangle(
            (state[0] - state[4] / 2.0, state[1] - state[5] / 2.0),
            state[4],
            state[5],
            fill=False,
            edgecolor=color,
            linewidth=1.2,
        )
        box.set_transform(
            Affine2D().rotate_around(state[0], state[1], state[3]) + axis.transData
        )
        axis.add_patch(box)

    light_normalizer = LightNormalizer()
    for slot, identifier in enumerate(sample["light_ids"]):
        if not identifier:
            continue
        for row in sample["lights"][slot]:
            xyz, state, valid = light_normalizer.decode_light(row)
            if valid:
                color = _light_color(state.name)
                axis.scatter(xyz[0], xyz[1], marker="s", s=22, c=color, edgecolors="black")
                break

    axis.add_patch(
        Circle((0.0, 0.0), observation_radius_m, fill=False, linestyle="--", color="#111827")
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(-observation_radius_m * 1.1, observation_radius_m * 1.1)
    axis.set_ylim(-observation_radius_m * 1.1, observation_radius_m * 1.1)
    axis.set_xlabel("local x (m)")
    axis.set_ylabel("local y (m)")
    axis.grid(alpha=0.15)
    figure.tight_layout()
    figure.savefig(destination)
    plt.close(figure)
    return destination


def _light_color(name: str) -> str:
    if "GREEN" in name:
        return "#16a34a"
    if "YELLOW" in name:
        return "#eab308"
    if "RED" in name:
        return "#dc2626"
    return "#6b7280"
