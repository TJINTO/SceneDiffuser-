from pathlib import Path

import numpy as np
import pytest

from topoworld.scenediffuserpp.normalization import AgentNormalizer
from topoworld.scenediffuserpp.normalization import LightNormalizer
from topoworld.scenediffuserpp.roadgraph import LightToken
from topoworld.scenediffuserpp.scene_builder import AgentState
from topoworld.scenediffuserpp.scene_builder import build_window
from topoworld.scenediffuserpp.scene_builder import candidate_windows
from topoworld.scenediffuserpp.scene_builder import count_light_state_transitions
from topoworld.scenediffuserpp.scene_builder import parse_fcd
from topoworld.scenediffuserpp.scene_builder import parse_tls_jsonl
from topoworld.scenediffuserpp.schema import AgentType
from topoworld.scenediffuserpp.schema import LightState
from topoworld.scenediffuserpp.schema import SceneSpec


def _state(x, y=0.0, heading=0.0, speed=10.0):
    return AgentState(
        x=float(x),
        y=float(y),
        z=0.0,
        heading=float(heading),
        speed=float(speed),
        length=4.5,
        width=2.0,
        height=1.75,
        type_id="car",
    )


def _base_tracks():
    return {
        "ego": {step: _state(step) for step in range(91)},
        "near": {step: _state(step, y=10.0) for step in range(91)},
    }


def test_window_reserves_agent_zero_for_av():
    window = build_window(
        _base_tracks(), av_id="ego", start_step=0, spec=SceneSpec.small()
    )

    assert window.agent_ids[0] == "ego"
    assert np.all(window.agents[0, :, -1] == 1.0)
    _, agent_type, valid = AgentNormalizer().decode_agent(window.agents[0, 10])
    assert agent_type is AgentType.AV
    assert valid is True


def test_agent_entering_80m_radius_changes_only_validity_before_entry():
    tracks = {
        "ego": {step: _state(step) for step in range(91)},
        "car": {step: _state(100 - step) for step in range(91)},
    }

    window = build_window(tracks, av_id="ego", start_step=0, spec=SceneSpec.small())
    slot = window.agent_ids.index("car")

    assert np.all(window.agents[slot, :10, :-1] == 0.0)
    assert np.all(window.agents[slot, :10, -1] == -1.0)
    assert window.agents[slot, 10, -1] == 1.0


def test_agent_exiting_80m_radius_keeps_slot_and_negative_validity_after_exit():
    tracks = {
        "ego": {step: _state(float(step)) for step in range(91)},
        "car": {step: _state(float(step), y=79.0 if step < 20 else 81.0) for step in range(91)},
    }

    window = build_window(tracks, av_id="ego", start_step=0, spec=SceneSpec.small())
    slot = window.agent_ids.index("car")

    assert window.agent_ids[slot] == "car"
    assert np.all(window.agents[slot, :20, -1] == 1.0)
    assert np.all(window.agents[slot, 20:, -1] == -1.0)
    assert np.all(window.agents[slot, 20:, :-1] == 0.0)


def test_reference_frame_uses_av_pose_at_last_history_step():
    tracks = {
        "ego": {
            step: _state(10.0, y=float(step), heading=np.pi / 2)
            for step in range(91)
        },
        "east": {
            step: _state(20.0, y=float(step), heading=np.pi / 2)
            for step in range(91)
        },
    }

    window = build_window(tracks, av_id="ego", start_step=0, spec=SceneSpec.small())
    ego, _, _ = AgentNormalizer().decode_agent(window.agents[0, 10])
    east_slot = window.agent_ids.index("east")
    east, _, _ = AgentNormalizer().decode_agent(window.agents[east_slot, 10])

    np.testing.assert_allclose(ego[:3], [0.0, 0.0, 0.0], atol=1e-6)
    assert ego[3] == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_allclose(east[:2], [0.0, -10.0], atol=1e-5)
    np.testing.assert_allclose(window.reference_world_pose.xy, [10.0, 10.0])
    assert window.reference_world_pose.heading == pytest.approx(np.pi / 2)


def test_fast_future_trajectory_stays_inside_normalized_position_range():
    tracks = {
        "ego": {step: _state(step * 3.0, speed=30.0) for step in range(91)},
        "near": {step: _state(step * 3.0, y=10.0, speed=30.0) for step in range(91)},
    }

    window = build_window(tracks, av_id="ego", start_step=0, spec=SceneSpec.small())
    valid = window.agents[..., -1] > 0.0

    assert np.max(np.abs(window.agents[..., :2][valid])) <= 1.0


def test_light_tensor_uses_connection_state_and_local_stop_line():
    light = LightToken(
        tls_id="tls0",
        link_index=0,
        incoming_lane_id="e0_0",
        outgoing_lane_id="e1_0",
        stop_line_xy=np.array([20.0, 0.0], dtype=np.float32),
        turn_direction="s",
    )
    tls = {step: {"tls0": "r" if step < 20 else "G"} for step in range(91)}

    window = build_window(
        _base_tracks(),
        av_id="ego",
        start_step=0,
        spec=SceneSpec.small(),
        light_tokens=[light],
        tls_states=tls,
    )
    xyz, state, valid = LightNormalizer().decode_light(window.lights[0, 10])

    np.testing.assert_allclose(xyz[:2], [10.0, 0.0], atol=1e-6)
    assert state is LightState.RED
    assert valid is True


def test_parse_fcd_converts_sumo_bearing_to_math_heading(tmp_path: Path):
    path = tmp_path / "fcd.xml"
    path.write_text(
        """<fcd-export>
  <timestep time="0.00"><vehicle id="north" x="1" y="2" angle="0" speed="3" type="car"/></timestep>
  <timestep time="0.10"><vehicle id="east" x="2" y="2" angle="90" speed="4" type="car"/></timestep>
</fcd-export>
""",
        encoding="utf-8",
    )

    tracks = parse_fcd(path)

    assert tracks["north"][0].heading == pytest.approx(np.pi / 2)
    assert tracks["east"][1].heading == pytest.approx(0.0)


def test_parse_tls_jsonl_rejects_duplicate_tls_step(tmp_path: Path):
    path = tmp_path / "tls.jsonl"
    row = '{"time_s":0.1,"tls_id":"t0","state":"r"}\n'
    path.write_text(row + row, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate TLS state"):
        parse_tls_jsonl(path, frequency_hz=10)


def test_candidate_windows_can_require_local_density_and_signal_context():
    light = LightToken(
        tls_id="tls0",
        link_index=0,
        incoming_lane_id="e0_0",
        outgoing_lane_id="e1_0",
        stop_line_xy=np.array([10.0, 5.0], dtype=np.float32),
        turn_direction="s",
    )

    selected = list(
        candidate_windows(
            _base_tracks(),
            SceneSpec.small(),
            stride=50,
            light_tokens=[light],
            min_reference_agents=2,
            require_reference_light=True,
        )
    )

    assert ("ego", 0) in selected
    assert list(
        candidate_windows(
            _base_tracks(),
            SceneSpec.small(),
            stride=50,
            light_tokens=[],
            min_reference_agents=2,
            require_reference_light=True,
        )
    ) == []


def test_light_transition_count_ignores_invalid_gaps_and_static_states():
    lights = np.zeros((2, 5, 13), dtype=np.float32)
    lights[..., -1] = -1.0
    lights[0, :, -1] = 1.0
    lights[0, :, 3:12] = -1.0
    lights[0, :2, 3] = 1.0
    lights[0, 2:, 4] = 1.0
    lights[1, [0, 2], -1] = 1.0
    lights[1, [0, 2], 3:12] = -1.0
    lights[1, 0, 3] = 1.0
    lights[1, 2, 4] = 1.0

    assert count_light_state_transitions(lights) == 1
