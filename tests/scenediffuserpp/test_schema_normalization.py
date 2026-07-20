import numpy as np
import pytest

from scenediffuserpp.normalization import AgentNormalizer
from scenediffuserpp.normalization import LightNormalizer
from scenediffuserpp.normalization import wrap_to_pi
from scenediffuserpp.schema import AGENT_CHANNELS
from scenediffuserpp.schema import LIGHT_CHANNELS
from scenediffuserpp.schema import AgentType
from scenediffuserpp.schema import dataset_build_config_from_mapping
from scenediffuserpp.schema import LightState
from scenediffuserpp.schema import SceneSpec


def test_small_scene_spec_has_paper_time_contract():
    spec = SceneSpec.small()

    assert spec.frequency_hz == 10
    assert spec.timesteps == 91
    assert spec.history_steps == 11
    assert spec.future_steps == 80
    assert spec.max_agents == 32
    assert spec.max_lights == 32
    assert spec.sampling_steps == 32
    assert spec.agent_channels == AGENT_CHANNELS
    assert spec.light_channels == LIGHT_CHANNELS
    assert spec.agent_channels[-1] == "validity"
    assert spec.light_channels[-1] == "validity"
    assert len(spec.agent_channels) == 12
    assert len(spec.light_channels) == 13


def test_paper_scene_spec_records_published_model_capacity():
    spec = SceneSpec.paper()

    assert spec.max_agents == 128
    assert spec.latent_queries == 192
    assert spec.hidden_dim == 512
    assert spec.transformer_layers == 8
    assert spec.attention_heads == 8
    assert spec.sampling_steps == 32


def test_scene_spec_rejects_inconsistent_time_partition():
    with pytest.raises(ValueError, match="history_steps plus future_steps"):
        SceneSpec(
            frequency_hz=10,
            timesteps=91,
            history_steps=10,
            future_steps=80,
            max_agents=32,
            max_lights=16,
            latent_queries=64,
            hidden_dim=128,
            transformer_layers=4,
            attention_heads=4,
            sampling_steps=16,
        )


def test_dataset_build_config_controls_shape_split_and_map_coverage():
    config = dataset_build_config_from_mapping(
        {
            "dataset": {
                "frequency_hz": 5,
                "timesteps": 31,
                "history_steps": 6,
                "future_steps": 25,
                "maximum_agents": 17,
                "maximum_lights": 9,
                "observation_radius_m": 70.0,
                "map_radius_m": 600.0,
                "map_point_spacing_m": 5.0,
                "maximum_map_points": 4096,
                "maximum_map_lanes": 512,
                "maximum_map_connections": 2048,
                "window_stride_steps": 7,
                "minimum_av_travel_m": 12.0,
                "minimum_reference_agents": 3,
                "require_reference_light": True,
                "minimum_light_state_transitions": 2,
                "shard_size": 23,
                "split_seed": 99,
            }
        }
    )

    assert config.scene_spec.frequency_hz == 5
    assert config.scene_spec.timesteps == 31
    assert config.scene_spec.max_agents == 17
    assert config.scene_spec.max_lights == 9
    assert config.observation_radius_m == 70.0
    assert config.map_radius_m == 600.0
    assert config.map_point_spacing_m == 5.0
    assert config.maximum_map_points == 4096
    assert config.maximum_map_lanes == 512
    assert config.maximum_map_connections == 2048
    assert config.window_stride_steps == 7
    assert config.minimum_reference_agents == 3
    assert config.require_reference_light is True
    assert config.minimum_light_state_transitions == 2
    assert config.shard_size == 23
    assert config.split_seed == 99


def test_dataset_build_config_keeps_observation_and_map_radii_distinct():
    config = dataset_build_config_from_mapping(
        {
            "dataset": {
                "frequency_hz": 10,
                "timesteps": 91,
                "history_steps": 11,
                "future_steps": 80,
                "maximum_agents": 32,
                "maximum_lights": 32,
                "observation_radius_m": 80.0,
                "map_radius_m": 1000.0,
            }
        }
    )

    assert config.observation_radius_m == 80.0
    assert config.map_radius_m == 1000.0
    assert config.map_point_spacing_m == 10.0


def test_dataset_build_config_rejects_map_smaller_than_observation_area():
    with pytest.raises(ValueError, match="map radius"):
        dataset_build_config_from_mapping(
            {
                "dataset": {
                    "frequency_hz": 10,
                    "timesteps": 91,
                    "history_steps": 11,
                    "future_steps": 80,
                    "maximum_agents": 32,
                    "maximum_lights": 32,
                    "observation_radius_m": 80.0,
                    "map_radius_m": 40.0,
                }
            }
        )


def test_agent_normalization_round_trips_valid_values():
    normalizer = AgentNormalizer()
    raw = np.array([12.0, -8.0, 0.0, np.pi / 2, 4.5, 2.0, 1.75])

    restored = normalizer.decode_continuous(normalizer.encode_continuous(raw))

    np.testing.assert_allclose(restored, raw, atol=1e-6)


def test_valid_agent_encodes_type_to_signed_one_hot():
    encoded = AgentNormalizer().encode_agent(
        continuous=np.array([0.0, 0.0, 0.0, 0.0, 4.5, 2.0, 1.75]),
        type_index=AgentType.CAR,
        valid=True,
    )

    np.testing.assert_array_equal(encoded[7:11], [-1.0, 1.0, -1.0, -1.0])
    assert encoded[-1] == 1.0


def test_invalid_agent_row_is_exactly_zero_except_validity():
    encoded = AgentNormalizer().encode_agent(
        continuous=np.full(7, 99.0), type_index=AgentType.CAR, valid=False
    )

    assert np.all(encoded[:-1] == 0.0)
    assert encoded[-1] == -1.0


def test_light_normalization_round_trips_valid_state():
    normalizer = LightNormalizer()
    encoded = normalizer.encode_light(
        xyz=np.array([16.0, -24.0, 0.0]),
        state=LightState.RED,
        valid=True,
    )
    xyz, state, valid = normalizer.decode_light(encoded)

    np.testing.assert_allclose(xyz, [16.0, -24.0, 0.0], atol=1e-6)
    assert state is LightState.RED
    assert valid is True


def test_wrap_to_pi_uses_half_open_interval():
    assert wrap_to_pi(np.pi) == pytest.approx(-np.pi)
    assert wrap_to_pi(-np.pi) == pytest.approx(-np.pi)
    assert wrap_to_pi(3 * np.pi) == pytest.approx(-np.pi)
