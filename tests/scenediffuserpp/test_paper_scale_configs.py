from pathlib import Path

import yaml

from scenediffuserpp.schema import dataset_build_config_from_mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_paper_scale_dataset_config_matches_paper_model_capacity():
    data_path = PROJECT_ROOT / "configs/scenediffuserpp/data_nanjing_10hz_paper.yaml"
    model_path = PROJECT_ROOT / "configs/scenediffuserpp/model_paper.yaml"

    data_config = dataset_build_config_from_mapping(
        yaml.safe_load(data_path.read_text(encoding="utf-8"))
    )
    model_config = yaml.safe_load(model_path.read_text(encoding="utf-8"))["model"]

    assert data_config.scene_spec.max_agents == model_config["maximum_agents"] == 128
    assert data_config.scene_spec.max_lights == model_config["maximum_lights"] == 64
    assert data_config.scene_spec.timesteps == model_config["timesteps"] == 91


def test_paper_scale_dataset_config_requires_signal_dynamics():
    data_path = PROJECT_ROOT / "configs/scenediffuserpp/data_nanjing_10hz_paper.yaml"

    data_config = dataset_build_config_from_mapping(
        yaml.safe_load(data_path.read_text(encoding="utf-8"))
    )

    assert data_config.minimum_light_state_transitions >= 1


def test_paper128_bp_diagnostic_config_keeps_full_paper_model_capacity():
    train_path = (
        PROJECT_ROOT / "configs/scenediffuserpp/train_paper128_bp_batch2_short.yaml"
    )

    config = yaml.safe_load(train_path.read_text(encoding="utf-8"))
    model_path = PROJECT_ROOT / config["model_config"]
    model_config = yaml.safe_load(model_path.read_text(encoding="utf-8"))["model"]

    assert config["training"]["task"] == "behavior_prediction"
    assert config["training"]["batch_size"] == 2
    assert model_config["maximum_agents"] == 128
    assert model_config["maximum_lights"] == 64
    assert model_config["hidden_dim"] == 512
    assert model_config["transformer_layers"] == 8
    assert model_config["latent_queries"] == 192


def test_dense_paper_scale_dataset_config_uses_wider_observation_without_scale_overflow():
    data_path = (
        PROJECT_ROOT / "configs/scenediffuserpp/data_nanjing_10hz_paper_dense500.yaml"
    )

    data_config = dataset_build_config_from_mapping(
        yaml.safe_load(data_path.read_text(encoding="utf-8"))
    )

    assert data_config.observation_radius_m == 500.0
    assert data_config.map_radius_m == 1000.0
    assert data_config.minimum_reference_agents == 4
    assert data_config.scene_spec.max_agents == 128
