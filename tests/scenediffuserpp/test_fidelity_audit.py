import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from scenediffuserpp.fidelity_audit import CLASSIFICATIONS
from scenediffuserpp.fidelity_audit import audit_to_json
from scenediffuserpp.fidelity_audit import audit_to_markdown
from scenediffuserpp.fidelity_audit import build_fidelity_audit
from scenediffuserpp.fidelity_audit import write_audit_artifacts
from scenediffuserpp.schema import dataset_build_config_from_mapping
from scenediffuserpp.storage import assign_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_IDS = {
    "source_data",
    "frequency_horizon",
    "agent_capacity",
    "light_capacity",
    "executed_agent_capacity",
    "hidden_size",
    "latent_queries",
    "transformer_shape",
    "context_encoder",
    "conditioning_fusion",
    "adaln_backbone",
    "scene_tensors",
    "sparse_loss",
    "task_mixture",
    "control_mask",
    "optimizer",
    "batch_training_steps",
    "training_exposure",
    "ema_selection",
    "sampling_steps",
    "sampling_transition",
    "paper_soft_clip",
    "signed_stable_clip",
    "long_rollout",
    "paper_metrics",
    "held_out_evaluation",
    "high_noise_denoising",
    "feature_normalization",
    "coordinate_frame",
    "roadgraph_tokenization",
    "diffusion_time_schedule",
    "dataset_split",
    "planner_world_separation",
}


def test_fidelity_audit_has_required_evidenced_entries():
    entries = build_fidelity_audit()
    by_id = {entry.id: entry for entry in entries}

    assert set(by_id) == REQUIRED_IDS
    assert len(entries) == len(by_id)
    for entry in entries:
        assert entry.classification in CLASSIFICATIONS
        assert entry.published.strip()
        assert entry.local.strip()
        assert entry.evidence
        assert all(item.strip() for item in entry.evidence)


def test_fidelity_audit_separates_published_renoise_from_interpreted_time_grid():
    by_id = {entry.id: entry for entry in build_fidelity_audit()}

    assert by_id["sampling_transition"].classification == "matched"
    assert "re-noise" in by_id["sampling_transition"].local
    assert by_id["diffusion_time_schedule"].classification == "interpreted"
    assert "linear" in by_id["diffusion_time_schedule"].local


def test_fidelity_audit_does_not_call_width_only_probe_paper_scale():
    by_id = {entry.id: entry for entry in build_fidelity_audit()}

    assert by_id["executed_agent_capacity"].classification == "reduced_scale"
    assert "32" in by_id["executed_agent_capacity"].local
    assert by_id["training_exposure"].classification == "reduced_scale"
    assert "5,000" in by_id["training_exposure"].local
    assert by_id["light_capacity"].classification == "interpreted"
    assert by_id["conditioning_fusion"].classification == "interpreted"
    assert by_id["sparse_loss"].classification == "interpreted"
    assert by_id["ema_selection"].classification == "missing"
    assert by_id["paper_metrics"].classification == "missing"
    assert by_id["high_noise_denoising"].classification == "reduced_scale"
    assert "9.87 m" in by_id["high_noise_denoising"].local
    assert "blocked" in by_id["dataset_split"].local


def test_fidelity_audit_renderers_preserve_ids_and_provenance():
    entries = build_fidelity_audit()

    payload = audit_to_json(entries, git_revision="abc123")
    markdown = audit_to_markdown(entries, git_revision="abc123")

    assert payload["schema_version"] == "scenediffuserpp-fidelity-v1"
    assert payload["git_revision"] == "abc123"
    assert {row["id"] for row in payload["entries"]} == REQUIRED_IDS
    assert "abc123" in markdown
    for entry_id in REQUIRED_IDS:
        assert f"`{entry_id}`" in markdown


def test_fidelity_artifacts_refuse_overwrite_without_force(tmp_path: Path):
    json_path = tmp_path / "audit.json"
    markdown_path = tmp_path / "audit.md"

    write_audit_artifacts(
        json_path,
        markdown_path,
        git_revision="first",
        force=False,
    )
    with pytest.raises(FileExistsError, match="already exists"):
        write_audit_artifacts(
            json_path,
            markdown_path,
            git_revision="second",
            force=False,
        )
    write_audit_artifacts(
        json_path,
        markdown_path,
        git_revision="second",
        force=True,
    )

    assert json.loads(json_path.read_text(encoding="utf-8"))["git_revision"] == "second"
    assert "second" in markdown_path.read_text(encoding="utf-8")


def test_model_configs_default_to_the_paper_sampler_and_validity_mode():
    small = yaml.safe_load(
        (PROJECT_ROOT / "configs/scenediffuserpp/model_small.yaml").read_text(
            encoding="utf-8"
        )
    )
    paper = yaml.safe_load(
        (PROJECT_ROOT / "configs/scenediffuserpp/model_paper.yaml").read_text(
            encoding="utf-8"
        )
    )
    reduced_paper = yaml.safe_load(
        (
            PROJECT_ROOT
            / "configs/scenediffuserpp/model_reduced_paper.yaml"
        ).read_text(encoding="utf-8")
    )
    large_32 = yaml.safe_load(
        (
            PROJECT_ROOT
            / "configs/scenediffuserpp/model_large_32.yaml"
        ).read_text(encoding="utf-8")
    )
    train_small = yaml.safe_load(
        (PROJECT_ROOT / "configs/scenediffuserpp/train_small.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert small["model"]["validity_mode"] == "paper"
    assert paper["model"]["validity_mode"] == "paper"
    assert reduced_paper["model"]["validity_mode"] == "paper"
    for values in (small, paper, reduced_paper, large_32):
        assert values["model"]["sampler"] == "paper_renoise"
        assert values["model"]["sampling_time_grid"] == "linear"
        assert values["model"]["validity_mode"] == "paper"
        assert (
            values["model"]["conditioning_fusion"]
            == "noisy_local_global_adaln"
        )
    assert paper["model"]["maximum_agents"] == 128
    assert large_32["model"]["maximum_agents"] == 32
    assert large_32["model"]["hidden_dim"] == 512
    assert (
        train_small["model_config"]
        == "configs/scenediffuserpp/model_reduced_paper.yaml"
    )
    for config_name in ("train_large_32_probe.yaml", "train_large_32_smoke.yaml"):
        values = yaml.safe_load(
            (PROJECT_ROOT / "configs/scenediffuserpp" / config_name).read_text(
                encoding="utf-8"
            )
        )
        assert values["model_config"] == "configs/scenediffuserpp/model_large_32.yaml"
    for legacy_name in (
        "train_paper_scale_probe.yaml",
        "train_paper_scale_reduced.yaml",
        "train_paper_scale_smoke.yaml",
    ):
        assert not (PROJECT_ROOT / "configs/scenediffuserpp" / legacy_name).exists()


def test_nanjing_data_config_preserves_local_observation_and_long_map_context():
    values = yaml.safe_load(
        (PROJECT_ROOT / "configs/scenediffuserpp/data_nanjing_10hz.yaml").read_text(
            encoding="utf-8"
        )
    )

    config = dataset_build_config_from_mapping(values)

    assert config.observation_radius_m == 80.0
    assert config.map_radius_m == 1000.0
    assert config.map_point_spacing_m == 10.0
    assert config.maximum_map_points == 12288
    assert config.maximum_map_lanes == 1024
    assert config.maximum_map_connections == 4096
    assert config.minimum_reference_agents == 2


def test_nanjing_run_grid_has_stratified_nonempty_grouped_splits():
    values = yaml.safe_load(
        (PROJECT_ROOT / "configs/scenediffuserpp/data_nanjing_10hz.yaml").read_text(
            encoding="utf-8"
        )
    )
    config = dataset_build_config_from_mapping(values)
    runs = [
        (float(period), f"period_{float(period):g}_seed_{int(seed)}")
        for period in values["runs"]["departure_period_s"]
        for seed in values["runs"]["seeds"]
    ]
    assignments = {
        run_id: assign_split(run_id, seed=config.split_seed)
        for _period, run_id in runs
    }

    assert Counter(assignments.values()) == {
        "train": 18,
        "validation": 3,
        "test": 3,
    }
    for split in ("validation", "test"):
        assert {
            period for period, run_id in runs if assignments[run_id] == split
        } == {0.5, 1.0, 2.0}
