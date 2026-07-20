import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import yaml

from topoworld.scenediffuserpp.multi_tensor_model import ModelConfig
from topoworld.scenediffuserpp.multi_tensor_model import MultiTensorDenoiser
from topoworld.scenediffuserpp.roadgraph import LaneToken
from topoworld.scenediffuserpp.roadgraph import Roadgraph
from topoworld.scenediffuserpp.scene_builder import AgentState
from topoworld.scenediffuserpp.scene_builder import build_window
from topoworld.scenediffuserpp.schema import SceneSpec
from topoworld.scenediffuserpp.storage import file_sha256
from topoworld.scenediffuserpp.storage import write_shard
from topoworld.scenediffuserpp.trainer import save_checkpoint
from topoworld.scenediffuserpp.trainer import SceneDiffuserTrainer
from topoworld.scenediffuserpp.trainer import TrainerConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_heldout_cli_runs_two_scenes_and_two_seeds(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    shard = write_shard(
        dataset_dir / "shard_00000.h5",
        [_scene_window("val-0"), _scene_window("val-1")],
        max_map_points=8,
    )
    manifest = {
        "schema_version": "scenediffuserpp-sumo-v1",
        "samples": [
            _manifest_row("val-0", sample_index=0),
            _manifest_row("val-1", sample_index=1),
        ],
        "shards": {shard.name: file_sha256(shard)},
    }
    manifest_path = dataset_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    model_path = tmp_path / "model.yaml"
    model_values = {
        "maximum_agents": 32,
        "maximum_lights": 32,
        "timesteps": 91,
        "history_steps": 11,
        "latent_queries": 4,
        "hidden_dim": 16,
        "transformer_layers": 1,
        "attention_heads": 4,
        "sampling_steps": 2,
        "sampler": "paper_renoise",
        "sampling_time_grid": "linear",
        "validity_mode": "signed_stable",
    }
    model_path.write_text(
        yaml.safe_dump({"model": model_values}), encoding="utf-8"
    )
    train_path = tmp_path / "train.yaml"
    training = {
        "device": "cpu",
        "precision": "fp32",
        "seed": 17,
        "learning_rate": 0.0003,
        "beta1": 0.9,
        "weight_decay": 0.01,
        "gradient_clip_norm": 1.0,
        "ema_decay": 0.999,
        "history_steps": 11,
    }
    train_path.write_text(
        yaml.safe_dump(
            {"model_config": str(model_path), "training": training}
        ),
        encoding="utf-8",
    )
    trainer = SceneDiffuserTrainer.create(
        MultiTensorDenoiser(
            ModelConfig(
                hidden_dim=16,
                attention_heads=4,
                transformer_layers=1,
                latent_queries=4,
                max_timesteps=91,
            )
        ),
        config=TrainerConfig(precision="fp32"),
        device="cpu",
        seed=17,
    )
    trainer.global_step = 20
    checkpoint = save_checkpoint(
        tmp_path / "step_20.pt",
        trainer,
        manifest_hash=file_sha256(manifest_path),
        run_config={},
    )
    train_log = tmp_path / "train.jsonl"
    train_log.write_text(
        "".join(
            json.dumps(
                {"global_step": step, "total_loss": 1.0 if step <= 10 else 0.5}
            )
            + "\n"
            for step in range(1, 21)
        ),
        encoding="utf-8",
    )
    eval_path = tmp_path / "eval.yaml"
    eval_path.write_text(
        yaml.safe_dump(
            {
                "evaluation": {
                    "split": "validation",
                    "minimum_samples": 2,
                    "seeds": [7, 8],
                    "weights": "ema",
                    "sampling_steps": 2,
                    "validity_mode": "signed_stable",
                    "max_speed_mps": 40.0,
                    "max_acceleration_mps2": 15.0,
                    "max_jerk_mps3": 100.0,
                }
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "evaluation"
    environment = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/scenediffuserpp/evaluate_heldout.py"),
            "--train-config",
            str(train_path),
            "--eval-config",
            str(eval_path),
            "--dataset",
            str(dataset_dir),
            "--checkpoint",
            str(checkpoint),
            "--train-log",
            str(train_log),
            "--out",
            str(output_dir),
            "--device",
            "cpu",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )

    assert result.returncode in {0, 2}, result.stderr
    report = json.loads(
        (output_dir / "heldout_evaluation.json").read_text(encoding="utf-8")
    )
    assert report["record_count"] == 4
    assert report["sample_count"] == 2
    assert report["run_count"] == 1
    assert {(row["sample_id"], row["seed"]) for row in report["records"]} == {
        ("val-0", 7),
        ("val-0", 8),
        ("val-1", 7),
        ("val-1", 8),
    }
    assert report["provenance"]["manifest_sha256"] == file_sha256(manifest_path)
    assert report["provenance"]["weights"] == "ema"
    assert report["provenance"]["sampler"] == "paper_renoise"
    assert report["provenance"]["sampling_time_grid"] == "linear"
    assert report["provenance"]["tensor_contract"]["agents"] == [32, 91, 12]
    assert report["provenance"]["speed_constraint"]["max_acceleration_mps2"] == 15.0
    assert report["provenance"]["speed_constraint"]["max_jerk_mps3"] == 100.0

    short_output_dir = tmp_path / "short-evaluation"
    short_result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/scenediffuserpp/evaluate_short.py"),
            "--config",
            str(train_path),
            "--dataset",
            str(dataset_dir),
            "--checkpoint",
            str(checkpoint),
            "--train-log",
            str(train_log),
            "--out",
            str(short_output_dir),
            "--weights",
            "raw",
            "--sampling-steps",
            "2",
            "--seed",
            "7",
            "--max-speed-mps",
            "40",
            "--max-acceleration-mps2",
            "15",
            "--max-jerk-mps3",
            "100",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )

    assert short_result.returncode in {0, 2}, short_result.stderr
    short_report = json.loads(
        (short_output_dir / "short_evaluation.json").read_text(encoding="utf-8")
    )
    denoising = short_report["denoising_diagnostics"]
    assert short_report["tensor_contract"]["lights"] == [32, 91, 13]
    assert short_report["manifest_sha256"] == file_sha256(manifest_path)
    assert short_report["train_log_sha256"] == file_sha256(train_log)
    assert len(short_report["sampling_trace"]) == 2
    assert short_report["sampling_trace"][0]["step"] == 0
    assert short_report["sampling_trace"][-1]["next_diffusion_time"] == 0.0
    assert short_report["speed_constraint"]["max_acceleration_mps2"] == 15.0
    assert short_report["speed_constraint"]["max_jerk_mps3"] == 100.0
    assert (
        "agent_unknown_valid_probability_above_half"
        in short_report["sampling_trace"][0]
    )
    assert "light_unknown_valid_probability_p50" in short_report["sampling_trace"][0]
    assert denoising["seeds"] == [7, 8]
    assert set(denoising["levels"]) == {
        "1.000000",
        "0.990000",
        "0.900000",
        "0.750000",
        "0.500000",
        "0.250000",
    }
    assert "validity_balanced_accuracy" in denoising["levels"]["1.000000"][
        "agents"
    ]

    mismatched_model_path = tmp_path / "paper-capacity-model.yaml"
    mismatched_model_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    **model_values,
                    "maximum_agents": 128,
                    "maximum_lights": 64,
                }
            }
        ),
        encoding="utf-8",
    )
    mismatched_train_path = tmp_path / "mismatched-train.yaml"
    mismatched_train_path.write_text(
        yaml.safe_dump(
            {
                "model_config": str(mismatched_model_path),
                "training": training,
            }
        ),
        encoding="utf-8",
    )
    mismatch_result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/scenediffuserpp/evaluate_short.py"),
            "--config",
            str(mismatched_train_path),
            "--dataset",
            str(dataset_dir),
            "--checkpoint",
            str(checkpoint),
            "--train-log",
            str(train_log),
            "--out",
            str(tmp_path / "mismatched-evaluation"),
            "--weights",
            "raw",
            "--sampling-steps",
            "2",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )

    assert mismatch_result.returncode == 1
    assert "agents shape mismatch" in mismatch_result.stderr

    manifest["tampered_after_training"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_mismatch_result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/scenediffuserpp/evaluate_short.py"),
            "--config",
            str(train_path),
            "--dataset",
            str(dataset_dir),
            "--checkpoint",
            str(checkpoint),
            "--train-log",
            str(train_log),
            "--out",
            str(tmp_path / "manifest-mismatched-evaluation"),
            "--weights",
            "raw",
            "--sampling-steps",
            "2",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )

    assert manifest_mismatch_result.returncode == 1
    assert "different dataset manifest" in manifest_mismatch_result.stderr


def _manifest_row(sample_id: str, *, sample_index: int) -> dict:
    return {
        "sample_id": sample_id,
        "run_id": "validation-run",
        "split": "validation",
        "frequency_hz": 10,
        "shard": "shard_00000.h5",
        "sample_index": sample_index,
    }


def _scene_window(sample_id: str):
    tracks = {
        "ego": {
            step: AgentState(
                step * 0.5, 0.0, 0.0, 0.0, 5.0, 4.5, 2.0, 1.75, "car"
            )
            for step in range(91)
        },
        "car": {
            step: AgentState(
                step * 0.4, 5.0, 0.0, 0.0, 4.0, 4.5, 2.0, 1.75, "car"
            )
            for step in range(91)
        },
    }
    xy = np.array([[0.0, 0.0], [100.0, 0.0]], dtype=np.float32)
    roadgraph = Roadgraph(
        lane_tokens=(
            LaneToken(
                lane_id="e0_0",
                edge_id="e0",
                xy=xy,
                tangent=np.tile([1.0, 0.0], (2, 1)).astype(np.float32),
                speed_limit_mps=13.9,
                lane_type="arterial",
                signalized=False,
            ),
        ),
        successors={"e0_0": ()},
        light_tokens=(),
    )
    return build_window(
        tracks,
        av_id="ego",
        start_step=0,
        spec=SceneSpec.small(),
        roadgraph=roadgraph,
        meta={"sample_id": sample_id, "run_id": "validation-run"},
    )
