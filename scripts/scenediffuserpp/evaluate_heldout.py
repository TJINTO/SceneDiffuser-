from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import replace
import json
import math
from pathlib import Path
import subprocess
import sys

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scenediffuserpp.evaluation import aggregate_evaluation_records
from scenediffuserpp.evaluation import evaluation_config_from_mapping
from scenediffuserpp.evaluation import EvaluationRecord
from scenediffuserpp.evaluation import HeldoutEvaluationConfig
from scenediffuserpp.evaluation import select_evaluation_samples
from scenediffuserpp.evaluation import verify_checkpoint_manifest
from scenediffuserpp.evaluation import verify_training_log
from scenediffuserpp.masks import behavior_prediction_mask
from scenediffuserpp.model_data_contract import validate_scene_model_contract
from scenediffuserpp.model_data_contract import validate_training_model_contract
from scenediffuserpp.multi_tensor_model import ModelConfig
from scenediffuserpp.multi_tensor_model import MultiTensorDenoiser
from scenediffuserpp.sampler import sample_scene
from scenediffuserpp.short_evaluation import summarize_short_evaluation
from scenediffuserpp.storage import file_sha256
from scenediffuserpp.storage import SceneDataset
from scenediffuserpp.trainer import load_checkpoint
from scenediffuserpp.trainer import SceneDiffuserTrainer
from scenediffuserpp.trainer import TrainerConfig


def main() -> int:
    parser = _argument_parser()
    args = parser.parse_args()
    config = _load_evaluation_config(args.eval_config, args)
    args.out.mkdir(parents=True, exist_ok=True)

    manifest_path = args.dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = select_evaluation_samples(
        manifest,
        split=config.split,
        max_runs=config.max_runs,
        minimum_samples=config.minimum_samples,
        allow_train=config.allow_train,
    )
    dataset = SceneDataset(
        [args.dataset / name for name in sorted(manifest["shards"])]
    )
    trainer, training, model_values, model_path = _load_trainer(
        args.train_config, device_override=args.device
    )
    metadata = load_checkpoint(args.checkpoint, trainer)
    manifest_hash = file_sha256(manifest_path)
    verify_checkpoint_manifest(metadata, manifest_hash=manifest_hash)
    if config.weights == "ema":
        trainer.model.load_state_dict(trainer.ema.shadow)

    validity_mode = config.validity_mode or str(
        model_values.get("validity_mode", "paper")
    )
    sampler = str(model_values.get("sampler", "paper_renoise"))
    sampling_time_grid = str(
        model_values.get("sampling_time_grid", "linear")
    )
    log_rows = _load_training_log(args.train_log)
    verify_training_log(log_rows, checkpoint_step=int(metadata["global_step"]))
    records: list[EvaluationRecord] = []
    tensor_contract: dict[str, list[int]] | None = None
    for selected_sample in selected:
        sample = dataset[selected_sample.dataset_index]
        current_contract = validate_scene_model_contract(sample, model_values)
        if tensor_contract is None:
            tensor_contract = current_contract
        elif current_contract != tensor_contract:
            raise ValueError("held-out samples do not share one tensor contract")
        frequency_hz = _sample_frequency_hz(
            selected_sample.manifest_row,
            expected_timestep_seconds=config.timestep_seconds,
        )
        agents, lights, roadgraph, map_padding = _sample_tensors(
            sample, trainer.device
        )
        topology = _topology_tensors(sample, trainer.device)
        history_steps = int(training["history_steps"])
        agent_mask, light_mask = _behavior_prediction_masks(
            agents, lights, history_steps=history_steps, device=trainer.device
        )
        generations = {
            seed: sample_scene(
                trainer.model,
                agent_context=agents,
                light_context=lights,
                agent_inpaint_mask=agent_mask,
                light_inpaint_mask=light_mask,
                roadgraph=roadgraph,
                roadgraph_padding_mask=map_padding,
                num_steps=config.sampling_steps,
                seed=seed,
                max_speed_mps=config.max_speed_mps,
                max_acceleration_mps2=config.max_acceleration_mps2,
                max_jerk_mps3=config.max_jerk_mps3,
                timestep_seconds=config.timestep_seconds,
                speed_limit_margin=config.speed_limit_margin,
                validity_mode=validity_mode,
                sampler=sampler,
                time_grid=sampling_time_grid,
                **topology,
            )
            for seed in config.seeds
        }
        for seed_index, seed in enumerate(config.seeds):
            comparison_seed = config.seeds[(seed_index + 1) % len(config.seeds)]
            generated = generations[seed]
            comparison = generations[comparison_seed]
            report = summarize_short_evaluation(
                log_rows,
                generated.agents,
                comparison.agents,
                agents,
                generated.lights,
                comparison.lights,
                lights,
                history_steps=history_steps,
                frequency_hz=frequency_hz,
                thresholds=config.thresholds,
            )
            report.update(
                {
                    "comparison_seed": comparison_seed,
                    "dataset_index": selected_sample.dataset_index,
                    "frequency_hz": frequency_hz,
                    "validity_mode": validity_mode,
                    "sampler": sampler,
                    "sampling_time_grid": sampling_time_grid,
                }
            )
            records.append(
                EvaluationRecord(
                    sample_id=selected_sample.sample_id,
                    run_id=selected_sample.run_id,
                    seed=seed,
                    report=report,
                )
            )

    report = aggregate_evaluation_records(
        records,
        provenance={
            "split": config.split,
            "allow_train": config.allow_train,
            "weights": config.weights,
            "sampling_steps": config.sampling_steps,
            "sampler": sampler,
            "sampling_time_grid": sampling_time_grid,
            "seeds": list(config.seeds),
            "validity_mode": validity_mode,
            "thresholds": asdict(config.thresholds),
            "speed_constraint": {
                "enabled": (
                    config.max_speed_mps is not None
                    or config.max_acceleration_mps2 is not None
                    or config.max_jerk_mps3 is not None
                ),
                "fallback_max_speed_mps": config.max_speed_mps,
                "max_acceleration_mps2": config.max_acceleration_mps2,
                "max_jerk_mps3": config.max_jerk_mps3,
                "road_speed_limit_margin": config.speed_limit_margin,
                "timestep_seconds": config.timestep_seconds,
            },
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_step": int(metadata["global_step"]),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "checkpoint_manifest_sha256": str(metadata["manifest_hash"]),
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": manifest_hash,
            "train_config": str(args.train_config.resolve()),
            "train_config_sha256": file_sha256(args.train_config),
            "eval_config": str(args.eval_config.resolve()),
            "eval_config_sha256": file_sha256(args.eval_config),
            "model_config": str(model_path),
            "model_config_sha256": file_sha256(model_path),
            "train_log": str(args.train_log.resolve()),
            "train_log_sha256": file_sha256(args.train_log),
            "git_revision": _git_revision(),
            "device": str(trainer.device),
            "tensor_contract": tensor_contract,
        },
        minimum_seeds_per_sample=2,
    )
    destination = args.out / "heldout_evaluation.json"
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == "passed" else 2


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run strict multi-scene SceneDiffuser++ held-out evaluation."
    )
    parser.add_argument("--train-config", required=True, type=Path)
    parser.add_argument(
        "--eval-config",
        type=Path,
        default=PROJECT_ROOT / "configs/scenediffuserpp/eval.yaml",
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--train-log", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device")
    parser.add_argument("--split")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--minimum-samples", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--weights", choices=("raw", "ema"))
    parser.add_argument("--sampling-steps", type=int)
    parser.add_argument("--validity-mode", choices=("paper", "signed_stable"))
    parser.add_argument("--max-speed-mps", type=float)
    parser.add_argument("--max-acceleration-mps2", type=float)
    parser.add_argument("--max-jerk-mps3", type=float)
    parser.add_argument("--allow-train", action="store_true")
    return parser


def _load_evaluation_config(
    path: Path, args: argparse.Namespace
) -> HeldoutEvaluationConfig:
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = evaluation_config_from_mapping(values)
    overrides = {
        "split": args.split,
        "max_runs": args.max_runs,
        "minimum_samples": args.minimum_samples,
        "seeds": None if args.seeds is None else tuple(args.seeds),
        "weights": args.weights,
        "sampling_steps": args.sampling_steps,
        "validity_mode": args.validity_mode,
        "max_speed_mps": args.max_speed_mps,
        "max_acceleration_mps2": args.max_acceleration_mps2,
        "max_jerk_mps3": args.max_jerk_mps3,
    }
    config = replace(
        config,
        **{key: value for key, value in overrides.items() if value is not None},
    )
    if args.allow_train:
        config = replace(config, allow_train=True)
    return config


def _load_trainer(
    train_config_path: Path, *, device_override: str | None
) -> tuple[SceneDiffuserTrainer, dict, dict, Path]:
    config = yaml.safe_load(train_config_path.read_text(encoding="utf-8"))
    training = config["training"]
    model_path = Path(config["model_config"])
    if not model_path.is_absolute():
        model_path = (PROJECT_ROOT / model_path).resolve()
    model_values = yaml.safe_load(model_path.read_text(encoding="utf-8"))["model"]
    validate_training_model_contract(model_values, training)
    model_config = ModelConfig(
        agent_channels=int(model_values.get("agent_channels", 12)),
        light_channels=int(model_values.get("light_channels", 13)),
        roadgraph_channels=int(model_values.get("roadgraph_channels", 8)),
        hidden_dim=int(model_values["hidden_dim"]),
        attention_heads=int(model_values["attention_heads"]),
        transformer_layers=int(model_values["transformer_layers"]),
        latent_queries=int(model_values["latent_queries"]),
        max_timesteps=int(model_values["timesteps"]),
        dropout=float(model_values.get("dropout", 0.0)),
    )
    trainer_config = TrainerConfig(
        learning_rate=float(training["learning_rate"]),
        beta1=float(training["beta1"]),
        weight_decay=float(training["weight_decay"]),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
        ema_decay=float(training["ema_decay"]),
        precision=str(training["precision"]),
    )
    trainer = SceneDiffuserTrainer.create(
        MultiTensorDenoiser(model_config),
        config=trainer_config,
        device=device_override or str(training["device"]),
        seed=int(training["seed"]),
    )
    return trainer, training, model_values, model_path


def _load_training_log(path: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("training log is empty")
    return rows


def _sample_tensors(
    sample: dict, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(sample["agents"]).unsqueeze(0).to(device),
        torch.from_numpy(sample["lights"]).unsqueeze(0).to(device),
        torch.from_numpy(sample["map_points"]).unsqueeze(0).to(device),
        torch.from_numpy(sample["map_padding_mask"]).unsqueeze(0).to(device),
    )


def _behavior_prediction_masks(
    agents: torch.Tensor,
    lights: torch.Tensor,
    *,
    history_steps: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    agent_mask = behavior_prediction_mask(
        entities=agents.shape[1],
        timesteps=agents.shape[2],
        channels=agents.shape[3],
        history=history_steps,
        device=device,
    ).unsqueeze(0)
    light_mask = behavior_prediction_mask(
        entities=lights.shape[1],
        timesteps=lights.shape[2],
        channels=lights.shape[3],
        history=history_steps,
        device=device,
    ).unsqueeze(0)
    return agent_mask, light_mask


def _topology_tensors(sample: dict, device: torch.device) -> dict[str, torch.Tensor]:
    names = {
        "map_point_lane_index": "roadgraph_point_lane_index",
        "map_lane_padding_mask": "roadgraph_lane_padding_mask",
        "map_successor_index": "roadgraph_successor_index",
        "map_successor_padding_mask": "roadgraph_successor_padding_mask",
    }
    present = [name in sample for name in names]
    if any(present) and not all(present):
        raise ValueError("scene sample has incomplete roadgraph topology")
    return {
        model_name: torch.from_numpy(sample[sample_name]).unsqueeze(0).to(device)
        for sample_name, model_name in names.items()
        if sample_name in sample
    }


def _sample_frequency_hz(
    manifest_row: dict, *, expected_timestep_seconds: float
) -> float:
    expected_frequency = 1.0 / expected_timestep_seconds
    frequency = float(manifest_row.get("frequency_hz", expected_frequency))
    if not math.isclose(frequency, expected_frequency, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(
            "dataset frequency does not match evaluation timestep: "
            f"{frequency} Hz vs {expected_frequency} Hz"
        )
    return frequency


def _git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


if __name__ == "__main__":
    raise SystemExit(main())
