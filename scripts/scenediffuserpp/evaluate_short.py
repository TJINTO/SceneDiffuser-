from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from topoworld.scenediffuserpp.diagnostics import plot_scene
from topoworld.scenediffuserpp.denoising_evaluation import evaluate_denoising_levels
from topoworld.scenediffuserpp.evaluation import verify_checkpoint_manifest
from topoworld.scenediffuserpp.evaluation import verify_training_log
from topoworld.scenediffuserpp.masks import behavior_prediction_mask
from topoworld.scenediffuserpp.model_data_contract import validate_scene_model_contract
from topoworld.scenediffuserpp.model_data_contract import validate_training_model_contract
from topoworld.scenediffuserpp.multi_tensor_model import ModelConfig
from topoworld.scenediffuserpp.multi_tensor_model import MultiTensorDenoiser
from topoworld.scenediffuserpp.sampler import sample_scene
from topoworld.scenediffuserpp.short_evaluation import summarize_short_evaluation
from topoworld.scenediffuserpp.storage import file_sha256
from topoworld.scenediffuserpp.storage import SceneDataset
from topoworld.scenediffuserpp.trainer import SceneDiffuserTrainer
from topoworld.scenediffuserpp.trainer import TrainerConfig
from topoworld.scenediffuserpp.trainer import load_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the SceneDiffuser++ short generation gate.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--train-log", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--weights", choices=("raw", "ema"), default="ema")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--sampling-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-speed-mps", type=float)
    parser.add_argument("--max-acceleration-mps2", type=float)
    parser.add_argument("--max-jerk-mps3", type=float)
    parser.add_argument("--speed-limit-margin", type=float, default=1.0)
    parser.add_argument("--timestep-seconds", type=float, default=0.1)
    parser.add_argument(
        "--validity-mode", choices=("paper", "signed_stable")
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    training = config["training"]
    model_path = (PROJECT_ROOT / config["model_config"]).resolve()
    model_values = yaml.safe_load(model_path.read_text(encoding="utf-8"))["model"]
    validate_training_model_contract(model_values, training)
    validity_mode = args.validity_mode or str(
        model_values.get("validity_mode", "paper")
    )
    sampler = str(model_values.get("sampler", "paper_renoise"))
    sampling_time_grid = str(
        model_values.get("sampling_time_grid", "linear")
    )
    model_config = ModelConfig(
        hidden_dim=int(model_values["hidden_dim"]),
        attention_heads=int(model_values["attention_heads"]),
        transformer_layers=int(model_values["transformer_layers"]),
        latent_queries=int(model_values["latent_queries"]),
        max_timesteps=int(model_values["timesteps"]),
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
        device=str(training["device"]),
        seed=int(training["seed"]),
    )
    metadata = load_checkpoint(args.checkpoint, trainer)
    if args.weights == "ema":
        trainer.model.load_state_dict(trainer.ema.shadow)
    log_rows = [
        json.loads(line)
        for line in args.train_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    verify_training_log(log_rows, checkpoint_step=int(metadata["global_step"]))

    manifest_path = args.dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_hash = file_sha256(manifest_path)
    verify_checkpoint_manifest(metadata, manifest_hash=manifest_hash)
    dataset = SceneDataset([args.dataset / name for name in sorted(manifest["shards"])])
    sample = dataset[args.sample_index]
    tensor_contract = validate_scene_model_contract(sample, model_values)
    device = trainer.device
    agents = torch.from_numpy(sample["agents"]).unsqueeze(0).to(device)
    lights = torch.from_numpy(sample["lights"]).unsqueeze(0).to(device)
    roadgraph = torch.from_numpy(sample["map_points"]).unsqueeze(0).to(device)
    map_padding = torch.from_numpy(sample["map_padding_mask"]).unsqueeze(0).to(device)
    topology = _topology_tensors(sample, device)
    history_steps = int(training["history_steps"])
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
    first = sample_scene(
        trainer.model,
        agent_context=agents,
        light_context=lights,
        agent_inpaint_mask=agent_mask,
        light_inpaint_mask=light_mask,
        roadgraph=roadgraph,
        roadgraph_padding_mask=map_padding,
        num_steps=args.sampling_steps,
        seed=args.seed,
        max_speed_mps=args.max_speed_mps,
        max_acceleration_mps2=args.max_acceleration_mps2,
        max_jerk_mps3=args.max_jerk_mps3,
        timestep_seconds=args.timestep_seconds,
        speed_limit_margin=args.speed_limit_margin,
        validity_mode=validity_mode,
        sampler=sampler,
        time_grid=sampling_time_grid,
        record_trace=True,
        **topology,
    )
    second = sample_scene(
        trainer.model,
        agent_context=agents,
        light_context=lights,
        agent_inpaint_mask=agent_mask,
        light_inpaint_mask=light_mask,
        roadgraph=roadgraph,
        roadgraph_padding_mask=map_padding,
        num_steps=args.sampling_steps,
        seed=args.seed + 1,
        max_speed_mps=args.max_speed_mps,
        max_acceleration_mps2=args.max_acceleration_mps2,
        max_jerk_mps3=args.max_jerk_mps3,
        timestep_seconds=args.timestep_seconds,
        speed_limit_margin=args.speed_limit_margin,
        validity_mode=validity_mode,
        sampler=sampler,
        time_grid=sampling_time_grid,
        **topology,
    )
    report = summarize_short_evaluation(
        log_rows,
        first.agents,
        second.agents,
        agents,
        first.lights,
        second.lights,
        lights,
        history_steps=history_steps,
    )
    denoising_diagnostics = evaluate_denoising_levels(
        trainer.model,
        agent_context=agents,
        light_context=lights,
        agent_inpaint_mask=agent_mask,
        light_inpaint_mask=light_mask,
        roadgraph=roadgraph,
        roadgraph_padding_mask=map_padding,
        noise_levels=(1.0, 0.99, 0.90, 0.75, 0.5, 0.25),
        seeds=(args.seed, args.seed + 1),
        **topology,
    )
    report.update(
        {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_step": metadata["global_step"],
            "weights": args.weights,
            "sampling_steps": args.sampling_steps,
            "sampler": sampler,
            "sampling_time_grid": sampling_time_grid,
            "sample_index": args.sample_index,
            "validity_mode": validity_mode,
            "tensor_contract": tensor_contract,
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": manifest_hash,
            "checkpoint_manifest_sha256": str(metadata["manifest_hash"]),
            "train_log": str(args.train_log.resolve()),
            "train_log_sha256": file_sha256(args.train_log),
            "denoising_diagnostics": denoising_diagnostics,
            "sampling_trace": list(first.trace),
            "speed_constraint": {
                "enabled": (
                    args.max_speed_mps is not None
                    or args.max_acceleration_mps2 is not None
                    or args.max_jerk_mps3 is not None
                ),
                "fallback_max_speed_mps": args.max_speed_mps,
                "max_acceleration_mps2": args.max_acceleration_mps2,
                "max_jerk_mps3": args.max_jerk_mps3,
                "road_speed_limit_margin": args.speed_limit_margin,
                "timestep_seconds": args.timestep_seconds,
            },
        }
    )
    (args.out / "short_evaluation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    np.savez_compressed(
        args.out / "generated_scene.npz",
        agents=first.agents.detach().cpu().numpy(),
        lights=first.lights.detach().cpu().numpy(),
        target_agents=agents.detach().cpu().numpy(),
        target_lights=lights.detach().cpu().numpy(),
    )
    plotted = {
        **sample,
        "agents": first.agents[0].detach().cpu().numpy(),
        "lights": first.lights[0].detach().cpu().numpy(),
    }
    plot_scene(plotted, args.out / "generated_scene.png", history_steps=history_steps)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"] == "passed" else 2


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


if __name__ == "__main__":
    raise SystemExit(main())
