from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from topoworld.scenediffuserpp.denoising_evaluation import evaluate_denoising_levels
from topoworld.scenediffuserpp.evaluation import verify_checkpoint_manifest
from topoworld.scenediffuserpp.evaluation import training_log_through_checkpoint
from topoworld.scenediffuserpp.masks import behavior_prediction_mask
from topoworld.scenediffuserpp.model_data_contract import validate_scene_model_contract
from topoworld.scenediffuserpp.model_data_contract import validate_training_model_contract
from topoworld.scenediffuserpp.multi_tensor_model import ModelConfig
from topoworld.scenediffuserpp.multi_tensor_model import MultiTensorDenoiser
from topoworld.scenediffuserpp.overfit_evaluation import behavior_prediction_baselines
from topoworld.scenediffuserpp.overfit_evaluation import summarize_t1_overfit_gate
from topoworld.scenediffuserpp.storage import file_sha256
from topoworld.scenediffuserpp.storage import SceneDataset
from topoworld.scenediffuserpp.trainer import load_checkpoint
from topoworld.scenediffuserpp.trainer import SceneDiffuserTrainer
from topoworld.scenediffuserpp.trainer import TrainerConfig
from topoworld.scenediffuserpp.training_provenance import verify_resume_contract
from topoworld.scenediffuserpp.training_sampler import resolve_diagnostic_sample_ids
from topoworld.scenediffuserpp.training_sampler import select_training_indices


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the fixed-t=1 memorization gate.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--train-log", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--sample-id")
    parser.add_argument("--weights", choices=("raw", "ema"), default="raw")
    parser.add_argument("--device")
    parser.add_argument("--seeds", nargs="+", type=int, default=(7, 8))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    training = config["training"]
    model_path = (PROJECT_ROOT / config["model_config"]).resolve()
    model_values = yaml.safe_load(model_path.read_text(encoding="utf-8"))["model"]
    validate_training_model_contract(model_values, training)

    configured_ids = resolve_diagnostic_sample_ids(training)
    sample_ids = (args.sample_id,) if args.sample_id else configured_ids
    if sample_ids is None or len(sample_ids) != 1:
        raise ValueError("t=1 overfit evaluation requires exactly one diagnostic sample_id")

    manifest_path = args.dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample_indices = select_training_indices(manifest, sample_ids=sample_ids)
    dataset = SceneDataset(
        [args.dataset / name for name in sorted(manifest["shards"])]
    )
    sample = dataset[sample_indices[0]]
    tensor_contract = validate_scene_model_contract(sample, model_values)

    trainer = SceneDiffuserTrainer.create(
        MultiTensorDenoiser(
            ModelConfig(
                hidden_dim=int(model_values["hidden_dim"]),
                attention_heads=int(model_values["attention_heads"]),
                transformer_layers=int(model_values["transformer_layers"]),
                latent_queries=int(model_values["latent_queries"]),
                max_timesteps=int(model_values["timesteps"]),
            )
        ),
        config=TrainerConfig(
            learning_rate=float(training["learning_rate"]),
            beta1=float(training["beta1"]),
            weight_decay=float(training["weight_decay"]),
            gradient_clip_norm=float(training["gradient_clip_norm"]),
            ema_decay=float(training["ema_decay"]),
            precision=str(training["precision"]),
        ),
        device=args.device or str(training["device"]),
        seed=int(training["seed"]),
    )
    metadata = load_checkpoint(args.checkpoint, trainer)
    if args.weights == "ema":
        trainer.model.load_state_dict(trainer.ema.shadow)

    manifest_hash = file_sha256(manifest_path)
    verify_checkpoint_manifest(metadata, manifest_hash=manifest_hash)
    verify_resume_contract(
        metadata["run_config"],
        training=training,
        model=model_values,
        tensor_contract=tensor_contract,
    )
    log_rows = [
        json.loads(line)
        for line in args.train_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    checkpoint_log_rows = training_log_through_checkpoint(
        log_rows, checkpoint_step=int(metadata["global_step"])
    )

    device = trainer.device
    agents = torch.from_numpy(sample["agents"]).unsqueeze(0).to(device)
    lights = torch.from_numpy(sample["lights"]).unsqueeze(0).to(device)
    roadgraph = torch.from_numpy(sample["map_points"]).unsqueeze(0).to(device)
    map_padding = torch.from_numpy(sample["map_padding_mask"]).unsqueeze(0).to(device)
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
    denoising = evaluate_denoising_levels(
        trainer.model,
        agent_context=agents,
        light_context=lights,
        agent_inpaint_mask=agent_mask,
        light_inpaint_mask=light_mask,
        roadgraph=roadgraph,
        roadgraph_padding_mask=map_padding,
        noise_levels=(1.0,),
        seeds=tuple(args.seeds),
        **_topology_tensors(sample, device),
    )
    baselines = behavior_prediction_baselines(agents, history_steps=history_steps)
    gate = summarize_t1_overfit_gate(
        denoising,
        future_only_target_points=int(baselines["future_only_target_points"]),
    )
    report = {
        "status": gate["status"],
        "purpose": "wiring_and_memorization_diagnostic_not_paper_fidelity",
        "sample_id": sample_ids[0],
        "sample_index": sample_indices[0],
        "weights": args.weights,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": int(metadata["global_step"]),
        "checkpoint_manifest_sha256": str(metadata["manifest_hash"]),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": manifest_hash,
        "train_log": str(args.train_log.resolve()),
        "train_log_sha256": file_sha256(args.train_log),
        "train_log_rows_through_checkpoint": len(checkpoint_log_rows),
        "tensor_contract": tensor_contract,
        "denoising": denoising,
        "baselines": baselines,
        "gate": gate,
    }
    output_path = args.out / "t1_overfit_evaluation.json"
    output_path.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
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
