from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from topoworld.scenediffuserpp.audit import audit_dataset
from topoworld.scenediffuserpp.audit import has_only_incomplete_corpus_failures
from topoworld.scenediffuserpp.masks import behavior_prediction_mask
from topoworld.scenediffuserpp.masks import mixed_multitensor_training_masks
from topoworld.scenediffuserpp.model_data_contract import validate_scene_model_contract
from topoworld.scenediffuserpp.model_data_contract import validate_training_model_contract
from topoworld.scenediffuserpp.multi_tensor_model import ModelConfig
from topoworld.scenediffuserpp.multi_tensor_model import MultiTensorDenoiser
from topoworld.scenediffuserpp.storage import SceneDataset
from topoworld.scenediffuserpp.trainer import SceneDiffuserTrainer
from topoworld.scenediffuserpp.trainer import TrainerConfig
from topoworld.scenediffuserpp.trainer import cuda_memory_metrics
from topoworld.scenediffuserpp.trainer import load_checkpoint
from topoworld.scenediffuserpp.trainer import reset_cuda_memory_peak
from topoworld.scenediffuserpp.trainer import save_checkpoint
from topoworld.scenediffuserpp.trainer import seed_torch
from topoworld.scenediffuserpp.trainer import train_step
from topoworld.scenediffuserpp.training_sampler import DeterministicEpochBatchSampler
from topoworld.scenediffuserpp.training_sampler import resolve_diagnostic_sample_ids
from topoworld.scenediffuserpp.training_sampler import select_training_indices
from topoworld.scenediffuserpp.training_provenance import build_checkpoint_run_config
from topoworld.scenediffuserpp.training_provenance import verify_resume_contract
from topoworld.scenediffuserpp.training_provenance import verify_training_start


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the SUMO SceneDiffuser++ reproduction.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--device")
    parser.add_argument("--precision", choices=("fp32", "bf16"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--allow-incomplete-corpus",
        action="store_true",
        help="allow explicitly labeled diagnostic training when the configured run grid is incomplete",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    audit = audit_dataset(args.dataset)
    override_eligible = has_only_incomplete_corpus_failures(audit)
    override_used = bool(args.allow_incomplete_corpus) and override_eligible
    audit["training_override"] = {
        "allow_incomplete_corpus": bool(args.allow_incomplete_corpus),
        "eligible": override_eligible,
        "used": override_used,
    }
    (args.out / "dataset_audit.json").write_text(
        json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8"
    )
    if audit["status"] != "passed" and not override_used:
        print(json.dumps(audit, indent=2), file=sys.stderr)
        return 2

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    training = config["training"]
    model_path = (PROJECT_ROOT / config["model_config"]).resolve()
    model_values = yaml.safe_load(model_path.read_text(encoding="utf-8"))["model"]
    validate_training_model_contract(model_values, training)
    diagnostic_sample_ids = resolve_diagnostic_sample_ids(training)
    dataset, indices, manifest = _load_training_split(
        args.dataset,
        sample_ids=diagnostic_sample_ids,
    )
    if not indices:
        raise RuntimeError("dataset has no training samples")
    tensor_contract = validate_scene_model_contract(dataset[indices[0]], model_values)
    model_config = ModelConfig(
        hidden_dim=int(model_values["hidden_dim"]),
        attention_heads=int(model_values["attention_heads"]),
        transformer_layers=int(model_values["transformer_layers"]),
        latent_queries=int(model_values["latent_queries"]),
        max_timesteps=int(model_values["timesteps"]),
    )
    seed_torch(int(training["seed"]))
    trainer_config = TrainerConfig(
        learning_rate=float(training["learning_rate"]),
        beta1=float(training["beta1"]),
        weight_decay=float(training["weight_decay"]),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
        ema_decay=float(training["ema_decay"]),
        precision=args.precision or str(training["precision"]),
    )
    device = args.device or str(training["device"])
    trainer = SceneDiffuserTrainer.create(
        MultiTensorDenoiser(model_config),
        config=trainer_config,
        device=device,
        seed=int(training["seed"]),
    )
    max_steps = int(args.max_steps or training["max_steps"])
    fixed_diffusion_time = training.get("diagnostic_fixed_diffusion_time")
    if fixed_diffusion_time is not None:
        fixed_diffusion_time = float(fixed_diffusion_time)
    channel_metrics_every = int(
        training.get(
            "channel_metrics_every",
            1 if fixed_diffusion_time is not None else 100,
        )
    )
    if channel_metrics_every <= 0:
        raise ValueError("channel_metrics_every must be positive")
    loss_settings = _training_loss_settings(training)
    manifest_hash = _sha256(args.dataset / "manifest.json")
    log_path = args.out / "train.jsonl"
    if args.resume:
        metadata = load_checkpoint(args.resume, trainer)
        if metadata["manifest_hash"] != manifest_hash:
            raise ValueError("resume checkpoint was trained on a different dataset manifest")
        verify_resume_contract(
            metadata["run_config"],
            training=training,
            model=model_values,
            tensor_contract=tensor_contract,
        )
        verify_training_start(log_path, checkpoint_step=trainer.global_step)
        if trainer.global_step >= max_steps:
            raise ValueError("resume checkpoint already reached the requested max steps")
    else:
        verify_training_start(log_path, checkpoint_step=None)
    batch_size = int(training["batch_size"])
    batch_sampler = DeterministicEpochBatchSampler(
        indices,
        batch_size=batch_size,
        seed=int(training["seed"]),
    )
    checkpoint_every = _resolve_checkpoint_every(
        training, override=args.checkpoint_every
    )
    for step in range(trainer.global_step, max_steps):
        selected = batch_sampler.batch(step)
        batch = _collate(
            [dataset[index] for index in selected],
            history_steps=int(training["history_steps"]),
            task=str(training["task"]),
            scene_generation_probability=float(
                training.get("scene_generation_probability", 0.5)
            ),
            control_feature_probability=float(
                training.get("control_feature_probability", 0.5)
            ),
            mask_seed=int(training["seed"]) + step,
        )
        reset_cuda_memory_peak(trainer.device)
        metrics = train_step(
            trainer,
            batch,
            diagnostic_dir=args.out / "diagnostics",
            fixed_diffusion_time=fixed_diffusion_time,
            record_channel_metrics=(
                trainer.global_step == 0
                or (trainer.global_step + 1) % channel_metrics_every == 0
            ),
            **loss_settings,
        )
        metrics.update(cuda_memory_metrics(trainer.device))
        task_flags = batch["task_is_scene_generation"]
        metrics.update(
            {
                "behavior_prediction_samples": int((~task_flags).sum().item()),
                "scene_generation_samples": int(task_flags.sum().item()),
                "agent_inpaint_fraction": float(
                    batch["agent_inpaint_mask"].float().mean().item()
                ),
                "light_inpaint_fraction": float(
                    batch["light_inpaint_mask"].float().mean().item()
                ),
                "data_epoch": batch_sampler.epoch_at_step(step),
            }
        )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, allow_nan=False) + "\n")
        if trainer.global_step % checkpoint_every == 0 or trainer.global_step == max_steps:
            save_checkpoint(
                args.out / f"step_{trainer.global_step:08d}.pt",
                trainer,
                manifest_hash=manifest_hash,
                run_config=build_checkpoint_run_config(
                    training=training,
                    model=model_values,
                    dataset=str(args.dataset.resolve()),
                    tensor_contract=tensor_contract,
                    incomplete_corpus_override=bool(
                        audit["training_override"]["used"]
                    ),
                    optimizer_provenance=config.get("optimizer_provenance"),
                    execution={
                        "effective_max_steps": max_steps,
                        "effective_checkpoint_every": checkpoint_every,
                        "resume_checkpoint": (
                            str(args.resume.resolve()) if args.resume else None
                        ),
                        "fixed_diffusion_time": fixed_diffusion_time,
                        "channel_metrics_every": channel_metrics_every,
                        **loss_settings,
                    },
                ),
            )
        print(json.dumps(metrics, allow_nan=False))
    return 0


def _load_training_split(
    root: Path,
    *,
    sample_ids: tuple[str, ...] | None = None,
) -> tuple[SceneDataset, list[int], dict]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    shard_paths = [root / name for name in sorted(manifest["shards"])]
    dataset = SceneDataset(shard_paths)
    indices = select_training_indices(manifest, sample_ids=sample_ids)
    return dataset, indices, manifest


def _collate(
    samples: list[dict],
    history_steps: int,
    *,
    task: str = "behavior_prediction",
    scene_generation_probability: float = 0.5,
    control_feature_probability: float = 0.5,
    mask_seed: int = 0,
) -> dict[str, torch.Tensor]:
    agents = torch.from_numpy(np.stack([sample["agents"] for sample in samples]))
    lights = torch.from_numpy(np.stack([sample["lights"] for sample in samples]))
    roadgraph = torch.from_numpy(np.stack([sample["map_points"] for sample in samples]))
    roadgraph_padding = torch.from_numpy(
        np.stack([sample["map_padding_mask"] for sample in samples])
    )
    if task == "behavior_prediction":
        agent_mask = behavior_prediction_mask(
            entities=agents.shape[1],
            timesteps=agents.shape[2],
            channels=agents.shape[3],
            history=history_steps,
        ).unsqueeze(0).expand(agents.shape[0], -1, -1, -1)
        light_mask = behavior_prediction_mask(
            entities=lights.shape[1],
            timesteps=lights.shape[2],
            channels=lights.shape[3],
            history=history_steps,
        ).unsqueeze(0).expand(lights.shape[0], -1, -1, -1)
        task_is_scene_generation = torch.zeros(agents.shape[0], dtype=torch.bool)
    elif task == "mixed":
        masks = mixed_multitensor_training_masks(
            agent_validity=agents[..., -1] > 0.0,
            light_validity=lights[..., -1] > 0.0,
            agent_channels=agents.shape[-1],
            light_channels=lights.shape[-1],
            history=history_steps,
            scene_generation_probability=scene_generation_probability,
            control_feature_probability=control_feature_probability,
            generator=torch.Generator().manual_seed(mask_seed),
        )
        agent_mask = masks.agent_mask
        light_mask = masks.light_mask
        task_is_scene_generation = masks.task_is_scene_generation
    else:
        raise ValueError(f"unsupported training task: {task}")
    batch = {
        "agents": agents,
        "lights": lights,
        "agent_inpaint_mask": agent_mask,
        "light_inpaint_mask": light_mask,
        "roadgraph": roadgraph,
        "roadgraph_padding_mask": roadgraph_padding,
        "task_is_scene_generation": task_is_scene_generation,
    }
    topology_keys = {
        "map_point_lane_index": "roadgraph_point_lane_index",
        "map_lane_padding_mask": "roadgraph_lane_padding_mask",
        "map_successor_index": "roadgraph_successor_index",
        "map_successor_padding_mask": "roadgraph_successor_padding_mask",
    }
    topology_counts = [
        sum(key in sample for key in topology_keys) for sample in samples
    ]
    if any(0 < count < len(topology_keys) for count in topology_counts):
        raise ValueError("scene sample has incomplete roadgraph topology")
    has_topology = [count == len(topology_keys) for count in topology_counts]
    if any(has_topology) and not all(has_topology):
        raise ValueError("cannot mix topology-aware and legacy scene shards")
    if all(has_topology):
        batch.update(
            {
                model_key: torch.from_numpy(
                    np.stack([sample[sample_key] for sample in samples])
                )
                for sample_key, model_key in topology_keys.items()
            }
        )
    return batch


def _training_loss_settings(training: dict) -> dict[str, float]:
    validity_transition_weight = float(
        training.get("validity_transition_weight", 1.0)
    )
    if (
        not math.isfinite(validity_transition_weight)
        or validity_transition_weight < 1.0
    ):
        raise ValueError("validity_transition_weight must be finite and at least 1.0")
    return {"validity_transition_weight": validity_transition_weight}


def _resolve_checkpoint_every(training: dict, *, override: int | None = None) -> int:
    value = int(training["checkpoint_every"] if override is None else override)
    if value <= 0:
        raise ValueError("checkpoint_every must be positive")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
