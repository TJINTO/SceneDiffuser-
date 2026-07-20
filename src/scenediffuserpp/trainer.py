from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers.optimization import Adafactor as _HfAdafactor

from scenediffuserpp.diffusion import forward_noise
from scenediffuserpp.diffusion import sparse_v_loss
from scenediffuserpp.diffusion import velocity_target
from scenediffuserpp.schema import AGENT_CHANNELS
from scenediffuserpp.schema import LIGHT_CHANNELS


PAPER_ADAFACTOR_DECAY_ADAM = 0.9999


@dataclass(frozen=True)
class TrainerConfig:
    learning_rate: float = 3e-4
    beta1: float = 0.9
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    ema_decay: float = 0.999
    precision: str = "fp32"

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.beta1 <= 0.0 or self.beta1 >= 1.0:
            raise ValueError("beta1 must be within (0, 1)")
        if self.weight_decay < 0.0 or self.gradient_clip_norm <= 0.0:
            raise ValueError("weight decay must be nonnegative and gradient clip positive")
        if self.ema_decay <= 0.0 or self.ema_decay >= 1.0:
            raise ValueError("ema_decay must be within (0, 1)")
        if self.precision not in {"fp32", "bf16"}:
            raise ValueError("precision must be fp32 or bf16")


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        current = model.state_dict()
        if current.keys() != self.shadow.keys():
            raise ValueError("EMA and model state keys differ")
        for name, value in current.items():
            if torch.is_floating_point(value):
                self.shadow[name].lerp_(value.detach(), 1.0 - self.decay)
            else:
                self.shadow[name].copy_(value)

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "shadow": {name: value.clone() for name, value in self.shadow.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if float(state["decay"]) != self.decay:
            raise ValueError("EMA decay differs from checkpoint")
        if state["shadow"].keys() != self.shadow.keys():
            raise ValueError("EMA state keys differ from model")
        for name, value in state["shadow"].items():
            self.shadow[name].copy_(value.to(self.shadow[name].device))


@dataclass
class SceneDiffuserTrainer:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    ema: ExponentialMovingAverage
    config: TrainerConfig
    device: torch.device
    generator: torch.Generator
    global_step: int = 0

    @classmethod
    def create(
        cls,
        model: nn.Module,
        *,
        config: TrainerConfig,
        device: str | torch.device,
        seed: int,
    ) -> "SceneDiffuserTrainer":
        target = torch.device(device)
        if config.precision == "bf16":
            if target.type != "cuda":
                raise RuntimeError("bf16 training requires CUDA")
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("requested CUDA device does not support bf16")
        model = model.to(target)
        generator = torch.Generator(device=target).manual_seed(int(seed))
        return cls(
            model=model,
            optimizer=build_optimizer(model, config),
            ema=ExponentialMovingAverage(model, config.ema_decay),
            config=config,
            device=target,
            generator=generator,
        )


def seed_torch(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    torch.manual_seed(seed)


class PaperAdafactor(_HfAdafactor):
    """Adafactor with the fixed Adam-style decay reported by SceneDiffuser++."""

    def __init__(
        self,
        params,
        lr=None,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        scale_parameter=True,
        relative_step=True,
        warmup_init=False,
        decay_adam: float = PAPER_ADAFACTOR_DECAY_ADAM,
    ):
        if decay_adam <= 0.0 or decay_adam >= 1.0:
            raise ValueError("decay_adam must be within (0, 1)")
        super().__init__(
            params,
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            decay_rate=decay_rate,
            beta1=beta1,
            weight_decay=weight_decay,
            scale_parameter=scale_parameter,
            relative_step=relative_step,
            warmup_init=warmup_init,
        )
        for group in self.param_groups:
            group["decay_adam"] = float(decay_adam)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.dtype in {torch.float16, torch.bfloat16}:
                    gradient = gradient.float()
                if gradient.is_sparse:
                    raise RuntimeError("Adafactor does not support sparse gradients.")

                state = self.state[parameter]
                gradient_shape = gradient.shape
                factored, use_first_moment = self._get_options(
                    group, gradient_shape
                )
                if len(state) == 0:
                    state["step"] = 0
                    if use_first_moment:
                        state["exp_avg"] = torch.zeros_like(gradient)
                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(
                            gradient_shape[:-1]
                        ).to(gradient)
                        state["exp_avg_sq_col"] = torch.zeros(
                            gradient_shape[:-2] + gradient_shape[-1:]
                        ).to(gradient)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(gradient)
                    state["RMS"] = 0
                else:
                    if use_first_moment:
                        state["exp_avg"] = state["exp_avg"].to(gradient)
                    if factored:
                        state["exp_avg_sq_row"] = state["exp_avg_sq_row"].to(
                            gradient
                        )
                        state["exp_avg_sq_col"] = state["exp_avg_sq_col"].to(
                            gradient
                        )
                    else:
                        state["exp_avg_sq"] = state["exp_avg_sq"].to(gradient)

                parameter_data = parameter
                if parameter.dtype in {torch.float16, torch.bfloat16}:
                    parameter_data = parameter_data.float()

                state["step"] += 1
                state["RMS"] = self._rms(parameter_data)
                learning_rate = self._get_lr(group, state)

                decay_adam = group.get("decay_adam")
                beta2t = (
                    float(decay_adam)
                    if decay_adam is not None
                    else 1.0 - math.pow(state["step"], group["decay_rate"])
                )
                update = gradient.square() + group["eps"][0]
                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]
                    exp_avg_sq_row.mul_(beta2t).add_(
                        update.mean(dim=-1), alpha=(1.0 - beta2t)
                    )
                    exp_avg_sq_col.mul_(beta2t).add_(
                        update.mean(dim=-2), alpha=(1.0 - beta2t)
                    )
                    update = self._approx_sq_grad(
                        exp_avg_sq_row, exp_avg_sq_col
                    )
                    update.mul_(gradient)
                else:
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2t).add_(update, alpha=(1.0 - beta2t))
                    update = exp_avg_sq.rsqrt().mul_(gradient)

                update.div_(
                    (self._rms(update) / group["clip_threshold"]).clamp_(min=1.0)
                )
                update.mul_(learning_rate)

                if use_first_moment:
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(group["beta1"]).add_(
                        update, alpha=(1.0 - group["beta1"])
                    )
                    update = exp_avg

                if group["weight_decay"] != 0:
                    parameter_data.add_(
                        parameter_data, alpha=(-group["weight_decay"] * learning_rate)
                    )
                parameter_data.add_(-update)

                if parameter.dtype in {torch.float16, torch.bfloat16}:
                    parameter.copy_(parameter_data)

        return loss


def build_optimizer(model: nn.Module, config: TrainerConfig) -> PaperAdafactor:
    return PaperAdafactor(
        model.parameters(),
        lr=config.learning_rate,
        beta1=config.beta1,
        weight_decay=config.weight_decay,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        decay_adam=PAPER_ADAFACTOR_DECAY_ADAM,
    )


def reset_cuda_memory_peak(device: torch.device | str) -> None:
    target = torch.device(device)
    if target.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target)


def cuda_memory_metrics(device: torch.device | str) -> dict[str, float | None]:
    target = torch.device(device)
    keys = {
        "cuda_memory_allocated_mib": None,
        "cuda_memory_reserved_mib": None,
        "cuda_peak_memory_allocated_mib": None,
        "cuda_peak_memory_reserved_mib": None,
    }
    if target.type != "cuda":
        return keys
    index = target.index if target.index is not None else torch.cuda.current_device()
    divisor = 1024.0 * 1024.0
    return {
        "cuda_memory_allocated_mib": torch.cuda.memory_allocated(index) / divisor,
        "cuda_memory_reserved_mib": torch.cuda.memory_reserved(index) / divisor,
        "cuda_peak_memory_allocated_mib": torch.cuda.max_memory_allocated(index)
        / divisor,
        "cuda_peak_memory_reserved_mib": torch.cuda.max_memory_reserved(index)
        / divisor,
    }


def train_step(
    trainer: SceneDiffuserTrainer,
    batch: dict[str, torch.Tensor],
    *,
    diagnostic_dir: str | Path | None = None,
    fixed_diffusion_time: float | None = None,
    record_channel_metrics: bool = True,
    validity_transition_weight: float = 1.0,
) -> dict[str, float | int | None]:
    required = (
        "agents",
        "lights",
        "agent_inpaint_mask",
        "light_inpaint_mask",
        "roadgraph",
        "roadgraph_padding_mask",
    )
    missing = [name for name in required if name not in batch]
    if missing:
        raise KeyError(f"training batch is missing: {missing}")
    if not isinstance(record_channel_metrics, bool):
        raise TypeError("record_channel_metrics must be boolean")
    validity_transition_weight = float(validity_transition_weight)
    if (
        not math.isfinite(validity_transition_weight)
        or validity_transition_weight < 1.0
    ):
        raise ValueError("validity_transition_weight must be finite and at least 1.0")
    topology_names = (
        "roadgraph_point_lane_index",
        "roadgraph_lane_padding_mask",
        "roadgraph_successor_index",
        "roadgraph_successor_padding_mask",
    )
    present_topology = [name for name in topology_names if name in batch]
    if present_topology and len(present_topology) != len(topology_names):
        missing_topology = [name for name in topology_names if name not in batch]
        raise KeyError(f"training batch has incomplete topology: {missing_topology}")
    input_names = required + (topology_names if present_topology else ())
    tensors = {name: batch[name].to(trainer.device) for name in input_names}
    _ensure_finite(tensors, "input", trainer, batch, diagnostic_dir)
    agents = tensors["agents"]
    lights = tensors["lights"]
    batch_size = agents.shape[0]
    if fixed_diffusion_time is None:
        time = torch.rand(
            batch_size,
            dtype=agents.dtype,
            device=trainer.device,
            generator=trainer.generator,
        )
    else:
        fixed_time = float(fixed_diffusion_time)
        if not math.isfinite(fixed_time) or not 0.0 <= fixed_time <= 1.0:
            raise ValueError("fixed_diffusion_time must be finite and within [0, 1]")
        time = torch.full(
            (batch_size,),
            fixed_time,
            dtype=agents.dtype,
            device=trainer.device,
        )
    agent_noise = _randn_like(agents, trainer.generator)
    light_noise = _randn_like(lights, trainer.generator)
    agent_z = forward_noise(agents, agent_noise, time)
    light_z = forward_noise(lights, light_noise, time)
    agent_target = velocity_target(agents, agent_noise, time)
    light_target = velocity_target(lights, light_noise, time)

    trainer.model.train()
    trainer.optimizer.zero_grad(set_to_none=True)
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if trainer.config.precision == "bf16"
        else nullcontext()
    )
    with autocast:
        topology_inputs = {
            name: tensors[name] for name in topology_names if name in tensors
        }
        output = trainer.model(
            agent_z=agent_z,
            light_z=light_z,
            agent_context=agents,
            light_context=lights,
            agent_inpaint_mask=tensors["agent_inpaint_mask"],
            light_inpaint_mask=tensors["light_inpaint_mask"],
            roadgraph=tensors["roadgraph"],
            roadgraph_padding_mask=tensors["roadgraph_padding_mask"],
            diffusion_time=time,
            **topology_inputs,
        )
        agent_loss, agent_parts = sparse_v_loss(
            output.agent_v.float(),
            agent_target.float(),
            agents[..., -1],
            tensors["agent_inpaint_mask"],
            validity_transition_weight=validity_transition_weight,
        )
        light_loss, light_parts = sparse_v_loss(
            output.light_v.float(),
            light_target.float(),
            lights[..., -1],
            tensors["light_inpaint_mask"],
            validity_transition_weight=validity_transition_weight,
        )
        total_loss = agent_loss + light_loss
    _ensure_finite(
        {
            "agent_prediction": output.agent_v,
            "light_prediction": output.light_v,
            "agent_loss": agent_loss,
            "light_loss": light_loss,
            "total_loss": total_loss,
        },
        "forward",
        trainer,
        batch,
        diagnostic_dir,
    )
    total_loss.backward()
    gradients = {
        name: parameter.grad
        for name, parameter in trainer.model.named_parameters()
        if parameter.grad is not None
    }
    _ensure_finite(gradients, "gradient", trainer, batch, diagnostic_dir)
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        trainer.model.parameters(), trainer.config.gradient_clip_norm
    )
    if not torch.isfinite(gradient_norm):
        _abort("gradient_norm", trainer, batch, diagnostic_dir, ["gradient_norm"])
    trainer.optimizer.step()
    _ensure_finite(
        dict(trainer.model.named_parameters()),
        "parameter",
        trainer,
        batch,
        diagnostic_dir,
    )
    trainer.ema.update(trainer.model)
    _ensure_finite(
        trainer.ema.shadow,
        "ema",
        trainer,
        batch,
        diagnostic_dir,
    )
    trainer.global_step += 1
    noise_metrics = _noise_binned_loss_metrics(
        output.agent_v.detach().float(),
        agent_target.detach().float(),
        agents[..., -1],
        tensors["agent_inpaint_mask"],
        output.light_v.detach().float(),
        light_target.detach().float(),
        lights[..., -1],
        tensors["light_inpaint_mask"],
        time.detach(),
        validity_transition_weight=validity_transition_weight,
    )
    channel_metrics = {}
    if record_channel_metrics:
        channel_metrics = {
            **_channel_loss_metrics(
                output.agent_v.detach().float(),
                agent_target.detach().float(),
                agents[..., -1],
                names=AGENT_CHANNELS,
                prefix="agent",
            ),
            **_channel_loss_metrics(
                output.light_v.detach().float(),
                light_target.detach().float(),
                lights[..., -1],
                names=LIGHT_CHANNELS,
                prefix="light",
            ),
        }
    return {
        "total_loss": float(total_loss.detach().cpu()),
        "agent_loss": float(agent_loss.detach().cpu()),
        "light_loss": float(light_loss.detach().cpu()),
        "agent_value_loss": float(agent_parts["value_loss"].detach().cpu()),
        "agent_validity_loss": float(agent_parts["validity_loss"].detach().cpu()),
        "agent_validity_transition_loss": float(
            agent_parts["validity_transition_loss"].detach().cpu()
        ),
        "light_value_loss": float(light_parts["value_loss"].detach().cpu()),
        "light_validity_loss": float(light_parts["validity_loss"].detach().cpu()),
        "light_validity_transition_loss": float(
            light_parts["validity_transition_loss"].detach().cpu()
        ),
        "agent_value_count": int(agent_parts["value_count"]),
        "agent_validity_count": int(agent_parts["validity_count"]),
        "agent_validity_transition_count": int(
            agent_parts["validity_transition_count"]
        ),
        "agent_validity_weight_sum": float(agent_parts["validity_weight_sum"]),
        "light_value_count": int(light_parts["value_count"]),
        "light_validity_count": int(light_parts["validity_count"]),
        "light_validity_transition_count": int(
            light_parts["validity_transition_count"]
        ),
        "light_validity_weight_sum": float(light_parts["validity_weight_sum"]),
        "validity_transition_weight": validity_transition_weight,
        "gradient_norm": float(gradient_norm.detach().cpu()),
        "global_step": trainer.global_step,
        **noise_metrics,
        **channel_metrics,
    }


def _noise_binned_loss_metrics(
    agent_prediction: torch.Tensor,
    agent_target: torch.Tensor,
    agent_validity: torch.Tensor,
    agent_inpaint_mask: torch.Tensor,
    light_prediction: torch.Tensor,
    light_target: torch.Tensor,
    light_validity: torch.Tensor,
    light_inpaint_mask: torch.Tensor,
    time: torch.Tensor,
    *,
    validity_transition_weight: float = 1.0,
) -> dict[str, float | int | None]:
    bins = (
        ("000_025", 0.0, 0.25, False),
        ("025_050", 0.25, 0.50, False),
        ("050_075", 0.50, 0.75, False),
        ("075_100", 0.75, 1.00, True),
    )
    metrics: dict[str, float | int | None] = {}
    for label, lower, upper, include_upper in bins:
        selected = (time >= lower) & (
            (time <= upper) if include_upper else (time < upper)
        )
        count = int(selected.sum().item())
        metrics[f"noise_t_{label}_count"] = count
        if not count:
            metrics[f"noise_t_{label}_loss"] = None
            continue
        agent_loss, _ = sparse_v_loss(
            agent_prediction[selected],
            agent_target[selected],
            agent_validity[selected],
            agent_inpaint_mask[selected],
            validity_transition_weight=validity_transition_weight,
        )
        light_loss, _ = sparse_v_loss(
            light_prediction[selected],
            light_target[selected],
            light_validity[selected],
            light_inpaint_mask[selected],
            validity_transition_weight=validity_transition_weight,
        )
        metrics[f"noise_t_{label}_loss"] = float(
            (agent_loss + light_loss).detach().cpu()
        )
    return metrics


def _channel_loss_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    validity: torch.Tensor,
    *,
    names: tuple[str, ...],
    prefix: str,
) -> dict[str, float | None]:
    if prediction.shape != target.shape or prediction.shape[-1] != len(names):
        raise ValueError("channel metric tensors do not match their channel names")
    if validity.shape != prediction.shape[:-1]:
        raise ValueError("channel metric validity shape mismatch")
    squared_error = (prediction - target).square()
    valid_values = validity > 0.0
    valid_count = int(valid_values.sum().item())
    if valid_count:
        value_losses = (
            squared_error[..., :-1]
            * valid_values.unsqueeze(-1).to(squared_error.dtype)
        ).sum(dim=tuple(range(prediction.ndim - 1))) / valid_count
        value_results: list[float | None] = value_losses.detach().cpu().tolist()
    else:
        value_results = [None] * (len(names) - 1)
    validity_loss = float(squared_error[..., -1].mean().detach().cpu())
    losses = (*value_results, validity_loss)
    return {
        f"{prefix}_channel_{name}_loss": loss
        for name, loss in zip(names, losses)
    }


def save_checkpoint(
    path: str | Path,
    trainer: SceneDiffuserTrainer,
    *,
    manifest_hash: str,
    run_config: dict[str, Any],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload = {
        "model": trainer.model.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "ema": trainer.ema.state_dict(),
        "trainer_config": asdict(trainer.config),
        "global_step": trainer.global_step,
        "generator_state": trainer.generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "manifest_hash": manifest_hash,
        "run_config": run_config,
    }
    try:
        torch.save(payload, temporary)
        torch.load(temporary, map_location="cpu", weights_only=False)
        temporary.replace(destination)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return destination


def load_checkpoint(
    path: str | Path, trainer: SceneDiffuserTrainer
) -> dict[str, Any]:
    payload = torch.load(path, map_location=trainer.device, weights_only=False)
    if payload["trainer_config"] != asdict(trainer.config):
        raise ValueError("checkpoint trainer configuration differs")
    trainer.model.load_state_dict(payload["model"])
    trainer.optimizer.load_state_dict(payload["optimizer"])
    trainer.ema.load_state_dict(payload["ema"])
    trainer.global_step = int(payload["global_step"])
    trainer.generator.set_state(payload["generator_state"].cpu())
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    if trainer.device.type == "cuda" and payload["cuda_rng_states"]:
        torch.cuda.set_rng_state_all(
            [state.cpu() for state in payload["cuda_rng_states"]]
        )
    return {
        "manifest_hash": payload["manifest_hash"],
        "run_config": payload["run_config"],
        "global_step": trainer.global_step,
    }


def _ensure_finite(
    tensors: dict[str, torch.Tensor | None],
    stage: str,
    trainer: SceneDiffuserTrainer,
    batch: dict[str, torch.Tensor],
    diagnostic_dir: str | Path | None,
) -> None:
    invalid = [
        name
        for name, value in tensors.items()
        if value is not None and torch.is_floating_point(value) and not torch.isfinite(value).all()
    ]
    if invalid:
        _abort(stage, trainer, batch, diagnostic_dir, invalid)


def _abort(
    stage: str,
    trainer: SceneDiffuserTrainer,
    batch: dict[str, torch.Tensor],
    diagnostic_dir: str | Path | None,
    invalid: list[str],
) -> None:
    if diagnostic_dir is not None:
        root = Path(diagnostic_dir)
        root.mkdir(parents=True, exist_ok=True)
        tensor_path = root / "nonfinite_batch.pt"
        temporary = tensor_path.with_suffix(".pt.tmp")
        torch.save({name: value.detach().cpu() for name, value in batch.items()}, temporary)
        temporary.replace(tensor_path)
        report = {
            "stage": stage,
            "invalid": invalid,
            "global_step": trainer.global_step,
            "batch_file": tensor_path.name,
        }
        (root / "nonfinite_batch.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    raise FloatingPointError(f"nonfinite values at {stage}: {invalid}")


def _randn_like(values: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(
        values.shape,
        dtype=values.dtype,
        device=values.device,
        generator=generator,
    )
