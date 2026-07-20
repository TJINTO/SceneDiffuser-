import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scenediffuserpp.trainer import SceneDiffuserTrainer
from scenediffuserpp.trainer import PaperAdafactor
from scenediffuserpp.trainer import PAPER_ADAFACTOR_DECAY_ADAM
from scenediffuserpp.trainer import TrainerConfig
from scenediffuserpp.trainer import build_optimizer
from scenediffuserpp.trainer import cuda_memory_metrics
from scenediffuserpp.trainer import load_checkpoint
from scenediffuserpp.trainer import save_checkpoint
from scenediffuserpp.trainer import seed_torch
from scenediffuserpp.trainer import train_step


class TinyDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.agent = nn.Linear(12, 12)
        self.light = nn.Linear(13, 13)

    def forward(self, **inputs):
        return SimpleNamespace(
            agent_v=self.agent(inputs["agent_z"]),
            light_v=self.light(inputs["light_z"]),
        )


class TopologyCheckingDenoiser(TinyDenoiser):
    def __init__(self):
        super().__init__()
        self.saw_topology = False

    def forward(self, **inputs):
        required = {
            "roadgraph_point_lane_index",
            "roadgraph_lane_padding_mask",
            "roadgraph_successor_index",
            "roadgraph_successor_padding_mask",
        }
        self.saw_topology = required <= inputs.keys()
        return super().forward(**inputs)


class TimeRecordingDenoiser(TinyDenoiser):
    def __init__(self):
        super().__init__()
        self.diffusion_time = None

    def forward(self, **inputs):
        self.diffusion_time = inputs["diffusion_time"].detach().clone()
        return super().forward(**inputs)


def _batch() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(27)
    agents = torch.randn(2, 3, 5, 12, generator=generator)
    lights = torch.randn(2, 2, 5, 13, generator=generator)
    agents[..., -1] = 1.0
    lights[..., -1] = 1.0
    agent_mask = torch.zeros_like(agents, dtype=torch.bool)
    light_mask = torch.zeros_like(lights, dtype=torch.bool)
    agent_mask[:, :, :2] = True
    light_mask[:, :, :2] = True
    return {
        "agents": agents,
        "lights": lights,
        "agent_inpaint_mask": agent_mask,
        "light_inpaint_mask": light_mask,
        "roadgraph": torch.randn(2, 6, 8, generator=generator),
        "roadgraph_padding_mask": torch.zeros(2, 6, dtype=torch.bool),
    }


def _trainer(model=None, seed: int = 91) -> SceneDiffuserTrainer:
    return SceneDiffuserTrainer.create(
        model or TinyDenoiser(),
        config=TrainerConfig(precision="fp32"),
        device="cpu",
        seed=seed,
    )


def test_optimizer_uses_published_explicit_hyperparameters():
    optimizer = build_optimizer(TinyDenoiser(), TrainerConfig())

    assert isinstance(optimizer, PaperAdafactor)
    assert optimizer.param_groups[0]["lr"] == 3e-4
    assert optimizer.param_groups[0]["weight_decay"] == 0.01
    assert optimizer.param_groups[0]["beta1"] == 0.9
    assert optimizer.param_groups[0]["decay_adam"] == PAPER_ADAFACTOR_DECAY_ADAM
    assert optimizer.param_groups[0]["scale_parameter"] is False
    assert optimizer.param_groups[0]["relative_step"] is False
    assert optimizer.param_groups[0]["warmup_init"] is False


def test_paper_adafactor_uses_constant_adam_decay_for_second_moment():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = PaperAdafactor(
        [parameter],
        lr=0.1,
        beta1=None,
        weight_decay=0.0,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        clip_threshold=1e9,
        decay_adam=0.5,
    )

    parameter.grad = torch.tensor([2.0])
    optimizer.step()
    parameter.grad = torch.tensor([4.0])
    optimizer.step()

    state = optimizer.state[parameter]
    torch.testing.assert_close(state["exp_avg_sq"], torch.tensor([9.0]))


def test_configured_seed_reproduces_model_parameter_initialization():
    seed_torch(20260720)
    first = TinyDenoiser()
    seed_torch(20260720)
    second = TinyDenoiser()

    assert all(
        torch.equal(first.state_dict()[name], value)
        for name, value in second.state_dict().items()
    )


def test_train_step_updates_model_and_ema_with_finite_named_losses():
    trainer = _trainer()
    before = copy.deepcopy(trainer.model.state_dict())

    metrics = train_step(trainer, _batch())

    assert metrics["total_loss"] > 0.0
    assert metrics["agent_value_count"] > 0
    assert metrics["light_validity_count"] > 0
    assert metrics["gradient_norm"] > 0.0
    assert any(
        not torch.equal(before[name], value)
        for name, value in trainer.model.state_dict().items()
    )
    assert trainer.global_step == 1


def test_cuda_memory_metrics_have_stable_keys_on_cpu():
    metrics = cuda_memory_metrics(torch.device("cpu"))

    assert metrics == {
        "cuda_memory_allocated_mib": None,
        "cuda_memory_reserved_mib": None,
        "cuda_peak_memory_allocated_mib": None,
        "cuda_peak_memory_reserved_mib": None,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_memory_metrics_report_numeric_cuda_values():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    metrics = cuda_memory_metrics(torch.device("cuda"))

    assert set(metrics) == {
        "cuda_memory_allocated_mib",
        "cuda_memory_reserved_mib",
        "cuda_peak_memory_allocated_mib",
        "cuda_peak_memory_reserved_mib",
    }
    assert all(value is not None and value >= 0.0 for value in metrics.values())


def test_train_step_reports_noise_binned_losses_for_every_sample():
    metrics = train_step(_trainer(seed=123), _batch())
    labels = ("000_025", "025_050", "050_075", "075_100")

    assert sum(int(metrics[f"noise_t_{label}_count"]) for label in labels) == 2
    for label in labels:
        count = int(metrics[f"noise_t_{label}_count"])
        loss = metrics[f"noise_t_{label}_loss"]
        if count:
            assert loss is not None
            assert torch.isfinite(torch.tensor(loss))
        else:
            assert loss is None


def test_train_step_reports_named_channel_losses():
    metrics = train_step(_trainer(seed=123), _batch())

    expected = {
        "agent_channel_x_loss",
        "agent_channel_heading_loss",
        "agent_channel_validity_loss",
        "light_channel_x_loss",
        "light_channel_state_red_loss",
        "light_channel_validity_loss",
    }
    assert expected <= metrics.keys()
    assert all(
        metrics[name] is not None
        and torch.isfinite(torch.tensor(metrics[name]))
        for name in expected
    )


def test_train_step_reports_weighted_validity_transition_metrics():
    batch = _batch()
    batch["agents"][0, 0, 2:, -1] = -1.0
    batch["lights"][0, 0, 3:, -1] = -1.0

    metrics = train_step(
        _trainer(seed=123),
        batch,
        validity_transition_weight=4.0,
    )

    assert metrics["validity_transition_weight"] == 4.0
    assert metrics["agent_validity_transition_count"] == 1
    assert metrics["light_validity_transition_count"] == 1
    assert metrics["agent_validity_weight_sum"] > metrics["agent_validity_count"]
    assert metrics["light_validity_weight_sum"] > metrics["light_validity_count"]
    assert metrics["agent_validity_transition_loss"] >= 0.0
    assert metrics["light_validity_transition_loss"] >= 0.0


def test_train_step_rejects_invalid_validity_transition_weight():
    with pytest.raises(ValueError, match="validity_transition_weight"):
        train_step(
            _trainer(seed=123),
            _batch(),
            validity_transition_weight=0.0,
        )


def test_train_step_can_skip_expensive_channel_diagnostics():
    metrics = train_step(
        _trainer(seed=123),
        _batch(),
        record_channel_metrics=False,
    )

    assert not any("_channel_" in name for name in metrics)


def test_train_step_supports_explicit_pure_noise_diagnostic_time():
    model = TimeRecordingDenoiser()

    metrics = train_step(
        _trainer(model, seed=123),
        _batch(),
        fixed_diffusion_time=1.0,
    )

    torch.testing.assert_close(model.diffusion_time, torch.ones(2))
    assert metrics["noise_t_075_100_count"] == 2
    assert metrics["noise_t_000_025_count"] == 0


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan")])
def test_train_step_rejects_invalid_fixed_diffusion_time(value: float):
    with pytest.raises(ValueError, match="fixed_diffusion_time"):
        train_step(
            _trainer(seed=123),
            _batch(),
            fixed_diffusion_time=value,
        )


def test_train_step_forwards_optional_roadgraph_topology():
    batch = _batch()
    batch.update(
        {
            "roadgraph_point_lane_index": torch.tensor(
                [[0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2]]
            ),
            "roadgraph_lane_padding_mask": torch.zeros(2, 3, dtype=torch.bool),
            "roadgraph_successor_index": torch.tensor(
                [[[0, 1], [1, 2]], [[0, 1], [1, 2]]]
            ),
            "roadgraph_successor_padding_mask": torch.zeros(
                2, 2, dtype=torch.bool
            ),
        }
    )
    model = TopologyCheckingDenoiser()
    trainer = _trainer(model)

    train_step(trainer, batch)

    assert model.saw_topology is True


def test_nonfinite_batch_aborts_and_writes_diagnostic(tmp_path: Path):
    batch = _batch()
    batch["agents"][0, 0, 0, 0] = float("nan")

    with pytest.raises(FloatingPointError):
        train_step(_trainer(), batch, diagnostic_dir=tmp_path)

    report = json.loads((tmp_path / "nonfinite_batch.json").read_text(encoding="utf-8"))
    assert report["stage"] == "input"
    assert (tmp_path / "nonfinite_batch.pt").is_file()


def test_checkpoint_resume_reproduces_next_loss_exactly(tmp_path: Path):
    torch.manual_seed(4)
    initial = TinyDenoiser()
    uninterrupted = _trainer(copy.deepcopy(initial), seed=123)
    train_step(uninterrupted, _batch())
    checkpoint = save_checkpoint(
        tmp_path / "step_1.pt",
        uninterrupted,
        manifest_hash="manifest-17",
        run_config={"name": "unit"},
    )
    expected = train_step(uninterrupted, _batch())

    resumed = _trainer(copy.deepcopy(initial), seed=999)
    metadata = load_checkpoint(checkpoint, resumed)
    actual = train_step(resumed, _batch())

    assert metadata["manifest_hash"] == "manifest-17"
    assert expected["total_loss"] == actual["total_loss"]
    assert resumed.global_step == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_bf16_train_step_computes_sparse_loss_in_fp32():
    trainer = SceneDiffuserTrainer.create(
        TinyDenoiser(),
        config=TrainerConfig(precision="bf16"),
        device="cuda",
        seed=5,
    )

    metrics = train_step(trainer, _batch())

    assert metrics["total_loss"] > 0.0
    assert torch.isfinite(torch.tensor(metrics["total_loss"]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_checkpoint_restores_rng_state_from_cpu_byte_tensor(tmp_path: Path):
    trainer = SceneDiffuserTrainer.create(
        TinyDenoiser(),
        config=TrainerConfig(precision="bf16"),
        device="cuda",
        seed=5,
    )
    train_step(trainer, _batch())
    path = save_checkpoint(
        tmp_path / "cuda.pt", trainer, manifest_hash="hash", run_config={}
    )
    resumed = SceneDiffuserTrainer.create(
        TinyDenoiser(),
        config=TrainerConfig(precision="bf16"),
        device="cuda",
        seed=6,
    )

    load_checkpoint(path, resumed)

    assert resumed.global_step == 1
