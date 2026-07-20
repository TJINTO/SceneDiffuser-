from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence


SCHEMA_VERSION = "scenediffuserpp-fidelity-v1"
PAPER_REFERENCE = (
    "https://openaccess.thecvf.com/content/CVPR2025/html/"
    "Tan_SceneDiffuser_City-Scale_Traffic_Simulation_via_a_Generative_"
    "World_Model_CVPR_2025_paper.html"
)
CLASSIFICATIONS = frozenset(
    {
        "matched",
        "reduced_scale",
        "interpreted",
        "intentional_deviation",
        "missing",
        "not_evaluated",
    }
)


@dataclass(frozen=True)
class FidelityEntry:
    id: str
    area: str
    published: str
    local: str
    classification: str
    evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.id or not self.area or not self.published or not self.local:
            raise ValueError("fidelity entry text fields cannot be empty")
        if self.classification not in CLASSIFICATIONS:
            raise ValueError(f"unsupported fidelity classification: {self.classification}")
        if not self.evidence or any(not item for item in self.evidence):
            raise ValueError("fidelity entries require nonempty evidence")


def build_fidelity_audit() -> tuple[FidelityEntry, ...]:
    return (
        _entry(
            "source_data",
            "Data",
            "WOMD scenes augmented with enlarged roadgraph regions.",
            "Synthetic 10 Hz SUMO trajectories from a Nanjing OSM network.",
            "intentional_deviation",
            "src/topoworld/scenediffuserpp/sumo_export.py",
            "src/topoworld/scenediffuserpp/scene_builder.py",
        ),
        _entry(
            "frequency_horizon",
            "Data",
            "10 Hz scenes with 91 physical timesteps.",
            "10 Hz scenes with 91 physical timesteps.",
            "matched",
            "configs/scenediffuserpp/data_nanjing_10hz.yaml",
            "src/topoworld/scenediffuserpp/schema.py:SceneSpec",
        ),
        _entry(
            "agent_capacity",
            "Scale",
            "128 jointly modeled agents.",
            "32 agents in the small run; 128 recorded in the paper config.",
            "reduced_scale",
            "configs/scenediffuserpp/model_small.yaml",
            "configs/scenediffuserpp/model_paper.yaml",
        ),
        _entry(
            "light_capacity",
            "Scale",
            "The maximum jointly modeled traffic-light count is not disclosed.",
            "The clean-room paper-shaped config chooses 64 light slots, while the completed diagnostic executes 32; neither value is a verified published hyperparameter.",
            "interpreted",
            "configs/scenediffuserpp/model_paper.yaml",
            "configs/scenediffuserpp/model_large_32.yaml",
        ),
        _entry(
            "executed_agent_capacity",
            "Scale",
            "The reported model trains and runs inference with all 128 agents.",
            "The strongest completed width-matched diagnostic still consumes HDF5 tensors with 32 agent slots and 32 light slots.",
            "reduced_scale",
            "configs/scenediffuserpp/data_nanjing_10hz.yaml",
            "src/topoworld/scenediffuserpp/schema.py:SceneSpec.small",
            "docs/scenediffuserpp-nanjing-reproduction-status.md",
        ),
        _entry(
            "hidden_size",
            "Scale",
            "Hidden dimension 512.",
            "Hidden dimension 128 small; 512 paper config.",
            "reduced_scale",
            "configs/scenediffuserpp/model_small.yaml",
            "configs/scenediffuserpp/model_paper.yaml",
        ),
        _entry(
            "latent_queries",
            "Scale",
            "192 scene-context latent queries.",
            "64 small; 192 paper config.",
            "reduced_scale",
            "configs/scenediffuserpp/model_small.yaml",
            "configs/scenediffuserpp/model_paper.yaml",
        ),
        _entry(
            "transformer_shape",
            "Scale",
            "Eight Transformer layers and eight attention heads.",
            "Four layers/heads small; eight layers/heads paper config.",
            "reduced_scale",
            "configs/scenediffuserpp/model_small.yaml",
            "configs/scenediffuserpp/model_paper.yaml",
        ),
        _entry(
            "context_encoder",
            "Architecture",
            "SceneDiffuser Perceiver-style global context encoder.",
            "Clean-room encoder uses one learned-query cross-attention stage after lane pooling and predecessor message passing; exact Perceiver IO depth is unavailable without official code.",
            "interpreted",
            "src/topoworld/scenediffuserpp/context_encoder.py:RoadgraphEncoder",
        ),
        _entry(
            "conditioning_fusion",
            "Architecture",
            "The SceneDiffuser diagram fuses noisy scene tokens with local and global context before AdaLN-Zero conditioning.",
            "The clean-room model now projects noisy and inpainted-local tokens together before cross-attending roadgraph context for AdaLN conditioning; projection depth and exact fusion remain interpreted without official code.",
            "interpreted",
            "src/topoworld/scenediffuserpp/multi_tensor_model.py:MultiTensorDenoiser",
            "tests/scenediffuserpp/test_multi_tensor_model.py:test_noisy_scene_tokens_enter_global_context_fusion_query",
        ),
        _entry(
            "adaln_backbone",
            "Architecture",
            "SceneDiffuser axial Transformer conditioned through AdaLN-Zero.",
            "Clean-room entity/time axial attention with AdaLN-Zero modulation.",
            "interpreted",
            "src/topoworld/scenediffuserpp/axial_transformer.py:AxialTransformerBlock",
            "src/topoworld/scenediffuserpp/multi_tensor_model.py:MultiTensorDenoiser",
        ),
        _entry(
            "scene_tensors",
            "Representation",
            "Joint agent and traffic-light scene tensors.",
            "Joint agent and traffic-light scene tensors.",
            "matched",
            "src/topoworld/scenediffuserpp/normalization.py",
            "src/topoworld/scenediffuserpp/multi_tensor_model.py:DenoiserOutput",
        ),
        _entry(
            "feature_normalization",
            "Representation",
            "Positions use 1/80 scaling; dimensions use published means and twice the standard deviations; categorical fields are normalized.",
            "Position and dimension constants match; heading and signed categorical encoding are clean-room interpretations.",
            "interpreted",
            "src/topoworld/scenediffuserpp/normalization.py",
            "tests/scenediffuserpp/test_schema_normalization.py",
        ),
        _entry(
            "coordinate_frame",
            "Representation",
            "Scene tensors are expressed in the AV frame immediately before simulation.",
            "Each window uses a fixed AV-centric reference pose at the history/future boundary.",
            "matched",
            "src/topoworld/scenediffuserpp/scene_builder.py:SceneWindow",
            "src/topoworld/scenediffuserpp/storage.py:reference_world_pose",
        ),
        _entry(
            "roadgraph_tokenization",
            "Representation",
            "Vectorized enlarged roadgraph tokens retain map structure used by the context encoder.",
            "SUMO lane polylines persist point-to-lane grouping and directed legal successors; the encoder pools lane tokens and applies predecessor message passing.",
            "interpreted",
            "src/topoworld/scenediffuserpp/roadgraph.py",
            "src/topoworld/scenediffuserpp/storage.py:_map_tensors",
            "src/topoworld/scenediffuserpp/context_encoder.py:RoadgraphEncoder",
        ),
        _entry(
            "sparse_loss",
            "Training",
            "All channels supervised when valid; only validity when invalid.",
            "V-prediction keeps inpainted entries supervised and applies the published binary loss mask, but mean reduction and equal agent/light loss weighting are clean-room choices because those details are not disclosed.",
            "interpreted",
            "src/topoworld/scenediffuserpp/diffusion.py:sparse_v_loss",
            "tests/scenediffuserpp/test_masks_diffusion.py:test_sparse_loss_keeps_inpainted_entries_under_published_sparse_weighting",
        ),
        _entry(
            "task_mixture",
            "Training",
            "Behavior prediction and SceneGen mixed with probability 0.5.",
            "Mixed config samples the two tasks with probability 0.5.",
            "matched",
            "configs/scenediffuserpp/train_small.yaml",
            "src/topoworld/scenediffuserpp/masks.py:mixed_multitensor_training_masks",
        ),
        _entry(
            "control_mask",
            "Training",
            "Factorized random controls over agent, time, and feature axes.",
            "Clean-room factorized subset sampler based on the paper equations.",
            "interpreted",
            "src/topoworld/scenediffuserpp/masks.py:_sample_factorized_control",
        ),
        _entry(
            "optimizer",
            "Training",
            "Adafactor with beta1 0.9 and Adam decay 0.9999.",
            "A local PaperAdafactor subclass keeps the Hugging Face Adafactor update structure while using fixed Adam-style decay_adam=0.9999; exact Waymo optimizer internals remain unavailable.",
            "interpreted",
            "src/topoworld/scenediffuserpp/trainer.py:build_optimizer",
            "configs/scenediffuserpp/train_small.yaml:optimizer_provenance",
        ),
        _entry(
            "batch_training_steps",
            "Scale",
            "Batch 1024 for 1.2 million optimization steps.",
            "Diagnostic configurations use batch 1-2 for 1K-5K optimization steps.",
            "reduced_scale",
            "configs/scenediffuserpp/train_small.yaml",
            "configs/scenediffuserpp/train_overfit_bp.yaml",
        ),
        _entry(
            "training_exposure",
            "Scale",
            "Training uses 1.2 million steps at batch 1024 on a large independent-scene corpus.",
            "The strongest completed width-matched diagnostic uses 5,000 batch-1 updates over eight overlapping windows from one SUMO run.",
            "reduced_scale",
            "configs/scenediffuserpp/train_large_32_probe.yaml",
            "docs/scenediffuserpp-nanjing-reproduction-status.md",
        ),
        _entry(
            "ema_selection",
            "Evaluation",
            "EMA and best validation checkpoint selected for final evaluation.",
            "EMA is stored and evaluated at 5,000 steps, but no best-validation checkpoint selection has been executed and both raw and EMA paper-mode generations fail all 13 strict gates.",
            "missing",
            "src/topoworld/scenediffuserpp/trainer.py:ExponentialMovingAverage",
            "scripts/scenediffuserpp/evaluate_short.py",
        ),
        _entry(
            "sampling_steps",
            "Sampling",
            "32 diffusion sampling steps.",
            "32 diffusion sampling steps.",
            "matched",
            "configs/scenediffuserpp/model_small.yaml",
            "configs/scenediffuserpp/model_paper.yaml",
        ),
        _entry(
            "sampling_transition",
            "Sampling",
            "At each inference step, predict the denoised solution, soft-clip sparse validity, then re-noise it with independent Gaussian noise at the next lower noise level.",
            "The named paper_renoise sampler performs v-recovery, paper soft clipping, and fresh independent Gaussian re-noise over both generated and inpainted values in that order.",
            "matched",
            "src/topoworld/scenediffuserpp/sampler.py:paper_denoise",
            "src/topoworld/scenediffuserpp/sampler.py:paper_renoise",
            "tests/scenediffuserpp/test_sampler.py:test_paper_denoise_and_renoise_transition_matches_fixed_equations",
            "tests/scenediffuserpp/test_sampler.py:test_inpainted_context_uses_independent_noise_at_each_paper_renoise_step",
        ),
        _entry(
            "diffusion_time_schedule",
            "Sampling",
            "Training samples continuous uniform diffusion time and inference uses 32 denoising steps; the exact 32-step time-grid discretization is not disclosed.",
            "Training uses continuous uniform scalar time; inference explicitly labels its 32-step linear time grid as a clean-room interpretation.",
            "interpreted",
            "src/topoworld/scenediffuserpp/trainer.py:train_step",
            "src/topoworld/scenediffuserpp/sampler.py:sampling_time_grid",
            "configs/scenediffuserpp/model_reduced_paper.yaml",
        ),
        _entry(
            "paper_soft_clip",
            "Sampling",
            "Scale values by validity probability; the paper text is ambiguous about whether the recursive validity channel stores probability or the original signed domain.",
            "The sampler keeps the recursive validity channel in the normalized signed domain [-1, 1] and uses probability only for value gating and diagnostics.",
            "interpreted",
            "src/topoworld/scenediffuserpp/sampler.py:soft_clip_sparse",
            "tests/scenediffuserpp/test_sampler.py:test_paper_soft_clip_keeps_recursive_validity_in_signed_training_domain",
            "tests/scenediffuserpp/test_sampler.py:test_paper_soft_clip_does_not_turn_negative_validity_positive_recursively",
        ),
        _entry(
            "signed_stable_clip",
            "Sampling",
            "No separate signed-domain recursive validity variant is published.",
            "The legacy signed_stable CLI value remains accepted as an alias for the signed-domain paper interpretation.",
            "interpreted",
            "src/topoworld/scenediffuserpp/sampler.py:soft_clip_sparse",
            "configs/scenediffuserpp/model_small.yaml",
            "scripts/scenediffuserpp/evaluate_short.py:--validity-mode",
        ),
        _entry(
            "long_rollout",
            "Evaluation",
            "Autoregressive city-scale trip simulation.",
            "No validated 60-second rollout; short generation gate only.",
            "missing",
            "docs/superpowers/plans/2026-07-20-scenediffuserpp-sumo-reproduction.md",
        ),
        _entry(
            "paper_metrics",
            "Evaluation",
            "Sliding-window distributional realism uses JS divergence for agent counts, entry/exit distances, off-road, collision, speed, traffic-light violations, and light transition matrices.",
            "Only per-sample diagnostic dynamics and sparse-count errors are implemented; the paper's long-rollout JS metric suite is absent.",
            "missing",
            "src/topoworld/scenediffuserpp/short_evaluation.py",
            "docs/superpowers/plans/2026-07-20-scenediffuserpp-fidelity-hardening.md:Task 7",
        ),
        _entry(
            "held_out_evaluation",
            "Evaluation",
            "Formal augmented-WOMD validation and test evaluation.",
            "A grouped held-out evaluator exists, but no independent validation/test corpus result has been produced.",
            "missing",
            "scripts/scenediffuserpp/evaluate_heldout.py",
            "docs/scenediffuserpp-nanjing-reproduction-status.md",
        ),
        _entry(
            "high_noise_denoising",
            "Evaluation",
            "Inference starts from pure Gaussian noise at diffusion time 1.",
            "The faithful mixed-objective 5,000-step probe reaches 0.801/0.787 agent/light balanced validity accuracy at time 1, but sample-0 agent XY RMSE is 38.70 m. An intentional BP-only fixed-t=1 diagnostic reaches 9.87 m across eight training windows after 2,000 steps, but no held-out high-noise or generation gate has passed.",
            "reduced_scale",
            "src/topoworld/scenediffuserpp/denoising_evaluation.py",
            "configs/scenediffuserpp/train_t1_overfit.yaml",
            "docs/scenediffuserpp-nanjing-reproduction-status.md",
        ),
        _entry(
            "dataset_split",
            "Evaluation",
            "Independent training, validation, and test scenes support model selection and final evaluation.",
            "The current eight-window diagnostic is blocked by the hardened audit: it contains one of 24 configured SUMO runs, all assigned to training, empty validation/test splits, no configured warmup, and only 180 of the configured 600 recording seconds.",
            "missing",
            "src/topoworld/scenediffuserpp/audit.py:_check_configured_corpus",
            "src/topoworld/scenediffuserpp/audit.py:_check_teacher_manifests",
            "src/topoworld/scenediffuserpp/storage.py:assign_split",
            "docs/scenediffuserpp-nanjing-reproduction-status.md",
        ),
        _entry(
            "planner_world_separation",
            "Rollout",
            "City-scale rollout separates AV control from generated world agents to avoid model collusion.",
            "No validated planner/world two-role rollout has been executed.",
            "missing",
            "docs/superpowers/specs/2026-07-20-scenediffuserpp-sumo-reproduction-design.md:Full-AR rollout",
        ),
    )


def audit_to_json(
    entries: Sequence[FidelityEntry], *, git_revision: str | None = None
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "paper_reference": PAPER_REFERENCE,
        "git_revision": git_revision,
        "entries": [asdict(entry) for entry in entries],
    }


def audit_to_markdown(
    entries: Sequence[FidelityEntry], *, git_revision: str | None = None
) -> str:
    lines = [
        "# SceneDiffuser++ Paper-Fidelity Audit",
        "",
        f"Paper: {PAPER_REFERENCE}",
        f"Git revision: `{git_revision or 'unknown'}`",
        "",
        "| ID | Area | Classification | Published | Local | Evidence |",
        "|---|---|---|---|---|---|",
    ]
    for entry in entries:
        evidence = "<br>".join(f"`{item}`" for item in entry.evidence)
        lines.append(
            "| "
            + " | ".join(
                (
                    f"`{entry.id}`",
                    _escape(entry.area),
                    f"`{entry.classification}`",
                    _escape(entry.published),
                    _escape(entry.local),
                    evidence,
                )
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def write_audit_artifacts(
    json_path: str | Path,
    markdown_path: str | Path,
    *,
    git_revision: str | None,
    force: bool,
) -> tuple[Path, Path]:
    json_destination = Path(json_path)
    markdown_destination = Path(markdown_path)
    existing = [path for path in (json_destination, markdown_destination) if path.exists()]
    if existing and not force:
        raise FileExistsError(f"audit artifact already exists: {existing[0]}")
    entries = build_fidelity_audit()
    _atomic_write(
        json_destination,
        json.dumps(
            audit_to_json(entries, git_revision=git_revision),
            indent=2,
            allow_nan=False,
        )
        + "\n",
    )
    _atomic_write(
        markdown_destination,
        audit_to_markdown(entries, git_revision=git_revision),
    )
    return json_destination, markdown_destination


def _entry(
    entry_id: str,
    area: str,
    published: str,
    local: str,
    classification: str,
    *evidence: str,
) -> FidelityEntry:
    return FidelityEntry(
        id=entry_id,
        area=area,
        published=published,
        local=local,
        classification=classification,
        evidence=tuple(evidence),
    )


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
