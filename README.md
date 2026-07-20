# SceneDiffuser++ SUMO Reproduction

This repository contains a clean-room implementation and diagnostic reproduction
of the SceneDiffuser++ traffic world-model pipeline, adapted to SUMO-generated
Nanjing traffic scenes.

It is **not** the official SceneDiffuser++ code release. The original paper did
not release implementation code at the time this repository was prepared, so
the architecture details here are reconstructed from the paper and validated
with local SUMO fixtures.

## Scope

Included:

- SceneDiffuser++-style sparse agent/light tensor schema.
- SUMO FCD and TLS export/parsing utilities.
- AV-centric 10 Hz scene-window builder.
- Roadgraph and traffic-light topology tensorization.
- Diffusion schedule, sparse validity/value masking, v-prediction loss.
- Multi-tensor denoiser with local/global context conditioning.
- Paper-mode sampler with signed-validity stabilization diagnostics.
- Short single-sample and grouped held-out evaluation scripts.
- Dataset and fidelity audit tools.
- Focused tests for masks, lifecycle validity, sampler stability, training,
  held-out aggregation, and SUMO scene conversion.

Excluded:

- Generated datasets.
- SUMO FCD/TLS outputs.
- Model checkpoints.
- Training logs and figures.
- Non-SceneDiffuser non-SceneDiffuser baseline and exploratory experiments.

## Current Status

The current reproduction is useful as an engineering and research probe, but it
is not yet a high-fidelity reproduction of the published CVPR 2025 result.

Latest local findings:

- A 384-window, 24-run Nanjing SUMO corpus passes structural audit.
- The 1,000-step 128-agent probe learned aggregate validity but initially failed
  lifecycle generation: generated birth/removal transitions were both zero.
- Continuing training to 4,000 steps improved single-sample lifecycle behavior:
  generated birth/removal changed from `0/0` to `2/1` on a target with `1/2`.
- Trajectory dynamics remain weak: acceleration and jerk still hit projection
  caps in short evaluation.
- Batch-size probing on an RTX 4090 showed `batch=3` and `batch=4` can start,
  but `batch=2` was more stable and higher-throughput under Windows/WDDM.

See [docs/reproduction-status.md](docs/reproduction-status.md) for the detailed
audit trail and current blockers.

## Environment

The tested environment is defined in `environment.yml`.

```powershell
conda env create -f environment.yml
conda activate scenediffuserpp-sumo
```

For local source imports:

```powershell
$env:PYTHONPATH = "src"
```

## Tests

Run the focused SceneDiffuser++ suite:

```powershell
$env:PYTHONPATH = "src"
D:\miniconda3\envs\scenediffuserpp-sumo\python.exe -m pytest tests\scenediffuserpp -q
```

Recent local focused verification before export:

```text
117 passed in 17.47s
```

## Main Commands

Generate a SUMO teacher corpus:

```powershell
$env:PYTHONPATH = "src"
D:\miniconda3\envs\scenediffuserpp-sumo\python.exe scripts\scenediffuserpp\generate_sumo_corpus.py --help
```

Build a scene dataset:

```powershell
D:\miniconda3\envs\scenediffuserpp-sumo\python.exe scripts\scenediffuserpp\build_scene_dataset.py --help
```

Audit a dataset:

```powershell
D:\miniconda3\envs\scenediffuserpp-sumo\python.exe scripts\scenediffuserpp\audit_scene_dataset.py --dataset outputs\scenediffuserpp\nanjing_paper_24_dataset_16perrun_dense500_scale1000_no_light_filter --out outputs\scenediffuserpp\nanjing_dataset_audit --skip-hashes
```

Train a 128-agent behavior-prediction probe:

```powershell
D:\miniconda3\envs\scenediffuserpp-sumo\python.exe scripts\scenediffuserpp\train.py `
  --config configs\scenediffuserpp\train_paper128_bp_batch2_short.yaml `
  --dataset outputs\scenediffuserpp\nanjing_paper_24_dataset_16perrun_dense500_scale1000_no_light_filter `
  --out outputs\scenediffuserpp\nanjing24_paper128_bp_batch2
```

Continue training with less frequent checkpointing:

```powershell
D:\miniconda3\envs\scenediffuserpp-sumo\python.exe scripts\scenediffuserpp\train.py `
  --config configs\scenediffuserpp\train_paper128_bp_batch2_short.yaml `
  --dataset outputs\scenediffuserpp\nanjing_paper_24_dataset_16perrun_dense500_scale1000_no_light_filter `
  --out outputs\scenediffuserpp\nanjing24_paper128_bp_batch2 `
  --resume outputs\scenediffuserpp\nanjing24_paper128_bp_batch2\step_00001000.pt `
  --max-steps 10000 `
  --checkpoint-every 500
```

Run a short generation gate:

```powershell
D:\miniconda3\envs\scenediffuserpp-sumo\python.exe scripts\scenediffuserpp\evaluate_short.py `
  --config configs\scenediffuserpp\train_paper128_bp_batch2_short.yaml `
  --dataset outputs\scenediffuserpp\nanjing_paper_24_dataset_16perrun_dense500_scale1000_no_light_filter `
  --checkpoint outputs\scenediffuserpp\nanjing24_paper128_bp_batch2\step_00004000.pt `
  --train-log outputs\scenediffuserpp\nanjing24_paper128_bp_batch2\train_step00004000.jsonl `
  --out outputs\scenediffuserpp\short_eval_step4000 `
  --weights raw `
  --sample-index 118 `
  --sampling-steps 32 `
  --validity-mode paper `
  --max-speed-mps 40 `
  --max-acceleration-mps2 15 `
  --max-jerk-mps3 100
```

Run grouped held-out evaluation:

```powershell
D:\miniconda3\envs\scenediffuserpp-sumo\python.exe scripts\scenediffuserpp\evaluate_heldout.py `
  --train-config configs\scenediffuserpp\train_paper128_bp_batch2_short.yaml `
  --eval-config configs\scenediffuserpp\eval.yaml `
  --dataset outputs\scenediffuserpp\nanjing_paper_24_dataset_16perrun_dense500_scale1000_no_light_filter `
  --checkpoint outputs\scenediffuserpp\nanjing24_paper128_bp_batch2\step_00004000.pt `
  --train-log outputs\scenediffuserpp\nanjing24_paper128_bp_batch2\train_step00004000.jsonl `
  --out outputs\scenediffuserpp\heldout_eval_step4000
```

## Notes on Reproduction Fidelity

The paper represents agent removal through the validity channel, not a separate
removal head or explicit deletion rule. This implementation follows that design:

- value channels are supervised only when valid;
- validity is supervised at all timesteps;
- history inpainting includes validity;
- intermediate denoise steps use soft sparse validity handling;
- final lifecycle quality is measured by birth/removal transition counts.

The current primary blocker is no longer an all-valid validity-domain bug. The
remaining blocker is held-out lifecycle and physical trajectory generation:
single-sample lifecycle transitions improve with more training, but full
held-out transition totals and natural acceleration/jerk remain unresolved.
