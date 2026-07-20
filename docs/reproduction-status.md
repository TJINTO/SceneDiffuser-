# SceneDiffuser++ Nanjing Reproduction Status

Snapshot date: 2026-07-20

## Latest Correction: Validity Domain

The earlier paper-mode all-valid diagnosis was partly caused by a validity-domain
mix in the sampler. The paper maps signed validity to a probability for value
gating, but its text is ambiguous about whether the recursive validity channel
should store that probability or remain in the normalized signed domain. The
implementation now uses one convention consistently: sparse validity channels
remain signed in `[-1, 1]`; probability is used only for value gating and
diagnostics.

With the same Nanjing dense500 checkpoint and no retraining, the corrected
paper-mode sampler no longer collapses lights or agents to all-valid:

| Evaluation | Status | Pass rate | Remaining failures |
|---|---:|---:|---|
| Sample 118 short gate, raw, 32 steps, speed/accel/jerk prior | passed | 1 / 1 | none |
| Validation run heldout, 16 samples x 2 seeds | blocked | 0.75 | `agent_removal_transition_error` |

The remaining failure is now narrower: generated trajectories keep history-active
agent slots valid for the whole future horizon, while SUMO targets include
birth/removal events as vehicles enter or leave the local scene. This is not a
threshold artifact: inspected generated raw validity for sample 118 is strongly
positive for slots 0-4 across all 80 future frames, while the target has two
removals and one birth. A 64x transition-weighted training probe created
birth/removal events but over-corrected and reduced heldout pass rate to 0.6875,
so transition weighting alone is not accepted as a fix.

Lifecycle audit after the validity-domain fix:

| Check | Current result |
|---|---:|
| Dataset samples | 384 |
| Tracked agent slots | 2,246 |
| Samples with agent birth | 60.94% |
| Samples with agent removal | 57.55% |
| Agent birth transitions in targets | 392 |
| Agent removal transitions in targets | 330 |
| Multi-segment agent slots | 4 |

The implementation checks now explicitly cover the paper removal mechanism:
fixed agent rows within a window, invalid rows set to zero with signed
validity `-1`, BP history inpainting including the validity channel, sparse
loss supervising invalid validity, and no separate removal head or hard delete
rule. Existing validation records show the remaining generation miss directly:
by record, targets contain 16 agent birth transitions and 46 removals, while
the generated samples contain 0 births and 0 removals.

## Verdict

The current result is a clean-room, reduced-data implementation probe. It is
not a high-fidelity reproduction of the CVPR 2025 result and it is not a usable
traffic world model.

The implementation now follows the disclosed sparse-validity mechanism more
closely. The latest Nanjing 128-slot, 1,000-step probe can pass one short
paper-mode sample gate after the signed-validity correction and kinematic
projection, but held-out evaluation remains blocked. Loss reduction is not the
limiting evidence: the 1,000-step training loss falls from 2.18 to 0.06 and
agent validity loss falls to about 0.015, yet recursive generation still misses
birth/removal transitions and pushes acceleration/jerk to the projection caps.

Primary references:

- SceneDiffuser++: City-Scale Traffic Simulation via a Generative World Model:
  [arXiv](https://arxiv.org/abs/2506.21976),
  [CVF PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Tan_SceneDiffuser_City-Scale_Traffic_Simulation_via_a_Generative_World_Model_CVPR_2025_paper.pdf)
- SceneDiffuser predecessor: https://proceedings.neurips.cc/paper_files/paper/2024/file/64ff8d0bf0b0fe2b872a42a0de9668f8-Paper-Conference.pdf

## What Matches

- 10 Hz, 91-step scenes with 11 history and 80 future steps.
- Joint agent and traffic-light tensors in one AV-centric coordinate frame.
- Cosine alpha/sigma schedule and v-prediction.
- Sparse supervision: all value channels when valid and only validity when invalid.
- BP/SceneGen mixture with nominal probability 0.5.
- Paper soft clipping and independent Gaussian re-noising over 32 steps.
- Hidden size 512, 8 layers, 8 heads, and 192 context queries in the largest run.
- Noisy, local inpainting, and global roadgraph context all enter AdaLN conditioning.

The final conditioning projection and Perceiver-style context encoder remain
clean-room interpretations because the authors did not release implementation
code. Matching interfaces and equations does not establish distributional
reproduction.

## Strongest Controlled Probe

Artifacts:

- Dataset: `outputs/scenediffuserpp/paper_dense_fixture_transition_v2`
- Checkpoint: `outputs/scenediffuserpp/large_32_noisy_fusion_1000/step_00005000.pt`
- Raw paper evaluation: `outputs/scenediffuserpp/large_32_noisy_fusion_eval_5000_raw_detailed_trace/short_evaluation.json`
- EMA paper evaluation: `outputs/scenediffuserpp/large_32_noisy_fusion_eval_5000_ema_detailed_trace/short_evaluation.json`
- Signed diagnostic: `outputs/scenediffuserpp/large_32_noisy_fusion_eval_5000_raw_signed_detailed_trace/short_evaluation.json`

The run is named `large_32`: model width/depth match the disclosed large
backbone, but the executed tensor capacity remains 32 agents instead of 128.

| Item | Current value | Published setup |
|---|---:|---:|
| Unique windows | 8 | Large WOMD-XLMap corpus |
| Independent runs | 1 of 24 configured | Independent train/validation/test scenes |
| Independent-run splits | 1 / 0 / 0 | Nonempty train / validation / test |
| Teacher warmup / recording | 0 s / 180 s | 60 s / 600 s configured |
| Agent tensor capacity | 32 | 128 |
| Light tensor capacity | 32 | Not disclosed |
| Model parameters | 58,458,649 | Exact count not disclosed |
| Batch size | 1 | 1024 |
| Optimization steps | 5,000 | 1,200,000 |
| Scene exposures | 5,000 | 1,228,800,000 |
| Checkpoint selection | Final raw/EMA diagnostics | Best validation model, then test |

The hardened dataset audit now correctly returns `blocked`: the configuration
declares 24 SUMO runs, but only one exists, validation/test are empty, and its
teacher records 180 seconds without the configured 60-second warmup instead of
recording 600 seconds after warmup. Diagnostic training requires the explicit
`--allow-incomplete-corpus` override, which is persisted in the audit and
checkpoint provenance. Source-manifest hash or 10 Hz contract failures cannot
use this override.

Training is finite and continuous from step 1 through 5,000. Mean loss falls
from 1.203 over the first 100 updates to 0.103 over the final 100. The final
100-step high-noise-bin mean remains 0.262, so aggregate loss hides the hard
near-pure-noise regime.

## Paper-Mode Generation Failure

| Metric | SUMO target | Raw, 5K | EMA, 5K |
|---|---:|---:|---:|
| Agent valid fraction | 0.2105 | 1.0000 | 1.0000 |
| Light valid fraction | 0.5219 | 1.0000 | 1.0000 |
| Mean speed | 6.56 m/s | 32.83 m/s | 20.26 m/s |
| P95 speed | 16.12 m/s | 66.20 m/s | 48.05 m/s |
| P95 acceleration | 5.00 m/s^2 | 1,134.65 m/s^2 | 796.22 m/s^2 |
| P95 jerk | 116.38 m/s^3 | 20,806.28 m/s^3 | 14,556.94 m/s^3 |
| Strict failed gates | 0 | 13 | 13 |

EMA improves magnitudes but does not restore sparse validity or physical
trajectories. The unpublished `signed_stable` diagnostic avoids the all-valid
fixed point, but is not a reproduction default and still fails 7 gates:
agent/light validity are 0.100/0.343,
mean speed is 22.26 m/s, P95 acceleration is 774.39 m/s^2, and P95 jerk is
13,163.48 m/s^3. It is not a substitute for the paper sampler.

## PaperAdafactor 128-Slot Short Probe

Artifacts:

- Dataset: `outputs/scenediffuserpp/paper128_light_fixture_short`
- Old HF-Adafactor strict paper evaluation: `outputs/scenediffuserpp/paper128_light_short_eval_100_strict_paper/short_evaluation.json`
- PaperAdafactor checkpoint: `outputs/scenediffuserpp/paper128_light_paperadafactor_100/step_00000100.pt`
- PaperAdafactor strict paper evaluation: `outputs/scenediffuserpp/paper128_light_paperadafactor_eval_100_strict_paper/short_evaluation.json`

The local optimizer now stores and executes the disclosed fixed
`decay_adam=0.9999` Adafactor semantics. The 2-step CUDA smoke test confirms
that `PaperAdafactor`, `decay_adam=0.9999`, and `validity_mode=paper` are
persisted in checkpoint provenance.

The 100-step apples-to-apples probe improves denoising loss but not generation:

| Metric | HF Adafactor, 100 | PaperAdafactor, 100 |
|---|---:|---:|
| Strict status | blocked | blocked |
| Loss ratio | 0.4649 | 0.3802 |
| Final short loss | 0.8838 | 0.7652 |
| Generated agent valid fraction | 1.0000 | 1.0000 |
| Target agent valid fraction | 0.0131 | 0.0131 |
| Generated light valid fraction | 1.0000 | 1.0000 |
| Target light valid fraction | 0.0303 | 0.0303 |
| Generated mean speed | 144.41 m/s | 156.12 m/s |
| Generated P95 speed | 289.24 m/s | 315.14 m/s |
| Generated P95 acceleration | 5,016.64 m/s^2 | 5,440.02 m/s^2 |
| Generated P95 jerk | 90,895.95 m/s^3 | 99,308.83 m/s^3 |

Direct denoising diagnostics still show that the model can approach target
sparsity at lower noise. At `t=0.25`, PaperAdafactor predicts an agent-valid
fraction of 0.0105 against a target of 0.0131. The full recursive sampler
nevertheless ends with 100% of unknown agent and light validity probabilities
above 0.5. This isolates the short-run failure to recursive paper soft-clipping
stability and insufficient high-noise/geometry quality, not to checkpoint
provenance or the old Adafactor decay approximation alone.

## Nanjing 24-Run Corpus Build

Artifacts:

- SUMO teacher root: `outputs/scenediffuserpp/nanjing_teacher_24`
- Bounded 24-run dataset: `outputs/scenediffuserpp/nanjing_paper_24_dataset_16perrun_no_light_filter`
- Dataset audit: `outputs/scenediffuserpp/nanjing_paper_24_dataset_16perrun_no_light_filter_audit/audit.json`

The 24 configured SUMO teachers now complete on the Nanjing network. Each run
simulates 660 seconds at 10 Hz, records 600 seconds after a 60-second warmup,
and writes FCD plus TLS JSONL. Mean teacher runtime is 78.9 seconds per run.

The first full dataset build attempt was too slow because
`minimum_light_state_transitions=1` performs an expensive AV-centric window
build before rejecting samples. A cProfile probe on one low-flow run showed
that collecting 8 accepted windows built 156 candidate windows and took
24.4 seconds. With the light-transition filter disabled, the same 8-window
probe took 15.9 seconds; the remaining fixed costs are FCD parsing, TLS JSONL
parsing, roadgraph loading, and per-window 1000-meter map tensorization.

The builder now supports `--max-windows-per-run` and caches the static
roadgraph for repeated runs on the same network. A bounded dataset with
16 windows per run and no per-sample light-transition requirement builds in
about 8.5 minutes and passes audit:

| Metric | Value |
|---|---:|
| Samples | 384 |
| Independent SUMO runs | 24 |
| Train / validation / test samples | 288 / 48 / 48 |
| Independent train / validation / test runs | 18 / 3 / 3 |
| Minimum teacher recording duration | 600 s |
| Mean agent valid fraction | 0.0150 |
| Mean light valid fraction | 0.0323 |
| Samples with light transitions | 32 |
| Total light-state transitions | 226 |
| Agent/light/map truncation fraction | 0.0 |

This is the first structurally valid 24-run Nanjing SceneDiffuser++ corpus in
this worktree. It is still a bounded engineering corpus, not a paper-scale
dataset. The low light-transition coverage is expected because the expensive
per-sample transition filter was disabled; a separate light-focused subset is
needed before evaluating traffic-light generation quality.

## Nanjing 24-Run 100-Step Probe

Artifacts:

- Checkpoint: `outputs/scenediffuserpp/nanjing24_16perrun_paper128_100/step_00000100.pt`
- Training log: `outputs/scenediffuserpp/nanjing24_16perrun_paper128_100/train.jsonl`
- Validation evaluation: `outputs/scenediffuserpp/nanjing24_16perrun_paper128_eval_100_validation_raw/heldout_evaluation.json`

The 100-step PaperAdafactor probe trains on the audited 24-run bounded corpus
without `--allow-incomplete-corpus`. It completes in 57 seconds on CUDA BF16
with peak allocated memory of about 8.13 GiB. No NaN, CUDA, or empty-graph
failure occurs.

Held-out evaluation uses raw weights, paper-mode validity, 32 sampling steps,
one validation run, 16 samples, and two seeds per sample. It remains blocked:

| Metric | Value |
|---|---:|
| Pass rate | 0 / 32 |
| Mean loss ratio | 0.3074 |
| Generated agent valid fraction | 1.0000 |
| Target agent valid fraction | 0.0142 |
| Generated light valid fraction | 1.0000 |
| Target light valid fraction | 0.0542 |
| Generated mean speed | 110.27 m/s |
| Target mean speed | 17.93 m/s |
| Generated P95 speed | 218.93 m/s |
| Generated P95 acceleration | 3,770.77 m/s^2 |
| Generated P95 jerk | 68,623.17 m/s^3 |

The new corpus fixes the structural experiment problem but does not fix
paper-mode recursive generation. Compared with the 128-slot fixture probe,
speed magnitudes are lower, yet all generated agent and light slots still end
valid. The next useful training run should not be reported as a reproduction
result until it passes at least the held-out short gate.

## High-Noise Diagnosis

One-step recovery uses fixed seeds and the raw 5K weights on sample index 0,
matching the canonical short-evaluation artifact:

| Noise time | Agent balanced validity | Agent recall | Agent XY RMSE | Light balanced validity | Light state accuracy |
|---:|---:|---:|---:|---:|---:|
| 1.00 | 0.801 | 0.670 | 38.70 m | 0.787 | 0.296 |
| 0.99 | 0.774 | 0.600 | 39.90 m | 0.802 | 0.337 |
| 0.90 | 0.927 | 0.873 | 31.83 m | 0.964 | 0.844 |
| 0.75 | 0.993 | 0.990 | 20.35 m | 0.994 | 0.990 |
| 0.50 | 0.998 | 0.997 | 10.50 m | 0.999 | 1.000 |
| 0.25 | 1.000 | 1.000 | 6.63 m | 1.000 | 1.000 |

Validity classification improves substantially over the 1K checkpoint, but
trajectory geometry remains poor. The sampler trace shows raw agent validity
crossing probability 0.5 at denoising step 25 and lights at step 9, ending at
all-valid. This is gradual recursive drift, not a final threshold-decoding bug.
An additional audit over all eight training windows and two seeds gives a
`t=1` XY RMSE of 28.30 m, so the sample-0 failure is not isolated.

The sampling-step sweep separates the two failure modes:

| Sampling steps | Agent valid fraction | Light valid fraction | Mean speed | P95 jerk |
|---:|---:|---:|---:|---:|
| 1 | 0.209 | 0.336 | 70.52 m/s | 50,695.67 m/s^3 |
| 8 | 0.741 | 0.930 | 68.74 m/s | 44,652.25 m/s^3 |
| 16 | 0.998 | 1.000 | 51.03 m/s | 31,660.26 m/s^3 |
| 32 | 1.000 | 1.000 | 32.83 m/s | 20,806.28 m/s^3 |

The first denoised sample is already physically invalid; additional steps
partly reduce motion magnitude while amplifying validity collapse.

## Fixed Pure-Noise Diagnostic

To distinguish a disconnected conditioning path from insufficient faithful
training, a separate diagnostic trains the small backbone on BP only with the
diffusion time fixed to `t=1`. This deliberately changes the published uniform
time distribution and is not a reproduction result.

Artifacts:

- Config: `configs/scenediffuserpp/train_t1_overfit.yaml`
- Checkpoint: `outputs/scenediffuserpp/t1_bp_overfit_small_1000/step_00002000.pt`

Across all eight training windows and two fixed noise seeds, raw `t=1` agent
XY RMSE falls from 16.13 m at 1,000 steps to 9.87 m at 2,000 steps. The
2,000-step result beats the same windows' static-position baseline of 20.82 m
and constant-velocity baseline of 14.72 m. Agent balanced validity accuracy is
0.939 and light-state accuracy is 0.905.

This shows that history context can influence future geometry through the
current conditioning and axial-attention path. It does not prove successful
overfitting: 9.87 m is still a large training-set error, and the diagnostic
removes SceneGen, randomized control masks, and the uniform diffusion-time
objective. Its EMA result is intentionally not used as a gate because the
undisclosed local decay choice of 0.999 severely lags at only 1K-2K updates.

## Controlled Findings

| Change | Observation | Conclusion |
|---|---|---|
| Add the missing noisy-token conditioning path | t=1 validity improves after 5K updates | The architecture correction matters but is insufficient |
| Continue 1K to 5K updates | t=1 agent/light balanced validity rises to 0.801/0.787 | More training learns sparse classes, not coherent trajectories |
| Trace all 32 soft-clipping steps | Validity crosses 0.5 late and converges to all-valid | Collapse is recursive, not final decoding |
| Use signed-domain diagnostic | Failed gates fall from 13 to 7 | Diagnostic only; paper probability recursion remains the reproduction path |
| Use fresh independent context noise per re-noise step | Metrics change negligibly | The prior fixed-noise context path was a fidelity bug, not the root cause |
| Sweep 1 to 32 sampling steps | One-step dynamics are worst; validity collapse grows with steps | Denoising quality and recursive stability are separate blockers |
| Replace HF Adafactor with fixed-decay PaperAdafactor | 100-step loss ratio improves from 0.465 to 0.380 but all-valid generation remains | Optimizer fidelity is necessary provenance hardening, not the root cause |
| Generate 24 Nanjing SUMO teachers | All 24 configured 10 Hz teachers complete with 600 s recording | Teacher corpus contract is now available |
| Build bounded 24-run dataset | 384-window dataset passes audit with nonempty validation/test splits | Ready for a small mixed-objective training probe |
| Harden corpus audit | 8-window corpus is now blocked as 1/24 runs with no held-out data | Structural validity must not be reported as experiment readiness |
| Validate source teacher contracts | The only run has 0 s warmup and 180 s recording against configured 60/600 s | The current dataset is a short fixture, not the declared corpus |
| Fix BP at pure noise for 2K diagnostic steps | Train-set XY RMSE reaches 9.87 m and beats constant velocity at 14.72 m | Conditioning is connected, but the faithful mixed objective remains unsolved |
| Log every output channel | Easy dimension/category channels diverge sharply from `x/y/heading` behavior | Aggregate diffusion loss is not a trajectory-quality proxy |

CUDA, BF16, topology tensors, deterministic model initialization, shuffled
epoch sampling, checkpoint resume, and finite-value checks operate without NaN,
empty-graph, or CUDA failures in this probe.

## Remaining Fidelity Gaps

1. **P0 - Held-out lifecycle generation fails.** Validation targets contain birth/removal transitions, but the current generated samples produce none.
2. **P0 - Continuous geometry remains weak.** Mean speed is plausible in some samples, but acceleration and jerk sit at the projection caps, so the trajectory dynamics are not learned cleanly.
3. **P0 - Generation quality is not proven without projection.** The latest pass uses speed, acceleration, and jerk projection; a faithful paper-mode short gate without these engineering priors is still required.
4. **P0 - Training scale is incomparable.** The current 384-window Nanjing corpus and 1,000-step probe are far below the paper's 1.2M-step, large-corpus setup.
5. **P0 - Validation selection is absent.** EMA/final-step diagnostics are not best-validation checkpoint selection.
6. **P0 - Long-rollout evaluation is absent.** There is no validated 600-step rollout, planner/world separation, or paper JS metric suite.
7. **P0 - Architecture is interpreted.** Exact Perceiver IO depth, local/global fusion, attention masks, and implementation details are unavailable without official code.
8. **P1 - Control-mask sampling is under-specified.** The paper does not disclose enough sampling-distribution detail to verify the clean-room choice.
9. **P1 - Sparse-loss reduction is under-specified.** The binary mask matches the equation, but reduction and agent/light weighting are clean-room choices.
10. **P1 - SUMO semantics are narrow.** The corpus is structurally valid but still single-city, bounded-window synthetic SUMO data.
11. **P1 - Short-run EMA is not calibrated.** The paper does not disclose EMA decay; local EMA behavior is still diagnostic.
12. **P1 - Transition-weighted loss is diagnostic only.** It exposed that transition supervision matters but is not part of the paper reproduction and over-corrected in heldout evaluation.

## Next Gates

Do not claim a SceneDiffuser++ reproduction score or start a formal comparison
table from these results. The next sequence is:

1. Add lifecycle transition totals to every held-out report and gate the next run on nonzero generated birth/removal when targets contain them.
2. Run a longer 128-agent Nanjing training probe on the audited 24-run corpus with ordinary diffusion loss, not transition-weighted loss.
3. Evaluate both projected and unprojected sampling; projection may be a safety guard, but it cannot be the only reason the sample passes.
4. Select checkpoints on validation metrics instead of final-step diagnostics.
5. Implement the 60-second rollout and paper sliding-window JS metrics before test reporting.

Current status: `blocked on lifecycle generation and physical trajectory quality`.
