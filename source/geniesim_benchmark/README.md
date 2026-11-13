# Benchmark

See [Benchmark usage](benchmark_test_docs.md) for setup, task discovery and per-task smoke commands.

Isaac tasks run with system Python 3.12 and Isaac Sim 6.2 in the dedicated
`geniesim4:latest` image (`geniesim docker` / `geniesim docker6.2`). Newton and
libuipc tasks use the same interpreter; native MuJoCo keeps its separate runtime.

## Weighted results and task assets

`POLICY_BUDGET_WEIGHTED` reports the budget-adjusted score. The
stage baseline is
`B = max(1, floor(floor(StepOut / 30) * 0.8))`. An accepted dispatch count
`N` scales that stage's native score by
`max(0, 1 - 2 * max(0, N - B) / B)`. Exceeding any stage baseline also
removes the completion bonus. Native checker scores, E2E and execution
limits remain unchanged. Missing budget evidence produces an unavailable
weighted score, excluded from its measured denominator; measured zeroes stay
in the denominator.

DexHand 2 reports use `evaluate_ret_00.json` across libuipc, MuJoCo Rubik
and Newton wok evaluation. They apply one dispatch budget per episode and
report native checkpoints and E2E separately from the budget-adjusted score.

An explicit `planning_exhausted` / `no_valid_hold` terminal reply closes an
episode using the last native state, without issuing another action or
incrementing its dispatch count. It does not declare task success.

Use the published task assets matching this benchmark release; see the
[asset download guide](../geniesim_cli/docs/assets-download.md) for download
and verification commands. The WBC roller-shutter task requires
`background_wbc_shutter.usda` and `WbcKeyLocked.usda`. The isolated background
disables the duplicate shutter; the foreground lock references the key in the
same workspace.

<a id="deformable-evaluation"></a>
<a id="deformable-scoring"></a>

Task launch and evaluation: [run a task](benchmark_test_docs.md#3-run-a-task).

Task-specific scoring: [scoop_popcorn rules](docs/scoop_popcorn_evaluation.md).

The libuipc cloth-hanger, fold-cloth and drawstring-bag tasks reuse one physics,
CPU MuJoCo and synchronized render session across their material layouts. All
episode plans are declared in startup YAML. Each episode seals its own physical
archive and score before the next native reset; material receipts, observations
and score records retain separate source epochs. Geometry changes are rejected
rather than rebuilt implicitly. The explicit GPU lifecycle diagnostic
[sequence probe](tests/native_libuipc_sequence_smoke.py) uses a deterministic hold
policy and is separate from model evaluation.

When a resume coordinator references an already scored layout, it must call
`LibuipcRunSession.skip_completed_episode(index)` for every skipped index in
sequence. This captures and validates the original material plan while physics
is paused, then seals the zero-action lifecycle entry. Resetting physics alone
leaves the renderer on an older material epoch and causes the first remaining
layout to fail with `material sequence cannot skip or replay an episode`.
Skipped layouts produce reuse receipts, no policy actions and no new score.

Clothes-hanger, folding and drawstring-bag tasks accept the same recording and
resume flags as other tasks:

```bash
geniesim benchmark run g2crsbwuji_dexmanip_cloth_hanger --infer-host localhost:8999 \
  --benchmark.output_dir=output/cloth_hanger --benchmark.record=true --benchmark.resume=true
```

Retry with the same output directory, task settings and inference service.
Completed, scored episodes are reused in `evaluate_ret_00.json`; interrupted
episodes run again. Keep the entire output directory until finalization.
Recording saves three camera videos and measured joint states; the head video
includes the instruction and zero-based episode index. Frames are captured at
observed control ticks. Native physical archives needed for scoring are always
kept, independently of `record`. Both `JOINT_ABS` and `EEF_ABS` arms are supported;
EEF uses shared hard IK and accepts `base_link` or `arm_base_link` coordinates.

Rubik tasks also accept `benchmark.resume` and optional `benchmark.resume_id`.
Keep the same output directory and settings across retries. Completed successes,
task failures and timeouts are retained; inference interruptions are retried.
Keep the native episode files, including recorded `raw-1000hz.npz`, with the
checkpoint. Once all episodes are committed, retrying only republishes scores.


Current interfaces are defined by:

| Contract | Source |
|---|---|
| Task runtime selection | [runtime.py](src/geniesim_benchmark/runtime.py) |
| Isaac task parameters | [params.py](src/geniesim_benchmark/config/params.py) |
| EEF conversion and IK acceptance | [eef_control.py](src/geniesim_benchmark/utils/eef_control.py), [hard_ik.py](src/geniesim_benchmark/utils/hard_ik.py), [usd_ik.py](src/geniesim_benchmark/utils/usd_ik.py) |
| Scores and episode summaries | [eval_utils.py](src/geniesim_benchmark/plugins/output_system/eval_utils.py) |
| Inference preflight | [check-inference](skills/check-inference/SKILL.md) |

[License](LICENSE) · [Skills](skills/README.md) · [Challenge workflow](skills/robocoliseum/)

Newton wok evaluation keeps one engine and one Isaac RTX worker alive across
layouts, together with persistent simulation/image/score clients. Each episode
opens a fresh inference connection. `geniesim benchmark run` and `run-layouts` validate that layouts differ
only in robot XY, then select each layout through `reset(instances=[id])`.
The engine restores robot/wok state, particle positions, zero velocities and
solver carryover; the runner verifies the published base pose. One session ID
and increasing reset epochs identify independent episodes. Native results stay
in per-layout directories. A shared observer recording carries a frame/epoch
sidecar and produces per-layout video clips; no physics replay is involved.
Topology, scoring or non-XY pose differences are rejected before starting.
Per-layout `results.json` and the top-level `batch_summary.json` retain native
JSON reporting; layout reuse does not change scoring thresholds or budgets.



## HDR backgrounds

Rubik's Isaac renderer and the two phone-packaging task configurations choose
from 11 reviewed office/home `.hdr` files in `assets/background/common/HDR/`.
Outdoor, rooftop and unreviewed backgrounds are excluded from both tasks.
The visible panorama and all
DomeLight textures in the task environment use the same selected file. Task
geometry, cameras, checker rules and light intensities are preserved; the old
HDR backdrop is hidden. The panorama has no physics schemas.

Both tasks resample the HDR for every episode, including repeated attempts on
the same layout. Phone packaging enables `generalization.hdr_texture` with
`resample_per_episode: true` and uses the existing benchmark seed plus layout,
instruction-file and episode coordinates. Rubik uses `benchmark.seed` plus the
scene identifier, source hash and episode index. Sampling uses an isolated
random generator, leaving model and physics RNG streams unchanged.

The same seed, episode coordinates and HDR pool reproduce the same choice;
changing the seed or episode produces a new random draw. Sampling is with
replacement, so draws may repeat. Reset updates the panorama and environment
light in the existing renderer process.

Rubik records `render_background` in each episode result and writes
`isaac-render/current-background.json`; phone packaging logs `HDR_BACKGROUND`
and includes the receipt in `recording_info.json`. Receipts contain the relative
asset path, chosen texture SHA256 and pool-name hash. Missing pools or changed
selected files fail explicitly instead of silently retaining the old backdrop.

## Common launch and output

```bash
geniesim benchmark run <TASK> --infer-host localhost:8999 \
  --app.headless=true --benchmark.record=true --benchmark.resume=true \
  --benchmark.output_dir=/path/to/results
```

Use the same command and output root on retries. Without an explicit path,
`GENIESIM_BENCHMARK_OUTPUT_ROOT` or `OUTPUT_DIR` selects the root; the default is
`output/benchmark`. Scores are saved under `ROOT/task_name/sub_task_name/`;
if the task has no separate sub-task, both directory names are the task name.
The score file is `evaluate_ret_00.json`. Recordings keep per-episode state and
aggregate `head.webm`, `hand_left.webm` and `hand_right.webm`; the head camera
shows the instruction and zero-based episode index. Camera recording requires
rendering to be enabled. `--benchmark.record_video_dir` additionally exports
scores, videos and state files to `PATH/sub_task_name/`.
