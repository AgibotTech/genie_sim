# scripts/tools/ — agent guide

Standalone debugging tools that load Newton models *without* ROS / Kit, so
adapters and solver configurations can be exercised in isolation.  These
exist for one reason: when something breaks at engine bringup, you should
be able to reproduce it without spinning up a full launcher + ROS graph.

Anything in this directory is a **debug surface**, not production. Keep
them small enough that an AI agent (or a tired human) can read end to end.

Several exceptions are `pytest` modules rather than tools:
`test_constraint_capacity.py`, `test_scene_articulation.py`, and the focused
OVRtx contract tests are real
regression tests and are meant to stay green. Run them with
`python3 -m pytest genie_sim_engine_newton/tools/ -q` from the workspace
`src/` directory. They must run **from this directory** — both resolve
sibling packages via `_HERE.parents[3]`, so copying one elsewhere breaks
collection with an `IndexError` before any test runs.

## Files

| File | Purpose |
|---|---|
| `test_chef_usda_wok_fit.py` | Pure synthetic-geometry regression for fitting the hidden particle-only wok proxy from the authored USDA bowl instead of assuming the procedural frame. |
| `test_chef_usda_noodle_mapping.py` | Pure regression for deterministic tetrahedral barycentric binding, authored vertex order, procedural fallback RNG stability, and all 16 Wuji noodle Mesh identities/hashes. |
| `test_ovrtx_deformable_bindings.py` | Pure validation and CPU-kernel regression for multi-Mesh tet-barycentric reconstruction into authored-order OVRtx point arrays. |
| `test_ovrtx_camera_zoom.py` | Pure focal-scale/clamping and runtime optical-zoom regression for inline OVRtx. |
| `test_scene_plugin_stage_hook.py` | Pins discovery of the plugin `on_stage` hook. |
| `test_newton_solver.py` | Load a USD/URDF, build an adapter + solver, optionally run a step loop with wbc-style command driver. The reference implementation for "what does it take to get a Newton model going from scratch." |
| `test_assemble_robot.py` | Per-stage harness for the build-time `urdf_to_usd` pipeline (xacro → URDF → Isaac importer → AS3 → post-process). Snapshots each stage's output, runs structural validators, supports re-running any contiguous range. |
| `test_robot_xml_static.py` | Structural diff for two MJCFs through pip-installed `mujoco`. Loads both, snapshots `opt` / `joints` / `actuators` / `equalities` / `bodies`, and reports `only_in_a` / `only_in_b` / `attr_diffs` per section. Auto-detects wrapper-namespace prefixes (`_genie_Physics_…`) so the dumped runtime XML lines up with bare reference MJCFs. Cheap (no stepping); CI-friendly (exits non-zero on suspicious findings). Use when the question is "did the converter change break the structure". |
| `test_robot_xml_dynamic.py` | Behavioural / rollout analysis for one or two MJCFs. With `--xml`/`--xml-a` only: single-XML stability check (per-joint amplitude, oscillation count, KE growth ratio). With `--xml-a` + `--xml-b`: same per-side stats AND cross-drift comparison. Supports `--ctrl-mode {hold,zero,sweep}` and a sinusoidal sweep on a chosen actuator for excited-system tests. Output is JSON + a per-side / cross / suspicious summary. Use when the question is "this joint oscillates at runtime, why" — a clean CPU-MuJoCo rollout isolates Newton/mjwarp behaviour from the wrapper. |
| `test_constraint_capacity.py` | **pytest**, not a debug tool. Pins `resolve_constraint_capacity`'s auto-scaling (20×nbody / 7×nbody, floored at 1024/512), `EngineNodeParams`' `_int_nonneg` semantics, and the MJXPBD launcher's deformable-scene capacity guard. |
| `test_scene_articulation.py` | **pytest**, not a debug tool. Ownership-based joint classification: all four `classify_scene_joint` branches, the `JK_SCENE_ARTICULATION` registration, the `drive_scene_articulation` yaml block, and the adapter's zero-gain mapping. `test_robot_joint_names_classify_unchanged` is the regression guard for "existing name-based callers are unaffected". |
| `AGENTS.md` | This file. |

## Pipeline (test_newton_solver.py)

`main()` is a 15-step pipeline.  Each step has a `# === STEP N: ===`
header comment in the source, and each step's log line begins with
`step N`.  Touch the right step when adding behavior:

```
 0   parse args, instantiate adapter
 1   adapter.register_custom_attributes(builder)
 2   builder.add_usd(...) / add_urdf(...) → model.finalize()
 3   _sanitize_model — zero-mass clamp + material defaults
 4   model.state() ×2, model.control(), gravity arrays
 5   adapter.prepare_model(model) + adapter.build_solver(model, …)
 6   _build_joint_map / _build_body_map / _build_mimic_map
 6b  _validate_pd_params  ← gated on --validate-pd or --validate-only
 7   _validate_init_pose_against_limits + _apply_init_pose
 8   adapter.post_joint_map (selective PD for Featherstone)
 9   sync state_0 + state_1 from model.joint_q
10–11 (currently unused)
12   post-load summary + NaN scan; exit if --validate-only
13   _warmup (one uncaptured frame; required before --capture)
14   ViewerGL open (optional)
15   step loop — optionally wrapped in wp.ScopedCapture
```

## Common extension recipes

### Adding a new validator

Validators are pure-Python functions of `(model, [adapter, joint_map, …])`
that walk the live state and log discrepancies.  They go in this file
alongside `_validate_pd_params` and `_validate_init_pose_against_limits`,
with two conventions:

1. Return an integer "suspicious count" so callers can decide whether to
   `return non-zero` and fail the run.
2. Gate the call from `main()` on a dedicated flag (`--validate-pd`,
   `--validate-init-pose`, etc.) AND implicitly fire under
   `--validate-only`.  This makes the tool a useful CI surface: a single
   `--validate-only` run touches every validator once.

Add the call after the step where the relevant data first exists.  For
example, PD params can't be validated until after `adapter.build_solver`
(layers 2+3 live on the solver), so `_validate_pd_params` runs at 6b
(after step 5 + 6).

### Adding a new wbc-style driver

`_WbcDriver` lives in this file (~240 lines).  It produces random
joint-space goals, advances when goals are reached or a timeout
expires, and writes to `control.joint_target_pos` directly.  If you
need a different controller shape (a trajectory, a sinusoid, a CSV
replay) add a sibling driver class and pick it from
`--cmd-mode trajectory|csv|…`.  Adapter-side PD setup is reused; only
the target-writing layer changes.

### Adding a new solver mode

The newton-direct engine's SolverAdapter pattern is mirrored here:
`make_adapter(args.solver)` returns the same adapter the production
engine uses.  To exercise a new adapter (XPBD, Style3D, a custom
solver), extend `_VALID_SOLVERS` in `_parse_args` and ensure the
adapter implements every method on `SolverAdapter`.

## Validator output reading guide

`--validate-pd` prints three sections:

**Layer 1** — Newton model arrays (what the adapter wrote in
`prepare_model`).  Per-DOF `mode`, `ke`, `kd`, `effort` and any flags.
A `mode=NONE` here means no actuator was emitted at all — usually a
typo in the adapter, not a real configuration issue.

**Layer 2** — `mjw_model.actuator_gainprm` / `actuator_biasprm`.  These
are mujoco's per-actuator parameters: `gainprm[0]=kp`, `biasprm=[0,
-kp, -kd, …]` for POSITION mode.  These get baked at
`_convert_to_mjc` time from Layer 1; a disagreement between the two
means `prepare_model` wrote *after* `build_solver` (don't do that).

**Layer 3** — `mjw_model.jnt_actfrcrange` / `jnt_actfrclimited`.  The
per-joint clamp on the total P+D actuator force.  Two failure modes
to watch for:

- `actfrclimited=True, range=(-0,+0)` → URDF didn't author
  `<limit effort=...>` AND your adapter's `_DEFAULT_EFFORT_LIMIT`
  floor isn't catching it → actuator force clamps to zero regardless
  of `kp` → JELLY.
- `actfrclimited=True, kp*0.1rad >> effort_limit` → the spring demands
  far more torque than the actuator can deliver even for a tiny error.
  At any non-zero velocity, `kd*v` damping consumes the saturated
  budget and the net force toward target drops to zero → JELLY again,
  but for a different reason.  Fix: scale `kp` per joint from
  `effort_limit` (the current strategy in `MuJoCoWarpAdapter.prepare_model`)
  or raise `effort_limit` in the URDF.

## Adapter contract (the bit you need to remember)

When extending adapters under `scripts/engine/newton/adapters/`:

- Mutations to `model.gravity` need solver-specific propagation.
  `model.gravity.assign(...)` updates Newton's array but mjwarp keeps
  its own `mjw_model.opt.gravity`; call `solver._update_model_properties()`
  at the top of every `substep` so the two stay in sync.  See
  `MuJoCoWarpAdapter.substep` for the canonical pattern.

- `prepare_model` runs BEFORE `build_solver`.  Anything baked into the
  mujoco/mjwarp internal model (`actuator_gainprm`, `actfrcrange`,
  collision shape ke/kd, etc.) gets frozen at `build_solver` time;
  edits to `model.*` after that point only affect Newton's
  arrays and *will not* propagate.

- Per-substep state mutations (gravity, particle_count,
  shape_contact_pair_count, etc.) that need to affect the captured
  CUDA graph must be done via `wp.array.assign` (a GPU memcpy that
  the graph captures), not Python attribute writes (which the graph
  does NOT capture).  Python writes happen once at capture time and
  freeze.

## When to split this file

Currently ~1400 lines.  Hit ~2000 and split into a package:

```
scripts/tools/test_newton_solver/
    __main__.py       (~50 lines — argparse + step dispatch)
    pipeline.py       (the 15 steps)
    validators.py     (_validate_pd_params, _validate_init_pose_*, _scan_for_nan)
    drivers.py        (_WbcDriver and driver interfaces)
    AGENTS.md         (this file, scoped to the package)
```

The pipeline's step boundaries are natural seam lines — each step
function takes the prior step's outputs and returns the next step's
inputs.  Pure data-flow, no shared state besides `args` and `logger`.

---

# `test_assemble_robot.py` — build-time pipeline harness

The OTHER half of the debug surface.  `test_newton_solver.py` exercises
the Newton runtime starting from an existing `robot.usda`.  This tool
exercises the BUILD that produces that USD: the 9-stage pipeline in
`genie_sim_engine_isaacsim/assemble/_robot_impl.py`, where xacro → URDF →
Isaac importer → AS3 transformer → post-process hooks each can fail
silently.

## Stages

The tool's stage list mirrors `_robot_impl._convert_urdf_to_usd_as3`'s
internal flow.  Stage numbers are stable — when Isaac changes its
importer API and we split stages 2/3/4 apart, the new sub-stages get
numbers in the 2.x / 3.x range without renumbering 5-9.

```
1  xacro              — xacro → robot.urdf + robot_raw.urdf (resolves package://)
2  isaac_pipeline     — Isaac URDF→USD + schema post-proc + AS3 (currently monolithic;
                        could be split when the importer exposes per-step hooks)
5  material_overrides — apply <material_override> PBR patches parsed from raw URDF
6  joint_fixup        — PhysX joint body0/body1 reference rewriting
7  mesh_bake          — bake xformOp transforms into mesh points / normals
8  collision_policy   — selective physics:collisionEnabled per the contact-surface table
9  mimic_overlay      — author per-class joint USD attrs (armature + master drive)
```

Stages 3 and 4 are folded into stage 2; the numbers are reserved for the
day Isaac's importer exposes hooks between schema-post-proc and AS3.

## Snapshot convention

Each stage saves a copy of its primary artifact under
`<workdir>/NN_<name>.<ext>` so an `ls` of the workdir reads the pipeline
top-to-bottom:

```
01_xacro.urdf
01_xacro.raw.urdf
04_isaac_as3.robot.usda
05_material_overrides.usda
06_joint_fixup.usda
07_mesh_bake.usda
08_collision_policy.usda
09_mimic_overlay.usda
robot.usda           # always points at the LATEST stage's output
robot.urdf
robot_raw.urdf
payloads/            # AS3 emits this; later stages mutate it in place
```

Originals (`robot.usda`, `robot.urdf`) stay alongside so subsequent
stages can chain onto them.

## CLI flow

```
# Full pipeline + validators
python3 test_assemble_robot.py --scene scene_flat_g2_sp --workdir test-results/assemble-robot --validate

# Re-run only post-AS3 stages (stages 5-9) against an existing USD
python3 test_assemble_robot.py --workdir test-results/assemble-robot --from-stage 5 --to-stage 9 --validate

# Inspect a single stage in isolation
python3 test_assemble_robot.py --workdir test-results/assemble-robot --only-stage mimic_overlay --validate

# Validators only — re-scan an existing workdir without re-running anything
python3 test_assemble_robot.py --workdir test-results/assemble-robot --validate-only --inspect

# Fresh start (rm -rf workdir first)
python3 test_assemble_robot.py --scene scene_flat_g2_sp --workdir test-results/assemble-robot --clean --validate
```

## Validator pattern

Each stage has TWO validation paths:

1. **Run-time validation** — fired immediately after the stage runs.
   Validator gets the stage's own metrics dict (e.g. `{joints_fixed: 4}`
   from `joint_fixup`, `{masters: 2, arm_shoulder: 4, …}` from
   `mimic_overlay`) and can check exact expected values.

2. **Post-hoc validation** — fired by `--validate-only` against an
   existing workdir, with a generic `_scan_usd_metrics()` dict.  The
   validator can't check stage-specific counters here, but it CAN
   re-walk the USD to confirm structural invariants (e.g. "armature
   authored on >0 joints", "drive only on `inner_joint1` joints").

Each validator detects which mode it's in and degrades gracefully:

```python
def _validate_xacro(result, logger):
    if "link_total" not in result.metrics:        # marker key for run-time mode
        logger.info("[xacro] (no run-time URDF metrics; skipped)")
        return 0
    # … real checks here
```

This split is deliberate: re-running validators is cheap (USD scan),
re-running stages is expensive (Isaac, AS3).

## Adding a new stage

1. Add a `_stage_<name>(args, workdir, logger) -> StageResult` function.
   It MUST snapshot its output to `<workdir>/NN_<name>.<ext>` before
   returning, and populate `metrics` with whatever counts the validator
   needs.

2. Add a `_validate_<name>(result, logger) -> int` function.  Return the
   number of suspicious entries (0 = clean).  Handle both run-time and
   post-hoc inputs (use `if "marker_key" not in result.metrics` to
   detect the post-hoc case and skip / fall back).

3. Append a `Stage(N, "<name>", "...", _stage_<name>, _validate_<name>)`
   to `STAGES`.  Pick N to fit the pipeline's logical order; gaps are
   fine (we leave 3 and 4 reserved).

## Dependencies

`_robot_impl.py` imports Isaac Sim modules (`urdf_usd_converter`,
`isaacsim.asset.transformer.rules`, etc.) for stages 1-4.  Stages 5-9
are pure pxr edits and DON'T need Isaac — that's why the tool's
`--from-stage 5` mode is the fast iteration path: re-run `mimic_overlay`
in 0.5s instead of waiting 30s for Isaac to redo the import.

When iterating on overlay logic, the loop is:

```
edit _robot_impl._apply_mimic_joint_overlay
python3 test_assemble_robot.py --workdir test-results/assemble-robot --only-stage mimic_overlay --validate
diff test-results/assemble-robot/09_mimic_overlay.usda <path-to-reference-mjcf>
```

## When to split this file

`test_assemble_robot.py` is currently ~1340 lines; the same 2000-line
threshold and per-stage seam lines apply.
