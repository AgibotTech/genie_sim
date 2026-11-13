# Benchmark usage

50 added scored tasks and 25 sandbox tasks. [License](LICENSE).

<a id="ai-smoke-testing"></a>

## 1. Build and start

Use `registry.agibot.com/genie-sim/geniesim4:latest` (container `geniesim4`),
with Isaac Sim 6.2, system Python 3.12 and ROS Jazzy. Isaac, Newton and libuipc
benchmarks select system Python 3.12 even when `omni_python` is available.
Requires an NVIDIA GPU and the task assets.

```bash
# Host, repository root; install this checkout's CLI if needed.
python3 -m venv .venv-cli
source .venv-cli/bin/activate
python3 -m pip install -e source/geniesim_cli
export GENIESIM_ASSETS_SRC=/path/to/geniesim_assets
geniesim docker6.2 build             # Optional: add --china.
geniesim docker6.2 up --headless
geniesim docker6.2 logs              # Wait for GENIESIM_READY.
geniesim docker6.2 into
```

```bash
# Container, one-time workspace preparation.
cd /workspace
export SIM_REPO_ROOT="$PWD"
export SIM_ASSETS=/geniesim_assets
export GENIESIM_ASSETS_PATH="$SIM_ASSETS"
source /opt/ros/jazzy/setup.bash
geniesim ros build dev
source devel/setup.bash
geniesim status
```

For a standalone Rubik environment, use Python 3.10–3.12 and install the runtime
dependencies before installing Benchmark with `--no-deps`:

```bash
# Host, repository root.
python3 -m venv .venv-rubik
source .venv-rubik/bin/activate
python -m pip install "mujoco==3.12.0" numpy scipy msgpack "websockets>=12" "kociemba==1.2.1" Pillow PyYAML
python -m pip install -e source/geniesim_cli
python -m pip install --no-deps -e source/geniesim_benchmark
export GENIESIM_ASSETS_PATH=/path/to/geniesim_assets
```

## 2. List tasks

```bash
geniesim benchmark categories
geniesim benchmark robots
geniesim benchmark list --category=finemanip
geniesim benchmark list --robot=g2crsbwuji --category=dexmanip
geniesim benchmark list --category=wbc
geniesim benchmark list --category=crsemb
geniesim benchmark list --category=sandbox
```

Counts: `finemanip=8`, `dexmanip=10`, `wbc=8`, `crsemb=24`, `sandbox=25`.

<a id="newton-wok"></a><a id="deformable-evaluation"></a><a id="mujoco-rubik"></a>

## 3. Run a task

Every task uses the same command. Replace `TASK` with a name from the task list;
the CLI selects the required runtime automatically.

```bash
export INFER_HOST=127.0.0.1:8999       # Set the actual policy endpoint.
geniesim benchmark check-inference --infer-host="$INFER_HOST"

TASK=g2crsb90d_finemanip_phone_packaging
geniesim benchmark run "$TASK" --infer-host="$INFER_HOST" \
  --benchmark.num_episode=1
```

`--benchmark.num_episode=1` requests one evaluation pass; tasks with multiple
instructions may execute one episode per instruction. Omit it to use task defaults.

### Joint and EEF control

Supported robots accept `JOINT_ABS` or `EEF_ABS` for both arms. EEF rows are
`[x,y,z,qx,qy,qz,qw]` in metres, with matching per-arm `base_link` set to
`base_link` (default) or `arm_base_link`. Effector channels retain their usual
dimensions, including all 40 Wuji finger targets.

IK/FK uses the bundled `ik_solver` wheel (Pinocchio + NLopt). It uses the
runtime robot USD when supplied, the loaded native model for MuJoCo Rubik,
and the matching embodiment URDF otherwise. EEF commands must satisfy joint
limits and pose tolerances; a solution outside those tolerances raises an
error without returning an approximate command. Legacy `robot.ik_backend`
values remain compatible aliases for the same solver.

The 6.2 image includes the matching native wheel. For a separately prepared
Python 3.12 environment, install the repository wheel:

```bash
python3.12 -m pip install --no-deps 3rdparty/ik_solver-0.4.3-cp312-cp312-linux_x86_64.whl
```

Run the kinematics checks from the repository root inside the 6.2 container:

```bash
GENIESIM_IK_TEST_ASSETS=/geniesim_assets python3 - <<'PY'
import isaacsim
import pytest
raise SystemExit(pytest.main([
    "-q", "-rs",
    "source/geniesim_benchmark/tests/test_hard_ik.py",
    "source/geniesim_benchmark/tests/test_ik_runtime_models.py",
]))
PY
```

These checks cover FK/IK, frames, joint limits, effector preservation and
available USD/native models. Missing USD assets are reported as skips;
they do not count as verified models or full task evaluations.

Rubik clips finite absolute arm joint targets to the declared joint limits before
execution. Arm limit saturation alone does not invalidate the episode.

### Resume an interrupted evaluation

Use the same command and output directory on every attempt:

```bash
geniesim benchmark run g2op_if_pick_block_color --infer-host="$INFER_HOST" \
  --benchmark.resume=true --benchmark.resume_id=job-123
```

`resume_id` is optional; when supplied, keep the same job/model identifier from
its first attempt. The inference address may change, but the evaluated model,
task inputs, seed, episode count and scoring configuration must stay the same.

Serial Corobot evaluations with a named subtask write `checkpoint_manifest.json` beside their result
under `output/benchmark/<task_name>/<sub_task_name>/`. Resume skips committed
episodes, retries the interrupted episode and updates the same result file;
`checkpoint.result_file` identifies that file. Successes, task failures and
timeouts remain in the score. Temporary inference errors are retried on restart.
A fatal simulator error remains a scored invalid attempt; its skipped instruction
range is retained and reported separately in the manifest.

With `--benchmark.record=true`, recordings also save per-episode `state.jsonl`
files containing measured joint names, positions and velocities, base pose and
velocity, simulation time and frame indices. State is sampled on the same ticks
as video; after finalization, files are named `recording_<episode>_*.state.jsonl`
alongside the videos, including under `record_video_dir` when configured.
Use `--benchmark.record_state=false` to explicitly disable state recording.

With `--benchmark.record=true`, committed recordings are reused. A retry after
all episodes finish only rebuilds/publishes artifacts; it does not evaluate again.
`record_video_dir` receives media and the final score file, while the canonical
score and checkpoint stay in the task output directory. Keep that directory,
including `_checkpoint_tasks`, until the job has been submitted successfully.

Without `resume`, a new evaluation starts. Missing manifests start a new run;
pre-checkpoint results cannot be resumed. Vectorized, preview, interactive and
non-Corobot modes and unnamed subtasks ignore resume with a warning. One worker may write a task
output directory at a time. Resuming preserves committed scores, rather than
promising identical physics or model randomness across process restarts.

## 4. Test every added task

Define this helper once. Each command below runs one task with the same parameters
and saves its log. Run tasks sequentially; use a policy endpoint that supports the
selected task and robot.

```bash
LOG_DIR="output/smoke/$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$LOG_DIR"
smoke() {
  local task_name="$1"
  geniesim benchmark run "$task_name" --infer-host="$INFER_HOST" \
    --benchmark.num_episode=1 >"$LOG_DIR/$task_name.log" 2>&1
  local task_status=$?
  printf '%s exit=%s log=%s\n' "$task_name" "$task_status" "$LOG_DIR/$task_name.log"
  return "$task_status"
}
```

### Fine manipulation (8)

```bash
smoke g2crsb90d_finemanip_insert_building_block
smoke g2crsb90d_finemanip_insert_coin_into_money_box
smoke g2crsb90d_finemanip_insert_ethernet_cable
smoke g2crsb90d_finemanip_insert_straw_into_milk_tea_cup
smoke g2crsb90d_finemanip_install_bearing_shaft_seat5mm
smoke g2crsb90d_finemanip_phone_packaging
smoke g2crsb90d_finemanip_plug_charging_cable_into_phone
smoke g2crsb90d_finemanip_put_the_cap_on_the_pen_new
```

### Dexterous manipulation (10)

```bash
smoke g2crsbwuji_dexmanip_cloth_hanger
smoke g2crsbwuji_dexmanip_cover_cup_lid
smoke g2crsbwuji_dexmanip_drawstring_bag
smoke g2crsbwuji_dexmanip_fold_cloth
smoke g2crsbwuji_dexmanip_pick_up_a_package
smoke g2crsbwuji_dexmanip_rubik_cube_solve
smoke g2crsbwuji_dexmanip_stack_bowls
smoke g2crsbwuji_dexmanip_take_the_book
smoke g2crsbwuji_dexmanip_take_the_scissors
smoke g2crsbwuji_dexmanip_wok_flip
```

### Whole-body manipulation (8)

```bash
smoke g2crsb90d_wbc_clean_the_blackboard
smoke g2crsb90d_wbc_match_and_pick_from_conveyor
smoke g2crsb90d_wbc_open_the_door
smoke g2crsb90d_wbc_open_the_roller_shutter
smoke g2crsb90d_wbc_pick_up_trash
smoke g2crsb90d_wbc_put_the_plates_on_the_shelf
smoke g2crsb90d_wbc_deliver_tissues
smoke g2crsb90d_wbc_transport_cartons_to_table
```

### Cross-embodiment (24)

```bash
smoke agilex_crsemb_hang_cups_on_cup_tree
smoke agilex_crsemb_keyboard_keycap_swap
smoke agilex_crsemb_open_short_trash_can_and_throw_garbage
smoke agilex_crsemb_pour_50ml_into_graduated_cylinder
smoke agilex_crsemb_scrape_chicken_bone_into_bin
smoke agilex_crsemb_write_uppercase_letter_with_pen
smoke arxone_crsemb_hang_cups_on_cup_tree
smoke arxone_crsemb_keyboard_keycap_swap
smoke arxone_crsemb_open_short_trash_can_and_throw_garbage
smoke arxone_crsemb_pour_50ml_into_graduated_cylinder
smoke arxone_crsemb_scrape_chicken_bone_into_bin
smoke arxone_crsemb_write_uppercase_letter_with_pen
smoke g1_crsemb_hang_cups_on_cup_tree
smoke g1_crsemb_keyboard_keycap_swap
smoke g1_crsemb_open_short_trash_can_and_throw_garbage
smoke g1_crsemb_pour_50ml_into_graduated_cylinder
smoke g1_crsemb_scrape_chicken_bone_into_bin
smoke g1_crsemb_write_uppercase_letter_with_pen
smoke g2op_crsemb_hang_cups_on_cup_tree
smoke g2op_crsemb_keyboard_keycap_swap
smoke g2op_crsemb_open_short_trash_can_and_throw_garbage
smoke g2op_crsemb_pour_50ml_into_graduated_cylinder
smoke g2op_crsemb_scrape_chicken_bone_into_bin
smoke g2op_crsemb_write_uppercase_letter_with_pen
```

### Sandbox (25)

```bash
smoke g2crsb90d_sandbox_home_clear_table
smoke g2crsb90d_sandbox_home_deliver_breakfast
smoke g2crsb90d_sandbox_home_open_bedroom_doors
smoke g2crsb90d_sandbox_home_store_bowls
smoke g2crsb90d_sandbox_home_tidy_sofa
smoke g2crsb90d_sandbox_hotel_deliver_cola
smoke g2crsb90d_sandbox_hotel_discard_tissue
smoke g2crsb90d_sandbox_hotel_retrieve_umbrella
smoke g2crsb90d_sandbox_hotel_store_suitcase
smoke g2crsb90d_sandbox_hotel_tidy_lounge
smoke g2crsb90d_sandbox_office_deliver_folder
smoke g2crsb90d_sandbox_office_discard_papers
smoke g2crsb90d_sandbox_office_supply_pens
smoke g2crsb90d_sandbox_office_supply_water
smoke g2crsb90d_sandbox_office_tuck_chairs
smoke g2crsb90d_sandbox_restaurant_clear_cup
smoke g2crsb90d_sandbox_restaurant_deliver_juice
smoke g2crsb90d_sandbox_restaurant_guide_guests
smoke g2crsb90d_sandbox_restaurant_replace_plate
smoke g2crsb90d_sandbox_restaurant_serve_tray
smoke g2crsb90d_sandbox_store_carry_carton
smoke g2crsb90d_sandbox_store_checkout_juice
smoke g2crsb90d_sandbox_store_discard_rubbish
smoke g2crsb90d_sandbox_store_return_juice
smoke g2crsb90d_sandbox_store_stack_crates
```

To run an entire category:

```bash
geniesim benchmark batch --category=finemanip --infer-host="$INFER_HOST" \
  --benchmark.num_episode=1
```

<a id="results"></a>

## 5. Check results

Check each process exit code, its log and newly generated artifacts under
`output/`. Result paths are printed in the run log. Smoke testing checks task
startup, policy communication and result generation; task success requires its
completion score. Sandbox tasks have no completion score.

Image qualification limits: [runtime acceptance](../geniesim_ros/docs/ISAACSIM_ACCEPTANCE.md).

<a id="published-task-isolation"></a>

Keep published `g2op_if_*`, `g2op_manip_*`, `g2op_robust_*`, `g2op_spatial_*`
inputs and behavior unchanged; new instructions/scenes/scoring require an
independent foreground and registration. Physics settings belong in scene/launcher YAML.

### Tissue delivery task name

The tissue-delivery entry is `g2crsb90d_wbc_deliver_tissues` (送纸巾). The CLI accepts
`g2crsb90d_wbc_sending_food` as a legacy alias, without listing a duplicate task.
The underlying `sending_food` scene, scorer and checkpoint keys remain unchanged
for compatibility; the instruction is `Deliver tissues`.
