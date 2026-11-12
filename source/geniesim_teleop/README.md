# geniesim_teleop — VR / Pico teleoperation 🎮

A VR-driven teleop loop that streams device poses into the simulator
over ROS 2, plus device drivers and a data-recording pipeline that
turns recorded rosbags into HDF5 episodes.

Current startup uses Isaac Sim **6.2**. Isaac Sim **4.5**, standalone **5.1**,
and the **5.1 + 6.2 combo** are **deprecated (E.O.L.)**. The retained
`run-teleop` skill describes the legacy 5.1 workflow; use this guide's 6.2
commands for current operation.

License: [Mozilla Public License Version 2.0](LICENSE)
Third-party dependency notices: [Wuji SDK (MIT)](src/geniesim_teleop/THIRD_PARTY_NOTICES.md).

---

## 🎮 Teleoperation

Support controller-pose tracking for both arms and the waist, plus gripper
and base control.

![teleop controller](docs/teleop_controller.png)

| No. | Function |
|:---:|---|
| 1 | left stick: forward/backward and lateral translation |
| 2 | control left gripper |
| 3 | backtracking action |
| 4 | X: reset both arms, waist and head to the task's initial pose |
| 5 | enable left arm pose tracking |
| 6 | B: start recording |
| 7 | right stick horizontal: turn left/right |
| 8 | control right gripper |
| 9 | hold A: use right-controller motion to control the five-joint waist |
| 10 | enable right arm pose tracking |

Releasing an arm grip holds the last commanded arm pose. Re-gripping starts
controller motion from a new origin while preserving that held target, even if
the simulated wrist has a small tracking error. Engaging the other hand also
preserves the active arm's target. X, playback, and input loss discard these
cached targets; the next grip starts from measured robot poses.

Hold **A** to anchor the current controller pose and waist pose; moving or
rotating the right controller then changes the `arm_base_link` target in
`base_link`, at half scale. MC solves the five waist joints. Release A to hold
the last waist target. If the right grip stays held, right-arm tracking resumes
from a fresh anchor so motion made while controlling the waist is not replayed
on the arm. A does not require the grip button. A five-joint waist cannot match
every arbitrary six-dimensional pose; tracking remains subject to its kinematics
and joint limits.

**X** has priority over pose tracking and resets arms, waist and head together;
release it to resume tracking from fresh anchors. Grippers are not reset.
Ordinary and Wuji collection prepare one shared arm pose for startup and X from
the selected task's initial states. The preparation keeps both wrists' positions
and orientations while adjusting nearby arm joint angles to satisfy MC's joint
and wrist IK bounds. An infeasible solve stops startup. The simulator sets joint
positions and drive targets together, and Pico sends the same prepared pose to
MC before accepting tracking input. Task configuration files are not rewritten.
With `--external-simulator`, start the external DummyEnv with the same
`GENIESIM_TELEOP_CONFIG` environment variable and make `geniesim_teleop`
importable in its Python environment.
**B** only starts recording and does not change right-controller tracking.
L3/R3 remain unassigned. In `--arm_only` mode, A does not engage the waist and X
resets only the arms.

### Chassis / WBC

Pico publishes `geometry_msgs/Twist` on `/cmd_twist` at 30 Hz. Left-stick Y
controls forward/backward; left-stick X controls sideways translation; right-stick
X controls yaw. Pushing right means rightward translation / clockwise rotation
(negative ROS `linear.y` / `angular.z`). Right-stick Y is unused. Speed follows
stick amplitude; the deadzone is 0.05 and diagonal translation is normalized.
Defaults are 0.4 m/s translation and 0.5 rad/s yaw. Optional overrides live in
the same task YAML:

```yaml
teleop:
  chassis:
    max_linear_speed: 0.4
    max_angular_speed: 0.5
    axis_deadzone: 0.05
    input_timeout_s: 0.25
```

The ROS `genie_sim_controllers` standalone node runs `OptimalServo` at 100 Hz,
using validated `/teleop/chassis_state`, and produces steering angles plus wheel velocities.
Both supported robots share the 4WD chassis: axle spacing 0.46 m, track 0.436 m,
wheel radius 0.07 m. `start_chassis.py` launches the existing ROS executable with
its `4wd.yaml` plus teleop-local `config/chassis_servo.yaml` overrides.
The bridge merges `/cmd_4ws`, MC `/hal/joint_cmd`, and Pico
`/teleop/gripper_command` into one `/joint_command` frame at 100 Hz.
Isaac consumes individual frames, so these sources must not publish directly
to `/joint_command`. `utils/chassis_command.py` selects only steering positions
and drive velocities from the original zero-filled `/cmd_4ws` message; unused
fields become NaN only inside teleop (zero is a real target).
After 0.3 s without an input source, the bridge zeros its velocity/effort
commands while retaining its position targets.

In `DummyEnv`, tasks with `robot.free_base: true` initialize under the world
anchor, enable wheel contact, set drive-wheel stiffness to zero, apply all task
joints and their drive targets together, then release the anchor. Fixed-base
tasks keep their anchor. Use the same task YAML for the simulator and teleop;
`benchmark.model_arc: pi` selects DemoPolicy/DummyEnv over the inference defaults.

Build the ROS workspace inside the container from the repository root with
`geniesim ros build dev` before using the full launcher. It sources
`devel/setup.bash` (or the installed `geniesim_ros` wheel overlay) and checks that
the chassis package exists before starting processes. To start the node separately
in a sourced ROS shell:

```bash
python3 -m geniesim_teleop.start_chassis
```

Releasing the stick publishes zero Twist. X reset and Y playback also stop the
base. After 0.25 s without a valid Pico packet, teleop sends zero Twist and stops
velocity-controlled grippers, then reanchors tracking on reconnection. The teleop
bridge independently commands zero wheel speed after 0.3 s without Twist,
complete wheel feedback or a servo command, retaining the last steering target.
Its watchdog uses monotonic time, including when simulation time is paused.
Only complete finite wheel feedback is forwarded to the teleop servo process.
These checks run in the teleop bridge.
Do not run a second chassis-command source on these topics at the same time.

### Pico Setup

1. Connect to the same **LAN** as the computer.
2. Start **AIDEA Vision App** in resource library.
3. Choose **Wireless Connection** and enter the **IP** of the computer.

With one active non-loopback interface containing a single IPv4 address,
teleop selects that address automatically. With multiple interfaces (including
VPNs), startup lists interface names and IPs. Select the Pico network interface:

```bash
# Inside the container, when launching the teleop loop directly:
geniesim teleop run --interface=YOUR_PICO_INTERFACE
# Or select a particular local IPv4 address with --host_ip=IP.

# Outside the container, for the launcher (replace the interface name):
GENIESIM_TELEOP_INTERFACE=YOUR_PICO_INTERFACE \
  ./source/geniesim_teleop/scripts/autoteleop.sh --isaac-version 6.2
```

For a multi-IP interface, select the local Pico-facing address with
`GENIESIM_TELEOP_HOST_IP=IP`. Set only one selector.
Network discovery requires `iproute2`. The selected address is printed and
used for the VR UDP listener; enter this same address in the Pico app.

> **Note:** The AIDEA Vision App has been uploaded; you can access it here:
> https://modelscope.cn/datasets/agibot_world/GenieSim3.0-Dataset/tree/master/app .
> APK installation for the Pico Enterprise Edition HMD can be performed via a
> computer. When the HMD is connected via USB, its internal storage becomes
> accessible as a standard drive on the computer. The APK file can then be
> installed by copying it directly to this drive.

### Launch Setup

On a new Linux workstation, prepare the NVIDIA driver, Docker Engine and NVIDIA
Container Toolkit, a graphical terminal, the repository (including Git LFS
files), and the matching asset package. From the repository root, install the
host CLI and point Docker at the downloaded asset package root (the directory
containing its `pyproject.toml`):

```bash
python3 -m venv .venv-cli
source .venv-cli/bin/activate
python -m pip install -e source/geniesim_cli
git lfs pull
export GENIESIM_ASSETS_SRC=/path/to/geniesim_assets
```

Teleop supports Isaac Sim **6.2**. Both collection launchers start simulation
and controls by default, using `docker/Dockerfile` and container `geniesim4`.
The simulator uses system `python3.12`; controls use system `python3` 3.12.
All three host scripts accept `--isaac-version 6.2`; other versions are rejected.

Prepare the container:

```bash
geniesim docker6.2 build
geniesim docker6.2 up
geniesim docker6.2 into
# Inside the container, once for ordinary chassis control:
cd /workspace
geniesim ros build dev
exit
```

An existing container must be recreated after rebuilding to use the new image
dependencies.
The ROS controller workspace is a separate prerequisite for ordinary chassis
control. Wuji arm/finger control does not launch the chassis servo.

#### 1. Data Collection

Run outside the container, from the repository root:

```bash
# Starts Isaac Sim 6.2 plus bridge, MC, chassis and Pico.
./source/geniesim_teleop/scripts/autoteleop.sh

# Attach controls to an existing simulator with the same task and ROS domain.
./source/geniesim_teleop/scripts/autoteleop.sh --external-simulator
```

Pico waits for joint states and TF from the simulator. Collection forces
`model_arc: pi`, `policy_class: DemoPolicy`, `env_class: DummyEnv`,
`interactive: true`, and `record: false`. This starts the task for manual control
without automatic evaluation or recording; Pico B starts recording.
An external simulator must use those same settings, task and ROS domain.

For task/scene selection, configuration examples and matching post-processing
commands, see [Changing collection tasks and scenes](#changing-collection-tasks-and-scenes).

The task JSON `benchmark/config/eval_tasks/<task_name>.json` supplies
`robot.robot_cfg`; teleop and motion control both follow this selection.
`G2_omnipicker` uses outer-actuator position control (idx41/idx81,
0 closed / 0.785 rad open); its inner joints are mimic followers. `G2_crsB_90d` uses
outer-joint velocity control. Initial poses and the motion-control URDF also
follow the task. You do not need to set the gripper type separately.

MC runs from a generated runtime under `app/bin/.cache/mc/<robot>/<task>/`.
The bundled binary requires the platform selector `G2`, but its generated URDF
and joint limits come from the selected task (including crsB 90d). The runtime
has its own model/configuration paths and enables `/hal/joint_cmd`; it does not
overwrite the bundled G2 model. The following prepares it without starting MC:

```bash
python3 -m geniesim_teleop.start_mc --config FILE --prepare-only
```

For a read-only package tree,
pass `--cache-dir DIR` to that module to use a writable runtime directory.

For a different YAML, set `GENIESIM_TELEOP_CONFIG` to its container-side path
when calling `autoteleop.sh`, or pass `--config FILE` to `geniesim teleop run`.
With the container running, the launcher starts Isaac Sim 6.2 with that YAML,
then the bridge, task-specific motion controller, ROS chassis servo and Pico.
Pico waits for simulator TF before initializing tracking.
`GENIESIM_TELEOP_SIM_PY` can select a prepared simulator interpreter
(default `python3.12`); controls use system `python3`.
The simulator runs from the mounted repository root. Saving/discarding a session
or exiting the script sends SIGINT to the simulator process group started by
this run and waits for recording shutdown to complete without a forced-kill
deadline. The recorder flushes the bag and writes `metadata.yaml` before the
simulator exits. The launcher marks only episodes created during this session
after shutdown; unfinished new recordings remain unmarked and are reported as
errors.
`--external-simulator` skips simulator startup when deliberately using an already
running instance with the same YAML.
Arm-only mode does not start the chassis servo. The container and ROS controller
build must already be prepared; these launchers do not build or create containers.

An optional `--robot_cfg` must agree with the task; mismatches are rejected.

#### 2. Data Post Process

Execute the following command under `genie_sim` root directory outside the docker
container:

```bash
genie_sim$ ./source/geniesim_teleop/scripts/autoteleop_post_process.sh --isaac-version 6.2 {SUB_TASK_NAME}
# for example
genie_sim$ ./source/geniesim_teleop/scripts/autoteleop_post_process.sh --isaac-version 6.2 open_door
```

> **Note:** To improve collection efficiency, data collection and data
> post-processing can be run in parallel.

#### Episode output

G2 omnipicker, crsB omnipicker, crsB 90D and Wuji recordings use the
`data_collection` output format through
`data_recording/sim_data_converter.py`. Geometry helpers, hardware templates,
joint metadata and FK URDFs ship with teleop. Post-processing runs independently
of the `data_collection` directory.

The exported episode includes:

- `aligned_joints.h5` and an identical `record/aligned_joints.h5`: canonical
  per-frame fields, joint-name attributes, radian grippers, 28-value end poses,
  reference orientations and monotonic timestamps.
- `parameters/sensor` and `parameters/hardware`: camera intrinsics, resolution,
  aligned/static extrinsics and robot/gripper hardware metadata.
- `meta_info.json`, `data_info.json`, `review/meta.json`, `review/state.json`,
  and the existing `observations/videos`.

Actions still describe measured teleop trajectories. Actual chassis poses are
retained. Unrecorded chassis commands and VR buttons use the collection schema's
placeholders, documented in `post_proc/teleop_source.json`; planner phase labels
remain empty. Wuji uses this same format with 20 measured joints per hand.

Completion is recorded in `post_proc/teleop_export.json` only after conversion
succeeds. Bags are retained and failed conversions retry. Valid intermediate
`state.json`/`aligned_joints_all.h5` can be upgraded even without the source bag;
recordings with incorrect joint metadata need their original bag.

### Changing collection tasks and scenes

Stop the current collection session before changing its configuration. Start
from a copy of the ordinary `config/teleop.yaml` or Wuji `config/wuji_teleop.yaml`,
preserving the ROS/camera settings and `benchmark.model_arc: pi`,
`policy_class: DemoPolicy`, and `env_class: DummyEnv`. Evaluation YAMLs may select
an inference policy or disable recording inputs.

Within `source/geniesim_benchmark/src/geniesim_benchmark/`, the task fields
select different parts of the scene:

- `benchmark.task_name` selects
  `benchmark/config/eval_tasks/<task_name>.json`, which defines
  the background scene, robot configuration and initial base pose.
- `benchmark.sub_task_name` selects the task's instructions and foreground
  instances under `benchmark/config/llm_task/<sub_task_name>/`. It also identifies
  the recording directory and the task's initial joint poses.
- Optional `benchmark.instance_ids: "0"` selects the existing instance `0`;
  use one ID per manual collection session. To change the layout, select another
  existing ID and restart the session. Wuji manual mode does not automatically
  advance to another instance.
  The foreground normally comes from `<instance>/scene.usda`; a task JSON may
  specify `scene.sub_usd_override_root` for an alternate foreground directory.

Use a matching task/sub-task pair. Existing examples in this checkout are:

| Collection mode | `task_name` | `sub_task_name` | Task's `robot.robot_cfg` |
|---|---|---|---|
| Ordinary Pico, Omnipicker (default) | `home_g2_op` | `open_door` | `G2_omnipicker.json` |
| Ordinary Pico, 90D | `phone_packaging_g2_crsb90d` | `phone_packaging` | `G2_crsB_90d.json` |
| Wuji (default) | `table_task_g2_crsB_wuji_take_the_scissors` | `take_the_scissors` | `G2_crsB_wuji.json` |

For example, prepare an ordinary 90D collection configuration on the host:

```bash
cp source/geniesim_benchmark/src/geniesim_benchmark/config/teleop.yaml \
  source/geniesim_benchmark/src/geniesim_benchmark/config/teleop_custom.yaml
```

Edit these fields in the copied YAML's existing `benchmark` section, keeping
its other settings:

```yaml
benchmark:
  task_name: phone_packaging_g2_crsb90d
  sub_task_name: phone_packaging
  instance_ids: "0"
```

Run on the host with the local Pico-facing IP selected as described above. The YAML
path passed to Docker is its **container-side** path (`/workspace` by default):

```bash
unset GENIESIM_TELEOP_INTERFACE
GENIESIM_TELEOP_CONFIG=/workspace/source/geniesim_benchmark/src/geniesim_benchmark/config/teleop_custom.yaml \
GENIESIM_TELEOP_HOST_IP=YOUR_LOCAL_PICO_IP \
  ./source/geniesim_teleop/scripts/autoteleop.sh --isaac-version 6.2

# In another terminal; process recordings for the selected sub-task.
./source/geniesim_teleop/scripts/autoteleop_post_process.sh --isaac-version 6.2 phone_packaging
```

For Wuji, copy its default YAML, choose a task whose robot family is
`G2_crsB_wuji`, and pass the copy with `--config`:

```bash
cp source/geniesim_teleop/src/geniesim_teleop/config/wuji_teleop.yaml \
  source/geniesim_teleop/src/geniesim_teleop/config/wuji_custom.yaml

./source/geniesim_teleop/scripts/autoteleop_wuji.sh --isaac-version 6.2 \
  --config /workspace/source/geniesim_teleop/src/geniesim_teleop/config/wuji_custom.yaml \
  --host-ip YOUR_LOCAL_PICO_IP \
  --left-sn YOUR_LEFT_GLOVE_SN --right-sn YOUR_RIGHT_GLOVE_SN
```

After editing the Wuji task fields, use that YAML's `sub_task_name` as the
post-processing argument too. The launcher preserves manual recording mode.

Robot selection follows the task's `robot.robot_cfg`: `G2_crsB_90d` uses velocity
gripper control; `G2_omnipicker` and `G2_crsB_omnipicker` use position control.
The URDF, MC configuration, initial/reset pose and recording joint mapping follow
that selection. Wuji uses the separate launcher and 20-joint glove input per hand.
There is no separate 90D/Omnipicker launch switch, and `--robot_cfg` only checks
consistency with the task. Use one of the supported families listed above.
A new scene or robot combination needs a matching task JSON,
scene assets and `robot_init_states.py` entry; changing a name alone does not
create them. For an external simulator, restart it with the same selected YAML
and ROS domain; changing the control configuration does not reload its scene.

## Wuji teleoperation (separate launcher)

Use [`scripts/autoteleop_wuji.sh`](scripts/autoteleop_wuji.sh) for Pico arm
tracking plus Wuji glove finger control. It starts Isaac Sim 6.2,
the bridge, MC, Pico and SDK hand input by default.
Use `--external-simulator` to attach to an existing Wuji simulator, which must
publish joint states and TF in the selected ROS domain (91 by default).
`--source retargeting` uses an external hand publisher that you start separately. Ordinary collection continues to use
`autoteleop.sh` and its existing configuration. The Wuji entry point uses its own process supervisor and reuses ordinary
B-to-record behavior. Its simulator uses `benchmark.interactive: true`, which
disables automatic evaluation and episode resets. After recording, enter
`Y` + Enter to keep or `N` + Enter to discard and stop a launcher-owned simulator.
The supervisor waits for recording shutdown before marking only new episodes;
Ctrl-C leaves them unmarked. External simulators must be finalized by their owner.
It currently controls arms and fingers; waist and chassis control are disabled.

The launcher runs inside an existing Genie Sim container with the source checkout
mounted at `/workspace`. It requires the sourced ROS environment, `xacro`, `iproute2`, the
bundled MC runtime, and (for SDK input) `wuji_sdk` exposing `RetargetSession` in
the container's system `python3`. The simulator uses `python3.12`; controls use
the system Python apt NumPy overlay. `docker/Dockerfile` (`geniesim4`)
bakes the teleop Python dependencies, Wuji SDK `2026.8.17`, and ROS/network tools
into system Python 3.12 and the operating system. Rebuild the image when its
dependency definitions change; updating the checkout alone does not add
packages to an existing image.
The ROS controller workspace must still be built for ordinary chassis control.

Replace the interface and glove serial numbers below with your devices:

```bash
# Preview commands without starting processes or connecting hardware (host needs xacro).
./source/geniesim_teleop/scripts/autoteleop_wuji.sh --dry-run --source retargeting

# Check dependencies, network selection and domain availability without connecting gloves.
./source/geniesim_teleop/scripts/autoteleop_wuji.sh \
  --check --interface YOUR_PICO_INTERFACE --right-sn YOUR_RIGHT_GLOVE_SN

# Run from an interactive terminal; Ctrl-C stops this session's process groups.
./source/geniesim_teleop/scripts/autoteleop_wuji.sh --isaac-version 6.2 \
  --interface YOUR_PICO_INTERFACE \
  --left-sn YOUR_LEFT_GLOVE_SN --right-sn YOUR_RIGHT_GLOVE_SN
```

If one interface has multiple IPv4 addresses, replace `--interface` with
`--host-ip <LOCAL_PICO_LAN_IP>` (alias `--host_ip`). An interface name alone
cannot identify the correct subnet; the launcher lists candidates instead of
selecting an arbitrary address. Do not pass both selectors.

One glove is supported: supply only its side's serial number. `--sdk-user`
selects an existing calibrated SDK user by ID or exact display name. This
launcher does not perform glove calibration. Pico grip buttons track the arms,
and X resets the arms; B starts recording. A, Y, sticks and gripper triggers are unused here.

The arm path reuses the ordinary Pico `TeleOp`, task-specific `prepare_mc`,
and bridge: `/joint_states` → `/hal/joint_state`, Pico → `/wbc/retarget` →
MC `/hal/joint_cmd` → merged `/joint_command`. After TF initialization the
Wuji entry sends the **IK-compatible initial arm positions derived from the task** through MC and waits
up to 15 seconds for both arm commands to reach those targets on `/joint_command`,
then enables tracking once arm feedback also reaches the prepared pose. Startup
and X use identical targets and the same generated Wuji model. This first target
starts MC's arm output; glove frames alone do not start the arms. Fingers remain
controlled by the glove. Hold the Pico grip buttons to move arms.
If startup fails, inspect `pico.log`, `mc.log`, and `bridge.log` in the printed
session directory; the supervisor prints the last lines of a failed child.

Defaults are **ROS domain 91**, **Pico UDP port 8080**, and the existing Wuji
scissors scene. Enter the printed IP and port in Pico. `--domain-id` and `--port`
override these values; domain 0 is rejected. The default matches the AIDEA App
fixed to UDP 8080, so no port argument is needed. Use `--port 8081` only when
the App can send to that port. Only the selected port is listened on. ROS domains
do not isolate UDP sockets: do not run ordinary teleop on the same IP and port
at the same time.
The selected IP must be reachable from the App; with Docker bridge networking,
forward the matching **UDP** port or use host networking.
Use `--config` with a container-side YAML path to
select another Wuji task; see
[Changing collection tasks and scenes](#changing-collection-tasks-and-scenes).
The default is
[`config/wuji_teleop.yaml`](src/geniesim_teleop/config/wuji_teleop.yaml).
`--external-simulator` skips simulator startup; the existing simulator must use
the same Wuji task and ROS domain. `--headless` applies to the launched simulator.

Container selection uses `GENIESIM_WUJI_CONTAINER` (default `geniesim4`), mount
location uses `GENIESIM_WUJI_WORKSPACE` (default `/workspace`), and simulator
interpreter selection uses `GENIESIM_WUJI_SIM_PY` (default `python3.12`).
These are separate from ordinary collection's environment variables.
**For concurrent ordinary collection, use a separate container and ROS domain.**
The bundled MC launcher rejects another MC process within its container;
separate ROS domains cannot isolate this singleton check. Both launchers stop
only their own process groups. Ordinary collection uses `teleop_process.sh`
to wait for recording finalization. A forced control shutdown fails the session.
Concurrent simulators still share GPU resources and may affect each
other's performance.

Post-process kept recordings with the same watcher and converter:

```bash
./source/geniesim_teleop/scripts/autoteleop_post_process.sh --isaac-version 6.2 take_the_scissors
```

The final `aligned_joints.h5` retains the ordinary per-frame `action` / `state`
structure. For Wuji, `left_effector/position` and `right_effector/position` each
contain 20 joint angles in radians, ordered thumb, index, middle, ring, pinky,
with joints 1–4 per finger. Each effector group's `name` attribute documents this
order, matching ordinary collection. Arms, head and waist are resolved by joint name, so finger count cannot
shift their columns. As in ordinary collection, action is derived from the
aligned measured trajectory, not a separately recorded command topic.
Missing or duplicate required joints fail conversion instead of emitting
misaligned columns. Wuji raw bags are retained for reprocessing; an old H5 that
already dropped fingers cannot reconstruct them without the source bag.

Wuji wrist calibration uses `opencvFisheye` intrinsics, including `k1`–`k4`,
under the same sensor directory and camera names as ordinary collection.
The recorder reads the authored lens model instead of pinhole-only attributes.
Depth topics retain the exact USD basename (including `Head_Front_Camera`),
while extracted camera files use canonical names such as `head_depth.png`.
Legacy `"None"` calibration values become JSON null; conversion completes with
invalid calibration flagged in `post_proc/calibration_status.json`,
`meta_info.json` and `data_info.json`. Missing intrinsics or unrecorded depth
cannot be recovered from joint data. Valid intermediate JSON/H5 files can be
upgraded without extracting the bag again.

Each launch writes logs, the resolved YAML, generated robot model, MC runtime
and process IDs beneath `output/wuji_teleop/<session>/`. The canonical ROS Wuji
xacro supplies all 40 finger joints and their limits. Generated models and MC
runtimes are scoped to the session.

For an already configured external retargeter, use `--source retargeting` instead
of serial numbers. Run the retargeter in the same ROS domain and publish
`sensor_msgs/JointState` to `/wuji_hand/joint_command`. Each frame must contain
all 20 joints of each included hand, with positions in radians. Accepted names
are `left_finger1_joint1` through `left_finger5_joint4` (and `right_…`), or the
canonical `hand_l_thumb_joint1` / `hand_r_…` names. Finger order is thumb, index,
middle, ring, pinky. A nonzero timestamp must use synchronized host wall time;
zero timestamps use receipt time. Do not also run the standalone Wuji hand bridge
to `/joint_command`: the Wuji teleop bridge merges fingers and MC arms into the
single publisher on that topic.

Invalid/incomplete hand frames are rejected and positions are clamped to the
generated URDF limits. With two SDK gloves, publishing begins after fresh input
from both; thereafter each hand updates independently. Input loss holds the last
position, reports a warning, and does not re-stamp cached glove frames as new
input.

The SDK capture, retargeting and name-mapping adapter belongs to
[`wuji/hand_bridge.py`](src/geniesim_teleop/wuji/hand_bridge.py), imported directly
by `wuji/sdk.py`. It also provides a standalone CLI for the separate Newton
chef workflow. From the checkout root in a sourced ROS shell:

```bash
PYTHONPATH="source/geniesim_teleop/src:${PYTHONPATH:-}" \
python3 -m geniesim_teleop.wuji.hand_bridge \
  --source sdk --right-sn YOUR_RIGHT_GLOVE_SN
```

With no arguments it bridges external `/wuji_hand/joint_command` input to
`/joint_command`; SDK mode publishes there directly as well. Use this standalone
entry only with a separate arm-only controller.
