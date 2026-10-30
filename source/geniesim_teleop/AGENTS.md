# geniesim_teleop — Agent Development Guide

The teleoperation runtime: a VR / Pico-driven teleop loop that streams
device poses into the simulator over ROS 2, plus device drivers and a
data-recording pipeline that turns recorded rosbags into HDF5 episodes.
The host scripts launch complete collection sessions. The **`geniesim teleop`
CLI verb**, owned by [`geniesim_cli`](../geniesim_cli/), launches individual
control processes.

Source: [source/geniesim_teleop/](.)
License: [Mozilla Public License Version 2.0](LICENSE)
Third-party notices live in `src/geniesim_teleop/THIRD_PARTY_NOTICES.md`;
the Wuji SDK's upstream MIT license is copied to `THIRD_PARTY_LICENSES/LICENSE.WUJI_SDK`
inside the Python package and to the repository's `THIRD_PARTY_LICENSES/` directory.
Keep both copies identical and include the package copy/notices in wheel data.
Skills: [skills/](skills/)

> **Maintenance contract** — when you add a subcommand, rename a module
> entry point, change the package layout, or touch the intra-package
> import paths, **update this file in the same diff**. Agents read this
> as the source of truth.

---

## 1. The CLI surface

Individual control processes use `geniesim teleop <subcommand>`. If the
`geniesim` binary isn't on `$PATH`, substitute
`python3 -m geniesim_cli teleop …` — same dispatch, same args.

| Subcommand | What it does |
|---|---|
| `run [flags...]` | Launch the VR / Pico teleop loop (`python -m geniesim_teleop.teleop`) |
| `bridge [--mode inprocess]` | Launch the command merger and joint feedback bridge (`python -m geniesim_teleop.bridge`) |

`geniesim teleop` (no subcommand) prints the usage table — keep that
output and this file in sync.

The isolated Wuji workflow has a dedicated shell entry point,
[`scripts/autoteleop_wuji.sh`](scripts/autoteleop_wuji.sh), for Pico arm tracking
and glove finger control. See [README — Wuji teleoperation](README.md#wuji-teleoperation-separate-launcher)
for prerequisites and commands. It launches `geniesim_teleop.wuji.launch` and
uses the canonical benchmark CLI for its 5.1 simulator child.
All three host scripts source `scripts/teleop_runtime.sh`. `--isaac-version 5.1`
selects `geniesim3` and `omni_python`, starting simulation unless explicitly
external. `--isaac-version 6.2` (default) selects `geniesim4` and always skips
simulation; Wuji forwards `--external-simulator`. Keep mode selection shared.
Control processes use system Python 3.12 in both modes.

### Forwarded flags (run)

Anything not consumed by the verb is forwarded verbatim to
`geniesim_teleop.teleop`'s `argparse`:

| Flag | Effect |
|---|---|
| `--client_host=H:P` | gRPC client host:port (default `localhost:50051`) |
| `--interface=NAME` | Select the network interface on the Pico LAN; required on multi-interface hosts unless `--host_ip` is supplied |
| `--host_ip=IP` | Select a local IPv4 address explicitly; mutually exclusive with `--interface` |
| `--port=N` | VR server port (default `8080`) |
| `--config=F` | Task YAML (default benchmark `config/teleop.yaml`); resolves robot from `eval_tasks/<task_name>.json` |
| `--robot_cfg=F` | Optional assertion; must match the task's `robot.robot_cfg` |
| `--device_type=T` | Teleop device (default `pico`) |
| `--arm_only` | Publish only arm targets; leave hand and whole-body joints to external controllers |

### Interpreter selection

Same order as `geniesim benchmark`, so the verb works in every
environment:

1. `$GENIESIM_PY_CMD` — explicit override, set by `geniesim docker`
2. `omni_python` if on `$PATH` (canonical Isaac Sim wrapper)
3. `sys.executable` (or `python3` as last resort)

---

## 2. Package layout

```
src/geniesim_teleop/
  teleop.py           # entry: TeleOp loop + main()  (geniesim-teleop console script)
  bridge.py           # command merger, teleop chassis adapter and feedback bridge
  start_chassis.py    # original ROS servo with teleop-local overrides
  wuji/               # isolated launcher, arm-only Pico, hand/arm merger, standalone hand bridge, SDK adapter, model generation
  utils/              # logger, ros_utils, ros_nodes, vr_server, name/transform utils
  devices/            # teleop_device (base), pico_device (Pico VR driver)
  config/             # robot_interface.py — robot descriptors (G2)
  data_recording/     # rosbag → HDF5 pipeline (extract_ros_bag, process_data, sim_data_converter)
  app/                # prebuilt motion-control runtime (binaries, robot cfg, vendored msgs) — shipped as package-data
```

All intra-package imports use the **absolute** `geniesim_teleop.…`
form (e.g. `from geniesim_teleop.utils.logger import Logger`). Do **not**
reintroduce bare top-level imports (`from utils.logger import …`) or
`sys.path` hacks — they break once the package is installed.

---

## 3. Path map

| Artifact | Location |
|---|---|
| Teleop entry | [`src/geniesim_teleop/teleop.py`](src/geniesim_teleop/teleop.py) |
| Bridge entry | [`src/geniesim_teleop/bridge.py`](src/geniesim_teleop/bridge.py) |
| Separate Wuji entry | [`scripts/autoteleop_wuji.sh`](scripts/autoteleop_wuji.sh) → [`wuji/launch.py`](src/geniesim_teleop/wuji/launch.py) |
| Standalone Wuji hand bridge | [`wuji/hand_bridge.py`](src/geniesim_teleop/wuji/hand_bridge.py) (`python3 -m geniesim_teleop.wuji.hand_bridge`) |
| Wuji task defaults | [`config/wuji_teleop.yaml`](src/geniesim_teleop/config/wuji_teleop.yaml) |
| Device drivers | [`src/geniesim_teleop/devices/`](src/geniesim_teleop/devices/) |
| Robot descriptors | [`src/geniesim_teleop/config/robot_interface.py`](src/geniesim_teleop/config/robot_interface.py) |
| Recording pipeline | [`src/geniesim_teleop/data_recording/`](src/geniesim_teleop/data_recording/) — generic rosbag extraction plus deformable `state.json`/NPZ conversion |
| Motion-control runtime | [`src/geniesim_teleop/app/`](src/geniesim_teleop/app/) |
| CLI dispatcher | [`source/geniesim_cli/src/geniesim_cli/commands/teleop.py`](../geniesim_cli/src/geniesim_cli/commands/teleop.py) |

### Cross-package dependency

`config/task_config.py` reads the task YAML, `eval_tasks/<task_name>.json`,
robot JSON/URDF, and `robot_init_states.py` as data from `geniesim_benchmark`.
It resolves the source tree via `SIM_REPO_ROOT`, the sibling layout, or a
package spec without importing the Isaac Sim runtime. Robot and initial-state
mismatches fail before ROS/device startup.

`config/gripper.py` owns the robot-specific mapping: omnipicker commands outer
actuators idx41/idx81 with `0.785 * cmd` (0 closed, 0.785 rad open). Inner
idx31/idx71 are USD mimic followers and cannot be commanded independently.
G2_crsB_90d sends outer joint velocity
commands `-5 * (1-cmd)` for `cmd < 0.8`, otherwise `+2`. Wuji requires arm-only
mode and an external hand controller. DummyEnv zeros crsB 90d actuator
stiffness for velocity control, preserving damping, force and joint limits.

`start_mc.py` generates a task-specific MC runtime beneath `app/bin/.cache/mc`
from the simulator robot URDF, including arm/waist limits. `--cache-dir` overrides
the writable output directory; `--prepare-only` does not start a process.
The shipped binary requires the G2 platform alias and resolves models relative
to its executable; copy the executable/launcher into the task runtime and put
the selected URDF in its `share/robot_model/urdf/G2/G2.urdf`. The task URDF
determines the geometry; G2 is the binary's platform selector. Shared libraries and
message resources are symlinked. Enable the generated `/hal/joint_cmd` publisher.
Explicit MC robot/config flags take precedence over cached selections.
MC is always launched with `--no-tool`; `/wbc/retarget` carries no gripper
groups. Grippers publish on `/teleop/gripper_command`. The required bridge merges
this with MC `/hal/joint_cmd` and chassis `/cmd_4ws` into `/joint_command`
at 100 Hz (`utils/joint_command_mux.py`). Preserve NaN for uncontrolled
fields; zero is a drive target. Stale source velocities/efforts are zeroed
after 0.3 s while positions hold. Playback releases upper-body ownership.
The benchmark `G2_crsB_90D/G2_crsB_90D.urdf` arm origins match the crsB USD
and canonical crsB xacro.
The two task URDFs' five torso joints match their simulator USD frames, axes
and limits. Waist pose control uses GROUP_WAIST (10), control_type 0,
frame `base_link`, target `arm_base_link`; the generated `model/waist.yaml`
must use that end frame. Joint resets use the same group with control_type 3
and task `body_state` in idx01..idx05 order, never reversed.
Generated profiles set IK `damp_v_cost` and `damp_a_cost` to zero: the bundled
solver otherwise stalls on the CRS open_door waist pose. Actuator limits and
the PVT/LPF trajectory layer remain enabled.

Pico X resets arms, waist and head together and takes priority over tracking.
Ordinary and Wuji collection use `TeleopTask.mc_initial_states()` for simulator
startup, MC bootstrap and X. It derives the pose from the selected task's
`robot_init_states.py` entry using `config/arm_reset.py`: preserve both wrist
SE(3) poses and minimize joint change subject to URDF joint bounds, MC position
thresholds and the RelaxedIKSQP wrist bound `pi/2 - 0.3`, with 0.02 rad clearance.
Do not clip wrist joints independently or replace the reset with measured startup
feedback. An infeasible solve stops startup instead of publishing an unsafe pose.
`GENIESIM_TELEOP_CONFIG` identifies the shared task in both launchers. DummyEnv
validates the robot/sub-task and writes all initial joint positions, zero
velocities and matching drive targets in one render-loop operation. PiEnv and
non-teleop benchmark states stay unchanged. An external simulator must receive
the same task environment and teleop package.
Wuji's session YAML stores `teleop.mc_model_urdf` relative to the YAML, pointing
to the same generated model used by MC. Direct launches without this entry
prepare the canonical Wuji model in a temporary directory for the reset solve.
MC startup waits for complete arm feedback, sends the same prepared joint targets
as X, and waits up to 15 seconds for those targets on the merged `/joint_command`
and matching arm feedback (0.02 rad tolerance).
A is a hold-to-track waist clutch on the right controller (half-scale relative
translation/rotation); releasing it keeps the last waist target and reanchors
right-arm tracking. Keep waist and arm anchors independent. B only starts
recording; it must not change pose coordinates or suppress the right arm.
Chassis mapping lives in `config/chassis.py`: left Y -> vx, negative left X ->
vy, negative right X -> wz. `teleop.chassis` in the task YAML owns analog scales,
deadzone and UDP timeout (defaults 0.4 m/s, 0.5 rad/s, 0.05, 0.25 s).
`SimNode.pub_chassis_twist` publishes `/cmd_twist`, including neutral/reset/playback
and input loss. Never publish wheel commands directly from Pico. The standalone
ROS `genie_sim_controllers` servo converts Twist + feedback to wheel commands;
`autoteleop.sh --isaac-version 5.1` starts Isaac Sim through `geniesim benchmark run` with the same
task YAML before opening control terminals; Pico waits for TF initialization.
`--external-simulator` skips that child. The script tracks its simulator process
group and sends SIGINT on keep/discard/exit so recordings can finalize. Record
the actual session leader inside `setsid`; its caller PID can differ when it forks.
Use `teleop_process.sh stop-wait` for the simulator: wait for its process group
to exit without a forced-kill deadline. The simulator waits for every rosbag
recorder to flush and exit before closing. Keep `stop` bounded for controls and
post-processing. Snapshot existing episode directories before simulator startup;
mark only new episodes after `metadata.yaml` exists.
The selected container (`geniesim docker5.1 up` or `geniesim docker6.2 up`) and ROS build are prerequisites. It sources the canonical dev/installed
overlay, checks the package,
and launches `geniesim_teleop.start_chassis` except arm-only. The bridge owns the only
teleop `/joint_command` publisher. `DummyEnv` releases the base for tasks
with `robot.free_base`, enabling wheel contact and velocity drives and applying
the initial joints/drive targets atomically before release. Simulator YAML must
set `benchmark.model_arc: pi` to select DummyEnv/DemoPolicy.
`VRServer` uses monotonic packet age; `PicoDevice` rejects invalid/nonfinite
packets. Input loss clears clutches and zeros velocity grippers; reconnect must
start with fresh tracking anchors. Arm clutch references must use one common TF
timestamp for both wrists and arm_base_link; independent latest lookups can turn
torso motion into a nonzero target on a stationary grip. A missing snapshot must
leave all references intact and retry on the next input without engaging the clutch.
Ordinary grip release must retain `TeleOp.last_eef_pub`: MC keeps that target.
On re-grip (including the other hand engaging), express the held arm-base-relative
target in the refreshed base frame instead of replacing it with lagging measured
wrist feedback. Cache only published targets; reset/playback/input loss clear them.
The clutch log reports the measured-to-held position gap in millimetres.
`start_chassis.py` runs the original executable with its `4wd.yaml` plus local
`config/chassis_servo.yaml`; only this process receives the geometry/timeout
overrides and `use_sim_time: false`. `utils/chassis_command.py` selects steering
positions and drive velocities from original zero-filled `/cmd_4ws` frames.
The bridge forwards complete wheel feedback on `/teleop/chassis_state` and stops
wheel velocities on neutral/playback or stale Twist/feedback/servo output while
holding steering. These monotonic watchdogs belong to teleop, not `ServoCore`.
L3/R3 are unused.
`--arm_only` ignores A and restricts X to arms.

### Recording format

`data_recording/sim_data_converter.py` is the only final-output converter.
It converts G2 omnipicker, crsB omnipicker, crsB 90D and Wuji teleop inputs
to the `data_collection` output schema. Conversion and FK execute within this
package.
`derive_fields.py`, `robot_fk.py`, `gripper_remap.py`, `hardware_templates.py`
and `sim_version_constants.py` are its local support modules. The joint-name
JSON and FK-only URDFs live in `data_recording/config/robot_cfg` and are package
data. Keep these resources in wheels. Post-processing needs NumPy/SciPy/h5py/
PyYAML/pytz and the bag/image tools.

The final H5 uses the collection gripper radian mappings, named joint groups,
28-value arm/effector poses in base_link, arm-base action/end poses, link3
reference quaternions, six-value chassis-velocity placeholders, and one-value
camera timestamp arrays. The benchmark alias `gp_90d_gripper` maps to `zhiyuan_gripper_ctek90d`.
Hardware/sensor calibration, metadata, review JSON,
and the identical `record/aligned_joints.h5` copy are generated by this local converter.
Arm/effector geometry uses the packaged URDF FK and gripper offsets with recorded
world/base and arm-base transforms. Camera poses use the OpenCV convention.

Teleop actions remain measured trajectories from `/joint_states`; they are
not recovered command messages. The current bag has no VR button stream or
chassis command velocity. Preserve actual `state/robot` poses and document the
canonical placeholders in `post_proc/teleop_source.json`. Manual collection has
no planner-phase labels, so those lists are empty.

`process_data` waits for `teleop_result`, extracts ordinary data in a temporary
directory, and retains source bags. Joint names and columns are validated;
required missing joints fail instead of being filled with zero. Empty ROS
velocity/effort arrays use zero placeholders. Final conversion is staged;
`post_proc/teleop_export.json` is published last and records input fingerprints
and output paths. Failed episodes retry
on subsequent scans. Legacy ordinary outputs are re-extracted when bags remain;
valid `state.json` plus `aligned_joints_all.h5` can also be upgraded without bags.
Wuji uses the same final writer and completion contract, with named 20-DOF effectors.

### Wuji recording format

`data_recording/wuji_layout.py` shares canonical finger names with
`wuji/contracts.py`. Extraction retains all 40 finger positions and uses wrist
TF frames for end poses. Conversion indexes arms/head/waist and fingers by name;
Wuji left/right effector position datasets each contain 20 angles in radians,
ordered thumb, index, middle, ring, pinky and joints 1–4 within each finger.
The group `name` attribute matches the ordinary collection schema. All other
H5 paths, types, end-pose layout and metadata structure use the common writer.
Hardware metadata identifies Wuji hands. The packaged `G2_crsB_wuji.urdf`
comes from the canonical ROS xacro and supplies wrist and index-link4 FK,
matching the simulator's configured end-effector frames.
Incomplete/duplicate joint lists fail explicitly. Raw Wuji bags remain available
for reprocessing. The converter keeps the ordinary measured-trajectory action
semantics; it does not substitute glove commands for measured joint state.

The benchmark recorder reads the authored camera distortion model. Wuji wrist
calibration retains `opencvFisheye` and k1–k4. Depth topic selection uses exact
USD basenames; extraction normalizes camera filenames without case sensitivity.
Final calibration tolerates legacy `"None"` values as JSON null, marks the
episode invalid in both metadata files and reports missing calibration in
`post_proc/calibration_status.json`. Legacy intermediate
JSON/H5 files can finish conversion without re-extracting bags. Missing depth
that was never recorded cannot be reconstructed.

### Isolated Wuji contracts

`wuji/pico.py` reuses arm-only tracking and the inherited B recording hotkey.
The supervised simulator forces
`benchmark.interactive=true` and `benchmark.record=false`: manual recording
without automatic evaluation/reset. Y/N followed by Enter stops owned processes,
waits for bag finalization and marks only newly created episodes. Ctrl-C leaves
data unmarked; external simulators remain under their owner's lifecycle. `wuji/bridge.py` is the only `/joint_command`
publisher for the Wuji session: MC supplies arms and the hand source supplies
complete 20-joint hands. It removes ordinary gripper/chassis command subscriptions,
rejects MC finger ownership, validates finite values and canonical/retargeter
names, clamps URDF limits, and clears hand caches across playback transitions.
Unknown, duplicate, partial or stale stamped hand frames must not enter the mux.
Each hand has its own cache so one disconnected glove cannot erase the other.

`wuji/pico.py` inherits the shared task-pose startup and reset from `TeleOp`.
Only the arms are sent to MC in arm-only mode; glove commands retain hand
ownership. Hand-only output is not arm readiness, and measured startup feedback
must not replace the task reset target.

`wuji/hand_bridge.py` owns the hand-name mapping, glove capture, SDK retargeting,
and standalone ROS hand bridge. Its independent entry point is
`python3 -m geniesim_teleop.wuji.hand_bridge`; it provides the
`--source retargeting|sdk` options and direct `/joint_command` output for the
split-process chef workflow. ROS 2 and `wuji_sdk` imports remain lazy.

`wuji/sdk.py` imports the adapter from `geniesim_teleop.wuji.hand_bridge` within
this package. It publishes only newly received frames at up to 100 Hz,
waits for both configured hands
initially, and holds the last target on input loss. External retargeting uses
`/wuji_hand/joint_command`;
SDK input uses `/teleop/wuji/hand_command`. Neither source publishes directly to
`/joint_command`.

`wuji/model.py` generates the canonical `g2_crsB_wuji.urdf.xacro` using a temporary
ament index. The legacy benchmark Wuji URDF lacks the 40 fingers; do not use it
for MC or hand limits. Generated files and MC caches belong to the session under
`output/wuji_teleop/`, never to a shared model/cache. The source checkout and xacro
are required. `wuji/launch.py` and `wuji/pico.py` default to UDP 8080 to match
the fixed AIDEA App port; the launcher uses domain 91. `--port` can override the
port when the App supports it. The launcher checks for occupied command
topics and the selected UDP port, and signals only its own child process groups
on exit. ROS domains do not isolate UDP sockets; never share the same IP/port
with ordinary teleop. Docker bridge networking requires UDP port forwarding.
Concurrent collection requires a separate container because the MC singleton
check operates by process name. It rejects another active MC without killing it.
Ordinary collection uses `scripts/teleop_process.sh` to track and stop only its
own simulator and control process groups. Both launchers wait for simulator
recording finalization without escalating to SIGKILL. A forced control shutdown
fails the session; it must not mark the recording complete.

---

## 4. Environment variables

| Var | Effect |
|---|---|
| `GENIESIM_TELEOP_ISAAC_VERSION` | Host script mode: `6.2` (default, controls only) or `5.1` (with simulation). `--isaac-version` takes precedence. |
| `GENIESIM_CONTAINER` | Override ordinary collection/post-processing container; defaults to `geniesim4` for 6.2, `geniesim3` for 5.1. |
| `GENIESIM_TELEOP_CONFIG` | Task YAML shared by the launcher, teleop and MC; for Docker use its container-side path. |
| `GENIESIM_PY_CMD` | Override the interpreter that launches the teleop modules. Set automatically by `geniesim docker`. |
| `GENIESIM_TELEOP_SIM_PY` | Ordinary 5.1 simulator interpreter, default `omni_python`; 6.2 never starts the simulator. |
| `GENIESIM_TELEOP_INTERFACE` | Default for `--interface`; also forwarded by `autoteleop.sh` during network preflight. |
| `GENIESIM_TELEOP_HOST_IP` | Default for `--host_ip`; also supported by `autoteleop.sh`. Set only one network selector. |
| `GENIESIM_WUJI_CONTAINER` | Wuji launcher container, default `geniesim4` for 6.2 / `geniesim3` for 5.1; independent of collection's selection. |
| `GENIESIM_WUJI_WORKSPACE` | Wuji container checkout path, default `/workspace`. |
| `GENIESIM_WUJI_SIM_PY` | Wuji 5.1 simulator interpreter, default `omni_python`; 6.2 skips simulation. ROS/SDK children use system Python. |

Network selection uses `ip -j -4 addr show up` (requires iproute2). Loopback
and operationally DOWN interfaces are excluded. VPNs and active virtual
interfaces remain candidates. Exactly one interface with one IPv4 address
is selected automatically; otherwise startup stops and lists the choices.
For multiple addresses on one interface, use `--host-ip` (alias `--host_ip`);
Wuji accepts both spellings as well. The VR UDP socket
binds to the selected address. `autoteleop.sh` checks selection before starting
the bridge, motion controller, or teleop loop.

---

## 5. Common workflows

### Convert a deformable-state rosbag

The generic converter consumes configurable joint, odometry, TF, and four-topic
deformable streams. It aligns robot messages and deformable snapshots on a
uniform rosbag receive-time grid, converts the odometry child pose to the
replay robot root through TF, splits simulation-time resets into separate
episode directories, and writes recording-world authored vertices, exact USDA
Prim paths, and aligned robot pose/joints:

```bash
PYTHONPATH=source/geniesim_teleop/src \
python3 -m geniesim_teleop.data_recording.extract_deformable_state \
  ./runs/<bag-dir> ./runs/<bag-dir>_decoded \
  --topic-prefix /genie_sim_engine/deformable \
  --scene-name <scene-name> \
  --scene-usd <asset-relative-scene.usd> \
  --robot-name <replay-robot-name>
```

The Wuji chef wrapper preserves the existing defaults and command:

```bash
PYTHONPATH=source/geniesim_teleop/src \
python3 -m geniesim_teleop.data_recording.extract_chef_state \
  ./runs/<bag-dir> ./runs/<bag-dir>_decoded
```

The shared message decoder accepts `tet_barycentric_v1` FEM bindings and
`particle_direct_v1` cloth bindings. New scenes may choose another topic
prefix but must publish the same `info`, `mapping`, `state`, and `rigid_state`
stream contract.

### Launch the teleop loop (inside the GUI container)

```bash
geniesim teleop run --device_type=pico --port=8080
```

### Launch the in-process bridge

```bash
geniesim teleop bridge --mode inprocess
```

### Run without the CLI on PATH

```bash
python3 -m geniesim_cli teleop run --device_type=pico
# or directly:
python3 -m geniesim_teleop.teleop --device_type=pico
```

---

## 6. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `geniesim: command not found` | Use `python3 -m geniesim_cli teleop …` or reinstall `geniesim_cli`. |
| `❌ geniesim_teleop is not importable.` | Editable install missing. Run `pip install -e source/geniesim_teleop`. |
| `ModuleNotFoundError: rclpy` / `geometry_msgs` | ROS 2 isn't sourced. Run inside the Genie Sim container or `source` your ROS overlay. |
| Task/initial-state resolution fails | Check the selected YAML, task JSON robot and matching `TASK_INFO_DICT` entry; set `SIM_REPO_ROOT` if the benchmark source cannot be located. |

---

## 7. Skills

| Skill | Trigger |
|---|---|
| [run-teleop](skills/run-teleop/SKILL.md) | "start teleop", "启动遥操作", "VR 采集", "run the teleop loop" |

---

## 8. Do not

- Don't reintroduce bare `from utils/devices/config …` imports or
  `sys.path.insert` hacks — use absolute `geniesim_teleop.*` imports.
- Don't pin a host/IP inside the code — pass `--host_ip` / `--client_host`
  at invocation time.
- Don't add a hard import of `geniesim_benchmark` at module top level;
  keep the `find_spec` lazy lookup so teleop runs without the benchmark
  package installed.
