---
name: run-teleop
description: >
  Start ordinary Pico or Wuji glove teleoperation with Isaac Sim 6.2, select
  the collection task/scene and Pico network address, keep recordings, and run
  teleop data post-processing. Use for 启动遥操作, 更换采集场景, Pico采集, Wuji采集, 遥操数据后处理,
  or requests to operate these collection launchers. Benchmark evaluation
  and the separate RT Engine teleop bridge have their own workflows.
license: MPL-2.0
metadata:
  author: genie-sim
  version: "1.3"
---

# Isaac Sim 6.2 teleoperation and recording

Run the host commands below from the repository root in an interactive terminal.
Choose **ordinary Pico** or **Wuji**; do not start both in the same container.
Use existing user choices for task, network and gloves. Ask only for missing
values needed to launch; never guess an IP address or glove serial number.

Both host collection scripts default to **Isaac Sim 6.2** and start simulation
and controls together. Only 6.2 is supported. Use `--external-simulator` only
when attaching to an existing simulator. `geniesim teleop run` starts only the
Pico loop, so it is not the one-command collection entry point.

## 1. Prepare the 6.2 container

The CLI must already be available; see
[CLI setup](../../../geniesim_cli/AGENTS.md#0-fresh-machine-setup).
The 6.2 image is built from `docker/Dockerfile`; its default container is
`geniesim4`. The repository and required scene assets must be available inside
that container, normally with the checkout mounted at `/workspace`.

To skip `geniesim docker5.1 build`, pull and retag the prebuilt `geniesim3`
Docker Hub image following [issue #201](https://github.com/AgibotTech/genie_sim/issues/201),
then continue with `geniesim docker5.1 up` below.

```bash
# Host: build when preparing or updating the local image.
geniesim docker6.2 build
# Host: start the prepared container.
geniesim docker6.2 up
```

An existing container must be recreated to use a rebuilt image. Pulling source
changes alone does not install dependencies into an old container. The current
Dockerfile includes teleop dependencies, `wuji_sdk`, xacro and network tools;
SDK/ROS controls run in system `python3` 3.12; the simulator uses `python3.12`.
The collection scripts require an already running container; they do not build
or create it. Ordinary collection also requires a supported desktop terminal
emulator and the chassis ROS workspace build:

```bash
# Host:
geniesim docker6.2 into
# Inside the container, once for ordinary chassis control:
cd /workspace
geniesim ros build dev
exit
```

Wuji controls arms and fingers only and does not start the chassis servo.

## 2A. Select the Pico network

Inspect local addresses with `ip -4 addr`. Select the **local machine's address
reachable by the Pico App**, not the headset's address. Both collection modes
listen on UDP **8080** by default, matching the fixed AIDEA App port.

- Automatic selection requires exactly one eligible interface with one IPv4
  address. Loopback and down interfaces are excluded; VPN and virtual interfaces
  can still make selection ambiguous.
- If the Pico interface has one IPv4 address, select that interface.
- If an interface has multiple IPv4 addresses, select the specific local IP.
  `--interface` alone will fail with `Selected interface has multiple IPv4 addresses`.
- Use one selector at a time. No interface name or IP is hardcoded.

| Launcher | Interface selector | Explicit local IPv4 selector |
|---|---|---|
| Ordinary `autoteleop.sh` | `GENIESIM_TELEOP_INTERFACE` environment variable | `GENIESIM_TELEOP_HOST_IP` environment variable |
| Wuji `autoteleop_wuji.sh` | `--interface` argument | `--host-ip` argument (alias `--host_ip`) |

The ordinary shell script does not accept these network CLI arguments; use its
shown environment variables. Wuji uses its explicit arguments. Enter the chosen
host IP and port in the Pico App where configurable. If Docker uses bridge
networking, the chosen IP must exist inside the container and the UDP port must
be forwarded; host networking avoids a separate container address. Different
ROS domains do not let two processes bind the same IP/UDP port.

Replace `YOUR_PICO_INTERFACE`, `YOUR_LOCAL_PICO_IP` and glove serial placeholders
in the examples with values discovered from the user's environment.

## 2B. Select the collection task and scene

Stop the old session before switching tasks. Copy the appropriate teleop YAML;
keep its ROS/camera settings and `benchmark.model_arc: pi`,
`policy_class: DemoPolicy`, `env_class: DummyEnv`, `interactive: true`, and
`record: false`. Both launchers enforce these manual-collection settings;
an external simulator must also use them. Change the existing
`benchmark.task_name` and `benchmark.sub_task_name` fields as a matching pair:

| Mode | `task_name` | `sub_task_name` | Selected robot |
|---|---|---|---|
| Ordinary Omnipicker | `home_g2_op` | `open_door` | `G2_omnipicker.json` |
| Ordinary 90D | `phone_packaging_g2_crsb90d` | `phone_packaging` | `G2_crsB_90d.json` |
| Wuji | `table_task_g2_crsB_wuji_take_the_scissors` | `take_the_scissors` | `G2_crsB_wuji.json` |

Within `source/geniesim_benchmark/src/geniesim_benchmark/`, `task_name` resolves
`benchmark/config/eval_tasks/<task_name>.json` for the background, robot and base pose.
`sub_task_name` selects instructions
and foreground instances under `benchmark/config/llm_task/<sub_task_name>/`,
initial joint states, and the recording directory. To select one existing
scene instance, add `benchmark.instance_ids: "0"`. Use one ID per manual session;
change the ID and restart to collect another layout. Wuji manual mode does not
automatically advance to another instance. Confirm the task JSON, instance assets and
robot's `robot_init_states.py` entry exist before launching a new combination.

For ordinary Pico, copy the default on the host:

```bash
cp source/geniesim_benchmark/src/geniesim_benchmark/config/teleop.yaml \
  source/geniesim_benchmark/src/geniesim_benchmark/config/teleop_custom.yaml
```

For the 90D example, edit the copy to set `task_name: phone_packaging_g2_crsb90d`
and `sub_task_name: phone_packaging`. Export the selection in the host terminal,
then use the launcher and network selector in section 3A:

```bash
export GENIESIM_TELEOP_CONFIG=/workspace/source/geniesim_benchmark/src/geniesim_benchmark/config/teleop_custom.yaml
```

For Wuji, copy its default before editing the task fields:

```bash
cp source/geniesim_teleop/src/geniesim_teleop/config/wuji_teleop.yaml \
  source/geniesim_teleop/src/geniesim_teleop/config/wuji_custom.yaml
```

Add `--config /workspace/source/geniesim_teleop/src/geniesim_teleop/config/wuji_custom.yaml`
to the Wuji commands in section 3B. Both launchers receive **container-side**
paths; replace `/workspace` if the checkout is mounted elsewhere. Ordinary
`autoteleop.sh` uses the environment variable and does not accept `--config`.

90D/Omnipicker selection follows the task JSON's `robot.robot_cfg`, including
gripper control, MC model and startup/reset pose. Supported ordinary families
are `G2_crsB_90d`, `G2_omnipicker` and `G2_crsB_omnipicker`; legacy `G2_90d`
is unsupported. `--robot_cfg` is a consistency check, not a robot override.
Wuji tasks require the Wuji launcher. External simulators must also load the
new YAML and matching ROS domain.

After switching, use the new **`sub_task_name`** for post-processing. For the
90D example:

```bash
./source/geniesim_teleop/scripts/autoteleop_post_process.sh --isaac-version 6.2 phone_packaging
```

## 3A. Ordinary Pico: start simulation and collection

For a single-address interface, run on the host:

```bash
# Clear a previously configured IP selector before selecting an interface.
unset GENIESIM_TELEOP_HOST_IP
GENIESIM_TELEOP_INTERFACE=YOUR_PICO_INTERFACE \
  ./source/geniesim_teleop/scripts/autoteleop.sh --isaac-version 6.2
```

Alternatively, for an interface with multiple addresses:

```bash
unset GENIESIM_TELEOP_INTERFACE
GENIESIM_TELEOP_HOST_IP=YOUR_LOCAL_PICO_IP \
  ./source/geniesim_teleop/scripts/autoteleop.sh --isaac-version 6.2
```

The script starts Isaac Sim, the command/feedback bridge, motion controller,
chassis servo and Pico. Do not start another Isaac Sim for this session. Wait
for simulator joint states and TF before tracking. The bridge owns the merged
`/joint_command`; Pico sends arm targets through MC, not directly to that topic.

The default YAML is `source/geniesim_benchmark/src/geniesim_benchmark/config/teleop.yaml`
(`home_g2_op` / `open_door`). For another task, set `GENIESIM_TELEOP_CONFIG` to
its **container-side** YAML path before launching. The task JSON selects the
robot and gripper; keep simulator, MC and Pico on the same task configuration.
`GENIESIM_CONTAINER` overrides the default `geniesim4` container and
`GENIESIM_WORKSPACE` overrides `/workspace`.

Hold the Pico grip buttons for arm tracking. Sticks drive the chassis, A holds
waist tracking, X resets arms/waist/head, B starts recording, and Y enters
playback. See the [package README](../../README.md) for the complete controls.

After recording, press **Y/y in the original host terminal** to keep, or **N/n**
to discard (this script reads one key without requiring Enter). It stops its
simulator, lets recording finalize, and marks `recording_info.json` for the
post-processing watcher. These terminal keys are separate from Pico buttons.

## 3B. Wuji: start simulation, Pico arms and glove fingers

Use the dedicated launcher, not ordinary `autoteleop.sh` with arm-only mode.
The default task is the scissors scene in
[`wuji_teleop.yaml`](../../src/geniesim_teleop/config/wuji_teleop.yaml).
It uses manual recording without automatic evaluation or timed scene resets.

For SDK gloves, use calibrated glove serial numbers and an explicit local IP:

```bash
# Optional preflight: checks dependencies/network without starting the devices.
./source/geniesim_teleop/scripts/autoteleop_wuji.sh --isaac-version 6.2 \
  --check --source sdk --host-ip YOUR_LOCAL_PICO_IP \
  --left-sn YOUR_LEFT_GLOVE_SN --right-sn YOUR_RIGHT_GLOVE_SN

# Start Isaac Sim, bridge, MC, Pico and SDK hand input together.
./source/geniesim_teleop/scripts/autoteleop_wuji.sh --isaac-version 6.2 \
  --source sdk --host-ip YOUR_LOCAL_PICO_IP \
  --left-sn YOUR_LEFT_GLOVE_SN --right-sn YOUR_RIGHT_GLOVE_SN
```

For a single-address interface, replace `--host-ip YOUR_LOCAL_PICO_IP` with
`--interface YOUR_PICO_INTERFACE`. One glove is supported by supplying only
that side's serial number. `--sdk-user` selects an existing calibrated SDK user
by ID or exact display name; the launcher does not calibrate gloves.

If hand retargeting is already supplied by another process:

```bash
./source/geniesim_teleop/scripts/autoteleop_wuji.sh --isaac-version 6.2 \
  --source retargeting --host-ip YOUR_LOCAL_PICO_IP
```

That mode requires a separately started publisher on `/wuji_hand/joint_command`
in the same ROS domain; it does not start the external glove producer. Wuji's
default domain is **91**. Each hand message must supply the complete named
20-joint hand; see the [hand input contract](../../AGENTS.md#isolated-wuji-contracts).

Hold Pico grips for the arms; gloves control fingers. X resets arms and B starts
recording. Waist and chassis control are disabled. When finished, enter **Y +
Enter** in the launcher terminal to keep, or **N + Enter** to discard and stop.
The supervisor waits for recording shutdown before marking the new episodes.
Ctrl-C leaves episodes unmarked, so it is not the keep-data operation.

Use `--config` with a container-side YAML to change tasks.
`GENIESIM_WUJI_CONTAINER` overrides `geniesim4` and `GENIESIM_WUJI_WORKSPACE`
overrides `/workspace`. Logs are printed under `output/wuji_teleop/<session>/`.
For both workflows, `--external-simulator` is only for deliberately attaching
to an existing simulator with matching task/robot/domain; omit it for one-command
6.2 simulation startup.

## 4. Post-process kept recordings

Run the same post-processing script on the host for either workflow. The
argument is the actual directory name beneath `output/recording_data`, normally
`benchmark.sub_task_name`, not the task JSON filename or Wuji session-log folder.

```bash
# Ordinary default task:
./source/geniesim_teleop/scripts/autoteleop_post_process.sh --isaac-version 6.2 open_door

# Wuji default task:
./source/geniesim_teleop/scripts/autoteleop_post_process.sh --isaac-version 6.2 take_the_scissors
```

An optional second positional argument sets the operator name (default
`GenieSim`). If Wuji uses a custom container/workspace, set `GENIESIM_CONTAINER`
and `GENIESIM_WORKSPACE` to those same values for post-processing; this script
does not read the `GENIESIM_WUJI_*` overrides.

This is a persistent watcher. It can run in a second terminal during collection,
but processing waits for the keep/discard result after recording finalizes.
Discarded episodes are deleted; kept source bags remain available. Failed
conversions retry. Stop the watcher with Ctrl-C when finished.

Outputs are inside each numeric episode directory:

- `aligned_joints.h5` and identical `record/aligned_joints.h5`.
- `meta_info.json`, `data_info.json`, `parameters/sensor`,
  `parameters/hardware`, `review`, and `observations/videos`.
- `post_proc/teleop_export.json`: final conversion completion marker.

Ordinary and Wuji recordings use one converter and the same final schema.
Ordinary left/right `effector/position` each contain one value; Wuji contains
20 measured angles per hand in radians, ordered thumb, index, middle, ring,
pinky, with joints 1–4 per finger. Each group's `name` attribute identifies them.
Actions describe measured trajectories. Wuji wrist calibration retains
`opencvFisheye` and k1–k4. Check `post_proc/calibration_status.json`: legacy missing
intrinsics are JSON null and mark metadata invalid even if export completes.
Unrecorded depth or missing calibration cannot be reconstructed from joint data.

## Troubleshooting and scope

- Container not running: start the selected 6.2 container before the host launcher.
- Missing chassis package: build the ROS workspace inside that container.
- Missing `wuji_sdk` or recording dependencies: rebuild/recreate the selected
  image; installing into Isaac's Python alone does not fix system Python controls.
- Multi-IP error: use the local IP selector, not just the interface name.
- Pico waits for TF: verify the simulator loaded the same task/robot; Wuji also
  needs the same domain. A running control process alone is not readiness.
- No final output: check the keep result, watcher errors and completion marker.
  `aligned_joints_all.h5` alone is an intermediate, not a finished export.

For implementation contracts use [AGENTS.md](../../AGENTS.md). The separate
[RT Engine bridge skill](../../../geniesim_ros/skills/teleop-bridge/SKILL.md)
is for that engine's workflow; do not substitute its MoveIt setup for this
Isaac Sim + MC collection flow.
