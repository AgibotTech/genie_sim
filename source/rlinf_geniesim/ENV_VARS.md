# Environment variables

Run configuration belongs in task initialization. These variables select
runtime paths, process placement, or opt-in logging.

| Variable | Purpose / default |
| --- | --- |
| `GENIESIM_REPO_ROOT` | Explicit checkout for `geniesim post-train`; takes precedence over `GENIESIM_REPO_PATH` / `SIM_REPO_ROOT` |
| `SIM_REPO_ROOT` | Checkout used by launchers; defaults to their own checkout |
| `SIM_ASSETS_ROOT` | Asset package root for host/launch helpers |
| `GENIESIM_ASSETS_PATH` | Asset path seen by the runtime; container default `/geniesim_assets` |
| `GENIESIM_ASSETS_SRC` | Host asset package mounted by `geniesim docker ... up` at `/geniesim_assets` |
| `GENIESIM_IMAGE` / `GENIESIM_CONTAINER` | Image tag / container name override for the selected Docker profile |
| `RLINF_WS_BUILD` | ROS build/install directory; default `/tmp/ros_ws_build` |
| `RLINF_WS_PACKAGES` | Optional selected ROS package list for workspace builds |
| `RLINF_NO_AUTO_REBUILD` | `1` retains an existing workspace despite source-stamp changes; default 0. A missing workspace is still built |
| `RLINF_RUN_ROOT` | Parent of runtime directories and spawn domain locks; default `/tmp`; spawn callers on one host must share it |
| `RLINF_SERVER_PYTHON` | Server interpreter: manual launcher defaults to `python3`; `spawn_server` defaults to `/usr/bin/python3` and requires an absolute path |
| `RLINF_ROS_SETUP` | ROS setup script for `spawn_server`; default `/opt/ros/$ROS_DISTRO/setup.bash` (`jazzy` when unset) |
| `GENIESIM_OWNER_PID` / `GENIESIM_OWNER_START` | Set internally by `spawn_server` to bind the private server to its owner PID and process start time; unset for external servers |
| `RLINF_ISAAC_PYTHON` | Isaac Python launcher; default `/isaac-sim/python.sh` |
| `RLINF_TILED_RENDER_SCRIPT` | Override the tiled node script; default resolves `isaacsim_tiled_render.py` in the sibling `geniesim_ros` source tree |
| `RLINF_FUSED_SIM_SCRIPT` | Override the fused node script; default resolves `isaacsim_fused_sim.py` in the sibling `geniesim_ros` source tree |
| `CUDA_VISIBLE_DEVICES` | GPUs visible to a server; set by `--device` |
| `ROS_DOMAIN_ID` | ROS isolation; launcher derives it from the port or `--domain` |
| `RLINF_SERVER_TAG` | Per-server resource suffix; launcher supplies `p<port>` |
| `GENIESIM_EXIT_WITH_PARENT` / `GENIESIM_PARENT_SERVER_PID` | Set by the server for its backend nodes: `1` and the server PID make a node exit when the server is killed. Nodes started without them are unaffected |
| `READY_TIMEOUT` | Server startup timeout in seconds; launcher default 600 |
| `RLINF_TILED_READY_TIMEOUT` | Tiled node startup/SHM-attach timeout in seconds; backend default 180, set to `READY_TIMEOUT` by the launcher |
| `RLINF_FUSED_READY_TIMEOUT` | Fused node startup timeout in seconds; backend default 180, launcher defaults to `READY_TIMEOUT` unless explicitly set |
| `RLINF_TILED_BARRIER_CAP_S` | Optional tiled render barrier timeout cap |
| `RLINF_TILED_FIRST_FRAME_CAP_S` | First-frame barrier timeout in seconds, including cold shader compilation; default 300 |
| `RLINF_TILED_BARRIER_RECOVERY` | Allow bounded previous-frame substitution after a barrier timeout; default 1. Reset/reload requires a fresh frame |
| `RLINF_TILED_BARRIER_MAX_STALLS` | Maximum consecutive previous-frame substitutions before raising; default 5 |
| `RLINF_TILED_BARRIER_SPIN_AFTER_S` | Seconds before barrier polling changes from 2 ms to 0.2 ms intervals; default 0.04 |
| `RLINF_TILED_FABRIC` | Enable Fabric rendering; launcher default 1 |
| `HEADLESS` | Headless rendering selection; launcher default 1 |
| `RLINF_TILED_HEADLESS` | Tiled node headless mode; default 1, set from `HEADLESS` by the launcher |
| `RLINF_ENV_SHELL` | Emit a foreground-free stage for the render node; default 1 |
| `RLINF_BAKED_GRAPHS` | Keep wrapper-stage OmniGraphs active; default 1. Does not affect per-environment foreground copies |
| `RLINF_STEP_CHUNK` | `0` disables batched chunk RPCs and uses individual steps; default 1 |
| `RLINF_JPEG_THREADS` | JPEG codec worker count; unset, nonpositive, or invalid values select half the CPU count, bounded to 2–16 |
| `RLINF_REST_BAKE_CACHE` | Rest-state cache; default `~/.cache/rlinf_geniesim/rest_bake` |
| `GENIESIM_PERF_LOG` | `1` writes handler timing CSVs; default off |
| `GENIESIM_RENDER_PERF_LOG` | Render performance tracing; launch environment defaults off |
| `TELEOP_PYTHON` | Local teleop interpreter |
| `TELEOP_CONTAINER` | Run `host_run_teleop.sh` inside this existing container as the host UID/GID; unset runs locally |
| `TELEOP_CONTAINER_REPO` | Checkout inside the teleop container; default `/workspace` |

`RLINF_WS_PKGS_DEFAULT` is an internal shell constant in `_ws_build_stamp.sh`,
not an environment override. Set `RLINF_WS_PACKAGES` to select a different
package list. Use `start_sim_servers.sh --rebuild-ws` to force a workspace
rebuild through the batch launcher.

The engine's detailed physical parameters are documented in
[physics_knobs.md](../geniesim_ros/src/ros_ws/src/genie_sim_engine_core/docs/physics_knobs.md).
`config/physics_overrides.yaml` contains optional contact overrides; environment
values take precedence. Do not use physical overrides to select processes,
ports, or tasks.

| Variable | Purpose / default |
| --- | --- |
| `RLINF_OBJECT_SOLREF` | MuJoCo object contact `timeconst,dampratio`; overrides `object_solref` in `config/physics_overrides.yaml`; fallback `0.010,1.0` |
| `RLINF_ROBOT_SOLREF` | MuJoCo robot contact `timeconst,dampratio`; overrides `robot_solref` in the same file; fallback `0.002,2.5` |
| `RLINF_DISABLE_ANCHOR_MIRROR` | `1` disables anchor mirroring in the separate-process renderer for diagnostics; default 0 |

The engine ignores `RLINF_IMPRATIO`, `RLINF_GRIPPER_FORCE_CAP`,
`RLINF_GRIPPER_KP`, `RLINF_PAD_DECOMP`, `RLINF_PAD_DECOMP_TH`,
`RLINF_PAD_DECOMP_MAXH`, `RLINF_SCENE_DECOMP_TH`, and `RLINF_BACKGROUND_STATIC`.
Set the corresponding solver, gripper, and scene-collider options in the
selected physics YAML profile instead.
