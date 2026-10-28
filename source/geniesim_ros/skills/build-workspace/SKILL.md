---
name: build-workspace
description: >
  Build the `geniesim_ros` colcon workspace inside the Genie Sim Docker
  container using the `geniesim ros build` CLI verb. Produces the
  `./devel` overlay that every `ros2 launch genie_sim_bringup …` step
  depends on.
  Trigger: When the user asks to "build ros workspace", "colcon build", "build genie_sim_bringup", "set up the ROS overlay",
  or after a fresh `geniesim docker into` shell where the workspace
  isn't sourced yet.
license: MPL-2.0
metadata:
  author: genie-sim
  version: "1.0"
prerequisites:
  - geniesim_cli:fresh-machine-setup   # see source/geniesim_cli/AGENTS.md § 0 (CLI + docker container available)
inputs:
  - name: build_type
    desc: "`dev` (RelWithDebInfo + symlink-install) or `release`"
    required: false
    default: dev
outputs:
  - desc: "`./devel`, `./devel_build`, `./devel_log` overlays sourced; `ros2 pkg list | grep genie_sim` lists 10 packages"
---

## When to Use

- First time entering the container in a fresh checkout.
- After pulling new commits that touched any package under
  `source/geniesim_ros/src/ros_ws/src/`.
- After running `geniesim ros build cleanup` and you want to rebuild
  from scratch.
- Before invoking the `launch-scene` skill (its `ros2 launch …` step
  needs the workspace sourced).

Do **not** use for:
- Installing the wheel layout that ships with `pip install geniesim_ros`
  — that's already built; just `source` the install tree.
- Building a Release wheel for deploy → use `geniesim deploy geniesim_ros`.

## Critical Patterns

1. **Run inside the container.** The colcon toolchain and rosdep tree
   live in the Isaac Sim + Jazzy image; outside the container the
   build will pick up the wrong Python / Boost / Eigen.
2. **Dev vs Release.** `geniesim ros build dev` produces a
   `RelWithDebInfo` + `--symlink-install` overlay at `./devel`,
   `./devel_build`, `./devel_log` (intentionally namespaced so they
   can't collide with a release build under `./install`).
   `geniesim ros build release` is for deploy, not for iteration.
3. **Always `source devel/setup.bash` after building.** The shell
   environment doesn't carry the overlay automatically.
4. **rosdep first if anything fails.** Missing apt-side dependencies
   are the #1 source of "package not found" build errors — run
   `geniesim ros doctor` before re-trying the build.

## Workflow

### Step 1 — Confirm you're inside the container

```bash
geniesim status                       # should report all distributions OK
echo $ROS_DISTRO                      # should print "jazzy"
```

If `$ROS_DISTRO` is empty, you're on the host — `geniesim docker into`
first.

### Step 2 — (optional) Repair rosdep

```bash
geniesim ros doctor                   # check & fix rosdep deps
```

### Step 3 — Build

```bash
cd /workspace                         # repo root (mounted by `geniesim docker up`)
geniesim ros build dev                # RelWithDebInfo + symlink-install -> ./devel
```

Iteration loop after editing C++ / xacro / launch files:

```bash
geniesim ros build dev                # re-runs incrementally thanks to symlink-install
```

### Step 4 — Source the overlay

```bash
source devel/setup.bash
```

Verify:

```bash
ros2 pkg list | grep genie_sim         # should list the 10 genie_sim_* packages
```

### Step 5 — Cleanup (only when something is wedged)

```bash
geniesim ros build cleanup             # interactive prompt before removing devel*/build/install/log
```

## Commands (copy-paste summary for the user)

```bash
# Inside the container, from the repo root:
geniesim ros doctor                    # optional — fix rosdep first
geniesim ros build dev                 # build the overlay
source devel/setup.bash                # overlay the built workspace
ros2 pkg list | grep genie_sim         # sanity check
```

## Notes

- `geniesim ros build` is a thin wrapper around `colcon build` — the
  same flags work if you call colcon directly, but the CLI sets the
  right `CMAKE_BUILD_TYPE`, picks up `$GENIESIM_WORKSPACE`, and routes
  output to the namespaced `./devel*` dirs.
- `geniesim ros graph` writes `colcon graph` to `geniesim_graph.png`
  in the cwd — handy when you need to reason about which packages
  pull in which.
- The bundled workspace is also installed system-wide via
  `pip install geniesim_ros`; that path is fine for *running* but not
  for iterating — every edit needs a rebuild, and `pip` won't pick up
  a symlink-install layout.

## Resources

- **CLI dispatcher**: [source/geniesim_cli/src/geniesim_cli/commands/ros.py](../../../geniesim_cli/src/geniesim_cli/commands/ros.py)
- **Workspace root**: [source/geniesim_ros/src/ros_ws/](../../src/ros_ws/)
- **Package routing**: [source/geniesim_ros/AGENTS.md](../../AGENTS.md)
