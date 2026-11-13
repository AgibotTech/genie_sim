# geniesim_ros

Documentation describes the current checkout in English. Keep local file and
checkout references within the main repository; do not depend on personal notes
or separate checkouts elsewhere on a maintainer's computer. Public web links,
upstream documentation, papers and download sources are welcome. Omit discarded
implementation history, and keep unimplemented work and future plans in
explicitly named roadmap documents.
Verification records must state the tested scope and current limitations.

ROS 2 workspace for the Genie Sim simulation stack — the physics +
render + MoveIt + ros2_control + visualization layer.

Source: [source/geniesim_ros/](.)
License: [Mozilla Public License Version 2.0](LICENSE)
Agent guide: [.agent/geniesim_ros.md](../../.agent/geniesim_ros.md)
Public intro: [README.md](README.md)
Shared documentation index: [docs/README.md](docs/README.md)

Maintain shared Python code for both Python 3.10 and Python 3.12. Keep
`requires-python = ">=3.10,<3.13"` and existing interpreter compatibility code.
Validate changed shared code on both maintained versions; SDK-specific imports
stay behind the existing interpreter/runtime boundaries. Do not raise the Python
minimum to follow Isaac 6.2. `geniesim docker` defaults to `docker6.2`;
Isaac Sim 4.5, standalone 5.1 and the 5.1 + 6.2 combo are deprecated (E.O.L.).
Retain their code until explicit retirement; do not add legacy qualification work.
Acceptance is recorded separately for each runtime.

Use uppercase Markdown filenames in `docs/` (for example, `STEP_MODES.md`).

---

## Maintainer rules — non-negotiable

These are the maintainer's rules for this workspace. An AI coding agent
**must not violate them**, and must not route around one "just to get a
demo running". If a rule blocks the change you want to make, stop and
raise it in the MR instead of adding a second path.

### 1. The engine is configured from exactly two YAML files

The physics engine takes its configuration from **the launcher YAML**
and **the scene YAML**. Nothing else is a configuration channel.

| Source | Home | Reaches the engine via |
|---|---|---|
| **Launcher YAML** | `genie_sim_bringup/config/launcher_*.yaml` | ROS node parameters on the engine node |
| ↳ physics-params sidecar | `genie_sim_engine_core/config/physics_params*.yaml` | the launcher's `physics_params_file` parameter — `session.py:277`, `load_physics_params` |
| ↳ `scene_yaml_defaults:` | a block **inside** the launcher YAML | deep-merged *underneath* the scene YAML, so the scene always wins — `session.py:134-152` |
| **Scene YAML** | `genie_sim_bringup/config/scene_*.yaml` | assemble manifest → `SceneConfig` |

The physics-params sidecar is not a third source: it is reachable only
because a launcher YAML names it. A knob no launcher can point at does
not exist.

**Forbidden, without exception:**

- **Environment variables.** No `RLINF_*`, no `GENIE_*`, no
  `os.environ.get(...)` read that changes physics, solver, scene, or
  pacing behaviour. Express tunables through reachable YAML keys.
- **Hardcoded module constants that shadow a tunable.** A value written
  as `_DEFAULT_FOO = 15.0` at module scope in an adapter and plumbed
  through a constructor kwarg is a second, undocumented route to the
  same physics. If a value is worth tuning it gets a YAML key and a row
  in the relevant `docs/` reference. A module constant is legitimate
  only as the fallback a *documented YAML key* resolves to when unset.
- **`/tmp` flag files, sentinel files, and debug switches** consulted at
  runtime to change behaviour.
- **CLI-only knobs** that no YAML can set.

**Corollary — one knob, one route.** The same physical quantity must not
be settable from two places. If the single-env and multi-env adapters
both need `impratio`, they read the same YAML key; they do not each
carry their own default.

### 2. Every tunable is documented where it is read

A new YAML key lands together with its row in the owning `docs/`
reference — [`physics_knobs.md`](src/ros_ws/src/genie_sim_engine_core/docs/physics_knobs.md),
[`STEP_MODES.md`](docs/STEP_MODES.md), and friends — including *why* the shipped value is the
value it is. Numbers that were fixed by measurement say so, and say what
was measured, so the next reader does not retune them on a hunch.

### 3. Python is formatted with `black --line-length 120`

Any change that touches Python in this workspace ends with the formatter,
run from the repo root:

```bash
black --line-length 120 source/geniesim_ros
```

120 columns is the workspace setting — do not run bare `black` (88
columns) and do not reformat with a different tool. An AI coding agent
runs this as the last step of a Python edit, so the diff carries no
formatting noise and no unrelated reflow.

---

## Pinned runtime versions

Isaac release declarations live in the repository's
[Docker runtime guide](../../docker/README.md). The dedicated 6.2 image uses base
`ISAACSIM_TAG=6.2.0.1`; its system-Python adapter registers the existing bundled
SDK with compatibility version `ISAACSIM_PIP_VERSION=6.2.0.0`.
Newton is installed by the SDK; the ROS package declares MuJoCo's coupled
requirement `mujoco~=3.12.0` from Newton 1.6.1's metadata.
CoACD (`coacd>=1.0.7`) is a default ROS distribution dependency, so ordinary
bootstrap and the Docker dependency collector install it. Installation does not
enable CPU decomposition; scene configuration still selects that behavior.

| Package | Current system runtime |
| --- | --- |
| Newton | 1.6.1 |
| MuJoCo / MuJoCo-Warp | 3.12.0 |
| newton-usd-schemas | 0.5.0 |
| mujoco-usd-converter | 0.6.1 |

Keep the MuJoCo requirement aligned when changing the Isaac wheel. Generated
SDK/colcon constraints and the Newton metadata guard validate system dependency
installation. Bundled and system interpreters require separate acceptance.

Verify actual imports, not only metadata, for the selected interpreter:

```bash
python3 -c "import newton, mujoco, warp; print(newton.__version__, newton.__file__, mujoco.__version__, warp.__version__)"
```

Version-dependent code names its branch version in a comment. Loaded paths and
startup/config readback determine which runtime a result applies to. See
[runtime acceptance](docs/ISAACSIM_ACCEPTANCE.md) and the [roadmap](docs/ROADMAP.md).

---

## ROS packages

**Doc convention.** Every ROS package ships **both** a `README.md`
(big-picture for humans — what the package does, when you'd touch
it, one usage hook) and an `AGENTS.md` (routing + mechanism for
agents — file layout, plugin registration, dispatch invariants).
The table below links both for every package, plus any `docs/`
deep-dive directory. New packages **must** ship both files before
the package is added to this table; the doc-coverage audit
(`geniesim tool docs --scope ros` — see end of this doc) enforces it.

| Package | Purpose | README | AGENTS |
|---|---|---|---|
| `genie_sim_common` | Shared CMake helpers (`genie_sim_package()` macro, test wrappers) consumed by every other package | [README](src/ros_ws/src/genie_sim_common/README.md) | [AGENTS](src/ros_ws/src/genie_sim_common/AGENTS.md) |
| `genie_sim_client` | Application-neutral Newton process, state, step/reset and lifecycle client | [README](src/ros_ws/src/genie_sim_client/README.md) | [AGENTS](src/ros_ws/src/genie_sim_client/AGENTS.md) |
| `genie_sim_bringup` | Launch orchestration, scene / launcher YAML, config | [README](src/ros_ws/src/genie_sim_bringup/README.md) | [AGENTS](src/ros_ws/src/genie_sim_bringup/AGENTS.md) |
| `genie_sim_engine_core` | `PhysicsEngine` ABC, C++ ROS 2 bridge, assemble-pipeline framework — deep-dives in [docs/](src/ros_ws/src/genie_sim_engine_core/docs/) | [README](src/ros_ws/src/genie_sim_engine_core/README.md) | [AGENTS](src/ros_ws/src/genie_sim_engine_core/AGENTS.md) |
| `genie_sim_engine_isaacsim` | Kit-hosted backends (Isaac PhysX / Isaac Newton), URDF→USD and `isaac_scene` stages | [README](src/ros_ws/src/genie_sim_engine_isaacsim/README.md) | [AGENTS](src/ros_ws/src/genie_sim_engine_isaacsim/AGENTS.md) |
| `genie_sim_engine_newton` | Newton-standalone (Kit-free) backend, solver adapters, `newton_extras` stage — deep-dives in [docs/](src/ros_ws/src/genie_sim_engine_newton/docs/) | [README](src/ros_ws/src/genie_sim_engine_newton/README.md) | [AGENTS](src/ros_ws/src/genie_sim_engine_newton/AGENTS.md) |
| `genie_sim_engine_libuipc` | Independent native IPC backend with CPU inputs and completed-state observations | [README](src/ros_ws/src/genie_sim_engine_libuipc/README.md) | [AGENTS](src/ros_ws/src/genie_sim_engine_libuipc/AGENTS.md) |
| `genie_sim_render` | OVRtx (C++) and Isaac Sim (Python) render nodes | [README](src/ros_ws/src/genie_sim_render/README.md) | [AGENTS](src/ros_ws/src/genie_sim_render/AGENTS.md) |
| `genie_sim_robot_model` | Robot xacro / URDF descriptions + pre-baked USDs + mesh tools | [README](src/ros_ws/src/genie_sim_robot_model/README.md) | [AGENTS](src/ros_ws/src/genie_sim_robot_model/AGENTS.md) |
| `genie_sim_rviz_plugins` | Custom RViz2 display plugins | [README](src/ros_ws/src/genie_sim_rviz_plugins/README.md) | [AGENTS](src/ros_ws/src/genie_sim_rviz_plugins/AGENTS.md) |
| `genie_sim_moveit` | MoveIt 2 config package for Genie G2 (SRDF, kinematics, OMPL, ros2_control wiring, WBC launch) | [README](src/ros_ws/src/genie_sim_moveit/README.md) | [AGENTS](src/ros_ws/src/genie_sim_moveit/AGENTS.md) |
| `genie_sim_moveit_plugins` | MoveIt 2 IK plugins (KDL-coupled, bio_ik-coupled, relaxed-IK) + RRT-Connect / TOPP-RA planner | [README](src/ros_ws/src/genie_sim_moveit_plugins/README.md) | [AGENTS](src/ros_ws/src/genie_sim_moveit_plugins/AGENTS.md) |
| `genie_sim_control` | ros2_control hardware interface + planar-base controller plugin | [README](src/ros_ws/src/genie_sim_ros_control/genie_sim_control/README.md) | [AGENTS](src/ros_ws/src/genie_sim_ros_control/genie_sim_control/AGENTS.md) |
| `genie_sim_controllers` | ros2_control 4WS chassis servo + MPC (OSQP) + ServoBase strategies | [README](src/ros_ws/src/genie_sim_ros_control/genie_sim_controllers/README.md) | [AGENTS](src/ros_ws/src/genie_sim_ros_control/genie_sim_controllers/AGENTS.md) |
| `genie_sim_planning` | Python chassis helpers + demo scripts | [README](src/ros_ws/src/genie_sim_planning/README.md) | [AGENTS](src/ros_ws/src/genie_sim_planning/AGENTS.md) |

---

## Build

`genie_sim_robot_model` is an independently maintained Git submodule at its
existing ROS package path. Initialize it before building:

```bash
git submodule update --init -- source/geniesim_ros/src/ros_ws/src/genie_sim_robot_model
```

Commit and publish description changes in the robot-model repository first,
then update the parent gitlink. The parent release command reads `.gitmodules`
and leaves submodule package versions unchanged. Do not vendor another copy.

```bash
geniesim ros build dev      # colcon build --symlink-install (development)
geniesim ros build release  # colcon build Release
```

Or directly:
```bash
cd source/geniesim_ros/src/ros_ws
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
source devel/setup.bash
```

---

## Wheel design

`geniesim_ros` ships as a single `pip`-installable wheel that contains
the **pre-built colcon install tree** of the ROS packages selected by the wheel build.
Consumers `pip install geniesim_ros-<ver>-py3-none-any.whl` and source
`setup.bash` from the path returned by `setup_bash_path()` — no colcon
required on the consumer side.

### Layout

```
geniesim_ros-<ver>-py3-none-any.whl   (zip, produced by setup.py)
└── geniesim_ros/
    ├── __init__.py                  ← imports + calls ensure_ros_install()
    ├── _bootstrap.py                 ← lazy-extract helper (this file)
    └── _ros_install.tar.gz           ← gzipped colcon --merge-install tree
        (extracted on first import to <pkg_dir>/_ros_install/)
            setup.bash
            share/<pkg>/package.xml
            lib/<pkg>/<entrypoint>.py     (mode 0o755 — preserved by tar)
            lib/python3.X/site-packages/<pkg>/...
            ...
```

### Why a tarball, not raw `package_data`

Setuptools' `package_data` glob copies files through
`distutils.file_util.copy_file` (which strips `st_mode`) and then
`bdist_wheel` writes every entry with the canonical wheel mode
`0o100664`. ROS 2 entrypoints under `lib/<pkg>/<script>.py` consequently
lose their `0o755` bit, and `ros2 run <pkg> <script>` fails because it
filters by `os.access(path, os.X_OK)`.

`tarfile` round-trips `st_mode` natively — wrapping the install tree in
a `.tar.gz` and shipping that single archive as `package_data`
preserves modes through the pip pipeline. The wheel is also smaller
(gzip beats zip-store on Python source) and faster to assemble (one
archive, one `RECORD` line).

### Build path (`pip install <path>` / `geniesim deploy geniesim_ros`)

`setup.py:BdistWheelWithColcon.run()` chains:

1. `_stage_colcon_into_package()` — runs `colcon build --merge-install
   Release` into `src/geniesim_ros/_ros_install/` (deploy-only build/log
   dirs, namespaced so they cannot collide with `geniesim ros build
   dev|release`).
2. `_prune_pycache()` — strips `__pycache__` so the tarball is clean.
3. `_tar_staged_install()` — tars `_ros_install/` →
   `_ros_install.tar.gz` with deterministic ordering / zeroed
   uid·gid·mtime (reproducible builds), then `rmtree`s the source dir
   so setuptools can't double-ship it.
4. `super().run()` — the stock `bdist_wheel` then assembles the wheel
   with `_ros_install.tar.gz` as the only `package_data` entry.

`pyproject.toml` declares `package-data = {"geniesim_ros":
["_ros_install.tar.gz"]}`. Override knob `GENIESIM_ROS_SKIP_BUILD=1`
skips step 1 (useful when a prior `geniesim ros build release` already
populated `_ros_install/`).

### Editable path (`pip install -e <path>`)

PEP 660 routes through `develop` / `editable_wheel`, which never reach
`BdistWheelWithColcon`. No tarball is built; the Python shim is
installed by reference; the dev runs `geniesim ros build dev`
themselves to populate `_ros_install/` next to the source tree.
`ensure_ros_install()` finds the directly-staged tree and uses it.

### Install path (consumer side)

`geniesim_ros/__init__.py` calls `ensure_ros_install()` on first import.
The helper:

1. **Wheel install case** — `_ros_install.tar.gz` is present beside
   `__init__.py`. SHA-256 the tarball; compare against the
   `_ros_install/.bootstrap_hash` sidecar.
   * Match → return `<pkg_dir>/_ros_install/`.
   * Mismatch (or no extraction yet) → `flock`-guarded `rmtree` +
     `tarfile.extractall(filter='data')` + write the new sidecar.
     `filter='data'` is path-traversal-safe on Python 3.12+; older
     Pythons fall through to plain `extractall` (we own the tarball
     content, so the threat surface is bounded).
2. **Editable / source-checkout case** — no tarball. If
   `<pkg_dir>/_ros_install/` already exists (dev built it), return it.
   Otherwise return `None` — caller must surface "user must build the
   workspace first".

Hot path is one `stat` + one string compare. Concurrent `import
geniesim_ros` calls (e.g. pytest workers) race the `flock`; the loser
double-checks inside the lock and returns immediately.

### Public API on `geniesim_ros`

| Function | Returns |
|---|---|
| `ensure_ros_install()` | path to extracted `_ros_install/`, or `None` if unavailable |
| `install_root()` | alias of the above (more obvious for shell discovery) |
| `setup_bash_path()` | path to `setup.bash` inside the install tree, or `None` |

Typical consumer use:

```bash
source $(python3 -c 'from geniesim_ros import setup_bash_path; print(setup_bash_path() or "")')
```

### Vendor consumer (ament workspaces)

When integrated into an ament workspace via a vendor meta-package, the
consumer doesn't go through pip. CMake unzips the wheel itself, then
`cmake -E tar xf` the inner `_ros_install.tar.gz`, then
`install(DIRECTORY … USE_SOURCE_PERMISSIONS)` overlays into
`CMAKE_INSTALL_PREFIX`. Both paths preserve modes; both treat
`_ros_install/` as a self-contained colcon `--merge-install` prefix.
`USE_SOURCE_PERMISSIONS` is non-negotiable on the CMake side — without
it CMake's default file mode resets the bug the tarball was built to
avoid.

### Layout invariants (the contract)

External consumers treat these as stable:

1. Wheel ships exactly one payload: `geniesim_ros/_ros_install.tar.gz`.
   No raw `_ros_install/<files>` entries inside the wheel.
2. Inside the tarball: colcon `--merge-install` layout, with
   `setup.bash` at the root and `share/genie_sim_engine_core/package.xml`
   present (commonly used as a post-extract sentinel by external
   tooling).
3. Scripts under `lib/<pkg>/` carry `0o755`; preserved by `tarfile` on
   write and by `extractall` / `USE_SOURCE_PERMISSIONS` on read.
4. Standalone `.so` libraries (`libgenie_sim_render_*.so`,
   `libgenie_sim_rviz_plugins.so`) live directly under `lib/`.

Changes to any of these need to be coordinated with downstream
integrators.

---

## Typical launch

```bash
ros2 launch genie_sim_bringup app.launch.py \
  scene:=scene_flat_acone \
  launcher_config:=launcher_ovrtx_isaac_physx \
  headless:=false
```

Newton/MuJoCo-Warp backend (Isaac's Newton wrapper, experimental):
```bash
ros2 launch genie_sim_bringup app.launch.py \
  scene:=scene_flat_acone \
  launcher_config:=launcher_ovrtx_isaac_newton \
  physics_hz:=200.0
```

## Operator skills

See the [skills index](skills/README.md) for end-to-end recipes. The
[Wuji chef Newton skill](skills/run-wuji-chef-newton/SKILL.md) is the canonical
Newton XPBD engine startup recipe for the Wuji wok scene.

---

## Key data flows

```
scene_*.yaml + launcher_*.yaml
        │
        ▼
app.launch.py ──► physics_backend.launch.py
                        │
                        ├─► python -m genie_sim_engine_core.assemble
                        │       ├─ urdf_to_usd    (URDF→USD, cache-gated via --skip-stage)
                        │       ├─ isaac_scene    (manifest.json, always runs)
                        │       └─ newton_extras  (self-skips without newton.entries)
                        ├─► genie_sim_engine_isaacsim_node.py  (Isaac PhysX / Isaac Newton physics loop)
                        ├─► genie_sim_engine_newton_node.py    (Newton-standalone physics loop, Kit-free)
                        ├─► render_ovrtx / render_isaacsim (optional)
                        └─► industrial_bridge (optional)
```

---

## Isaac Sim compatibility & asset layouts

The active runtime is Isaac Sim 6.2. Deprecated (E.O.L.) 4.5, 5.1 and combo
paths retain their importer and asset-layout compatibility code until retirement.
The table below describes the retained dispatch code. The `physics_engine`
selector is the user-visible switch for the active runtime:

| `physics_engine` | Solver | URDF→USD importer / Asset layout | Mimic-joint enforcement | Cloth / soft bodies |
|---|---|---|---|---|
| (E.O.L. Isaac Sim 4.5 / 5.1 — retained PhysX-only code) | PhysX (`omni.physx`) | `URDFParseAndImportFile` (kit command); **AS2 / flat** single-file `robot.usda`, every link direct child of `/<robot_prefix>` | engine-side broadcast (manifest `mimic_joints` block) | yes via PhysX (PhysxSchema) |
| `isaac_physx` (Isaac Sim 6.x) | PhysX 5 (`omni.physx`) | `urdf_usd_converter.Converter` + `importer_utils.convert_joints_attributes` (PhysX post-pass) + `run_asset_transformer_profile`; **AS3** multi-file package with links nested under `/<prefix>/Geometry/.../link`, physics split across `payloads/Physics/{physics,physx,mujoco}.usda` | `PhysxMimicJointAPI:<axis>` (PhysX articulation solver) | yes via PhysX (`PhysxParticleClothAPI`, etc.) |
| `isaac_newton` (Isaac Sim 6.x) | Newton via `isaacsim.physics.newton` wrapper | `urdf_usd_converter.Converter` (raw, no PhysX post-pass); **AS3** — same nesting | `NewtonMimicAPI` (Newton native constraint) | **no** — wrapper bridges only `NewtonArticulationRootAPI` / `NewtonMaterialAPI` / `NewtonMimicAPI` (rigid-body schemas only); `newton_usd_schemas` defines no cloth/particle/deformable APIs |

The bare `physx` and `newton` ids are **rejected** —
`runtime.bootstrap._validate_engine_id` raises `ValueError` on anything
outside the canonical set.

### How the workspace dispatches

- **Conversion** — `genie_sim_engine_isaacsim/assemble/_robot_impl.py:_convert_urdf_to_usd`
  reads `importlib.metadata.version("isaacsim")` and routes to either
  `_convert_urdf_to_usd_as3` (AS3 path) or `_convert_urdf_to_usd_as2`
  (AS2-flat path). Neither path performs URDF pre-processing at runtime —
  OBJ name normalization and inertial-block injection
  are all offline mesh-developer steps owned by `genie_sim_robot_model`
  (see [Mesh-developer tools](src/ros_ws/src/genie_sim_robot_model/AGENTS.md#mesh-developer-tools)).
- **Layout detection at runtime** —
  `genie_sim_engine_isaacsim/kit/stage.py:_detect_asset_format` returns
  `"as3"` when a `payloads/` directory sits next to `robot.usda`,
  otherwise `"flat"`. The flag selects `_collect_joints_as3` (sublayer-aware
  joint walk) vs `_collect_joints` (single-stage walk).
- **Body discovery is layout-agnostic** — `_collect_body_paths` walks the
  full subtree under `/<prefix>` via `Usd.PrimRange` filtered by
  `HasAPI(RigidBodyAPI)`. Every URDF link that becomes a real rigid body
  carries that schema in all three pipelines, so this discriminator works
  uniformly without inspecting `_asset_format`.
- **`/tf_render` protocol** — `child_frame_id` is the absolute USD prim
  path; transform is **local relative to immediate USD parent**, not world.
  Sending world transforms "works" for flat 4.x/5.x by accident (parent at
  identity) but composes through the parent chain a second time on AS3 and
  visibly disassembles the robot. Sending local lets the renderer use its
  ordinary `xformOp:translate`/`xformOp:orient` write path, with USD's
  standard composition reconstructing the world pose through the matching
  kinematic hierarchy. See
  [genie_sim_render/AGENTS.md](src/ros_ws/src/genie_sim_render/AGENTS.md).
- **Physics engine selection** — runtime knob
  `physics_engine:=isaac_physx|isaac_newton` (with
  `physics_solver:=mujoco|xpbd|...` for `isaac_newton`). The same
  `robot.usda` is consumed by both 6.x backends; neither rewrites the
  asset on switch. Validated by
  `runtime.bootstrap._validate_engine_id` — anything outside the
  canonical set raises `ValueError`.

### Gotchas the workspace already handles

- **Newton-first 6.x converter authoring** — when the URDF root link is
  empty (UR / Robotiq / G2 / Aloha put inertia on a separate `*_inertia`
  link connected by a fixed joint), the 6.x converter takes its
  `is_ghost_link` branch and authors `body0 = default_prim` for **every**
  joint, fixed joints included. PhysX cannot construct an articulation
  from such USD. **Fixed at the URDF layer**, not at runtime — the
  `genie_sim_robot_model` package ships
  `scripts/diagnose_urdf.py` which detects links missing
  `<inertial>` and (in fix mode) injects a placeholder block in the
  xacro source. Run it whenever you add or modify a URDF link. See
  [engine_core/docs/pipeline.md → URDF/xacro authoring as the source of truth](src/ros_ws/src/genie_sim_engine_core/docs/pipeline.md#urdfxacro-authoring-as-the-source-of-truth).
- **Mimic-joint discriminator drift** — the URDF→USD converters all apply
  `DriveAPI` to every revolute joint, so a `HasAPI(DriveAPI)` discriminator
  silently misclassifies all five Robotiq mimic followers as masters,
  fights the constraint through the master, and freezes the gripper. The
  unified discriminator in `_configure_drives` checks for
  `PhysxMimicJointAPI:*` **or** `NewtonMimicAPI`; absence of both falls
  into the master-drive path, which is the right behavior for both
  6.x mimics (constraint solver enforces) and the 4.x/5.x no-mimic case
  (constraint isn't enforced — known importer limitation, use 6.x if
  you need URDF mimic semantics).
- **AS3 articulation root path** — `base_link` lives at `/<prefix>/base_link`
  (flat) or `/<prefix>/Geometry/base_link` (AS3). `snapshot_odom` walks the
  subtree to resolve the path on first call, then caches it.
- **OBJ object-name de-duplication** — the 6.x converter keys each authored
  geometry by the .obj file's `o <name>` directive and de-dupes by name
  across files. Most .obj files exported from 3ds Max / Blender carry
  generic default object names (`Cylinder001`, `Box001`, `s`, `2`, `1`)
  that collide aggressively across a multi-link robot: first occurrence
  wins, the rest get a `_N` suffix or are silently dropped, and per-mesh
  references in `instances.usda` end up pointing at the wrong geometry.
  ARX acone is the canonical case — 14+ visual meshes share four distinct
  internal names, leaving the assembled USD missing arm_r_* and half the
  gripper visuals. UR5 doesn't trip it because its `.dae` meshes carry
  per-link names. **Fixed at the mesh-source-of-truth**, not at runtime —
  the `genie_sim_robot_model` package ships a developer tool
  `scripts/normalize_obj_names.py` that rewrites every `.obj` in place to
  carry exactly one `o <file_stem>` directive, eliminating cross-file
  collisions deterministically. Mesh developers run it (with `--dry-run`
  to diagnose) after every batch of DCC exports. See
  [genie_sim_robot_model/AGENTS.md → Mesh-developer tools](src/ros_ws/src/genie_sim_robot_model/AGENTS.md#mesh-developer-tools).

For per-component design rules (joint authoring, drive overrides, mass
authoring, articulation gain handling) see
[engine_core/docs/pipeline.md → Cross-version invariants](src/ros_ws/src/genie_sim_engine_core/docs/pipeline.md#cross-version-invariants).

---

## Routing rules

- Launch entry point → `genie_sim_bringup/launch/app.launch.py`
- Physics loop (Isaac Sim engines) → `genie_sim_engine_isaacsim/scripts/genie_sim_engine_isaacsim_node.py`
- Physics loop (Newton-standalone) → `genie_sim_engine_newton/scripts/genie_sim_engine_newton_node.py`
- Render node → `genie_sim_render/scripts/isaacsim_render.py` or `src/render_node.cpp`
- Robot descriptions → `genie_sim_robot_model/xacro/robot.xacro`
- MoveIt config + launches → `genie_sim_moveit/`
- IK / planner plugins → `genie_sim_moveit_plugins/`
- ros2_control hardware + planar-base → `genie_sim_ros_control/genie_sim_control/`
- 4WS chassis servo → `genie_sim_ros_control/genie_sim_controllers/`
- RViz2 plugins → `genie_sim_rviz_plugins/src/`
- Shared launch helpers → `genie_sim_bringup/launch/utils.py`

---

## 🔗 ROS-package DAG — methodology

**The rendered diagram lives in [`README.md`](README.md).** This section explains how it's generated so contributors can extend it; the diagram itself is read on the README page (GitHub renders Mermaid inline).

### Generator

```bash
geniesim tool ros-dag             # verify the block in source/geniesim_ros/README.md is current
geniesim tool ros-dag --fix       # regenerate the block in place
```

Source of truth: every `package.xml` under `src/ros_ws/src/<pkg>/`. Each XML tag becomes an edge in a defined order — the first matching tag for a given `(src, dst)` wins so build-time deps never get mis-rendered as exec.

| `package.xml` tag | Mermaid edge | Meaning |
|---|---|---|
| `<buildtool_depend>` | `-->|buildtool|` | Build toolchain (e.g. `ament_cmake`, `ament_cmake_python`) |
| `<build_depend>` | `-->|build|` | C++/CMake build-time dependency |
| `<exec_depend>` | `==>|exec|` | Runtime dependency |
| `<depend>` | `==>|exec|` | Combined build + exec — rendered as exec (the runtime contract is what's interesting in a workspace diagram) |
| `<test_depend>` | `-.->|test|` | Test-only dependency |

External packages (system libs like `rclpy`, `python3-yaml`, `ament_cmake`) are filtered out — only edges between packages that **both live in `ros_ws/src/`** are emitted. This keeps the diagram focused on intra-workspace coupling.

### Edge taxonomy alignment

The arrow taxonomy is intentionally the same as the Python-peer DAG in [`source/README.md`](../README.md):

- **Thin `-->`** = build / packaging coupling.
- **Thick `==>`** = runtime / import coupling.
- **Dashed `-.->`** = optional or conditional.

A reader who's seen one diagram can read the other without re-learning the legend.

### Marker contract

The generator writes between `<!-- AUTOGEN:ros-dag start -->` and `<!-- AUTOGEN:ros-dag end -->` in `source/geniesim_ros/README.md`. Both must be present; missing markers raise a hard error. Do not hand-edit between them — CI runs `geniesim tool ros-dag` (without `--fix`) and fails on drift.

### Adding a new edge category

To surface a relationship that isn't in `package.xml` metadata (e.g. a runtime ROS-topic contract), add a new emission pass inside `_emit_ros_mermaid` in [`source/geniesim_cli/src/geniesim_cli/commands/tool.py`](../geniesim_cli/src/geniesim_cli/commands/tool.py) and define a new Mermaid arrow style. The `_ROS_DEPEND_TAGS` table is the source-of-truth ordering — append there, don't reorder.

---

## Doc-coverage audit

The convention "every package ships README.md + AGENTS.md and gets a
row in the packages table above" is enforced by the central audit:

```bash
geniesim tool docs --scope ros          # human-readable, exit 1 on violations
geniesim tool docs --scope ros --quiet  # silent on success (CI hook form)
```

Exits 0 with a green banner when every package satisfies the rules;
exits 1 and lists the violations otherwise. Four invariants checked:

1. Every directory under `src/ros_ws/src/` with a `package.xml`
   (excluding `external/` / `build/` / `install/`) has both
   `README.md` and `AGENTS.md` siblings.
2. Every such package has a row in this file's packages table.
3. Every markdown link in this file (and in every per-package
   `README.md` / `AGENTS.md`) points at a file that exists.
4. Every `package.xml` uses the standard maintainer and license
   lines (see `Package metadata convention` below) and ships a
   `LICENSE` file alongside it.

Wire `geniesim tool docs --scope ros` into pre-commit / CI to make
new packages fail the merge if they don't ship both files.

See [`source/geniesim_cli/src/geniesim_cli/commands/tool.py`](../geniesim_cli/src/geniesim_cli/commands/tool.py) for the audit source.

---

## Package metadata convention

Every `package.xml` under `src/ros_ws/src/` uses the same maintainer
and license lines, and every package ships its own `LICENSE` file
(a copy of the workspace-level MPL-2.0 text — the `<license
file="LICENSE">` reference is resolved relative to the package
directory, not the workspace root):

```xml
<maintainer email="yangwendisheng@agibot.com"> Genie Sim Team </maintainer>
<license file="LICENSE">Mozilla Public License Version 2.0</license>
```

Every package also declares `<depend>genie_sim_common</depend>` and
calls `find_package(genie_sim_common REQUIRED)` + `genie_sim_package()`
at the top of its `CMakeLists.txt` — see
[`genie_sim_common/cmake/genie_sim_package.cmake`](src/ros_ws/src/genie_sim_common/cmake/genie_sim_package.cmake)
for the macro (C++20 default, `-Wall -Wextra -Wpedantic`, opt-in
`WERROR`, test wrappers).

## Newton Wuji evaluation configuration

`launcher_newton_wuji_eval` pairs with the separate
`scene_flat_g2_sp_chef_wuji_eval` procedural baseline. Newton owns physics and
requests frozen-state observations from a separate Isaac RTX worker. The default
authored chef scene has its own physics and recording configuration.
