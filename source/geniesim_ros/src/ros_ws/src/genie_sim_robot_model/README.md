# genie_sim_robot_model

Robot descriptions for GenieSim — xacro / URDF sources, meshes (`.dae`,
`.STL`, `.obj`), pre-baked USDs, and the mesh-prep tools that keep
those assets clean enough for Isaac Sim's URDF→USD importer.

## Robots shipped

| Tier | Robots | What it means |
|---|---|---|
| **Tier 1 — actively supported** | Genie G2 (`crs` / `crsB` arms; Omnipicker, Swiftpicker, and crsB/Wuji hands) | Primary maintained models and physics/launch test scenarios. Runtime results depend on the selected scene and backend. |
| **Tier 2 — reference only** | `agilex/aloha`, `agilex/piper`, `arx/x5`, `arx/acone`, `franka/fr3`, `universal_robots/ur5` | Reference models; validate gains, contacts, and end-effector behavior for the selected platform. |

Bring-your-own robots are supported: drop a new xacro tree under
`robots/<vendor>/<model>/`, generate a flat URDF, and the engine's
assemble pipeline takes it from there.

The universal entry point is `xacro/robot.xacro`. The build generates 25
URDF variants; `gripper:=no_gripper` shares the same arm sources.

| Family (`robot_model`) | Arm | Gripper choices | Identified parameter selector |
|---|---|---|---|
| `genie` (`body:=g2`) | `crs` | `omnipicker`, `swiftpicker`, `no_gripper` | — |
| `genie` (`body:=g2`) | `crsB` | `omnipicker`, `swiftpicker`, `wuji`, `no_gripper` | `inertia_source:=sysid` |
| `arx` | `x5`, `acone` | Vendored default, `no_gripper` | `param_source:=sysid` |
| `agilex` | `piper`, `aloha` | Vendored default, `no_gripper` | — |
| `franka` | `fr3` | Vendored default, `no_gripper` | — |
| `universal_robots` | `ur5` | `robotiq_85`, `robotiq_140`, `no_gripper` | — |

Parameter selectors default to `default`. See
[OVERLAYS.md](robots/OVERLAYS.md) and [SYSID_PARAMS.md](robots/SYSID_PARAMS.md)
for measured values, fit conditions, and the distinction between authored
parameters and runtime solver values.

## Layout

```
robots/                               # per-robot xacro / URDF sources + meshes
├── genie/g2/                         # Tier 1
├── agilex/{aloha,piper}/             # Tier 2
├── arx/{x5,acone}/                   # Tier 2
├── franka/fr3/                       # Tier 2
├── universal_robots/ur5/             # Tier 2
└── */*/overlays/                     # identification overlays — see robots/OVERLAYS.md
urdf/                                 # generated flat URDFs (consumed by MoveIt / sim)
scripts/                              # mesh-prep / diagnostic tools
└── diagnose_urdf.py, normalize_obj_names.py, fix_dae_units.py,
    recompute_inertia.py, copy_dae_material.py, …
```

## Mesh-prep tools

DCC exports rarely arrive in a state the Isaac Sim importer is happy
with. The `scripts/` tools fix the common issues:

- **`diagnose_urdf.py`** — reports inertial, root, wheel-collider,
  tensor, and mesh-unit issues without rewriting files. Empty kinematic
  frames do not need placeholder mass on Isaac Sim 6.2; physical bodies
  need meaningful mass and inertia. Use `--dry-run` to fail on every finding.
- **`normalize_obj_names.py`** — rewrites every `.obj`'s `o <name>`
  directive to match the file stem, eliminating cross-file name
  collisions the Isaac Sim 6.x converter de-dupes against.
- **`fix_dae_units.py`** — scans `.dae` files exported in millimetres
  and rescales them to metres (matches the URDF / SDF / Newton / PhysX
  convention).
- **`copy_dae_material.py`** — copies the COLLADA material plumbing
  from a donor DAE into a target DAE (e.g. pick up the crsB look on a
  crs arm mesh exported without materials).
- **`recompute_inertia.py`** — recomputes inertial tensors from
  meshes under uniform density.

Run these once after every batch of DCC exports.

## When you'd touch this package

- Adding a new robot — drop xacros under `robots/<vendor>/<model>/`.
- Fixing meshes that don't import cleanly into Isaac Sim — start with
  the diagnostic tools in `scripts/`.
- Tuning a robot's URDF (joint limits, mass, inertia, mimic
  constraints).
- Adding a new gripper / end-effector variant — extend the existing
  G2 xacro composition pattern (`g2_<arm>_<gripper>.urdf.xacro`).

## Mechanics

See [AGENTS.md](AGENTS.md) for the robot composition pattern, mesh
conventions per format (`.dae` / `.obj` / `.STL`), the Isaac Sim
URDF→USD path each importer takes, and the gotchas the tools fix.
