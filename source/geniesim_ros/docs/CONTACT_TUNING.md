# Contact tuning guide

Per-class contact parameter tuning for MuJoCo-based engines (Newton with
`mujoco-warp` / `mujoco-warp-multi-env` adapters).

## Overview

Contact behavior is controlled by four per-class parameters in scene YAML:
- **`solref`** — contact compliance (spring stiffness + damping)
- **`solimp`** — constraint impedance (how constraint violation is corrected)
- **`friction`** — three-axis friction coefficients
- **`condim`** — contact dimensionality (which friction axes are active)

Parameters are set per object **class**, not per asset. The classifier
automatically buckets shapes by topology and prim path into five classes:

| Class | What it matches |
|-------|-----------------|
| `robot` | colliders under the active `robot_prefix` |
| `floor` | the `add_ground_plane()` plane |
| `passive` | free scene rigid bodies (dropped objects, articulated scene props) |
| `static_prop` | world-static USD shapes (tables, walls, shelves) |
| `other` | unmatched (fallback) |

Plus one **refinement**, which is not a sixth class:

| Refinement | What it matches |
|-------|-----------------|
| `gripper_pad` | the `robot` shapes whose body is an annotated fingertip pad |

`gripper_pad:` inherits every value `robot` resolved to and overrides only the
keys it names. Membership is not in the YAML: the pads are the bodies carrying
the assemble-time `newton:isGripperPad` annotation — the same set the CoACD
pad decomposition runs over, with a gripper-classified `link4` fallback for
assets that predate the annotation. The YAML supplies the parameters, the asset
supplies which bodies are pads, so a differently-named gripper needs no YAML
edit and no scene can name a body that does not exist.

It is opt-in on the presence of the block. Without one the pads stay in
`robot`. With one present and no pad body identifiable, the engine warns by
name at bring-up and leaves everything on `robot` rather than pretending it
applied.

## Scene YAML schema

```yaml
newton:
  mjc_contact:
    robot:
      solref: [0.002, 1.0]             # [timeconst, dampratio] (2 floats)
      solimp: [0.9, 0.95, 0.001, 0.5, 2.0]  # [d0, dwidth, width, mid, pow] (5 floats)
      friction: [1.5, 0.5, 0.2]        # [tangential, torsional, rolling] (3 floats)
      condim: 6                        # 1/3/4/6 (int, optional)
    passive:
      friction: [1.0, 1.0, 0.0001]
      condim: 4
    gripper_pad:
      condim: 6                        # inherits robot's solref/solimp/friction
    floor:
      solref: [0.02, 1.0]
```

Each class gets its own block. Keys are optional — omit a class to use
defaults (MuJoCo stock), omit a key to use its default.

**Validation:** wrong-length tuples warn at build time and keep the default
rather than silently half-applying. So do an unknown class name
(`gripper_pads:`, `static-prop:`) and an unknown key inside a class
(`frictoin:`) — otherwise a configuration typo is indistinguishable from "I chose
not to override that". A typo fails loud, not silent.

## condim: contact dimensionality

`condim` controls which of the three friction coefficients are **active**:

| condim | Active friction axes | Use case |
|--------|----------------------|----------|
| 1 | none (frictionless) | debugging, ice rinks |
| 3 | tangential only | **MuJoCo default** — most scenes |
| 4 | tangential + torsional | stops resting objects from spinning |
| 6 | tangential + torsional + rolling | fingertip pads, high-fidelity grasping |

**Sharp edge:** authoring `friction: [1.5, 0.5, 0.2]` under the default
condim 3 silently ignores the `0.5` (torsional) and `0.2` (rolling). The
coefficients sit in the file, but MuJoCo never reads them.

**When to raise condim:**
- Objects spinning at rest despite zero angular velocity command → condim 4
  (torsional friction damps the spurious spin)
- Objects twisting/rolling out of gripper despite high friction values →
  condim 6 on `gripper_pad`, **not** on `robot` (see below)

**Cost:** Each dimension is another constraint Jacobian row per contact.
Moving every non-robot contact from 3 to 4 is roughly +33% rows. Budget
`mujoco_njmax` headroom (see launcher capacity section) before raising condim
across a contact-rich scene, or you'll start dropping contacts silently.

This is what `gripper_pad` exists for. `robot.condim: 6` covers every robot
shape — arm links, torso, base, all of it — and measured roughly 27% of
real-time (87 Hz against 120) for fidelity that only matters where the robot
actually grips. Put the `6` under `gripper_pad:` and leave `robot:` on `3`.

## Grasp-fidelity example

For manipulation tasks where gripper contact fidelity matters, use:

```yaml
newton:
  mjc_contact:
    robot:
      friction: [1.5, 0.5, 0.2]
      condim: 3          # arm and body links: tangential only
    gripper_pad:
      condim: 6          # fingertips only: all three slots live
    passive:
      friction: [1.0, 1.0, 0.0001]
      condim: 4
    floor:
      friction: [1.0, 1.0, 0.0001]
      condim: 4
    static_prop:
      friction: [1.0, 1.0, 0.0001]
      condim: 4
```

**Why these values:**
- `robot` tangential 1.5: Coulomb sliding friction for manipulation
- `robot` torsional 0.5 / rolling 0.2: fingertip pad grip (condim 6 required,
  which is why they are inherited by `gripper_pad` and inert everywhere else
  on the robot at condim 3)
- Non-robot torsional 1.0: stops resting objects from spinning at condim 4
  (mjwarp lacks implicitfast midpoint integration, so free bodies pick up
  spurious rotational drift; stock 0.005 barely damps it, 1.0 cuts ~91%)
- Non-robot rolling 0.0001: placeholder (condim 4 doesn't read it)

These are starting points, not universal constants. Tune per scene.

## Defaults (when omitted)

When a class or key is absent from the scene YAML, MuJoCo stock defaults apply:

```python
solref:   [0.002, 1.0]
solimp:   [0.9, 0.95, 0.001, 0.5, 2.0]
friction: [1.0, 0.005, 0.0001]
condim:   (not written → MuJoCo uses 3)
```

These match MuJoCo's own defaults, so an engine without tuning behaves like
vanilla MuJoCo.

## Where to put it

**Scene YAML**, not launcher YAML. Contact tuning is per-scene physics,
not a node parameter. The engine reads `newton.mjc_contact:` from the scene
config, not from ROS params.

**Exception:** a launcher may carry a `scene_yaml_defaults:` block, which is a
scene-YAML *fragment*, not a params block. It is deep-merged **underneath** the
scene YAML the manifest names — per key, recursing into mappings, treating
lists as leaves — so the scene's own keys always win and a scene naming only
`robot:` does not delete the launcher's `gripper_pad:`. Use it for per-fleet
defaults that every scene should inherit; use the scene YAML for anything one
scene needs to disagree about.

`launcher_newton_mjwarp_multi_env.yaml` supplies the grasp-fidelity profile in
that block. `test_h1_h2_pads.py` checks its per-class and pad settings.

## Stability check

The engine logs a warning when `substep_dt > min(timeconst)/2` across the
classes present in the scene. Above that threshold, MJW's Baumgarte
stabilizer silently softens contact every other substep, producing
"gripper walks through the box" symptoms no amount of tuning can fix.

Fix: raise `physics_solver_substep` (in launcher YAML) or relax the tightest
`solref` timeconst (in scene YAML). The log line names the needed substep count.

## Implementation

- `engine/newton/mjc_contact.py::_resolve_params()` — reads and validates
- `engine/newton/mjc_contact.py::apply_pre_finalize()` — writes per-shape
  into the builder's custom-attribute store
- `engine/newton/mjc_contact.py::_apply_override()` — folds one YAML block
  onto the resolved defaults; every rejection warns and keeps the prior value
- `common/gripper_annotations.py::read_pad_body_names()` — pad membership,
  read once by `setup/model.py` and passed to `apply_pre_finalize`
- `common/session.py::_deep_merge_under()` — the `scene_yaml_defaults` merge
- One summary log line per class with chosen values + stability check; the
  `gripper_pad` line is tagged `= robot` when it names no override of its own
