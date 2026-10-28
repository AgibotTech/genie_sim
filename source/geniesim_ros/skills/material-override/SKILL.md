---
name: material-override
description: >
  Tune PBR metallic / roughness on URDF visual meshes without
  re-exporting DAE — author an inline `<material_override>` element
  inside any `<visual>`, and let the engine's URDF→USD pipeline patch
  the converter's defaults post-conversion. Useful for sim-to-real
  visual fidelity (DAE materials round-trip poorly; Isaac Sim's
  importer lands every embedded material at roughness=0.5,
  metallic=0.0 by default).
  Trigger: When the user asks to "tune material", "change roughness /
  metallic", "make the gripper shinier", "fix dull PBR", "override
  material", "PBR override", or any time the simulator looks washed
  out after a URDF→USD import.
license: MPL-2.0
metadata:
  author: genie-sim
  version: "1.1"
prerequisites:
  - geniesim_ros:build-workspace
inputs:
  - name: urdf_or_xacro_path
    desc: URDF / xacro file containing the `<visual>` to tune
    required: true
  - name: roughness
    desc: "PBR roughness 0..1 (omit to leave unchanged)"
    required: false
  - name: metallic
    desc: "PBR metallic 0..1 (omit to leave unchanged)"
    required: false
outputs:
  - desc: "`<material_override>` element added inside the target `<visual>`; engine regenerates `robot.usda` with patched metallic / roughness on next launch"
---

## When to Use

- Robot or asset visual meshes import with washed-out / flat
  shading after the URDF→USD converter runs.
- User wants to A/B PBR values without re-exporting DAE meshes from
  Blender / Maya.
- Polishing a robot for sim-to-real visual transfer.
- Author of a new URDF (see `add-robot` skill) wants the meshes to
  look right without round-tripping DAE materials.

Do **not** use for:
- Replacing albedo / textures → must be done in the DAE export.
- Per-prim material at runtime → that's a runtime USD edit, not the
  `<material_override>` element.
- Adding a brand-new material — only patches metallic / roughness on
  the embedded one.

## Critical Patterns

1. **Author inside `<visual>`, not on the link.** The override
   auto-scopes to exactly the mesh it sits under. A link with N
   `<visual>` elements just gets N independent overrides.
2. **Isaac Sim 6.2 AS3 assembly owns the extension.**
   The `urdf_to_usd` stage strips the element from the URDF it feeds the
   converter, then applies the values after the AS3 transformer via
   `_parse_material_override_blocks` / `_apply_material_overrides` in
   `genie_sim_engine_isaacsim/assemble/_robot_impl.py`.
3. **Only `roughness` and `metallic` are exposed.** Other UsdPreviewSurface
   inputs (specular, opacity, IOR) are not on the contract.
4. **Range 0..1.** Finite out-of-range values warn and clamp. Malformed,
   NaN and infinite fields warn and remain unchanged.
5. **No attributes needed.** Scoping is positional. Don't try to
   target a sibling visual by name — author one override per visual. The
   converter copy assigns stable names; private material references isolate
   shared materials. Source xacro and payload materials remain unchanged.
6. **Only the 6.2 path applies these overrides.** Isaac 4.5/5.1 is unchanged.
   Regenerate old caches after updating the converter naming implementation.

## Schema

```xml
<visual>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <geometry>
    <mesh filename="package://.../base_link.dae"/>
  </geometry>
  <material_override>
    <roughness>0.45</roughness>
    <metallic>0.85</metallic>
  </material_override>
</visual>
```

Either child is optional — author only what you need. Default for
an absent child is "leave as the converter wrote it" (typically
0.5 / 0.0 for DAE imports).

## PBR cheat-sheet

| Look | `roughness` | `metallic` | Notes |
|---|---|---|---|
| Brushed steel | 0.40 | 0.95 | Most arm shells |
| Anodised aluminium | 0.55 | 0.30 | UR-style structural |
| Polished chrome | 0.05 | 1.00 | Decorative trims, mirror parts |
| Matte plastic | 0.70 | 0.00 | Cable guards, end-stops |
| Rubber pads | 0.85 | 0.00 | Soft contact surfaces |
| Anodised gold (gripper bushings) | 0.30 | 0.90 | Highlights |

These are starting points; A/B against a photograph of the real
robot if you have one.

## Workflow

### Step 1 — Find the visual you want to tune

```bash
# Locate the URDF / xacro under genie_sim_robot_model
grep -nR "<visual>" source/geniesim_ros/src/ros_ws/src/genie_sim_robot_model/robots/ | head
```

### Step 2 — Add the override

Open the xacro, drop `<material_override>` into the `<visual>`
block. Existing reference: aloha arm xacro carries an override on
every visual:

```xml
<visual>
  <origin xyz="0 0 0" rpy="0 0 0" />
  <geometry>
    <mesh filename="${mesh_dir}/base_link.dae" />
  </geometry>
  <material_override><roughness>0.45</roughness><metallic>0.85</metallic></material_override>
</visual>
```

### Step 3 — Rebuild the workspace

```bash
geniesim ros build dev && source devel/setup.bash
```

### Step 4 — Force a USD regenerate

The robot USD is cache-gated by `manifest.json`. Bust the cache:

```bash
rm -rf assets/scenes/<scene>/
# OR pass at launch:
ros2 launch genie_sim_bringup app.launch.py … always_regenerate_robot_usd:=true
```

### Step 5 — Launch and inspect

```bash
ros2 launch genie_sim_bringup app.launch.py \
  scene:=<scene> \
  launcher_config:=launcher_ovrtx_isaac_physx \
  headless:=false   # local workstation with a screen; flip to true on a remote/headless host
```

In the OVRtx viewport, compare against the unmodified baseline.
Iterate values, rebuild, re-launch.

### Step 6 — Verify the converted USD

Read the composed USD, including its payloads, rather than grepping the
root file alone. Use the USD-enabled Python in the prepared 6.2 runtime:

```python
from pxr import Usd, UsdShade
stage = Usd.Stage.Open("assets/scenes/<scene>/robot.usda")
for prim in stage.Traverse():
    if prim.IsA(UsdShade.Material) and "/GenieMaterialOverrides/" in str(prim.GetPath()):
        mat = UsdShade.Material(prim)
        print(prim.GetPath(), {name: mat.GetInput(name).Get()
                              for name in ("roughness", "metallic")})
```

Verify the intended visuals and preserved albedo in OVRtx or Isaac rendering.
Regression tests: `python3 -m pytest
source/geniesim_ros/src/ros_ws/src/genie_sim_engine_isaacsim/tests/test_material_overrides.py`.
The installed 6.2 converter enables the fresh-URDF SDK tests for repeated boxes
and meshes; other environments skip those cases explicitly.

## Notes

- The stage walks each `<visual>` in author order; if a visual has
  multiple meshes (rare in our robots), the override applies to every
  embedded material of that visual.
- The override is stripped only from the converter copy. MoveIt / RSP /
  urdfdom consume the authored xacro and ignore the extension.
- For brand-new materials (no DAE-embedded source), prefer authoring
  a real `<material>` block — material_override is patch-only.
- Material is rendered by OVRtx + Isaac Sim. The Newton GL viewer
  uses simple shading and won't reflect roughness / metallic — use
  OVRtx for visual QA.

## Resources

- **Engine source**: [genie_sim_engine_isaacsim/assemble/_robot_impl.py](../../src/ros_ws/src/genie_sim_engine_isaacsim/genie_sim_engine_isaacsim/assemble/_robot_impl.py) → `_parse_material_override_blocks` / `_apply_material_overrides`
- **Reference usage**: [source/geniesim_ros/src/ros_ws/src/genie_sim_robot_model/robots/agilex/aloha/aloha.urdf.xacro](../../src/ros_ws/src/genie_sim_robot_model/robots/agilex/aloha/aloha.urdf.xacro)
- **Pipeline doc** → "<material_override>": [genie_sim_engine_core/docs/pipeline.md](../../src/ros_ws/src/genie_sim_engine_core/docs/pipeline.md)
- **Related skills**: `add-robot`, `launch-scene`
