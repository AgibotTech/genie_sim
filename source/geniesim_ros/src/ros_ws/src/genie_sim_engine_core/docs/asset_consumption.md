# Asset consumption by physics backend

The [assembly pipeline](pipeline.md) generates `robot.usda`, its payloads and the
scene manifest. Runtime consumes the manifest and applies session configuration;
it does not silently repair or rebake source robot files.

## Asset layout

Isaac 6.2 conversion produces an AS3 package with `robot.usda`, `payloads/base.usda`,
geometry/material files and physics variants under `payloads/Physics/`.
Archive the complete directory. Isaac 4.5/5.1 import uses the flat AS2 layout.
Runtime discovers rigid-body owners by schema rather than assuming that mesh
wrapper names are link owners.

The Isaac assembly path selects `Physics=physx` before overlays and discovery.
Runtime selects the appropriate physics on the private scene reference before
robot updates. A root USDA default variant is not enough to establish the
composed physics variant.

## Consumption matrix

| Concern | Isaac PhysX | Isaac Newton wrapper (experimental) | Standalone Newton |
| --- | --- | --- | --- |
| Stage owner | Kit `IsaacSimStage` | Kit `IsaacSimStage` | USD stage and Newton model builder |
| Solver | PhysX | Wrapper's MuJoCo-Warp | Selected solver adapter |
| USD parsing | Isaac physics schemas | Wrapper's importer | `ModelBuilder.add_usd` with Newton, PhysX and Mjc resolvers |
| Class tuning | Profile-driven USD seeds and articulation gains | Seeds, tensors and model settings | Adapter fallbacks before construction |
| Selected sysid | Native 6.2 axis properties and SI gains | Final model overrides | SI model overrides before compilation |
| Base policy | Free or weld; mocap warns and uses weld | Same Kit policy | Free/weld or supported mocap adapter |
| Deformables | Outside the maintained rigid scene path | Unsupported | Scene entries/plugin and selected particle solver |

## Runtime boundaries

Standalone stage setup applies initial base pose and fixed-joint policies before
`add_usd`. A retained root weld is normalized only if its anchor misplaces the
robot. The model builder and adapter seed coordinates/targets and run FK before
stepping. Mimic policy, class tuning and sysid selection must resolve before the
solver compiles its actuator/passive arrays.

Kit stage setup applies profile and selected per-joint values to USD seeds before
reset, then configures SI articulation tensors. The wrapper also defends selected
values from its passive class writes. A live tensor or compiled-model readback is
stronger evidence than inspecting the saved root USDA.

`/tf_render` body paths identify physics-owned transforms. Transform messages are
local to the immediate USD parent; `/tf_scene` uses world poses. AS3 parent chains
make confusing these conventions visible as detached meshes or double offsets.

## Verification

- [Assembly harness](../../genie_sim_engine_newton/genie_sim_engine_newton/tools/AGENTS.md)
  produces per-stage snapshots and structural checks.
- Newton solver tools inspect model, actuator and compiled MJCF values.
- [Joint runtime tests](../../genie_sim_engine_isaacsim/AGENTS.md#runtime-sysid-precedence)
  check native axis properties, tensor gains and initialized models.
- [Bringup criteria](../../genie_sim_bringup/README.md#pass-criteria-matrix) cover
  fresh conversion, cache reuse, visual ownership and commanded motion.

The [joint parameter reference](../../../../../docs/JOINT_PARAMS.md) owns exact
field support and unit conventions.
