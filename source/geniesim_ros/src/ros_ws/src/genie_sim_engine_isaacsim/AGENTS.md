# genie_sim_engine_isaacsim

| Runtime | PhysX | Isaac Newton |
|---|---|---|
| 6.2 | AS3 variant selection and overlay inventory checks | Public configuration API and MuJoCo scene schema |
| 5.1 / 4.5 | Flat conversion/runtime paths retained | Not covered by the 6.2 public configuration contract |

This package owns Kit backends and Isaac-specific assembly. Keep Isaac/USD
imports lazy in assemble helpers; core must remain independent of all backends.
Use only launcher/scene YAML configuration routes described in
[geniesim_ros/AGENTS.md](../../../../AGENTS.md).

## Implementation

- `kit/launch_config.py`: shared, lightweight Kit startup compatibility.
  Isaac 6.2 disables `rtx.hydra.supportMultiTickRate` before app creation so
  explicit Replicator captures survive stage replacement and return RGB/depth.
  Apply it to converter, physics and renderer startup. Other SDK versions retain
  their launch config; this is an API compatibility requirement, not YAML tuning.
  Identify bundled Python by its executable under `ISAAC_PATH/kit/python` and
  read that SDK's `VERSION`; system Python uses installed wheel metadata.
  The dedicated docker6.2 image registers its bundled SDK through a small
  system-Python adapter distribution, so version metadata remains available
  without a second SDK payload. `kit/bootstrap.py` reads its
  `geniesim-bundled.json` manifest for apps/converter paths; the AS3 converter
  uses that same resolver. Pip-only and combo runtimes retain their resolution.

- `assemble/physics_variant.py`: select the known AS3 `physx` or legacy `PhysX`
  variant and check movable joints against URDF inventory. Prefer `physx` when
  both exist. No arbitrary variant fallback.
- Newton's spherical-joint axes retain their imported gains and force limits;
  do not apply scalar robot-joint tuning or guess unknown DOF names.
- `assemble/urdf_to_usd_stage.py`: construct AS3 manifest `robot.body_paths`
  from rigid link owners after selecting physics in a private session layer.
  A geometry-only walk can confuse mesh wrappers with links and omit
  `base_link`, leaving the chassis visual at the origin. Preserve source USD
  contents and variant selection; flat assets retain their existing walk.
- `assemble/_robot_impl.py`: the collision policy expands only instanceable
  geometry wrappers before editing collider attributes. Default USD traversal
  hides AS3 mesh proxies; proxies cannot receive per-instance policy edits.
  Keep rigid bodies and material instances intact, and use the literal `sdf`
  mesh-approximation token (it is a PhysX extension, absent from UsdPhysics.Tokens).
  Selection and count gates precede collision,
  material and joint overlays. Pad/coupler counts derive from URDF topology.
  Required annotation failures propagate; output is saved only after checks.
- Material overrides are scoped to the Isaac 6.2 AS3 converter. The converter
  copy gives each overridden visual a stable name and strips extension tags;
  the authored/raw URDF stays intact. Apply PBR after the transformer, with
  private material references/bindings on the selected visual in the generated
  root layer. Never mutate shared payload materials or fall back to every
  visual on a link. Finite scalars clamp to [0, 1]; invalid/non-finite fields
  warn and remain unchanged. Post-process failures must not produce a success
  stamp. Regenerate old caches after changes to the converter naming scheme.
- Isaac 6.2 assembly checks cache permissions before fetching/staging URDF or
  starting Kit. Replacing AS3 payloads needs write/search access in every nested
  directory, as well as writable root files. Refuse root writes into user-owned
  output directories (including a new scene's existing parent). Do not follow
  staged package symlinks or change ownership automatically. Container exec must
  use the host UID: `geniesim docker6.2 into` already does this; bare `docker exec`
  inherits the container's root setup user and bypasses the entrypoint privilege
  drop. Repair ownership only within the affected generated scene cache. On
  converter failure, print/flush the original traceback before `os._exit(1)`;
  otherwise Python never reports the failing recursive-copy operation.
- `kit/stage.py`: select the robot reference's variant before robot updates;
  collect composed joints rather than inventing paths from payload specs.
- `kit/episode_reset.py`: capture live articulation/root/target and free-body
  poses after initial joint targets are written. Exact USD inventory paths are
  required; PhysX wildcard matches do not recurse. Both Kit engines restore env 0,
  clear velocities and efforts, then advance `reset_epoch` after writes succeed.
  Newton additionally evaluates FK, copies both state buffers, updates Fabric
  and synchronizes the device. Keep the solver and runtime gains intact.
  Empty env lists do nothing; nonzero envs, seeded randomization and instance
  switching fail explicitly before writes. ROS command setpoints remain latched
  by the existing core contract: controllers must replace their previous command
  after reset. PhysX joint publication reads the live tensor state to avoid pairing
  a new epoch with old USD attributes.
- `kit/newton_config.py`: public `NewtonConfig`, `MuJoCoSolverConfig`,
  `configure_newton`, `get_newton_config` and `switch_newton_solver` contracts.
  Constructor values are configured before `IsaacSimStage` creates/reset its
  World. Iterations use `mjc:option:iterations`, consumed by Newton 1.6.1's
  registered MODEL/ONCE custom attribute. Do not put iterations back onto
  `MuJoCoSolverConfig` or restore the unused Kit `solver_type` setting.
- `kit/isaac_newton.py`: verify configuration after reset, retain existing
  model tuning and verify live cone/contact/iteration options on the first
  step before diagnostics or older model patches can mask a mismatch. A missing
  NewtonStage is an error, not a PhysX fallback. Read the live solver for the
  banner. Do not reinstate
  dataclass class/default mutation or private configuration imports.

Generated-asset checks run on the generated copy, never on source URDF/USD.
To recheck old caches, regenerate in a fresh output directory; counts alone
cannot establish whether cached asset tuning matches the current source.

## Verification

`tests/` is registered with `ament_add_pytest_test`. CPU configuration tests
run with `python3 -m pytest tests/test_newton_config.py`; USD tests require a
USD-enabled interpreter, and scene-schema tests require an initialized Isaac
app. A normal host run reports these as skipped, not as live acceptance.

The explicit `tests/isaac62_smoke.py` runtime smoke is documented in the migration roadmap. It must
read live solver options after stepping; reading only the pending public
configuration does not close the runtime gate. Keep 5.1/4.5 regressions and the
full scene/task/render matrix open until they have their own evidence.

## Runtime sysid precedence

`kit/joint_params.py` applies selected `robot.joint_overlays` after USD class
seeds, before reset, then defends drive values from the articulation class pass.
USD and tensor gains must agree after SI/per-degree conversion. Isaac 6.2
`PhysxJointAxisAPI` carries passive viscosity, static/dynamic friction effort and
armature independently of the drive. Do not use legacy `jointFriction` for a
Coulomb torque or add passive damping to drive damping. Mimic followers keep
zero drive gains. Older PhysX must reject selected passive overlays explicitly.
The Isaac Newton wrapper reapplies Channel B after its passive class writes.

`tests/isaac62_joint_params_smoke.py` runs the regression suite with native
schemas and live PhysX tensor readback. Plain-host skips of those checks are
not live acceptance; `tests/test_runtime_joint_params.py` also exercises actual
Newton CPU models and all MuJoCo-Warp adapter factory routes.

Material regression coverage: `python3 -m pytest tests/test_material_overrides.py`
checks repeated shapes, shared and AS3-instanced materials, preserved texture
connections, save/reopen behavior, value validation and conversion-failure
stamps. With the 6.2 converter installed it also converts a fresh two-visual
URDF using the actual SDK without starting Kit. Host skips of those SDK-specific
cases do not count as importer validation.
