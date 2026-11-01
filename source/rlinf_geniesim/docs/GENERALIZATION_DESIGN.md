# Randomization

Randomization is configured under the top-level `randomization` task-init
section. Physics, render, and sensor domains remain separate because they
execute in different processes. `RandomizationSpec` resolves configuration;
`Sampler` derives deterministic plans from a seed and independent substreams.

## Sources and precedence

Explicit task-init keys override `config/task_generalization.yaml`, which
overrides the benchmark eval task's generalization values. Built-in defaults
supply missing values. `use_task_generalization: false` bypasses the per-task
file; `use_benchmark_generalization: false` bypasses benchmark defaults.
A per-task `null` disables that dimension.

The per-task file currently supplies physics defaults. Render settings come
from the init payload. Sensor settings run in `GenieSimVecEnv`; callers must
supply a resolved sensor domain to that wrapper, as its sensor configuration
is not inferred from the remote render state.

## Seed contract

A reset seed of `-1` restores authored values without seeded perturbations.
Nonnegative seeds select deterministic plans. Sampler domains such as render,
sensor, background, and joint PD incorporate environment identity, so equal
seeds need not produce equal values in different environments. The scalar
`reset(seed=...)` also derives distinct per-environment seeds.
The legacy `randomization.enabled` key does not control this contract.

Object positions, arm joints, and robot base perturbations use the physics
layer. Instance-specific robot base poses take precedence over base offsets.
Joint PD and background assignments are built with the scene. Lights,
materials, and camera poses are applied on reset by supported render bridges;
sensor noise, occlusion, and dropped frames are applied on the client.

Example additional init sections, usable with `init_task.py --config FILE`:

```yaml
randomization:
  object_xy_radius: 0.03
  object_constraints: true
  render:
    lights:
      enable: true
      intensity_scale: [0.5, 1.5]
      distant_rotate_deg: 45.0
    material:
      enable: true
      mode: ml_prefix
seed: 0
```

Available dimensions and their validation are defined in `randomize/spec.py`.
Material targeting supports `ml_prefix`, `keyword`, or `both`; exclude robot
prims when configuring broader keyword matches. Camera thresholds use metres
for translation and degrees for rotation. Background randomization requires
compatible decorative rooms and refuses tasks that rely on physical background
supports; it also requires the external background assets.

When a rest-state bake provides support geometry, object offset constraints
keep stacks together, prevent new overlaps, and retain objects on their
supports. See [instance body poses](INSTANCE_BODY_POSES.md).

Use [readback](RANDOMIZATION_READBACK_API.md) to inspect the requested,
resolved, and applied values. A requested dimension is not proof that a
particular backend applied it.
