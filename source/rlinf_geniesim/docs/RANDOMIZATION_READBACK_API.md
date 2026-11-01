# Randomization readback

`client.randomization_readback()` fetches the active server's configuration
and applied plans over ZMQ. `env.effective_randomization()` adds the local
sensor plans known to that `GenieSimVecEnv`. Both are read-only; neither
resets nor rebuilds the world.

| Field | Meaning |
| --- | --- |
| `requested` | The raw randomization block last supplied by the client |
| `resolved` | Normalized physics, render, and sensor configuration domains |
| `resolved_available` | Whether a stage has supplied physics/sensor resolution |
| `applied.build_time` | Per-environment joint PD/background assignments from the build |
| `applied.last_reset` | Latest render reset generation and per-environment plans/readback |

`resolved.physics` and `resolved.sensor` are `null` before the process has
built a stage. An empty mapping means resolution is available and that domain
is disabled. Render resolution does not require a stage. Applied sections may
also be `null` when no build/reset has occurred or the backend cannot report
them. Do not interpret missing evidence as a disabled configuration.

`applied.last_reset.per_env` is a mapping keyed by string environment IDs. Its
entries contain the reset plan and supported readback details, including
material and camera-pose application counts. The wrapper fills each entry's
sensor section from its own state. If server-side reset readback is unavailable,
`env.effective_randomization()` exposes its local plans in `applied.sensor_only`.

```python
report = env.effective_randomization()
requested = report["requested"]
applied = report["applied"]
```

The normalized `resolved` shape is not an init payload. To modify configuration,
update the original `randomization` section and call `env.init(...)`; sending
resolved output back would lose the original precedence and disable semantics.

SHM does not have the variable-length control channel required by this API and
raises `NotImplementedError`. It must not return an empty report as if the
read had succeeded.
