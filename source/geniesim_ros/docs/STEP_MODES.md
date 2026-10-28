# Run-loop step modes

`EngineRunLoop` owns the tick body. A `LoopPacer` decides when that body runs and
when its state publishes. The implementation is
[common/pacing.py](../src/ros_ws/src/genie_sim_engine_core/genie_sim_engine_core/common/pacing.py).

## Modes and configuration

| `step_mode` | Tick release | Publication | Use |
| --- | --- | --- | --- |
| `realtime` (default) | Wall clock, scaled by `realtime_factor` | DDS cadence | Interactive simulation |
| `lockstep` | Command activity releases a K-tick burst | Last tick of the burst | Client-driven requests |
| `freerun` | No wall-clock pacing | DDS cadence | Batch execution |

```yaml
genie_sim_engine:
  ros__parameters:
    step_mode: lockstep
    step_lockstep_ticks: 4
    step_idle_timeout: 0.0
```

The launcher owns these keys. `step_idle_timeout` pauses realtime stepping after
inactivity; zero disables it, and it arms only after the first command.
`step_lockstep_ticks` selects burst length; the pacer's default applies when zero.
The shared client requires an explicit positive K.

`EngineNodeParams.from_dict` normalizes mode strings and nonnegative numeric
values. An empty mode selects realtime; an unknown mode raises with the valid
choices. Inactive knobs are logged. `realtime_factor` applies only to realtime.
No environment variable configures pacing.

Lockstep delegates to realtime during startup until command activity is usable.
A backend without an activity signal warns and remains wall-paced; a signal that
becomes available later can engage it. Check the startup description before
interpreting a run as lockstep.

## Pacer interface

```python
decision = pacer.before_tick(now)
published = pacer.after_step(ok)
```

| Decision | Step physics | Publish without a step | Reset spin baseline |
| --- | --- | --- | --- |
| `RUN` | Yes | No | No |
| `IDLE` | No | No | No |
| `PAUSED` | No | No | Yes |
| `FLUSH` | No | Yes | Yes |

`next_deadline(now)` supplies the render budget. Unpaced modes return `now`, not
infinity. `notify_resumed(now)` re-baselines pacing after lifecycle resume/reset;
`note_overrun` handles late ticks. Mode registration lives in `STEP_MODES`.
A new pacing policy belongs in a pacer class rather than a branch in the loop.

## Lifecycle and timing invariants

- A bounded `IDLE` wait counts toward the next `spin_ms`. An unbounded `PAUSED`
  wait does not. Lifecycle pauses also reset the loop's spin baseline.
- Resume/reset notify the pacer so wall-clock debt and stale idle time do not
  trigger catch-up bursts or an immediate idle pause.
- Lockstep resume preserves the remaining burst and consumed-command identity;
  pausing mid-burst does not shorten or replay the request.
- Between bursts, a pending reset triggers `FLUSH`. The loop calls
  `apply_commands` with empty command dictionaries, drains reset state, and
  publishes without integrating physics. Reapplying latched commands here would
  overwrite the reset targets. A zero 4WS timestamp brakes a chassis once.
- Realtime drains resets through its ordinary tick and does not emit `FLUSH`.

## Completion limits

A pacer burst is not a command-ID acknowledgement. The
[shared client's `arrival_v1` contract](../src/ros_ws/src/genie_sim_client/DESIGN.md#4-shared-step-contract-arrival_v1)
waits for new joint and body messages. Startup free-running, separate command
latching and asynchronous GPU mirrors prevent it from proving exact command
ownership or committed K-tick completion. Publisher stamps are wall clock, not
simulation time.

The pacing suite covers cadence, startup, activity, reset flush, resume, timing
accounting and parameter forwarding with controlled clocks. Unit tests do not
establish live solver overhead, multi-world physics or exact committed stepping.
Those remain separate runtime acceptance requirements.

## Verification

From the repository root:

```bash
python3 -m pytest -q source/geniesim_ros/src/ros_ws/src/genie_sim_engine_core/tests/pacing
```

See [performance metrics](../src/ros_ws/src/genie_sim_engine_core/docs/perf.md),
[configuration ownership](../src/ros_ws/src/genie_sim_engine_core/docs/physics_knobs.md)
and the [core maintainer contract](../src/ros_ws/src/genie_sim_engine_core/AGENTS.md).
