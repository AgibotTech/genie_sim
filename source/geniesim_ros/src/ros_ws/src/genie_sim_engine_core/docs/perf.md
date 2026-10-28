# Performance profiling and tuning


## Stats output

The engine logs a performance summary every ~1 second (controlled by the C++ `stats_interval_s`
in `pybinding.cpp`).  The format is:

```
--- Physics Stats (last 500 steps) ---
  Steps: 4820  Actual Hz: 99.87 (target 100)  overruns: 3/500
  Step:     mean=7.21  min=5.83  max=14.30  std=1.12  p95=9.40  p99=12.10 ms
  Interval: mean=10.01  min=9.87  max=11.42  jitter=0.18  p95=10.31  p99=10.89 ms (target=10.00 ms)
  Solver:   mean=3.14  max=6.80  p99=5.91 ms  (sim.step — backend-agnostic)
  Render:   mean=28.40  max=45.20 ms  (47 of 500 ticks rendered)
  Publish:  mean=0.82  max=1.94  p99=1.21 ms
  Spin:     mean=2.63  max=4.11  p99=3.80 ms
  Step by tick type: render-tick mean=34.20 ms  non-render mean=6.90 ms
  Scheduler: rendered=47  skipped(period)=2  skipped(budget)=12
  Cmd:      mean=0.09  max=0.31 ms  (pop_commands + apply_commands)
  Extras:   mean=0.12  max=0.85 ms  (tick_extras + hooks)
  GPU-sync: mean=0.28  max=1.40 ms  (47 render ticks, sync_ms − dispatch_ms)
  Budget:   mean=8.03ms (80%)  worst=15.10ms (151%)  of 10.0ms target
```

### Field glossary

| Field | Source | Meaning |
|---|---|---|
| `overruns` | C++ ring | Ticks where `step_ms > dt`. Occasional is fine; persistent means the loop can't hold the target Hz. |
| `Step` | C++ ring | Wall time from `step_start` to end of render: physx + extras + render. `p99` is the tail to watch. |
| `Interval` | C++ ring | Tick-to-tick wall time. `jitter` (std) measures OS scheduling noise. |
| `Solver` | C++ ring | `sim.step(dt)` return value — time spent inside the physics solver. Backend-agnostic: PhysX in `isaac_physx`, Newton in `isaac_newton`, mjwarp/Featherstone in `newton_standalone`. |
| `Render` | C++ ring | `render_sync_ms` — Kit `simulation_app.update()` or GL viewer, **including** any explicit GPU sync. |
| `Publish` | C++ ring | `_publish_tick` — clock + joint_states + body_tf + odom combined. |
| `Spin` | C++ ring | Idle time from end of previous tick's publish to start of next tick's command apply. High = headroom; near-zero = saturated. |
| `Step by tick type` | C++ ring | Mean step time split by whether a render fired. Shows how much render steals from the physics budget. |
| `Scheduler` | C++ ring | `skipped(period)` = render skipped because it wasn't time yet; `skipped(budget)` = render skipped because insufficient budget remained. High `skipped(budget)` means renders are being shed to protect physics. |
| `Cmd` | Python | `pop_commands + apply_commands` latency. |
| `Extras` | Python | `tick_extras + post_step_hooks` latency. Near-zero for `isaac_physx`; non-trivial for `newton` (cloth/Fabric writeback). |
| `GPU-sync` | Python | `render_sync_ms − render_ms` on render ticks. Time the CPU stalled waiting for GPU after dispatch. Non-zero only when `GENIESIM_RENDER_SYNC=1`. |
| `Budget` | Python | `(step_ms + publish_ms) / dt`. Mean and worst-case tick utilisation as a percentage of the target period. |

---

## Bottleneck signatures

| Symptom | Root cause | Section |
|---|---|---|
| `Solver mean` high (>5 ms at 100 Hz) | Solver CPU/GPU saturated | [§1 Physics solver](#1-physics-solver) |
| `render-tick step mean` >> `non-render step mean` | RTX frame budget overflow | [§2 OVRTX render](#2-ovrtx-render) |
| `Spin mean` ≈ 0 + `Budget worst` >100% + frequent overruns | Loop can't hold wall-clock | [§3 Loop saturation](#3-loop-saturation) |
| `Publish mean` >1 ms | ROS serialisation overhead | [§4 Publish overhead](#4-publish-overhead) |
| `GPU-sync max` >2 ms | Explicit GPU sync blocking CPU | [§5 GPU sync](#5-gpu-sync) |
| `Interval jitter (std)` >0.5 ms with normal Solver + Spin | OS scheduling preemption | [§6 OS scheduling](#6-os-scheduling) |
| `skipped(budget)` high, `skipped(period)` low | render_safety_ms too tight | [§2 OVRTX render](#2-ovrtx-render) |

---

## §1 Physics solver

**Signal**: `Solver mean` is a large fraction of `Step mean`, or `Step` p99 tracks `Solver` p99.

The "Solver" line measures `sim.step()` time, which is whichever physics backend the engine
config chose:

- `isaac_physx` — PhysX 5 articulation solver
- `isaac_newton` — Newton solver hosted inside Kit's Isaac framework
- `newton_standalone` — Newton solver with mjwarp or Featherstone adapter

Tuning levers below are PhysX-specific (Isaac PhysX path); for Newton-backed engines see
the adapter docs in `engine/newton/adapters/`.

### Enable GPU articulation pipeline

The most impactful single change for complex robots. PhysX 5 moves the articulation solver to GPU
when the scene is non-trivial.  Verify the context is created with:

```python
# kit/stage.py — PhysicsContext construction
PhysicsContext(
    physics_dt=1.0 / physics_hz,
    use_gpu_pipeline=True,
    use_gpu_dynamics=True,
    gpu_max_rigid_contact_count=524288,
    gpu_max_rigid_patch_count=81920,
)
```

If `use_gpu_pipeline=False`, all articulation solve is on CPU.  For a 7-DOF arm at 100 Hz this is
typically 3–8 ms CPU vs <1 ms GPU.

### Reduce solver effort

Raise `render_safety_ms` so fewer render ticks fire (see §2) and give the solver more of the
budget.  Alternatively reduce contact complexity:

- Disable collision on non-contact links via the selective collision policy
  (see `pipeline.md` — default policy already does this for arm links).
- Verify `MeshCollisionAPI` approximation is `convexHull` or `sdf`, not `meshSimplification`
  (which PhysX meshes at solve time).

---

## §2 OVRTX render

**Signal**: `render-tick step mean` >> `non-render step mean`;  `skipped(budget)` high.

The render scheduler already sheds frames when budget is tight (`skipped(budget)` counter).
These levers control how aggressively it sheds and how expensive each kept frame is.

### Lower `render_hz`

```yaml
# physics_params.yaml
stepping:
  render_target_hz: 15.0   # default 30.0
```

Halves render load.  Downstream consumers (RViz, recording) typically don't need >15 Hz.

### Raise `render_safety_ms`

```yaml
# physics_params.yaml
stepping:
  render_safety_ms: 8.0    # default 2.0
```

The scheduler only fires a render when `budget_remaining > render_safety_ms`.  Raising this
sheds more renders automatically and protects the physics deadline.

### Switch to raster mode

```bash
ros2 launch genie_sim_bringup app.launch.py render_mode:=raster
```

Drops from RTX path-tracing to rasterization.  Typically 5–10× faster per frame, indistinguishable
for robot-state visualisation.  Path-tracing is only needed for photorealistic output.

### Reduce viewport resolution

Edit the `RenderProduct` resolution in `render_layer.usda` (written by the Isaac scene stage).
960×540 vs 1920×1080 reduces RTX work roughly 4×.

---

## §3 Loop saturation

**Signal**: `Spin mean` ≈ 0, `Budget worst` >100%, persistent overruns.

The Python process is saturating its CPU core.  Physics, ROS pub, and pybind overhead compete on
one thread.

### CPU core affinity

Pins the process to specific cores and prevents OS migration:

```bash
taskset -c 4-7 ros2 run genie_sim_engine_isaacsim genie_sim_engine_isaacsim_node.py ...
```

Pick isolated cores (check `lscpu` for NUMA topology; keep GPU driver cores free).

### Real-time scheduling priority

```bash
chrt -f 80 ros2 run genie_sim_engine_isaacsim genie_sim_engine_isaacsim_node.py ...
```

`SCHED_FIFO` priority 80 prevents any `SCHED_OTHER` task from preempting the physics thread.
Requires either `ulimit -r unlimited` (per-session) or a `/etc/security/limits.conf` entry:

```
*  -  rtprio  99
```

### Disable CPU frequency scaling

```bash
cpupower frequency-set -g performance
```

Prevents the CPU from downclocking mid-run (C-state transitions add 0.2–1 ms spikes).

### Reduce physics_hz

Counterintuitive: dropping from 100 Hz to 50 Hz doubles the budget per tick.  If the solver
is stable at the lower rate (contact forces allow it), this is the cleanest fix.

---

## §4 Publish overhead

**Signal**: `Publish mean` >1 ms, especially on robots with many joints.

### Publish at half rate

Add a counter in the entry point's `post_step_hooks` to publish every 2nd tick:

```python
_pub_n = [0]
def _throttle_pub(sim_time):
    _pub_n[0] += 1
    # half-rate bodies TF (joint_states always at full rate)
    ...
session.post_step_hooks.append(_throttle_pub)
```

For `/tf` body transforms specifically: external consumers rarely need >50 Hz.  `joint_states`
at 100 Hz is more important for controllers.

### Profile the publish breakdown

`note_publish_phase` tracks clock/joints/bodies/odom sub-timings.  These are not currently
surfaced in `log_stats_if_due` — add a temporary `print` in `_publish_tick` to see which
sub-step dominates before optimising.

---

## §5 GPU sync

**Signal**: `GPU-sync max` >1 ms.

This is always `render_sync_ms − render_ms`.  Non-zero means `GENIESIM_RENDER_SYNC=1` is set
somewhere in the launch environment.  This forces `wp.synchronize_device("cuda:0")` after each
render frame, stalling the CPU until the GPU finishes.  It exists only for diagnosing
render-vs-physics CUDA race conditions.

```bash
# Verify it is not set:
echo $GENIESIM_RENDER_SYNC   # should be empty
```

Never set this in production or benchmark runs.

---

## §6 OS scheduling

**Signal**: `Interval jitter (std)` >0.5 ms with normal `Solver` and healthy `Spin`.

The physics loop is being preempted by the OS between ticks.  Render or ROS system calls can
trigger context switches.

- CPU affinity (§3) is the primary fix.
- Check competing processes: `htop -d 1`, sort by CPU.  Isaac Sim's background threads
  (USD sync, telemetry, extension manager) can compete on shared cores.
- Disable kernel NMI watchdog temporarily: `echo 0 > /proc/sys/kernel/nmi_watchdog`.
- For persistent high jitter on an otherwise idle machine: check IRQ affinity.
  Network and storage IRQs default to core 0.  Move them off the physics cores:
  ```bash
  # example: move all IRQs away from cores 4-7
  for irq in /proc/irq/*/smp_affinity; do echo f > $irq 2>/dev/null; done
  ```

---

## Quick checklist for `launcher_ovrtx_physx`

```
[ ] stats collected: Solver mean, Render mean, Spin mean, Budget worst
[ ] GPU articulation pipeline enabled (use_gpu_pipeline=True in PhysicsContext)
[ ] render_mode:=raster unless photorealistic output required
[ ] render_target_hz: 15.0 unless 30 Hz viewport is needed
[ ] GENIESIM_RENDER_SYNC not set in env
[ ] CPU affinity set (taskset) on physics process
[ ] CPU frequency governor = performance
[ ] PhysX and GPU on same NUMA node (check with numactl --hardware)
```
