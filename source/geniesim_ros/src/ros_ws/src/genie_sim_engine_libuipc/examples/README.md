# Libuipc examples and native acceptance

These tools belong to the libuipc backend. Container creation, dependency
installation and workspace builds use the existing Docker/ROS commands.
Ordinary ROS builds include the backend; these tools install no dependencies.

## Offline particle capture

`dump_fold_cloth_particles.py` captures one declared cloth/tet object's completed
particle range. It needs an installed ROS overlay, an assembled libuipc manifest
and a compatible native runtime. From the repository root, after the normal ROS
build and sourcing its overlay:

```bash
python3 source/geniesim_ros/src/ros_ws/src/genie_sim_engine_libuipc/examples/dump_fold_cloth_particles.py \
  --manifest /path/to/assembled/manifest.json \
  --out /path/to/new-capture --ticks 240
```

After installation, the same entry point is available as:

```bash
ros2 run genie_sim_engine_libuipc dump_fold_cloth_particles.py \
  --manifest /path/to/assembled/manifest.json \
  --out /path/to/new-capture --ticks 240
```

Engine settings come from installed `launcher_libuipc.yaml`. Use `--launcher`
with a launcher YAML path to select a different physics profile, rate or pacing
mode. An independent package installation without bringup must pass that YAML
explicitly. To capture in freerun, set `step_mode: freerun` in that YAML; the example
does not silently override launcher pacing or duplicate physics defaults.
The referenced scene YAML still owns CPU-worker/viewer settings. A headless
engine does not override a scene's optional CPU viewer configuration.

Output must be a new directory. The example writes:

```text
new-capture/
  frames.json
  deformable/<object_name>/000000.npz
  deformable/<object_name>/000001.npz
  ...
```

Each NPZ contains world-space `points`. The metadata records the object path,
frame pattern, particle count, actual physics rate and completed simulation
times. It is written only after the requested number of frames completes.
An interrupted/failed run can leave partial NPZs without a completed manifest;
choose a new output directory for the next run. Existing data is never deleted
or overwritten. The original scene callback and owned engine resources are
restored/closed on exit, including failures.

This example uses private scene hooks and supports exactly one deformable object.
It is not an episode recorder, MP4 encoder or benchmark integration. CPU MuJoCo
remains asynchronous; selecting freerun does not establish deterministic coupled
CPU/IPC stepping. For live images, use the normal `launcher_libuipc_ovrtx` or
`launcher_libuipc_isaacsim` render path rather than a separate demo renderer.

## Explicit native GPU smoke test

The native probe is [test/native_smoke.py](../test/native_smoke.py). Run it directly
inside the compatible CPython 3.12/native/GPU environment:

```bash
python3 source/geniesim_ros/src/ros_ws/src/genie_sim_engine_libuipc/test/native_smoke.py
```

After installation it is also available through
`ros2 run genie_sim_engine_libuipc native_smoke.py`. It requires no ROS runtime
when executed directly from source and can run independently of colcon discovery.
Both tools support `--help` without starting an engine or loading native IPC.

The probe runs cloth, rigid ABD and mixed cases, checking native
`World.init/advance/retrieve/recover(0)`, finite state and gravity motion. It is
not named `test_*.py`, so normal pytest/CTest collection does not execute it.
Full task contacts, reset/renderer synchronization and hardware compatibility
remain separate gates.
See [DESIGN.md](../DESIGN.md) and the
[native artifact notes](../../../../../../../3rdparty/pyuipc-README.md).
