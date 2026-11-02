# genie_sim_engine_libuipc

Independent libuipc physics backend. The native World solves rigid-body, cloth
and tet contact; Newton's SDK supplies model construction, state arrays and FK.
The package imports neither `genie_sim_engine_newton` nor
`genie_sim_engine_isaacsim`, including during asset assembly. Benchmark and RLinf
are not backend dependencies; the core `PhysicsEngine` ABI is unchanged.

## Installation and build

Use the shared system-Python 3.12/Jazzy SDK environment. Newton 1.6.1,
MuJoCo/MuJoCo-Warp 3.12.0 and Warp follow that SDK's constraints, with no separate
standalone pin set. Newton's `SolverMuJoCo` export path requires MuJoCo-Warp even
with CPU MuJoCo. The ROS distribution installs CoACD (`coacd>=1.0.7`); scenes
select decomposition explicitly with `cpu_collision.method: coacd`.

Install the native wheel through `geniesim bootstrap`, or retry just that optional
step with `geniesim bootstrap --libuipc-only`. The [native artifact guide](../../../../../../3rdparty/pyuipc-README.md)
owns Python/CUDA/GPU requirements, selection, SDK protection and licensing.
Native loading alone does not qualify GPU solves or complete backend operation.
Package discovery does not load `uipc`.

Ordinary `geniesim ros build dev` builds this backend. Follow the
[build setup](../../../../../geniesim_benchmark/README.md#deformable-evaluation)
to load the matching ROS overlay. Scene and launcher selection still determines
which backend runs.

Cold URDF conversion additionally invokes the external Isaac importer in a
separate process. Local prebaked-USD composition needs neither Kit nor a sibling
backend. Keep bare-process USD and Kit USD providers separate; do not install
another `usd-core` over the runtime's `usd-exchange` provider.

## Launch and evaluation

```bash
ros2 launch genie_sim_bringup app.launch.py \
  scene:=scene_flat_g2_wuji_ipc_fold_cloth \
  launcher_config:=launcher_libuipc headless:=true
```

`launcher_libuipc_ovrtx` and `launcher_libuipc_isaacsim` add separate rendering;
`headless:=false` selects the Newton SDK GL viewer. The four original IPC scenes
retain their `libuipc:` block.

The [three-task benchmark](../../../../../geniesim_benchmark/README.md#deformable-evaluation)
uses separate `scene_flat_g2_wuji_ipc_<task>_eval` configs and
`launcher_libuipc_wuji_eval` for `cloth_hanger`, `fold_cloth` and `drawstring_bag`.
It adds a CPU barrier, committed actions, full physical archives and coherent
three-camera observations. `fold_paper` is outside that benchmark. Physics
continues during inference/render waits; configured rates describe simulation
time, not achieved wall-clock throughput. Task asset identity and clean archives
do not establish source/target physical equivalence or model task success.

[DESIGN.md](DESIGN.md) owns CPU coupling, SDK/USD, state, failure and archive
contracts. [Examples](examples/README.md) cover particle capture and the explicit
`test/native_smoke.py` GPU probe.
[AGENTS.md](AGENTS.md) provides contributor guidance.
