# Runtime verification

The backend integration targets system Python 3.12, ROS Jazzy and Isaac Sim 6.2.
The standalone Newton runtime is Newton 1.6.1, MuJoCo 3.12.0 and Warp 1.18.0.
Shared core code is also checked with Python 3.10 / ROS Humble.

| Check | Result |
| --- | --- |
| Jazzy build/install | Common, core, Isaac, Newton, robot model, render and bringup packages pass |
| Backend, render and bringup Python suites | 440 passed, 16 skipped |
| Newton tool regressions | 396 passed, 4 skipped |
| Humble core suites | 347 passed |
| Jazzy core CTest suites | All 8 targets pass |
| Renderer CTest suites | All 3 targets pass |
| Native Isaac 6.2 sysid checks | 24 passed, including live PhysX tensor readback |

Skipped checks are not runtime acceptance. Full scene motion, camera attachment,
long-run stability, cache behavior and autonomous task success require the
[bringup checklist](../src/ros_ws/src/genie_sim_bringup/README.md#scene-testing).
The Isaac Newton rigid wrapper remains experimental. Bundled Python and older
Isaac runtimes are outside this system-Python qualification.

## Live G2 smoke

Fresh `scene_flat_g2_sp` assembly and ROS execution were checked with the migrated
PhysX and Newton MJ-Warp launchers, headless and without RViz.

| Check | Isaac PhysX + OVRtx | Newton MJ-Warp |
| --- | --- | --- |
| Fresh URDF/USD/scene assembly | Pass | Pass |
| Joint feedback and robot TF | 42 joints | 42 joints |
| Requested head motion | 0.12 rad | 0.12 rad |
| Measured head motion | 0.120009 rad | 0.120000 rad |
| Advancing simulation clock | Pass | Pass |
| RGB and depth | 640×400 rgba8 and 32FC1 | No renderer selected |

The PhysX probe took about 266 seconds including cold OVRtx first-frame warmup;
physics and ROS feedback continued during that wait. Topic discovery alone did
not count as image delivery. Newton's headless comparison does not validate GL
or inline OVRtx visuals. The lifecycle listener treats context shutdown as normal
termination and still reports unexpected executor failures.

## Image qualification records

These measurements apply to the identified images and source snapshots. They
are recorded evidence, not a rerun of the current checkout or qualification of
another image tag. Use the [bringup matrix](../src/ros_ws/src/genie_sim_bringup/README.md#scene-testing)
for scenario pass criteria and the [workspace roadmap](ROADMAP.md) for open gates.
`geniesim docker` and `geniesim docker6.2` select the dedicated 6.2 image;
explicit `geniesim docker5.1` selects the combo image.

### geniesim3 system Python, 2026-10-25

The measured local clean rebuild has image configuration ID
`sha256:31a683881401097ca8562db74ab97c4fe9a86671d245ed81502b831ad7af7a49`
and size **51,576,579,322 bytes (51.6 GB)**. Its NVIDIA 5.1 base manifest is
`sha256:f3563cb2ba0c18af0b2fb321360dcb73a917b899f879e3213623d6bee484fa54`.
The system SDK payload was copied once from a cached installation without
inheriting donor image layers. This does not establish fresh upstream-download
reproducibility. OS/ROS dependencies used source
`e4759a02008f412ce6451c865dbbcc1c2afffa2b` and the `isaac` dependency profile.
Bundled `/isaac-sim/kit/python` was preserved and is outside this qualification.

| Check | Recorded result |
| --- | --- |
| Physics/render versions | Python 3.12.3, Isaac 6.2.0.0, Newton 1.6.1, MuJoCo/MuJoCo-Warp 3.12.0, OVRtx 0.4.1.364340 |
| Imported system packages | NumPy 2.3.1, absl 2.4.0, fsspec 2026.2.0, pydantic-core 2.33.2; typing-extensions 4.12.2 files verified |
| Dependencies / CPU | `pip check` and Newton guards pass; 139 SDK/colcon constraints, zero repairs; 93 CPU tests pass |
| Fresh R01 assembly | `scene_flat_g2_sp` regenerates on PhysX and Newton; crsB/Swiftpicker sysid selections apply after class defaults |
| PhysX + OVRtx | 42 finite joint positions, TF/clock, 640×400 RGB/depth and exact YAML CameraInfo K; RGB standard deviation 110.42; all 256000 depth pixels finite positive |
| PhysX head command | +0.1 rad target; error 0.017153 rad, within the 0.02 rad probe tolerance |
| Newton MJ-Warp head command | Headless, no visualizer; 42 finite joint positions and TF/clock; +0.1 rad target, error 0.016708 rad |
| Direct SDK camera | 48×64 RGB standard deviation 91.90; cube-centre depth 4.0 m; imported NumPy explicitly checked |

Imported package versions and file records matched the installed wheels, with
no duplicate distribution metadata in system pip's installation directory.
Metadata alone is insufficient for an import audit. Both ROS runs used a freshly
built overlay and private scene caches. These probes do not qualify full robot
sweeps, camera attachment under chassis motion, cache/reset criteria, Newton
GL/inline OVRtx or other scenes on this image.

### Dedicated Isaac 6.2, 2026-10-22

The measured image used NVIDIA base `6.2.0.1` and Isaac wheel `6.2.0.0`.
These component measurements do not transfer automatically to another image or
asset revision.

| Check | Recorded coverage |
| --- | --- |
| Tiled and fused cameras | RGB for two environments with heights 48/64, correct env-major order and zero padding |
| Depth | 48×64 RGB/depth; cube centre 4.0 m |
| G2 omnipicker, PhysX | RGB/depth/CameraInfo, 46 finite joints, TF/scene transforms/clock; 100 Hz physics |
| Experimental Isaac Newton wrapper | Camera/ROS graph passes; paced loop 60.7–61.1 Hz against a 100 Hz target |
| Fresh G2 omnipicker / Swiftpicker conversion | Assembly passes; intended colliders, material overrides and face-subset bindings apply |
| Kit episode reset | PhysX and wrapper pass two resets restoring root/joints/targets and 14 free-body poses, clearing velocity/effort and advancing epochs; max readback errors 0 / 1.1e-7 |
| Calibration | Off-centre fx=180/fy=110 cube projection centre 139.5/99.5 versus 140/100, size 90×56 versus 90×55 pixels; ROS retains exact YAML K |

The wrapper's direct warm solver calls measured 96.7–97.3 Hz. Those calls and the
paced ROS/render loop are different workloads on a shared GPU; neither establishes
isolated 100 Hz acceptance. Bundled 5.1 and Isaac 4.5 runtime acceptance are not
established by these system-Python results.

## Reproduce

Build each interpreter's ROS workspace separately; do not reuse a Python 3.12
CMake cache for Python 3.10. Source the matching ROS underlay and workspace
`devel/setup.bash`, then run shared checks from the repo root:

```bash
python3 -m pytest -q \
  source/geniesim_ros/src/ros_ws/src/genie_sim_engine_isaacsim/tests \
  source/geniesim_ros/src/ros_ws/src/genie_sim_engine_core/tests/lifecycle \
  source/geniesim_ros/src/ros_ws/src/genie_sim_bringup/test/test_assemble_pipeline.py \
  source/geniesim_ros/src/ros_ws/src/genie_sim_render/test/test_isaacsim_transport.py
```

Use the [scene test commands](../src/ros_ws/src/genie_sim_bringup/README.md#one-command-for-every-matrix-cell)
for fresh and cached launches. Retain image/source/asset identity, launcher and
scene configuration, complete AS3 output, logs and camera samples with each run.
Shared CPU checks and SDK-only skips do not establish live GPU acceptance.
