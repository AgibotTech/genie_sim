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
