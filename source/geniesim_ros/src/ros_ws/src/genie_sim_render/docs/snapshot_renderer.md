# Isaac6 frozen snapshot renderer

`isaacsim_snapshot_render.py` renders Newton viewer geometry in a separate
headless Isaac Sim process. Newton remains the sole physics owner. The
worker creates no `World` or solver. Scene geometry arrives as Newton mesh
snapshots; a robot can instead be referenced from its authored visual USD to
retain MDL materials, texture maps, UVs and material/subset bindings. The parent
then removes that robot's Newton instance rows and supplies its body poses,
so the authored reference replaces its snapshot geometry without duplication.
Referenced physics and native sensor payloads are disabled for rendering.

Build the Python entry points and docs from a sourced ROS workspace:

```bash
colcon build --packages-select genie_sim_render --cmake-args -DGENIE_SIM_RENDER_BUILD_OVRTX=OFF
source install/setup.bash
```

`GENIE_SIM_RENDER_BUILD_OVRTX=OFF` bypasses OVRTX SDK discovery and all C++
node/plugin/gtest targets. Snapshot pytest remains registered when
`BUILD_TESTING=ON`. The default is `ON` so ordinary builds retain the
existing OVRTX targets. This controls package construction only; it does
not change Newton physics, scene data or a renderer's runtime behavior.

Launch from the parent's current Isaac6-capable interpreter:

```python
from multiprocessing import Pipe
import subprocess
import sys

parent, child = Pipe(duplex=True)
process = subprocess.Popen(
    [sys.executable, worker_path, "--ipc-fd", str(child.fileno())],
    pass_fds=(child.fileno(),),
    # Route stdout/stderr to a log owned by the caller.
)
child.close()
ready = parent.recv()  # {"ready": True, "renderer": "isaacsim"}
```

The FD is transport plumbing, not a physics/configuration knob. All caller
selection and geometry/camera values come from the evaluation launcher's
configuration. Use system Python 3.12 with the bundled Isaac Sim 6.2 SDK
in the dedicated container. The 5.1 + 6.2 combo is deprecated (E.O.L.).
The worker uses the shared `prepare_launch_config` compatibility helper before
Kit starts. On SDK 6.2 this disables multi-tick scheduling, matching the other
Isaac render entry points; explicit synchronous Replicator captures must not
inherit the unsupported scheduling mode after stage replacement.

Application initialization and the first render can take time compiling
shaders; the parent must use bounded readiness/render timeouts and own
process cleanup when a native startup failure prevents a reply.

The duplex Pipe is private and trusted. `Connection.recv()` uses Python
object deserialization; never expose it to a network or untrusted sender.
Kit stdout/stderr contain normal logs and are not a message channel.

## Messages

Send a dictionary with `op: "render"`:

| Field | Meaning |
|---|---|
| `sequence` | Integer request credential, returned unchanged |
| `scene.meshes` | `{name: mesh_patch}`; first occurrence supplies `points` and `indices`; later patches merge with cached data |
| `scene.instances` | Complete `{batch_name: instance_batch}` for this frame; absent batches are removed |
| `scene.authored_robot` | Optional `{asset_path: str, root_prim?: str}` visual USD descriptor; repeated each capture with the same identity |
| `scene.body_poses` | With an authored robot, complete `{body_name: float32[7]}` world poses as `xyz + quaternion xyzw`, supplied afresh every capture |
| `scene.authored_scene` | Optional fixed `{asset_path, root_prim, hide_prims, use_asset_lights, body_visuals}` background/selected-body visual descriptor; see below |
| `scene.visual_body_poses` | Complete fresh `{body_name: xyz+xyzw}` Newton world poses for the authored scene's `body_visuals` drivers |
| `cameras` | Nonempty list of distinct `{name, eye, target, up, fov_degrees}` in world metres |
| `width`, `height` | Positive integer render resolution, shared by this request's cameras |

Mesh patches accept finite `points: Nx3`, flat integer triangle `indices`,
optional vertex `normals: Nx3`, vertex `uvs: Nx2`, `texture`, `color: RGB`,
`roughness`, `metallic`, `hidden` and `backface_culling`. Arrays may be NumPy
arrays. UVs become a vertex-interpolated USD `st` primvar. `texture` accepts an
existing local image path or a `uint8 H×W×3/4` array; arrays are written to
content-addressed PNGs in an owned temporary directory. A `UsdUVTexture` and
`UsdPrimvarReader_float2` network reads `st` and supplies diffuse RGB, with the
provided RGB tint. A texture without matching UVs is an error. Hidden mesh
prototypes remain hidden at their source location,
while their instances can be visible. Unreferenced meshes with
`hidden=False`, such as chef food meshes, render directly.

Instance batches require `mesh`, `xforms: Nx7` as `xyz + quaternion xyzw`,
`scales: Nx3`; optional `colors: Nx3`, `materials: Nx4` and `hidden`.
Newton's material vector stores roughness, metallic, checker and texture
flags; this worker uses roughness/metallic and the texture-enable flag for
generic textured instances. Procedural checker rendering is not implemented.
Quaternions
are normalized; zero quaternions and invalid triangle indices are errors.

## Authored robot materials

The parent resolves the visual asset through the assembled manifest. For Wuji,
the scene selects `robot/G2_crsB_wuji/robot.usda` using
`robot_source.prebaked.visual_only: true`; Newton physics still consumes the
staged URDF asset. An older `robot_visual.usda` that merely references the
converted physics `robot.usda` does not restore original hand glove textures;
reassemble the scene with the explicit visual source.

The worker references the descriptor's default prim, or explicit `root_prim`,
under `/World/AuthoredRobot` and retains the original Looks/material networks.
Relative texture paths stay anchored to their original source layers. It
disables physics/sensor variants, joints, native cameras and collision display
geometry. Only link Xforms are driven by the incoming body poses, preserving
the authored local geometry/scales and material graph. Asset identity cannot
change during one worker session; missing/ambiguous visual drivers are errors.

The source Wuji asset has a flat link layout. Its 88 manifest body names match
the authored `/genie/<body_name>` prims; the parent maps those exact names to
Newton body indices and removes only their corresponding snapshot instance
rows. Shared prototype geometry and nonrobot rows in the same batch remain
available for the wok/stove/food. The `body_poses` mapping is sent on every
capture, even when mesh deltas are empty. Full authored MDL and face-varying
UV data stay in the referenced USD, rather than being converted into the generic
mesh adapter's vertex UV/RGB texture representation.

Chef's wok, handle, stove, support ring, noodles, ham and vegetables are
programmatically constructed meshes/primitives with RGB colors. The plugin
defines no image textures or authored roughness/metallic values for these
objects. Their original colors and geometry are preserved; no original wood,
food or metal image maps exist to recover. Any newly configured PBR appearance
for those objects is a visual adaptation, not a restored source texture.

## Authored background and dynamic object visuals

The parent can select a real background USD through scene YAML
`newton.authored_visuals`. It resolves the asset path, filters the replaced
Newton visual rows, and sends this render-only descriptor in each snapshot:

```python
"authored_scene": {
    "asset_path": "/absolute/path/to/assets/background/home/fried_noodles/background.usda",
    "root_prim": "/World",
    "hide_prims": ["/World/objects"],
    "use_asset_lights": True,
    "body_visuals": [{
        "body": "chef_wok",
        "source_prim": "/World/objects/benchmark_pot_013",
        "reference_pose": [0.9949, 0, 1.09, 0, 0, 0, 1],
    }],
},
"visual_body_poses": {"chef_wok": current_newton_xyz_xyzw},
```

The static root is referenced at `/World/AuthoredScene`, retaining its
materials, textures, face-varying UVs, authored scales and light hierarchy.
`root_prim`, `hide_prims` and `source_prim` use absolute paths in the source
USD. Hidden subtrees are deactivated only in the worker layer. The supplied
background already contains its table, stove, pot and 42 rigid food instances;
deactivating `/World/objects` removes that static pot/food copy. The existing
46 deformable Newton food objects and their scoring remain unchanged.
Referencing `/World` also excludes the source stage's sibling `/Render`
products/settings, so the worker keeps ownership of its sensor and observer
cameras and their requested resolution.

Each selected body has an independent parent under `/World/AuthoredVisualBodies`
and a full source-subtree reference child. The child retains the source's
**world** transform, including ancestor offsets and its internal visual
transforms/material graph. The parent's matrix is
`inverse(reference_pose) * current_pose` in Gf's row-vector convention.
`reference_pose` identifies the Newton rigid-body frame at authored rest, not
the source mesh origin. The supplied pot's biased origin and internal Rx(90°)
are therefore retained rather than placing that origin at the wok rim center.
All poses use metres and quaternion **xyzw**, normalized before conversion.

Missing assets, prims or fresh body poses, invalid quaternions, duplicate body
drivers, or changes to the descriptor within one worker session are errors.
The descriptor may be omitted after its first successful use; the complete
`visual_body_poses` mapping is still required on every capture. The worker
never keeps a previously rendered pose in place of a missing driver.

The worker disables referenced physics APIs, PhysicsScene, joints, native
cameras and collision-display prims. It never imports the background's rigid
physics into Newton or runs another solver. When `use_asset_lights=True`, its
default dome/sun are deactivated and the authored lights remain active.
When false, authored lights are deactivated and worker lights stay active.
An authored scene always suppresses the fallback ground. One-time stderr
audit lines identify the source, hidden prims, body drivers/reference frames,
material bindings and active light count. They are logs, not protocol replies.

The generic scene uses Z up and metres, with a dome and distant light. A fallback
visual ground is added only while no visible mesh/batch name contains
`ground`, `floor` or `plane`, and no authored scene is configured. It has no collision or solver behavior.
`fov_degrees` is vertical FOV, matching Newton's Camera projection; camera
up preserves wrist roll. Each camera has its own RenderProduct. Cameras
removed from the next request and changed resolutions release their old
annotator/product resources.

After applying the entire snapshot, the worker calls Isaac6 Replicator
`orchestrator.step(rt_subframes=2, delta_time=0.0, pause_timeline=True,
wait_for_render=True)`. It then reads each RGB annotator on CPU and copies
the RGB channels. The reply is:

```python
{"sequence": sequence, "renderer": "isaacsim",
 "images": {camera_name: uint8_array_of_shape_H_W_3}}
```

No image resize, fallback pixels or duplicated camera images are used.
Invalid dtype/resolution is an error. Each returned array comes from that
camera's render product. Rendering does not advance timeline time.

The caller must associate the sequence with its completed Newton snapshot,
publish the three sensors with one capture stamp after this reply, and only
then acknowledge the observation barrier. The worker does not infer a
snapshot from asynchronous TF caches. For video, send the observer view
only on the caller's simulation-clock samples (for example every two ticks
at 60 Hz for 30 fps). Sensor captures during inference must not be appended
as video frames. The worker has no wall-clock capture loop.

Send `{"op": "close"}` and wait for process exit to release annotators,
render products, generated temporary texture files and SimulationApp. Parent EOF also exits. A Python error
returns `{"error": "Type: detail", "sequence": ..., "renderer":
"isaacsim"}` and exits with status 1 after resource cleanup; stdout/stderr
retain the traceback. A native crash cannot send a Python error, so the
parent must also inspect child exit/timeout and close its Pipe endpoint.

## CPU checks

From the repository root, with NumPy, pytest and `usd-core` available:

```bash
python3 -m pytest -q source/geniesim_ros/src/ros_ws/src/genie_sim_render/test/test_isaac_snapshot.py
```

These checks use real in-memory USD to verify reference visibility,
mesh deltas, transforms, materials, camera axes and removed instances.
Authored-background cases additionally verify hidden static copies, local
texture resolution, light selection, ancestor/mesh offsets, rotation and
scale preservation, dynamic body deltas and fresh-driver rejection. Sensor
and observer requests use the same scene and render barrier in those checks.
Protocol/render-product ownership checks use test doubles and never start
SimulationApp, GPU services or a model. They establish no claim that a
particular simulator task has been validated; actual Isaac RGB must be
verified separately in the intended runtime. The 2026-10-20 GPU render/control
record used the earlier basic RGB materials. It does not verify the new
authored-robot material/texture or generic mesh UV/texture rendering path.
