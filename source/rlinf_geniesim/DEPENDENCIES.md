# Dependencies and distribution boundaries

The [package license](LICENSE) applies to this package's source files,
subject to file-specific notices. Dependencies keep their own licenses.
An import in source and a binary installed in a published container have
different distribution contents; review the contents of the artifact being
released. Mozilla explains MPL's file-level scope in its
[official FAQ](https://www.mozilla.org/en-US/MPL/2.0/FAQ/).

## Standalone client distribution

[`rlinf-geniesim-client`](CLIENT_README.md) installs the transport clients,
vector environment, and randomization helpers under the `rlinf_geniesim`
import namespace. Its source distribution and wheel include this package's
MPL-2.0 `LICENSE`. Server transports, launch and debug scripts, physics and
rendering backends, benchmark tasks, and assets are excluded.

The base installation depends on NumPy, PyZMQ, and Pillow. The `shm` extra adds
`posix_ipc`; the `sensor` extra adds SciPy. PyTorch is supplied by RLinf rather
than installed by the client package. The declared dependencies are in
[pyproject.toml](pyproject.toml). They retain their own licenses, including any
native libraries bundled in dependency wheels.

The broader source/server dependency profiles below do not describe the
contents of the standalone client. `requirements-client.txt` remains a shared
dependency list for source-based scripts.

## Source and runtime dependencies

| Component | Use | License evidence |
| --- | --- | --- |
| posix_ipc | SHM semaphores | BSD-3-Clause in the [upstream repository](https://github.com/osvenskan/posix_ipc). |
| PyZMQ | Client/server transport | BSD-3-Clause for the binding; see [upstream LICENSE](https://github.com/zeromq/pyzmq/blob/main/LICENSE.md). Binary wheels may also bundle libzmq and libsodium. |
| CoACD | Convex decomposition in the simulation runtime | MIT in the [upstream LICENSE](https://github.com/SarahWeiii/CoACD/blob/main/LICENSE). |
| OVRTX | RTX rendering; installed by the server Dockerfile | The `0.4.1.364340` wheel from [NVIDIA's package index](https://pypi.nvidia.com/ovrtx/) declares `LicenseRef-NvidiaProprietary`. It ships the NVIDIA Software License Agreement, Product Specific Terms for NVIDIA AI Products, and third-party notices. |
| [Newton](https://github.com/newton-physics/newton), [MuJoCo](https://github.com/google-deepmind/mujoco), [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp), [Warp](https://github.com/NVIDIA/warp) | Physics backends | Upstream source uses Apache-2.0; binary packages also carry component-specific licenses. The shared image retains the Isaac SDK versions under its dependency constraints; native libuipc installation runs through `geniesim bootstrap`. Inventory the selected runtime and retain its licenses and notices. |
| pynput | Installed in the optional teleop image | LGPLv3 in the [upstream project](https://github.com/moses-palmer/pynput); retain `COPYING.LGPL` and notices, and provide required source access for the version distributed. |
| NumPy, SciPy, PyYAML, Pillow, optional OpenCV/PyTorch, pxr, ROS packages | Numerical, image, scene and middleware interfaces | Installed by the selected environment; retain the notices supplied with those versions. |
| GenieSim benchmark, teleop and engine packages | Tasks, success checkers, Pico input, IK helpers and simulation | Sibling packages with their own license files and runtime dependencies. |
| GenieSim assets | Scenes, robots and other task assets | Supplied separately; this package does not grant redistribution rights to assets. |

The Dockerfiles do not define a complete, version-locked dependency inventory.
The server's added Python dependencies are pinned in
[`post-train-runtime.txt`](../../docker/post-train-runtime.txt), with OVRTX pinned
in its Dockerfile. The optional teleop layer installs `pynput` without a pin.
The documented [`Dockerfile`](../../docker/Dockerfile) build creates the default
`registry.agibot.com/genie-sim/geniesim4:latest` base locally with NVIDIA's Isaac Sim
`6.2.0.1` image and the `6.2.0.0` system-Python package. The NGC image tag and
Python package version are paired releases with different version strings. The
post-training layer uses this image as its default `BASE_IMAGE`.
For a container release, record the base-image digest and installed packages,
and retain their applicable license texts and notices. Consult the terms
shipped with that exact simulator distribution; source licensing alone does
not determine the terms of Kit components or assets. See NVIDIA's
[Isaac Sim licensing documentation](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/common/licenses-isaac-sim.html).
For OVRTX, review the agreement shipped with the pinned wheel and retain its
`LICENSE`, `THIRD-PARTY-NOTICES.txt`, and component license files in any
distributed image. The package's MPL license does not cover OVRTX binaries.

Publishing the whole repository also distributes its tracked `3rdparty/`
wheels and sibling binary trees. Mounting them at runtime or omitting them
from a Docker build does not exclude them from a repository archive.

## Asset package

The local `geniesim_assets` 4.0.0 snapshot inspected for this release declares
`CC-BY-NC-SA-4.0` in its package metadata and ships the corresponding
`LICENSE`. Its attribution, noncommercial and share-alike conditions are
separate from this wrapper's MPL license. Verify the license and file-specific
notices of the exact asset distribution used; this finding does not establish
the terms of other asset sets or versions.

## Optional IK solver binary

[`docker/Dockerfile.post-train-teleop`](../../docker/Dockerfile.post-train-teleop) installs
`3rdparty/ik_solver-0.4.3-cp312-cp312-linux_x86_64.whl`. The wheel contains
native code and bundled libraries, including Assimp, Boost, Coal, Draco,
NLopt, OctoMap, Pinocchio, urdfdom and yaml-cpp.

The inspected cp312 wheel (SHA-256
`f339166959d9ef466d8da18106cf0ea6cca8ab8b86f099ecb5cc9ea38b877dcc`)
contains no license metadata or LICENSE/NOTICE files.
Its owner's distribution terms, build/source revisions and complete bundled
component notices must be established before publishing this wheel or an
image that contains it. The package's MPL notice does not resolve those
missing records. The repository's existing `THIRD_PARTY_LICENSES` directory
covers some named components, but is not a verified manifest of this wheel.

The cp311 wheel in `3rdparty/` bundles additional libraries and requires a
separate inventory if it is included in a release. Removing the temporary
wheel file after installation does not remove its installed binary code.

The Pico debug path imports `geniesim_teleop.utils.vr_server`. A published
`geniesim_teleop` wheel additionally includes its `app/` tree, which contains
prebuilt motion-control and vendor libraries. If that sibling package is
distributed, its binary provenance, applicable notices and any required
source must be checked separately; importing the Pico Python helper does
not establish those records.

## Retained source attribution

`render/bridges/_multi_env_helpers.py` retains attribution to the legacy
`rl_renderer.py`. Its same-repository source is available at commit
`31cf127842dd60cfa0f40a83f1cf698ef9d2ac37`, path
`source/rlinf_geniesim/renderer/rl_renderer.py`; that file carries the
AgiBot copyright and MPL-2.0 notices. Preserve those notices and this
provenance when distributing the extracted helpers.
