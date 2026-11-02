# GenieSim Docker images

This directory contains the Docker image definitions and runtime scripts for running GenieSim with NVIDIA Isaac Sim and ROS 2.

The Docker images are intended to be used as **base development/runtime images**. Users normally bind-mount the GenieSim workspace into `/workspace`, and the container entrypoint installs the mounted tier-1 GenieSim packages in editable mode each time the container starts. This keeps the heavy simulator and ROS layers baked into the image while allowing active GenieSim source development without rebuilding the image.

## Quick start

From the repository root:

```bash
# Build the default combo image: Isaac Sim 5.1 + 6.2 + ROS 2 Jazzy
geniesim docker build

# Build using mainland China mirrors for nvcr.io, apt, and pip
geniesim docker build --china

# Start the container with GUI/X11 support
geniesim docker up

# Start without GUI mounts, for CI or batch workflows
geniesim docker up --headless

# Enter the running container
geniesim docker into

# Stop and remove the container
geniesim docker down
```

`geniesim docker` is an alias for `geniesim docker5.1`, the current default variant.

## Image variants

| Command | Dockerfile | Image | Container | Isaac Sim | OS | ROS 2 | Python strategy | Status |
|---|---|---|---|---|---|---|---|---|
| `geniesim docker` / `geniesim docker5.1` | `Dockerfile.5.1` | `registry.agibot.com/genie-sim/geniesim3:latest` | `geniesim3` | 5.1 + 6.2 (wheel `6.2.0.0`) | Ubuntu 24.04 | Jazzy | `omni_python` for Isaac Sim 5.1, system `python3.12` for Isaac Sim 6.2 + ROS | Default dispatch; recorded system 6.2 checks pass; bundled 5.1 unqualified |
| `geniesim docker4.5` | `Dockerfile.4.5` | `registry.agibot.com/genie-sim/geniesim2:latest` | `geniesim2` | 4.5 | Ubuntu 22.04 | Humble | system `python3.10` | ⚠️ E.O.L. |
| `geniesim docker6.2` | `Dockerfile` | `registry.agibot.com/genie-sim/geniesim4:latest` | `geniesim4` | 6.X (exact base `6.2.0.1`, wheel `6.2.0.0`) | Ubuntu 24.04 | Jazzy | system `python3.12` | Dedicated system runtime; see image-specific acceptance |

The minor version is intentionally not encoded in the main image family name. For example, `geniesim3:latest` tracks the current 5.1-based image line.

`geniesim docker` stays mapped to `docker5.1` until `docker6.2` passes runtime
acceptance. The combo keeps bundled Isaac 5.1 and system-Python Isaac 6.2.

### Isaac Sim 6.X — the GenieSim 4.0 target

The dedicated variant selects base `6.2.0.1` and wheel `6.2.0.0`. Runtime qualification is recorded per image and backend in [runtime acceptance](../source/geniesim_ros/docs/ISAACSIM_ACCEPTANCE.md); a built image alone does not establish scene coverage.

Every variant pins its Isaac Sim version in two places that must agree:

| Dockerfile | series spec | `ISAACSIM_TAG` (NGC base image) | `ISAACSIM_PIP_VERSION` (PyPI wheel) |
|---|---|---|---|
| `Dockerfile` | `6.2.0.1` | `6.2.0.1` | `6.2.0.0` |
| `Dockerfile.5.1` | *(none)* | `5.1.0` | `6.2.0.0` *(the 6.X co-install)* |
| `Dockerfile.4.5` | *(none)* | `4.5.0` | `4.5.0` |

The base is pinned to `nvcr.io/nvidia/isaac-sim:6.2.0.1`. Its four-component
image tag is a 6.2 respin; the published Python wheel remains `6.2.0.0`.
The resolver explicitly pairs these strings rather than requiring an
unpublished `6.2.0.1` wheel.

Where a variant declares a **series spec**, those two are *derived* rather than
maintained: `isaacsim_series` in `docker.py`'s `VARIANTS` names the series,
`geniesim_cli._isaacsim` resolves it against nvcr.io's tag list and
pypi.nvidia.com's wheel index, and `geniesim docker6.2 build` passes the result as
`--build-arg`. Check what a spec resolves to without starting a build:

```sh
python3 -m geniesim_cli._isaacsim 6.2.0.1 --python-tag cp312
# ISAACSIM_TAG=6.2.0.1
# ISAACSIM_PIP_VERSION=6.2.0.0
```

`docker6.2` uses the exact spec `6.2.0.1`, so normal builds retain that base.
A broad `6.X` spec also resolves to this pair using the explicit respin mapping.
The wheel must be available for the target interpreter; unknown respins do
not automatically reuse an earlier wheel.

The `ISAACSIM_TAG` / `ISAACSIM_PIP_VERSION` values in the table remain the
Dockerfile `ARG` defaults and are what a build falls back to when resolution
fails (offline, registry error, no match) — a warning, never a build failure. So
**keep them current**, and bump both by hand for the variants with no spec.
Either way the third string below must match.

A third string lives outside this directory and belongs to the same release:
`mujoco~=3.12.0` in `source/geniesim_ros/pyproject.toml`. Isaac Sim owns the
Newton that consumes MuJoCo (6.2 ships Newton 1.6.1 with MuJoCo and MuJoCo-Warp
3.12.0, Warp 1.17.0), and newton 1.6.1 in turn requires `mujoco~=3.12.0` — the
pin is Newton's own specifier, not a number we picked. The dependency layer
installs it *over* Isaac's copy, so it must name the same release as the two
above.
`source/geniesim_ros/AGENTS.md` carries the full Isaac-owned version table.

The [runtime guide](../source/geniesim_ros/AGENTS.md) describes current assembly, physics and render requirements. Open release gates are in the [platform roadmap](../docs/ROADMAP.md).

`check_newton_pins.py` runs after dependency installation and fails the build when installed MuJoCo/MuJoCo-Warp versions violate Newton's requirements. SDK/colcon constraints govern transitive resolution; omitting a direct requirement from `SKIP_EXACT` does not protect its installed version.

## Layer design

### Post-training profiles

The post-training simulation images reuse an existing GenieSim base:

```text
NVIDIA Isaac Sim 6.2.0.1
  └─ Dockerfile → registry.agibot.com/genie-sim/geniesim4:latest (ROS 2 Jazzy, Newton/MuJoCo/Warp)
      └─ Dockerfile.post-train → geniesim-post-train:server
          └─ Dockerfile.post-train-teleop → geniesim-post-train:teleop (optional Pico/IK)
```

The RLinf training/client image is separate; it installs the lightweight
`rlinf-geniesim-client` package and connects over ZMQ. These profiles prepare
the simulation side and keep source and assets mounted at runtime.

From the repository root on the simulation host, outside containers:

```bash
python3 -m pip install -e source/geniesim_cli
geniesim docker6.2 build
geniesim docker --profile post-train build
export GENIESIM_ASSETS_SRC=/path/to/geniesim_assets
geniesim docker --profile post-train up --headless
geniesim docker --profile post-train into
```

Inside the server container, from `/workspace`:

```bash
geniesim post-train server start \
  --count 1 --base-port 5601 --devices 0
```

The RLinf client chooses the task and physics/render backends when initializing
the server. `geniesim post-train server status/stop` operates the selected ports;
`geniesim post-train server logs --port 5601 --follow` streams a server's log.
See [the server workflow](../source/rlinf_geniesim/README.md) for
task initialization and [teleoperation](../source/rlinf_geniesim/TELEOP_USAGE.md)
for the optional image. To inspect or remove the server container, run on the host:

```bash
geniesim docker --profile post-train logs -f
geniesim docker --profile post-train down
```

| Profile | Default image | Default container | Parent build argument |
|---|---|---|---|
| `post-train` | `geniesim-post-train:server` | `post_train_server` | `BASE_IMAGE` (default `registry.agibot.com/genie-sim/geniesim4:latest`) |
| `post-train-teleop` | `geniesim-post-train:teleop` | `post_train_teleop` | `SERVER_IMAGE` (default `geniesim-post-train:server`) |

Append `--build-arg BASE_IMAGE=IMAGE` or `--build-arg SERVER_IMAGE=IMAGE` to
the corresponding build command to select a local or mirrored parent. The
server parent must supply system Python 3.12, ROS Jazzy, bundled Isaac Sim 6.2.x
at `/isaac-sim` including `/isaac-sim/python.sh`, the matched physics libraries,
and GenieSim's runtime user/tools. Both builds check `/isaac-sim/VERSION` before
installing dependencies and reject other Isaac Sim versions. A custom teleop
parent must be a server image built with the `post-train` profile.
The default `geniesim docker build` produces a 5.1-based image, which is not a
compatible parent even if its system Python also has a newer Isaac Sim package.
A stock NVIDIA Isaac image alone is insufficient. `--china` selects
apt/pip mirrors for the added layers; it does not mirror the parent image or
the NVIDIA OVRTX wheel index.

The base build uses the paired NGC tag and wheel versions described above
and retains Isaac Sim's Newton/MuJoCo/Warp versions.
Build the base locally before the post-training layer; the post-training
build uses that local image without contacting the registry.

[`post-train-runtime.txt`](post-train-runtime.txt) pins the server layer's added
Python packages. These pins do not replace the
base image's dependency inventory.

Both profiles reuse the shared launcher, host UID/GID mapping, seeded shader
cache volumes, and host networking, with 8 GB shared memory. `GENIESIM_IMAGE`,
`GENIESIM_CONTAINER`, `GENIESIM_WORKSPACE`, and `GENIESIM_CACHE_ROOT` override
their respective defaults. Existing containers must have matching checkout
and profile labels. `up` fails if the entrypoint exits or is not ready within
`GENIESIM_START_TIMEOUT` seconds (default 300); inspect `logs` before retrying.
It does not start simulation servers automatically.

The profiles use the entrypoint selected by the image. Changes to
`entrypoint.post-train.sh` require rebuilding the image; the common
`entrypoint.sh` and simulation source remain mounted from the checkout.
The profile entrypoint prepares the bundled Kit's runtime directories for the
mapped user. These images disable the inherited Kit-log healthcheck because
idle containers and teleop clients do not require a running Kit process.
Use `geniesim post-train server status` inside the server container to check
its simulation processes.

Base-image dependency conflicts and GPU backend acceptance remain in the
[runtime validation TODOs](../source/rlinf_geniesim/tests/README.md#pending-runtime-validation).

### Base image layers

Each image is built in the same conceptual order:

```text
NVIDIA Isaac Sim base image
  └─ OS tools, compiler toolchain, graphics/runtime libraries, locale
      └─ ROS 2 distribution for the base Ubuntu release
          └─ Isaac Sim Python package layer when needed
              └─ third-party Python dependencies for tier-1 GenieSim packages
                  └─ rosdep system dependencies for geniesim_ros packages
                      └─ shared GenieSim entrypoint and /workspace mountpoint
```

### 1. NVIDIA Isaac Sim base

All variants start from NVIDIA's published Isaac Sim image:

- 5.1 variant: `nvcr.io/nvidia/isaac-sim:5.1.0`
- 4.5 variant: `nvcr.io/nvidia/isaac-sim:4.5.0`
- 6.X variant: `nvcr.io/nvidia/isaac-sim:6.2.0.1`

The NVIDIA base image already contains the `isaac-sim` user with UID/GID `1234:1234` and, for bundled Isaac Sim variants, `/isaac-sim` plus `/isaac-sim/python.sh`.

The GenieSim image keeps the build-time user as `1234:1234`. Host UID/GID remapping is done at container startup by `entrypoint.sh`, not at build time.

### 2. OS and graphics/runtime packages

The images install common system packages used by development, ROS, Isaac Sim, and GenieSim:

- shell/dev tools: `bash`, `git`, `git-lfs`, `sudo`, `vim`, `curl`, `wget`, `jq`
- build tools: `build-essential`, `cmake`, `python3-dev`, `libeigen3-dev`
- graphics/runtime libraries: Vulkan, X11, EGL/OpenGL, `ffmpeg`
- ROS support tools: `ros-dev-tools`, `rosdep`
- graph tooling: `graphviz`, `libgraphviz-dev`

Locale is configured as `en_US.UTF-8`.

### 3. ROS 2 layer

The ROS distribution follows the Ubuntu base:

- Ubuntu 24.04 / Noble images use **ROS 2 Jazzy**.
- Ubuntu 22.04 / Jammy images use **ROS 2 Humble**.

Installed ROS packages include the desktop distribution, CycloneDDS RMW, RViz vision message plugins, joystick support, and development tools.

At runtime, the entrypoint configures ROS for local-container discovery:

- `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`
- `ROS_LOCALHOST_ONLY=1` for Humble
- `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST` for Jazzy
- a loopback-only CycloneDDS URI in `/etc/profile.d/geniesim.sh`

### 4. Isaac Sim Python layer

Different Isaac Sim releases use different Python strategies.

#### 5.1 combo image: `Dockerfile.5.1`

This is the default image and intentionally contains two Isaac Sim runtimes:

```text
omni_python  -> /isaac-sim/python.sh, Isaac Sim 5.1 bundled Python
python3      -> system Python 3.12, ROS 2 Jazzy, pip-installed Isaac Sim 6.X
```

Use this image when you need the stable Isaac Sim 5.1 environment but also want access to Isaac Sim 6.X Python packages from system Python.

Important details:

- `/usr/local/bin/omni_python` wraps `/isaac-sim/python.sh`.
- `omni_python` is the default runtime interpreter for this variant.
- Isaac Sim 6.X is also installed into system `python3.12` with `pip install "isaacsim[all,extscache]==6.2.0.0"`.
- Tier-1 GenieSim packages are installed into both interpreters at runtime.
- `numpy==1.26.4` and `scipy==1.13.1` are pinned for the 5.1 bundled Python.
- `3rdparty/ik_solver-0.4.3-cp311-cp311-linux_x86_64.whl` is installed into `omni_python`.

#### 4.5 image: `Dockerfile.4.5`

The 4.5 image installs Isaac Sim 4.5 into system Python 3.10:

```text
python3 -> system Python 3.10, ROS 2 Humble, pip-installed Isaac Sim 4.5
```

⚠️ **E.O.L. in the open-source GitHub repository.** The recipe and CLI remain available for compatibility with Isaac Sim 4.5 / ROS 2 Humble workflows.

The Dockerfile includes two compatibility fixes before adding the ROS repository:

1. Downgrade `libbrotli1` to the Ubuntu Jammy version required by `libbrotli-dev`.
2. Install `libfreetype6-dev` and `libfontconfig1-dev` from Ubuntu repositories before ROS packages are resolved.

#### 6.X image: `Dockerfile`

The 6.X image is shaped like a system-Python-only image:

```text
python3 -> system Python 3.12, ROS 2 Jazzy, pip-installed Isaac Sim 6.X
```

Base image `isaac-sim:6.2.0.1`, wheel `isaacsim[all,extscache]==6.2.0.0`,
resolved from the exact `6.2.0.1` spec. Build and runtime acceptance require a compatible checkout and the selected physics/render environment.

`geniesim docker` selects `docker5.1`. Use `docker6.2` explicitly for the dedicated system runtime.

#### System-Python runtime

Use **system Python 3.12 / Isaac 6.2** for the system runtime. Bundled
`omni_python` is a separate environment and needs independent validation.

Build this configuration with the normal command:

```bash
geniesim docker5.1 build
```

The image retains Newton 1.6.1 and MuJoCo/MuJoCo-Warp 3.12.0 through its SDK
constraints.
The system dependency layer also prepares OVRtx's
`bin/mdl/omniverse_exts` directory: its root-owned installed parent prevents a
remapped non-root renderer from creating that material search directory.

Use the system-interpreter CLI configuration with the combo image:

```bash
GENIESIM_IMAGE=registry.agibot.com/genie-sim/geniesim3:latest \
GENIESIM_CONTAINER=geniesim3-system geniesim docker6.2 up --headless
GENIESIM_CONTAINER=geniesim3-system geniesim docker6.2 into
# Inside the container:
python3 --version
geniesim ros build dev
source /workspace/devel/setup.bash
```

`docker5.1 up` selects bundled `omni_python`; overriding its image alone does
not select the system runtime. See [runtime acceptance](../source/geniesim_ros/docs/ISAACSIM_ACCEPTANCE.md)
for the measured image and GPU/ROS checks.

The [recorded geniesim3 system-Python qualification](../source/geniesim_ros/docs/ISAACSIM_ACCEPTANCE.md#geniesim3-system-python-2026-10-25)
identifies the measured clean rebuild, SDK provenance and scene coverage limits.
Those measurements apply to that image identity rather than every `latest` tag.

#### Bundled Isaac 5.1 runtime status

Bundled Python 3.11 uses its own NumPy prebundle and dependency set. Validate
its imports and rendering separately from system Python.

### 5. GenieSim dependency layer

The Dockerfiles do **not** copy the entire source tree into the image. They copy only selected `pyproject.toml` files and run `docker/collect_deps.py` to extract third-party dependencies for tier-1 GenieSim packages.

This keeps Docker layer invalidation smaller:

- changing Python source does not force rebuilding the heavy image layers;
- changing dependencies in a tier-1 `pyproject.toml` does force rebuilding the dependency layer;
- the active source tree is mounted at runtime and installed editable by the entrypoint.

`collect_deps.py` reads tier-1 package names from `source/geniesim/pyproject.toml`, skips internal `geniesim*` dependencies, and skips packages provided by the simulator/runtime layer such as `isaacsim`, `cv_bridge`, `numpy`, and `scipy`.

Tier-2 packages such as teleop, scene agent, and world are intentionally not baked into the default image. Users opt into them inside the container, for example:

```bash
pip install -e "source/geniesim/[teleop]"
pip install -e "source/geniesim/[scene_agent]"
pip install -e "source/geniesim/[world]"
pip install -e "source/geniesim/[all]"
```

### Isaac/colcon package versions

The system-Python dependency layers in `Dockerfile` and `Dockerfile.5.1` use
`collect_runtime_constraints.py` to read installed Isaac Sim and colcon
`Requires-Dist` metadata. Platform markers and activated dependency extras are
respected. Missing or incompatible packages are explicitly selected for repair;
GenieSim requirements resolve under the combined SDK constraints.

For Isaac 6.2, this preserves NumPy 2.3.1, absl-py 2.4.0, etils 1.13.0,
fsspec 2026.2.0 and glfw 2.10.0, while setuptools follows colcon's `<80,>=30.3.0`
range. The versions are read from metadata, not duplicated in a hand-maintained
lock file. The installer retains compatible installed packages, and the Isaac
profile's final `pip check` fails the build if declared dependencies disagree.
Newton's optional sim requirements are also checked by `check_newton_pins.py`.

The generated constraints remain at `/opt/geniesim/docker/isaac-constraints.txt`.
Apply them explicitly when installing additional packages into system Python.
The bundled 5.1 interpreter and opt-in standalone Newton profile use their own
version contracts; the Isaac constraints are not exported globally.
These constraints govern packages installed by pip; they do not pin the pip
executable itself.

## Runtime design

The image is only half of the design. The shared scripts make the image usable as a development base.

### `start.sh`

`start.sh` is called by the CLI when users run `geniesim docker up`.

It starts `docker run` with:

- GPU access: `--gpus all`
- host networking: `--network=host`
- privileged mode for simulator/device workflows
- host UID/GID environment variables
- the workspace bind-mounted at `/workspace`
- editable `geniesim_assets` bind-mounted at `/geniesim_assets`
- persistent Isaac Sim cache/config/log/data mounts
- optional X11 and input-device mounts for GUI mode
- named volumes for in-package Isaac Sim shader caches

By default, the workspace is the repository root located by the CLI. You can override it:

```bash
GENIESIM_WORKSPACE=/path/to/repo geniesim docker up
```

The cache root defaults to `~/docker/isaac-sim`:

```bash
GENIESIM_CACHE_ROOT=/fast/disk/isaac-cache geniesim docker up
```

### `entrypoint.sh`

`entrypoint.sh` runs every time the container starts. It is responsible for:

1. remapping the in-container `isaac-sim` user from `1234:1234` to the host UID/GID;
2. preparing the ROS, CycloneDDS, Isaac Sim, and GenieSim environment;
3. writing `/etc/profile.d/geniesim.sh` so interactive shells inherit the environment;
4. installing tier-1 GenieSim packages from `/workspace/source/*` in editable mode;
5. dropping privileges from root to the remapped `isaac-sim` user before running the command.

Warp kernels use `/tmp/geniesim-newton-warp-<uid>` rather than the shared host cache.
This prevents another container running as the image's original UID 1234 from
making the Newton process's JIT cache unwritable after host UID remapping.

This means users can develop on top of the base image by editing files on the host and restarting or re-entering the container, rather than rebuilding the Docker image for every source change.

### `into.sh`

`into.sh` is a small helper that enters an already-running container with the current host UID/GID, `HOME=/home/isaac-sim`, and working directory `/workspace`. The CLI implements the same behavior for `geniesim docker into`.

## Cache and volume layout

`start.sh` creates this host-side cache layout under `GENIESIM_CACHE_ROOT`, defaulting to `~/docker/isaac-sim`:

```text
cache/main/ov              -> /home/isaac-sim/.cache/ov
cache/main/warp            -> /home/isaac-sim/.cache/warp
cache/computecache         -> /home/isaac-sim/.nv/ComputeCache
config                     -> /home/isaac-sim/.nvidia-omniverse/config
data                       -> /home/isaac-sim/.local/share/ov/data
logs                       -> /home/isaac-sim/.nvidia-omniverse/logs
pkg                        -> /home/isaac-sim/.local/share/ov/pkg
```

Some Isaac Sim package-internal cache directories are mounted as **named Docker volumes**, not host bind mounts. This matters because Docker named volumes are seeded from the image on first creation. A plain empty host bind mount would hide files shipped in the image, such as shader cache metadata, and can break rendering.

## Developing on top of the base image

There are two recommended extension patterns.

### Option A: develop from a bind-mounted workspace

Use this when you are changing GenieSim source code or adding Python packages in the repo.

```bash
geniesim docker up
geniesim docker into

# Inside the container
python3 -m pip install -e source/my_package --break-system-packages
```

For the default 5.1 image, install packages into `omni_python` too if they need to be imported by Isaac Sim 5.1 code:

```bash
omni_python -m pip install -e source/my_package
```

This approach is fast and does not require image rebuilds for source edits.

### Option B: build a downstream image

Use this when you want to distribute a repeatable image with extra system packages, Python dependencies, or prebuilt assets.

Example for the default image:

```dockerfile
FROM registry.agibot.com/genie-sim/geniesim3:latest

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    your-system-package \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --break-system-packages your-python-package

# If your code must run under Isaac Sim 5.1's bundled Python:
RUN omni_python -m pip install your-omni-python-package

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/geniesim-entrypoint"]
CMD ["bash"]
```

For downstream images, keep these conventions unless you have a specific reason to change them:

- keep `/workspace` as the user workspace;
- keep `/geniesim_assets` as the assets mount point;
- keep `/usr/local/bin/geniesim-entrypoint` as the entrypoint;
- keep the `isaac-sim` user available;
- do not bake active GenieSim source into the image unless you want an immutable runtime image;
- install 5.1-specific Python dependencies with `omni_python`, and system/ROS/6.X dependencies with `python3`.

## Build arguments

`geniesim <variant> build` accepts these flags before the ones it forwards to
`docker build`:

| Flag | Purpose |
|---|---|
| `--mirrors NAME` | Select a preset from [`mirrors.toml`](mirrors.toml) |
| `--china` | Alias for `--mirrors china` |
| `--isaacsim <spec>` | Override the variant's series spec for this build (`--isaacsim 6.2.X`, or the exact base `--isaacsim 6.2.0.1`) |
| `--no-resolve` | Skip version resolution entirely; use the Dockerfile `ARG` defaults |

Anything else is forwarded verbatim (`--no-cache`, `--progress=plain`, …). Two
env vars bypass resolution and pin both strings directly — set **both** or
neither:

| Env var | Purpose |
|---|---|
| `GENIESIM_ISAACSIM_TAG` | NGC base image tag to build against |
| `GENIESIM_ISAACSIM_PIP_VERSION` | PyPI wheel version to install |

Base Dockerfiles support these build arguments. Derived profiles accept
`APT_MIRROR` and `PIP_INDEX_URL` without defaults and inherit their parent:

| Build arg | Default | Purpose |
|---|---|---|
| `ISAACSIM_REGISTRY` | `nvcr.io/nvidia` | Registry/path prefix for the NVIDIA Isaac Sim base image |
| `ISAACSIM_TAG` | variant-specific | Isaac Sim **base image** tag (`6.2.0.1` / `5.1.0` / `4.5.0`). Passed explicitly by `geniesim docker6.2 build` from the resolved series spec; the `ARG` default is the offline fallback |
| `ISAACSIM_PIP_VERSION` | variant-specific | Isaac Sim **PyPI wheel** version (`6.2.0.0` / `4.5.0`) — must name the same release as `ISAACSIM_TAG`; resolved alongside it |
| `APT_MIRROR` | `upstream` | Official Ubuntu archive/security sources, or a mirror URL; empty inherits |
| `PIP_INDEX_URL` | `https://pypi.org/simple` | Image-wide primary pip index; empty inherits |
| `ISAACSIM_PYPI_INDEX` | `https://pypi.nvidia.com` | NVIDIA pip index for Isaac Sim wheels |

Base builds default to `upstream`. The `china` preset selects DaoCloud for the
NVIDIA base image and TUNA for Ubuntu and ordinary Python packages. Add custom
tables to [`mirrors.toml`](mirrors.toml) using the same build-argument names.
Only arguments declared by the selected Dockerfile are passed. Explicit
`--build-arg NAME=VALUE` values take precedence over the preset; the CLI prints
the selected settings and their source. Bare `--build-arg NAME` retains Docker's
environment forwarding.

| Build | No mirror flag | Explicit `--mirrors NAME` |
|---|---|---|
| Base (`docker4.5`, `docker5.1`, `docker6.2`) | Upstream | Apply the selected preset |
| Derived (`--profile post-train`, `--profile post-train-teleop`) | Inherit parent | Override inherited defaults |

```bash
geniesim docker6.2 build
geniesim docker6.2 build --china
geniesim docker --profile post-train build
geniesim docker --profile post-train build --mirrors upstream
```

The selected APT/pip settings persist in the image. The shared
`geniesim-configure-mirrors` helper updates Ubuntu sources and `/etc/pip.conf`;
the image's `PIP_CONFIG_FILE=/etc/pip.conf` makes the same pip defaults apply to
root and runtime users. Runtime environment `PIP_INDEX_URL` or pip's explicit
`--index-url` can override them. Empty helper values preserve inherited settings.
Unrelated APT repositories and global pip options are preserved. Rebuild the
base image before building derived profiles that use this helper.

Mirror selection leaves Isaac metadata discovery and the NVIDIA package index
upstream. Use `--no-resolve` or the explicit version pins above when metadata
discovery is unavailable. A mirror selection does not change the Isaac release.


Manual build example:

```bash
docker build \
  -f docker/Dockerfile.5.1 \
  -t registry.agibot.com/genie-sim/geniesim3:latest \
  .
```

## Runtime environment variables

The CLI injects variant-specific environment variables into `start.sh` and the container entrypoint.

| Variable | Purpose |
|---|---|
| `ROS_DISTRO` | ROS distribution to source, usually `jazzy` or `humble` |
| `GENIESIM_PY_CMD` | default variant interpreter, usually `python3` or `omni_python` |
| `GENIESIM_BREAK_SYSTEM_PKGS` | whether entrypoint pip calls should use `--break-system-packages` |
| `GENIESIM_CHOWN_KIT_PATH` | pip-installed Isaac Sim `kit` path that must be writable after UID remap |
| `GENIESIM_EXTRA_CACHE_DIRS` | extra cache subdirectories to create under `cache/main` |
| `GENIESIM_ISAACSIM_KIT_CACHE_PATH` | in-package Isaac Sim kit cache path mounted as a named volume |
| `GENIESIM_OVRTX_CACHE_PATH` | in-package ovrtx cache path mounted as a named volume |

User-facing overrides:

| Variable | Default | Purpose |
|---|---|---|
| `GENIESIM_IMAGE` | variant image tag | Run or build a different image tag |
| `GENIESIM_CONTAINER` | variant container name | Use a different container name |
| `GENIESIM_WORKSPACE` | repo root | Host path mounted at `/workspace` |
| `GENIESIM_CACHE_ROOT` | `~/docker/isaac-sim` | Host path for Isaac Sim cache/config/log/data mounts |

## Common workflows

### Rebuild after dependency changes

If a tier-1 `pyproject.toml` changes, rebuild the image so the dependency layer is updated:

```bash
geniesim docker down
geniesim docker build
geniesim docker up
```

### Source-only changes

If only Python/C++ source files changed, usually no rebuild is needed. Restarting the container re-runs editable installs:

```bash
geniesim docker down
geniesim docker up
```

Or, inside the container, manually reinstall the edited package:

```bash
python3 -m pip install -e source/geniesim_ros --break-system-packages
```

### Use the 4.5 compatibility image

⚠️ `geniesim2` is E.O.L. in the open-source GitHub repository.

```bash
geniesim docker4.5 build
geniesim docker4.5 up
geniesim docker4.5 into
```

### Use a custom image tag

```bash
GENIESIM_IMAGE=my-registry/geniesim-custom:dev geniesim docker build
GENIESIM_IMAGE=my-registry/geniesim-custom:dev geniesim docker up
```

## File map

| File | Purpose |
|---|---|
| `Dockerfile.5.1` | Default image: Isaac Sim 5.1 bundled runtime plus Isaac Sim 6.X in system Python |
| `Dockerfile.4.5` | ⚠️ E.O.L. compatibility image for Isaac Sim 4.5 + ROS 2 Humble |
| `Dockerfile` | Isaac Sim 6.X (series `6.X`) + ROS 2 Jazzy image for GenieSim 4.0 |
| `start.sh` | Shared container launch script used by the CLI |
| `entrypoint.sh` | Shared runtime setup, UID remap, environment setup, editable install |
| `into.sh` | Helper for entering an existing container |
| `collect_deps.py` | Extracts tier-1 third-party Python dependencies from `pyproject.toml` files |
| `check_newton_pins.py` | Build-time guard: fails the build if the installed MuJoCo does not satisfy Isaac's Newton |
| `AGENTS.md` | Maintainer-facing notes for agents and contributors |

## Troubleshooting

### `geniesim_assets is not pip-installed (editable) on the host`

`start.sh` needs to locate the editable `geniesim_assets` source root and mount it at `/geniesim_assets`.

Install it on the host first:

```bash
pip install -e /path/to/geniesim_assets
```

### Files created in `/workspace` have the wrong owner

Use the CLI path (`geniesim docker up`) so `HOST_UID` and `HOST_GID` are passed automatically. If running Docker manually, pass:

```bash
-e HOST_UID=$(id -u) -e HOST_GID=$(id -g)
```

### GUI does not open

Use GUI mode, not headless mode:

```bash
geniesim docker up
```

Make sure the host has a working X11 session and that Docker can access `/tmp/.X11-unix`. `start.sh` runs `xhost +local:` automatically for local containers.

### Need CI or batch mode

Use headless mode:

```bash
geniesim docker up --headless
```

This skips X11, Xauthority, and `/dev/input` mounts.


## Native libuipc runtime

Jazzy base builds install the bundled architecture-specific libuipc wheel into
system Python 3.12. The CLI selects host GPU 0 automatically:

```bash
geniesim docker build
geniesim docker6.2 build
# Select a deployment GPU architecture explicitly, including on a GPU-less host:
geniesim docker build --build-arg LIBUIPC_SM=89
geniesim docker6.2 build --build-arg LIBUIPC_SM=120
```

Only SM89 and SM120 are supported. Unsupported or unreadable GPU capability
stops the build before Docker, with a manual override hint. Raw `docker build`
requires `--build-arg LIBUIPC_SM=89` or `120`; the Docker build stage itself
needs no GPU. Native linking requires the image's complete CUDA 13 provider.
Both artifacts require Linux x86_64 CPython 3.12; bundled `omni_python` 3.11,
the Humble base and derived post-training layers do not receive this build step.

The installer verifies SHA256, preserves installed SDK dependency versions and
restores the previous native installation if publication fails. The image
records the selected architecture and artifact hash. Its entrypoint calls
`python3 -m geniesim_cli bootstrap --libuipc-only` through the mounted CLI.
Marked images verify the artifact, dependencies and native loading without pip;
unmarked compatible environments retain installation/repair. Optional native
failures warn and leave ordinary container startup available.

`geniesim bootstrap` checks libuipc after required peers, including when healthy
peers are left unchanged. `geniesim bootstrap --libuipc-only` runs just the native
step without peer probes or prompts. Required peer failures still fail bootstrap.
For platform requirements, hashes, license and provenance, see
[the native wheel notes](../3rdparty/pyuipc-README.md). CPU tests do not establish
an image build, GPU simulation or task success.
