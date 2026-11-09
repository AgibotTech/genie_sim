# Isaac Sim 6.2 container

This image starts from NVIDIA's official `nvcr.io/nvidia/isaac-sim:6.2.0.1`,
the same base release configured by the repository's root Dockerfile.
It includes this module's Python tools, skills, API references, and example
library. External assets, model weights, generated data, and local caches
are not copied into the image.

The default `runtime` target supports asset catalogs, DSL/direct/hybrid USD
generation, mesh and support-surface inspection, example search, and headless
Isaac Sim rendering. The optional `vl` target adds GPU video embeddings.
No Open WebUI, MCP services, ROS installation, collection service, second
Isaac Sim installation, or additional GUI stack is added.

This scene-agent image is sufficient for standalone scene design, USD
validation and Isaac Sim previews. The data-collection image is a separate,
optional runtime for executing robot policies and recording episodes. Building
or running this image does not pull or install the collection image.

## Build

The default image name is `registry.agibot.com/genie-sim/geniesim3-scene-agent:latest`,
and the container name is `geniesim3-scene-agent`. The optional VL build uses
the same image repository with the `vl` tag.

Run from this module's root (the directory containing `pyproject.toml`):

```bash
python3 docker/build_image.py

# Optional GPU embedding dependencies; this is a much larger build.
python3 docker/build_image.py --target vl

# Shared presets and custom URLs (set the variables to your own URLs first):
python3 docker/build_image.py --china
python3 docker/build_image.py --mirrors upstream --dry-run
python3 docker/build_image.py --china \
  --build-arg APT_MIRROR="$APT_MIRROR" \
  --build-arg PIP_INDEX_URL="$PIP_INDEX_URL"
```

Run these commands from `source/geniesim_scene_agent/`.
The monorepo root is not the build context. Pip installation and wheel
deployment do not trigger this build or an image pull. `docker build` can fetch
the official base image if it is absent; `docker run` can fetch the requested
scene-agent image if it is absent. Build arguments:

| Argument | Default | Purpose |
| --- | --- | --- |
| `ISAACSIM_IMAGE` | `nvcr.io/nvidia/isaac-sim:6.2.0.1` | Official base image or an exact mirror of it |
| `ISAACSIM_REGISTRY` | `nvcr.io/nvidia` | Registry prefix; an explicit `ISAACSIM_IMAGE` takes precedence |
| `ISAACSIM_TAG` | `6.2.0.1` | Official base release; an explicit `ISAACSIM_IMAGE` takes precedence |
| `SCENE_AGENT_VERSION` | checkout `VERSION` through the build script | Materialize the release version inside the module-only context |
| `APT_MIRROR` | inherit parent | Ubuntu mirror URL, or `upstream` |
| `PIP_INDEX_URL` | inherit parent | Persistent primary pip index; official base defaults to upstream |

The build script uses the checkout's lightweight CLI sources and shared
`docker/mirrors.toml`; no editable CLI install or scene runtime is required on
the host. It supports `--image`, `--target runtime|vl|mirrors`, `--dry-run`,
`--china`/`--mirrors NAME`, and ordinary Docker build arguments. `--china` is
an alias for `--mirrors china`; explicit `--build-arg` values override presets.
`mirrors` builds only the configuration stage, without application dependencies.
Use Python 3.11+ on the build host, or Python 3.10 with `tomli` installed for
reading the shared TOML presets. Building images requires Docker with BuildKit
and named-context support.

Without a preset, empty or omitted APT/pip values retain parent settings.
`--mirrors upstream` explicitly resets them. The shared helper configures
Ubuntu repositories and `/etc/pip.conf` before dependency installation;
`PIP_CONFIG_FILE=/etc/pip.conf` persists into both `runtime` and `vl`, including
the authoring venv and Isaac Python. Ordinary later pip installs read that file;
explicit pip options/environment variables can override it. Model download
endpoints are independent runtime settings. Mirror changes require rebuilding
the image and do not replace packages already present in its parent.

For a direct BuildKit build, supply the shared helper as an additional context:

```bash
docker build -f docker/Dockerfile.6.2 \
  --build-context geniesim_mirrors=../../docker \
  --build-arg SCENE_AGENT_VERSION="$(cat VERSION)" \
  -t registry.agibot.com/genie-sim/geniesim3-scene-agent:latest .
```

When exporting only this module, provide the shared `docker/` resources through
that named context. Raw Docker accepts URL build arguments, not preset flags.
The main build context stays this module; external assets and models stay mounted.
The build script passes the checkout's resolved `VERSION`, because its monorepo
symlink points outside this context. Direct builds must pass the value as above,
or export the module with a regular, nonempty `VERSION` file.

The image does not contain an LLM agent application. Run an agent that reads
the bundled skills and invokes these Python tools, or invoke the tools
directly. Model weights and embedding databases remain optional runtime data.

## Python environments

- `python` and the `scene-assets`/`scene-scenes` commands use
  `/opt/scene-agent-venv`, an isolated environment created with the base
  image's Python 3.12. It contains NumPy, Pillow, and standalone USD bindings.
- `ISAACSIM_PYTHON=/isaac-sim/python.sh` selects the original Isaac Sim
  interpreter for the render subprocess. Its USD plugins load during
  `SimulationApp` startup. The image does not install standalone `usd-core`
  or upgrade NumPy in this interpreter.
- GPU embedding dependencies, when selected, go into the isolated tools
  environment. `docker/constraints.txt` fixes the Qwen implementation's
  directly relevant versions. Remaining transitive dependencies are not
  fully locked.

The authoring constraints retain NumPy 1.26.4, USD 25.11 and the existing
optional VL stack; migration does not require upgrading those dependencies.
The renderer imports NumPy, Pillow and USD after `SimulationApp` initializes
the SDK's extension paths. Batch previews recreate cameras, annotators and
render products for each scene and release capture resources before replacing
the scene reference. Isaac Sim 6.2 starts with
`--/rtx/hydra/supportMultiTickRate=false` for explicit Replicator captures
after scene replacement; older configured SDKs keep their launch settings.

## Run

For rendering, the host needs a compatible NVIDIA GPU/driver and NVIDIA
Container Toolkit. NVIDIA's container supports standalone Python rendering
in headless mode; a desktop display is not required.
Pass `ACCEPT_EULA=Y` when accepting the Isaac Sim license for a render run.

The default container user is the official image's `isaac-sim` user
(UID/GID 1234). Prepare writable host output/cache directories for that user,
or use Docker-managed volumes as below. The asset mount should be read-only:

```bash
export GENIESIM_ASSETS=/path/to/assets
docker run --name geniesim3-scene-agent --rm -it --gpus all --init \
  -e ACCEPT_EULA=Y \
  --mount "type=bind,source=$GENIESIM_ASSETS,target=/assets,readonly" \
  --mount type=volume,source=scene-agent-output,target=/workspace/output \
  --mount type=volume,source=scene-agent-cache,target=/workspace/cache \
  --mount type=volume,source=scene-agent-home,target=/tmp/scene-agent-home \
  registry.agibot.com/genie-sim/geniesim3-scene-agent:latest
```

No ports or network services are started. A shell is the default command.
Inside the container:

```bash
python -m scene_agent_scene.similar "fruit on shelves" --limit 5
python -m scene_agent_scene.find_example search "open microwave" --limit 5
python -m scene_agent_assets catalog-build \
  --assets-root "$GENIESIM_ASSETS" --catalog .cache/catalog

python -m scene_agent_scene build --mode dsl \
  --recipe skills/geniesim-scene-generation/examples/table_block_dsl.py \
  --assets-root "$GENIESIM_ASSETS" --scene-id table_block \
  --output-root "$SCENE_AGENT_OUTPUT" --isaac-python "$ISAACSIM_PYTHON"
```

The entrypoint binds `examples/assets` when an assets directory containing
`objects/` is available. Missing assets do not prevent example-code browsing
or USD authoring with generated geometry. Actual asset-backed generation needs
the matching assets and their textures/referenced files.

Generated scenes reference mounted assets. To load an exported scene outside
the container, preserve or rebind the relative asset-reference layout; copying
only `scene.usda` to an arbitrary host directory will not resolve those paths.

For the `vl` target, also mount the model directory read-only at `/models`
with weights in `/models/Qwen3-VL-Embedding-2B`. `QWEN3_VL_MODEL` can override
that path. Build an embedding database explicitly with `vl-build`; full-library
indexing takes a long time. Cached vectors live outside the asset tree. Use
the directory-index skill fallback when VL is unavailable or incomplete.

## Validate generated scenes

After running the build command above, validate its actual scene output inside
the same container:

```bash
python -m scene_agent_scene validate "$SCENE_AGENT_OUTPUT/table_block/0" --require-renders
```

Inspect `validation.json` and show `renders/front.png`, `renders/side.png`, and
`renders/top.png` to review the scene. Robot motion validation uses the optional
collection runtime when task execution is requested.
