---
name: generate-world
description: >
  Drive `geniesim_world` to produce a photorealistic, explorable 3D
  world from a single equirectangular panorama — uses the
  `geniesim_world create` CLI (Click subcommand), pairs SHARP +
  DA360 to fuse panorama RGB with metric depth, and optionally
  upscales with Real-ESRGAN.
  Trigger: When the user asks to "generate a world", "生成 3D 世界",
  "pano to 3D", "PanoRecon", "make a scene from a photo",
  "create a world from a panorama", or references
  `geniesim_world` / a `.png` panorama input.
license: MPL-2.0
metadata:
  author: genie-sim
  version: "1.0"
prerequisites: []
inputs:
  - name: panorama
    desc: Equirectangular RGB image (2:1 aspect)
    required: true
  - name: work_dir
    desc: Output directory
    required: false
    default: "runs/<timestamp>"
  - name: device
    desc: Torch device
    required: false
    default: cuda:0
  - name: depth_max
    desc: Clip predicted depth at N metres
    required: false
  - name: super_sample
    desc: Run Real-ESRGAN before fusing (needs optional binary)
    required: false
    default: "false"
outputs:
  - desc: "Work dir with `cubes/`, `depth/`, and a fused world export (`.gsp` / `.ply`). RT Engine load path is W.I.P. — for now the output is consumed by `geniesim_world`'s own tooling."
---

## When to Use

- User has an equirectangular panorama (2:1 aspect, e.g.
  `4096×2048`) and wants a 3D world (Gaussian splat / depth map /
  fused cubes) out of it.
- User wants to seed a Genie Sim scene with a generated environment
  rather than a hand-authored USD.
- Researcher reproducing the PanoRecon paper (arXiv 2604.07105).

Do **not** use for:
- USD scene authoring without a panorama input → that's the
  `geniesim-scene-generation` skill in `geniesim_scene_agent`.
- Asset / object library search → `geniesim-asset-search` in
  `geniesim_scene_agent`.
- Importing an existing USD scene into the RT Engine → just point
  the scene yaml's `scene_usda` at the existing USD (see `add-robot`
  / `launch-scene`). This skill is **not** the path for that —
  `geniesim_world` produces `.ply` / `.gsp` Gaussians today, and
  loading them into a `scene_*.yaml` is 🚧 W.I.P.

## Critical Patterns

1. **Out-of-band install.** `geniesim_world` is not pulled in by
   `geniesim bootstrap` — it lives behind heavy CUDA deps (PyTorch +
   ml-sharp + DA360 + optional Real-ESRGAN). Install in its own
   conda env, separate from the rest of the stack.
2. **Three external dependencies, not on pip.** Need
   `external/ml-sharp/`, `external/DA360/` (with checkpoint), and
   optionally `external/realesrgan-ncnn-vulkan-…/`. The `external/`
   tree sits next to `geniesim_world/` under `source/`.
3. **DA360 checkpoint is mandatory.** Either drop it at
   `external/DA360/DA360_large.pth`, pass `--da360-checkpoint`, or
   set `GENIESIM_DA360_CHECKPOINT`. Without it, depth prediction
   fails immediately.
4. **Tested on RTX 5090 / CUDA 12.8.** Other GPUs work but need a
   matching `requirements-cu<XX>.txt`. Don't paste the cu128 line
   verbatim onto a cu118 box.
5. **Real-ESRGAN is optional.** It improves visual fidelity but
   introduces synthetic texture — disable for tasks where
   ground-truth pixel statistics matter.

## Workflow

### Step 1 — Verify the layout

```
source/
├── geniesim_world/                              # this project
└── external/
    ├── ml-sharp/                                # git clone https://github.com/apple/ml-sharp.git
    ├── DA360/                                   # git clone https://github.com/DepthAnything/DA360.git
    │   └── DA360_large.pth                      # checkpoint (download separately)
    └── realesrgan-ncnn-vulkan-20220424-ubuntu/  # optional, super-sample binary
        └── realesrgan-ncnn-vulkan
```

If any external is missing, run the clone / download from the
`geniesim_world/README.md` § "Prepare Dependencies".

### Step 2 — Create a clean env and install

```bash
conda create -n geniesim_world python=3.11 -y
conda activate geniesim_world
pip install --upgrade "pip==24.0" "setuptools==69.5.1" "wheel==0.43.0"

cd source/geniesim_world
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements-cu128.txt
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e .
```

Verify:

```bash
geniesim_world --help                            # Click group
geniesim_world create --help                     # subcommand surface
```

### Step 3 — Generate a world from a panorama

```bash
geniesim_world create \
  --panorama path/to/scene_pano.png \
  --work-dir runs/$(date +%Y%m%d_%H%M%S) \
  --device cuda:0
```

Optional knobs:

| Flag | Effect |
|---|---|
| `--da360-checkpoint <path>` | Override checkpoint path |
| `--da360-root <dir>` | Override DA360 repo root |
| `--checkpoint-path <path>` | SHARP checkpoint override |
| `--depth-max <float>` | Clip predicted depth at N metres |
| `--no-depth-gt-init` | Stock SHARP (ignore depth_gt override) |
| `--super-sample` | Run Real-ESRGAN before fusing (needs the optional binary) |

### Step 4 — Inspect the work-dir

Typical artifacts under `runs/<stamp>/`:

```
runs/<stamp>/
├── pano_input.png             # the panorama (copied for provenance)
├── depth/                     # per-cube depth maps (EXR or PNG)
├── cubes/                     # cubemap faces fused from pano + depth
└── world.gsp / world.ply / …  # exported world (format depends on flags)
```

### Step 5 — (🚧 W.I.P.) Feed into a Genie Sim scene

Loading the generated world into the RT Engine via a `scene_*.yaml`
is **not yet supported**. The current contract is that
`geniesim_world` produces `.ply` / `.gsp` Gaussians (plus per-face
EXR depth + cubemap RGB); downstream consumption is via the package's
own debug tooling (`geniesim_world debug`) and external viewers, not
the RT Engine launcher. The RT-Engine load path is planned — track
the **Roadmap & Updates** section in the root README.

## Commands (copy-paste summary for the user)

```bash
# Once-off setup
conda create -n geniesim_world python=3.11 -y && conda activate geniesim_world
cd source/geniesim_world
pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r requirements-cu128.txt
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e .

# Generate
geniesim_world create \
  --panorama path/to/scene_pano.png \
  --work-dir runs/demo \
  --device cuda:0
```

## Notes

- The CLI is a Click group: `geniesim_world` exposes subcommands,
  with `create` being the primary one. Run `geniesim_world --help`
  for the full list.
- "Generate from text" and "Generate from sparse images" are flagged
  *COMING SOON* in `geniesim_world/README.md` — don't promise them
  in the workflow.
- The env is intentionally isolated from the rest of the stack so
  `pip install` here can't break a working `geniesim_ros` shell.
- For sim-to-real research, the generated world's depth statistics
  are more reliable without `--super-sample` (Real-ESRGAN
  introduces hallucinated texture).

## Resources

- **CLI source**: [source/geniesim_world/src/geniesim_world/cli_pano.py](../../src/geniesim_world/cli_pano.py)
- **README**: [source/geniesim_world/README.md](../../README.md)
- **AGENTS.md**: [source/geniesim_world/AGENTS.md](../../AGENTS.md)
- **Paper**: https://arxiv.org/abs/2604.07105
- **External deps**: `apple/ml-sharp`, `DepthAnything/DA360`, optional `xinntao/Real-ESRGAN`
