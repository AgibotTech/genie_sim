# geniesim_world — Agent Development Guide 🌍

Release version: the local `VERSION` links to the repository root `VERSION`
(**3.5.0**). Wheel metadata and runtime `__version__` use that release value.
`MANIFEST.in` includes `VERSION` in source archives so they can be rebuilt.

Multimodal spatial world model: turns a single equirectangular panorama
into a photorealistic, explorable 3D world (Gaussian splat / depth / cube
geometry) via the **PanoRecon** pipeline (SHARP + DA360 fusion). Owns the
`geniesim_world` console script.

Source: [source/geniesim_world/](.)
License: [Mozilla Public License Version 2.0](LICENSE)
Public intro: [README.md](README.md)
Skills: [skills/](skills/)
Paper: https://arxiv.org/abs/2604.07105

> **Maintenance contract** — when you change CLI flags, add/remove a
> subcommand, alter the env var registry (`GENIESIM_*`), swap an
> external dep (ml-sharp / DA360 / Real-ESRGAN), or change the depth
> normalisation contract, **update this file in the same diff**.
> Agents read this as the source of truth.

---

## 1. Why this distribution exists

`geniesim_world` is the platform's **scene-from-photo** front-end. It
sits outside the `geniesim bootstrap` install path because its CUDA / ML
deps (PyTorch + SHARP + DA360) collide with the rest of the stack. Users
spin up a dedicated conda env, generate a world, and consume the
`.ply` / `.gsp` Gaussian output via `geniesim_world`'s own tooling.

Loading the generated world into a `scene_*.yaml` for `geniesim_ros` is not
supported. The [platform roadmap](../../docs/ROADMAP.md) tracks proposed
integration. Document the package's current exports and debug tools.

---

## 2. Layout

```
source/geniesim_world/
├── AGENTS.md              ← this file
├── README.md              ← user-facing intro + install
├── pyproject.toml         ← build config
├── setup.py               ← entry_points: geniesim_world=…cli_pano:cli
├── requirements-cu128.txt ← tested on RTX 5090 / CUDA 12.8
├── skills/
│   └── generate-world/SKILL.md
└── src/geniesim_world/
    ├── cli_pano.py        ← Click group; `create` subcommand
    ├── cli_debug.py       ← `debug` subcommand (EXR / depth diff)
    ├── predict_run.py     ← SHARP wrapper, predictor orchestration
    ├── predictor.py       ← `DepthGuidedRGBGaussianPredictor`
    ├── gaussian_ops.py    ← frustum culling, splat ops
    └── utils/
        ├── cubes.py       ← cubemap split / merge / face I/O
        ├── da360_depth.py ← DA360 depth inference + relative-depth normalisation
        ├── merge.py       ← PLY post-processing + merge
        └── pano_fuse.py   ← cubemap fusion
```

---

## 3. CLI surface

Console script: `geniesim_world` (Click group).

| Subcommand | Purpose |
|---|---|
| `geniesim_world create` | Panorama → 3D world (the primary one) |
| `geniesim_world debug` | EXR PNG + depth diff utilities |

Common flags on `create`:

| Flag | Effect |
|---|---|
| `--panorama PATH` | Equirectangular RGB input (required) |
| `--work-dir PATH` | Output directory (default: cwd / `runs/<stamp>/`) |
| `--da360-root DIR` | Override `external/DA360/` location |
| `--da360-checkpoint PATH` | Override DA360 checkpoint location |
| `--checkpoint-path PATH` | SHARP checkpoint override |
| `--depth-max FLOAT` | Clip predicted depth at N metres |
| `--device STR` | Torch device (default: `default` → cuda:0 if available) |
| `--no-depth-gt-init` | Stock SHARP (ignore `depth_gt` override) |
| `--super-sample` | Run Real-ESRGAN before fusing (needs the optional binary) |

`geniesim_world --help` / `geniesim_world create --help` are the
authoritative live docs.

---

## 4. External dependencies

Loaded via `sys.path` (not pip), so the repo layout matters:

| Dep | Location | How acquired |
|---|---|---|
| `ml-sharp` | `external/ml-sharp/` | `git clone https://github.com/apple/ml-sharp.git` |
| `DA360` repo + checkpoint | `external/DA360/` + `DA360_large.pth` | clone + download checkpoint separately |
| Real-ESRGAN (optional) | `external/realesrgan-ncnn-vulkan-…/` | download binary release |

Override paths via `GENIESIM_DA360_CHECKPOINT`, `--da360-root`, etc.

---

## 5. Environment variables

| Var | Effect |
|---|---|
| `GENIESIM_DA360_CHECKPOINT` | Absolute path to DA360 checkpoint (overrides `external/DA360/DA360_large.pth`) |
| `GENIESIM_REALESRGAN_BIN` | Absolute path to Real-ESRGAN binary |

All env vars are namespaced under `GENIESIM_*`. Do not introduce
non-namespaced vars or legacy `GENIE_*` names (those are deprecated).

---

## 6. Architectural rules (do not violate)

1. **Keep DA360 as a runtime sys.path dep.** Do not force a pip-installable DA360 — upstream doesn't ship one cleanly, and the checkpoint path is fluid.
2. **Depth normalisation contract** (see `da360_depth.py`):
   - `da360` / `both` modes → DA360 **relative** depth (min-normalized), clamped `[1e-4, 1e4]` (matches SHARP disparity band).
   - `sharp` mode → SHARP **meters**.
   - `fuse` mode → DA360 median-aligned to SHARP in `cli_pano`, then fused → meters.
3. **`--depth-max` semantics:**
   - When set → maps clip to `[0, depth_max]`; inverse-depth resize uses `(eps < d < depth_max)` for support.
   - When unset → EXR depth is **≥ 0** only (no global far clip).
4. **Depth resize:** `resize_depth_map` treats unknown support as 0. Cube EXR refine pass clears masked depth. Equirect→cube uses linear interp + optional `--cube-depth-refine`.
5. **CLI compat is a contract.** Preserve flag names + semantics across versions unless the user asks for breaking changes. README examples are the regression suite.
6. **Use current identifiers.** The package is `geniesim_world`; configuration uses `GENIESIM_*` names and the current CLI surface.

---

## 7. Skills

| Skill | Trigger |
|---|---|
| [generate-world](skills/generate-world/SKILL.md) | "generate a world", "pano to 3D", "PanoRecon", "make a scene from a photo" |

---

## 8. Verification recipes

| Goal | Command |
|---|---|
| Confirm CLI installs | `geniesim_world --help` (after `pip install -e .` in this dir) |
| Smoke-test `create` | `geniesim_world create --panorama assets/sample.png --work-dir /tmp/test` |
| Inspect a previous run's depth | `geniesim_world debug …` (see `cli_debug.py --help`) |
| Lint check | `black --line-length 120 --check src/` |

---

## 9. Do not

- Don't add this package to `geniesim bootstrap`'s default install set — its CUDA / SHARP / DA360 deps belong in a dedicated conda env, not the shared stack.
- Don't update README example flags without re-running the example — flag drift between docs and reality is the #1 reported bug here.
- Don't bypass `cli_pano.py` to call `predict_run.py` directly from a user-facing skill. The CLI is the public surface.
- Keep unsupported text/sparse-image generation and RT Engine integration in the [platform roadmap](../../docs/ROADMAP.md).

---

## 📚 Related Docs

- [`README.md`](README.md) — user-facing intro + install
- [`skills/generate-world/SKILL.md`](skills/generate-world/SKILL.md) — agent-driven workflow
- [`../AGENTS.md`](../AGENTS.md) — repository module map
- [`../../AGENTS.md`](../../AGENTS.md) — repo root agent guide (boot sequence)
