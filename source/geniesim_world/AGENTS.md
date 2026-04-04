# geniesim_world Agent Guide 🤖

This file is for coding agents only.
User-facing documentation belongs in `README.md`.

## 🎯 Scope

- Package root: `geniesim_world/src/geniesim_world/`
- Main CLI: `geniesim_world create`, `geniesim_world debug`
- Runtime external dependencies:
  - `../external/ml-sharp`
  - `../external/DA360`
  - optional Real-ESRGAN binary

## 🧭 Routing Map

- CLI args, orchestration, `create` + `debug` registration -> `src/geniesim_world/cli_pano.py`
- `debug` subcommand (EXR PNG + depth diff) -> `src/geniesim_world/cli_debug.py`
- SHARP wrapper and predictor interaction -> `src/geniesim_world/predict_run.py`, `src/geniesim_world/predictor.py` (`DepthGuidedRGBGaussianPredictor`: `depth_gt` merge for init), `gaussian_ops.cull_gaussians_outside_pinhole_frustum` (91° ref frustum; `--ply-frustum-fov-deg`; `--no-ply-frustum-cull`; `--no-ndc-frustum-mask`)
- Cubemap split/merge and face I/O -> `src/geniesim_world/utils/cubes.py`
- DA360 depth inference and DA360-like exports -> `src/geniesim_world/utils/da360_depth.py` (min-normalized relative depth; clamp `[1e-4, 1e4]` like SHARP disparity band; no `depth_max` in this module)
- PLY post-processing and merge -> `src/geniesim_world/utils/merge.py`
- Packaging metadata -> `setup.py`, `pyproject.toml`, `requirements-cu128.txt`
- User docs -> `README.md`

## ⚙️ Operational Rules

1. Keep env var names under `GENIESIM_*`.
2. Keep DA360 as a runtime repo dependency (loaded via `sys.path`); do not force pip-installable DA360.
3. Cubemap `*_depth.exr`: **`da360` / `both`** use DA360 **relative** depth (min-normalized). **`sharp`** uses SHARP meters. **`fuse`** median-aligns DA360 to SHARP in `cli_pano` then fuses to meters. When CLI **`--depth-max`** is set, maps clip to **`[0, depth_max]`**; inverse-depth resize uses `(eps < d < depth_max)` for support. When unset, EXR depth is **≥ 0** only (no global far clip). Depth resize: `resize_depth_map` (unknown support → 0); cube EXR: refine + **mask** clears depth; equirect→cube: linear + optional `--cube-depth-refine`.
4. Preserve CLI compatibility unless user asks for breaking changes.
5. When behavior changes, update `README.md` in the same task.

## ✅ Completion Checklist

- `--help` text reflects current behavior.
- README examples still match real flags/env vars.
- No stale legacy identifiers (`genie_sim_world_model`, `GENIE_*`, old CLI names).
- If packaging changed, re-check editable install assumptions.

## 📚 Related Docs

- `README.md` (user-facing)
- `../AGENTS.md` (repo-wide routing)
- `../external/AGENTS.md` (external capability registry)
