# Genie Sim World 🌍

`Genie Sim World` creates immersive explorable 3D world for robot manipulation task within seconds.
This project implements the methods from ***Genie Sim PanoRecon** - Fast Immersive Scene Generation from a Single Equirectangular Panorama*

![image.png](./assets/project.png)
<div align="center">
  <a href="https://arxiv.org/abs/2604.07105" style="text-decoration:none;">
    <img src="https://img.shields.io/badge/arXiv-2604.07105-red.svg?logo=arxiv&logoColor=white" alt="arXiv Paper: 2604.07105">
  </a>
  <a href="https://github.com/AgibotTech/genie_sim/source/geniesim_world">
    <img src="https://img.shields.io/badge/GitHub-grey?logo=GitHub" alt="GitHub">
  </a>
</div>

---


## ✨ Why this project

- 🚀 Fast end-to-end pipeline (PanoRecon within seconds)
- 🧠 Expand ml-sharp's ability to panorama image, creating photorealistic 360 consistent world
- 🖼️ Generate diverse environments from text or image (*COMING SOON*) at low cost
- 🧩 Seamlessly integrated into geniesim, providing SimReady scene assets (*COMING SOON*) for robot tasks

## 🗂️ Expected Layout

Keep `geniesim_world/` and `external/` side by side:

```text
source/
  geniesim_world/            # this project
  external/
    ml-sharp/
    DA360/
    realesrgan-ncnn-vulkan-20220424-ubuntu/  # optional
```

## 📦 Prepare Dependencies

Clone required external repos:

```bash
cd external
external$ git clone https://github.com/apple/ml-sharp.git
external$ git clone https://github.com/DepthAnything/DA360.git
```

Prepare DA360 checkpoint:

- put it at `external/DA360/DA360_large.pth`, or
- pass `--da360-checkpoint`, or
- set `GENIESIM_DA360_CHECKPOINT`

(Optional) Real-ESRGAN binary for `--super-sample`:

> NOTE: This improves the visual effect but introduces fake textures into the image

- download from https://github.com/xinntao/Real-ESRGAN/releases
- place executable at
  `external/realesrgan-ncnn-vulkan-20220424-ubuntu/realesrgan-ncnn-vulkan`
- or set `GENIESIM_REALESRGAN_BIN=/abs/path/to/realesrgan-ncnn-vulkan`

## ⚙️ Install

Create environment and pin packaging tools:

```bash
conda create -n geniesim_world python=3.11 -y
conda activate geniesim_world
pip install --upgrade "pip==24.0" "setuptools==69.5.1" "wheel==0.43.0"
```

Install project:

```bash
cd /path/to/source/geniesim_world
# we tested cuda12.8 on RTX5090 32G, for other GPUs please switch to the workable version then install this package
geniesim_world$ pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements-cu128.txt
geniesim_world$ pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e .
```

Quick sanity check:

```bash
python -c "import sharp"
```

## 🚀 Run World Generator Pipeline

Basic Dataflow:

`ERP Image` -> `DA360 Depth` -> `Cubemap Faces` -> `SHARP per Face` -> `merged PLY`

Basic run:

```bash
geniesim_world create -p <input-pano-image>.png -o .
```

Run with super-sampling:

```bash
geniesim_world create -p <input-pano-image>.png -o . --super-sample realesrgan-x4plus
```

## 📝 Run 360 Panorama Pipeline (DiT360 + ComfyUI)

The ERP panorama images can be collected from reaf life or from AI models

If you do not have an ERP panorama yet, you can first generate one from text prompts with ComfyUI + DiT360, then feed that panorama into `geniesim_world create`

Brief DiT360 pipeline:

1. Install ComfyUI (base workflow runtime).
2. Install `ComfyUI-DiT360` custom nodes.
3. Load FLUX.1-dev + DiT360 LoRA in ComfyUI. (*You should accept the license of the FLUX.1-dev before downloading*)
4. Generate a 2:1 equirectangular panorama (`.png`).
5. Run `geniesim_world create -p <your_pano>.png -o .` for 3D reconstruction.

Useful links:

- ComfyUI installation guide: https://github.com/comfyanonymous/ComfyUI#installing
- ComfyUI-DiT360 repository: https://github.com/cedarconnor/ComfyUI-DiT360.git

## 🧰 Useful Options

- `--skip-sharp` : stop after DA360 + cubemap export
- `--skip-merge` : keep only per-face PLY outputs
- `--no-ndc-frustum-mask` : do not drop SHARP Gaussians outside a tight NDC xy frustum; **sky-heavy faces** can look empty / “disappeared” without this (try it first if a face PLY is missing).
- `--no-ply-frustum-cull` : keep PLY Gaussians even when their mean falls **outside** the widened cube-face pyramid (default **on**: cull before merge).
- `--ply-frustum-fov-deg` (default **91**) : full FOV used for that cull vs a **90°** nominal cube face; ``tan(fov/2)/tan(45°)`` widens the acceptance slightly so seams are less harsh than a hard 90° cut.
- `--ply-frustum-margin-px` (default **0**) : optional pixel inset on the reference image when computing edge tangents (tightens the cull).
- `--pano-depth-engine {da360,sharp,both,fuse}` (default **da360**) —
  - `da360`: old behavior, DA360 is depth source
  - `sharp`: DA360 skipped, SHARP pano depth is depth source
  - `both`: DA360 is depth source, also run SHARP pano depth and save SHARP-vs-DA360 diff
  - `fuse`: run both DA360 + SHARP pano depth; **DA360 relative depth is median-ratio aligned to SHARP meters** before pyramid fusion, then the fused map (meters) drives cubemap + SHARP
- `--pano-depth-fuse-method {laplacian,gaussian}` (default **laplacian**) — fuse-engine method.
- `--pano-depth-fuse-sharp-weight` (default **0.6**) — fuse-engine SHARP contribution (0~1).
- `--pano-depth-fuse-levels` (default **4**) — fuse-engine pyramid levels for laplacian mode.
- `--pano-depth-engine-debug` : save extra engine debug artifacts (e.g. DA360 exported files when DA360 is active)
- `--sharp-ply-radius-cap` (default **40**) — when exporting each SHARP face `.ply`, if `|xyz|` exceeds this threshold `T`, apply log compression to the radius (no hard cutoff) and scale splat sizes accordingly; set **0** to disable.
- `--pano-depth-resize-space {inverse,depth}` (default **inverse**) — float EXR depth is resized in **inverse depth (disparity)** space by default (metrically saner than interpolating raw depth); use `depth` to interpolate depth values directly (legacy / comparison).
- `--pano-depth-interpolation {linear,cubic,lanczos4}` (default **linear**) — OpenCV interpolation used when resizing the chosen quantity (disparity or depth) to match ERP size (`linear` is the most geometry-stable default; try `cubic/lanczos4` only if you prefer sharper but riskier edges).
- `--cube-face-size` (default **1536**) — cubemap face resolution; keep aligned with SHARP’s internal face size.
- `--cube-depth-refine {none,bilateral}` (default **bilateral**) — light filter on inverse depth per cube face to soften alias / stair-steps (disable with `none` if you need raw warped depth). Invalid / unmasked pixels are encoded as **NaN** in `*_depth.exr` (not filled to a synthetic far plane), so downstream can detect “holes” reliably.
- `--depth-max` (optional) — when set, caps **metric** depth (meters), fuse/align valid masks, robust stats, and fixed-range debug PNGs. **Default: no global cap** (avoids stacking geometry at one far bound). Legacy **uint16 PNG** cube depth still requires an explicit scale — pass `--depth-max` or use **EXR** depth inputs for SHARP.

**Depth units:** With **`--pano-depth-engine da360`** or **`both`**, pano and cube **`.exr`** depths from DA360 are **min-normalized relative** values (not SI meters). With **`sharp`** or **`fuse`**, pano/cube depths are **meters**, **≥ 0** in EXR; if **`--depth-max`** is set they are also clipped to **`[0, depth_max]`**. Invalid / unknown regions are encoded as **NaN** holes. SHARP consumes cube EXRs in the same units as written for that run.

**DA360 is not scaled per cube face:** the network gives one **continuous** equirect **min-normalized relative** depth map. After native inference and after ERP resize, values are **clamped to `[1e-4, 1e4]`** (same numeric band as SHARP’s `disparity.clamp(min=1e-4, max=1e4)`). No global affine to SI meters. **`gen_cubes` only samples** that field onto six faces. For **metric** pano depth (meters), use **`--pano-depth-engine sharp`** or **`fuse`**; use **`--depth-max`** only when you want an explicit far clip or fixed PNG visualization scale.

**Face PLY empty / “disappeared” (large sky):** depth-guided SHARP applies `apply_ndc_frustum_mask`, which can remove most splats on uniform sky. Use **`--no-ndc-frustum-mask`**; defaults also use wider NDC margins (2.0) than stock (1.0).

## 🐛 Debug a finished run (`geniesim_world debug`)

After `geniesim_world create`, visualize depth EXRs and compare DA360 vs SHARP per face:

```bash
geniesim_world debug --dir ./output/mansion_01
```

All outputs are **flat under `<run_dir>/debug/`** (no subfolders) for easy side-by-side comparison:

- **`diff_da360_vs_sharp_{face}.exr`** / **`.png`** / **`_color.png`** — |DA360 − SHARP| per face (`.png` is linear uint16; **`_color.png`** uses the same **matplotlib turbo** + `val_max` scheme as `sharp predict`, default **50 m** via `METRIC_DEPTH_MAX_CLAMP_METER`, or `--depth-max` when set).
- **`<flattened>_…png`** + **`<flattened>_…_color.png`** — one pair per **pipeline** `.exr` (paths like `cubes/front/front_depth.exr` → `cubes_front_front_depth.png` + `cubes_front_front_depth_color.png`; excluding files under `debug/`).
- **`depth_diff_summary.txt`** — per-face MAE / max / RMSE.

More verbose logs:

```bash
geniesim_world debug --dir ./output/mansion_01 -v
```

By default, **`debug`** scales each PNG from that file’s **p99 vs max** (no global cap). To pin visualization to a fixed meter range:

```bash
geniesim_world debug --dir ./output/mansion_01 --depth-max 80
```

## 🩺 Troubleshooting

1. `Set --da360-root ... or export GENIESIM_DA360_ROOT`
   - DA360 repo is not found.
   - Ensure your layout matches:
     `source/geniesim_world` and `source/external/DA360`.
   - Or pass `--da360-root /abs/path/to/DA360`.

2. `DA360 checkpoint not found`
   - Put the checkpoint at `external/DA360/DA360_large.pth`, or
   - pass `--da360-checkpoint /abs/path/to/DA360_large.pth`, or
   - set `GENIESIM_DA360_CHECKPOINT`.

3. `Could not find realesrgan-ncnn-vulkan binary` (when using `--super-sample`)
   - Download Real-ESRGAN ncnn binary release.
   - Place it at
     `external/realesrgan-ncnn-vulkan-20220424-ubuntu/realesrgan-ncnn-vulkan`,
     or set `GENIESIM_REALESRGAN_BIN=/abs/path/to/realesrgan-ncnn-vulkan`.
   - Make sure the binary is executable (`chmod +x <binary>`).

4. EXR read/write failures (OpenCV)
   - Install `opencv-python-headless` from requirements.
   - Some OpenCV builds may not include OpenEXR support; switch to a compatible build/environment.
   - If EXR fails, first verify with a fresh environment and the pinned dependency set in this README.

5. CUDA / PyTorch mismatch
   - This project is tested with PyTorch 2.8.0 + CUDA 12.8 wheels.
   - Reinstall with:
     `pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements-cu128.txt`
   - Then reinstall this package with the same extra index URL.

## 📁 Output Overview

For input `foo.png` with `-o .`, outputs go to `./foo/`:

- `input.png`
- `input_depth.exr` — float panorama depth resized (anti-aliased) to current input RGB resolution; content follows `--pano-depth-engine` (`da360` or `sharp`).
- optional `<stem>.4x.png`
- `cubes/` — per-face RGB, DA360 depth (`<face>_depth.exr`), intr/extr, masks
- `sharp/` — per face: `<face>_image.ply` (SHARP Gaussians) and `<face>_image_sharp_depth.exr` (SHARP monodepth after alignment, same H×W as the cube face and DA360 `*_depth.exr`)
- optional pano SHARP-vs-DA360 comparison (always in `--pano-depth-engine both`; also in `sharp` mode without diff):
  - `sharp_pano_depth.exr` (same size as upsampled panorama / `input_depth.exr`)
  - `sharp_vs_da360_pano_absdiff.exr` (only when DA360 is active)
  - `sharp_vs_da360_pano_absdiff.png` (only when DA360 is active)
- optional fused pano depth (`--pano-depth-engine fuse`):
  - `fused_pano_depth.exr`
- `debug/` — only after `geniesim_world debug`: flat `.exr`/`.png` exports + `depth_diff_summary.txt`
- `merged_gaussians.ply` (unless `--skip-merge`)
- optional DA360 artifacts (`*.npy`, `*.png`, `*.exr`, `*.jpg`, `*.ply`)

## 🔧 Environment Variables

- `GENIESIM_DA360_ROOT`
- `GENIESIM_DA360_CHECKPOINT`
- `GENIESIM_REALESRGAN_BIN`

## 📄 License and Third-Party Notes

- `LICENSE` applies to this repository's original source code only.
- This repository does **not** relicense third-party repositories, binaries, or other assets.
- Genie Sim World is built upon several great third-party projects:
  - `ml-sharp` (Apple) - Used for 3D reconstruction
  - `DA360` (Insta360 Research Team) - Used for panorama depth estimation
  - Real-ESRGAN (optional) - Command-line tool wrapper for super-sampling functionality
- All third-party components remain under their own licenses and terms.
- Check `THIRD_PARTY_LICENSES` for detailed license information.

Please consider citing our work either way below if it helps your research.

```
@misc{li2026geniesimpanoreconfast,
      title={Genie Sim PanoRecon: Fast Immersive Scene Generation from Single-View Panorama}, 
      author={Zhijun Li and Yongxin Su and Di Yang and Jichao Wang and Zheyuan Xing and Qian Wang and Maoqing Yao},
      year={2026},
      eprint={2604.07105},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2604.07105}, 
}
```
