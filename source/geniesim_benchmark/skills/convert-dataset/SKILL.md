---
name: convert-dataset
description: >
  Convert robot trajectory datasets between formats — currently agibot v1 →
  LeRobot v2.1 (parquet + HEVC/PNG-encoded MP4). Uses the
  `geniesim dataset convert agibot-to-lerobot` CLI verb, which wraps the
  `geniesim_benchmark.dataset.convert.agibot_to_lerobot` Python API.
  Trigger: When the user asks to "convert agibot to lerobot",
  "convert dataset", "transcode trajectory data", "build a LeRobot dataset",
  "把 agibot 数据转成 lerobot", or provides an agibot episode dir / batch dir
  and wants the LeRobot v2.1 layout (`data/chunk-*/*.parquet` + `videos/…/*.mp4`
  + `meta/`).
license: MPL-2.0
metadata:
  author: genie-sim
  version: "1.0"
prerequisites:
  - geniesim_cli:fresh-machine-setup   # see source/geniesim_cli/AGENTS.md § 0
inputs:
  - name: agibot_dir
    desc: agibot episode dir (single) or parent dir containing multiple episode subdirs
    required: true
  - name: output_dir
    desc: Destination for the LeRobot dataset
    required: true
  - name: lerobot_ref_dir
    desc: Reference LeRobot dataset to fill missing fisheye / head_back extrinsic columns from
    required: false
  - name: fps
    desc: Video frame rate
    required: false
    default: "30"
outputs:
  - desc: "LeRobot v2.1 dataset at output_dir (data/chunk-NNN/episode_*.parquet, videos/chunk-NNN/<key>/episode_*.mp4, meta/info.json + tasks.jsonl + episodes.jsonl + episodes_stats.jsonl)"
---

## When to Use

- User has agibot v1 trajectory data and wants the LeRobot v2.1 layout (e.g.
  to feed an upstream LeRobot training pipeline, or compare against an
  existing LeRobot reference).
- User provides a parent dir of multiple episode subdirs — the converter
  auto-detects single vs batch from layout.

Do **not** use for:
- Just running a benchmark task → `run-benchmark` skill.
- Probing an inference server → `check-inference` skill.

## Prerequisites

- `geniesim_benchmark` installed (tier-1 peer — comes with `geniesim
  bootstrap`).
- **`ffmpeg` on `PATH`.** Used for both RGB encoding (HEVC / libx265) and
  depth encoding (PNG / gray16le). The converter pre-flights `ffmpeg`; if
  missing it surfaces the install hint (`sudo apt install ffmpeg` on
  Debian/Ubuntu, `brew install ffmpeg` on macOS).
- `h5py`, `numpy`, `pyarrow` are declared deps of `geniesim_benchmark`;
  nothing to install separately.

## Workflow

### Single episode

```bash
geniesim dataset convert agibot-to-lerobot \
  --agibot-dir ./agibot/episode_000 \
  --output-dir ./lerobot_out
```

`--agibot-dir` is treated as a **single episode** iff it contains
`aligned_joints.h5` directly. The resulting dataset has
`total_episodes = 1`.

### Batch (auto-detect)

```bash
geniesim dataset convert agibot-to-lerobot \
  --agibot-dir ./agibot \
  --output-dir ./lerobot_out
```

When `--agibot-dir` does **not** contain `aligned_joints.h5` directly, the
converter scans for episode subdirectories (each must contain
`aligned_joints.h5`). Episodes are indexed in sorted order of their
directory name.

### With a reference LeRobot dataset

```bash
geniesim dataset convert agibot-to-lerobot \
  --agibot-dir ./agibot \
  --output-dir ./lerobot_out \
  --lerobot-ref-dir /path/to/reference/lerobot_dataset
```

When the agibot episode is missing the fisheye / head_back extrinsics
(common — those cameras aren't on every rig), the converter pulls the
missing columns from
`<lerobot-ref-dir>/data/chunk-000/episode_000000.parquet`. Omit
`--lerobot-ref-dir` to leave those columns empty.

### Tune FPS

```bash
--fps 60   # default is 30
```

`--fps` is passed to ffmpeg (`-r`, `-framerate`) **and** baked into the
v2.1 timestamps (frame_index / fps). The `meta/info.json` always records
`fps: 30` regardless — match this if you need consistency across a
collection.

## Programmatic use

The same conversion is callable from Python:

```python
from pathlib import Path
from geniesim_benchmark.dataset.convert.agibot_to_lerobot import convert_agibot_to_lerobot

manifest = convert_agibot_to_lerobot(
    agibot_dir=Path("./agibot"),
    output_dir=Path("./lerobot_out"),
    lerobot_ref_dir=Path("./ref_lerobot"),  # optional
    fps=30.0,
)
print(manifest["total_episodes"], manifest["total_frames"])
```

The Python API raises `RuntimeError` for missing `ffmpeg`, missing heavy
deps, or no detected episodes. The CLI wrapper catches those and prints
the error to stderr with exit code `1`.

## Verify it worked

```bash
ls -R lerobot_out/
# → data/chunk-000/episode_000000.parquet, ...
# → videos/chunk-000/{top_head,hand_left,hand_right,top_head_depth,...}/episode_*.mp4
# → meta/{info.json,tasks.jsonl,episodes.jsonl,episodes_stats.jsonl}

python3 -c "
import pyarrow.parquet as pq
t = pq.read_table('lerobot_out/data/chunk-000/episode_000000.parquet')
print(t.schema)
print('rows:', t.num_rows)
"
```

`observation.state` must be a `fixed_size_list<float32, 159>` and `action`
a `fixed_size_list<float32, 40>` — those widths are part of the v2.1
contract and the converter writes them literally. The `index` column is
globally contiguous across the batch, while `frame_index` restarts at zero for
each episode. Per-camera statistics are sampled from the RGB frames using the
LeRobot v2.1 sampling heuristic and stored in `episodes_stats.jsonl`.

## Troubleshooting

- **`ffmpeg is not on PATH`** — install `ffmpeg`; see Prerequisites.
- **`No episode directories found`** — `--agibot-dir` neither contains
  `aligned_joints.h5` directly nor has any subdir containing one. Re-check
  the path; common mistake is pointing at a parent that's one level too
  high.
- **`ERROR encoding <key>: …`** — ffmpeg printed something to stderr.
  Common causes: missing input frames (`camera/<N>/<stem>.jpg` glob is
  sparse), unsupported codec (older ffmpeg without `libx265` — install
  `ffmpeg` with HEVC support, e.g. the `nasm`/`libx265` variant), or write
  permission errors on `--output-dir`.
- **Stats look wrong** — numeric statistics read back the parquet rows and
  video statistics sample the raw RGB frames. If the parquet wasn't written,
  the stats entry is `{}`; a missing RGB frame raises a conversion error.
  Inspect the parquet and the corresponding `camera/<frame>/..._color.jpg`
  files first.

## Resources

- Logic: [agibot_to_lerobot.py](../../src/geniesim_benchmark/dataset/convert/agibot_to_lerobot.py)
- CLI dispatcher: [geniesim_cli/commands/dataset.py](../../../geniesim_cli/src/geniesim_cli/commands/dataset.py)
- LeRobot v2.1 reference layout: HuggingFace `lerobot` repo (search for
  `info.json` `codebase_version: v2.1`).
