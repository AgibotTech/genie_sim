# geniesim_benchmark — Benchmark tasks, scoring, LLM eval 🧪

Declarative task configs + a runtime that loads a scene, drives a
robot, evaluates a policy, and records scores. The canonical entry is
the `geniesim benchmark` CLI verb (owned by
[`geniesim_cli`](../geniesim_cli/)).

License: [Mozilla Public License Version 2.0](LICENSE)
Agent doc: see [`../../.agent/geniesim_benchmark.md`](../../.agent/geniesim_benchmark.md)
Skills: [`skills/`](skills/)

---

## 📦 Install

```bash
pip install -e source/geniesim_benchmark/
```

Pulled in automatically by `geniesim bootstrap`. Heavy runtime deps
(Isaac Sim, MuJoCo, open3d, …) come from this package.

---

## 🛠️ What you can do

### Run a task against an inference server

```bash
geniesim benchmark run g2op_if_pick_block_color \
  --infer-host=<IP>:8999
```

### Probe an inference server before sinking minutes into a sim launch

```bash
geniesim benchmark check-inference \
  --infer-host=<IP>:8999 --arch=corobot
```

### Feed history image observations (server-driven, corobot)

The `corobot` policy can attach a rolling buffer of past head-camera frames
to each inference request under `params.history`, letting temporal models see
what happened while the previous action chunk was replaying. This is **opt-in
per response and controlled by the server** — nothing to configure on the
benchmark side:

- The server returns an integer `hist_frame_interval` in its `result`.
  `> 0` enables capture and sets the sampling stride (a head frame is grabbed
  every N chunk-replay steps); `0` / omitted disables it.
- The captured frames ride along on the *next* request as
  `params.history = {"interval": N, "images": [...]}`. The first request of an
  episode (before the server has opted in) carries no history.

A server that never returns the field behaves exactly as before, so this is
backward compatible with existing inference servers.

### Discover tasks

```bash
geniesim benchmark categories         # show category counts
geniesim benchmark robots             # show robot counts
geniesim benchmark list --robot=g2op --category=instruction_following
```

### Batch-evaluate a sweep

```bash
geniesim benchmark batch --category=instruction_following --robot=g2op
```

### Convert collected datasets between formats

The benchmark stack ships dataset utilities under
`geniesim_benchmark.dataset.*`. The first converter goes from
**agibot v1 → LeRobot v2.1** (parquet + HEVC/PNG-encoded MP4s):

```bash
geniesim dataset convert agibot-to-lerobot \
  --agibot-dir ./agibot \
  --output-dir ./lerobot_out
```

The `--agibot-dir` argument accepts either a single-episode dir
(contains `aligned_joints.h5` directly) or a parent dir of multiple
episode subdirs — auto-detected at runtime. Pass
`--lerobot-ref-dir <path>` to fill missing fisheye / head_back
extrinsic columns from a reference dataset; omit it to leave those
columns empty. Requires **ffmpeg on `PATH`** (RGB → HEVC, depth → PNG).

---

## 🤖 Skills

| Skill | Purpose |
|---|---|
| [run-benchmark](skills/run-benchmark/SKILL.md) | Launch a benchmark task locally against a user-provided inference server |
| [check-inference](skills/check-inference/SKILL.md) | Probe a model inference WebSocket server and validate the response |

---

## 📂 Layout

```
src/geniesim_benchmark/
├── app/app.py            # runtime entry, called by `geniesim benchmark run`
├── config/               # *.yaml task configs (the work-list)
├── dataset/              # dataset utilities (format conversion, …)
│   └── convert/
│       └── agibot_to_lerobot.py   # public convert_agibot_to_lerobot() + convert_cli()
└── …
```

`config/*.yaml` is the source of truth for what's a benchmark task —
robot, scene, policy, scoring rule. The runtime is config-driven; new
tasks land as new yaml files, not new code.

`dataset/` is the home for off-line data utilities (format converters,
schema inspectors). Each converter exposes a plain-Python API plus a
`convert_cli(argv)` wrapper used by the `geniesim dataset convert …`
dispatcher — `argparse` only lives in the wrapper, the API is usable
from notebooks.

---

## 🔗 Pointers

- 🗺️ Module map: [`../README.md`](../README.md)
- 🏠 Repo root: [`../../README.md`](../../README.md)
- 🤖 Agent dispatcher: [`../../.agent/geniesim_benchmark.md`](../../.agent/geniesim_benchmark.md)
- 🏆 Leaderboard / public scores: [`../../README.md`](../../README.md) § Genie Sim Benchmark Leaderboard
