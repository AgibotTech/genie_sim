---
name: challenge-download-assets
description: >
  Query, download, and verify GenieSim simulation assets by task or track with
  geniesim assets. Supports resumable transfers, ModelScope/HuggingFace source
  selection, and container asset mounts. Use for scenes, robots, and benchmark
  runtime assets; use challenge-download-datasets for LeRobot training data.
---

# Download GenieSim simulation assets

First obtain the **simulator code**, install its CLI, then download the assets.
This skill supplies instructions; the downloader and versioned asset index come
from the simulator code.

Use `list` and `plan` to identify the required tasks, download size, and destination.
Then download and run `verify` with the same selectors. Asset downloads require no
platform login or evaluation job. A request to download assets does not require
starting a container or simulation.

## 1. Obtain the simulator code and install the CLI

Clone the simulator code, or use the source delivered by the code provider:

```bash
git clone https://github.com/AgibotTech/genie_sim.git
cd genie_sim
```

If the source is already available, enter its repository root instead of cloning
again. Install the CLI in the Python environment used for downloading:

```bash
python3 -m pip install -e source/geniesim_cli

# List indexed tasks without downloading anything.
geniesim assets list
```

Skip installation only if this environment already uses the CLI from that source.
Copying this skill does not update the installed CLI or its index. If
`geniesim assets` is unavailable, update the simulator code as instructed by its
provider and reinstall the CLI.

Downloading needs only the lightweight CLI; Isaac Sim, ROS, USD, and the ModelScope
SDK are not prerequisites. To run simulation afterward, prepare the container and
runtime dependencies following the simulator code's setup instructions.
`geniesim bootstrap` installs or repairs required modules; it does not download
these assets. Do not suggest `pip install geniesim` or `pip install geniesim_assets`.

## 2. Inspect tasks and download size

```bash
# List every indexed task, or filter by robot/category prefix.
geniesim assets list
geniesim assets list g2op_if

# Inspect the file count and size without downloading assets.
geniesim assets plan g2op_if

# Optionally save the complete selected file manifest.
geniesim assets plan g2op_if --output g2op-if-assets.json
```

Multiple selectors form a union of dependencies:

| Selector | Examples |
| --- | --- |
| Exact task | `g2op_if_pick_block_color`, optionally with `.yaml` |
| Robot/category prefix | `g2op_if`, `g2op_crsemb` |
| Task category | `manip`, `spatial`, `dexmanip`, `finemanip`, `wbc`, `crsemb`, `robust`, `sandbox` |

## 3. Download and verify

```bash
# Download all indexed g2op_if tasks.
geniesim assets download g2op_if --dest "$HOME/sim/assets-g2op-if" --workers 8

# Download one task into a separate directory.
geniesim assets download g2op_if_pick_block_color --dest "$HOME/sim/assets-pick-block"

# Select several categories together, using their own destination.
geniesim assets download manip spatial --dest "$HOME/sim/assets-manip-spatial"

# Check file sizes and SHA256 hashes offline.
geniesim assets verify g2op_if --dest "$HOME/sim/assets-g2op-if"
```

### Commands by track

Choose only the requested tracks. Each command downloads all indexed tasks for
that track into a separate directory; instruction-following uses `g2op_if`:

```bash
geniesim assets download g2op_if --dest "$HOME/sim/assets-g2op-if"
geniesim assets download manip --dest "$HOME/sim/assets-manip"
geniesim assets download spatial --dest "$HOME/sim/assets-spatial"
geniesim assets download dexmanip --dest "$HOME/sim/assets-dexmanip"
geniesim assets download finemanip --dest "$HOME/sim/assets-finemanip"
geniesim assets download wbc --dest "$HOME/sim/assets-wbc"
geniesim assets download crsemb --dest "$HOME/sim/assets-crsemb"
geniesim assets download robust --dest "$HOME/sim/assets-robust"
geniesim assets download sandbox --dest "$HOME/sim/assets-sandbox"
```

`s2r` is not in the index for selective download. The installed CLI's
`geniesim assets list` is authoritative for available tasks.

`--workers` defaults to `8` and accepts `1–32`. The plan's `unique` size accounts
for content deduplication; downloading also requires at least 256 MiB of free
space beyond the remaining content size.

Keep the same selected content and benchmark baseline for a destination.
Adding tasks that change the selected file set requires a new directory; request
all needed selectors together. Switching the download source is supported for the
same content. `--overwrite` does not override a destination's release mismatch.

## 4. Resume or switch sources

ModelScope is the default source. Repeat the same command after interruption:
completed files are checked and reused, and partial content resumes when the
server supports HTTP Range requests. Otherwise the partial transfer restarts.

Switch to HuggingFace while retaining the same selected content and destination:

```bash
geniesim assets download g2op_if --source huggingface --dest "$HOME/sim/assets-g2op-if"
```

Existing files with different release hashes cause an error by default. Add
`--overwrite` only when replacing those files is intended.

The index pins source revisions and file hashes; it does not scan future remote
changes. `--index FILE` selects another audited index with the supported schema.
Only use an index supplied for the intended simulator code and asset release.
`verify` confirms published bytes, not scene initialization or task success.

## 5. Use the downloaded assets in simulation

Before creating the GenieSim container, set the asset source on the host:

```bash
export GENIESIM_ASSETS_SRC="$HOME/sim/assets-g2op-if"
geniesim docker up
geniesim docker into
```

Inside a prepared simulation environment, preview a layout with the standard CLI:

```bash
geniesim benchmark run g2op_if_pick_block_color -- \
  --benchmark.preview=true --benchmark.instance_ids=0 \
  --benchmark.record=false --app.headless=true
```

Container mounts are fixed at creation. An existing container must be recreated
with the intended asset source for the new mount to take effect.

## Notes

- Available tasks depend on the installed index. The index this skill describes contains
  150 tasks, including 50 robust tasks and 22 of the 25 sandbox tasks.
- `sandbox_hotel_store_suitcase`, `sandbox_restaurant_guide_guests`, and
  `sandbox_store_carry_carton` are excluded because their scene metadata references
  objects missing from the published asset revision. Do not fabricate replacements.
- A task selection includes dependencies of all its indexed layouts; selection by
  `instance_id` is not supported. Each closure includes the published common
  HDR/EXR backgrounds.
