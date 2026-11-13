---
name: challenge-download-datasets
description: >
  Download the Simulation Challenge LeRobot v2.1 training datasets from ModelScope using
  ./scripts/download_dataset.sh. Pulls task-suite data (instruction / manipulation / sim2real /
  cross_embodiment_transfer / dexterous_manipulation / precision_manipulation /
  whole_body_coordination) from
  the agibot_world/GenieSim3.0-Dataset repo into a local dir. Trigger: When the user asks to
  "download training data", "download the dataset", "fetch LeRobot data",
  "get the lerobot v2.1 data", "download task suite", mentions download_dataset.sh, ModelScope
  GenieSim3.0-Dataset, or needs training data to train a challenge model.
metadata:
  author: zy
  version: "1.1"
---

# challenge-download-datasets — Fetch LeRobot v2.1 training data

Download the official **LeRobot v2.1** training datasets for the Simulation Challenge from
ModelScope (`agibot_world/GenieSim3.0-Dataset`) via the bundled `download_dataset.sh`. The data is
organized into **task suites**; pick one or grab all of them.

> **Self-contained:** the downloader ships with this skill at `scripts/download_dataset.sh` — you
> don't need the genie-sim repo checked out. The repo also has it at `./scripts/download_dataset.sh`;
> either works. Examples below use `$SKILL_DIR/scripts/download_dataset.sh` where `$SKILL_DIR` is
> this skill's directory.

This produces the training corpus consumed when training/finetuning a contestant model — once you
have a checkpoint, hand off to `challenge-baseline-model` (provision/run inference) and the rest of
the `challenge-help` pipeline.

---

## Prerequisite — modelscope CLI

The script shells out to the `modelscope` downloader. Install it once if missing:

```bash
command -v modelscope >/dev/null || pip install modelscope
```

## Available task suites

| Suite | Remote path | Notes |
|-------|-------------|-------|
| `instruction`  | `task_suite/instruction/**`  | instruction-following demos |
| `manipulation` | `task_suite/manipulation/**` | manipulation demos |
| `sim2real`     | `task_suite/sim2real/**`     | sim2real demos |
| `cross_embodiment_transfer` | `task_suite/cross_embodiment_transfer/**` | cross-embodiment transfer demos |
| `dexterous_manipulation` | `task_suite/dexterous_manipulation/**` | dexterous manipulation demos |
| `precision_manipulation` | `task_suite/precision_manipulation/**` | precision manipulation demos |
| `whole_body_coordination` | `task_suite/whole_body_coordination/**` | whole-body coordination demos |

Each suite is downloaded in **LeRobot v2.1** format.

## Usage

Signature: `download_dataset.sh [SUITE_NAME] [LOCAL_DIR]`. Resolve the bundled script path first
(works regardless of cwd), then call it:

```bash
SCRIPT="$(dirname "$0")/scripts/download_dataset.sh"   # or hard-code this skill's scripts/ path

# Download ONE suite to ./data/<suite>/  (default LOCAL_DIR is ./data/)
"$SCRIPT" instruction

# Download a suite to a custom base dir → /path/to/save/sim2real/
"$SCRIPT" sim2real /path/to/save

# Download other suites to a custom base dir
"$SCRIPT" cross_embodiment_transfer /path/to/save
"$SCRIPT" dexterous_manipulation /path/to/save
"$SCRIPT" precision_manipulation /path/to/save
"$SCRIPT" whole_body_coordination /path/to/save

# Download ALL seven suites
"$SCRIPT"

# Help
"$SCRIPT" -h
```

`LOCAL_DIR` is the **base** dir — output lands at `<LOCAL_DIR>/<suite>/`, so it doesn't matter which
directory you invoke the script from.

**Output layout:** a suite lands at `<LOCAL_DIR>/<suite>/` (default `./data/<suite>/`). The script
downloads to a temp dir first, then copies `task_suite/<suite>/` contents into the target — so the
`task_suite/` prefix is stripped in the final layout.

## Notes

- **Pick a suite to limit size** — omitting `SUITE_NAME` downloads all seven, which is large. Prefer
  naming the specific suite the user needs.
- **Resumable:** `modelscope download` caches/resumes; re-running after an interruption continues
  rather than restarting from scratch.
- **Invalid suite names fail fast** — the script only accepts the seven names in the table above;
  anything else exits with the valid list.
- **Disk + network:** downloads can be tens of GB. Run with `run_in_background` if driving from the
  assistant so the session stays responsive, and confirm the target disk has room first.
- After data is in place, training is out of scope for this skill — see `challenge-baseline-model`
  to stand up inference once you have a checkpoint, and `challenge-help` for the full job pipeline.
