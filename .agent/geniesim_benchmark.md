# geniesim_benchmark — repository-level dispatcher

> 🧭 **Canonical source**: [`source/geniesim_benchmark/README.md`](../source/geniesim_benchmark/README.md) is the source of truth for the task runtime, config schema, CLI verbs, and scoring contract.

This file is a 30-second pointer. Do not duplicate content here that lives at the canonical source — duplication is what makes dispatchers rot.

---

## What it is

The benchmark task runtime: declarative `.yaml` configs in `config/` describe what to load, which robot to drive, and what to evaluate. Tasks are launched through the `geniesim benchmark` CLI verb (owned by [`geniesim_cli`](geniesim_cli.md)) and run against a user-provided inference server (`--infer-host=IP:PORT`).

The package runs Isaac evaluation directly. The canonical README describes its
runtime requirements; benchmark task configs use a separate schema from ROS scenes.

## Where to look

| Topic | File |
|---|---|
| Canonical CLI surface + config naming convention | [`source/geniesim_benchmark/README.md`](../source/geniesim_benchmark/README.md) |
| User-facing intro + leaderboard pointer | [`source/geniesim_benchmark/README.md`](../source/geniesim_benchmark/README.md) |
| Agent skills (run, check-inference) | [`source/geniesim_benchmark/skills/`](../source/geniesim_benchmark/skills/) |
| Task configs (the work-list) | [`source/geniesim_benchmark/src/geniesim_benchmark/config/`](../source/geniesim_benchmark/src/geniesim_benchmark/config/) |
| App entry | [`source/geniesim_benchmark/src/geniesim_benchmark/app/app.py`](../source/geniesim_benchmark/src/geniesim_benchmark/app/app.py) |
| Dataset utilities (converters) | [`source/geniesim_benchmark/src/geniesim_benchmark/dataset/`](../source/geniesim_benchmark/src/geniesim_benchmark/dataset/) — `agibot → LeRobot v2.1` lives in `dataset/convert/agibot_to_lerobot.py`; CLI entry is `geniesim dataset convert agibot-to-lerobot` |

## Invariants the rest of the repo relies on

- **Separate configuration schemas.** Benchmark task configs and ROS scene YAMLs have different contracts. Use the runtime-specific loader documented in the README.
- **Config naming is `<robot>_<category>_<task>.yaml`.** The CLI splits on the **second** token as category; deeper tags (e.g. `g2op_probe_if_pick_*`) belong to the task name. Robot prefixes (`g2op`, `g290d`, `arxone`, `g1op`, `aloha`) are **internal to this package** today — they're not the same vocabulary as `geniesim_ros` scene yamls.
- **The `--infer-host=H:P` shorthand** is rewritten to `--benchmark.infer_host=…` (run/batch) or `--host …  --port …` (check-inference). Don't pin a host inside a `.yaml` — keep it at invocation time.
- **`config.yaml` / `template.yaml` / `teleop.yaml`** are templates / defaults, **not** runnable tasks. `geniesim benchmark list` filters them out.
- **Unknown `--key=value` flags are forwarded verbatim** to `app/app.py`'s `ParameterServer`. The full key space is the `@dataclass` tree in `config/params.py`. Don't strip "unknown" flags in the CLI dispatch.
- **`batch` semantics:** sequential per-config `subprocess.run`; exits non-zero if **any** config failed. Long jobs need `tmux` / log pipe.
- **Leaderboard tables in the root README (`GenieSim-Instruction` / `-Robust` / `-Manipulation` / `-Sim2Real`)** are part of the published contract. Don't rename these strings.
