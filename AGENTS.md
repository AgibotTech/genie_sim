# Agent Development Guide

This file is the entry point for agentic coding agents operating in this repository.
Start here, then follow the links below for details on each subsystem.

**Reasoning rule**: Agents must follow [`.agent/FIRST_PRINCIPLES.md`](./.agent/FIRST_PRINCIPLES.md) —
derive every action from evidence in this repo and from explicit contracts; never guess or pattern-match.

---

## 🚀 Agent boot sequence

When you pick up a task in this repository, read in this order. Stop as soon as the question is answered; don't burn context on files that aren't relevant.

1. **This file** (`AGENTS.md`) — the dispatch layer. Tells you which subsystem owns the task.
2. **[`source/AGENTS.md`](./source/AGENTS.md)** — repository module map (one line per `geniesim_*` peer). Picks the package you'll be editing.
3. **`source/<pkg>/AGENTS.md`** — canonical guide for that one package. File layout, command surface, architectural rules, troubleshooting. This is the source of truth for "how does X work".
4. **`source/<pkg>/skills/<name>/SKILL.md`** — concrete recipe for a common workflow (run a benchmark, launch a scene, add a robot, …). Self-contained: prerequisites, copy-paste commands, gotchas.
5. **Source files referenced by 1–4.** Only after the doc trail above narrows the scope.

Side trail for repo-wide architecture (rarely needed):

- [`.agent/`](./.agent/) — thin redirects to the per-package `AGENTS.md` of the most-asked-about peers. Treat as a dispatcher; the per-package file is canonical.

> **Anti-pattern**: starting with `grep` / `find` before reading `source/AGENTS.md`. The package table tells you which directory to scope a search to.

---

## 🧞 Canonical commands

The `geniesim` CLI is the single entry point for every operator-level workflow. Prefer these over ad-hoc invocations; they handle path resolution, interpreter selection, and environment normalisation.

| Verb | What it does |
|---|---|
| `geniesim bootstrap` | Install every peer distribution in topological order (the umbrella) |
| `geniesim status` | Per-distribution health probe — never raises on missing siblings |
| `geniesim doctor` | Diagnose & repair (status + rosdep + env) |
| `geniesim docker {build,up,into,down,logs}` | Manage the Genie Sim container (Isaac Sim 5.1 / 6.0 / 4.5) |
| `geniesim ros build {dev,release,cleanup}` | `colcon build` the ROS 2 workspace |
| `geniesim ros doctor` | Repair rosdep |
| `geniesim benchmark run <CONFIG>` | Run one benchmark task |
| `geniesim benchmark check-inference` | Probe an inference WebSocket server |
| `geniesim teleop run` | Launch the VR / Pico teleop loop |
| `geniesim deploy [MODULE]` | Build pure-Python wheel(s) into `./deploy/` |
| `geniesim version` / `geniesim env` / `geniesim completion bash\|zsh` | Operator utilities |

Full surface: [`source/geniesim_cli/AGENTS.md`](./source/geniesim_cli/AGENTS.md).

> **Never** suggest `pip install geniesim` or `pip install geniesim_assets` — those distributions are **not on PyPI**. Always redirect through `geniesim bootstrap`.

---

## Core References

| Document | Purpose |
|---|---|
| [.agent/FIRST_PRINCIPLES.md](./.agent/FIRST_PRINCIPLES.md) | Evidence-first reasoning rules — derive every action from repo facts, never guess |
| [source/AGENTS.md](./source/AGENTS.md) | Repository module map — one row per `geniesim_*` peer with its docs + skills |
| [source/README.md](./source/README.md) | GitHub-facing module index — same map, reader-oriented |

`.agent/` is a thin dispatch layer for cross-cutting peers; each redirects to its canonical `source/<pkg>/AGENTS.md`. Leaf peers go straight to source.

| Peer | Guide |
|---|---|
| `geniesim` (umbrella) | [`.agent/geniesim.md`](./.agent/geniesim.md) |
| `geniesim_cli` | [`.agent/geniesim_cli.md`](./.agent/geniesim_cli.md) |
| `geniesim_benchmark` | [`.agent/geniesim_benchmark.md`](./.agent/geniesim_benchmark.md) |
| `geniesim_generator` | [`.agent/geniesim_generator.md`](./.agent/geniesim_generator.md) |
| `geniesim_ros` | [`.agent/geniesim_ros.md`](./.agent/geniesim_ros.md) |
| `geniesim_teleop` | [`.agent/geniesim_teleop.md`](./.agent/geniesim_teleop.md) |
| `geniesim_world` | [`source/geniesim_world/AGENTS.md`](./source/geniesim_world/AGENTS.md) |
| `agent2act` | [`source/agent2act/AGENTS.md`](./source/agent2act/AGENTS.md) |
| `data_collection` | [`source/data_collection/AGENTS.md`](./source/data_collection/AGENTS.md) |
| `rlinf_geniesim` | [`source/rlinf_geniesim/README.md`](./source/rlinf_geniesim/README.md) |
| `scene_reconstruction` | [`source/scene_reconstruction/README.md`](./source/scene_reconstruction/README.md) |

---

## Repository Layout (quick map)

```
source/
├── geniesim/             umbrella meta-package (no code, only deps)
├── geniesim_cli/         CLI dispatcher; owns the `geniesim` console script
├── geniesim_benchmark/   benchmark tasks, scoring, LLM eval configs
├── geniesim_generator/   scene generation, procedural layout
├── geniesim_ros/         Genie Sim RT Engine — ROS 2 workspace
├── geniesim_teleop/      VR / Pico teleoperation bridge
├── geniesim_world/       multimodal spatial world model (pano → 3D)
│
│   ── separately-maintained (not `geniesim_*` peers) ──
├── agent2act/            LLM-to-action runtime and CoRobot service (own uv env)
├── data_collection/      data collection client/server
├── rlinf_geniesim/       RL training (RLinf, human-in-the-loop)
├── scene_reconstruction/ 3D reconstruction pipeline
└── external/             vendored third-party code
```

See [source/AGENTS.md](./source/AGENTS.md) for the full package table.

---

## Do Not

- Generate or guess URLs — only use URLs present in local files.
- Hardcode absolute paths outside the repo root.
- Skip `geniesim_cli._style` colors when writing CLI output.
- Put ROS-only deps (`rclpy`, `cv-bridge`, …) in the core `geniesim` or `geniesim_cli` dependencies.
- Suggest `pip install geniesim` or `pip install geniesim_assets` — not on PyPI; redirect to `geniesim bootstrap`.
- Duplicate `.agent/*.md` content from the per-package `AGENTS.md` it points at. The `.agent/` files are dispatchers, not canonical sources.
