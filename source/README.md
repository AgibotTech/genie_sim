# source/ — Genie Sim Module Map

Genie Sim is shipped as a fan-out of **peer Python packages** plus a couple
of non-`geniesim` modules. Each module lives under `source/<name>/` and is
self-contained: it has its own `pyproject.toml`, its own docs, and (for
the agent-driven workflows) its own `skills/` directory.

This file is the **entry index** for GitHub readers. Start here, then
click into the module you actually need.

> 🧞 **First time?** Install the CLI and bootstrap the rest of the stack
> following [`source/geniesim_cli/AGENTS.md` § 0 — Fresh-machine setup](geniesim_cli/AGENTS.md) — that's the canonical onboarding path.

---

## 📦 Genie Sim Python packages

All editable-installable via `pip install -e source/<pkg>/`. The
`geniesim` umbrella pulls them in as deps; `geniesim bootstrap` runs the
installs in topological order.

| Module | Role | Entry doc | Skills |
|---|---|---|---|
| 🧞 [`geniesim_cli/`](geniesim_cli/) | CLI dispatcher — the `geniesim` console script | [AGENTS.md](geniesim_cli/AGENTS.md) | — |
| 🌐 [`geniesim/`](geniesim/) | Umbrella meta-package — pulls in every peer | — | — |
| 🧪 [`geniesim_benchmark/`](geniesim_benchmark/) | Benchmark tasks and scoring with Isaac, Newton, and native MuJoCo runtimes | [README.md](geniesim_benchmark/README.md) | [run-benchmark](geniesim_benchmark/skills/run-benchmark/SKILL.md), [check-inference](geniesim_benchmark/skills/check-inference/SKILL.md) |
| 🏗️ [`geniesim_scene_agent/`](geniesim_scene_agent/) | Asset search, scene authoring, and task orchestration | [AGENTS.md](geniesim_scene_agent/AGENTS.md) | [geniesim-scene-generation](geniesim_scene_agent/skills/geniesim-scene-generation/SKILL.md), [geniesim-asset-search](geniesim_scene_agent/skills/geniesim-asset-search/SKILL.md), [scene-to-data](geniesim_scene_agent/skills/geniesim-scene-to-data/SKILL.md) |
| ⚡ [`geniesim_ros/`](geniesim_ros/) | **Genie Sim RT Engine** — realtime interactive ROS 2 simulation | [README.md](geniesim_ros/README.md) · [AGENTS.md](geniesim_ros/AGENTS.md) | [build-workspace](geniesim_ros/skills/build-workspace/SKILL.md), [launch-scene](geniesim_ros/skills/launch-scene/SKILL.md), [moveit-wbc](geniesim_ros/skills/moveit-wbc/SKILL.md), [add-robot](geniesim_ros/skills/add-robot/SKILL.md), [teleop-bridge](geniesim_ros/skills/teleop-bridge/SKILL.md), [record-episode](geniesim_ros/skills/record-episode/SKILL.md), [debug-physics](geniesim_ros/skills/debug-physics/SKILL.md), [material-override](geniesim_ros/skills/material-override/SKILL.md) |
| 🎮 [`geniesim_teleop/`](geniesim_teleop/) | VR / Pico teleoperation bridge | [AGENTS.md](geniesim_teleop/AGENTS.md) | [run-teleop](geniesim_teleop/skills/run-teleop/SKILL.md) |
| 🌍 [`geniesim_world/`](geniesim_world/) | Multimodal spatial world model (panorama → 3D world) | [README.md](geniesim_world/README.md) · [AGENTS.md](geniesim_world/AGENTS.md) | [generate-world](geniesim_world/skills/generate-world/SKILL.md) |
| 🌙 [`geniesim_dream/`](geniesim_dream/) | G1/G2 world-model inference, policy and VLM evaluation; own environments | [README.md](geniesim_dream/README.md) · [AGENTS.md](geniesim_dream/AGENTS.md) | [run-dream](geniesim_dream/skills/run-dream/SKILL.md) |

### Where do I start?

Install and manage containers with `geniesim_cli`, then select the workflow:

```
geniesim_cli      → install + docker
   │
   ├── geniesim_benchmark   task-selected Isaac / Newton / native MuJoCo
   │                        — run / score / evaluate policies headless
   │
   └── geniesim_ros         Genie Sim RT Engine (ROS 2 native)
                            — interactive sim, MoveIt, teleop, recording
```

Benchmark and RT Engine configurations have separate schemas and launch graphs.
Managed Newton and IPC tasks use the RT Engine client; direct Isaac and native
MuJoCo tasks retain their own runtime entry points. Proposed convergence is
tracked in the [platform roadmap](../docs/ROADMAP.md).

Side branches:

- **Author a new scene from a panorama** → `geniesim_world`
- **Synthesise scenes / search assets via LLM** → `geniesim_scene_agent`
- **Train an RL policy in the loop** → `rlinf_geniesim`
- **Connect LLM planning to robot actions** → [`agent2act`](agent2act/README.md)
- **Generate robot videos with a world model** → [`geniesim_dream`](geniesim_dream/README.md) (separate installation)

---

## Installation tiers

The [umbrella metadata](geniesim/pyproject.toml) declares required peers and
optional extras. First-class package status does not imply automatic installation.

| Install group | Packages | Setup |
|---|---|---|
| Tier 1 — required | `geniesim_cli`, `geniesim_assets`, `geniesim_benchmark`, `geniesim_ros` | Install the CLI first; `geniesim bootstrap` installs or repairs required peers and the `geniesim` umbrella. Download asset sources separately. |
| Tier 2 — opt-in | `geniesim_teleop`, `geniesim_scene_agent`, `geniesim_world` | Follow the owning module's setup guide. World uses a dedicated environment. Default bootstrap skips these packages. |
| Separate first-class package | `geniesim_dream` | Install its world-model, policy and evaluation environments separately. It is outside the umbrella extras. |
| Independently maintained | `agent2act`, `data_collection`, `rlinf_geniesim`, `scene_reconstruction` | Follow each module's installation guide; these are outside both tiers. |

## 🔗 Module dependency DAG

How the `geniesim_*` peers depend on each other at build time and runtime.

**Legend:** `-->|build|` build-system requires · `==>|exec|` runtime dependency · `-.->|[X] extra|` optional extra · `-.->|refactor: layer atop|` planned refactor target. Methodology + how to regenerate: see [`AGENTS.md` § Module dependency DAG — methodology](AGENTS.md#-module-dependency-dag--methodology).

<!-- AUTOGEN:deps-dag start -->
```mermaid
graph TD
  %% Auto-generated by `geniesim tool deps-dag --fix`.
  %% Edges: -->|build| build-system requires; ==>|exec| runtime deps;
  %%        -.->|[X] extra| optional extras.
  %% Source: source/geniesim_*/pyproject.toml + _ANNOTATIONS in
  %% source/geniesim_cli/src/geniesim_cli/commands/tool.py.

  geniesim["<b>geniesim</b><br/>meta-package"]
  geniesim_cli["<b>geniesim_cli</b><br/>console script"]
  geniesim_benchmark["<b>geniesim_benchmark</b><br/>Isaac / Newton / MuJoCo"]
  geniesim_dream["<b>geniesim_dream</b><br/>🌙 world-model inference (own envs)"]
  geniesim_ros["<b>geniesim_ros</b><br/>⚡ RT Engine"]
  geniesim_scene_agent["<b>geniesim_scene_agent</b><br/>🏗️ scene gen (optional extra)"]
  geniesim_teleop["<b>geniesim_teleop</b><br/>🎮 VR / Pico"]
  geniesim_world["<b>geniesim_world</b><br/>🌍 pano → 3D world (own env)"]

  geniesim -->|build| geniesim_cli
  geniesim -->|build| geniesim_benchmark
  geniesim -->|build| geniesim_ros
  geniesim_benchmark ==>|exec| geniesim_cli
  geniesim_ros ==>|exec| geniesim_cli
  geniesim_teleop ==>|exec| geniesim_cli
  geniesim_teleop ==>|exec| geniesim_ros
  geniesim -."[teleop,all,full] extra".-> geniesim_teleop
  geniesim -."[scene_agent,all,full] extra".-> geniesim_scene_agent
  geniesim -."[world,all,full] extra".-> geniesim_world
  geniesim_benchmark -..->|Newton client| geniesim_ros

  classDef legacy fill:#fff3cd,stroke:#856404,color:#856404
  classDef leaf fill:#e9ecef,stroke:#6c757d,color:#495057
  classDef placeholder fill:#f8d7da,stroke:#842029,color:#842029
  classDef umbrella fill:#cfe2ff,stroke:#084298,color:#084298
  class geniesim_dream,geniesim_world leaf
  class geniesim umbrella
```
<!-- AUTOGEN:deps-dag end -->

---

## 🧩 Separately-maintained modules

These directories live under `source/` but are **not** part of the
`geniesim_*` peer set. They have their own build / run conventions,
their own release cadence, and they are not pulled in by
`geniesim bootstrap`. Some predate the `geniesim_*` reorganisation;
others are active out-of-band collaborations.

| Directory | Description |
|---|---|
| [`agent2act/`](agent2act/README.md) | LLM-to-action runtime with motion/policy/auto execution, CoRobot WebSocket service, and a separate uv environment |
| [`data_collection/`](data_collection/) | Independent Isaac Sim 5.1 / cuRobo v2 collection and replay |
| 🎓 [`rlinf_geniesim/`](rlinf_geniesim/) | RL training (RLinf, human-in-the-loop, distributed) |
| [`scene_reconstruction/`](scene_reconstruction/) | 3D reconstruction pipeline (Dockerfile, third-party deps) |
| [`external/`](external/) | Vendored third-party code (ml-sharp, DA360, …) |

---

## 🤖 Agent SKILLs

Every `skills/<name>/SKILL.md` is self-contained and invocable by
agentic coding agents (Claude Code et al.). The skill file carries
its own trigger phrases, prerequisites, workflow, and copy-paste
commands — so an agent can drive the simulator end-to-end without
the user reading the source.

Human consumers can `cat` any skill file for the same recipe:

```bash
cat source/geniesim_ros/skills/launch-scene/SKILL.md
```

Skill conventions:

- YAML frontmatter with `name`, `description`, `license`, `metadata`.
- Sections: **When to Use** · **Critical Patterns** · **Workflow** ·
  **Commands** · **Notes** · **Resources**.
- Cross-reference related skills by name in Resources.

---

## 📜 Conventions

- **`pip install -e source/<pkg>/`** for every Python package — never
  `pip install geniesim` (not on PyPI).
- **`AGENTS.md`** is the engineer-facing routing doc for each package.
- **`README.md`** (where present) is the user-facing pitch / quickstart.
- **`skills/`** is the agent-facing recipe library — same target
  audience as a human user looking for a copy-paste guide.

When in doubt, follow the trail:
**root README** → `source/AGENTS.md` → per-package `AGENTS.md` →
per-package `skills/<name>/SKILL.md`.

---

## 🔗 Top-level pointers

- Root README: [`../README.md`](../README.md)
- Repository-wide agent doc: [`../AGENTS.md`](../AGENTS.md)
- First-principles reasoning rule: [`../.agent/FIRST_PRINCIPLES.md`](../.agent/FIRST_PRINCIPLES.md)
- Umbrella deep-dive: [`../.agent/geniesim.md`](../.agent/geniesim.md)
- CLI deep-dive: [`../.agent/geniesim_cli.md`](../.agent/geniesim_cli.md)
- RT Engine deep-dive: [`../.agent/geniesim_ros.md`](../.agent/geniesim_ros.md)
