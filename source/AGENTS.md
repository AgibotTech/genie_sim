# source/ — Repository Module Map

This file is the **AGENTS routing table** for the repository's source tree.
For the GitHub-facing user-oriented version (with badges + a "where do I
start" diagram), see [`README.md`](README.md).

> **Maintenance contract** — when you add a peer distribution, rename a
> module, or move a package between the first-class / legacy tiers,
> update this file in the same diff.

---

## 🧞 First-class Genie Sim packages

Pip-installable distributions under `pip install -e source/<pkg>/`. The
`geniesim` umbrella pulls them in as deps; `geniesim bootstrap` runs the
installs in topological order.

| Directory | Distribution | README | AGENTS | Skills |
|---|---|---|---|---|
| `geniesim/` | umbrella (`geniesim`) | [README](geniesim/README.md) | — | — |
| `geniesim_cli/` | `geniesim_cli` — owns the `geniesim` console script | [README](geniesim_cli/README.md) | [AGENTS](geniesim_cli/AGENTS.md) | — |
| `geniesim_benchmark/` | `geniesim_benchmark` — benchmark tasks + scoring (**legacy stack — Isaac Sim direct; refactor planned onto `geniesim_ros`**) | [README](geniesim_benchmark/README.md) | — | [skills/](geniesim_benchmark/skills/) |
| `geniesim_generator/` | `geniesim_generator` — LLM-driven scene generation (gated) | [README](geniesim_generator/README.md) | [AGENTS](geniesim_generator/AGENTS.md) | [skills/](geniesim_generator/skills/) |
| `geniesim_ros/` | `geniesim_ros` — Genie Sim RT Engine (ROS 2 workspace) | [README](geniesim_ros/README.md) | [AGENTS](geniesim_ros/AGENTS.md) | [skills/](geniesim_ros/skills/) |
| `geniesim_teleop/` | `geniesim_teleop` — VR / Pico teleop bridge | [README](geniesim_teleop/README.md) | [AGENTS](geniesim_teleop/AGENTS.md) | [skills/](geniesim_teleop/skills/) |
| `geniesim_world/` | `geniesim_world` — multimodal spatial world model (pano → 3D) | [README](geniesim_world/README.md) | [AGENTS](geniesim_world/AGENTS.md) | [skills/](geniesim_world/skills/) |

> `geniesim_world` was previously listed as "legacy"; it is now first-class
> with its own README + skill, and is installed out-of-band (heavy CUDA +
> ML deps in its own conda env, not via `geniesim bootstrap`). See its
> README for the install path.

> ⚠️ **`geniesim_benchmark` and `geniesim_ros` are independent parallel
> stacks today.** The benchmark package drives Isaac Sim directly; the RT
> Engine is the ROS 2 native engine. Don't unify their scene formats /
> configs / launch graphs prematurely — the roadmap is to refactor the
> benchmark runtime into a benchmark **layer on top of** `geniesim_ros`.
> See [`.agent/geniesim_benchmark.md`](../.agent/geniesim_benchmark.md)
> § "Architectural status".

---

## 🔗 Module dependency DAG — methodology

**The rendered diagram lives in [`source/README.md`](README.md).** This section explains how it's generated so contributors can extend the audit; the diagram itself is read on the README page (GitHub renders Mermaid inline).

### Generator

```bash
geniesim tool deps-dag             # verify the block in source/README.md is current
geniesim tool deps-dag --fix       # regenerate the block in place
```

Source of truth, in evaluation order:

1. **`source/geniesim_*/pyproject.toml`** — every peer with a `[project]` table is enumerated by globbing `source/geniesim*/pyproject.toml`. Sibling modules that don't match the `geniesim_*` glob (`agent2act`, `rlinf_geniesim`, `data_collection`, `scene_reconstruction`, `external`) are deliberately out of scope.
2. **`[build-system].requires`** → emitted as **build edges** (`-->|build|`). External-only requirements (`setuptools`, `wheel`) produce no edges; the edge fires when a peer needs *another peer* at build time.
3. **`[project].dependencies`** → emitted as **runtime/exec edges** (`==>|exec|`), with one special case: when the source peer carries `role: umbrella` in `_ANNOTATIONS`, its `[project].dependencies` are reclassified as **build edges** (`-->|build|`). The umbrella ships no code beyond `__version__`, so those deps are packaging declarations ("install these alongside me"), not runtime imports.
4. **`[project.optional-dependencies]`** → emitted as **extra edges** (`-.->|[X] extra|`). Edges with the same `(src, dst)` collapse — `geniesim[generator,full]` becomes one labelled edge.
5. **`_ANNOTATIONS`** in [`source/geniesim_cli/src/geniesim_cli/commands/tool.py`](geniesim_cli/src/geniesim_cli/commands/tool.py) — the only hand-edited input. Carries `status` (legacy / leaf / placeholder), `role` (umbrella / cli), `note` (short label), `refactor_target` (planned merge target), and `requires_agents` (exempt from the docs audit).
6. **`refactor_target`** in `_ANNOTATIONS` → emitted as a **refactor edge** (`-.->|refactor: layer atop|`).

### Edge taxonomy

| Edge style | Mermaid syntax | Source |
|---|---|---|
| Build | `-->|build|` | `[build-system].requires` cross-peer **or** `[project].dependencies` of a peer with `role: umbrella` (the meta-package's "install alongside" set) |
| Exec (runtime) | `==>|exec|` | `[project].dependencies` cross-peer (non-umbrella sources) |
| Optional extra | `-.->|[X] extra|` | `[project.optional-dependencies][X]` cross-peer |
| Refactor target | `-.->|refactor: layer atop|` | `_ANNOTATIONS[name]["refactor_target"]` |

### Marker contract

The generator writes between `<!-- AUTOGEN:deps-dag start -->` and `<!-- AUTOGEN:deps-dag end -->`. Both must be present in the target file; missing markers raise a hard error. Do not hand-edit between them — CI runs `geniesim tool deps-dag` (without `--fix`) and fails on drift. Run `--fix` after any pyproject change or `_ANNOTATIONS` update and commit the result.

### Adding a new edge category

To surface a relationship that isn't in pyproject metadata (e.g. a runtime ROS topic contract), add a new emission pass inside `_emit_mermaid` in `tool.py` and define a new Mermaid arrow style. The `_RUNTIME_EDGES` constant existed for this purpose previously and is currently empty — that's where one-off runtime contracts that don't have a pip-level dep should be declared.
---

## 🧩 Separately-maintained modules

These directories live under `source/` but are **not** part of the
`geniesim_*` peer set: they have their own build / run conventions,
their own release cadence, and they are not pulled in by `geniesim
bootstrap`. Some predate the `geniesim_*` reorganisation; others are
active out-of-band collaborations. Don't extend without a plan, and
don't cross-reference them as if they were peer distributions.

| Directory | Description |
|---|---|
| `agent2act/` | LLM-to-action runtime and CoRobot WebSocket service; own uv environment — see [README.md](./agent2act/README.md) and [AGENTS.md](./agent2act/AGENTS.md) |
| `data_collection/` | Data collection client/server, ROS nodes, aimdk protocol — see [AGENTS.md](./data_collection/AGENTS.md) |
| `rlinf_geniesim/` | RL training pipeline (RLinf, human-in-the-loop, distributed) — see [README.md](./rlinf_geniesim/README.md) |
| `scene_reconstruction/` | 3D reconstruction pipeline (Dockerfile, third-party deps) |
| `external/` | Vendored third-party code (ml-sharp, DA360, …) |

---

## 🤖 Agent skills convention

Every first-class peer that needs operator-driven workflows ships a
`skills/` directory; each `skills/<name>/SKILL.md` is a self-contained
recipe (frontmatter → When to Use → Critical Patterns → Workflow →
Commands → Notes → Resources). They are invocable directly by agentic
coding agents and readable by humans via `cat`.

When you add a SKILL.md, also link it from the package's
[README](README.md), [AGENTS](AGENTS.md), and any `.agent/<pkg>.md`
dispatcher.

---

## 📜 Doc-coverage rules

Per-package contracts:
- Every first-class peer ships **`README.md`** (user-facing pitch + install).
- Every peer with non-trivial dispatch logic ships **`AGENTS.md`** (agent / contributor routing).
- Every peer with operator workflows ships **`skills/<name>/SKILL.md`** files.

The ROS workspace has a stricter audit:
- Every package under `geniesim_ros/src/ros_ws/src/` ships both `README.md` and `AGENTS.md` — enforced by `geniesim tool docs --scope ros`.

### Canonical AGENTS.md skeleton

When you author a new `source/<pkg>/AGENTS.md`, follow this heading skeleton so agents reading any package find the same structure:

```
# <pkg> — Agent Development Guide
> Maintenance contract — when X changes, update this file in the same diff.

1. What this is (one paragraph, why this distribution exists)
2. Layout (filesystem tree)
3. CLI / API surface (table)
4. Architectural rules / invariants (numbered, do-not-violate)
5. Skills index (table — link to skills/README.md)
6. Common workflows (links to SKILL.md files, not inline duplicates)
7. Troubleshooting (table)
8. Verification recipes (table)
9. Do not (bullet list of foot-guns)
```

Existing AGENTS.md files predate this skeleton and don't all match exactly; treat the skeleton as the **target** for new files and incremental refactors, not a forced migration.

### Canonical SKILL.md frontmatter

```yaml
---
name: <skill-name>
description: >
  One paragraph + trigger phrases (en + 中文).
license: MPL-2.0
metadata:
  author: genie-sim
  version: "1.0"
prerequisites:               # other skills that must run first
  - <pkg>:<skill>
inputs:                      # machine-readable input declaration
  - name: <key>
    desc: …
    required: true|false
    default: …               # optional
outputs:
  - desc: What the skill leaves behind on success
---
```

Body sections (in order): **When to Use · Critical Patterns · Workflow · Commands · Notes · Resources**.

---

## 🔗 Pointers

- Root README: [`../README.md`](../README.md)
- Root AGENTS.md (boot sequence + canonical commands): [`../AGENTS.md`](../AGENTS.md)
- First-principles reasoning rule: [`../.agent/FIRST_PRINCIPLES.md`](../.agent/FIRST_PRINCIPLES.md)
- `.agent/` thin redirects: [`../.agent/`](../.agent/)
