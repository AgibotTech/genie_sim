# AGENTS.md — genie_sim_engine_core

Notes for AI agents editing this package. Read before touching code;
the invariants here protect the contract every backend depends on.
Start with **Who owns this package** below — it sets the bar for what
may be changed here at all, and the rest of this file is the detail of
that bar.

Companion doc:
- [README.md](./README.md) — public surface, the `PhysicsEngine` ABC,
  the run loop, the assemble pipeline, and how backends plug in.

## Who owns this package

**Designer / maintainer: yangwendisheng@agibot.com.**

`genie_sim_engine_core` is the *single* place that holds the contracts
and protocols of the `genie_sim_engine_*` ecosystem — the
`PhysicsEngine` ABC and its factory, the assemble pipeline and its
`AssembleStage` / `Manifest` / middleware surface, the parameter
schema, and the pybind ROS bridge. Every backend is written against
what is defined here. There is no second copy of any of it, and a
change made here is a change made to every backend at once, including
ones not in this repository.

That is why the bar for editing this package is different from the bar
for editing a backend:

- **Do not change the design, and do not add logic that works around
  it, without the designer's approval.** Not a new kwarg on a contract
  surface, not a special case in the factory, not a field that only one
  backend understands, not a shortcut that "just needs to work for
  now". If you are reaching for one of those, the design is telling you
  the thing belongs on the other side of the boundary.
- **This applies to AI agents in full.** An agent working here has read
  a slice of the repo and none of the deployments, the tuning history,
  or the downstream consumers. Plausible-looking is not the standard;
  the invariants in this file are load-bearing for reasons that are not
  all visible from the code. When a change looks obviously right and
  this file says otherwise, this file wins — raise it, do not route
  around it.
- **Porting is not approval.** Code arriving from another branch,
  another fork, or a vendor drop gets reviewed against these rules on
  its own merits. "It was already written that way over there" carries
  no weight here. Review imported code against the current contracts.
- **Record approved design decisions** in the owning documentation and tests.
  Commit messages follow the repository's one-line format.

What does *not* need approval: fixing a bug in an implementation that
already respects the contract, adding tests, improving docs and
comments, and anything confined to a backend package. When in doubt
about which side of the line a change falls on, ask before writing it,
not after.

## Scope

This package is the **cross-backend contract layer**. It owns:

- the `PhysicsEngine` ABC and its factory/registry;
- the run loop (`EngineRunLoop`), session (`EngineSession`), and
  parameter surface (`EngineNodeParams`, `early_params`);
- shared helpers with no backend content (mimic-joint, scene-yaml
  flag parsers);
- the **assemble pipeline** — orchestrator, `AssembleStage` ABC,
  `Pipeline` / `PipelineRunner`, `AssembleContext`, `Manifest`
  schema, middleware, the `Pipeline.register` decorator registry,
  and the CLI (`python -m genie_sim_engine_core.assemble`);
- the C++ pybind ROS bridge (`genie_sim_engine_py` — `/joint_states`,
  `/tf_scene`, `/odom`, `/clock` publishers + realtime scheduler +
  stats ring).

**Nothing engine-specific lives here.** No Isaac Sim, no Newton, no
libuipc, and no other physics runtime. If you find yourself adding
`import <backend_runtime>`, `from pxr import`, `import newton`,
`import uipc`, `import omni.*`, or any backend-specific extension,
stop — the code belongs in the backend package, and the *contract*
it implements belongs here.

## The zero-backend-imports rule

The load-bearing property of this package: importing any core module
must NOT drag in a backend runtime. Consumers include core-CI machines
that ship with none of the backend runtimes installed, and every
backend's assemble discovery (see below) loads every registered entry
point — a stray backend import at core-module scope would force that
runtime on every consumer.

Concretely:

- No `import` of a backend runtime at module scope in any file under
  `genie_sim_engine_core/`.
- No `from typing import TYPE_CHECKING; ...` shim to launder a runtime
  import — those still fail at type-checker load time.
- Data-structure fields that reference backend types are `str` or
  `dict` at the core boundary; the backend deserializes at call time.
- If a helper genuinely needs a light common library (e.g. `numpy`,
  `yaml`, `jsonschema`), it goes in `package.xml` as a `depend`; those
  are the only permitted "third-party" runtime deps.

## The config-is-the-only-source rule

Every tunable the engine reads comes from
`genie_sim_engine_core/config/physics_params.yaml` or a sibling profile
next to it (`physics_params_rlinf.yaml`, …). There is no second channel.

**Do not add an environment-variable override for a physics or engine
parameter — not in core, and not in a backend that reads a core
parameter type.** A logged override still changes physics without changing the
selected configuration. Development workspaces use `--symlink-install`, so YAML
edits do not require an image rebuild. Keep tuning in version-controlled
profiles so a run's configuration and metadata identify the applied values.

Concretely:

- No `os.environ` / `os.getenv` read for a value that could be a config
  field. If a knob is worth setting, it is worth a yaml key.
- No "env wins over yaml" priority chains, and no helper that
  generalizes one. A logged override is still an override.
- A parameter that a specific run needs to vary gets its own profile
  yaml, selected by the launcher, not an env var.
- The narrow exception is genuine *environment* facts, not tuning:
  paths, `ROS_DOMAIN_ID`, display/headless detection, CI markers — and
  secrets, which must NOT go in a yaml. If it changes what the simulation
  computes, it is not one of these.

## Core depends on nothing downstream

`genie_sim_engine_core` must not acquire a dependency on any
`genie_sim_engine_*` backend package — not an import, not a
`package.xml` `depend`, not a runtime attribute lookup that only
resolves when a backend is installed. The dependency arrow points one
way: backends build-depend on core and register into it.

This is the same property as the zero-backend-imports rule above, one
level up: that rule keeps a backend *runtime* (`newton`, `omni.*`,
`pxr`) out of core, this one keeps a backend *package* out. Both exist
so that core can be imported, tested, and released on a machine that
has no backend installed at all.

Concretely:

- No `import genie_sim_engine_newton…` / `genie_sim_engine_isaacsim…`
  anywhere under `genie_sim_engine_core/`, at module scope or inside a
  function. A lazy import is still a dependency; it just fails later,
  on a live run instead of at import.
- Naming a backend module in a docstring or comment is fine and is the
  intended way to point at the other side of a contract. Referencing
  one in *code* is not.
- When core needs something a backend has, invert it: core defines the
  ABC / registry / dataclass, and the backend supplies the
  implementation through it. If that feels impossible, the thing being
  reached for is backend content and belongs in the backend package.
- Backend-specific config keys are still allowed in the core yaml —
  the yaml is data, and the schema deliberately owns every backend's
  parameter surface so all of it is in one reviewable place.

## PhysicsEngine ABC and factory

- `PhysicsEngine.register(name)` is a decorator; the name is the
  string the launcher's `physics_engine` param maps to.
- `PhysicsEngine.create(name, **kwargs)` runs
  `_validate_create_kwargs` first — the ABI mismatch check. Don't
  short-circuit that call; the focused `TypeError` it produces is how
  we localize breakage to the factory boundary instead of blowing up
  deep inside a backend init.
- Constructor kwargs are a versioned surface. Adding a new kwarg to
  `EngineSession → PhysicsEngine.create` breaks every strict backend
  (backends without `**_unused`). Coordinate the change: land the
  backend updates first, then the session change. If you can't
  coordinate, add the kwarg to `_unused_by_default` handling in
  `PhysicsEngine.create` and file a migration item.
- The `registered_backend_abis()` reflection surface is public; the
  README documents a smoke check that uses it. Don't rename the fields
  (`class_name`, `required_names`, `accepts_extra_kwargs`) without
  updating the docs.

## Assemble pipeline invariants

The assemble subsystem is where the "many backends" pressure hits
hardest. The rules below make the pipeline safe to extend without
touching core for every new backend.

### Discovery

- Pipelines register via `@Pipeline.register(name)` on a factory
  function that returns a `Pipeline`. Registration happens at module
  import time; the CLI accepts `--pipeline-module <import.path>`
  (repeatable) so launchers list backend packages to import before
  dispatch — matching how `PhysicsEngine` backends are imported today.
- The registered factory must NOT import backend runtimes when
  executed. Backend imports live inside each `AssembleStage.run()`.
- If a backend's assemble module fails to import (backend runtime
  missing), the CLI logs and continues — a broken backend shouldn't
  kill the launcher's ability to inspect other pipelines. Same policy
  applies inside `Manifest.merge`: a malformed slice from one stage
  doesn't corrupt the rest.

### AssembleContext and AssembleArtifacts

- `AssembleContext.inputs` is a frozen dataclass. Stages that mutate
  it (via `object.__setattr__` or `dataclasses.replace` re-binding to
  `ctx.inputs`) are wrong. If a stage needs to communicate downstream,
  it goes through `StageResult`.
- `AssembleArtifacts.files` is a `dict[str, Path]` keyed by *logical*
  name (`"robot.usda"`, `"manifest.json"`), not filesystem path.
  Downstream stages look up by logical name so the stage-dir path
  layout can change without touching stages.
- `StageResult.manifest_slice` is scoped: stages must write only under
  `physics.<engine_key>.*` or a documented core-owned key. Core
  rejects merges that touch top-level keys the stage doesn't own.

### AssembleStage ABC

- `name` is the stage identifier used in logs and middleware. Snake
  case, backend-prefixed if the stage is backend-specific (e.g.
  `newton_extras`).
- `produces` and `requires` are `frozenset[str]` of logical file
  names. `Pipeline.validate()` runs a DAG check: every `requires`
  must be produced by either the inputs (`AssembleInputs` provides
  the scene YAML implicitly) or a prior stage. Missing prerequisites
  fail at pipeline construction, not at stage execution.
- `optional` and `skip_reason(ctx)` are the two skip mechanisms.
  `optional=True` is a static declaration ("this stage is skippable
  under any config"); `skip_reason` is dynamic. Use `skip_reason` for
  scene-yaml-conditional stages (e.g. "no cloth entries → skip
  newton_extras"). Never let a stage silently no-op inside `run()`
  without going through one of these two paths.
- `run(ctx) -> StageResult` is the only side-effect surface. A stage
  that raises propagates through core; middleware sees the exception
  in `after(err=…)`. Don't catch and swallow.

### Manifest schema

- Top-level keys are core-owned; `schema/manifest.py::Manifest` defines
  the complete field set and version policy. Changing a required field
  requires a version bump and coordinated reader updates.
- Backend-specific fields go under `physics.<engine_key>.*`, opaque
  to core. Core's `ManifestValidatorMiddleware` runs a JSON-schema
  check only on the core-owned keys; the reading engine validates its
  own slice.
- Bumping `MANIFEST_VERSION` is a coordinated change: update the
  schema, update `Manifest`, and teach `Manifest.read` how to
  upgrade old manifests (or refuse them with a clear error).

### Middleware

- Middleware are single-instance, ordered, and stateless between
  runs. Instance state during one pipeline run is fine (`Timing`
  keeps a per-stage clock); across runs is not.
- `before(stage, ctx)` runs before `skip_reason`; if the stage skips,
  `after(stage, ctx, result=None, err=None)` still fires so timers
  can close cleanly.
- `InvariantMiddleware` fails the pipeline if a stage returns without
  producing every file in its `produces` set. Do not weaken that
  check to accommodate "sometimes we don't need this file" cases —
  use `skip_reason` or split the stage instead.

### CLI

- `python -m genie_sim_engine_core.assemble` is the single entry
  point. Flag surface is stable: `--scene`, `--output-dir`,
  `--base-path`, `--physics-engine`, plus `--debug-dir` for
  `DebugDumpMiddleware`. Adding flags is fine; renaming or removing
  is a coordinated launcher change.
- Exit codes: `0` success, `1` pipeline validation failure, `2` stage
  runtime failure, `3` manifest write failure. Downstream launchers
  gate on these; don't collapse them into a single non-zero code.

## The pybind ROS bridge (`genie_sim_engine_py`)

- Lives under `src/` and builds as a pybind11 module. All backends
  import it via `import genie_sim_engine_py as _core` in
  `EngineSession.__init__`.
- Publishers (`/joint_states`, `/tf_scene`, `/odom`, `/clock`) are
  cross-backend contracts. Rate control, message shape, and QoS live
  here — backends only supply the numbers.
- The realtime scheduler and stats ring are exposed to Python. When
  editing either, remember: Python holds the GIL, the C++ RT thread
  does not. Any shared state is behind the ring's lock-free primitives
  in `src/realtime_buffer.cpp`. Don't add `PyGILState_Ensure` inside
  the RT loop.

## Python compatibility

- Python 3.10 and 3.12 must both work. Don't use 3.11+ syntax
  (`typing.Self`, `except*`, `LiteralString`) in any file consumers
  will import.
- Import `override` from `typing_extensions`, not `typing`.
- Prefer `dataclasses` + `typing` collection aliases (`list[X]`,
  `dict[K, V]`) — those work on 3.10 with `from __future__ import
  annotations`.

## Testing

Core has no backend-dependent tests. Suitable smoke checks when
editing:

- `PhysicsEngine.registered_backend_abis()` reflection (README
  documents the exact invocation).
- Pipeline DAG validation: construct a `Pipeline` with a `requires`
  that no earlier stage `produces`; expect `Pipeline.validate()` to
  raise.
- Manifest round-trip: write a `Manifest`, read it back, assert
  equality. Version-mismatch cases in `tests/schema/test_manifest.py`
  cover refusal of incompatible manifests.
- Registry discovery: import two toy backend assemble modules to
  trigger their `@Pipeline.register` decorators, then run
  `python -m genie_sim_engine_core.assemble --list-pipelines
  --pipeline-module toy_a.assemble --pipeline-module toy_b.assemble`
  and assert both appear.

## Files you edit here vs. elsewhere

Core changes:

- `genie_sim_engine_core/engine/base.py` — ABC + factory.
- `genie_sim_engine_core/common/{loop,session,params,early_params,scene_config}.py`
- `genie_sim_engine_core/assemble/{context,stage,pipeline,manifest,middleware}.py`
- `genie_sim_engine_core/assemble/__main__.py` — CLI.
- `src/*.cpp`, `src/pybinding.cpp` — pybind ROS bridge.

Elsewhere:

- New `AssembleStage` implementations — the owning backend package.
- New `PhysicsEngine` implementations — the owning backend package.
- Launcher wiring — `genie_sim_bringup`.

## Files you probably shouldn't edit without strong reason

- `PhysicsEngine.create` kwargs surface — coordinated change; every
  strict backend breaks otherwise.
- The `Manifest` top-level keys — coordinated change; every backend
  reader breaks otherwise.
- The `@Pipeline.register` name is a cross-backend contract — every
  registered pipeline name is a value the scene YAML's
  `physics_engine` field can pick. Renaming a registered pipeline
  breaks every scene YAML that referenced it.
- `MANIFEST_VERSION` — bump requires an upgrader in `Manifest.read`
  and a fixture in `tests/fixtures/`.
- `config/physics_params*.yaml` field names — read by every backend's
  parameter parsing; renaming a key silently falls back to the code
  default rather than erroring, so a rename must land with the
  `common/params.py` parser in the same commit.

## Runtime joint overlays (Channel B)

`common/joint_params.py::JointParamResolver` consumes resolved overlay paths,
validates `applies_to`, and returns sparse SI values by exact URDF joint name.
It imports no backend or robot-model package. The in-tree backend selection
helper resolves explicit `robot.joint_overlays` package URIs; core never discovers
robot assets. Last overlay wins per field, including zero; omitted fields retain
the existing class fallback. Backends own unit conversion, mimic constraints,
application timing and solver readback. Command delay and torque constant remain
unconsumed and are logged as such. No general Channel A URDF patch is enabled.
