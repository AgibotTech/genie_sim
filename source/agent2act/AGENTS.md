# Agent2Act development

Read `README.md` for commands and `docs/architecture.md` for dependency direction.

- Use uv for dependencies and commands. Update `uv.lock` whenever dependency declarations change.
- Keep `domain`, `application` and `execution` independent of simulator, network and model adapters. Architecture tests enforce this.
- Register implementations in `bootstrap` through `ComponentRegistry`; do not add backend-name conditionals to the runtime or CLI.
- New planners/backends must implement `reset` and `close`. Register cleanup as soon as resources are created and test construction-failure cleanup.
- Keep the CoRobot wire envelope inside adapters/service. The application API accepts `Observation` and returns `ActionChunk`.
- Preserve explicit fixed/auto routing, reject unsupported constraints, and distinguish issued actions from observed success.
- Public core APIs require types. Run `make check` and `uv build` after architectural changes.
- Do not use live LLM credentials or GPU services in automated tests. Local protocol tests use ephemeral loopback ports.
- Do not treat successful fixture execution as a validated Isaac Sim task.
- For checker development: run large local labeled asset regressions, add relevant collision evidence, review screenshots and decisions, confirm the results before promotion, and preserve explicit version isolation for historical tasks.
- Do not add features to the upstream Genie Sim checkout for Agent2Act. Keep planner improvements here and preserve the original simulator tasks, action budgets, robot limits, and scoring.
