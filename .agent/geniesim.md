# geniesim — umbrella meta-package

> 🧭 **Canonical source**: [`source/geniesim/README.md`](../source/geniesim/README.md) is the source of truth for the umbrella distribution's deps + extras; [`source/geniesim/pyproject.toml`](../source/geniesim/pyproject.toml) is authoritative for the actual dep list.

This file is a 30-second pointer. Do not duplicate content here that lives at the canonical source — duplication is what makes dispatchers rot.

---

## What it is

`geniesim` is a PEP 621 meta-package: it ships no Python code beyond `__init__.py` (which only sets `__version__`). Its sole job is to pull in every peer distribution as required deps so `pip install geniesim` lands the full SDK in one shot.

## Where to look

| Topic | File |
|---|---|
| Install variants + dep list | [`source/geniesim/README.md`](../source/geniesim/README.md) |
| Authoritative dep list | [`source/geniesim/pyproject.toml`](../source/geniesim/pyproject.toml) |
| Authoritative version | [`source/geniesim/VERSION`](../source/geniesim/VERSION) |
| Per-peer dispatch table | [`source/AGENTS.md`](../source/AGENTS.md) |
| Fresh-machine setup | [`source/geniesim_cli/AGENTS.md` § 0](../source/geniesim_cli/AGENTS.md) |

## Invariants the rest of the repo relies on

- The umbrella ships **no SDK code** beyond `__version__`. Heavy runtime deps belong in the peer that needs them.
- Adding a new required peer means updating `pyproject.toml` `dependencies` **and** the `_STATUS_DISTRIBUTIONS` / `_INIT_TARGETS` tables in [`source/geniesim_cli/src/geniesim_cli/cli.py`](../source/geniesim_cli/src/geniesim_cli/cli.py) in the same diff. The CLI's `status` / `bootstrap` verbs are the test surface.
- Optional peers go behind `[project.optional-dependencies]` extras (`[scene_agent]`, `[full]`, …) and the `extras` field of the relevant `_STATUS_DISTRIBUTIONS` entry.
- The umbrella is **not on PyPI** — never suggest `pip install geniesim`. Always redirect through `geniesim bootstrap` (which installs editable from the local checkout).
