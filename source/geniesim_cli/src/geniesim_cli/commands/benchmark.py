# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""``geniesim benchmark {run,list,batch,generate-layouts,run-layouts,categories,robots,check-inference}`` — drive
``geniesim_benchmark/app/app.py`` from the CLI, plus a probe for the inference server it talks to.

The underlying invocation this verb wraps is::

    omni_python <geniesim_benchmark>/app/app.py \\
        --config <geniesim_benchmark>/config/<NAME>.yaml \\
        --benchmark.infer_host=<HOST:PORT> [other --key=value...]

The CLI layer here resolves ``<NAME>`` to a concrete YAML (basename,
substring, or a literal path), filters tasks by robot prefix /
category, and forwards any extra ``--key=value`` flags verbatim to
``app.py``'s ``ParameterServer.override_from_cli`` (see
``geniesim_benchmark/config/params.py``).

Configs follow ``<robot>_<category>_<task>.yaml`` in
``geniesim_benchmark/config/``. Splitting the stem on ``_`` and looking
up the second token in :data:`_KNOWN_CATEGORIES` gives us a simple,
deterministic tag inference without having to maintain per-file
metadata.

The task runtime selects the interpreter in the combo image: legacy Isaac
tasks use bundled 5.1 through ``omni_python``; the Newton wok task uses
system ``python3.12`` with Isaac 6.2 and ROS Jazzy. Generic
``GENIESIM_PY_CMD`` container defaults do not override this task selection.
A custom prepared Python path still applies. Newton tasks load their built ROS
overlay automatically before exec.
Native MuJoCo tasks use ``benchmark.python`` or the CLI interpreter and bypass
the Isaac and ROS bootstrap.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

from geniesim_cli import _env
from geniesim_cli._style import BOLD, CYAN, DIM, GREEN, MAGENTA, RED, RST, WHITE, YELLOW

# Second token in `<robot>_<category>_<task>.yaml`. Stable enough to
# hard-code: new categories are rare and an unknown one just shows up
# as the inferred robot — no failure, just a less precise tag.
_KNOWN_CATEGORIES: frozenset[str] = frozenset(
    {
        "if",
        "zeroshot",
        "s2r",
        "dev",
        "wbc",
        "lp",
        "manip",
        "finemanip",
        "dexmanip",
        "crsemb",
        "agent",
        "probe",
        "robust",
        "spatial",
        "vln",
        "pick",
        "ci",
    }
)

# --- check-inference -------------------------------------------------------
# The inference probe lives under ``benchmark`` because the payload it
# sends is generated from a benchmark task's observation, and the server
# it pings is the one ``--benchmark.infer_host`` points at — the probe is
# meaningless outside that context.
#
# The payload is a corobot JSON-RPC ``{"method": "infer", "params": {...}}``
# envelope. A canonical ``corobot_payload.pkl`` ships next to the probe
# script and is used by default; the caller can pass an explicit payload
# (e.g. a fresh ``debug_preview/debug_NNNN.pkl`` from the corobot policy's
# debug dump) to override it.
#
# The probe script and its default payload live *inside* the benchmark
# package (under ``<geniesim_benchmark>/scripts/``), so both are resolved
# via :func:`_benchmark_root` like ``config/`` and ``app.py`` — not via a
# repo-root walk.
_CHECK_SCRIPT_REL = "scripts/check_inference.py"
_CHECK_DEFAULT_PAYLOAD_REL = "scripts/corobot_payload.pkl"

# Flags on ``check_inference.py`` that consume a follow-on value when
# written as ``--flag value`` (vs ``--flag=value``). Used by the
# positional scanner so we don't mistake ``10.0.0.5`` (a host) for the
# payload pkl. Permissive superset is fine — unknown flags forward
# verbatim either way.
_CHECK_FLAGS_WITH_VALUES: frozenset[str] = frozenset({"--host", "--port", "--iters", "--max-dims"})


def _benchmark_root() -> Path:
    """Return ``<geniesim_benchmark>/`` (the dir with ``app/`` and ``config/``).

    Resolves via :func:`importlib.util.find_spec` so editable installs
    (``pip install -e source/geniesim_benchmark``) and wheel installs
    are both discovered without a hard-coded layout.
    """
    spec = importlib.util.find_spec("geniesim_benchmark")
    if spec is None or not spec.submodule_search_locations:
        print(f"{RED}❌ geniesim_benchmark is not importable.{RST}")
        print(f"   {DIM}Install it from a local checkout, then retry:{RST}")
        print(f"     {CYAN}pip install -e source/geniesim_benchmark{RST}")
        sys.exit(1)
    return Path(list(spec.submodule_search_locations)[0])


def _config_dir() -> Path:
    return _benchmark_root() / "config"


def _app_py() -> Path:
    return _benchmark_root() / "app" / "app.py"


def _python_cmd() -> str:
    """Select bundled Isaac 5.1, independently of the container's default."""
    override = _prepared_python_override()
    if override:
        return override
    if shutil.which("omni_python"):
        return "omni_python"
    raise RuntimeError("This Isaac benchmark requires bundled Isaac Sim 5.1 (omni_python). Use the combo image.")


def _prepared_python_override() -> str | None:
    """Keep custom prepared environments, not generic container defaults."""
    override = os.environ.get("GENIESIM_PY_CMD")
    if override and Path(override).name not in {"omni_python", "python3", "python3.12"}:
        return override
    return None


def _rt_python_cmd() -> str:
    """Select the system 3.12 ABI for Isaac 6.2, ROS Jazzy runtime."""
    override = _prepared_python_override()
    if override:
        return override
    interpreter = shutil.which("python3.12")
    if interpreter:
        return interpreter
    raise RuntimeError("This RT benchmark requires system Python 3.12 with Isaac Sim 6.2 and ROS Jazzy.")


def _all_configs() -> list[Path]:
    cfg = _config_dir()
    if not cfg.is_dir():
        return []
    # Skip private (_foo.yaml) files and the structural single-token
    # files `config.yaml` / `teleop.yaml` / `template.yaml` that hold
    # defaults / templates rather than runnable task configs.
    return sorted(
        p for p in cfg.iterdir() if p.is_file() and p.suffix == ".yaml" and not p.name.startswith("_") and "_" in p.stem
    )


def _split_name(stem: str) -> tuple[str | None, str | None]:
    """Parse a config stem into (robot, category).

    Matches ``<robot>_<category>_<rest>``. If the second token isn't a
    recognised category we treat the file as having only a robot tag.
    If the *first* token is a known category, the file has no robot
    (rare: ``lp_straighten_object.yaml``, ``spatial_*.yaml``).
    """
    parts = stem.split("_")
    if not parts:
        return None, None
    if parts[0] in _KNOWN_CATEGORIES:
        return None, parts[0]
    if len(parts) >= 2 and parts[1] in _KNOWN_CATEGORIES:
        return parts[0], parts[1]
    return parts[0], None


def _filter_configs(robot: str | None, category: str | None, needle: str | None) -> list[Path]:
    out: list[Path] = []
    for p in _all_configs():
        r, c = _split_name(p.stem)
        if robot and r != robot:
            continue
        if category and c != category:
            continue
        if needle and needle.lower() not in p.stem.lower():
            continue
        out.append(p)
    return out


def _resolve_config(arg: str) -> Path:
    """Resolve a user-supplied config arg to an absolute YAML path.

    Order of attempts:

    1. As-is, if it points to an existing file.
    2. ``<config_dir>/<arg>.yaml`` (auto-suffix).
    3. ``<config_dir>/<arg>``.
    4. Substring match against config stems — unique hit only.
    """
    p = Path(arg).expanduser()
    if p.is_file():
        return p.resolve()

    cfg_dir = _config_dir()
    candidates: list[Path] = []
    if not arg.endswith(".yaml"):
        candidates.append(cfg_dir / f"{arg}.yaml")
    candidates.append(cfg_dir / arg)

    for c in candidates:
        if c.is_file():
            return c.resolve()

    matches = [pp for pp in _all_configs() if arg.lower() in pp.stem.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print(f"{RED}❌ Ambiguous config '{arg}'. {len(matches)} matches:{RST}")
        for m in matches[:10]:
            print(f"     {WHITE}{m.stem}{RST}")
        if len(matches) > 10:
            print(f"     {DIM}... and {len(matches) - 10} more{RST}")
        sys.exit(1)

    print(f"{RED}❌ No config matches '{arg}'.{RST}")
    print(f"   {DIM}Try{RST} {CYAN}geniesim benchmark list{RST}{DIM} to enumerate.{RST}")
    sys.exit(1)


def _extract_flag(args: list[str], name: str) -> tuple[str | None, list[str]]:
    """Pop ``--name=value`` (or ``--name value``) out of ``args``.

    Returns the value (or None) and the remaining args. Used to peel
    short-hand flags like ``--infer-host`` off before we forward the
    rest to ``app.py``.
    """
    rest: list[str] = []
    val: str | None = None
    i = 0
    while i < len(args):
        a = args[i]
        if a == f"--{name}":
            if i + 1 < len(args):
                val = args[i + 1]
                i += 2
                continue
        elif a.startswith(f"--{name}="):
            val = a[len(f"--{name}=") :]
            i += 1
            continue
        rest.append(a)
        i += 1
    return val, rest






def _task_runtime(config_path: Path, forwarded: list[str]) -> str:
    try:
        from geniesim_benchmark.runtime import runtime_for_config
    except ModuleNotFoundError as error:
        if error.name != "geniesim_benchmark.runtime":
            raise
        # Older benchmark wheels expose only the Isaac app.
        return "isaac"
    return runtime_for_config(config_path, forwarded)


def is_native_run(args: list[str]) -> bool:
    """Identify explicit native runs before bootstrapping unrelated engines."""
    if len(args) < 2 or args[0] != "run":
        return False
    _, arguments = _extract_flag(args[1:], "infer-host")
    if not arguments:
        return False
    if importlib.util.find_spec("geniesim_benchmark") is None:
        return False
    from geniesim_benchmark.runtime import runtime_for_config

    return runtime_for_config(_resolve_config(arguments[0]), arguments[1:]) == "mujoco"


def _native_python(config_path: Path, forwarded: list[str]) -> str:
    """Select a configured native interpreter without consulting Isaac wrappers."""
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    override, _ = _extract_flag(forwarded, "benchmark.python")
    interpreter = override if override is not None else config.get("benchmark", {}).get("python")
    if interpreter is not None and (not isinstance(interpreter, str) or not interpreter):
        raise ValueError("benchmark.python must be a nonempty interpreter path")
    return interpreter or sys.executable or "python3"


def _build_app_cmd(config_path: Path, forwarded: list[str], infer_host: str | None) -> list[str]:
    runtime = _task_runtime(config_path, forwarded)
    if runtime == "mujoco":
        cmd = [
            _native_python(config_path, forwarded),
            str(_benchmark_root() / "scripts" / "run_mujoco.py"),
            "--config",
            str(config_path),
        ]
    elif runtime == "newton":
        cmd = [_rt_python_cmd(), "-m", f"geniesim_benchmark.{runtime}.runner", "--config", str(config_path)]
    elif runtime == "isaac":
        cmd = [_python_cmd(), str(_app_py()), "--config", str(config_path)]
    else:
        raise ValueError(f"Unsupported benchmark runtime: {runtime!r}")
    if infer_host:
        cmd.append(f"--benchmark.infer_host={infer_host}")
    cmd.extend(forwarded)
    return cmd


def _runtime_environment(config_path: Path, forwarded: list[str]) -> dict[str, str] | None:
    runtime = _task_runtime(config_path, forwarded)
    if runtime == "mujoco":
        return None
    if runtime != "newton" or "--dry-run" in forwarded:
        return dict(os.environ)
    from geniesim_benchmark.rt_engine.environment import runtime_environment

    return runtime_environment(config_path)


def _prepared_app_cmd(config_path: Path, forwarded: list[str], infer_host: str | None):
    cmd = _build_app_cmd(config_path, forwarded, infer_host)
    environment = _runtime_environment(config_path, forwarded)
    if environment is not None and not _prepared_python_override():
        virtualenv = environment.get("VIRTUAL_ENV")
        if (
            _task_runtime(config_path, forwarded) == "newton"
            and "--dry-run" not in forwarded
            and virtualenv
            and Path(virtualenv).name == "rt_python"
            and (Path(virtualenv) / "bin/python").is_file()
        ):
            cmd[0] = str(Path(virtualenv) / "bin/python")
    if environment is not None:
        # Render/engine children must inherit the task choice, not the image default.
        environment["GENIESIM_PY_CMD"] = cmd[0]
    return cmd, environment


def _print_usage() -> None:
    print(f"{BOLD}{MAGENTA}🧪 geniesim benchmark{RST}")
    print()
    print(f"{BOLD}Usage:{RST} geniesim benchmark {CYAN}<subcommand>{RST} [args...]")
    print()
    print(f"{BOLD}Subcommands:{RST}")
    print(f"  {CYAN}run{RST} {WHITE}<CONFIG>{RST} [--infer-host=H:P] [extras...]   ▶️  Run one task")
    print(f"  {CYAN}list{RST} [--robot=R] [--category=C] [SUBSTR]      📋 List configs")
    print(f"  {CYAN}batch{RST} --category=C [--robot=R] [extras...]    🧬 Run every matching task")
    print(f"  {CYAN}generate-layouts{RST} [args...]                       🧬 Generate a portable Wuji layout bundle")
    print(f"  {CYAN}run-layouts{RST} --bundle PATH --infer-host H:P --output-dir PATH [--dry-run]")
    print(f"     {DIM}Run each Wuji layout in a fresh ROS engine; --layout-ids 0,1 selects a subset.{RST}")
    print(f"  {CYAN}categories{RST}                                    🏷  Distinct categories + counts")
    print(f"  {CYAN}robots{RST}                                        🤖 Distinct robot prefixes + counts")
    print(f"  {CYAN}check-inference{RST} [PAYLOAD] [--infer-host=H:P] [extras...]")
    print(f"     {DIM}🔌 Probe the inference server with a corobot payload pkl.{RST}")
    print(f"     {DIM}Defaults to the bundled corobot_payload.pkl; pass a path to override.{RST}")
    print()
    print(f"{BOLD}CONFIG resolution:{RST}")
    print(f"  {DIM}1. literal path (absolute / relative){RST}")
    print(f"  {DIM}2. basename: 'g2op_if_pick_block_color' (auto-adds .yaml){RST}")
    print(f"  {DIM}3. unique substring match against config stems{RST}")
    print()
    print(f"{BOLD}Examples:{RST}")
    print(f"  {CYAN}geniesim benchmark run{RST} g2op_if_pick_block_color {WHITE}--infer-host=localhost:8999{RST}")
    print(f"  {CYAN}geniesim benchmark list{RST} --robot=g2op --category=if")
    print(f"  {CYAN}geniesim benchmark batch{RST} --category=if --robot=g2op --infer-host=localhost:8999")
    print(f"  {CYAN}geniesim benchmark check-inference{RST} --infer-host=localhost:8999   {DIM}# bundled payload{RST}")
    print(f"  {CYAN}geniesim benchmark check-inference{RST} debug_preview/debug_0001.pkl --infer-host=localhost:8999")
    print()
    print(f"{BOLD}Forwarded flags:{RST}")
    print(f"  {DIM}Anything after the CONFIG (or any unknown flag) is forwarded verbatim to app.py.{RST}")
    print(f"  {DIM}Example: '{RST}{CYAN}--app.headless=true{RST} {CYAN}--benchmark.num_episode=5{RST}{DIM}'.{RST}")


def _do_run(args: list[str]) -> None:
    if not args:
        print(f"{RED}❌ Missing CONFIG.{RST} {DIM}Try{RST} {CYAN}geniesim benchmark list{RST}.")
        sys.exit(1)

    infer_host, args = _extract_flag(args, "infer-host")
    config_arg = args[0]
    forwarded = args[1:]
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    config_path = _resolve_config(config_arg)
    cmd, environment = _prepared_app_cmd(config_path, forwarded, infer_host)

    print(f"{BOLD}{MAGENTA}🧪 geniesim benchmark run{RST}")
    print(f"   {DIM}Config:{RST}    {CYAN}{config_path}{RST}")
    if infer_host:
        print(f"   {DIM}Inference:{RST} {CYAN}{infer_host}{RST}")
    print(f"   {DIM}Python:{RST}    {CYAN}{cmd[0]}{RST}")
    print()
    print(f"   {YELLOW}$ {' '.join(cmd)}{RST}")
    print()
    # execvp so app.py owns the tty: Ctrl-C, stdout, exit code all
    # belong directly to the child. Matches `geniesim docker into`.
    if environment is not None:
        os.environ.update(environment)
    sys.stdout.flush()
    os.execvp(cmd[0], cmd)


def _do_list(args: list[str]) -> None:
    robot, args = _extract_flag(args, "robot")
    category, args = _extract_flag(args, "category")
    needle = args[0] if args else None

    configs = _filter_configs(robot, category, needle)

    label_bits: list[str] = []
    if robot:
        label_bits.append(f"robot={robot}")
    if category:
        label_bits.append(f"category={category}")
    if needle:
        label_bits.append(f"~{needle}")
    label = ", ".join(label_bits) or "all"

    if not configs:
        print(f"{YELLOW}⚠️  No configs match ({label}).{RST}")
        return

    print(f"{BOLD}{MAGENTA}📋 {len(configs)} configs ({label}){RST}")
    print()
    for p in configs:
        r, c = _split_name(p.stem)
        tag = f"{r or '-'}/{c or '-'}"
        print(f"  {DIM}[{tag:<14}]{RST}  {WHITE}{p.stem}{RST}")


def _do_categories(_: list[str]) -> None:
    cats: dict[str, int] = {}
    for p in _all_configs():
        _, c = _split_name(p.stem)
        if c:
            cats[c] = cats.get(c, 0) + 1
    if not cats:
        print(f"{YELLOW}⚠️  No configs found.{RST}")
        return
    print(f"{BOLD}{MAGENTA}🏷  Categories{RST}")
    print()
    for name in sorted(cats, key=lambda k: (-cats[k], k)):
        print(f"  {WHITE}{name:<10}{RST}  {DIM}{cats[name]:>4} configs{RST}")


def _do_robots(_: list[str]) -> None:
    robots: dict[str, int] = {}
    for p in _all_configs():
        r, _ = _split_name(p.stem)
        if r:
            robots[r] = robots.get(r, 0) + 1
    if not robots:
        print(f"{YELLOW}⚠️  No configs found.{RST}")
        return
    print(f"{BOLD}{MAGENTA}🤖 Robots{RST}")
    print()
    for name in sorted(robots, key=lambda k: (-robots[k], k)):
        print(f"  {WHITE}{name:<10}{RST}  {DIM}{robots[name]:>4} configs{RST}")


def _do_batch(args: list[str]) -> None:
    infer_host, args = _extract_flag(args, "infer-host")
    robot, args = _extract_flag(args, "robot")
    category, args = _extract_flag(args, "category")

    if not category and not robot:
        print(f"{RED}❌ batch requires at least --category or --robot.{RST}")
        print(f"   {DIM}List options:{RST} {CYAN}geniesim benchmark categories{RST} / {CYAN}robots{RST}.")
        sys.exit(1)

    configs = _filter_configs(robot, category, None)
    if not configs:
        print(f"{YELLOW}⚠️  No configs match the filter.{RST}")
        return

    forwarded = args[:]
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    print(f"{BOLD}{MAGENTA}🧬 geniesim benchmark batch{RST} {DIM}({len(configs)} tasks){RST}")
    if infer_host:
        print(f"   {DIM}Inference:{RST} {CYAN}{infer_host}{RST}")
    print()

    failed: list[str] = []
    for i, p in enumerate(configs, 1):
        cmd, environment = _prepared_app_cmd(p, forwarded, infer_host)
        print(f"  {BOLD}[{i}/{len(configs)}]{RST} {CYAN}{p.stem}{RST}")
        print(f"     {YELLOW}$ {' '.join(cmd)}{RST}")
        result = subprocess.run(cmd, **({"env": environment} if environment is not None else {}))
        if result.returncode != 0:
            print(f"     {RED}❌ exit {result.returncode}{RST}")
            failed.append(p.stem)
        else:
            print(f"     {GREEN}✅ ok{RST}")
        print()

    if failed:
        print(f"{RED}❌ {len(failed)}/{len(configs)} failed:{RST}")
        for f in failed:
            print(f"     {WHITE}{f}{RST}")
        sys.exit(1)
    print(f"{GREEN}✅ All {len(configs)} tasks succeeded.{RST}")


# --- check-inference -------------------------------------------------------


def _check_script() -> Path:
    """Locate the inference probe inside the benchmark package.

    Resolved via :func:`_benchmark_root` (``find_spec``) so it works for
    both editable and wheel installs without a repo-root walk.
    """
    return _benchmark_root() / _CHECK_SCRIPT_REL


def _python_check_cmd() -> str:
    """Pick the interpreter that runs ``check_inference.py``.

    Pure-Python deps (msgpack, numpy, websockets) — no Isaac Sim
    bootstrap needed, so we deliberately do *not* go through
    :func:`_python_cmd`. Plain ``python3`` keeps the probe snappy.
    """
    return shutil.which("python3") or sys.executable or "python3"


def _has_flag(args: list[str], *names: str) -> bool:
    """True iff any ``--name`` / ``--name=...`` is present in ``args``."""
    for a in args:
        for n in names:
            if a == f"--{n}" or a.startswith(f"--{n}="):
                return True
    return False


def _scan_inference_positional(args: list[str]) -> tuple[str | None, list[str]]:
    """Pull the first positional out of a flag/positional argv.

    ``--flag value`` pairs (where ``flag`` is in
    :data:`_CHECK_FLAGS_WITH_VALUES`) stay glued so we don't mistake
    e.g. ``10.0.0.5`` for the payload pkl. Everything else that
    doesn't start with ``-`` is the payload (first wins).
    """
    positional: str | None = None
    others: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a in _CHECK_FLAGS_WITH_VALUES:
            others.append(a)
            if i + 1 < len(args):
                others.append(args[i + 1])
                i += 2
                continue
        elif a.startswith("-"):
            others.append(a)
        elif positional is None:
            positional = a
        else:
            others.append(a)
        i += 1
    return positional, others


def _resolve_payload(arg: str) -> Path | None:
    """Return the absolute path of the payload pkl, or None if missing.

    Accepts an absolute path or one relative to cwd; also tries
    ``$GENIESIM_REPO_ROOT`` (where the corobot policy dumps
    ``debug_preview/``) when that override is set.
    """
    p = Path(arg).expanduser()
    if p.is_absolute():
        return p if p.is_file() else None
    if p.is_file():
        return p.resolve()
    override = _env.repo_root()
    if override:
        cand = (Path(override) / arg).resolve()
        if cand.is_file():
            return cand
    return None


def _do_check_inference(args: list[str]) -> None:
    script = _check_script()
    if not script.is_file():
        print(f"{RED}❌ Could not locate the inference probe: {script}{RST}")
        print(f"   {DIM}Install the benchmark package:{RST} {CYAN}pip install -e source/geniesim_benchmark{RST}")
        sys.exit(1)

    # Strip our --infer-host shorthand before scanning the rest of
    # argv so it doesn't muddle positional detection.
    infer_host, args = _extract_flag(args, "infer-host")

    positional, forwarded = _scan_inference_positional(args)

    if positional is None:
        # Fall back to the canonical payload bundled with the package.
        payload = _benchmark_root() / _CHECK_DEFAULT_PAYLOAD_REL
        if not payload.is_file():
            print(f"{RED}❌ Bundled default payload missing: {payload}{RST}")
            print(f"   {DIM}Pass an explicit corobot payload pkl, e.g. a fresh{RST}")
            print(f"   {DIM}{WHITE}debug_preview/debug_NNNN.pkl{RST}{DIM} from the corobot policy debug dump.{RST}")
            sys.exit(1)
    else:
        payload = _resolve_payload(positional)
        if payload is None:
            print(f"{RED}❌ Payload not found: {positional}{RST}")
            sys.exit(1)

    # Decompose --infer-host=H:P into --host / --port. Only fill slots
    # the user didn't already set explicitly — explicit wins.
    if infer_host:
        if not _has_flag(forwarded, "host"):
            if ":" in infer_host:
                host, port = infer_host.rsplit(":", 1)
                forwarded.extend(["--host", host])
                if port and not _has_flag(forwarded, "port"):
                    forwarded.extend(["--port", port])
            else:
                forwarded.extend(["--host", infer_host])

    cmd = [_python_check_cmd(), str(script), str(payload), *forwarded]

    print(f"{BOLD}{MAGENTA}🔌 geniesim benchmark check-inference{RST}")
    print(f"   {DIM}Script:{RST}  {CYAN}{script}{RST}")
    print(f"   {DIM}Payload:{RST} {CYAN}{payload}{RST}")
    print()
    print(f"   {YELLOW}$ {' '.join(cmd)}{RST}")
    print()
    # execvp so the probe owns the tty: emoji-rich output, exit code,
    # and Ctrl-C all go straight to the user.
    os.execvp(cmd[0], cmd)


def _do_layouts(module: str, args: list[str]) -> None:
    """Exec the benchmark's pure dispatch entry with the current prepared ROS Python."""
    package = _benchmark_root()
    if not (package / "newton" / f"{module}.py").is_file():
        print(f"{RED}❌ The benchmark package does not provide Newton layout commands.{RST}", file=sys.stderr)
        sys.exit(1)
    interpreter = os.environ.get("GENIESIM_PY_CMD") or sys.executable or "python3"
    command = [interpreter, "-m", f"geniesim_benchmark.newton.{module}", *args]
    os.execvp(interpreter, command)


def run(args: list[str]) -> None:
    if not args or args[0] in ("-h", "--help", "help"):
        _print_usage()
        return

    sub = args[0]
    rest = args[1:]

    if sub in ("generate-layouts", "run-layouts"):
        _do_layouts("layouts" if sub == "generate-layouts" else "layout_batch", rest)
        return
    if sub == "run":
        _do_run(rest)
        return
    if sub == "list":
        _do_list(rest)
        return
    if sub == "batch":
        _do_batch(rest)
        return
    if sub == "categories":
        _do_categories(rest)
        return
    if sub == "robots":
        _do_robots(rest)
        return
    if sub == "check-inference":
        _do_check_inference(rest)
        return

    # A bare config name dispatches to `benchmark run`.
    _do_run(args)
