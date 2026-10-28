# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""``geniesim data-collection {install,config,llm,replay,list,tasks,robots,run,build,up,into,down}`` —
drive the standalone ``source/data_collection`` automated trajectory-collection
module from the CLI.

Unlike ``geniesim benchmark`` (which execs a single ``app.py`` *inside* an
already-running container), ``data_collection`` is a host-orchestrated,
two-process module: ``scripts/run_data_collection.sh`` does ``docker run -d``
against its dedicated collection image and the in-container
entrypoint launches both ``data_collector_server.py`` and
``run_data_collection.py``. This verb therefore wraps those shell scripts on
the host rather than exec-ing a single process — see
``source/data_collection/AGENTS.md`` for the full driving model.

The editable ``geniesim-data-collection`` locator registers this source tree.
Without it, discovery walks up from cwd / this file (or ``$GENIESIM_REPO_ROOT``)
to ``source/data_collection``. Runtime dependencies are supplied by its image.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from geniesim_cli import _env
from geniesim_cli._style import BOLD, CYAN, DIM, GREEN, MAGENTA, RED, RST, WHITE, YELLOW

# Path of the module relative to a repo checkout, plus a marker that
# uniquely identifies the tree once we're standing on (or above) it.
_DC_REL = Path("source") / "data_collection"
_DC_MARKER = Path("scripts") / "run_data_collection.sh"
_TASKS_DIRNAME = "tasks"

# Container defaults match the public v2 Dockerfiles and launch scripts.
_DC_IMAGE = "registry.agibot.com/genie-sim/geniesim4-data-collection:latest"
_DC_CONTAINER = "geniesim4-data-collection"


def _dc_root() -> Path:
    """Return the ``source/data_collection`` directory.

    Resolution order: an importable ``data_collection`` package (parity with
    ``geniesim_benchmark``'s ``find_spec`` — present only when the locator shim
    is ``pip install -e``'d), then explicit ``$GENIESIM_REPO_ROOT``, then walk
    up from cwd and from this module's file. Each candidate is accepted only if
    it actually holds the module marker, so a stale install can't mislead us.
    """
    try:
        spec = importlib.util.find_spec("data_collection")
    except (ImportError, ValueError):
        spec = None
    if spec is not None and spec.submodule_search_locations:
        cand = Path(list(spec.submodule_search_locations)[0]).resolve()
        if (cand / _DC_MARKER).is_file():
            return cand

    override = _env.repo_root()
    if override:
        cand = Path(override).expanduser().resolve() / _DC_REL
        if (cand / _DC_MARKER).is_file():
            return cand

    for start in (Path.cwd().resolve(), Path(__file__).resolve()):
        for c in (start, *start.parents):
            if c.name == "data_collection" and (c / _DC_MARKER).is_file():
                return c
            cand = c / _DC_REL
            if (cand / _DC_MARKER).is_file():
                return cand

    print(f"{RED}❌ Could not locate 'source/data_collection'.{RST}")
    print(f"   {DIM}Set {BOLD}$GENIESIM_REPO_ROOT{RST}{DIM} or run from inside the repo.{RST}")
    sys.exit(1)


def _tasks_root() -> Path:
    return _dc_root() / _TASKS_DIRNAME


def _all_tasks() -> list[Path]:
    root = _tasks_root()
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*.json") if p.is_file())


def _split_task(path: Path, tasks_root: Path) -> tuple[str | None, str | None, str | None]:
    """Parse a task json path into ``(collection, task, robot)``.

    The on-disk layout is ``tasks/<collection>/<task>/<robot>/<name>.json``
    (e.g. ``geniesim_2025/sort_fruit/g2/sort_..._g2.json``). Shallower paths
    degrade gracefully: missing levels come back as ``None``.
    """
    try:
        parts = path.relative_to(tasks_root).parts
    except ValueError:
        return None, None, None
    dirs = parts[:-1]
    collection = dirs[0] if dirs else None
    if len(dirs) >= 3:
        return collection, dirs[-2], dirs[-1]
    if len(dirs) == 2:
        return collection, dirs[1], None
    return collection, None, None


def _extract_flag(args: list[str], name: str) -> tuple[str | None, list[str]]:
    """Pop ``--name=value`` (or ``--name value``) out of ``args``.

    Returns the value (or ``None``) and the remaining args. Mirrors
    ``commands/benchmark.py:_extract_flag``.
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


def _filter_tasks(robot: str | None, task: str | None, needle: str | None) -> list[tuple[Path, str | None, str | None]]:
    root = _tasks_root()
    out: list[tuple[Path, str | None, str | None]] = []
    for p in _all_tasks():
        _, tsk, rob = _split_task(p, root)
        if robot and (rob or "").lower() != robot.lower():
            continue
        if task and (tsk or "").lower() != task.lower():
            continue
        if needle:
            haystack = f"{tsk or ''}/{rob or ''}/{p.stem}".lower()
            if needle.lower() not in haystack:
                continue
        out.append((p, tsk, rob))
    return out


def _print_usage() -> None:
    print(f"{BOLD}{MAGENTA}🤖 geniesim data-collection{RST}")
    print()
    print(f"{BOLD}Usage:{RST} geniesim data-collection {CYAN}<subcommand>{RST} [args...]")
    print()
    print(f"{BOLD}Subcommands:{RST}")
    print(f"  {CYAN}install{RST} [--task-root=PATH] [--assets-root=PATH] [--dry-run]  📦 Install and save defaults")
    print(f"  {CYAN}config{RST}                                    ⚙️ Show saved settings and active task root")
    print(f"  {CYAN}llm list{RST} [--robot=R] [--json]                📋 List external LLM tasks")
    print(f"  {CYAN}llm show{RST} TASK                               🔎 Show metadata and available scenes")
    print(f"  {CYAN}llm run{RST} TASK [--scene=ID] [--run-times=N]    ▶️ Collect an LLM task (recording enabled)")
    print(
        f"  {CYAN}replay{RST} EPISODE_OR_TASK [--config=JSON]        🎬 Replay an episode or a task's latest recording"
    )
    print(f"  {CYAN}list{RST} [--robot=R] [--task=T] [SUBSTR]      📋 List task templates")
    print(f"  {CYAN}tasks{RST}                                     🏷  Distinct tasks (families) + counts")
    print(f"  {CYAN}robots{RST}                                    🤖 Distinct robots + counts")
    print(f"  {CYAN}run{RST} {WHITE}<LLM_TASK>{RST} [--scene=ID] [--run-times=N] [--dry-run]")
    print(f"     {DIM}Default: collect an LLM task. Use --task-json PATH for a JSON job/template.{RST}")
    print(f"  {CYAN}build{RST} [--image=TAG] [docker-build args...] 🔨 Build the collection/replay image")
    print(f"  {CYAN}up{RST} [--container-name=N]                  🐳 Create + enter interactive GUI container")
    print(
        f"  {CYAN}into{RST} {DIM}/{RST} {CYAN}down{RST} [--container-name=N]         🐳 Enter running container / stop + remove"
    )
    print()
    print(f"{BOLD}JSON task resolution (run --task-json PATH):{RST}")
    print(f"  {DIM}1. literal path (absolute / relative to the data_collection dir){RST}")
    print(f"  {DIM}2. basename: 'sort_the_fruit_into_the_box_apple_g2'{RST}")
    print(f"  {DIM}3. unique substring match against task stems{RST}")
    print()
    print(f"{BOLD}Examples:{RST}")
    print(f"  {CYAN}geniesim data-collection llm list{RST} --robot g2")
    print(
        f"  {CYAN}geniesim data-collection run{RST} place_long_block_across_two_supports_g2 --scene 2026121696_166a0404"
    )
    print(f"  {CYAN}geniesim data-collection replay{RST} place_long_block_across_two_supports_g2")
    print(f"  {CYAN}geniesim data-collection list{RST} --robot=g2 sort_fruit")
    print(
        f"  {CYAN}geniesim data-collection run{RST} --task-json sort_the_fruit_into_the_box_apple_g2 --headless --standalone"
    )
    print(
        f"  {CYAN}geniesim data-collection run{RST} --task-json apple_g2 --dry-run   {DIM}# resolve + print the command, don't launch{RST}"
    )


def _do_list(args: list[str]) -> None:
    robot, args = _extract_flag(args, "robot")
    task, args = _extract_flag(args, "task")
    needle = args[0] if args else None

    rows = _filter_tasks(robot, task, needle)

    bits: list[str] = []
    if robot:
        bits.append(f"robot={robot}")
    if task:
        bits.append(f"task={task}")
    if needle:
        bits.append(f"~{needle}")
    label = ", ".join(bits) or "all"

    if not rows:
        print(f"{YELLOW}⚠️  No tasks match ({label}).{RST}")
        return

    print(f"{BOLD}{MAGENTA}📋 {len(rows)} tasks ({label}){RST}")
    print()
    tags = [f"{fam or '-'}/{rob or '-'}" for _, fam, rob in rows]
    width = min(max((len(t) for t in tags), default=0), 44)
    for (p, _fam, _rob), tag in zip(rows, tags):
        print(f"  {DIM}[{tag:<{width}}]{RST}  {WHITE}{p.stem}{RST}")


def _do_tasks(_: list[str]) -> None:
    root = _tasks_root()
    counts: dict[str, int] = {}
    for p in _all_tasks():
        _, fam, _rob = _split_task(p, root)
        if fam:
            counts[fam] = counts.get(fam, 0) + 1
    if not counts:
        print(f"{YELLOW}⚠️  No tasks found.{RST}")
        return
    print(f"{BOLD}{MAGENTA}🏷  Tasks{RST}")
    print()
    width = min(max((len(k) for k in counts), default=0), 44)
    for name in sorted(counts, key=lambda k: (-counts[k], k)):
        print(f"  {WHITE}{name:<{width}}{RST}  {DIM}{counts[name]:>4} variants{RST}")


def _do_robots(_: list[str]) -> None:
    root = _tasks_root()
    counts: dict[str, int] = {}
    for p in _all_tasks():
        _, _fam, rob = _split_task(p, root)
        if rob:
            counts[rob] = counts.get(rob, 0) + 1
    if not counts:
        print(f"{YELLOW}⚠️  No tasks found.{RST}")
        return
    print(f"{BOLD}{MAGENTA}🤖 Robots{RST}")
    print()
    width = min(max((len(k) for k in counts), default=0), 44)
    for name in sorted(counts, key=lambda k: (-counts[k], k)):
        print(f"  {WHITE}{name:<{width}}{RST}  {DIM}{counts[name]:>4} tasks{RST}")


def _pop_flag(args: list[str], flag: str) -> tuple[bool, list[str]]:
    """Remove a boolean ``flag`` from ``args``; report whether it was present."""
    if flag in args:
        return True, [a for a in args if a != flag]
    return False, args


def _ambiguous(arg: str, matches: list[Path]) -> None:
    print(f"{RED}❌ Ambiguous task '{arg}'. {len(matches)} matches:{RST}")
    for m in matches[:10]:
        print(f"     {WHITE}{m.stem}{RST}")
    if len(matches) > 10:
        print(f"     {DIM}... and {len(matches) - 10} more{RST}")
    sys.exit(1)


def _rel_to_dc(resolved: Path, dc_root: Path) -> str:
    """Return ``resolved`` as a path relative to ``dc_root`` (posix).

    The container only bind-mounts the data_collection tree, so a task
    outside it would be invisible inside the container — reject early.
    """
    try:
        return resolved.relative_to(dc_root).as_posix()
    except ValueError:
        print(f"{RED}❌ Task is outside the data_collection tree:{RST} {resolved}")
        print(f"   {DIM}Only files under {BOLD}{dc_root}{RST}{DIM} are mounted into the container.{RST}")
        sys.exit(1)


def _resolve_task(arg: str, dc_root: Path) -> str:
    """Resolve a task arg to a path relative to ``dc_root``.

    Order: literal path (cwd / dc_root / tasks-root, with optional ``.json``
    suffix), then exact basename (stem) match, then unique substring match
    against task stems.
    """
    tasks_root = dc_root / _TASKS_DIRNAME

    p = Path(arg).expanduser()
    if p.is_file():
        return _rel_to_dc(p.resolve(), dc_root)

    suffixes = [""] if arg.endswith(".json") else ["", ".json"]
    for base in (dc_root, tasks_root):
        for suf in suffixes:
            cand = base / f"{arg}{suf}"
            if cand.is_file():
                return _rel_to_dc(cand.resolve(), dc_root)

    stem = arg[:-5] if arg.endswith(".json") else arg
    all_tasks = _all_tasks()
    exact = [t for t in all_tasks if t.stem == stem]
    if len(exact) == 1:
        return _rel_to_dc(exact[0].resolve(), dc_root)
    if len(exact) > 1:
        _ambiguous(arg, exact)

    matches = [t for t in all_tasks if arg.lower() in t.stem.lower()]
    if len(matches) == 1:
        return _rel_to_dc(matches[0].resolve(), dc_root)
    if len(matches) > 1:
        _ambiguous(arg, matches)

    print(f"{RED}❌ No task matches '{arg}'.{RST}")
    print(f"   {DIM}Try{RST} {CYAN}geniesim data-collection list{RST}{DIM} to enumerate.{RST}")
    sys.exit(1)


_RUN_BOOL_FLAGS = ("--headless", "--no-record", "--standalone")


def _resolve_assets(dc_root: Path, *, required: bool = True) -> str:
    """Use the standalone module's asset locator, without importing its runtime."""
    env = dict(os.environ)
    configured = _env.assets_path()
    if configured:
        env[_env.GENIESIM_ASSETS_PATH.name] = configured
    result = subprocess.run(
        [sys.executable, str(dc_root / "scripts" / "resolve_assets.py")],
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode:
        if required:
            print(f"{RED}{result.stderr.strip()}{RST}")
            sys.exit(result.returncode)
        return ""
    return result.stdout.strip()


def _resolve_llm_tasks(dc_root: Path) -> str:
    env = dict(os.environ)
    configured = _env.llm_task_root()
    if configured:
        env[_env.GENIESIM_LLM_TASK_ROOT.name] = configured
    result = subprocess.run(
        [sys.executable, str(dc_root / "scripts" / "resolve_llm_task_root.py")],
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode:
        print(f"{RED}{result.stderr.strip()}{RST}")
        sys.exit(result.returncode)
    return result.stdout.strip()


def _do_run(args: list[str]) -> None:
    json_task, rest = _extract_flag(args, "task-json")
    if json_task is not None:
        if not json_task or json_task.startswith("--"):
            print(f"{RED}--task-json needs a JSON file path.{RST}")
            sys.exit(1)
        _do_json_run([json_task, *rest])
        return
    if "--task-json" in args:
        print(f"{RED}--task-json needs a JSON file path.{RST}")
        sys.exit(1)
    from geniesim_cli.commands.data_collection_workflows import run

    run("run", args)


def _do_json_run(args: list[str]) -> None:
    if args and args[0] in ("-h", "--help", "help"):
        _print_usage()
        return

    dry_run, args = _pop_flag(args, "--dry-run")
    container_name, args = _extract_flag(args, "container-name")

    passthru: list[str] = []
    unknown: list[str] = []
    task_arg: str | None = None
    for a in args:
        if a in _RUN_BOOL_FLAGS:
            if a not in passthru:
                passthru.append(a)
        elif a.startswith("-"):
            unknown.append(a)
        elif task_arg is None:
            task_arg = a
        else:
            unknown.append(a)

    if task_arg is None:
        print(f"{RED}❌ Missing TASK.{RST} {DIM}Try{RST} {CYAN}geniesim data-collection list{RST}.")
        sys.exit(1)
    if unknown:
        print(f"{RED}❌ Unsupported option(s) for data-collection run:{RST} {WHITE}{' '.join(unknown)}{RST}")
        print(
            f"   {DIM}run_data_collection.sh only accepts{RST} "
            f"{CYAN}--headless --no-record --standalone --container-name{RST}"
            f"{DIM} (plus{RST} {CYAN}--dry-run{RST}{DIM}).{RST}"
        )
        print(
            f"   {DIM}Unlike{RST} {CYAN}benchmark run{RST}{DIM}, it does not forward arbitrary --key=value flags.{RST}"
        )
        sys.exit(1)

    dc_root = _dc_root()
    rel = _resolve_task(task_arg, dc_root)

    assets_src = _resolve_assets(dc_root, required=not dry_run)
    llm_tasks = _resolve_llm_tasks(dc_root)
    job = json.loads((dc_root / rel).read_text(encoding="utf-8"))
    if job.get("mode") == "llm" and job.get("run_mode") != "replay":
        from geniesim_cli.commands.data_collection_workflows import _validate_run_times

        if not llm_tasks:
            print(f"{RED}Error: No LLM task library configured{RST}", file=sys.stderr)
            raise SystemExit(1)
        task_name = job["task_name"]
        if Path(task_name).name != task_name or task_name in (".", ".."):
            print(f"{RED}Error: Use a task directory name{RST}", file=sys.stderr)
            raise SystemExit(1)
        try:
            _validate_run_times(
                dc_root, Path(llm_tasks) / task_name, job.get("scene_dir", "random"), job.get("run_times", 1)
            )
        except ValueError as exc:
            print(f"{RED}Error: {exc}{RST}", file=sys.stderr)
            raise SystemExit(1)

    cmd = ["bash", "scripts/run_data_collection.sh", "--task", rel, *passthru]
    if container_name:
        cmd += ["--container-name", container_name]

    flags = " ".join(passthru) + (f" --container-name {container_name}" if container_name else "")
    print(f"{BOLD}{MAGENTA}📥 geniesim data-collection run{RST}")
    print(f"   {DIM}Task:{RST}    {CYAN}{rel}{RST}")
    print(f"   {DIM}Assets:{RST}  {CYAN}{assets_src or 'assets not found — configure before launching'}{RST}")
    print(f"   {DIM}LLM tasks:{RST} {CYAN}{llm_tasks or '(not configured)'}{RST}")
    print(f"   {DIM}Workdir:{RST} {CYAN}{dc_root}{RST}")
    print(f"   {DIM}Flags:{RST}   {CYAN}{flags.strip() or '(none)'}{RST}")
    print()
    print(f"   {YELLOW}$ (cd {dc_root} && {' '.join(cmd)}){RST}")
    print()

    if dry_run:
        print(f"{DIM}--dry-run: resolved command only; not launching.{RST}")
        return

    env = dict(os.environ)
    env[_env.GENIESIM_ASSETS_PATH.name] = assets_src
    env["SIM_ASSETS"] = assets_src
    if llm_tasks:
        env[_env.GENIESIM_LLM_TASK_ROOT.name] = llm_tasks
        env["LLM_TASK_ROOT"] = llm_tasks
    result = subprocess.run(cmd, cwd=str(dc_root), env=env)
    sys.exit(result.returncode)


def _ensure_docker() -> None:
    """Refuse container ops inside the container or when docker is absent."""
    if _env.in_container():
        print(f"{RED}❌ container ops can't run inside the container.{RST}")
        print(f"   {DIM}Run this on the host machine.{RST}")
        sys.exit(1)
    if shutil.which("docker") is None:
        print(f"{RED}❌ docker is not on PATH.{RST}")
        sys.exit(1)


def _container_state(name: str) -> str | None:
    """Return the container's docker state (e.g. ``running``) or ``None`` if absent."""
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Status}}", name],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _do_install(args: list[str]) -> None:
    from geniesim_cli.commands.data_collection_workflows import run

    run("install", args)


def _do_build(args: list[str]) -> None:
    _ensure_docker()
    dry_run, args = _pop_flag(args, "--dry-run")
    image, args = _extract_flag(args, "image")
    image = image or os.environ.get("DATA_COLLECTION_IMAGE") or _DC_IMAGE
    dc_root = _dc_root()
    dockerfile = "docker/Dockerfile"
    cmd = ["docker", "build", "-f", dockerfile, "-t", image]
    cmd += [*args, "."]
    print(f"{BOLD}{MAGENTA}🔨 geniesim data-collection build{RST}")
    print(f"   {DIM}Dockerfile:{RST} {CYAN}{dc_root / dockerfile}{RST}")
    print(f"   {DIM}Tag:{RST}        {CYAN}{image}{RST}")
    print()
    print(f"   {YELLOW}$ (cd {dc_root} && {' '.join(cmd)}){RST}")
    print()
    if dry_run:
        print(f"{DIM}--dry-run: command only; not building.{RST}")
        return
    env = dict(os.environ)
    env["DOCKER_BUILDKIT"] = "1"
    result = subprocess.run(cmd, cwd=str(dc_root), env=env)
    sys.exit(result.returncode)


def _do_up(args: list[str]) -> None:
    _ensure_docker()
    container, _ = _extract_flag(args, "container-name")
    container = container or _DC_CONTAINER

    dc_root = _dc_root()
    assets_src = _resolve_assets(dc_root)
    llm_tasks = _resolve_llm_tasks(dc_root)
    cmd = ["bash", "scripts/start_gui.sh", "run", container]
    print(f"{BOLD}{MAGENTA}🐳 geniesim data-collection up{RST} {DIM}(interactive GUI container){RST}")
    print(f"   {DIM}Container:{RST} {CYAN}{container}{RST}")
    print(f"   {YELLOW}$ (cd {dc_root} && {' '.join(cmd)}){RST}")
    print()
    env = dict(os.environ)
    env[_env.GENIESIM_ASSETS_PATH.name] = assets_src
    env["SIM_ASSETS"] = assets_src
    if llm_tasks:
        env[_env.GENIESIM_LLM_TASK_ROOT.name] = llm_tasks
        env["LLM_TASK_ROOT"] = llm_tasks
    result = subprocess.run(cmd, cwd=str(dc_root), env=env)
    sys.exit(result.returncode)


def _do_into(args: list[str]) -> None:
    _ensure_docker()
    container, _ = _extract_flag(args, "container-name")
    container = container or _DC_CONTAINER
    state = _container_state(container)
    if state != "running":
        print(f"{RED}❌ Container {BOLD}{container}{RST}{RED} is not running (state: {state}).{RST}")
        print(f"   {DIM}Start it with{RST} {CYAN}geniesim data-collection up{RST}{DIM}.{RST}")
        sys.exit(1)
    dc_root = _dc_root()
    result = subprocess.run(["bash", "scripts/start_gui.sh", "exec", container], cwd=str(dc_root))
    sys.exit(result.returncode)


def _do_down(args: list[str]) -> None:
    _ensure_docker()
    container, _ = _extract_flag(args, "container-name")
    container = container or _DC_CONTAINER
    state = _container_state(container)
    if state is None:
        print(f"{DIM}No container named {BOLD}{container}{RST}{DIM}; nothing to do.{RST}")
        return
    print(f"{BOLD}{MAGENTA}🛑 geniesim data-collection down{RST}")
    print(f"   {DIM}Container:{RST} {CYAN}{container}{RST} {DIM}(state: {state}){RST}")
    if state == "running":
        subprocess.run(["docker", "stop", container], check=False)
    subprocess.run(["docker", "rm", "-f", container], check=False)
    print(f"   {GREEN}✅ Removed{RST}")


def run(args: list[str]) -> None:
    if not args or args[0] in ("-h", "--help", "help"):
        _print_usage()
        return

    sub = args[0]
    rest = args[1:]

    if sub in ("config", "llm", "replay"):
        from geniesim_cli.commands.data_collection_workflows import run

        run(sub, rest)
        return
    if sub == "install":
        _do_install(rest)
        return
    if sub == "list":
        _do_list(rest)
        return
    if sub == "tasks":
        _do_tasks(rest)
        return
    if sub == "robots":
        _do_robots(rest)
        return
    if sub == "run":
        _do_run(rest)
        return
    if sub == "build":
        _do_build(rest)
        return
    if sub == "up":
        _do_up(rest)
        return
    if sub == "into":
        _do_into(rest)
        return
    if sub == "down":
        _do_down(rest)
        return

    print(f"{RED}❌ Error: unknown data-collection subcommand '{sub}'{RST}")
    print()
    _print_usage()
    sys.exit(1)
