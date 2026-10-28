# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""geniesim command-line interface — thin dispatcher.

All subcommand logic lives in ``geniesim_cli.commands.*``.
This module is the console-script entry point only.
"""

from __future__ import annotations

import signal
import sys

from geniesim_cli._style import BOLD, CYAN, DIM, GREEN, MAGENTA, RED, RST, WHITE, YELLOW

# Verbs whose underlying implementation is still in flux. Hitting one of
# these prints a W.I.P. banner and waits for the user to press ENTER so
# the warning is acknowledged before the heavy work starts. Set
# ``GENIESIM_WIP_ACK=1`` (or pass ``--yes`` to the verb) to skip the
# prompt — useful in CI / scripted runs.
_WIP_VERBS: dict[str, str] = {
    "teleop": (
        "geniesim_teleop is undergoing refactor — APIs, device protocol, "
        "and CLI flags may change between commits without notice."
    ),
}


def _wip_prompt(verb: str, message: str) -> None:
    """Show a W.I.P. banner for ``verb`` and block until the user
    presses ENTER. No-ops when stdin isn't a TTY or when the user has
    set ``GENIESIM_WIP_ACK=1`` / passed ``--yes`` (consumed by the
    caller). Ctrl-C / EOF aborts with a clear message."""
    import os

    if os.environ.get("GENIESIM_WIP_ACK") == "1":
        return
    if not sys.stdin.isatty():
        # Non-interactive: print the banner once, do not block.
        print(f"{YELLOW}🚧 W.I.P. verb {BOLD}geniesim {verb}{RST}{YELLOW} — {message}{RST}")
        return

    bar = "─" * 64
    print(f"{YELLOW}{bar}{RST}")
    print(f"{YELLOW}🚧 {BOLD}geniesim {verb}{RST}{YELLOW} is a Work-In-Progress verb.{RST}")
    print(f"   {DIM}{message}{RST}")
    print(f"{YELLOW}{bar}{RST}")
    try:
        input(f"{BOLD}Press {CYAN}[ENTER]{RST}{BOLD} to continue, or Ctrl-C to abort: {RST}")
    except (KeyboardInterrupt, EOFError):
        print()
        print(f"{RED}❌ Aborted by user.{RST}")
        sys.exit(130)


# Verbs that can run without the rest of the tier-1 stack installed —
# they're the recovery / inspection surface and must work on a fresh
# install where siblings aren't here yet.
# Verbs that can run without the rest of the tier-1 stack installed —
# they're the recovery / inspection surface, host-side container
# orchestration, or contributor utilities that work directly on the
# source tree without importing siblings. None of these should trigger
# the auto-bootstrap prompt.
#
# In particular: `geniesim docker *` is **host-only** in the new
# model — the entrypoint inside the container installs every tier-1
# peer from the bind-mounted workspace, so the host CLI must never
# pip-install anything. `geniesim tool *` parses pyproject.toml /
# package.xml directly. `geniesim deploy` runs `python -m build`
# without importing the wheel's contents.
_BOOTSTRAP_SKIP_VERBS: frozenset[str] = frozenset(
    {
        "bootstrap",
        "status",
        "doctor",
        "version",
        "completion",
        "env",
        "help",
        "-h",
        "--help",
        "docker",
        "docker4.5",
        "docker5.1",
        "docker6.2",
        "post-train",
        "tool",
        "deploy",
        "data-collection",
    }
)


def _ensure_bootstrap(cmd: str) -> None:
    """First-invocation auto-bootstrap.

    Detect a fresh install (no tier-1 sibling distributions importable)
    and offer to run ``geniesim bootstrap`` before dispatching the
    user's actual command. Setuptools cmdclass install-hooks no longer
    fire under modern pip (PEP 660 editable, PEP 517 wheel), so this is
    the only deterministic place to intercept "first install" — the
    user's very next invocation after ``pip install -e source/geniesim_cli/``.

    Skipped for:
      * ``_BOOTSTRAP_SKIP_VERBS`` — recovery / inspection verbs that
        must work *before* siblings exist (otherwise we'd loop).
      * ``GENIESIM_SKIP_AUTOBOOT=1`` — escape hatch for CI / containers.
      * Non-TTY stdin — prints a hint and exits 1 (won't block scripted
        runs in a confused state).
    """
    import importlib.util
    import os

    if cmd in _BOOTSTRAP_SKIP_VERBS:
        return
    if os.environ.get("GENIESIM_SKIP_AUTOBOOT") == "1":
        return

    try:
        from geniesim_cli._tiers import tier1

        siblings = [p for p in tier1() if p != "geniesim_cli"]
    except Exception:
        # Tier read failed (e.g. running from a wheel with no source tree
        # and no installed `geniesim` umbrella) — can't determine what to
        # bootstrap, so defer to the user.
        return

    missing = [p for p in siblings if importlib.util.find_spec(p) is None]
    if not missing:
        return

    bar = "─" * 64
    print(f"{YELLOW}{bar}{RST}")
    print(f"{YELLOW}🪄 {BOLD}Fresh install detected.{RST}")
    print(f"   {DIM}Tier-1 peers missing:{RST} {', '.join(f'{WHITE}{p}{RST}' for p in missing)}")
    print(f"   {DIM}These are pulled in by{RST} {CYAN}geniesim bootstrap{RST}{DIM}.{RST}")
    print(f"{YELLOW}{bar}{RST}")

    if not sys.stdin.isatty():
        print(f"{YELLOW}⚠️  Non-interactive shell — refusing to auto-bootstrap.{RST}")
        print(
            f"   Run {CYAN}geniesim bootstrap{RST} manually, or set "
            f"{CYAN}GENIESIM_SKIP_AUTOBOOT=1{RST} to ignore this check."
        )
        sys.exit(1)

    try:
        answer = input(f"{BOLD}Run {CYAN}geniesim bootstrap{RST}{BOLD} now? " f"[Y/n]: {RST}").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print()
        print(f"{RED}❌ Aborted. Run {CYAN}geniesim bootstrap{RST} when ready.")
        sys.exit(130)

    if answer not in ("", "y", "yes"):
        print(f"{DIM}Skipped. Run {CYAN}geniesim bootstrap{RST}{DIM} when ready.{RST}")
        sys.exit(0)

    print()
    from geniesim_cli.commands import bootstrap as _b

    _b.run([])
    # Bootstrap returns; fall through and dispatch the user's command on
    # the freshly-installed stack.
    print()
    print(f"{GREEN}✅ Bootstrap done — continuing with {BOLD}geniesim {cmd}{RST}{GREEN}.{RST}")
    print()


def _print_usage() -> None:
    from geniesim_cli.commands.deploy import DEPLOY_MODULES

    print(f"{BOLD}{MAGENTA}🧞 geniesim{RST} {DIM}— Genie Sim CLI (geniesim_cli){RST}")
    print()
    print(f"{BOLD}Usage:{RST} geniesim {CYAN}<command>{RST} [args...]")
    print()
    print(f"{BOLD}Commands:{RST}")
    print(f"  {CYAN}ros build{RST} dev|release               ⚙️  Build ROS 2 colcon workspace")
    print(f"  {CYAN}ros build cleanup{RST}                   🧹 Remove colcon output dirs (with confirmation)")
    print(f"  {CYAN}ros graph{RST}                           📊 colcon graph text, then PNG (graphviz)")
    print(f"  {CYAN}deploy{RST} <MODULE|all> [--reuse-cache]   🚀 Build pure-Python wheel(s) into ./deploy")
    print(f"  {CYAN}deploy list{RST}                         📋 List deploy candidates")
    print(f"  {CYAN}benchmark run{RST} <CONFIG> [...]        🧪 Run a benchmark task (wraps app/app.py)")
    print(f"  {CYAN}benchmark list{RST} [--robot/--category] 📋 List task configs by robot / category")
    print(f"  {CYAN}benchmark batch{RST} --category=C [...]  🧬 Run every task in a category sequentially")
    print(f"  {CYAN}benchmark check-inference{RST} [...]     🔌 Probe the inference server with a saved payload")
    print(
        f"  {CYAN}teleop run{RST} [...]                    🎮 Launch the VR / Pico teleoperation loop {YELLOW}(🚧 W.I.P.){RST}"
    )
    print(
        f"  {CYAN}teleop bridge{RST} [...]                 🌉 Launch the in-process image pub/sub bridge {YELLOW}(🚧 W.I.P.){RST}"
    )
    print(f"  {CYAN}data-collection install{RST}              📦 Register the collection module from this checkout")
    print(f"  {CYAN}data-collection list{RST} [--robot/--task]   📋 List data_collection task templates")
    print(f"  {CYAN}data-collection llm list/show{RST}           📋 Browse external LLM tasks and scenes")
    print(f"  {CYAN}data-collection replay{RST} <TASK|EPISODE>  🎬 Replay a saved recording")
    print(
        f"  {CYAN}data-collection run{RST} <TASK> [...]        📥 Collect an LLM task; --task-json selects a JSON job"
    )
    print(f"  {CYAN}status{RST}                              🔍 Health-check all geniesim distributions")
    print(f"  {CYAN}doctor{RST}                              🩺 Diagnose & repair (status + rosdep + more)")
    print(
        f"  {CYAN}docker{RST} <build|up|down|into|logs>    🐳 Manage the Genie Sim container {DIM}(default → docker5.1){RST}"
    )
    print(
        f"  {CYAN}docker6.2{RST} <build|up|down|into|logs>  🐳 Isaac Sim 6.2 variant (geniesim4) {DIM}— behaviour unverified{RST}"
    )
    print(f"  {CYAN}docker5.1{RST} <build|up|down|into|logs>  🐳 Isaac Sim 5.1 variant (geniesim3, default)")
    print(
        f"  {CYAN}docker --profile{RST} post-train|post-train-teleop <command>  🐳 Post-training server / teleop image layer{RST}"
    )
    print(f"  {CYAN}docker4.5{RST} <build|up|down|into|logs>  🐳 Isaac Sim 4.5 variant (geniesim2 E.O.L.)")
    print(f"  {CYAN}post-train{RST} <server|teleop> [...]         🧪 Manage post-training simulation services{RST}")
    print(f"  {CYAN}bootstrap{RST}           🪄 Bootstrap / re-initialize the geniesim stack")
    print(f"  {CYAN}tool{RST} <deps-dag|ros-dag|docs> [--fix]   🛠️  Contributor repo-maintenance utilities")
    print(
        f"  {CYAN}dataset convert{RST} <FROM-to-TO> [...]    🗂️  Dataset format conversion {DIM}(agibot-to-lerobot){RST}"
    )
    print(f"  {CYAN}env{RST} [--all|--unset]                 🌍 Show GENIESIM_* env vars")
    print(
        f"  {CYAN}version{RST}                             📋 Show version information ({DIM}--bump|--sync|--check{RST})"
    )
    print(f"  {CYAN}completion{RST} bash|zsh                 🐚 Generate shell completion script")
    print()
    print(
        f"{DIM}🚧 W.I.P. verbs prompt for [ENTER] before running. Set {CYAN}GENIESIM_WIP_ACK=1{RST}{DIM} to skip.{RST}"
    )
    print()
    print(f"{BOLD}📦 Deploy modules:{RST}")
    for name in sorted(DEPLOY_MODULES):
        print(f"  {WHITE}{name}{RST}")
    print(f"  {DIM}(omit MODULE to deploy all){RST}")


def main(argv: list[str] | None = None) -> None:
    # Behave like a normal Unix tool when piped into `head`/`less`: restore the
    # default SIGPIPE handler so an early-closed stdout terminates us quietly
    # instead of raising BrokenPipeError mid-print. No-op where unavailable
    # (Windows / non-main-thread).
    try:
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except (AttributeError, ValueError):
        pass

    args = argv if argv is not None else sys.argv[1:]

    if not args or args[0] in ("-h", "--help", "help"):
        _print_usage()
        sys.exit(0)

    cmd = args[0]
    sub_args = args[1:]

    # W.I.P. gate — for verbs whose implementation is still in flux,
    # prompt the user for ENTER acknowledgement before dispatching.
    # ``--yes`` skips the prompt and is consumed before forwarding.
    if cmd in _WIP_VERBS:
        if "--yes" in sub_args:
            sub_args = [a for a in sub_args if a != "--yes"]
        else:
            _wip_prompt(cmd, _WIP_VERBS[cmd])

    # First-invocation auto-bootstrap. On a fresh install the tier-1
    # peers (`geniesim`, `geniesim_benchmark`, `geniesim_ros`,
    # `geniesim_assets`) aren't installed yet — offer to install them
    # before the user's command runs. Skipped for recovery verbs
    # (status / doctor / bootstrap / version / completion / env).
    _ensure_bootstrap(cmd)

    if cmd == "version":
        from geniesim_cli.commands import version as _m

        _m.run(sub_args)
        sys.exit(0)

    if cmd == "status":
        from geniesim_cli.commands import status as _m

        _m.run(sub_args)
        sys.exit(0)

    if cmd == "doctor":
        from geniesim_cli.commands import doctor as _m

        _m.run(sub_args)
        sys.exit(0)

    if cmd == "docker":
        from geniesim_cli.commands import docker as _m

        _m.run(sub_args, variant="5.1", invoked_as="docker")
        sys.exit(0)

    if cmd == "post-train":
        from geniesim_cli.commands import post_train as _m

        _m.run(sub_args)
        sys.exit(0)

    if cmd == "docker6.2":
        from geniesim_cli.commands import docker as _m

        _m.run(sub_args, variant="", invoked_as="docker6.2")
        sys.exit(0)

    if cmd == "docker4.5":
        from geniesim_cli.commands import docker as _m

        _m.run(sub_args, variant="4.5", invoked_as="docker4.5")
        sys.exit(0)

    if cmd == "docker5.1":
        from geniesim_cli.commands import docker as _m

        _m.run(sub_args, variant="5.1", invoked_as="docker5.1")
        sys.exit(0)

    if cmd == "env":
        from geniesim_cli.commands import env as _m

        _m.run(sub_args)
        sys.exit(0)

    if cmd == "bootstrap":
        from geniesim_cli.commands import bootstrap as _m

        _m.run(sub_args)
        sys.exit(0)

    if cmd == "tool":
        from geniesim_cli.commands import tool as _m

        _m.run(sub_args)
        sys.exit(0)

    if cmd == "dataset":
        from geniesim_cli.commands import dataset as _m

        _m.run(sub_args)
        sys.exit(0)

    if cmd == "completion":
        from geniesim_cli.commands import completion as _m

        _m.run(sub_args)
        sys.exit(0)

    if cmd == "deploy":
        from geniesim_cli.commands import deploy as _m

        _m.run(sub_args)
        sys.exit(0)

    if cmd == "benchmark":
        from geniesim_cli.commands import benchmark as _m

        _m.run(sub_args)
        sys.exit(0)

    if cmd == "teleop":
        from geniesim_cli.commands import teleop as _m

        _m.run(sub_args)
        sys.exit(0)

    if cmd == "data-collection":
        from geniesim_cli.commands import data_collection as _m

        _m.run(args[1:])
        sys.exit(0)

    if cmd == "ros":
        from geniesim_cli.commands import ros as _m

        _m.run(sub_args)
        sys.exit(0)

    print(f"{RED}❌ Error: unknown command '{cmd}'{RST}")
    print()
    _print_usage()
    sys.exit(1)


if __name__ == "__main__":
    main()
