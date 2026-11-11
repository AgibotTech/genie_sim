# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""``geniesim teleop {run,bridge}`` — drive the ``geniesim_teleop`` package.

``run`` launches the VR / Pico teleoperation loop (``geniesim_teleop.teleop``);
``bridge`` launches the in-process image pub/sub bridge
(``geniesim_teleop.bridge``). Both wrap a module invocation::

    <python> -m geniesim_teleop.teleop  [--client_host ... --port ... ...]
    <python> -m geniesim_teleop.bridge  [--mode inprocess]

Unknown flags are forwarded verbatim to the underlying ``argparse`` in
each module, so this verb never has to mirror their option tables.

Controls use system Python 3.12 with ROS Jazzy in the Isaac Sim 6.2
environment. ``GENIESIM_PY_CMD`` can select a prepared interpreter explicitly.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys

from geniesim_cli._style import BOLD, CYAN, DIM, MAGENTA, RED, RST, WHITE, YELLOW


def _teleop_importable() -> bool:
    """True iff the ``geniesim_teleop`` package can be located."""
    try:
        return importlib.util.find_spec("geniesim_teleop") is not None
    except ModuleNotFoundError:
        return False


def _require_teleop() -> None:
    if not _teleop_importable():
        print(f"{RED}❌ geniesim_teleop is not importable.{RST}")
        print(f"   {DIM}Install it from a local checkout, then retry:{RST}")
        print(f"     {CYAN}pip install -e source/geniesim_teleop{RST}")
        sys.exit(1)


def _python_cmd() -> str:
    """Pick the interpreter used to launch the teleop modules."""
    override = os.environ.get("GENIESIM_PY_CMD")
    if override:
        return override
    interpreter = shutil.which("python3.12")
    if interpreter:
        return interpreter
    raise RuntimeError("Teleop requires system Python 3.12 with ROS Jazzy. Use geniesim docker6.2.")


def _exec_module(module: str, forwarded: list[str], label: str) -> None:
    _require_teleop()
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    cmd = [_python_cmd(), "-m", module, *forwarded]

    print(f"{BOLD}{MAGENTA}🎮 geniesim teleop {label}{RST}")
    print(f"   {DIM}Module:{RST} {CYAN}{module}{RST}")
    print(f"   {DIM}Python:{RST} {CYAN}{cmd[0]}{RST}")
    print()
    print(f"   {YELLOW}$ {' '.join(cmd)}{RST}")
    print()
    # execvp so the child owns the tty: Ctrl-C, stdout, exit code all
    # belong directly to the teleop process. Matches `geniesim benchmark run`.
    os.execvp(cmd[0], cmd)


def _print_usage() -> None:
    print(f"{BOLD}{MAGENTA}🎮 geniesim teleop{RST}")
    print()
    print(f"{BOLD}Usage:{RST} geniesim teleop {CYAN}<subcommand>{RST} [args...]")
    print()
    print(f"{BOLD}Subcommands:{RST}")
    print(f"  {CYAN}run{RST} [--client_host H:P] [--port N] [--robot_cfg F] [--device_type T]")
    print(f"     {DIM}▶️  Launch the VR / Pico teleoperation loop (geniesim_teleop.teleop).{RST}")
    print(f"  {CYAN}bridge{RST} [--mode inprocess]")
    print(f"     {DIM}🌉 Launch the in-process image pub/sub bridge (geniesim_teleop.bridge).{RST}")
    print()
    print(f"{BOLD}Common run flags{RST} {DIM}(forwarded verbatim to geniesim_teleop.teleop):{RST}")
    print(f"  {WHITE}--client_host{RST} {DIM}gRPC client host:port (default localhost:50051){RST}")
    print(f"  {WHITE}--host_ip{RST}     {DIM}VR host IP (auto-detected if omitted){RST}")
    print(f"  {WHITE}--port{RST}        {DIM}VR server port (default 8080){RST}")
    print(f"  {WHITE}--robot_cfg{RST}   {DIM}Robot config json (default G2_omnipicker.json){RST}")
    print(f"  {WHITE}--device_type{RST} {DIM}Teleop device (default pico){RST}")
    print()
    print(f"{BOLD}Examples:{RST}")
    print(f"  {CYAN}geniesim teleop run{RST} --device_type=pico --port=8080")
    print(f"  {CYAN}geniesim teleop bridge{RST} --mode inprocess")
    print()
    print(f"{DIM}Anything after the subcommand (or any unknown flag) is forwarded verbatim.{RST}")


def run(args: list[str]) -> None:
    if not args or args[0] in ("-h", "--help", "help"):
        _print_usage()
        return

    sub = args[0]
    rest = args[1:]

    if sub == "run":
        _exec_module("geniesim_teleop.teleop", rest, "run")
        return
    if sub == "bridge":
        _exec_module("geniesim_teleop.bridge", rest, "bridge")
        return

    print(f"{RED}❌ Unknown teleop subcommand '{sub}'.{RST}")
    print()
    _print_usage()
    sys.exit(1)
