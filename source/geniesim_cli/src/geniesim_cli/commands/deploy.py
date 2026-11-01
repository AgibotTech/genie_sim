# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""``geniesim deploy [MODULE]`` — build pure-Python wheels into ./deploy."""

from __future__ import annotations

import os
import sys

from geniesim_cli._lazy import distribution_source_root
from geniesim_cli._style import BOLD, CYAN, DIM, GREEN, MAGENTA, RED, RST, WHITE, YELLOW

DEPLOY_MODULES: dict[str, str] = {
    "geniesim": "geniesim",
    "geniesim_benchmark": "geniesim_benchmark",
    "geniesim_cli": "geniesim_cli",
    "geniesim_scene_agent": "geniesim_scene_agent",
    "geniesim_ros": "geniesim_ros",
    "geniesim_teleop": "geniesim_teleop",
}

# Modules under the `geniesim_*` umbrella that exist but deliberately
# don't ship a pip wheel via `geniesim deploy` — distributed
# out-of-band (HuggingFace / ModelScope datasets, conda env, etc.).
# Non-`geniesim_*` modules under `source/` (rlinf_geniesim,
# data_collection, scene_reconstruction, external) are out of scope for
# this verb entirely and not listed here.
SKIP_DEPLOY: frozenset[str] = frozenset({"geniesim_assets", "geniesim_world"})


def _clean_stale_build_artifacts(source_root) -> None:
    import shutil
    from pathlib import Path

    targets: list[Path] = []
    for egg_info in source_root.rglob("*.egg-info"):
        if egg_info.is_dir():
            targets.append(egg_info)
    for dirname in ("build", "dist"):
        path = source_root / dirname
        if path.is_dir():
            targets.append(path)

    if not targets:
        return

    removed: list[Path] = []
    blocked: list[tuple[Path, OSError]] = []
    for path in targets:
        try:
            shutil.rmtree(path)
            removed.append(path)
        except OSError as exc:
            blocked.append((path, exc))

    if blocked:
        print(f"   {RED}❌ Cannot remove stale build artifacts (permission denied):{RST}")
        for path, exc in blocked:
            print(f"      {BOLD}{path}{RST}  {DIM}({exc}){RST}")
        print()
        print(f"   {DIM}These are usually created by the dev container running pip as root.{RST}")
        rel = blocked[0][0].relative_to(source_root) if blocked[0][0].is_relative_to(source_root) else blocked[0][0]
        print(f"   {CYAN}sudo rm -rf {rel}{RST}")
        sys.exit(1)

    if removed:
        rel_paths = [str(p.relative_to(source_root)) if p.is_relative_to(source_root) else str(p) for p in removed]
        print(f"   {DIM}Cleaned stale artifacts:{RST} {', '.join(rel_paths)}")


def _deploy(module: str | None, *, reuse_cache: bool = False) -> None:
    import subprocess
    from pathlib import Path

    deploy_dir = Path.cwd() / "deploy"
    deploy_dir.mkdir(parents=True, exist_ok=True)

    if module is None:
        targets = list(DEPLOY_MODULES.keys())
    else:
        if module in SKIP_DEPLOY:
            print(
                f"{YELLOW}⚠️  Skipping {BOLD}{module}{RST}{YELLOW} — excluded from {BOLD}geniesim deploy{RST}{YELLOW}.{RST}"
            )
            print(f"   {DIM}Assets are distributed out-of-band, not as wheels.{RST}")
            return
        if module not in DEPLOY_MODULES:
            print(f"{RED}❌ Error: unknown module '{module}'{RST}")
            print()
            print(f"{CYAN}📦 Available modules:{RST}")
            for name in sorted(DEPLOY_MODULES):
                print(f"   {WHITE}{name}{RST}")
            sys.exit(1)
        targets = [module]

    print(f"{BOLD}{MAGENTA}🚀 geniesim deploy{RST}")
    print(f"   {DIM}Deploy directory:{RST} {CYAN}{deploy_dir}{RST}")
    print(f"   {DIM}Modules:{RST}          {CYAN}{', '.join(targets)}{RST}")
    print()

    for tgt in targets:
        source_root = Path(distribution_source_root(tgt))
        print(f"   {DIM}─ {tgt} source root:{RST} {CYAN}{source_root}{RST}")
        _clean_stale_build_artifacts(source_root)
        build_cmd = [sys.executable, "-m", "build", "--wheel", "--outdir", str(deploy_dir), str(source_root)]

        is_ros = tgt == "geniesim_ros"
        if is_ros:
            build_cmd.append("--no-isolation")
            print(f"   {DIM}Build isolation:{RST} {YELLOW}disabled (ROS/CMake needs ambient interpreter){RST}")

        print(f"   {YELLOW}📝 Building wheel for {BOLD}{tgt}{RST}{YELLOW} ...{RST}")

        if is_ros:
            ros_env = os.environ.copy()
            if reuse_cache:
                ros_env["GENIESIM_ROS_KEEP_BUILD_CACHE"] = "1"
                print(f"   {DIM}Cache:{RST}           {YELLOW}reusing (--reuse-cache){RST}")
            print(f"   {DIM}─────── colcon log ───────{RST}")
            r = subprocess.run(build_cmd, env=ros_env)
            print(f"   {DIM}──────────────────────────{RST}")
        else:
            r = subprocess.run(build_cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"   {RED}❌ Build failed:{RST}")
                print(r.stdout)
                print(r.stderr)

        if r.returncode != 0:
            print(f"   {RED}❌ Build failed (exit {r.returncode}){RST}")
            sys.exit(r.returncode)
        print(f"   {GREEN}✅ Wheel built ({tgt}){RST}")

    wheels = sorted(deploy_dir.glob("*.whl"))
    if not wheels:
        print(f"   {RED}❌ No wheels found after build{RST}")
        sys.exit(1)

    print()
    print(f"   {GREEN}📦 Deployed to {deploy_dir}/:{RST}")
    for whl in wheels:
        size_kb = whl.stat().st_size / 1024
        print(f"      {WHITE}{whl.name}{RST}  {DIM}({size_kb:.0f} KB){RST}")
    print()
    print(f"   {BOLD}🎉 Done!{RST}")


def _list() -> None:
    import contextlib
    import io
    from pathlib import Path

    print(f"{BOLD}{MAGENTA}📋 geniesim deploy list{RST}")
    print()
    print(f"{BOLD}Deploy candidates:{RST}")
    for name in sorted(DEPLOY_MODULES):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                root = Path(distribution_source_root(name))
            root_str = f"{CYAN}{root}{RST}"
        except SystemExit:
            root_str = f"{DIM}(not installed){RST}"
        except Exception as exc:
            root_str = f"{DIM}(error: {exc}){RST}"
        print(f"  {GREEN}✅ {WHITE}{name}{RST}  {DIM}→{RST} {root_str}")

    if SKIP_DEPLOY:
        print()
        print(f"{BOLD}Always skipped:{RST}")
        for name in sorted(SKIP_DEPLOY):
            print(f"  {YELLOW}⏭️  {WHITE}{name}{RST}  {DIM}(distributed out-of-band){RST}")


def _help() -> None:
    print(f"{BOLD}{MAGENTA}🚀 geniesim deploy{RST}")
    print()
    print(f"{BOLD}Usage:{RST} geniesim deploy {CYAN}[--reuse-cache] <MODULE|all>{RST}")
    print()
    print(f"{BOLD}Arguments:{RST}")
    print(f"  {CYAN}all{RST}               Build wheels for all deploy modules")
    print(f"  {CYAN}<MODULE>{RST}          Build wheel for a specific module")
    print(f"  {CYAN}list{RST}              List deploy candidates and skipped modules")
    print()
    print(f"{BOLD}Flags:{RST}")
    print(f"  {WHITE}--reuse-cache{RST}   {DIM}Skip wiping the colcon build cache (geniesim_ros only){RST}")
    print()
    print(f"{BOLD}Modules:{RST}")
    for name in sorted(DEPLOY_MODULES):
        print(f"  {WHITE}{name}{RST}")


def run(args: list[str]) -> None:
    if not args or args[0] in ("-h", "--help"):
        _help()
        return

    if args[0] == "list":
        _list()
        return

    reuse_cache = "--reuse-cache" in args
    args = [a for a in args if a != "--reuse-cache"]
    module = args[0] if args else None
    if module == "all":
        module = None
    _deploy(module, reuse_cache=reuse_cache)
