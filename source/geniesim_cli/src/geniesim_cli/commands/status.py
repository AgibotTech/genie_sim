# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""``geniesim status`` — health-check report across all geniesim distributions.

The canonical distribution table lives in ``geniesim_cli._status_spec`` and
is **derived** from each peer's ``pyproject.toml`` + the actual installed
package layout (``pkgutil.iter_modules``). The only hand-maintained state
is presentation (emojis) and the ~3 pip→import name aliases for packages
whose import name differs from the wheel name. Adding a new peer is now
a zero-edit change to this module: drop a new ``source/geniesim_*/``
directory with a pyproject and it shows up automatically.

``bootstrap`` re-uses the same spec + the ``inspect`` function below to
decide which peers still need ``pip install``.
"""

from __future__ import annotations

import importlib.util
import os
import sys

from geniesim_cli._lazy import probe_module
from geniesim_cli._status_spec import status_distributions
from geniesim_cli._style import BOLD, CYAN, DIM, GREEN, MAGENTA, RED, RST, WHITE, YELLOW


def _distributions() -> tuple[dict, ...]:
    """Return the (derived) status spec.

    Wrapped in a function so the cache built by :func:`status_distributions`
    is shared with ``bootstrap`` / ``doctor`` without re-deriving the spec.
    """
    return status_distributions()


# Back-compat alias — ``bootstrap``, ``doctor`` historically imported
# ``DISTRIBUTIONS`` as a tuple. Re-export the cached spec under the old
# name so those modules keep working without code changes.
DISTRIBUTIONS = _distributions()


def inspect(spec: dict) -> dict:
    """Gather installation + health info for a single distribution.

    Returns a dict with keys:
        installed:  bool
        version:    str | None
        editable:   bool | None     (None when undetermined)
        location:   str | None      (path to the package directory)
        import_ok:  bool | None     (None when not installed)
        import_err: str | None
        deps:       list[(pip_name, import_name, ok, err)]
        submodules: list[(name, ok, err)]
        extras:     list[(extra_label, [pip_names], [import_names], ok)]
    """
    import json
    from importlib.metadata import PackageNotFoundError, distribution

    info: dict = {
        "installed": False,
        "version": None,
        "editable": None,
        "location": None,
        "import_ok": None,
        "import_err": None,
        "deps": [],
        "submodules": [],
        "extras": [],
    }

    try:
        dist = distribution(spec["dist"])
    except PackageNotFoundError:
        return info

    info["installed"] = True
    info["version"] = dist.version

    try:
        raw = dist.read_text("direct_url.json")
        if raw:
            data = json.loads(raw)
            dir_info = data.get("dir_info") or {}
            info["editable"] = bool(dir_info.get("editable"))
    except (OSError, ValueError):
        info["editable"] = None

    spec_obj = importlib.util.find_spec(spec["top"])
    if spec_obj is not None:
        if spec_obj.submodule_search_locations:
            info["location"] = list(spec_obj.submodule_search_locations)[0]
        elif spec_obj.origin:
            info["location"] = spec_obj.origin

    ok, err = probe_module(spec["top"])
    info["import_ok"] = ok
    info["import_err"] = err

    for pip_name, import_name in spec["deps"]:
        ok, err = probe_module(import_name)
        info["deps"].append((pip_name, import_name, ok, err))

    for sub in spec["submodules"]:
        ok, err = probe_module(sub)
        info["submodules"].append((sub, ok, err))

    for label, pip_names, import_names in spec["extras"]:
        all_ok = all(probe_module(name)[0] for name in import_names)
        info["extras"].append((label, list(pip_names), list(import_names), all_ok))

    return info


def health_issues(spec: dict, info: dict) -> list[str]:
    """Shared status/doctor verdict; absence is allowed only for optional peers."""
    if not info["installed"]:
        return [] if spec.get("tier") == "optional" else ["not installed"]
    issues = ["import failed"] if info["import_ok"] is False else []
    issues.extend(f"runtime dependency: {name}" for name, _, ok, _ in info["deps"] if not ok)
    issues.extend(f"submodule: {name}" for name, ok, _ in info["submodules"] if not ok)
    return issues


def _render_distribution_dag(rows: list[tuple[dict, dict]]) -> None:
    """Render the distribution-level dependency DAG as an ASCII tree.

    Builds a directed graph from the ``deps`` field of each distribution
    spec — edges flow ``dependent → dependency``. Roots (nodes with no
    incoming edges) are walked depth-first using the same ``├──/└──/│``
    connectors as the rest of ``geniesim status``. When a node has already
    been printed earlier in the traversal (diamond), the second occurrence
    is suffixed with a dim ``↑`` marker to break the visual cycle without
    re-expanding the subtree.
    """
    adj: dict[str, list[str]] = {}
    health: dict[str, str] = {}
    emoji: dict[str, str] = {}
    incoming: dict[str, int] = {}

    for spec, info in rows:
        name = spec["dist"]
        emoji[name] = spec["emoji"]
        adj.setdefault(name, [])
        incoming.setdefault(name, 0)
        if not info["installed"]:
            # Optional peers (tier-2) get a softer "absent" badge in the
            # graph too — same logic as the report verdict.
            health[name] = "optional-absent" if spec.get("tier") == "optional" else "missing"
        elif info["import_ok"]:
            health[name] = "ok"
        else:
            health[name] = "broken"
        for pip_name, _import_name in spec["deps"]:
            adj[name].append(pip_name)
            incoming[pip_name] = incoming.get(pip_name, 0) + 1
            adj.setdefault(pip_name, [])
            emoji.setdefault(pip_name, "📦")

    roots = [n for n in adj if incoming.get(n, 0) == 0]
    if not roots:
        roots = list(adj.keys())

    declaration_order = {spec["dist"]: idx for idx, (spec, _info) in enumerate(rows)}

    def _sort_key(name: str) -> int:
        return declaration_order.get(name, 1_000_000)

    roots.sort(key=_sort_key)
    for name in adj:
        adj[name].sort(key=_sort_key)

    seen: set[str] = set()

    def _label(name: str, is_repeat: bool) -> str:
        glyph = emoji.get(name, "📦")
        h = health.get(name, "ok")
        if h == "missing":
            badge = f" {YELLOW}⚠️  missing{RST}"
        elif h == "optional-absent":
            badge = f" {DIM}⏭️  optional, not installed{RST}"
        elif h == "broken":
            badge = f" {RED}❌{RST}"
        else:
            badge = ""
        repeat = f" {DIM}↑{RST}" if is_repeat else ""
        return f"{glyph} {WHITE}{name}{RST}{badge}{repeat}"

    def _walk(name: str, prefix: str, is_last: bool, is_root: bool) -> None:
        branch = "" if is_root else ("└── " if is_last else "├── ")
        is_repeat = name in seen
        line_prefix = f"{DIM}{prefix}{branch}{RST}" if branch else ""
        print(f"{line_prefix}{_label(name, is_repeat)}")
        if is_repeat:
            return
        seen.add(name)
        children = adj.get(name, [])
        if not children:
            return
        if is_root:
            child_prefix = ""
        else:
            child_prefix = prefix + ("    " if is_last else "│   ")
        for idx, child in enumerate(children):
            _walk(child, child_prefix, idx == len(children) - 1, is_root=False)

    for r_idx, root in enumerate(roots):
        _walk(root, prefix="", is_last=(r_idx == len(roots) - 1), is_root=True)
        if r_idx != len(roots) - 1:
            print(f"{DIM}│{RST}")


def run(args: list[str]) -> None:
    """Pretty-printed health report for the geniesim multi-distribution stack."""
    import shutil
    from pathlib import Path

    print(f"{BOLD}{MAGENTA}🧞 geniesim status{RST} {DIM}— health check across distributions{RST}")
    print()

    overall_ok = True
    any_missing = False
    rows: list[tuple[dict, dict]] = []
    for spec in DISTRIBUTIONS:
        info = inspect(spec)
        rows.append((spec, info))
        if health_issues(spec, info):
            overall_ok = False
        if not info["installed"] and spec.get("tier") != "optional":
            any_missing = True

    print(f"{BOLD}📦 Distributions{RST}")
    last_dist_idx = len(rows) - 1
    for d_idx, (spec, info) in enumerate(rows):
        emoji = spec["emoji"]
        dist_name = spec["dist"]
        is_last_dist = d_idx == last_dist_idx
        dist_branch = "└──" if is_last_dist else "├──"
        dist_pipe = "    " if is_last_dist else "│   "

        if not info["installed"]:
            if spec.get("tier") == "optional":
                print(
                    f"{DIM}{dist_branch}{RST} {emoji} {WHITE}{dist_name}{RST}  {DIM}⏭️  not installed (optional){RST}  {DIM}— install with{RST} {CYAN}pip install -e \"source/geniesim/[{dist_name.replace('geniesim_', '')}]\"{RST}"
                )
            else:
                print(
                    f"{DIM}{dist_branch}{RST} {emoji} {WHITE}{dist_name}{RST}  {YELLOW}⚠️  not installed{RST}  {DIM}— run{RST} {CYAN}geniesim bootstrap{RST}"
                )
            if not is_last_dist:
                print(f"{DIM}│{RST}")
            continue

        version_str = f"{GREEN}{info['version']}{RST}"
        if info["editable"] is True:
            mode_str = f"{CYAN}editable{RST}"
        elif info["editable"] is False:
            mode_str = f"{DIM}wheel{RST}"
        else:
            mode_str = f"{DIM}?{RST}"

        if info["import_ok"]:
            health = f"{GREEN}✅ healthy{RST}"
        else:
            health = f"{RED}❌ import failed{RST}"

        print(
            f"{DIM}{dist_branch}{RST} {emoji} {WHITE}{dist_name}{RST} {version_str} {DIM}({mode_str}{DIM}){RST} {health}"
        )

        sections: list[tuple[str, list[tuple[str, str]]]] = []

        if info["location"]:
            sections.append(("location", [("ok", f"{BOLD}{info['location']}{RST}")]))

        if info["import_ok"] is False and info["import_err"]:
            sections.append(("error", [("err", f"{RED}{info['import_err']}{RST}")]))

        if info["deps"]:
            dep_rows: list[tuple[str, str]] = []
            for pip_name, import_name, ok, err in info["deps"]:
                if ok:
                    dep_rows.append(("ok", f"{GREEN}✅{RST} {WHITE}{pip_name}{RST} {DIM}(import {import_name}){RST}"))
                else:
                    dep_rows.append(("err", f"{RED}❌{RST} {WHITE}{pip_name}{RST} {DIM}(import {import_name}){RST}"))
                    if err:
                        dep_rows.append(("err", f"{RED}{err}{RST}"))
            sections.append(("runtime deps", dep_rows))

        if info["submodules"]:
            sub_rows: list[tuple[str, str]] = []
            for sub, ok, err in info["submodules"]:
                if ok:
                    sub_rows.append(("ok", f"{GREEN}✅{RST} {WHITE}{sub}{RST}"))
                else:
                    sub_rows.append(("err", f"{RED}❌{RST} {WHITE}{sub}{RST}"))
                    if err:
                        sub_rows.append(("err", f"{RED}{err}{RST}"))
            sections.append(("submodules", sub_rows))

        if info["extras"]:
            ext_rows: list[tuple[str, str]] = []
            for label, pip_names, _import_names, ok in info["extras"]:
                joined = " ".join(pip_names)
                if ok:
                    ext_rows.append(("ok", f"{GREEN}✅{RST} {WHITE}[{label}]{RST} {DIM}({joined}){RST}"))
                else:
                    ext_rows.append(("skip", f"{DIM}⏭️  [{label}]{RST} {DIM}not installed ({joined}){RST}"))
            sections.append(("optional extras", ext_rows))

        for s_idx, (section_name, items) in enumerate(sections):
            is_last_section = s_idx == len(sections) - 1
            sec_branch = "└──" if is_last_section else "├──"
            sec_pipe = "    " if is_last_section else "│   "
            print(f"{DIM}{dist_pipe}{sec_branch}{RST} {DIM}{section_name}{RST}")
            for i_idx, (_kind, text) in enumerate(items):
                is_last_item = i_idx == len(items) - 1
                item_branch = "└──" if is_last_item else "├──"
                print(f"{DIM}{dist_pipe}{sec_pipe}{item_branch}{RST} {text}")

        if not is_last_dist:
            print(f"{DIM}│{RST}")

    print()
    print(f"{BOLD}🕸️  Dependency graph{RST} {DIM}(distribution-level){RST}")
    _render_distribution_dag(rows)

    print()
    print(f"{BOLD}🔗 Console script{RST}")
    geniesim_path = shutil.which("geniesim")
    if geniesim_path:
        print(f"{DIM}└──{RST} {GREEN}✅{RST} {BOLD}{geniesim_path}{RST}")
    else:
        print(
            f"{DIM}└──{RST} {YELLOW}⚠️  not on PATH{RST} {DIM}(check{RST} {CYAN}{Path(sys.executable).parent}{RST}{DIM}){RST}"
        )

    print()
    print(f"{BOLD}🤖 ROS 2{RST}")
    ros_distro = os.environ.get("ROS_DISTRO")
    ros_version = os.environ.get("ROS_VERSION")
    ros_python = os.environ.get("ROS_PYTHON_VERSION")
    ros_localhost = os.environ.get("ROS_LOCALHOST_ONLY")
    ament_prefix = os.environ.get("AMENT_PREFIX_PATH", "")
    sourced = bool(ros_distro) and bool(ros_version)
    if sourced:
        ros_lines: list[str] = [
            f"{GREEN}✅ sourced{RST} {DIM}({WHITE}{ros_distro}{RST}{DIM} / v{ros_version}){RST}",
        ]
        if ros_python:
            ros_lines.append(f"{DIM}python:{RST}         ROS_PYTHON_VERSION={BOLD}{ros_python}{RST}")
        if ros_localhost:
            ros_lines.append(f"{DIM}localhost-only:{RST} ROS_LOCALHOST_ONLY={BOLD}{ros_localhost}{RST}")
        if ament_prefix:
            paths = ament_prefix.split(":")
            tail = f"  {DIM}(+{len(paths) - 1} more){RST}" if len(paths) > 1 else ""
            ros_lines.append(f"{DIM}AMENT_PREFIX:{RST}   {BOLD}{paths[0]}{RST}{tail}")
        for r_idx, txt in enumerate(ros_lines):
            branch = "└──" if r_idx == len(ros_lines) - 1 else "├──"
            print(f"{DIM}{branch}{RST} {txt}")
    else:
        print(
            f"{DIM}└──{RST} {YELLOW}⚠️  not sourced{RST} {DIM}— run{RST} {CYAN}source /opt/ros/<distro>/setup.bash{RST}"
        )

    print()
    print(f"{BOLD}🐍 Environment{RST}")
    from geniesim_cli import _env

    workspace = _env.workspace_or_cwd()
    ws_origin = _env.workspace_origin()
    no_color = os.environ.get("NO_COLOR")
    env_lines: list[str] = [
        f"{DIM}python:{RST}    {WHITE}{sys.version.split()[0]}{RST} {DIM}({sys.executable}){RST}",
        f"{DIM}workspace:{RST} {BOLD}{workspace}{RST} {DIM}({ws_origin}){RST}",
    ]
    if no_color:
        env_lines.append(f"{DIM}NO_COLOR:{RST}  {WHITE}{no_color}{RST}")
    for e_idx, txt in enumerate(env_lines):
        branch = "└──" if e_idx == len(env_lines) - 1 else "├──"
        print(f"{DIM}{branch}{RST} {txt}")
    print()

    shell = os.environ.get("SHELL", "")
    completion_enabled = False
    home = os.path.expanduser("~")
    if "zsh" in shell:
        rc = os.path.join(home, ".zshrc")
        hint = f'eval "$(geniesim completion zsh)"'
    else:
        rc = os.path.join(home, ".bashrc")
        hint = f'eval "$(geniesim completion bash)"'
    if os.path.isfile(rc):
        try:
            with open(rc) as f:
                completion_enabled = "geniesim completion" in f.read()
        except OSError:
            pass
    print(f"{BOLD}🐚 Shell completion{RST}")
    if completion_enabled:
        print(f"{DIM}└──{RST} {GREEN}✅ enabled{RST}")
    else:
        print(f"{DIM}└──{RST} {YELLOW}⚠️  not enabled{RST} {DIM}— add to {BOLD}{rc}{RST}{DIM}:{RST} {CYAN}{hint}{RST}")

    print()

    if overall_ok:
        print(f"{BOLD}{GREEN}🎉 All systems go!{RST}")
    else:
        print(f"{BOLD}{YELLOW}⚠️  Some components need attention (see above).{RST}")
        if any_missing:
            print(f"   {DIM}Run{RST} {CYAN}geniesim bootstrap{RST} {DIM}to install the missing distributions.{RST}")
    sys.exit(0)
