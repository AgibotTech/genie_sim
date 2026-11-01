# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""``geniesim tool`` — repo-maintenance abilities for contributors.

The ``tool`` verb is the noun-space for repo-level utilities that read
the source tree, check structural invariants, and (with ``--fix``)
regenerate derived artifacts in-place. Sub-commands here are
deliberately **for contributors**, not operators — they do not
install, configure, or run any simulation.

Sub-commands:
    deps-dag    Derive a Mermaid module-dependency DAG from every
                source/geniesim_*/pyproject.toml and either verify the
                rendered block in source/README.md is current
                (default / ``--check``) or regenerate it (``--fix``).
    ros-dag     Derive a Mermaid ROS-package DAG from every
                source/geniesim_ros/src/ros_ws/src/*/package.xml,
                splice into source/geniesim_ros/README.md. Replaces
                the old ``geniesim ros graph`` colcon-PNG pipeline —
                same domain, but rendered as a CI-checkable Mermaid
                block instead of a one-shot PNG.
                ``--check`` (default) / ``--fix``.
    docs        Repo-wide doc-coverage audit. Four independent checks
                (coverage / index / links / metadata) × three scopes
                (all / cli / ros). Reports violations and exits
                non-zero. There is no ``--fix`` — missing files and
                broken links can't be auto-generated.

Adding a sub-command:
    1. Implement ``_<name>_run(argv) -> int`` in this file.
    2. Add a row to ``_SUBCOMMANDS`` and the usage banner below.
    3. Document the sub-command in source/geniesim_cli/AGENTS.md §3.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from geniesim_cli._style import BOLD, CYAN, DIM, GREEN, MAGENTA, RED, RST, WHITE, YELLOW

_USAGE = f"""{BOLD}{MAGENTA}🧞 geniesim tool{RST} {DIM}— repo-maintenance abilities{RST}

Usage:
    {CYAN}geniesim tool deps-dag{RST}                Check source/README.md dep DAG is current
    {CYAN}geniesim tool deps-dag --fix{RST}          Regenerate the dep DAG in source/README.md

    {CYAN}geniesim tool ros-dag{RST}                 Check source/geniesim_ros/README.md ROS-pkg DAG
    {CYAN}geniesim tool ros-dag --fix{RST}           Regenerate the ROS-package DAG

    {CYAN}geniesim tool docs{RST}                    Full doc-coverage audit (all scopes, all checks)
    {CYAN}geniesim tool docs --quiet{RST}            Silent on success (CI hook form)
    {CYAN}geniesim tool docs --scope ros{RST}        Limit to the ROS workspace
    {CYAN}geniesim tool docs --scope cli{RST}        Limit to the Python peers
    {CYAN}geniesim tool docs --check links{RST}      Only run the broken-link audit
    {CYAN}geniesim tool docs --check coverage{RST}   Only run the file-presence audit
    {CYAN}geniesim tool docs --check index{RST}      Only run the index-coverage audit
    {CYAN}geniesim tool docs --check metadata{RST}   Only run the ROS package.xml metadata audit
        {DIM}--check / --scope may be repeated to stack.{RST}
"""


# --------------------------------------------------------------------------
# Architectural annotations — single source of truth for cross-cutting
# facts that aren't in pyproject.toml. Update here when re-tiering a peer
# or changing the refactor roadmap.
# --------------------------------------------------------------------------

_ANNOTATIONS: dict[str, dict] = {
    # status: "legacy" | "leaf" | "placeholder"
    # refactor_target: another peer this is planned to fold into
    # note: short text rendered in the node label
    # requires_agents: False → exempt from `tool docs` AGENTS.md presence
    #                  check (defaults True). The umbrella ships no code
    #                  beyond __version__, so AGENTS.md isn't authored.
    "geniesim": {"role": "umbrella", "note": "meta-package", "requires_agents": False},
    "geniesim_cli": {"role": "cli", "note": "console script"},
    "geniesim_benchmark": {
        "note": "Isaac Sim direct",
        "requires_agents": False,
    },
    "geniesim_ros": {"note": "⚡ RT Engine"},
    "geniesim_teleop": {"note": "🎮 VR / Pico"},
    "geniesim_world": {"status": "leaf", "note": "🌍 pano → 3D world (own env)"},
    "geniesim_dream": {"status": "leaf", "note": "🌙 world-model inference (own envs)"},
    "geniesim_scene_agent": {"note": "🏗️ scene gen (optional extra)"},
}

# Cross-package data-flow edges that are not in pyproject.toml deps but
# are real runtime contracts. Currently empty: every meaningful runtime
# coupling is now reflected in pyproject deps (e.g. geniesim_teleop →
# geniesim_ros). Add an entry here when a new runtime relationship
# exists *without* a pip-level dep. Each entry: (from, to, label, style).
# style: "data" | "refactor"
_RUNTIME_EDGES: list[tuple[str, str, str, str]] = []


def _autogen_markers(key: str) -> tuple[str, str]:
    """Marker pair for a given AUTOGEN region key. Used by every
    sub-command that splices a generated block into a markdown file —
    each key gets its own `<!-- AUTOGEN:<key> start/end -->` pair so
    multiple generators can coexist in the same target file."""
    return f"<!-- AUTOGEN:{key} start -->", f"<!-- AUTOGEN:{key} end -->"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _repo_root() -> Path | None:
    """Walk up from cwd looking for a ``source/`` dir that holds peer
    pyproject.toml files. Returns None when the user is outside a
    checkout — caller is responsible for the error message."""
    cur = Path.cwd().resolve()
    for candidate in (cur, *cur.parents):
        src = candidate / "source"
        if src.is_dir() and any(src.glob("geniesim*/pyproject.toml")):
            return candidate
    return None


def _read_pyproject(path: Path) -> dict:
    """Parse pyproject.toml. Uses stdlib ``tomllib`` on Python 3.11+,
    falls back to the ``tomli`` backport on 3.10. Returns the
    [project] table merged with the package name's annotation block;
    missing fields default to safe values."""
    try:
        import tomllib  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # type: ignore[import-not-found, no-redef]
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "geniesim tool deps-dag needs `tomllib` (Python 3.11+) "
                "or the `tomli` backport on older Pythons. Install with "
                "`pip install tomli` and re-run."
            ) from exc

    with path.open("rb") as fh:
        data = tomllib.load(fh)
    project = data.get("project", {}) or {}
    build = data.get("build-system", {}) or {}
    return {
        "name": project.get("name", path.parent.name),
        "deps": list(project.get("dependencies", []) or []),
        "build_deps": list(build.get("requires", []) or []),
        "extras": {k: list(v or []) for k, v in (project.get("optional-dependencies", {}) or {}).items()},
    }


def _strip_specifier(req: str) -> str:
    """``geniesim_cli>=3.2.0`` -> ``geniesim_cli``. Lightweight, only
    handles the operators that appear in our pyproject files."""
    for op in (">=", "<=", "==", "~=", ">", "<", "!=", "["):
        idx = req.find(op)
        if idx != -1:
            req = req[:idx]
    return req.strip()


def _collect_peers(repo_root: Path) -> dict[str, dict]:
    """Glob source/geniesim_*/pyproject.toml, merge with annotations."""
    out: dict[str, dict] = {}
    for pj in sorted((repo_root / "source").glob("geniesim*/pyproject.toml")):
        info = _read_pyproject(pj)
        name = info["name"]
        ann = _ANNOTATIONS.get(name, {})
        out[name] = {**info, **ann, "_dir": pj.parent.name}
    return out


# --------------------------------------------------------------------------
# Mermaid emission
# --------------------------------------------------------------------------


def _node_id(name: str) -> str:
    """Mermaid id: dots / dashes are unsafe, the underscore is fine."""
    return name.replace("-", "_")


def _node_line(name: str, info: dict) -> str:
    note = info.get("note", "")
    role = info.get("role")
    label_top = f"<b>{name}</b>"
    label_bot = f"<br/>{note}" if note else ""
    return f'  {_node_id(name)}["{label_top}{label_bot}"]'


def _emit_mermaid(peers: dict[str, dict]) -> str:
    """Produce the full ```mermaid ... ``` fenced block."""
    lines: list[str] = []
    lines.append("```mermaid")
    lines.append("graph TD")
    lines.append("  %% Auto-generated by `geniesim tool deps-dag --fix`.")
    lines.append("  %% Edges: -->|build| build-system requires; ==>|exec| runtime deps;")
    lines.append("  %%        -.->|[X] extra| optional extras.")
    lines.append("  %% Source: source/geniesim_*/pyproject.toml + _ANNOTATIONS in")
    lines.append("  %% source/geniesim_cli/src/geniesim_cli/commands/tool.py.")
    lines.append("")

    # Stable node order: umbrella + cli first, then alphabetical.
    head = [n for n in ("geniesim", "geniesim_cli") if n in peers]
    rest = sorted(n for n in peers if n not in head)
    ordered = head + rest

    for name in ordered:
        lines.append(_node_line(name, peers[name]))
    lines.append("")

    # Build-system edges (thin `-->`) from pyproject `[build-system].requires`.
    # External-only `setuptools` / `wheel` requirements produce no edges
    # in practice; the edge fires when a peer needs another peer at
    # build time (e.g. setuptools_scm-style codegen helpers).
    emitted: set[tuple[str, str, str]] = set()
    for name in ordered:
        info = peers[name]
        for raw in info["build_deps"]:
            dep = _strip_specifier(raw)
            if dep in peers and dep != name:
                key = (name, dep, "build")
                if key in emitted:
                    continue
                emitted.add(key)
                lines.append(f"  {_node_id(name)} -->|build| {_node_id(dep)}")

    # Runtime / exec dep edges from `[project].dependencies`.
    #
    # Special case for the umbrella meta-package: it ships no code,
    # so its `[project].dependencies` are packaging declarations
    # ("install these alongside me"), not runtime imports. Render them
    # as build edges so the diagram reads "umbrella aggregates these"
    # (thin arrow) instead of "umbrella calls into these" (thick arrow).
    # Annotated peers with `role: umbrella` opt into this reclassification.
    for name in ordered:
        info = peers[name]
        is_umbrella = info.get("role") == "umbrella"
        arrow = "-->" if is_umbrella else "==>"
        label = "build" if is_umbrella else "exec"
        for raw in info["deps"]:
            dep = _strip_specifier(raw)
            if dep in peers and dep != name:
                key = (name, dep, label)
                if key in emitted:
                    continue
                emitted.add(key)
                lines.append(f"  {_node_id(name)} {arrow}|{label}| {_node_id(dep)}")

    # Optional dep edges (-.->) from [project.optional-dependencies].
    # Group by (src, dst) so multiple extras pointing at the same peer
    # collapse into one edge with a combined label (e.g. geniesim's
    # `[scene_agent]` and `[full]` both pull in geniesim_scene_agent).
    # Skip self-edges (`pkg[extra1,extra2]` recursive-extra pattern).
    optional_edges: dict[tuple[str, str], list[str]] = {}
    for name in ordered:
        info = peers[name]
        for extra_name, extra_deps in info["extras"].items():
            for raw in extra_deps:
                dep = _strip_specifier(raw)
                if dep in peers and dep != name:
                    optional_edges.setdefault((name, dep), []).append(extra_name)
    for (src, dst), extra_names in optional_edges.items():
        label = ",".join(extra_names)
        lines.append(f'  {_node_id(src)} -."[{label}] extra".-> {_node_id(dst)}')

    # Runtime data-flow edges (cyan dashed).
    for src, dst, label, style in _RUNTIME_EDGES:
        if src in peers and dst in peers:
            arrow = "-..->" if style == "data" else "-.->"
            lines.append(f"  {_node_id(src)} {arrow}|{label}| {_node_id(dst)}")

    # Refactor-target edges (orange dashed) from _ANNOTATIONS.
    for name, info in peers.items():
        target = info.get("refactor_target")
        if target and target in peers:
            lines.append(f"  {_node_id(name)} -.->|refactor: layer atop| {_node_id(target)}")

    # Styling.
    lines.append("")
    lines.append("  classDef legacy fill:#fff3cd,stroke:#856404,color:#856404")
    lines.append("  classDef leaf fill:#e9ecef,stroke:#6c757d,color:#495057")
    lines.append("  classDef placeholder fill:#f8d7da,stroke:#842029,color:#842029")
    lines.append("  classDef umbrella fill:#cfe2ff,stroke:#084298,color:#084298")
    legacy = [_node_id(n) for n, i in peers.items() if i.get("status") == "legacy"]
    leaf = [_node_id(n) for n, i in peers.items() if i.get("status") == "leaf"]
    placeholder = [_node_id(n) for n, i in peers.items() if i.get("status") == "placeholder"]
    umbrella = [_node_id(n) for n, i in peers.items() if i.get("role") == "umbrella"]
    if legacy:
        lines.append(f"  class {','.join(legacy)} legacy")
    if leaf:
        lines.append(f"  class {','.join(leaf)} leaf")
    if placeholder:
        lines.append(f"  class {','.join(placeholder)} placeholder")
    if umbrella:
        lines.append(f"  class {','.join(umbrella)} umbrella")

    lines.append("```")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# AUTOGEN-region splice
# --------------------------------------------------------------------------


def _splice_autogen(content: str, block: str, marker_start: str, marker_end: str, target_label: str) -> str:
    """Replace whatever sits between the AUTOGEN markers in `content`
    with `block` (no trailing newline duplication). Raises if the
    markers aren't found.

    Generic over marker pair so the same splicer is reused by every
    sub-command that writes a generated region — see ``_autogen_markers``."""
    start = content.find(marker_start)
    end = content.find(marker_end)
    if start == -1 or end == -1 or end < start:
        raise RuntimeError(
            f"AUTOGEN markers missing in {target_label}. "
            f"Add\n  {marker_start}\n  {marker_end}\n"
            f"where the generated block should live."
        )
    return content[: start + len(marker_start)] + "\n" + block + "\n" + content[end:]


# --------------------------------------------------------------------------
# Sub-commands
# --------------------------------------------------------------------------


def _deps_dag_run(argv: list[str]) -> int:
    if "--help" in argv or "-h" in argv:
        print(_USAGE)
        return 0

    fix = "--fix" in argv or "--write" in argv

    root = _repo_root()
    if root is None:
        print(
            f"{RED}❌ Could not locate a Genie Sim source checkout from {Path.cwd()}.{RST}\n"
            f"   {DIM}Run this inside a clone; it looks upward for source/geniesim_*/pyproject.toml.{RST}"
        )
        return 2

    target = root / "source" / "README.md"
    if not target.is_file():
        print(f"{RED}❌ source/README.md not found at {target}{RST}")
        return 2

    peers = _collect_peers(root)
    if not peers:
        print(f"{RED}❌ No source/geniesim_*/pyproject.toml files found under {root / 'source'}{RST}")
        return 2

    print(
        f"{BOLD}{MAGENTA}🧞 geniesim tool deps-dag{RST} " f"{DIM}({len(peers)} peers · {target.relative_to(root)}){RST}"
    )

    new_block = _emit_mermaid(peers)
    current = target.read_text()

    marker_start, marker_end = _autogen_markers("deps-dag")
    try:
        next_content = _splice_autogen(current, new_block, marker_start, marker_end, "source/README.md")
    except RuntimeError as exc:
        print(f"{RED}❌ {exc}{RST}")
        return 2

    if next_content == current:
        print(f"{GREEN}✅ deps-dag in source/README.md is up to date.{RST}")
        return 0

    if fix:
        target.write_text(next_content)
        print(f"{GREEN}✅ Rewrote source/README.md deps-dag block.{RST}")
        return 0

    # Drift detected, --fix not passed: report and exit non-zero.
    print(f"{YELLOW}⚠️  source/README.md deps-dag block is stale.{RST}")
    print(f"   {DIM}Run {CYAN}geniesim tool deps-dag --fix{RST}{DIM} to regenerate.{RST}")
    return 1


_SUBCOMMANDS = {
    "deps-dag": _deps_dag_run,
    "ros-dag": None,  # filled in after _ros_dag_run is defined below
    "docs": None,  # filled in after _docs_run is defined below
}


# ==========================================================================
# Sub-command: ros-dag — ROS-package dependency DAG inside source/geniesim_ros
# ==========================================================================
#
# Source of truth: every ``source/geniesim_ros/src/ros_ws/src/<pkg>/package.xml``.
# Each <depend>, <exec_depend>, <build_depend>, <buildtool_depend>, and
# <test_depend> tag becomes an edge — same build/exec arrow taxonomy as the
# Python deps-dag so the two diagrams read the same.
# --------------------------------------------------------------------------

_ROS_WORKSPACE_SRC = "source/geniesim_ros/src/ros_ws/src"
_ROS_README_TARGET = "source/geniesim_ros/README.md"

# package.xml depend tag → (arrow, label). Order matters: scanned in
# this order, first hit wins so build-tools always render as build edges
# even if a package also declares the same dep as exec.
_ROS_DEPEND_TAGS: tuple[tuple[str, str, str], ...] = (
    ("buildtool_depend", "-->", "buildtool"),
    ("build_depend", "-->", "build"),
    ("exec_depend", "==>", "exec"),
    ("depend", "==>", "exec"),  # `<depend>` = build+exec, render as exec
    ("test_depend", "-.->", "test"),
)


def _read_package_xml(path: Path) -> dict:
    """Parse a ROS 2 package.xml. Returns {"name": str, "deps":
    {tag: [pkg_name, ...]}}. Uses stdlib xml.etree only; depends on
    well-formed XML which every ROS package guarantees."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(path)
    root = tree.getroot()
    name_el = root.find("name")
    name = (name_el.text or path.parent.name).strip() if name_el is not None else path.parent.name
    deps: dict[str, list[str]] = {}
    for tag, _arrow, _label in _ROS_DEPEND_TAGS:
        deps[tag] = [el.text.strip() for el in root.findall(tag) if el.text]
    return {"name": name, "deps": deps}


def _collect_ros_packages(repo_root: Path) -> dict[str, dict]:
    """Glob `source/geniesim_ros/src/ros_ws/src/*/package.xml`,
    skipping `external/`. Returns {pkg_name: parsed_metadata}."""
    ws_src = repo_root / _ROS_WORKSPACE_SRC
    if not ws_src.is_dir():
        return {}
    out: dict[str, dict] = {}
    for pkg_xml in sorted(ws_src.rglob("package.xml")):
        rel = pkg_xml.relative_to(ws_src).parts
        if any(p in {"external", "build", "install", "devel", "devel_build", "devel_log"} for p in rel):
            continue
        info = _read_package_xml(pkg_xml)
        info["_dir"] = pkg_xml.parent.name
        out[info["name"]] = info
    return out


def _emit_ros_mermaid(pkgs: dict[str, dict]) -> str:
    """Produce the Mermaid block for the ROS-package DAG. Same arrow
    taxonomy as the Python deps-dag — build vs exec is the headline
    distinction."""
    lines: list[str] = []
    lines.append("```mermaid")
    lines.append("graph TD")
    lines.append("  %% Auto-generated by `geniesim tool ros-dag --fix`.")
    lines.append("  %% Edges: -->|build|/|buildtool| build-time deps;")
    lines.append("  %%        ==>|exec| runtime deps; -.->|test| test deps.")
    lines.append("  %% Source: source/geniesim_ros/src/ros_ws/src/*/package.xml.")
    lines.append("")

    ordered = sorted(pkgs)
    for name in ordered:
        lines.append(f'  {_node_id(name)}["<b>{name}</b>"]')
    lines.append("")

    emitted: set[tuple[str, str, str]] = set()
    for tag, arrow, label in _ROS_DEPEND_TAGS:
        for name in ordered:
            for dep in pkgs[name]["deps"].get(tag, []):
                # Skip system / external deps — we only edge between
                # packages that live in the same workspace.
                if dep not in pkgs or dep == name:
                    continue
                key = (name, dep, label)
                if key in emitted:
                    continue
                emitted.add(key)
                lines.append(f"  {_node_id(name)} {arrow}|{label}| {_node_id(dep)}")

    lines.append("```")
    return "\n".join(lines)


def _ros_dag_run(argv: list[str]) -> int:
    if "--help" in argv or "-h" in argv:
        print(_USAGE)
        return 0

    fix = "--fix" in argv or "--write" in argv

    root = _repo_root()
    if root is None:
        print(
            f"{RED}❌ Could not locate a Genie Sim source checkout from {Path.cwd()}.{RST}\n"
            f"   {DIM}Run this inside a clone; it looks upward for source/geniesim_*/pyproject.toml.{RST}"
        )
        return 2

    target = root / _ROS_README_TARGET
    if not target.is_file():
        print(f"{RED}❌ {_ROS_README_TARGET} not found at {target}{RST}")
        return 2

    pkgs = _collect_ros_packages(root)
    if not pkgs:
        print(f"{RED}❌ No ROS packages found under {root / _ROS_WORKSPACE_SRC}{RST}")
        return 2

    print(
        f"{BOLD}{MAGENTA}🧞 geniesim tool ros-dag{RST} "
        f"{DIM}({len(pkgs)} packages · {target.relative_to(root)}){RST}"
    )

    new_block = _emit_ros_mermaid(pkgs)
    current = target.read_text()

    marker_start, marker_end = _autogen_markers("ros-dag")
    try:
        next_content = _splice_autogen(current, new_block, marker_start, marker_end, _ROS_README_TARGET)
    except RuntimeError as exc:
        print(f"{RED}❌ {exc}{RST}")
        return 2

    if next_content == current:
        print(f"{GREEN}✅ ros-dag in {_ROS_README_TARGET} is up to date.{RST}")
        return 0

    if fix:
        target.write_text(next_content)
        print(f"{GREEN}✅ Rewrote {_ROS_README_TARGET} ros-dag block.{RST}")
        return 0

    print(f"{YELLOW}⚠️  {_ROS_README_TARGET} ros-dag block is stale.{RST}")
    print(f"   {DIM}Run {CYAN}geniesim tool ros-dag --fix{RST}{DIM} to regenerate.{RST}")
    return 1


_SUBCOMMANDS["ros-dag"] = _ros_dag_run


# ==========================================================================
# Sub-command: docs — repo-wide doc-coverage audit
# ==========================================================================

# Directories whose contents are NOT subject to the audit. Combined with
# rglob filters so we never recurse into them. Add new pollution
# patterns here when they show up.
_DOCS_SKIP_DIRS = {
    "build",
    "install",
    "dist",
    "log",
    "external",
    "node_modules",
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "env",
    "devel",
    "devel_build",
    "devel_log",
}


def _docs_should_skip(path: Path) -> bool:
    """True iff any path component is in the skip set or matches the
    *.egg-info / *.dist-info patterns. Cheap to evaluate."""
    for part in path.parts:
        if part in _DOCS_SKIP_DIRS:
            return True
        if part.endswith(".egg-info") or part.endswith(".dist-info"):
            return True
    return False


# Markdown link regex. Captures ``[label](path)`` — skips angle-bracket
# autolinks ``<https://...>`` and inline ``[ref][id]`` reference links.
_LINK_RE = re.compile(r"\[(?P<label>[^\]]+)\]\((?P<path>[^)#\s]+)(?:#[^)]*)?\)")


def _is_external_url(target: str) -> bool:
    return (
        target.startswith("http://")
        or target.startswith("https://")
        or target.startswith("mailto:")
        or target.startswith("ftp://")
        or target.startswith("file://")
    )


# --- scope enumerators ----------------------------------------------------


def _enumerate_cli_peers(repo: Path) -> list[Path]:
    """Return every first-class peer dir under source/geniesim_*/ that
    has a pyproject.toml. Sorted, stable order."""
    return sorted(p.parent for p in (repo / "source").glob("geniesim*/pyproject.toml"))


def _enumerate_ros_packages(repo: Path) -> list[Path]:
    """Every directory under geniesim_ros/src/ros_ws/src/ with a
    package.xml, minus the skip set."""
    ws_src = repo / "source" / "geniesim_ros" / "src" / "ros_ws" / "src"
    if not ws_src.is_dir():
        return []
    pkgs: list[Path] = []
    for pkg_xml in ws_src.rglob("package.xml"):
        if _docs_should_skip(pkg_xml.relative_to(ws_src)):
            continue
        pkgs.append(pkg_xml.parent)
    return sorted(pkgs)


# --- check: coverage ------------------------------------------------------


def _audit_coverage_cli(repo: Path, peers: dict[str, dict]) -> list[str]:
    """Every first-class peer ships README.md. AGENTS.md is required
    unless the peer's annotation block sets requires_agents=False."""
    violations: list[str] = []
    for name, info in sorted(peers.items()):
        pkg_dir = repo / "source" / info["_dir"]
        if not (pkg_dir / "README.md").is_file():
            violations.append(f"  source/{info['_dir']}/: missing README.md")
        requires_agents = info.get("requires_agents", True)
        if requires_agents and not (pkg_dir / "AGENTS.md").is_file():
            violations.append(f"  source/{info['_dir']}/: missing AGENTS.md")
    return violations


def _audit_coverage_ros(repo: Path) -> list[str]:
    """Every ROS package ships README.md + AGENTS.md (no exemptions —
    the ROS workspace contract is stricter than the Python peer
    contract)."""
    violations: list[str] = []
    for pkg in _enumerate_ros_packages(repo):
        rel = pkg.relative_to(repo)
        missing = []
        if not (pkg / "README.md").is_file():
            missing.append("README.md")
        if not (pkg / "AGENTS.md").is_file():
            missing.append("AGENTS.md")
        if missing:
            violations.append(f"  {rel}: missing {' + '.join(missing)}")
    return violations


# --- check: metadata -------------------------------------------------------

_ROS_EXPECTED_MAINTAINER_EMAIL = "yangwendisheng@agibot.com"
_ROS_EXPECTED_MAINTAINER_NAME = "Genie Sim Team"
_ROS_EXPECTED_LICENSE_FILE = "LICENSE"
_ROS_EXPECTED_LICENSE_TEXT = "Mozilla Public License Version 2.0"


def _audit_metadata_ros(repo: Path) -> list[str]:
    """Every ROS package.xml uses the shared maintainer + license lines
    and ships a LICENSE file alongside it. Enforces the convention
    documented in geniesim_cli/AGENTS.md."""
    import xml.etree.ElementTree as ET

    violations: list[str] = []
    for pkg in _enumerate_ros_packages(repo):
        rel = pkg.relative_to(repo)
        xml_path = pkg / "package.xml"
        if not xml_path.is_file():
            continue

        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError as exc:
            violations.append(f"  {rel}/package.xml: unparseable ({exc})")
            continue

        maintainers = root.findall("maintainer")
        if len(maintainers) != 1:
            violations.append(f"  {rel}/package.xml: expected exactly one <maintainer>, found {len(maintainers)}")
        else:
            m = maintainers[0]
            got_email = (m.get("email") or "").strip()
            got_name = (m.text or "").strip()
            if got_email != _ROS_EXPECTED_MAINTAINER_EMAIL or got_name != _ROS_EXPECTED_MAINTAINER_NAME:
                violations.append(
                    f"  {rel}/package.xml: <maintainer> must be "
                    f'email="{_ROS_EXPECTED_MAINTAINER_EMAIL}" text="{_ROS_EXPECTED_MAINTAINER_NAME}" '
                    f'(got email="{got_email}" text="{got_name}")'
                )

        licenses = root.findall("license")
        if len(licenses) != 1:
            violations.append(f"  {rel}/package.xml: expected exactly one <license>, found {len(licenses)}")
        else:
            lic = licenses[0]
            got_file = (lic.get("file") or "").strip()
            got_text = (lic.text or "").strip()
            if got_file != _ROS_EXPECTED_LICENSE_FILE or got_text != _ROS_EXPECTED_LICENSE_TEXT:
                violations.append(
                    f"  {rel}/package.xml: <license> must be "
                    f'file="{_ROS_EXPECTED_LICENSE_FILE}" text="{_ROS_EXPECTED_LICENSE_TEXT}" '
                    f'(got file="{got_file}" text="{got_text}")'
                )

        if not (pkg / _ROS_EXPECTED_LICENSE_FILE).is_file():
            violations.append(f'  {rel}: missing LICENSE file (referenced by <license file="LICENSE">)')

    return violations


# --- check: index ---------------------------------------------------------


def _read_link_targets(md: Path) -> set[Path]:
    """Parse markdown links in `md`, return the absolute resolved paths
    of every relative target (skipping http/https/mailto)."""
    if not md.is_file():
        return set()
    targets: set[Path] = set()
    for m in _LINK_RE.finditer(md.read_text()):
        raw = m.group("path").strip()
        if _is_external_url(raw):
            continue
        try:
            resolved = (md.parent / raw).resolve()
        except OSError:
            continue
        targets.add(resolved)
    return targets


def _audit_index_cli(repo: Path, peers: dict[str, dict]) -> list[str]:
    """source/AGENTS.md table should list every peer (README and/or
    AGENTS link target's parent matches the peer dir)."""
    index_md = repo / "source" / "AGENTS.md"
    if not index_md.is_file():
        return [f"  source/AGENTS.md is missing — can't audit the peer index"]
    targets = _read_link_targets(index_md)
    listed_dirs = {t.parent for t in targets if t.name in ("README.md", "AGENTS.md")}
    violations: list[str] = []
    for name, info in sorted(peers.items()):
        pkg_dir = (repo / "source" / info["_dir"]).resolve()
        if pkg_dir not in listed_dirs:
            violations.append(f"  source/AGENTS.md: no row for source/{info['_dir']}/")
    return violations


def _audit_index_ros(repo: Path) -> list[str]:
    """source/geniesim_ros/AGENTS.md packages table should list every
    ROS package."""
    index_md = repo / "source" / "geniesim_ros" / "AGENTS.md"
    if not index_md.is_file():
        return [f"  source/geniesim_ros/AGENTS.md is missing — can't audit the ROS package index"]
    targets = _read_link_targets(index_md)
    listed_dirs = {t.parent for t in targets if t.name in ("README.md", "AGENTS.md")}
    violations: list[str] = []
    for pkg in _enumerate_ros_packages(repo):
        if pkg.resolve() not in listed_dirs:
            violations.append(f"  source/geniesim_ros/AGENTS.md: no row for {pkg.relative_to(repo)}")
    return violations


# --- check: links ---------------------------------------------------------

# Files whose link audit we run. Everything else is ignored.
_LINK_AUDIT_GLOBS = (
    "AGENTS.md",
    "README.md",
    "source/AGENTS.md",
    "source/README.md",
    ".agent/*.md",
    "source/*/AGENTS.md",
    "source/*/README.md",
    "source/*/skills/*/SKILL.md",
    "source/*/skills/README.md",
    "source/geniesim_ros/AGENTS.md",
    "source/geniesim_ros/src/ros_ws/src/*/AGENTS.md",
    "source/geniesim_ros/src/ros_ws/src/*/README.md",
)


def _audit_links(repo: Path, scope: str) -> list[str]:
    """Walk every doc file in scope, parse markdown links, flag any
    relative target that doesn't resolve. Skips http(s)/mailto."""
    violations: list[str] = []

    files: list[Path] = []
    for pattern in _LINK_AUDIT_GLOBS:
        for hit in repo.glob(pattern):
            if not hit.is_file():
                continue
            if _docs_should_skip(hit.relative_to(repo)):
                continue
            # Scope filter.
            if scope == "ros" and "geniesim_ros" not in hit.parts:
                continue
            if scope == "cli" and "ros_ws" in hit.parts:
                continue
            files.append(hit)

    for md in sorted(set(files)):
        for m in _LINK_RE.finditer(md.read_text()):
            raw = m.group("path").strip()
            if _is_external_url(raw):
                continue
            try:
                target = (md.parent / raw).resolve()
            except OSError:
                violations.append(f"  {md.relative_to(repo)}: '{raw}' → invalid path")
                continue
            if not target.exists():
                violations.append(f"  {md.relative_to(repo)}: '{raw}' → not found")

    return violations


# --- argv parsing ---------------------------------------------------------


def _parse_docs_args(argv: list[str]) -> tuple[set[str], set[str], bool, bool]:
    """Returns (checks, scopes, quiet, help). Accepts ``--check X``,
    ``--check=X``, repeated; same for ``--scope``."""
    checks: set[str] = set()
    scopes: set[str] = set()
    quiet = False
    show_help = False

    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("-h", "--help"):
            show_help = True
            i += 1
        elif a in ("-q", "--quiet"):
            quiet = True
            i += 1
        elif a == "--check" and i + 1 < len(argv):
            checks.add(argv[i + 1])
            i += 2
        elif a.startswith("--check="):
            checks.add(a.split("=", 1)[1])
            i += 1
        elif a == "--scope" and i + 1 < len(argv):
            scopes.add(argv[i + 1])
            i += 2
        elif a.startswith("--scope="):
            scopes.add(a.split("=", 1)[1])
            i += 1
        else:
            print(f"{RED}❌ Unknown arg '{a}' for tool docs{RST}")
            print(_USAGE)
            sys.exit(2)

    if not checks:
        checks = {"coverage", "index", "links", "metadata"}
    if not scopes:
        scopes = {"all"}

    valid_checks = {"coverage", "index", "links", "metadata"}
    valid_scopes = {"all", "cli", "ros"}
    bad_checks = checks - valid_checks
    bad_scopes = scopes - valid_scopes
    if bad_checks:
        print(f"{RED}❌ Invalid --check value(s): {', '.join(sorted(bad_checks))}{RST}")
        print(f"   {DIM}Valid: {', '.join(sorted(valid_checks))}{RST}")
        sys.exit(2)
    if bad_scopes:
        print(f"{RED}❌ Invalid --scope value(s): {', '.join(sorted(bad_scopes))}{RST}")
        print(f"   {DIM}Valid: {', '.join(sorted(valid_scopes))}{RST}")
        sys.exit(2)

    # `all` is equivalent to {cli, ros} for enumeration purposes.
    if "all" in scopes:
        scopes = {"cli", "ros"}

    return checks, scopes, quiet, show_help


# --- entry point ----------------------------------------------------------


def _docs_run(argv: list[str]) -> int:
    if "-h" in argv or "--help" in argv:
        print(_USAGE)
        return 0

    checks, scopes, quiet, _ = _parse_docs_args(argv)

    root = _repo_root()
    if root is None:
        print(
            f"{RED}❌ Could not locate a Genie Sim source checkout from {Path.cwd()}.{RST}\n"
            f"   {DIM}Run this inside a clone; it looks upward for source/geniesim_*/pyproject.toml.{RST}"
        )
        return 2

    peers = _collect_peers(root) if "cli" in scopes else {}
    ros_pkgs = _enumerate_ros_packages(root) if "ros" in scopes else []

    if not quiet:
        scope_label = "+".join(sorted(scopes))
        check_label = "+".join(sorted(checks))
        print(f"{BOLD}{MAGENTA}🧞 geniesim tool docs{RST} " f"{DIM}(scope={scope_label}, check={check_label}){RST}")
        if "cli" in scopes:
            print(f"  {DIM}· {len(peers)} Python peer(s) under source/geniesim_*/{RST}")
        if "ros" in scopes:
            print(f"  {DIM}· {len(ros_pkgs)} ROS package(s) under geniesim_ros/ros_ws/src/{RST}")

    violations: list[str] = []

    if "coverage" in checks:
        if "cli" in scopes:
            violations.extend(_audit_coverage_cli(root, peers))
        if "ros" in scopes:
            violations.extend(_audit_coverage_ros(root))

    if "index" in checks:
        if "cli" in scopes:
            violations.extend(_audit_index_cli(root, peers))
        if "ros" in scopes:
            violations.extend(_audit_index_ros(root))

    if "links" in checks:
        # Link audit takes a scope string; pass the merged scope so a
        # single pass covers both worlds when --scope=all.
        link_scope = "all"
        if scopes == {"cli"}:
            link_scope = "cli"
        elif scopes == {"ros"}:
            link_scope = "ros"
        violations.extend(_audit_links(root, link_scope))

    if "metadata" in checks:
        if "ros" in scopes:
            violations.extend(_audit_metadata_ros(root))

    if violations:
        print(
            f"{RED}❌ tool docs found {len(violations)} violation(s):{RST}",
            file=sys.stderr,
        )
        for v in violations:
            print(v, file=sys.stderr)
        print(
            f"\n  {DIM}Fix:{RST}\n"
            f"   • {DIM}coverage{RST} → add the missing README.md / AGENTS.md\n"
            f"   • {DIM}index{RST}    → add a row to the parent AGENTS.md table\n"
            f"   • {DIM}links{RST}    → repoint or remove the broken markdown link\n"
            f"   • {DIM}metadata{RST} → align package.xml <maintainer>/<license> with the shared convention (see geniesim_cli/AGENTS.md)",
            file=sys.stderr,
        )
        return 1

    if not quiet:
        print(f"{GREEN}✅ docs audit clean: {len(violations)} violation(s).{RST}")
    return 0


_SUBCOMMANDS["docs"] = _docs_run


# --------------------------------------------------------------------------
# Dispatcher entry point
# --------------------------------------------------------------------------


def run(argv: list[str]) -> None:
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(_USAGE)
        sys.exit(0)

    sub = argv[0]
    handler = _SUBCOMMANDS.get(sub)
    if handler is None:
        print(f"{RED}❌ Unknown subcommand 'tool {sub}'{RST}")
        print()
        print(_USAGE)
        sys.exit(1)

    sys.exit(handler(argv[1:]))
