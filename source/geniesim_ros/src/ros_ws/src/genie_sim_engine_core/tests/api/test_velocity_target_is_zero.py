# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""The PhysX velocity target must stay zero on every position-driven joint.

What this protects
------------------
``schema/joint_params.py`` converts a mujoco/newton sysid set onto PhysX by
folding mujoco's passive damping into the PhysX drive::

    mujoco:  tau = ke*(q* - q) - kd*qd - c*qd
    PhysX:   tau = kp*(q* - q) + d*(qd* - qd)
    hence:   d = damping_drive + damping

That identity is **exact, not approximate** -- but only because ``qd*`` is
zero. PhysX damps velocity *error*, so ``d*(qd* - qd)`` collapses to ``-d*qd``
at ``qd* = 0`` and the single PhysX term is then precisely the two mujoco
terms summed. Set a nonzero velocity target on such a joint and the folded
``damping`` starts driving the joint *towards* that speed, which is not what
was identified. The conversion would still run, still look right, and be
wrong by ``damping * qd*``.

Nothing else in the tree states that dependency, so this is the guard. The
day someone adds a nonzero velocity target to a position-driven joint, this
test is what says the fold stopped being exact.

The 4WS carve-out
-----------------
One writer is deliberately nonzero: the 4-wheel-steering command path sets
``targetVelocity`` from ``cmd_4ws_drive_vel``, because a road wheel is
velocity-commanded by design. Those joints classify as ``JK_CHASSIS_DRIVE`` /
``JK_CHASSIS_WHEEL`` (``genie_sim_engine_newton`` ``joint_classification``),
they are not position-driven, and no sysid overlay in the tree names one --
so the fold never reaches them. The carve-out is allowed by name, and the
name is asserted, so routing anything else through it fails here.

AST-only, in the manner of ``test_call_site_signatures.py``: importing these
modules needs Kit, USD and a GPU, none of which a precondition check should
require. Precision over reach -- each writer is resolved positively and
anything unresolvable is reported rather than assumed benign.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List, Tuple

import pytest

_SRC = Path(__file__).resolve().parents[3]
_PKG = "genie_sim_engine_isaacsim"

#: The one dict whose values may legitimately become a nonzero velocity
#: target: 4WS road-wheel commands. Named rather than pattern-matched so a
#: second velocity-commanded path has to be added here consciously.
_VELOCITY_COMMANDED_SOURCE = "cmd_4ws_drive_vel"


def _files() -> List[Path]:
    root = _SRC / _PKG / _PKG
    if not root.is_dir():
        pytest.skip(f"{_PKG} not checked out")
    return sorted(root.rglob("*.py"))


#: Calls that cannot change a value, only its shape, dtype or device. The
#: scan recurses through these and through nothing else: a wrapper that may
#: scale (``np.full``, ``* scale``) has to read as nonzero, or the guard
#: would bless the very thing it exists to catch. ``_view_input`` is the
#: numpy -> CUDA-torch transfer in ``kit/stage.py`` and returns its argument
#: unchanged for non-Newton engines.
_VALUE_PRESERVING = frozenset(
    {"expand_dims", "asarray", "array", "ascontiguousarray", "reshape", "ravel", "flatten", "astype", "_view_input", "float", "int", "float32", "float64"}
)

#: Calls that produce zeros regardless of their arguments.
_ZERO_PRODUCING = frozenset({"zeros", "zeros_like"})


def _is_zero(node: ast.AST) -> bool:
    """True only when ``node`` is *provably* zero, by construction."""
    if isinstance(node, ast.Constant):
        return node.value == 0
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return _is_zero(node.operand)
    if isinstance(node, ast.Call):
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
        if name in _ZERO_PRODUCING:
            return True
        if name in _VALUE_PRESERVING and node.args:
            return _is_zero(node.args[0])
    return False


def _in_4ws_loop(path: Path, node: ast.AST, tree: ast.AST) -> bool:
    """True when ``node`` sits inside a ``for ... in cmd_4ws_drive_vel`` body."""
    for loop in ast.walk(tree):
        if not isinstance(loop, (ast.For, ast.AsyncFor)):
            continue
        it = loop.iter
        # ``for name, value in cmd_4ws_drive_vel.items()`` and
        # ``for name in list(cmd_4ws_drive_vel)``.
        src = ast.dump(it)
        if f"'{_VELOCITY_COMMANDED_SOURCE}'" not in src:
            continue
        for inner in ast.walk(loop):
            if inner is node:
                return True
    return False


def _velocity_target_writes(path: Path) -> List[Tuple[int, ast.AST, bool]]:
    """``(lineno, value_node, is_4ws)`` for every velocity-target write."""
    tree = ast.parse(path.read_text(), filename=str(path))
    found: List[Tuple[int, ast.AST, bool]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not isinstance(fn, ast.Attribute):
            continue

        # (a) ``...TargetVelocityAttr().Set(x)`` / ``tvel_attr.Set(x)``
        if fn.attr == "Set" and node.args:
            recv = fn.value
            recv_name = ""
            if isinstance(recv, ast.Call) and isinstance(recv.func, ast.Attribute):
                recv_name = recv.func.attr
            elif isinstance(recv, ast.Name):
                recv_name = recv.id
            elif isinstance(recv, ast.Attribute):
                recv_name = recv.attr
            if "TargetVelocityAttr" in recv_name or recv_name in ("tvel_attr",):
                found.append((node.lineno, node.args[0], _in_4ws_loop(path, node, tree)))

        # (b) the tensor-API form, ``set_joint_velocity_targets(...)``
        if fn.attr == "set_joint_velocity_targets":
            args = list(node.args) + [kw.value for kw in node.keywords if kw.arg in ("velocities", None)]
            if not args:
                pytest.fail(f"{path.name}:{node.lineno}: set_joint_velocity_targets with no resolvable argument")
            found.append((node.lineno, args[0], _in_4ws_loop(path, node, tree)))
    return found


def test_the_damping_fold_precondition_holds():
    """Every position-drive velocity target is provably zero."""
    writes = [(f, ln, val, is_4ws) for f in _files() for ln, val, is_4ws in _velocity_target_writes(f)]
    assert writes, "no velocity-target write found at all -- the scan broke, not the tree"

    offenders = [
        f"{f.relative_to(_SRC)}:{ln}: velocity target is not provably zero ({ast.unparse(val)})"
        for f, ln, val, is_4ws in writes
        if not is_4ws and not _is_zero(val)
    ]
    assert not offenders, (
        "A nonzero velocity target on a position-driven joint breaks the exactness of\n"
        "d = damping_drive + damping in schema/joint_params.py (PHYSX_FIELD_MAP). Either\n"
        "keep the target at zero, or demote the fold to an approximation and say so in\n"
        "the converter's report:\n  " + "\n  ".join(offenders)
    )


def test_the_only_nonzero_target_is_the_4ws_wheel_command():
    """The carve-out is one path, and it is the velocity-commanded one.

    If a second writer starts setting nonzero targets, the fold's precondition
    has changed even though the first test still passes -- so the exception is
    pinned by name and counted, not merely tolerated.
    """
    nonzero = [
        (f.relative_to(_SRC), ln, ast.unparse(val))
        for f in _files()
        for ln, val, is_4ws in _velocity_target_writes(f)
        if not _is_zero(val)
    ]
    assert len(nonzero) == 1, f"expected exactly the 4WS drive-wheel write, got {nonzero}"
    _, _, expr = nonzero[0]
    assert "value" in expr, expr  # the commanded wheel speed, scaled

    # And it is reached only from the 4WS command dict.
    assert all(
        is_4ws
        for f in _files()
        for _, val, is_4ws in _velocity_target_writes(f)
        if not _is_zero(val)
    )


def test_the_scan_would_catch_a_nonzero_target(tmp_path: Path):
    """The guard's own sensitivity, on synthetic source.

    Both tests above pass today, which means neither one demonstrates that
    the scan can fail. This does: each writer shape is fed in with a zero and
    with a nonzero value, and the verdict has to flip.
    """
    shapes = [
        "drive.GetTargetVelocityAttr().Set({v})",
        "tvel_attr.Set({v})",
        "art_view.set_joint_velocity_targets(velocities={v})",
        "art_view.set_joint_velocity_targets({v})",
    ]
    for shape in shapes:
        for value, want_zero in (
            ("0.0", True),
            ("np.zeros(num_dof)", True),
            ("_view_input(np.expand_dims(np.zeros(n, dtype=np.float32), 0), engine=e)", True),
            ("0.35", False),
            ("float(value) * scale", False),
            ("np.full(num_dof, 0.35)", False),  # value-producing, not value-preserving
            ("cmd", False),  # an unresolvable name is not evidence of zero
        ):
            p = tmp_path / "snippet.py"
            p.write_text(shape.format(v=value) + "\n")
            writes = _velocity_target_writes(p)
            assert len(writes) == 1, f"{shape} with {value}: not detected as a write"
            (_, val, is_4ws) = writes[0]
            assert not is_4ws
            assert _is_zero(val) is want_zero, f"{shape} with {value}: wrong verdict"


def test_the_4ws_loop_detector_requires_the_named_dict(tmp_path: Path):
    """The carve-out must not be a blanket pass for any enclosing loop."""
    p = tmp_path / "snippet.py"
    p.write_text(
        "for name, value in cmd_4ws_drive_vel.items():\n"
        "    drive.GetTargetVelocityAttr().Set(float(value) * scale)\n"
        "for name, value in cmd_positions.items():\n"
        "    drive.GetTargetVelocityAttr().Set(float(value) * scale)\n"
    )
    writes = _velocity_target_writes(p)
    assert [is_4ws for _, _, is_4ws in writes] == [True, False]
