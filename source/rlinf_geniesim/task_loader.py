# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Build physics/render manifests from benchmark task and instance data.

task_name selects benchmark/config/eval_tasks/<name>.json for the background
and robot; sub_task_name selects llm_task/<name>/<instance>/scene.usda for
the foreground. Manifest construction uses OpenUSD to correct spawn placement
and bake composed rigid-body poses before superset deduplication. Optional
render-layer authoring uses the same USD runtime."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Optional

from rlinf_geniesim.randomize import RandomizationSpec, Sampler, check_base_xy_channel_conflict, task_defaults
from rlinf_geniesim.randomize.background import discover_background_rooms, resolve_room_usda
from rlinf_geniesim.task_background import TASK_NEEDS_BACKGROUND, resolve_base_z, room_key
from rlinf_geniesim.task_lighting import LIGHT_RIGS, LightRig, resolve_light_rig


def _quat_to_yaw(quat) -> float:
    """Yaw (radians about world Z) from a (w, x, y, z) quaternion — for the
    engine's init_base_pose.theta (benchmark stores a full quaternion; the
    engine takes a single yaw, which matches for upright robot bases)."""
    import math  # noqa: PLC0415

    w, x, y, z = (float(v) for v in (list(quat) + [0, 0, 0, 0])[:4])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _benchmark_config_root() -> Path:
    """Locate benchmark/config, which contains eval_tasks and llm_task data."""
    import geniesim_benchmark  # noqa: PLC0415

    return Path(geniesim_benchmark.__file__).resolve().parent / "benchmark" / "config"


def _assets_root() -> Path:
    # Prefer the container convention (entrypoint sets GENIESIM_ASSETS_PATH), so
    # the assets root resolves even when the geniesim_assets package isn't
    # pip-installed. Fall back to the installed package (host default).
    env = os.environ.get("GENIESIM_ASSETS_PATH")
    if env:
        return Path(env)
    from geniesim_assets import ASSETS_PATH  # noqa: PLC0415

    return Path(ASSETS_PATH)


# ---------------------------------------------------------------------------
# Physics overrides: environment > config/physics_overrides.yaml > code default
# ---------------------------------------------------------------------------

_PHYSICS_OVERRIDES_YAML = "physics_overrides.yaml"

# Upper bound on solref timeconst: beyond this, finger penetration outweighs
# the benefit (about 1 cm at 0.02).
_SOLREF_TC_MAX = 0.05


def _physics_overrides(path: Optional[str] = None) -> dict:
    """Read optional physics overrides, returning {} when absent or unreadable.

    Report parse/read failures before falling back to code defaults.
    """
    p = Path(path) if path else Path(__file__).resolve().parent / "config" / _PHYSICS_OVERRIDES_YAML
    if not p.is_file():
        return {}
    try:
        import yaml  # noqa: PLC0415

        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001 — a bad yaml must not stop the build
        print(
            f"[task_loader] WARN: cannot read {p} ({exc}); NO overrides are "
            f"in effect and the code defaults are used",
            flush=True,
        )
        return {}


def _parse_solref(raw, *, source: str, substep: int = 10) -> Optional[list]:
    """Parse [timeconst, dampratio] from a YAML list or comma-separated string.

    None or an empty value means no override. Reject malformed values;
    substep determines the warning threshold of 2 * sub_dt.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return None
        # Do not filter empty fields with ``if x.strip()``: that reads a
        # trailing comma such as ``"0.002,"`` as a single value, defaulting
        # dampratio to 1.0 and changing the stiffness by 25x with no error.
        parts = [x.strip() for x in raw.split(",")]
        if any(not x for x in parts):
            raise ValueError(
                f"{source}={raw!r} has an empty field (trailing " f"comma?); expected <timeconst>[,<dampratio>]"
            )
    elif isinstance(raw, (list, tuple)):
        parts = list(raw)
        if not parts:
            return None
    else:
        raise ValueError(f"{source}={raw!r} has the wrong type; expected " f"'<tc>[,<dr>]' or [tc, dr]")

    try:
        vals = [float(x) for x in parts]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}={raw!r} is not numeric ({exc})") from exc
    if not 1 <= len(vals) <= 2 or any(v <= 0 for v in vals):
        raise ValueError(f"{source}={raw!r} expects <timeconst>[,<dampratio>], " f"both greater than 0")

    tc = vals[0]
    dr = vals[1] if len(vals) > 1 else 1.0
    if tc > _SOLREF_TC_MAX:
        raise ValueError(
            f"{source} timeconst={tc} > {_SOLREF_TC_MAX}: " f"finger penetration outweighs the benefit above this"
        )
    # Lower bound 2*sub_dt: below it the contact integrator is unstable and
    # Baumgarte silently softens contacts every other substep, which shows up
    # as the gripper passing through objects and no other parameter can fix
    # it. Warn rather than reject, so "raise substep first, then lower
    # timeconst" stays available.
    floor = 2.0 / (100.0 * max(int(substep), 1))
    if tc < floor:
        print(
            f"[task_loader] WARN: {source} timeconst={tc} < {floor:.5g} "
            f"= 2*sub_dt at substep={substep}; contacts will soften "
            f"silently (the gripper passes through objects). Raise substep "
            f"first.",
            flush=True,
        )
    return [tc, dr]


def _solref_note(solref: list, factory: list, label: str) -> str:
    """One line describing how stiff this setting actually is.

    solref is (timeconst, dampratio) while the engine writes
    ``ke = (1/(tc*dr))^2``, so two similar-looking solrefs can differ by 25x
    in stiffness. Reporting solref alone does not add up later.
    """
    ke = (1.0 / (solref[0] * solref[1])) ** 2
    fke = (1.0 / (factory[0] * factory[1])) ** 2
    return (
        f"{label} solref={solref} (ke~{ke:.0f}; default {factory} gives "
        f"ke={fke:.0f}, so stiffness / {fke / ke:.3g})"
    )


def _resolve_solref(
    *, env_var: str, yaml_key: str, factory: list, label: str, overrides: Optional[dict] = None, substep: int = 10
) -> list:
    """Resolve this side's solref by environment > yaml > default, and print
    what took effect.

    The environment must win: the tuning tools inject it per run to do A/B
    comparisons, and a yaml that could override it would silently make those
    reports describe a different setting than the one they print.
    """
    ov = _physics_overrides() if overrides is None else overrides
    contact = (ov.get("contact") or {}) if isinstance(ov, dict) else {}

    got = _parse_solref(os.environ.get(env_var), source=env_var, substep=substep)
    if got is not None:
        print(f"[task_loader] {env_var} -> {_solref_note(got, factory, label)}", flush=True)
        return got

    got = _parse_solref(contact.get(yaml_key), source=f"{_PHYSICS_OVERRIDES_YAML}:contact.{yaml_key}", substep=substep)
    if got is not None:
        print(f"[task_loader] {_PHYSICS_OVERRIDES_YAML} -> " f"{_solref_note(got, factory, label)}", flush=True)
        return got

    print(f"[task_loader] {label} solref={factory} (default, no override)", flush=True)
    return list(factory)


def resolve_benchmark_task(
    task_name: str,
    sub_task_name: str,
    instance_id: int = 0,
    *,
    bench_root: Optional[str] = None,
    assets_root: Optional[str] = None,
) -> dict:
    """Resolve a benchmark task to its concrete asset paths (no USD loaded).

    ``bench_root`` / ``assets_root`` are injectable for testing; default to the
    installed ``geniesim_benchmark`` / ``geniesim_assets`` packages.
    """
    bench = Path(bench_root) if bench_root else _benchmark_config_root()
    assets = Path(assets_root) if assets_root else _assets_root()

    eval_task = json.loads((bench / "eval_tasks" / f"{task_name}.json").read_text())
    robot_cfg_name = eval_task["robot"]["robot_cfg"]  # e.g. "G2_omnipicker.json"

    # app/robot_cfg/ is a sibling of benchmark/ (bench = .../benchmark/config)
    app_root = bench.parent.parent
    robot_cfg = json.loads((app_root / "app" / "robot_cfg" / robot_cfg_name).read_text())
    # robot_usd / base_prim_path live under the "robot" block in the real
    # robot_cfg; tolerate a flat layout too.
    rc_robot = robot_cfg.get("robot", robot_cfg)
    robot_usd_rel = rc_robot["robot_usd"]  # relative to assets root
    robot_prefix = str(rc_robot.get("base_prim_path", "/genie")).lstrip("/")

    # Foreground scene for this instance (RL always loads this).
    scene_usda = bench / "llm_task" / sub_task_name / str(instance_id) / "scene.usda"

    # BACKGROUND room (shelf/counter/door/...) — eval_task.scene.scene_usd, under
    # the assets root (e.g. background/market/market_1/background.usda). RL loads
    # it only when ``use_background`` says so — resolved per task by
    # ``task_background.resolve_use_background``; the objects + robot are
    # authored in THIS scene's world frame, so with the background loaded the real
    # workbench supports them. May be a list (benchmark picks one) -> take first.
    scene_blk = eval_task.get("scene", {}) or {}
    bg_rel = scene_blk.get("scene_usd", "")
    if isinstance(bg_rel, list):
        bg_rel = bg_rel[0] if bg_rel else ""
    background_usda = str(assets / bg_rel) if bg_rel else ""

    # Static background props whose world AABB the eval checkers need. Scene
    # props have no rigid body, so the engine must publish their boxes
    # explicitly or every ``Inside``/``InBBox`` checker naming one scores 0%
    # (see ``static_props.py`` / the isaac_fused static-prop path engine-side).
    # Scene-scoped, hence it lives in the eval task's scene block next to the
    # background it belongs to, and is shared by every sub-task and instance
    # in that room. Paths are in the dialect the checkers emit (``/World/<name>``).
    static_prop_aabbs = [str(x) for x in (scene_blk.get("eval_static_props") or []) if str(x).strip()]

    return {
        "scene_usda": str(scene_usda),
        "background_usda": background_usda,
        "static_prop_aabbs": static_prop_aabbs,
        "robot_usda": str(assets / robot_usd_rel),
        "robot_fixed_attachments": rc_robot.get("fixed_attachments", []),
        "robot_prefix": robot_prefix,
        "robot_from_urdf": False,
        "robot_cfg_name": robot_cfg_name,
        "task_name": task_name,
        "sub_task_name": sub_task_name,
        "instance_id": int(instance_id),
    }


def resolve_task_cameras(
    task_name: str,
    *,
    bench_root: Optional[str] = None,
    obs_width: Optional[int] = None,
) -> list[dict]:
    """Resolve observation cameras from the benchmark task and robot configuration.

    Use recording_setting.camera_list for camera selection and the robot camera
    specifications for native dimensions. obs_width selects a common output
    width; None selects the first camera's native width. Preserve each camera's
    aspect ratio, round output dimensions to even values, and retain native_hw
    for calibrated intrinsics. prim_path is relative to the robot root.
    """
    if obs_width is not None and int(obs_width) <= 0:
        raise ValueError(f"obs_width must be positive, got {obs_width}")
    bench = Path(bench_root) if bench_root else _benchmark_config_root()
    eval_task = json.loads((bench / "eval_tasks" / f"{task_name}.json").read_text())
    cam_list = (eval_task.get("recording_setting") or {}).get("camera_list") or []
    robot_cfg_name = eval_task["robot"]["robot_cfg"]
    robot_cfg = json.loads((bench.parent.parent / "app" / "robot_cfg" / robot_cfg_name).read_text())
    cam_specs = robot_cfg.get("camera") or {}
    prefix = str((robot_cfg.get("robot", robot_cfg)).get("base_prim_path", "/genie"))

    # Share a width while preserving each camera's native aspect ratio.
    common_w: Optional[int] = None if obs_width is None else int(obs_width)
    cams: list[dict] = []
    for full in cam_list:
        if full.startswith(prefix + "/"):  # /genie/head_link3/... -> head_link3/...
            rel = full[len(prefix) + 1 :]
        else:
            rel = full.lstrip("/").split("/", 1)[-1]  # drop the leading robot-root segment
        spec = cam_specs.get(full) or cam_specs.get(rel) or []
        w, h = (int(spec[0]), int(spec[1])) if len(spec) >= 2 else (640, 400)
        if common_w is None:  # common width = first (head) cam's
            common_w = w
        obs_w = int(common_w)
        obs_h = int(round(obs_w * h / w))  # preserve THIS cam's aspect
        # Use even dimensions for the tiled renderer and image codecs.
        obs_w -= obs_w % 2
        obs_h = max(2, obs_h - obs_h % 2)
        cams.append(
            {
                "name": rel.rsplit("/", 1)[-1],
                "link": rel.split("/", 1)[0],
                "prim_path": rel,
                "image_hw": [obs_h, obs_w],  # per-camera obs resolution
                # This camera's NATIVE benchmark resolution — the obs resolution above
                # is a clean aspect-preserving downscale of it (width-matched).
                "native_hw": [h, w],
            }
        )
    return cams


def resolve_robot_init_joints(sub_task_name: str, robot_cfg_name: str) -> dict:
    """Resolve the INITIAL JOINT pose benchmark eval applies, by REUSING the
    benchmark config registries — exact parity, maximal code reuse.

    benchmark's PiEnv.reset() snaps the robot to:
      arm/waist/head <- TaskInfo(TASK_INFO_DICT[sub_task][robot]).init_pose()
      gripper        <- ROBOT_CONFIGS[robot]["init_gripper_open"]
    mapping the values onto ROBOT_CONFIGS[robot]'s joint-name lists.

    Returns ``{joint_name: value}`` (the engine's init_joint_pos), or ``{}`` if
    geniesim_benchmark isn't importable or the task/robot has no entry.

    NOTE: this is NOT the teleop ``init_arm_pose`` (robot_init_pose.json) — that
    is teleop-only and has no G2 entry; benchmark EVAL uses TaskInfo.init_pose().
    """
    try:  # benchmark config registries are pure-Python (no Isaac)
        from geniesim_benchmark.utils.name_utils import ROBOT_CONFIGS, robot_type_mapping  # noqa: PLC0415
        from geniesim_benchmark.benchmark.config.robot_init_states import TASK_INFO_DICT  # noqa: PLC0415
        from geniesim_benchmark.utils.infer_pre_process import TaskInfo  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return {}

    robot = robot_type_mapping(str(robot_cfg_name).split(".")[0])
    cfg = ROBOT_CONFIGS.get(robot)
    ti_cfg = (TASK_INFO_DICT.get(sub_task_name) or {}).get(robot)
    if cfg is None or ti_cfg is None:
        return {}

    init_arm, init_head, init_waist, _init_hand, _init_gripper = TaskInfo(ti_cfg, robot).init_pose()
    init_gripper = cfg.get("init_gripper_open", [0.0, 0.0])  # PiEnv uses cfg, not TaskInfo
    # TaskInfo returns radians (Isaac convention); the engine's init_joint_pos
    # wants DEGREES for revolute joints (arm/waist/head) and metres for prismatic
    # (gripper). ALL G2 groups incl. the gripper masters are revolute -> degrees.
    _deg = 180.0 / math.pi
    out: dict = {}
    for names, vals, conv in (
        (cfg.get("arm_joints"), init_arm, _deg),
        (cfg.get("waist_joints"), init_waist, _deg),
        (cfg.get("head_joints"), init_head, _deg),
        # G2 omnipicker gripper MASTER joints (outer_joint1) are REVOLUTE, so
        # they follow the same degrees convention as arm/waist/head. Passing the
        # benchmark's 0.785 rad through raw made the engine read 0.785 DEGREES
        # (~0.014 rad) -> gripper started CLOSED instead of open. (A prismatic
        # gripper robot would need conv=1.0 metres here.)
        (cfg.get("gripper_joints"), init_gripper, _deg),
    ):
        if names and vals is not None:
            for name, val in zip(names, vals):
                out[name] = float(val) * conv
    return out


def _rlinf_cameras_to_assemble_cfg(cameras: list[dict]) -> list[dict]:
    """Convert task-init cameras (``{name, link, image_hw, ...}``) to the camera
    config ``author_render_layer`` expects (``{prim_path, sensor{width,height},
    topic{rgb,depth}, intrinsic, extrinsic, model}``).

    ``prim_path`` is the camera prim relative to the robot root: prefer an
    explicit ``prim_path`` (mirror mode — robot.usda has the camera baked in);
    fall back to ``link`` (synthesize mode — author_render_layer derives
    intrinsics from ``intrinsic``). Pure dict mapping (host-testable)."""
    out: list[dict] = []
    for i, c in enumerate(cameras or []):
        h, w = c.get("image_hw") or [800, 1280]
        name = c.get("name", f"cam_{i}")
        out.append(
            {
                "prim_path": c.get("prim_path") or c.get("link") or name,
                "sensor": {
                    "width": int(w),
                    "height": int(h),
                    "min_range": float(c.get("min_range", 0.01)),
                    "max_range": float(c.get("max_range", 10000.0)),
                },
                "topic": {
                    "rgb": c.get("topic", f"/rlinf/{name}/image_raw"),
                    "depth": c.get("depth_topic", ""),
                },
                "intrinsic": c.get("intrinsic", {}),
                "extrinsic": c.get("extrinsic", {}),
                "model": c.get("model", ""),
            }
        )
    return out


def _import_author_render_layer():
    """Import the upstream ``author_render_layer`` (extracted from assemble_scene).

    Tries the installed ``genie_sim_engine`` package, then locates
    ``assemble_scene.py`` in the repo and loads it by path. The module imports
    cleanly without pxr (pxr is lazy), so this resolves on host; only *calling*
    the returned function needs pxr + a real robot USD (container)."""
    try:
        from genie_sim_engine_isaacsim.assemble._scene_impl import (  # type: ignore  # noqa: PLC0415
            author_render_layer,
        )

        return author_render_layer
    except Exception:
        pass
    import importlib.util  # noqa: PLC0415

    rel = (
        "geniesim_ros",
        "src",
        "ros_ws",
        "src",
        "genie_sim_engine_isaacsim",
        "genie_sim_engine_isaacsim",
        "assemble",
        "_scene_impl.py",
    )
    for parent in Path(__file__).resolve().parents:
        cand = parent.joinpath(*rel)
        if cand.is_file():
            spec = importlib.util.spec_from_file_location("genie_assemble_scene", str(cand))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            return mod.author_render_layer
    raise ImportError("could not locate assemble_scene.author_render_layer")


def _deactivate_baked_omnigraphs(st) -> None:
    """Deactivate embedded OmniGraphs in the authored wrapper stage.

    Graphs must be disabled before opening the stage because registration and
    relocation can crash Isaac. The override preserves meshes, colliders and
    joints, which the benchmark controls through engine APIs.

    This wrapper override does not modify foreground_usda or its per-environment
    copies; sanitizing those requires editing their shared input layer.
    """
    from pxr import Usd  # noqa: PLC0415

    # Traverse everything, including prims already switched off upstream, so a
    # graph inside an inactive subtree cannot be missed if it is reactivated.
    victims = [
        p.GetPath()
        for p in Usd.PrimRange.Stage(st, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate))
        if p.GetTypeName() in ("OmniGraph", "ComputeGraph")
    ]
    if not victims:
        return
    for path in victims:
        st.OverridePrim(path).SetActive(False)
    shown = ", ".join(str(p) for p in victims[:3])
    print(
        f"[task_loader] deactivated {len(victims)} baked OmniGraph(s) that "
        f"ship inside object assets (they evaluate in the simulation "
        f"pipeline and crash PhysX startup): {shown}"
        f"{' …' if len(victims) > 3 else ''}",
        flush=True,
    )


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() not in ("0", "false", "no", "off")


def _emit_env_shell() -> bool:
    """Is the render node handed a foreground-free stage? (``RLINF_ENV_SHELL``)

    On by default. Set ``RLINF_ENV_SHELL=0`` to go back to the render node
    opening ``scene_usda`` and deactivating the strays.
    """
    return _env_flag("RLINF_ENV_SHELL", True)


def _baked_graphs_run() -> bool:
    """Whether wrapper-stage OmniGraphs remain active (RLINF_BAKED_GRAPHS).

    Enabled by default. This flag does not affect per-environment copies loaded
    from foreground_usda.
    """
    return _env_flag("RLINF_BAKED_GRAPHS", True)


def _omnigraph_guard_on() -> bool:
    """Deactivate wrapper graphs unless both env-shell rendering and graphs are enabled.

    Opening the foreground-containing wrapper and relocating live graphs can
    crash Isaac. Disabling env-shell rendering therefore enables this guard.
    The renderer and manifest builder must agree on env_shell_usda support.
    """
    return not (_emit_env_shell() and _baked_graphs_run())


def _author_rlinf_env(st, *, light_rig: Optional[LightRig], floor_half: float) -> None:
    """Author shared lights and a neutral floor under /RLinfEnv.

    Both the wrapper and render shell use this function. The global location
    keeps these lights outside per-environment cloning and randomization.
    """
    from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade  # noqa: PLC0415

    def _author_light(prim, *, intensity, exposure, kelvin):
        # Write both USD attribute spellings for renderer compatibility.
        vals = [
            ("intensity", float(intensity), Sdf.ValueTypeNames.Float),
            ("exposure", float(exposure), Sdf.ValueTypeNames.Float),
        ]
        if kelvin is not None:
            vals += [
                ("colorTemperature", float(kelvin), Sdf.ValueTypeNames.Float),
                ("enableColorTemperature", True, Sdf.ValueTypeNames.Bool),
            ]
        for name, value, ty in vals:
            prim.CreateAttribute(f"inputs:{name}", ty).Set(value)
            prim.CreateAttribute(name, ty).Set(value)

    rig = light_rig if light_rig is not None else LIGHT_RIGS["legacy"]

    dome = UsdLux.DomeLight.Define(st, "/RLinfEnv/Dome")
    if rig.dome_rotate is not None:
        UsdGeom.Xformable(dome.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(*rig.dome_rotate))
    _author_light(dome.GetPrim(), intensity=rig.dome_intensity, exposure=rig.dome_exposure, kelvin=rig.dome_kelvin)
    if not rig.dome_visible_in_primary_ray:
        # Omit the override when it matches the renderer's visible default.
        dome.GetPrim().CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool).Set(False)

    sun = UsdLux.DistantLight.Define(st, "/RLinfEnv/Sun")
    UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(*rig.sun_rotate))
    _author_light(sun.GetPrim(), intensity=rig.sun_intensity, exposure=rig.sun_exposure, kelvin=rig.sun_kelvin)

    # --- default floor: large neutral plane at z=0, render-only (no collision) ---
    h = float(floor_half)
    floor = UsdGeom.Mesh.Define(st, "/RLinfEnv/Floor")
    floor.CreatePointsAttr([(-h, -h, 0.0), (h, -h, 0.0), (h, h, 0.0), (-h, h, 0.0)])
    floor.CreateFaceVertexCountsAttr([4])
    floor.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    floor.CreateNormalsAttr([(0, 0, 1)] * 4)
    floor.SetNormalsInterpolation("vertex")
    floor.CreateExtentAttr([(-h, -h, 0.0), (h, h, 0.0)])
    mat = UsdShade.Material.Define(st, "/RLinfEnv/FloorMat")
    shd = UsdShade.Shader.Define(st, "/RLinfEnv/FloorMat/PBR")
    shd.CreateIdAttr("UsdPreviewSurface")
    shd.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.5, 0.5, 0.5))
    shd.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    mat.CreateSurfaceOutput().ConnectToSource(shd.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(floor.GetPrim()).Bind(mat)


def write_env_shell_usda(
    out_path: str,
    background_usda: str = "",
    light_rig: Optional[LightRig] = None,
    floor_half: float = 15.0,
) -> str:
    """Write the global environment shell without foreground objects.

    The render node instantiates foregrounds under each env_i; loading them in
    the shell would relocate asset OmniGraphs and can crash Isaac Sim. A real
    background room supplies its own lights/floor, so the shell stays empty in
    that case. Preserve the foreground's resolved USD distance units, including
    its default 0.01 metersPerUnit, for unscaled robot references."""
    from pxr import Usd, UsdGeom  # noqa: PLC0415

    st = Usd.Stage.CreateNew(out_path)
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    # ``/World`` must exist and be the defaultPrim: the render node authors
    # ``/World/envs`` onto this stage, and consumers resolve the default prim.
    world = st.DefinePrim("/World", "Xform")
    st.SetDefaultPrim(world)

    bg_abs = str(Path(background_usda).resolve()) if background_usda else ""
    if not (bg_abs and Path(bg_abs).is_file()):
        _author_rlinf_env(st, light_rig=light_rig, floor_half=floor_half)

    st.GetRootLayer().Save()
    return out_path


def _wrap_scene_with_default_env(
    scene_usda: str,
    out_path: str,
    background_usda: str = "",
    light_rig: Optional[LightRig] = None,
    floor_half: float = 15.0,
) -> str:
    """Create a wrapper USD that sublayers the foreground scene and gives the
    renderer something to look at, returning the wrapper path.

    Two cases, and they are exclusive:

    * ``background_usda`` given — the task's real room. It carries its own
      floor, walls, shelves and lights, so it is sublayered as-is and NO
      default floor or lights are added. Adding them anyway would put a second
      floor through the room's own and over-expose everything.
    * no background — the foreground alone. RL foreground scenes carry no room
      and no lights, so the render is otherwise dark with nothing for the robot
      to stand on. Add a default backdrop:
        * a large neutral floor plane at z=0 (render-only — no collision, so
          the physics model is unchanged),
        * a DomeLight fill + an angled DistantLight key, per ``light_rig``.

    WHICH lights is a per-task decision, not this function's: pick the rig with
    ``task_lighting.resolve_light_rig(sub_task_name)`` and pass it in. Omitting
    it takes the legacy pair. That file carries the two rigs and why a task
    would be on either one.

    The PHYSICS side already loads the background on its own (``engine.py``
    raw-references it so the importer picks up its collision geometry, which
    is what holds objects up on a shelf), so rendering must see the same
    ``background_usda`` too — otherwise an object can rest correctly on a
    shelf while the picture shows it floating over an empty plane, the shelf
    nowhere to be seen.

    A sublayer composes the scene at its original prim paths (geometry
    unchanged — physics still sees the same model). CONTAINER-ONLY (needs
    pxr)."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade  # noqa: PLC0415

    scene_abs = str(Path(scene_usda).resolve())
    st = Usd.Stage.CreateNew(out_path)
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    root = st.GetRootLayer()
    root.subLayerPaths.append(scene_abs)
    bg_abs = str(Path(background_usda).resolve()) if background_usda else ""
    if bg_abs and Path(bg_abs).is_file():
        # A separate /Background reference avoids foreground path collisions.
        # This wrapper owns room composition; the engine must not add it again.
        st.DefinePrim("/Background", "Xform").GetReferences().AddReference(bg_abs)
    # Preserve the scene's defaultPrim so consumers reading it still resolve.
    scene_layer = Sdf.Layer.FindOrOpen(scene_abs)
    if scene_layer and scene_layer.defaultPrim:
        root.defaultPrim = scene_layer.defaultPrim

    if bg_abs and Path(bg_abs).is_file():
        # The room brings its own floor and lights; stop here.
        if _omnigraph_guard_on():
            _deactivate_baked_omnigraphs(st)
        st.GetRootLayer().Save()
        return out_path

    # The global env (dome + key + floor). Shared verbatim with
    # ``write_env_shell_usda`` -- see that function for why the two must not
    # drift.
    _author_rlinf_env(st, light_rig=light_rig, floor_half=floor_half)

    if _omnigraph_guard_on():
        _deactivate_baked_omnigraphs(st)
    st.GetRootLayer().Save()
    return out_path


def author_task_render_layer(
    out_dir: str,
    robot_usda: str,
    robot_prefix: str,
    cameras: list[dict],
    free_cam_hw: tuple = (720, 1280),
) -> str:
    """Author ``render_layer.usda`` for a task by reusing the upstream
    ``author_render_layer``. Returns the render-layer path.

    CONTAINER-ONLY: opens the robot USD and authors USD (needs pxr). The camera
    conversion and the import resolve on host, but the call does not.
    """
    author_render_layer = _import_author_render_layer()
    from pxr import Usd  # noqa: PLC0415

    robot_stage = Usd.Stage.Open(robot_usda, load=Usd.Stage.LoadAll)
    cam_cfg = _rlinf_cameras_to_assemble_cfg(cameras)
    fh, fw = free_cam_hw
    render_layer_path, _authored, _lidars = author_render_layer(
        out_dir, robot_usda, robot_stage, robot_prefix, cam_cfg, [], int(fw), int(fh)
    )

    # Foreground-only RL scenes skip the background USD — which is where the
    # scene lights live — so OVRTX would render pure black. Add an ambient
    # DomeLight to the render layer so the scene is actually visible.
    try:
        from pxr import UsdLux  # noqa: PLC0415

        st = Usd.Stage.Open(render_layer_path)
        dome = UsdLux.DomeLight.Define(st, "/RenderOVRTX/RLinfDomeLight")
        dome.CreateIntensityAttr(1000.0)
        st.GetRootLayer().Save()
        print("[rlinf-render] added ambient DomeLight (foreground-only scene)", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[rlinf-render] WARN: could not add DomeLight: {exc!r}", flush=True)
    return render_layer_path


def resolve_instance_robot_init(
    sub_task_name: str,
    instance_ids: list,
    build_instance: int,
    *,
    bench_root: Optional[str] = None,
) -> dict:
    """Per-instance robot initial state -- only present for tasks that ship a
    ``robot_init.json``.

    Why this is needed: some tasks must reproduce training data where the robot's starting
    pose was **randomized** per episode. The benchmark's existing mechanism
    (``eval_task.robot_init_pose`` + ``TASK_INFO_DICT``) stores **one value per task**, which
    cannot express that.

    Returns::

        {"robot_base_xy": {iid: [dx, dy]},        # **delta** relative to the build instance
         "robot_joints":  {iid: {joint: rad}}}    # absolute value (radians)

    Why the base is a delta rather than an absolute value: the engine side goes through the
    mocap channel (``apply_base_mocap_offsets``), which adds an offset to the
    ``_base_mocap_init_pos`` snapshot taken at build time. An absolute value has no meaning
    there.

    **Only xy is handled.** This channel does not write z or orientation (in
    ``mujoco_warp.py``, z is forced to init and orientation is never written at all), so a
    task using this mechanism must keep orientation and z identical across its instances. If
    a future task also varies orientation, this function must be extended first -- an
    orientation mismatch here **raises loudly** rather than silently degrading.

    Tasks with no ``robot_init.json`` return ``{}``, and callers skip accordingly.
    """
    bench = Path(bench_root) if bench_root else _benchmark_config_root()
    root = bench / "llm_task" / sub_task_name
    ids = [int(i) for i in (instance_ids or [])]
    if int(build_instance) not in ids:
        ids.append(int(build_instance))
    files = {i: root / str(i) / "robot_init.json" for i in ids}
    if not any(f.is_file() for f in files.values()):
        return {}

    recs: dict = {}
    for i, f in files.items():
        if not f.is_file():
            raise RuntimeError(
                f"{sub_task_name} instance {i} is missing robot_init.json, but other "
                f"instances of the same task have it -- per-instance robot poses are only "
                f"half done, and the eval would silently run with the wrong pose"
            )
        recs[i] = json.loads(f.read_text())

    base = recs[int(build_instance)]["base_pose"]
    ref_q = [float(v) for v in base["quaternion"]]
    ref_z = float(base["position"][2])
    # Tolerance is set by **physical scale**, not by floating-point bits: the data is a 4x4
    # matrix dumped out of the simulation, and z carries nanometre-scale noise, so comparing
    # entry by entry would treat that noise as a real difference. 0.1mm / 0.01 degrees is far
    # smaller than any meaningful difference and far larger than the noise.
    Z_TOL, Q_TOL = 1e-4, 1e-4
    out_xy, out_j = {}, {}
    for i, rec in recs.items():
        bp = rec["base_pose"]
        if max(abs(float(a) - float(b)) for a, b in zip(bp["quaternion"], ref_q)) > Q_TOL:
            raise RuntimeError(
                f"{sub_task_name} instance {i}'s base orientation differs from the "
                f"reference instance -- the engine's mocap channel only writes xy, so "
                f"orientation would be silently dropped. Supporting it requires extending "
                f"apply_base_mocap_offsets first"
            )
        if abs(float(bp["position"][2]) - ref_z) > Z_TOL:
            raise RuntimeError(
                f"{sub_task_name} instance {i}'s base z differs from the reference "
                f"instance by {abs(float(bp['position'][2]) - ref_z)*1000:.3f}mm -- same as "
                f"above, the mocap channel's z is forced to init and cannot be changed"
            )
        out_xy[str(i)] = [
            float(bp["position"][0]) - float(base["position"][0]),
            float(bp["position"][1]) - float(base["position"][1]),
        ]
        out_j[str(i)] = {k: float(v) for k, v in rec["joint_pos_rad"].items()}
    return {"robot_base_xy": out_xy, "robot_joints": out_j}


def resolve_robot_init_pose(
    task_name: str,
    sub_task_name: str,
    instance_id: int = 0,
    *,
    bench_root: Optional[str] = None,
) -> dict:
    """Resolve the benchmark robot base pose as position and wxyz quaternion.

    Select the workspace from ``scene_info.scene_id``. If it is absent from
    the configured pose table, use the first workspace. A missing pose table
    returns the identity pose; a flat position/quaternion mapping is accepted.
    """
    bench = Path(bench_root) if bench_root else _benchmark_config_root()
    eval_task = json.loads((bench / "eval_tasks" / f"{task_name}.json").read_text())
    rip = (eval_task.get("robot", {}) or {}).get("robot_init_pose", {}) or {}

    identity = {"position": [0.0, 0.0, 0.0], "quaternion": [1.0, 0.0, 0.0, 0.0]}
    if not isinstance(rip, dict) or not rip:
        return identity
    # Already flattened to {position, quaternion}?
    if "position" in rip:
        return {
            "position": list(rip.get("position", identity["position"])),
            "quaternion": list(rip.get("quaternion", identity["quaternion"])),
        }

    # Keyed by workspace. Pick workspace_id like benchmark (scene_info.scene_id),
    # falling back to the single workspace present.
    ws_id = None
    inst = bench / "llm_task" / sub_task_name / str(instance_id) / "scene_info.json"
    if inst.is_file():
        try:
            sid = json.loads(inst.read_text()).get("scene_id", "")
            ws_id = str(sid).split("/")[-1] if sid else None
        except Exception:  # noqa: BLE001
            ws_id = None
    if ws_id not in rip:
        ws_id = next(iter(rip)) if rip else None  # single-workspace fallback
    chosen = rip.get(ws_id, {}) if ws_id else {}
    return {
        "position": list(chosen.get("position", identity["position"])),
        "quaternion": list(chosen.get("quaternion", identity["quaternion"])),
    }


def _spawn_lift(lo, hi, supports, max_pen: float, clearance: float):
    """How far to raise a spawn box so it clears the support under it.

    Pure arithmetic on axis-aligned boxes -- no USD -- so the RULE has one
    implementation and both callers (the built scene and the per-reset pose
    table) are guaranteed to agree. ``lo``/``hi`` are the object's world AABB;
    ``supports`` is a list of ``(lo, hi)`` static boxes.

    Returns ``(dz, top)``, or ``(None, None)`` when the object rests above
    everything, is free-falling, or is deeper than ``max_pen`` -- that last one
    is not a spawn penetration but a bbox artifact of a hollow container (an
    object INSIDE a bin overlaps the bin's rim-height bbox).
    """
    bottom = float(lo[2])
    area = max((float(hi[0]) - float(lo[0])) * (float(hi[1]) - float(lo[1])), 1e-9)
    top = None
    for slo, shi in supports:
        st = float(shi[2])
        if not (bottom - 0.05 < st <= bottom + max_pen):
            continue
        ox = min(float(hi[0]), float(shi[0])) - max(float(lo[0]), float(slo[0]))
        oy = min(float(hi[1]), float(shi[1])) - max(float(lo[1]), float(slo[1]))
        if ox <= 0 or oy <= 0 or (ox * oy) / area < 0.3:
            continue
        top = st if top is None else max(top, st)
    if top is None or top <= bottom:
        return None, None
    return (top - bottom) + clearance, top


def _static_support_boxes(bbox_cache, static_prims, background_path):
    """World AABBs of everything an object can rest on: the non-dynamic
    foreground prims plus every mesh in the background room."""
    from pxr import Usd, UsdGeom  # noqa: PLC0415

    out = []
    for o in static_prims:
        r = bbox_cache.ComputeWorldBound(o).ComputeAlignedRange()
        if not r.IsEmpty():
            out.append((r.GetMin(), r.GetMax()))
    if background_path:
        bg = Usd.Stage.Open(background_path)
        if bg is not None:
            bg.Load("/")
            bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
            for p in bg.Traverse():
                if p.GetTypeName() != "Mesh":
                    continue
                r = bc.ComputeWorldBound(p).ComputeAlignedRange()
                if not r.IsEmpty():
                    out.append((r.GetMin(), r.GetMax()))
    return out


def _split_dynamic(objs_root):
    """``(dynamic, static)`` children of ``/World/Objects``."""
    from pxr import Usd  # noqa: PLC0415

    def _is_dynamic(prim) -> bool:
        for d in Usd.PrimRange(prim):
            a = d.GetAttribute("physics:rigidBodyEnabled")
            if a and a.IsValid() and bool(a.Get()):
                return True
        return False

    dyn, static = [], []
    for o in objs_root.GetAllChildren():
        (dyn if _is_dynamic(o) else static).append(o)
    return dyn, static


def depenetrate_spawn(
    scene_path: str,
    background_path: str,
    out_path: str,
    *,
    max_pen: float = 0.03,
    clearance: float = 0.002,
) -> str:
    """Lift dynamic foreground objects out of SHALLOW spawn penetration.

    Benchmark scenes author object heights against the VISUAL support surface;
    the physics collider can sit a few mm proud (e.g. market shelf: authored can
    bottom z=0.822 vs collider top z=0.8258 -> 3.8mm penetration). PhysX clamps
    de-penetration velocity so eval tolerates this; mjwarp's stiff contact
    (ke=250000) LAUNCHES the object instead — upright cans get ejected and tip
    over at t=0.

    For every dynamic object (any descendant with physics:rigidBodyEnabled=1),
    find the highest STATIC support top under its footprint — background meshes +
    non-dynamic foreground objects — and, if the object's collision-bbox bottom
    penetrates it by <= ``max_pen`` (deeper "penetrations" are bbox artifacts of
    hollow containers — an object INSIDE a bin overlaps the bin's rim-height bbox
    — and are left alone), raise it so the bottom clears by ``clearance``.

    Writes the adjusted root layer to ``out_path`` (payload arcs kept) and
    returns it. Relative asset paths are anchored before a layer is copied to
    another directory, even when no lift is needed. Otherwise returns the
    original path when nothing penetrates (or pxr is unavailable).
    """
    try:
        from pxr import Usd, UsdGeom, Gf, Sdf, UsdUtils  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return scene_path

    # Work on a private layer: editing Stage.Open(scene_path) changes every
    # open stage sharing USD's layer cache, even without saving that file.
    source = Sdf.Layer.FindOrOpen(scene_path)
    if source is None:
        raise ValueError(f"cannot open foreground scene {scene_path!r}")
    layer = Sdf.Layer.CreateAnonymous("spawn.usda")
    layer.TransferContent(source)
    anchored = False

    def anchor(path):
        nonlocal anchored
        resolved = Sdf.ComputeAssetPathRelativeToLayer(source, path)
        anchored |= resolved != path
        return resolved

    UsdUtils.ModifyAssetPaths(layer, anchor)

    def output(lifted=False):
        if lifted or anchored:
            layer.Export(out_path)
            return out_path
        return scene_path

    fg = Usd.Stage.Open(layer)
    if fg is None:
        return output()
    fg.Load("/")
    objs_root = fg.GetPrimAtPath("/World/Objects")
    if not objs_root or not objs_root.IsValid():
        return output()

    dyn, static_fg = _split_dynamic(objs_root)
    if not dyn:
        return output()

    bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    supports = _static_support_boxes(bc, static_fg, background_path)
    if not supports:
        return output()

    lifted = 0
    for o in dyn:
        r = bc.ComputeWorldBound(o).ComputeAlignedRange()
        if r.IsEmpty():
            continue
        lo, hi = r.GetMin(), r.GetMax()
        bottom = float(lo[2])
        dz, top = _spawn_lift(lo, hi, supports, max_pen, clearance)
        if dz is None:  # resting above / free-falling: fine
            continue
        t_attr = o.GetAttribute("xformOp:translate")
        if not (t_attr and t_attr.IsValid()):
            continue
        v = Gf.Vec3d(t_attr.Get())
        t_attr.Set(Gf.Vec3d(v[0], v[1], v[2] + dz))
        print(
            f"[task_loader] depenetrate_spawn: {o.GetName()} +{dz*1000:.1f}mm "
            f"(bottom {bottom:.4f} < support {top:.4f})",
            flush=True,
        )
        lifted += 1
    if not lifted:
        return output()
    return output(lifted=True)


# Probe heights along the robot's torso. The base is at z=0 and the robot is
# 1.79 m tall, so these suffice to decide whether it is entirely inside a
# convex hull; a room shell will not swallow only half of it.
_SHELL_PROBE_Z = (0.1, 0.5, 1.0, 1.5)


def depenetrate_instance_poses(
    instance_poses: dict,
    scene_path: str,
    background_path: str,
    *,
    max_pen: float = 0.03,
    clearance: float = 0.002,
) -> int:
    """Raise shallowly penetrating dynamic objects in a reset-pose table, in place.

    Parked slots and objects clear of support colliders remain unchanged.
    Returns the number of raised entries. Manifest construction bakes poses
    separately and does not call this helper."""
    if not instance_poses:
        return 0
    try:
        from pxr import Usd, UsdGeom, Gf  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        print(
            "[task_loader] WARN: no pxr, instance_poses will not de-interpenetrate -- "
            "objects will spawn after reset interpenetrating by a few millimetres, and "
            "may get knocked over",
            flush=True,
        )
        return 0

    fg = Usd.Stage.Open(scene_path)
    if fg is None:
        return 0
    fg.Load("/")
    objs_root = fg.GetPrimAtPath("/World/Objects")
    if not objs_root or not objs_root.IsValid():
        return 0
    dyn, static_fg = _split_dynamic(objs_root)
    if not dyn:
        return 0
    bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    supports = _static_support_boxes(bc, static_fg, background_path)
    if not supports:
        return 0

    # Per-slot geometry, taken ONCE from the built scene: the object's AABB
    # expressed relative to its own authored translate and orient. Poses in the
    # table differ from the authored one, so the box is re-oriented per pose by
    # the delta rotation below -- for the yaw-only placements the benchmark
    # actually authors that leaves the height exactly alone and just turns the
    # footprint.
    local = {}
    for o in dyn:
        r = bc.ComputeWorldBound(o).ComputeAlignedRange()
        if r.IsEmpty():
            continue
        t = o.GetAttribute("xformOp:translate")
        if not (t and t.IsValid()):
            continue
        tv = Gf.Vec3d(t.Get())
        q = o.GetAttribute("xformOp:orient")
        qv = q.Get() if (q and q.IsValid()) else None
        rot = Gf.Quatd(float(qv.GetReal()), Gf.Vec3d(*qv.GetImaginary())) if qv is not None else Gf.Quatd(1.0)
        local[o.GetName()] = (Gf.Vec3d(r.GetMin()) - tv, Gf.Vec3d(r.GetMax()) - tv, rot)

    lifted, worst = 0, 0.0
    for iid, table in instance_poses.items():
        for slot, pose in table.items():
            geom = local.get(slot)
            if geom is None or len(pose) < 7:
                continue
            blo, bhi, rot0 = geom
            target = Gf.Quatd(float(pose[3]), Gf.Vec3d(float(pose[4]), float(pose[5]), float(pose[6])))
            delta = Gf.Matrix4d().SetRotate(target * rot0.GetInverse())
            corners = [Gf.Vec3d(x, y, z) for x in (blo[0], bhi[0]) for y in (blo[1], bhi[1]) for z in (blo[2], bhi[2])]
            rc = [delta.TransformDir(c) for c in corners]
            base = Gf.Vec3d(float(pose[0]), float(pose[1]), float(pose[2]))
            lo = Gf.Vec3d(*[min(c[i] for c in rc) + base[i] for i in range(3)])
            hi = Gf.Vec3d(*[max(c[i] for c in rc) + base[i] for i in range(3)])
            dz, _top = _spawn_lift(lo, hi, supports, max_pen, clearance)
            if dz is None:
                continue
            pose[2] = float(pose[2]) + dz
            lifted += 1
            worst = max(worst, dz)
    if lifted:
        print(
            f"[task_loader] depenetrate_instance_poses: lifted {lifted} "
            f"(instance, object) entries, largest +{worst*1000:.1f}mm -- otherwise reset "
            f"would write the benchmark's original height back, and the object would "
            f"spawn interpenetrating and get knocked over",
            flush=True,
        )
    return lifted


def _hull_swallows(vertices, queries) -> "tuple[bool, float]":
    """Whether any of ``queries`` lies inside the convex hull of ``vertices``,
    and how deep in metres. Separate from USD so the pure geometry can be
    tested without a room asset.
    """
    import numpy as np  # noqa: PLC0415
    from scipy.spatial import ConvexHull, Delaunay  # noqa: PLC0415

    verts = np.asarray(vertices, dtype=float)
    pts = np.asarray(queries, dtype=float)
    if len(verts) < 4:
        return False, 0.0
    hull = ConvexHull(verts)
    inside = Delaunay(verts[hull.vertices]).find_simplex(pts) >= 0
    if not inside.any():
        return False, 0.0
    eq = hull.equations
    depth = float(np.max(-(pts[inside] @ eq[:, :3].T + eq[:, 3]).min(axis=1)))
    return True, depth


def disable_swallowing_room_shells(
    background_usda: str,
    robot_xy: "tuple[float, float]",
    out_path: str,
) -> str:
    """Disable collision on room shells that would swallow the robot, and
    return the background USD path to use.

    A ``type="mesh"`` geom is convex in the solver, which takes its hull
    first. A room shell is concave by nature, so its hull is one solid block
    with everything in the room inside it.

    To be precise, MuJoCo does support non-convex collision through SDF
    geoms, height fields, and convex decomposition. We hit convex hulling
    because this pipeline only ever produces mesh geoms: the USD import turns
    meshes into ``Mesh`` geo and the solver exposes no entry point for
    emitting one as an SDF geom. The cost of the SDF path has not been
    measured.

    The benchmark is unaffected because it runs PhysX, which supports static
    concave triangle meshes; ``physics:approximation = "none"`` on the asset
    means exactly "do not approximate".

    Only shells that would SWALLOW the robot are disabled, not every wall: a
    flat wall's hull is roughly itself and is harmless. The criterion is
    collision enabled, ``approximation="none"``, and the robot inside the
    hull.

    Floors are unaffected — a slab's hull is still that slab — and so is the
    door, which is a separate asset under ``objects``.

    This writes an overlay layer and does not touch the original asset. With
    no match it returns the original path and writes nothing.
    """
    if not background_usda:
        return background_usda
    try:
        from pxr import Sdf, Usd, UsdGeom, Gf  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        # Returning silently guarantees a NaN run nobody can explain.
        print(
            f"[task_loader] WARN: cannot inspect the room shell ({exc!r}); "
            f"loading the background as-is. If the shell is concave and has "
            f"collision, physics will diverge on frame 1",
            flush=True,
        )
        return background_usda

    stage = Usd.Stage.Open(background_usda)
    if stage is None:
        print(f"[task_loader] WARN: cannot open background " f"{background_usda!r}; loading as-is", flush=True)
        return background_usda
    stage.Load("/")
    probes = np.array([[robot_xy[0], robot_xy[1], z] for z in _SHELL_PROBE_Z])

    def _collides(prim) -> bool:
        for p in (prim, prim.GetParent()):
            if not p or not p.IsValid():
                continue
            if "PhysicsCollisionAPI" in p.GetAppliedSchemas():
                a = p.GetAttribute("physics:collisionEnabled")
                return (a.Get() is not False) if (a and a.IsValid()) else True
        return False

    targets, checked = [], 0
    for prim in stage.Traverse():
        if "PhysicsMeshCollisionAPI" not in prim.GetAppliedSchemas():
            continue
        approx = prim.GetAttribute("physics:approximation")
        if not (approx and approx.IsValid() and approx.Get() == "none"):
            continue
        if not _collides(prim):
            continue
        pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if pts is None or len(pts) < 4:
            continue
        checked += 1
        xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        world = np.array([tuple(xf.Transform(Gf.Vec3d(*p))) for p in pts])
        # Sample when there are many vertices: the hull depends only on the
        # outer ones, so uniform sampling barely changes the result. The seed
        # is fixed, so one asset always gives the same answer.
        if len(world) > 20000:
            world = world[np.random.default_rng(0).choice(len(world), 20000, replace=False)]
        try:
            swallows, depth = _hull_swallows(world, probes)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[task_loader] WARN: convex hull of {prim.GetPath()} "
                f"failed ({exc!r}); treating it as not swallowing",
                flush=True,
            )
            continue
        if swallows:
            targets.append((str(prim.GetPath()), depth))

    if not targets:
        # Say so even when nothing matched, or "no match" and "this code
        # never ran" are indistinguishable.
        print(
            f"[task_loader] room shell check: {checked} concave colliding "
            f"meshes, none swallowing the robot; loading the background "
            f"as-is",
            flush=True,
        )
        return background_usda

    out = Usd.Stage.CreateNew(out_path) if not Path(out_path).exists() else Usd.Stage.Open(out_path)
    if out is None:
        print(f"[task_loader] WARN: cannot write {out_path!r}; loading the " f"background as-is", flush=True)
        return background_usda
    root = out.GetRootLayer()
    root.subLayerPaths = [str(Path(background_usda).resolve())]
    src = stage.GetRootLayer()
    if src.defaultPrim:
        root.defaultPrim = src.defaultPrim
    for path, depth in targets:
        over = out.OverridePrim(path)
        over.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(False)
        print(
            f"[task_loader] room shell: disabling collision on {path}; its "
            f"convex hull swallows the robot by {depth:.2f}m",
            flush=True,
        )
    root.Save()
    return out_path


#: Mount point of the background room in the scene USD. The labels the
#: checker uses to read scene joint angles carry this prefix, so changing it
#: makes tasks like open_door silently fail to score.
BACKGROUND_PRIM_PATH = "/Background"


def compose_background_into_scene(
    scene_usda: str,
    background_usda: str,
    out_path: str,
) -> str:
    """Mount the background room into the scene USD and return the path to
    hand the engine.

    The background is already inside when the engine opens the scene, so the
    engine needs to do nothing for it. With no background, or a missing file,
    the original path is returned and nothing is written.

    The room is REFERENCED rather than sublayered: a reference mounts its
    contents under ``BACKGROUND_PRIM_PATH``, while a sublayer merges the two
    roots and the room's prims would land at the scene root without that
    prefix.
    """
    if not background_usda:
        return scene_usda
    bg = Path(background_usda)
    if not bg.is_file():
        print(
            f"[task_loader] WARN: background {background_usda!r} does not " f"exist; loading the scene as-is",
            flush=True,
        )
        return scene_usda

    from pxr import Usd  # noqa: PLC0415

    src = Usd.Stage.Open(scene_usda)
    if src is None:
        print(f"[task_loader] WARN: cannot open scene {scene_usda!r}; the " f"background was not mounted", flush=True)
        return scene_usda

    Path(out_path).unlink(missing_ok=True)
    out = Usd.Stage.CreateNew(out_path)
    if out is None:
        print(f"[task_loader] WARN: cannot write {out_path!r}; the background " f"was not mounted", flush=True)
        return scene_usda
    root = out.GetRootLayer()
    root.subLayerPaths = [str(Path(scene_usda).resolve())]
    src_root = src.GetRootLayer()
    if src_root.defaultPrim:
        root.defaultPrim = src_root.defaultPrim
    prim = out.DefinePrim(BACKGROUND_PRIM_PATH, "Xform")
    prim.GetReferences().AddReference(str(bg.resolve()))

    # A second copy under defaultPrim. Both reference the same asset and each
    # serves one consumer:
    #   * the root ``/Background`` — the engine opens this layer directly and
    #     the checker's joint labels come from here;
    #   * the one under defaultPrim — the render node references this layer
    #     under ``{env0}/World``, and a reference takes only the defaultPrim
    #     subtree, discarding root siblings. Without it the render side has no
    #     room and a wrist camera close to an object renders black.
    dp = src_root.defaultPrim
    if dp:
        under = out.DefinePrim(f"/{dp}{BACKGROUND_PRIM_PATH}", "Xform")
        under.GetReferences().AddReference(str(bg.resolve()))
    root.Save()
    print(f"[task_loader] background room mounted at " f"{BACKGROUND_PRIM_PATH}: {bg.name}", flush=True)
    return out_path


_OBJ_ANY_PAYLOAD_RE = None


_ASSET_ARC_RE = None


def _asset_arc_re():
    """Match payload/reference arcs used by scene-object and transform parsing.

    Captures an optional leading ``delete `` (group 1) and the referenced
    asset path (group 2). The ``delete`` group is not decorative: some prims
    author BOTH a ``delete payload = @old@`` (retracting a stale reference)
    and a ``prepend payload = @new@`` (the one actually in effect) on the
    SAME prim, with the ``delete`` arc authored first. Callers MUST skip
    matches where group 1 is truthy -- a plain first-match ``search()``
    would return the RETRACTED asset, not the one the prim actually uses,
    silently merging two geometrically different objects as "the same
    asset" with no error raised anywhere.
    """
    import re  # noqa: PLC0415

    global _ASSET_ARC_RE
    if _ASSET_ARC_RE is None:
        _ASSET_ARC_RE = re.compile(r"(delete\s+)?(?:payload|references)\s*=\s*@([^@]+)@")
    return _ASSET_ARC_RE


def _prim_is_deactivated(body: str) -> bool:
    """Read only the prim's own active flag; inactive children do not hide it."""
    import re  # noqa: PLC0415

    header = re.match(r"\s*\((.*?)\)\s*\{", body, re.S)
    return bool(header) and re.search(r"\bactive\s*=\s*false\b", header.group(1)) is not None


def _usda_delimiters(text: str, start: int = 0):
    """Yield structural delimiters, ignoring quoted values, assets and comments."""
    import re  # noqa: PLC0415

    pattern = re.compile(r'"(?:\\.|[^"\\])*"|@[^@]*@|\#[^\n]*|[(){}]')
    for token in pattern.finditer(text, start):
        if token.group() in ("(", ")", "{", "}"):
            yield token.start(), token.group()


def _own_prim_properties(body: str) -> str:
    """Keep a prim's metadata and properties while excluding descendant bodies."""
    tokens = _usda_delimiters(body)
    parentheses = 0
    for pos, value in tokens:
        if value == "(":
            parentheses += 1
        elif value == ")":
            parentheses -= 1
        elif value == "{" and parentheses == 0:
            start = pos + 1
            break
    else:
        return ""
    parts = [body[:start]]
    depth = 1
    for pos, value in tokens:
        if value == "{":
            if depth == 1:
                parts.append(body[start:pos])
            depth += 1
        elif value == "}":
            depth -= 1
            if depth == 0:
                parts.append(body[start:pos])
                break
            if depth == 1:
                start = pos + 1
    return "".join(parts)


def _parse_scene_objects(scene_usda: str) -> list:
    """Parse ``[(prim_name, asset_id, [x,y,z,qw,qx,qy,qz]), ...]`` from a
    benchmark foreground scene.usda, in AUTHORED order.

    ``asset_id`` is the payload/references asset DIRECTORY (the
    ``Aligned.usda`` filename stripped) — the object's real identity. The prim
    NAME carries a random hash suffix that differs between instances even when
    the asset is identical (``table_7643d40d`` vs ``table_a8ad262d`` both
    payload ``.../benchmark_table_019/Aligned.usda``), which is why identity
    must key off the asset, not the name. Prims with no payload/references
    arc are not objects and are skipped entirely (see below).

    Authored order matters: ``_pair_by_asset`` pairs same-asset objects
    positionally, so this list's order IS the pairing contract.

    Regex-based (machine-generated files with a stable format) to keep this
    module pxr-free / host-testable. Orient defaults to identity when the prim
    authors no orient op.

    An object must carry ``xformOp:translate`` and reference a non-``delete``
    payload or references arc. The type keyword is optional: the benchmark
    contains both ``def Xform "name" (...)`` and ``def "name" (...)``.

    Read each complete object subtree, then exclude descendant properties so
    child transforms cannot replace the object's own pose.
    """
    import re  # noqa: PLC0415

    global _OBJ_ANY_PAYLOAD_RE
    if _OBJ_ANY_PAYLOAD_RE is None:
        # Any prim declaration (typed or not) whose header opens directly
        # with a payload/references arc -- used ONLY for the diagnostic
        # below, never to build `out`, so it can't change parsed output.
        _OBJ_ANY_PAYLOAD_RE = re.compile(
            r'def\s+(?:\w+\s+)?"([A-Za-z_0-9]+)"\s*\(\s*\n?\s*'
            r"(?:delete\s+|prepend\s+|append\s+|add\s+)?(?:payload|references)\s*="
        )
    text = Path(scene_usda).read_text()
    out: list = []
    for name, block, type_kw in _extract_object_xform_blocks(scene_usda):
        body = _own_prim_properties(block[len(f'def {type_kw}"{name}"') :])
        if _prim_is_deactivated(body):
            # ``active = false`` means the prim is not in the composed scene at
            # all. Treating it as an object hands every consumer a member that
            # cannot be resolved: per-instance body-pose baking opens the
            # composed stage, finds nothing at /World/Objects/<name>, and
            # aborts the whole task ("missing object ... for slot ..."). Only
            # pour_workpiece authors this today -- all 20 instances deactivate
            # exactly 3 accessories to vary how many the instance has.
            #
            # Skipping here rather than at the consumer keeps ONE definition of
            # "this instance's objects": the slot table, the alias pairing and
            # the parked-slot fallback are all derived from this list, so the
            # slot still exists (other instances use it) and this instance
            # simply parks it -- which is exactly what an absent object means.
            continue
        t = re.search(r"double3 xformOp:translate = \(([^)]+)\)", body)
        if not t:
            continue
        # First NON-deleted payload/references arc (see _asset_arc_re() above).
        a = next((cand for cand in _asset_arc_re().finditer(body) if not cand.group(1)), None)
        if a is None:
            # A prim with a pose but no asset reference is not an object.
            # This excludes nested joints, which have a translate and no
            # payload, without affecting any previously resolved prim.
            continue
        pos = [float(v) for v in t.group(1).split(",")]
        q = re.search(r"quat[df] xformOp:orient = \(([^)]+)\)", body)
        quat = [float(v) for v in q.group(1).split(",")] if q else [1.0, 0.0, 0.0, 0.0]  # authored (w, x, y, z)
        asset = a.group(2).rsplit("/", 1)[0]
        out.append((name, asset, pos + quat))
    # Loud warning for the KNOWN GAP documented above: a prim declaration that
    # carries a payload/references arc but was never captured into `out`
    # (typically because it omits the `Xform` type keyword). Such objects are
    # invisible to instance-group pairing and pose extraction -- the checker
    # will silently see no such object at all, which is exactly the "physics
    # runs, score never lights up, nothing raises" failure this module exists
    # to surface, not hide.
    parsed_names = {name for name, _, _ in out}
    missing = sorted(
        {cand.group(1) for cand in _OBJ_ANY_PAYLOAD_RE.finditer(text) if cand.group(1) not in parsed_names}
    )
    if missing:
        print(
            f"[task_loader] WARN: {scene_usda} has {len(missing)} object "
            f"prim(s) with a payload/references arc that _parse_scene_objects "
            f'did NOT capture (likely `def "name" (...)` missing the '
            f"`Xform` type keyword): {missing[:5]}"
            f"{', ...' if len(missing) > 5 else ''} -- these objects are "
            f"invisible to instance-group pairing/pose extraction; the "
            f"checker will silently see no such object.",
            flush=True,
        )
    return out


def _parse_scene_object_poses(scene_usda: str) -> dict:
    """``{object_name: [x, y, z, qw, qx, qy, qz]}`` — the name-keyed view of
    ``_parse_scene_objects`` (kept for callers that don't need asset identity:
    ``resolve_superset_instances``, ``_write_superset_scene_usda``)."""
    return {name: pose for name, _, pose in _parse_scene_objects(scene_usda)}


_TABLE_MATERIAL_RE = None


def _table_material_re():
    """Compiled pattern for ``table_000``'s tabletop material override:
    ``over "entity" { over "imagetostl_mesh1" { over "a09___Default" { rel
    material:binding = <...> } } }`` inside a benchmark ``scene.usda``.
    Compiled once, mirrors ``_asset_arc_re()``'s pattern above."""
    import re  # noqa: PLC0415

    global _TABLE_MATERIAL_RE
    if _TABLE_MATERIAL_RE is None:
        _TABLE_MATERIAL_RE = re.compile(
            r'imagetostl_mesh1"[\s\S]*?a09___Default"[\s\S]*?' r"rel material:binding = <([^>]+)>"
        )
    return _TABLE_MATERIAL_RE


def _parse_table_material(scene_usda: str) -> Optional[str]:
    """The material ``table_000``'s tabletop mesh (``entity/imagetostl_mesh1/a09___Default``)
    authors in this instance's ``scene.usda`` -- the composed absolute path with the leading
    ``/`` stripped off (a relative path, for the render node to prepend
    ``/World/envs/env_<id>/`` to), or ``None`` if there is no such ``over`` override (falling
    back to the shared asset ``Aligned.usda``'s own default material, i.e.
    ``a09___Default`` itself -- none of the three tasks
    ``place_sprite_bottle_on_coaster``/``_train``/``_train2`` have this override in their
    scene.usda, so this function naturally returns ``None`` for them, unaffected).

    Only searches within the block for this one object, ``table_000`` (first uses the same
    object-boundary technique as ``_parse_scene_objects`` to isolate ``table_000``'s block,
    then looks for this material override inside that block), to avoid mismatching some
    other object's binding further down."""
    import re  # noqa: PLC0415

    text = Path(scene_usda).read_text()
    m = re.search(r'def (?:Xform )?"table_000"(.*?)(?=def (?:Xform )?"|\Z)', text, re.S)
    if not m:
        return None
    b = _table_material_re().search(m.group(1))
    if not b:
        return None
    return b.group(1).lstrip("/")


def _pair_by_asset(build_objs: list, member_objs: list) -> list:
    """Return ``(build_prim, member_prim)`` pairs, one per build object.

    Match identical asset/name pairs first to keep live build names stable.
    Pair remaining copies of each asset by authored order. Pose and checker
    alias tables must use this same pairing so each checker name identifies
    the body placed at its authored pose. Raise if asset multisets differ.
    """
    from collections import defaultdict  # noqa: PLC0415

    buckets: dict = defaultdict(list)
    for name, asset, _ in member_objs:
        buckets[asset].append(name)
    # Rule 1: exact (asset, name) matches claim their partner up front.
    taken: dict = {}
    for name, asset, _ in build_objs:
        cand = buckets.get(asset) or []
        if name in cand:
            cand.remove(name)
            taken[name] = name
    # Rule 2: the rest take what's left of their bucket, in authored order.
    out: list = []
    for name, asset, _ in build_objs:
        if name in taken:
            out.append((name, taken[name]))
            continue
        cand = buckets.get(asset) or []
        if not cand:
            raise ValueError(
                f"asset multiset mismatch: build prim {name!r} needs another " f"{asset!r}, member has none left"
            )
        out.append((name, cand.pop(0)))
    return out


def resolve_instance_group(
    sub_task_name: str,
    instance_id: int,
    *,
    bench_root: Optional[str] = None,
) -> dict:
    """Find instances sharing the build instance's asset multiset.

    These instances can switch by changing object poses without rebuilding the
    model. Return ``group_instances``, ``instance_poses`` keyed by build prim,
    and ``name_alias`` mapping each member's prims to build prims. Both tables
    come from the same ``_pair_by_asset`` result. Instances with different
    asset multisets require a superset stage or a separate task initialization.
    """
    from collections import Counter  # noqa: PLC0415

    bench = Path(bench_root) if bench_root else _benchmark_config_root()
    base_dir = bench / "llm_task" / sub_task_name
    build_objs = _parse_scene_objects(str(base_dir / str(instance_id) / "scene.usda"))
    ref_sig = Counter(a for _, a, _ in build_objs)
    poses: dict = {}
    alias: dict = {}
    for inst_dir in sorted(base_dir.iterdir(), key=lambda p: (len(p.name), p.name)):
        scene = inst_dir / "scene.usda"
        if not inst_dir.is_dir() or not scene.exists():
            continue
        try:
            objs = _parse_scene_objects(str(scene))
        except Exception:  # noqa: BLE001
            continue
        if Counter(a for _, a, _ in objs) != ref_sig:
            continue
        pairs = _pair_by_asset(build_objs, objs)
        pose_of = {n: p for n, _, p in objs}
        iid = int(inst_dir.name)
        poses[iid] = {bp: pose_of[mp] for bp, mp in pairs}
        alias[iid] = {mp: bp for bp, mp in pairs}
    return {
        "group_instances": sorted(poses.keys()),
        "instance_poses": {str(k): v for k, v in sorted(poses.items())},
        "name_alias": {str(k): v for k, v in sorted(alias.items())},
    }


def _slot_reference_pose(per_inst_objs, instance_ids, variants, variant_key, k):
    """Reference pose for an idle (unused-by-this-instance) slot: the AUTHORED
    pose of the ``k``-th object of this slot's (asset, variant) in the first
    instance that actually has that many copies. Using a real authored pose
    (rather than a synthetic one) keeps the parked z at the same height as the
    live scene, so the parked copy doesn't spawn underground or float absurdly
    high before park_base/park_pitch are added.

    Matching the VARIANT too, not just the asset, keeps that property once an
    asset's slots are split: a variant may exist only in instances that place
    it somewhere else entirely, and borrowing another variant's height would
    park it at the wrong altitude.
    """
    asset, variant = variant_key
    for iid in instance_ids:
        same = [p for n, a, p in per_inst_objs[iid] if a == asset and variants.get((iid, n)) == variant]
        if len(same) > k:
            return same[k]
    return [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]


class _UnsavedUsdLayerError(ValueError):
    """Refreshing a source layer would discard a caller's unsaved USD edits."""


def _object_variants(base, instance_ids, per_inst_objs) -> dict:
    """Return physics-compatible variant keys for each (instance, member prim).

    Use the pose baker's signature so slot sharing respects mass, collider
    layout, scale, material and rigid-body structure, excluding free-body pose.
    Relationship targets use a provisional asset-only alias map; the final baker
    validates compatibility after slots are resolved.

    Discovery failures return {}, allowing asset-only pairing before the baker's
    validation. Reject unsaved USD layers rather than discarding caller edits.
    """
    try:
        from pxr import Usd, UsdGeom, UsdPhysics  # noqa: PLC0415

        from rlinf_geniesim.instance_body_poses import _signature, signature_key  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        print(
            f"[task_loader] WARN: no USD, superset slots will only deduplicate by asset "
            f"and won't distinguish variants ({exc!r})"
        )
        return {}
    out: dict = {}
    try:
        for iid in instance_ids:
            scene = str(Path(base) / str(iid) / "scene.usda")
            stage = Usd.Stage.Open(scene, load=Usd.Stage.LoadAll)
            if stage is None:
                raise ValueError(f"cannot open {scene!r}")
            # USD caches source and payload layers in the process. A task may
            # be rebuilt after either file changes; use their saved contents.
            for layer in stage.GetUsedLayers():
                if layer.realPath and layer.dirty:
                    raise _UnsavedUsdLayerError(
                        f"unsaved USD layer {layer.realPath!r} is in use; save or "
                        "discard those edits before resolving variants"
                    )
            stage.Reload()
            cache = UsdGeom.XformCache()
            cursor: dict = {}
            aliases: dict = {}
            for name, asset, _ in per_inst_objs[iid]:
                k = cursor.get(asset, 0)
                cursor[asset] = k + 1
                aliases[name] = f"s{k}_{asset.rsplit('/', 1)[-1]}"
            joints = [p for p in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()) if p.IsA(UsdPhysics.Joint)]
            for name, _, _ in per_inst_objs[iid]:
                root = stage.GetPrimAtPath(f"/World/Objects/{name}")
                if not root or not root.IsActive():
                    raise ValueError(f"instance {iid}: no active object {name!r}")
                bodies = [
                    p
                    for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
                    if p.HasAPI(UsdPhysics.RigidBodyAPI) and UsdPhysics.RigidBodyAPI(p).GetRigidBodyEnabledAttr().Get()
                ]
                out[(iid, name)] = signature_key(_signature(root, cache, bodies, aliases, joints))
    except _UnsavedUsdLayerError:
        raise
    except Exception as exc:  # noqa: BLE001
        print(
            f"[task_loader] WARN: could not compute variant signatures, superset slots "
            f"will only deduplicate by asset ({exc!r})"
        )
        out = {}
    if out:
        # Only variant identity is needed from here on. Number signatures so
        # the returned map does not retain their mesh digests.
        seen: dict = {}
        for iid in sorted(instance_ids):
            for name, _, _ in per_inst_objs[iid]:
                seen.setdefault(out[(iid, name)], len(seen))
        out = {k: seen[v] for k, v in out.items()}
    return out


def resolve_superset_instances(
    sub_task_name: str,
    instance_ids: list[int],
    *,
    bench_root: Optional[str] = None,
    park_base: float = 20.0,
    park_pitch: float = 10.0,
    park_cols: int = 8,
) -> dict:
    """Deduplicate instance objects into slots grouped by asset and physics variant.

    Each group reserves its maximum multiplicity across instances. Slot names
    use s{k}_{asset_leaf}; duplicate leaf names from different assets raise
    ValueError. Build instance_poses and name_alias in the same pairing pass so
    checker names always refer to the body at the corresponding authored pose.
    Unused slots are parked on a grid indexed by their global slot order.
    """
    if park_pitch < 5.0:
        raise ValueError(
            f"park_pitch={park_pitch} < 5.0 m: parked donors could overlap "
            f"each other or the live scene (tabletop diagonal bound)"
        )
    bench = Path(bench_root) if bench_root else _benchmark_config_root()
    base = bench / "llm_task" / sub_task_name

    # 1) Every instance's objects (authored order), and each asset's slot
    #    count = the max multiplicity of that asset across instances. Also
    #    each instance's OWN table_000 material override (if any) — needed
    #    so a later reset(instances=...) can rebind the shared render
    #    layer's table material to match whichever instance it switches to
    #    (see sim_handler.py's reset()); the superset build itself only
    #    ever splices ONE donor instance's material into the shared scene
    #    (_write_superset_scene_usda), so this table is the only place that
    #    remembers what every OTHER instance actually authored. Paths are
    #    RAW here (still say "table_000", not yet the deduped slot name --
    #    the slot name is only known once step 2 below has run) and get
    #    slot-corrected further down.
    per_inst_objs: dict = {}
    slot_count: dict = {}  # (asset, variant) -> slots needed
    table_materials_raw: dict = {}
    table_asset: Optional[str] = None
    for iid in instance_ids:
        scene_usda = str(base / str(iid) / "scene.usda")
        objs = _parse_scene_objects(scene_usda)
        per_inst_objs[iid] = objs
        table_materials_raw[str(iid)] = _parse_table_material(scene_usda)
        for name, asset, _ in objs:
            if name == "table_000" and table_asset is None:
                table_asset = asset

    # A slot shares one geometry copy, so physical variants require separate slots.
    variants = _object_variants(base, instance_ids, per_inst_objs)
    for iid in instance_ids:
        c: dict = {}
        for name, asset, _ in per_inst_objs[iid]:
            key = (asset, variants.get((iid, name)))
            c[key] = c.get(key, 0) + 1
        for key, n in c.items():
            slot_count[key] = max(slot_count.get(key, 0), n)

    # Variants of one asset are ordered by FIRST APPEARANCE over the instances
    # in ascending id order, so the slot table does not depend on the order the
    # caller happened to list them in -- sim_server resolves name_alias from a
    # separate call and both must agree on every slot name.
    variant_order: dict = {}
    for iid in sorted(instance_ids):
        for name, asset, _ in per_inst_objs[iid]:
            seen = variant_order.setdefault(asset, [])
            variant = variants.get((iid, name))
            if variant not in seen:
                seen.append(variant)

    # 2) Slot table: f"s{k}_{asset_dir_name}". Names are decoupled from every
    #    instance's own prim names -- see docstring for why (take_wrong_item_shelf).
    #    Loud guard: two DIFFERENT full asset paths that happen to share the
    #    same LEAF directory name (e.g. .../a/widget_001 vs .../b/widget_001)
    #    would otherwise compute the identical slot name and silently clobber
    #    each other in `tbl` below -- a slot vanishes from the pose table with
    #    no exception raised, exactly the failure mode this whole redesign
    #    exists to avoid, just relocated to naming instead of pairing.
    #
    #    A split asset keeps ONE run of ``k`` across all of its variants (the
    #    first variant takes s0..., the next continues where it stopped), so a
    #    task with a single variant per asset -- 35 of the 38 -- produces byte
    #    identical slot names to the pre-variant build.
    slots: list = []  # [(slot_name, asset), ...], deterministic order
    leaf_owner: dict = {}  # leaf -> the (only) asset allowed to claim it
    slot_of_variant: dict = {}  # (asset, variant) -> [slot_name, ...]
    for asset in sorted(variant_order):
        leaf = asset.rsplit("/", 1)[-1]
        if leaf in leaf_owner and leaf_owner[leaf] != asset:
            raise ValueError(
                f"slot-name collision in {sub_task_name!r}: assets "
                f"{leaf_owner[leaf]!r} and {asset!r} share the leaf name "
                f"{leaf!r}, so both would compute the slot name "
                f"f's{{k}}_{leaf}' -- refusing to build a superset that "
                f"would silently drop one asset's slots."
            )
        leaf_owner[leaf] = asset
        k = 0
        for variant in variant_order[asset]:
            names = []
            for _ in range(slot_count[(asset, variant)]):
                names.append(f"s{k}_{leaf}")
                slots.append((f"s{k}_{leaf}", asset))
                k += 1
            slot_of_variant[(asset, variant)] = names
    slot_of_asset: dict = {}  # asset -> [slot_name, ...]
    for name, asset in slots:
        slot_of_asset.setdefault(asset, []).append(name)

    # Retarget table_000 material overrides to the assembled superset slot.
    # Tasks without that table keep an empty material mapping.
    table_materials: dict = {"mesh_rel": None, "by_instance": {}}
    if (
        table_asset is not None
        and table_asset in slot_of_asset
        and any(v is not None for v in table_materials_raw.values())
    ):
        table_slot = slot_of_asset[table_asset][0]
        table_materials["mesh_rel"] = f"World/Objects/{table_slot}/entity/imagetostl_mesh1/a09___Default"
        for iid_str, raw in table_materials_raw.items():
            target = raw if raw is not None else ("World/Objects/table_000/entity/Looks/a09___Default")
            table_materials["by_instance"][iid_str] = target.replace(
                "World/Objects/table_000/", f"World/Objects/{table_slot}/", 1
            )

    # 3) Per-instance pairing + parking. instance_poses and name_alias are
    #    built from the SAME pass -- see the hard invariant above.
    poses_out: dict = {}
    alias_out: dict = {}
    for iid in instance_ids:
        cursor: dict = {}
        alias: dict = {}
        used: dict = {}  # slot_name -> this member's pose
        for name, asset, pose in per_inst_objs[iid]:
            key = (asset, variants.get((iid, name)))
            k = cursor.get(key, 0)
            cursor[key] = k + 1
            slot = slot_of_variant[key][k]
            alias[name] = slot
            used[slot] = list(pose)
        tbl = {}
        slot_variant = {n: key for key, names in slot_of_variant.items() for n in names}
        for idx, (slot_name, asset) in enumerate(slots):
            if slot_name in used:
                tbl[slot_name] = used[slot_name]
                continue
            # Idle slot -> park on the 2D grid. rank is the SLOT's own index
            # (no "donor instance" concept once assets are deduped), which
            # still guarantees every idle slot lands in a distinct cell.
            key = slot_variant[slot_name]
            ref = _slot_reference_pose(
                per_inst_objs, instance_ids, variants, key, slot_of_variant[key].index(slot_name)
            )
            parked = list(ref)
            parked[0] += park_base + park_pitch * (idx % park_cols)
            parked[1] += park_pitch * (idx // park_cols + 1)
            tbl[slot_name] = parked
        poses_out[str(iid)] = tbl
        alias_out[str(iid)] = alias

    return {
        "group_instances": list(instance_ids),
        "instance_poses": poses_out,
        "name_alias": alias_out,
        "table_materials": table_materials,
    }


_XFORM_NAME_RE = None


def _extract_object_xform_blocks(scene_usda: str) -> list:
    """Return ``(prim_name, block_text, type_kw)`` for authored object prims.

    Preserve typed Xform versus untyped declarations for later renaming.
    Brace matching keeps each complete subtree without trailing scene scopes.
    An object must have its own translate op and non-delete asset arc; nested
    descendants do not make container prims qualify as objects.
    """
    import re  # noqa: PLC0415

    global _XFORM_NAME_RE
    if _XFORM_NAME_RE is None:
        _XFORM_NAME_RE = re.compile(r'def (Xform )?"([A-Za-z_0-9]+)"')
    text = Path(scene_usda).read_text()
    out = []
    for m in _XFORM_NAME_RE.finditer(text):
        type_kw = m.group(1) or ""
        name = m.group(2)
        i = m.end()
        pm = re.match(r"\s*\(", text[i:])
        meta = ""
        if pm:  # skip prim metadata, e.g.
            depth, j = 0, len(text)
            for pos, value in _usda_delimiters(text, i + pm.end() - 1):
                if value == "(":
                    depth += 1
                elif value == ")":
                    depth -= 1
                    if depth == 0:
                        j = pos + 1
                        break
            meta = text[i:j]
            i = j
        # First NON-deleted payload/references arc, same rule as
        # _parse_scene_objects -- a container ("World"/"Objects"/"Render")
        # has no arc of its own at all and is skipped here (see _asset_arc_re()).
        if not any(not cand.group(1) for cand in _asset_arc_re().finditer(meta)):
            continue
        brace_start = next(pos for pos, value in _usda_delimiters(text, i) if value == "{")
        depth, k = 0, len(text)
        for pos, value in _usda_delimiters(text, brace_start):
            if value == "{":
                depth += 1
            elif value == "}":
                depth -= 1
                if depth == 0:
                    k = pos + 1
                    break
        if "xformOp:translate" not in _own_prim_properties(text[brace_start:k]):
            continue
        out.append((name, text[m.start() : k], type_kw))
    return out


def _scene_kinematic_prim_names(scene_usda: str) -> set:
    """Return object names authored with ``physics:kinematicEnabled = 1``.

    Newton demotes these objects to static colliders before assigning body
    labels. The checker uses this set to suppress expected missing-body alias
    warnings; kinematic objects still participate in instance pairing.
    """
    return {
        name for name, block, _kw in _extract_object_xform_blocks(scene_usda) if "physics:kinematicEnabled = 1" in block
    }


_SUPERSET_TRANSLATE_RE = None


def _write_superset_scene_usda(
    sub_task_name: str,
    instance_ids,
    out_path: str,
    *,
    bench_root: Optional[str] = None,
    active_instance_id: Optional[int] = None,
    scene_paths: Optional[dict] = None,
    instance_group: Optional[dict] = None,
) -> str:
    """Assemble one foreground subtree per deduplicated asset slot.

    Reuse the group's exact ``name_alias`` pairing. Active slots donate their
    own instance block; idle slots use the first real owner, matching the
    baker's parking reference. Payloads, child prims and original property
    targets survive the copy.

    Manifest builds provide ``scene_paths`` and ``instance_group`` from the
    already-corrected per-instance bake. Preserve full composed locator
    transforms (including ancestor scale), place parked slots in their full
    parking frames, and verify startup body poses against the baked table.
    Additional parent transforms keep original transform ops effective for
    graph/property writers. USD is required on this production path.

    Without prepared sources this helper retains its text-only behavior for
    callers that just inspect slot assembly: copy donor blocks and optionally
    rewrite their locator translations. This path does not produce body tables.
    """
    import re  # noqa: PLC0415

    global _SUPERSET_TRANSLATE_RE
    if _SUPERSET_TRANSLATE_RE is None:
        _SUPERSET_TRANSLATE_RE = re.compile(r"(xformOp:translate = \()[^)]*(\))")
    bench = Path(bench_root) if bench_root else _benchmark_config_root()
    base = bench / "llm_task" / sub_task_name
    instance_ids = list(instance_ids)

    full = (
        instance_group
        if instance_group is not None
        else resolve_superset_instances(sub_task_name, instance_ids, bench_root=bench_root)
    )
    # slot -> owning member prim, per instance (inverse of name_alias; only
    # holds entries for slots that instance actually USES -- idle/parked
    # slots for that instance are absent here, same as in name_alias).
    slot_owner: dict = {
        iid: {slot: member for member, slot in full["name_alias"][str(iid)].items()} for iid in instance_ids
    }
    # Canonical slot order: every per-instance pose table was built by
    # resolve_superset_instances iterating the SAME internal `slots` list, so
    # any one of them carries that order via dict insertion order (Python
    # dicts preserve it) -- reuse it here so slot declaration order is
    # deterministic without recomputing the asset/slot-count logic.
    slot_order = list(next(iter(full["instance_poses"].values())).keys())

    active_iid = int(active_instance_id) if active_instance_id is not None else None
    active_tbl = full["instance_poses"].get(str(active_iid)) if active_iid is not None else None

    blocks_cache: dict = {}

    def _blocks_for(iid):
        if iid not in blocks_cache:
            scene = scene_paths[str(iid)] if scene_paths is not None else base / str(iid) / "scene.usda"
            blocks_cache[iid] = {
                name: (block, type_kw) for name, block, type_kw in _extract_object_xform_blocks(str(scene))
            }
        return blocks_cache[iid]

    blocks: list = []
    donors = {}
    for slot_name in slot_order:
        src_iid = None
        if active_iid is not None and slot_name in slot_owner.get(active_iid, {}):
            src_iid = active_iid
        else:
            for iid in instance_ids:
                if slot_name in slot_owner[iid]:
                    src_iid = iid
                    break
        if src_iid is None:
            # Can't happen: slot_order only contains slots that came from
            # resolve_superset_instances' own slot table, and every such slot
            # is, by construction, used by at least one of instance_ids.
            raise ValueError(
                f"slot {slot_name!r} has no owning instance among " f"{instance_ids} -- this should be unreachable"
            )
        member_name = slot_owner[src_iid][slot_name]
        donors[slot_name] = (src_iid, member_name)
        block, type_kw = _blocks_for(src_iid)[member_name]
        block = block.replace(f'def {type_kw}"{member_name}"', f'def {type_kw}"{slot_name}"', 1)
        if scene_paths is None and active_tbl is not None and slot_name in active_tbl:
            x, y, z = active_tbl[slot_name][:3]
            block = _SUPERSET_TRANSLATE_RE.sub(lambda m: f"{m.group(1)}{x!r}, {y!r}, {z!r}{m.group(2)}", block, count=1)
        blocks.append(block)
    body = "\n\n".join(blocks)
    usda = (
        "#usda 1.0\n"
        "(\n"
        '    defaultPrim = "World"\n'
        "    metersPerUnit = 1\n"
        '    upAxis = "Z"\n'
        ")\n\n"
        'def Xform "World"\n'
        "{\n"
        '    def Scope "Objects"\n'
        "    {\n"
        f"{body}\n"
        "    }\n"
        "}\n"
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(usda)
    if scene_paths is not None and active_tbl is not None:
        # Splicing object blocks removes World/Objects ancestors. Preserve
        # their full composed transform (including scale) on the slot itself,
        # then apply the selected/parked complete locator pose. Child paths and
        # graph topology remain intact.
        from pxr import Gf, Usd, UsdGeom

        stages = {}
        merged = Usd.Stage.Open(str(out))
        cache = UsdGeom.XformCache()
        for slot, (iid, member) in donors.items():
            if iid not in stages:
                stages[iid] = Usd.Stage.Open(scene_paths[str(iid)])
            root = stages[iid].GetPrimAtPath(f"/World/Objects/{member}")
            matrix = Gf.Matrix4d(cache.GetLocalToWorldTransform(root))
            # Parking may replace orientation as well as position. Retain
            # donor stretch/scale while replacing its complete rigid frame.
            donor_pose = Gf.Matrix4d().SetRotate(Gf.Transform(matrix).GetRotation())
            donor_pose.SetTranslateOnly(matrix.ExtractTranslation())
            row = active_tbl[slot]
            target_pose = Gf.Matrix4d().SetRotate(Gf.Quatd(row[3], Gf.Vec3d(*row[4:])))
            target_pose.SetTranslateOnly(Gf.Vec3d(*row[:3]))
            matrix = matrix * donor_pose.GetInverse() * target_pose
            xform = UsdGeom.Xformable(merged.GetPrimAtPath(f"/World/Objects/{slot}"))
            # Keep the original ops live: an OmniGraph may target translate
            # or orient directly. Add the lost ancestor/parking transform
            # outside those ops instead of replacing their xformOpOrder.
            extra = xform.GetLocalTransformation().GetInverse() * matrix
            if not Gf.IsClose(extra, Gf.Matrix4d(1), 1e-10):
                ops = xform.GetOrderedXformOps()
                reset_stack = xform.GetResetXformStack()
                parent_op = xform.AddTransformOp(opSuffix="bakedInstanceParent")
                parent_op.Set(extra)
                xform.SetXformOpOrder([parent_op, *ops], reset_stack)
        merged.GetRootLayer().Save()
        # Check the actual assembled startup, including nested reset stacks.
        # Reject an unsupported assembly instead of publishing a reset table
        # that disagrees with the stage on its first frame.
        import numpy as np

        cache.Clear()
        for key, pose in full["instance_body_poses"][str(active_iid)].items():
            body = merged.GetPrimAtPath(f"/World/Objects/{key}")
            matrix = cache.GetLocalToWorldTransform(body)
            quat = Gf.Transform(matrix).GetRotation().GetQuat().GetNormalized()
            q = np.array([quat.GetReal(), *quat.GetImaginary()])
            if (
                not np.allclose(matrix.ExtractTranslation(), pose[:3], atol=1e-6, rtol=0)
                or abs(float(np.dot(q, pose[3:]))) < 1 - 1e-6
            ):
                raise ValueError(
                    f"superset startup body {key!r} disagrees with instance "
                    f"{active_iid}; regroup or rebuild the scene"
                )
    return str(out)


# Keep the most recent resolved spec and build-time plans in this process.
# None means no completed resolution; empty domains mean resolved but disabled.
_LAST_BUILD_RANDOMIZATION: Optional[dict] = None


def _record_build_randomization(*, resolved: dict, build_time_plans: dict) -> None:
    """Records this build's normalized spec and build-time plan (see ``_LAST_BUILD_RANDOMIZATION``)."""
    global _LAST_BUILD_RANDOMIZATION
    _LAST_BUILD_RANDOMIZATION = {"resolved": resolved, "build_time_plans": build_time_plans}


def last_build_randomization() -> Optional[dict]:
    """The generalization readback record for the most recent ``build_stage_manifest``, or
    ``None`` if no build has happened.

    ``None`` **must** be faithfully conveyed by the caller as "don't know yet", and must not
    be collapsed into an empty dict -- otherwise "this process has never built a stage" and
    "built, but generalization is entirely off" would look identical on the wire, exactly
    the kind of false positive this repo keeps deleting."""
    return _LAST_BUILD_RANDOMIZATION


def build_stage_manifest(
    task_name: str,
    sub_task_name: str,
    instance_id: int,
    cameras: list[dict],
    out_path: str,
    *,
    extra_newton_cfg: Optional[dict] = None,
    render_layer_usda: str = "",
    author_render: bool = False,
    use_background: bool = False,
    randomization: Optional[dict] = None,
    bench_root: Optional[str] = None,
    assets_root: Optional[str] = None,
    superset_instance_ids: Optional[list] = None,
    num_envs: int = 1,
    seed: Optional[int] = None,
    env_spacing: float = 15.0,
) -> str:
    """Build a benchmark stage manifest and return its output path.

    cameras contains task-init observation camera definitions. When author_render
    is enabled, author a render layer unless the caller supplies one.

    superset_instance_ids selects a merged foreground and shared instance-switch
    table; instance_id must identify its build instance. None selects the single
    instance foreground. Robot configuration, cameras and background remain
    based on the selected benchmark task.

    num_envs, seed and env_spacing define build-time background assignment when
    randomization.render.background.enable is set. This seed comes from init,
    independently of per-reset randomization seeds.
    """
    r = resolve_benchmark_task(
        task_name,
        sub_task_name,
        instance_id,
        bench_root=bench_root,
        assets_root=assets_root,
    )

    from rlinf_geniesim.instance_body_poses import bake_instance_body_poses

    # Resolve pairing once, then capture each original composed scene BEFORE
    # selecting a shared subtree. Its own child overrides determine body poses.
    if superset_instance_ids is not None:
        grp = resolve_superset_instances(sub_task_name, superset_instance_ids, bench_root=bench_root)
    else:
        grp = resolve_instance_group(sub_task_name, instance_id, bench_root=bench_root)
    grp["build_instance"] = int(instance_id)
    bench = Path(bench_root) if bench_root else _benchmark_config_root()
    out = Path(out_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    from rlinf_geniesim.robot_attachments import bake_fixed_attachments

    r["robot_usda"] = bake_fixed_attachments(
        r["robot_usda"], r.get("robot_fixed_attachments", []), str(out.with_name(out.stem + "_robot.usda"))
    )
    corrected_scenes = {}
    for iid in grp["group_instances"]:
        original = bench / "llm_task" / sub_task_name / str(iid) / "scene.usda"
        corrected_scenes[str(iid)] = depenetrate_spawn(
            str(original),
            r.get("background_usda", "") if use_background else "",
            str(out.with_name(f"{out.stem}_instance_{iid}_depen.usda")),
        )
    grp.update(bake_instance_body_poses(scene_paths=corrected_scenes, group=grp))

    if superset_instance_ids is not None:
        r["scene_usda"] = _write_superset_scene_usda(
            sub_task_name,
            superset_instance_ids,
            str(out.with_name(out.stem + "_superset_scene.usda")),
            bench_root=bench_root,
            active_instance_id=instance_id,
            scene_paths=corrected_scenes,
            instance_group=grp,
        )
    else:
        r["scene_usda"] = corrected_scenes[str(instance_id)]

    # Mount the background room into the scene the engine opens, so the
    # upstream engine needs no background handling of its own.  After
    # depenetrate_spawn, which wants foreground and background separate.
    if use_background:
        _composed = compose_background_into_scene(
            r["scene_usda"],
            r.get("background_usda", ""),
            str(Path(out_path).resolve().with_name(Path(out_path).stem + "_bg.usda")),
        )
        if _composed != r["scene_usda"]:
            r["scene_usda"] = _composed
            # Clear this once merged into the scene: the render node is its
            # only consumer, and leaving it mounts the room a second time.
            # With two overlapping copies, a wrist camera close to a surface
            # ends up inside a doubled wall and renders black, while the head
            # camera is far enough away to look fine.
            r["background_usda"] = ""

    # Author render_layer.usda on demand (render ON). Reuses the upstream
    # author_render_layer; needs pxr + the robot USD (container). Skipped when
    # render is off (render_layer stays empty -> null bridge / physics-only).
    if author_render and not render_layer_usda:
        # The authorer overwrites render_layer.usda inside its output directory.
        # Isolate that directory as well as the scene/config filenames.
        render_out_dir = Path(out_path).resolve().with_name(Path(out_path).stem + "_render")
        render_out_dir.mkdir(parents=True, exist_ok=True)
        render_layer_usda = author_task_render_layer(str(render_out_dir), r["robot_usda"], r["robot_prefix"], cameras)

    # The render node opens scene_usda as its base stage; the foreground scene
    # has no room and no lights, so wrap it with a default environment (floor +
    # bright global lighting) when rendering. (No wrap when render is off —
    # physics-only path keeps the scene verbatim.)
    # Robot base pose -> scene-cfg the engine reads via manifest["scene_yaml"].
    # The engine's _apply_init_base_pose authors translate+orient on the robot
    # root (== benchmark's XFormPrim on /genie); theta = yaw from the quaternion.
    rip = resolve_robot_init_pose(task_name, sub_task_name, instance_id, bench_root=bench_root)
    _pos = list(rip.get("position", [0.0, 0.0, 0.0]))
    # A background room's floor is not necessarily at z=0, while the
    # benchmark's robot_init_pose usually omits z. Correct it from the room
    # table when a background is loaded, lifting only on real penetration.
    # Both outputs must use the SAME z, or the engine and the client believe
    # different things.
    if use_background and r.get("background_usda", ""):
        _z, _why = resolve_base_z(room_key(r["background_usda"]), float(_pos[2]))
        print(f"[task_loader] robot base z: {_why}", flush=True)
        _pos[2] = _z
        rip = {**rip, "position": _pos}
        # Disable collision on room shells that swallow the robot. After
        # depenetrate_spawn: that step ignores collision flags when finding a
        # support, but disabling first would make this criterion disagree with
        # what the engine actually loads.
        r["background_usda"] = disable_swallowing_room_shells(
            r["background_usda"],
            (float(_pos[0]), float(_pos[1])),
            str(Path(out_path).resolve().with_name(Path(out_path).stem + "_bg_noshell.usda")),
        )
    scene_usda = r["scene_usda"]
    # Empty unless the env-shell path is on; the manifest field defaulting to
    # "" is what makes the render node fall back to the old open-and-deactivate
    # behaviour.
    env_shell_usda = ""
    # The room is authored here for BOTH consumers, so this cannot be gated on
    # rendering. The engine opens this same file as its physics stage, and
    # ``render=none`` runs (headless physics checks, ``author_render=False``
    # callers) would otherwise get a scene with no room at all -- e.g. a door
    # task compiling to an MJCF with zero door geoms. Physics must always see
    # the room it needs, independent of whether anything renders it. With no
    # background and no render there is nothing to wrap: the synthetic floor
    # and lights exist only for the camera.
    _bg_for_scene = r.get("background_usda", "") if use_background else ""
    if author_render or _bg_for_scene:
        # The same ``use_background`` decision the physics side acts on -- the
        # two must not disagree about which rooms exist (see the wrapper's
        # docstring for what that looked like when they did).
        #
        # Deliberately AFTER ``disable_swallowing_room_shells``: that function
        # rewrites ``r['background_usda']`` to a copy with the room's outer
        # shell stripped of collision, because MuJoCo convex-hulls a shell
        # mesh into a solid block that swallows the robot. Referencing the
        # original instead puts the robot inside that hull, which produces
        # NaN joint states at the startup reset.
        _rig, _rig_name, _rig_src = resolve_light_rig(sub_task_name)
        print(f"[task_loader] light rig: {_rig_name} (from {_rig_src})", flush=True)
        # The default floor is authored ONCE at ``/RLinfEnv/Floor``, outside
        # the GridCloner's envs_scope -- it is global/shared, not cloned per
        # env (see the multi-env comment on ``foreground_usda`` below). The
        # cloner lays out ``num_envs`` envs on a single row (grid_spacing
        # apart, centered on the world origin -- ``GridCloner(num_per_row=
        # num_envs)`` above), so a fixed half-extent only covers the env(s)
        # nearest the origin; with many envs, the ones further out would
        # stand over an empty grey plane. Size it to the full grid instead:
        # each env owns a spacing-wide cell, so half the total grid width
        # (num_envs * env_spacing / 2) reaches every cell's edge. ``max()``
        # keeps the single-env case using the same 15.0 default as before.
        _floor_half = max(15.0, num_envs * env_spacing / 2.0)
        scene_usda = _wrap_scene_with_default_env(
            r["scene_usda"],
            str(Path(out_path).resolve().with_name(Path(out_path).stem + "_scene_env.usda")),
            background_usda=_bg_for_scene,
            light_rig=_rig,
            floor_half=_floor_half,
        )
        # The stage the render node opens: same global env, no foreground.
        # See ``write_env_shell_usda``.
        if _emit_env_shell():
            env_shell_usda = write_env_shell_usda(
                str(Path(out_path).resolve().with_name(Path(out_path).stem + "_env_shell.usda")),
                background_usda=_bg_for_scene,
                light_rig=_rig,
                floor_half=_floor_half,
            )
            print(
                f"[task_loader] env shell: {Path(env_shell_usda).name} "
                f"(render node opens this instead of the full scene)",
                flush=True,
            )

    scene_cfg = {
        "robot": {
            # ``robot.robot_source.pin_base_to_world`` is deliberately NOT set, so
            # it stays at the engine's default of False and the run takes
            # ``pin_base_mode=free`` (plain substep: real gravity and contacts).
            #
            # Setting it True looks right for a stationary tabletop robot -- it is
            # what the engine's own aloha / ur5 / fr3 bringup configs do -- but
            # those are arm benches with nothing on the table. Here it selects
            # ``_substep_body_kinematic_control``, which zeroes gravity AND
            # contacts for the WHOLE model, so every block hangs frozen in
            # mid-air at its spawn height. The engine says as much
            # in ``_bind_substep_body``: that regime "freezes every dynamic object
            # in a benchmark manipulation scene".
            #
            # The base is still held under ``free``, because the robot's root weld
            # is never actually dropped: ``_deactivate_root_joint`` requires the
            # joint to name a ``body0``, and both shipped G2 robots author theirs
            # with body0 ABSENT. So the effective regime is "plain substep with a
            # surviving weld", which is what these tasks want but is not one of the
            # three named modes. Anyone who makes that deactivation work must pair
            # it with ``pin_base_mode=mocap`` (real physics, base clamped per
            # substep) or the robot will start falling over.
            "init_base_pose": {
                "x": float(_pos[0]),
                "y": float(_pos[1]),
                "z": float(_pos[2]),
                "theta": _quat_to_yaw(rip.get("quaternion", [1.0, 0.0, 0.0, 0.0])),
            },
        }
    }
    # Per-reset seeded object randomization spec (engine _randomize_world_objects
    # reads scene_cfg.randomization): {object_xy_radius, exclude, min_gap, ...}.
    # The engine has no separate `enabled` gate -- a real per-reset seed (>=0) is
    # the WHOLE trigger for object-xy randomization, subject only to
    # object_xy_radius > 0 (default 0.03, so present unless a caller
    # explicitly zeroes it). Any `enabled` key a caller still passes here is
    # vestigial from the engine's point of view -- it is simply never read.
    # The only real way to opt an instance's OBJECTS out of randomization
    # while still passing a real seed (for arm/base generalization
    # elsewhere) is `object_xy_radius: 0`; to opt everything out, pass
    # seed=-1 at reset time instead (see rlinf_geniesim's own seed contract:
    # seed>=0 => generalize, seed==-1 => verbatim, no config gate between).
    if randomization:
        scene_cfg["randomization"] = dict(randomization)
    # Startup and reset use the same already-corrected instance sources.
    # A failed bake must abort, never silently disable instance switching.
    _rb = resolve_instance_robot_init(sub_task_name, grp["group_instances"], int(instance_id), bench_root=bench_root)
    if _rb:
        grp["robot_base_xy"] = _rb["robot_base_xy"]
        grp["robot_joints"] = _rb["robot_joints"]
        print(
            f"[task_loader] {sub_task_name}: per-instance robot initial state "
            f"{len(_rb['robot_joints'])} entries (reference = instance {instance_id})"
        )
    scene_cfg["instance_switch"] = grp

    # BENCHMARK GENERALIZATION (L1a, physics-side keys): merge the eval task's
    # init_joint / init_base thresholds into the seeded-reset spec. Explicit
    # yaml keys win over the benchmark block; gate with
    # randomization.use_benchmark_generalization: false to opt out.
    # Logic lives in randomize.spec.RandomizationSpec; key names and priority
    # order are unchanged. The per-task defaults (config/task_generalization.yaml)
    # sit between the explicit yaml keys and the benchmark block; see
    # randomize.task_defaults.
    spec = RandomizationSpec.resolve(
        task_name,
        scene_cfg.get("randomization"),
        bench_root=bench_root,
        task_defaults=task_defaults.for_task(sub_task_name, instance_switch=grp),
    )
    if spec:
        scene_cfg["randomization"] = spec.to_dict()

    # Joint PD gain randomization (joint_pd, physics side): turns the parameter domain
    # already normalized by RandomizationSpec.resolve() above (``spec.physics["joint_pd"]``,
    # one range as ``{enable, kp_scale, kd_scale}``) into **specific per-env multipliers**,
    # written back into ``scene_cfg["randomization"]["joint_pd"]``. After this step runs,
    # this key's shape changes from "one range" to "specific values keyed by per-env string"
    # (``{"0": {"kp_scale": ..., "kd_scale": ...}, ...}``, the same convention as
    # ``instance_switch.robot_base_xy``) -- the engine side
    # (``lifecycle.py::_apply_joint_pd_randomization``) only applies this shape, it does not
    # sample (see the "joint_pd sub-stream" section of ``randomize/sampler.py``'s module
    # docstring for why this dimension must be fully sampled here in one shot, rather than
    # leaving a range for the engine to sample fresh by seed the way the other three items
    # like object_xy_radius do -- ``scene_cfg`` is written once at stage-build time and is
    # static afterwards, unlike ``reset(seeds=...)``, which can carry a fresh value every
    # time).
    #
    # When left at the default off (``enable`` is not true), no ``joint_pd`` key is left in
    # ``scene_cfg["randomization"]`` at all -- the ``elif`` branch below clears out the
    # "domain" shape too, so the invariant "this key existing = at least one env really has
    # a specific multiplier configured" holds at the config level itself.
    #
    # ``_jpd_build_plans`` keeps the full per-env sampled plan (with seed) for
    # ``_record_build_randomization`` to retain for readback; this build does not consume it.
    _jpd_build_plans: dict = {}
    joint_pd_spec = (spec.physics or {}).get("joint_pd") or {}
    if joint_pd_spec.get("enable", False):
        # ``seed`` is the top-level task-init seed used for the one-time assignment at build
        # time (not a reset seed, since there isn't one yet at this point) -- the same
        # discipline as background room assignment (see the matching check in the background
        # section below); its absence is a config problem, and refusing loudly is better than
        # silently not taking effect.
        if seed is None:
            raise ValueError(
                f"randomization.joint_pd.enable=true for sub_task "
                f"{sub_task_name!r}, but no top-level task-init `seed` was "
                f"given to build_stage_manifest -- per-env joint_pd gain "
                f"assignment happens once at stage-build time (scene_cfg."
                f"randomization is static after build; there is no "
                f"reset(seeds=[...]) yet at this point -- see randomize/"
                f"sampler.py's module docstring for why), so it needs the "
                f"task-init payload's top-level `seed` to derive a "
                f"deterministic per-env multiplier. Add `seed:` to the "
                f"task-init payload."
            )
        _jpd_seed = int(seed)
        if _jpd_seed < 0:
            # -1 is the sentinel for the byte-for-byte reproduction contract -- this
            # dimension, like every other one, produces no plan at all when it's -1,
            # consistent with Sampler.sample()'s existing convention.
            scene_cfg["randomization"]["joint_pd"] = {}
        else:
            # Only consumes the spec for this one physics sub-key, joint_pd -- otherwise
            # this call would also draw a lights/material/camera-pose plan that's never
            # consumed, and could cause the physics-side keys spec.to_dict() already wrote
            # (e.g. object_xy_radius) to get unexpectedly rebuilt and lost.
            _jpd_only_spec = RandomizationSpec(physics={"joint_pd": joint_pd_spec})
            _jpd_seeds_map = {i: _jpd_seed for i in range(int(num_envs))}
            _jpd_plans = Sampler(_jpd_only_spec).sample(_jpd_seeds_map)
            # Kept for readback (see ``_jpd_build_plans`` above).
            _jpd_build_plans = {str(i): p.to_dict() for i, p in _jpd_plans.items()}
            _jpd_per_env: dict = {}
            for _i in range(int(num_envs)):
                _plan = _jpd_plans.get(_i)
                _jpd = _plan.physics.get("joint_pd") if _plan is not None else None
                if _jpd:
                    _jpd_per_env[str(_i)] = _jpd
            scene_cfg["randomization"]["joint_pd"] = _jpd_per_env
    elif "joint_pd" in (scene_cfg.get("randomization") or {}):
        # Turned off, but ``spec.to_dict()`` left a domain-shaped value behind (an explicit
        # ``randomization.joint_pd: {enable: false, ...}`` was given). Delete it so "this
        # key exists" keeps meaning "at least one env has a specific multiplier configured"
        # -- a domain dict's keys (``enable``/``kp_scale``/``kd_scale``) could never collide
        # with a per-env string index, so the engine would not misread it either way, but
        # leaving it would make it easy to mistake for live config.
        del scene_cfg["randomization"]["joint_pd"]

    # Per-env background room (render side): reuses the spec.render already resolved by the
    # RandomizationSpec.resolve() call above, rather than opening a second entry point that
    # parses randomization.render itself.
    #
    # When left at the default off (``enable`` is not true), this whole section does not
    # execute at all: discover_background_rooms is not called, Sampler is not called, and
    # background_usda_per_env does not appear as a key in the manifest either (the
    # manifest-dict construction below uses ``background_usda_per_env is not None`` as the
    # switch).
    background_spec = (spec.render or {}).get("background") or {}
    background_usda_per_env: Optional[list[str]] = None
    # Keeps the full per-env sampled plan (with seed) for readback, same as joint_pd's
    # ``_jpd_build_plans`` above.
    _bg_build_plans: dict = {}
    if background_spec.get("enable", False):
        # Gate: tasks marked True in task_background.TASK_NEEDS_BACKGROUND have a background
        # that is a PHYSICAL SUPPORT (the object rests on a door/stove/shelf that lives in
        # the background). Swapping rooms pulls that support out from under the object --
        # it falls silently, no error, just worse scoring -- so this raises rather than warns.
        #
        # Tasks not listed in the table at all are likewise refused: nobody has verified
        # whether their background is a physical support, and guessing wrong costs the same
        # silent breakage. Better to hard-code a table with a test watching it for gaps than
        # to miss one.
        #
        # Only the TASK_NEEDS_BACKGROUND table is checked here, not ``use_bg``
        # (task_background.resolve_use_background) -- most decorative tasks marked False in
        # the table also have use_bg False (the foreground brings its own support surface,
        # so no background is attached by default), but the point of this dimension is to
        # let them attach a purely decorative, freely swappable background room too. use_bg
        # must not be used as the criterion for "can this be randomized" -- the two concerns
        # are layered separately.
        _needs_bg = TASK_NEEDS_BACKGROUND.get(sub_task_name)
        if _needs_bg is True:
            raise ValueError(
                f"randomization.render.background.enable=true, but sub_task "
                f"{sub_task_name!r} is marked True in "
                f"task_background.TASK_NEEDS_BACKGROUND -- its background "
                f"room is a PHYSICAL SUPPORT (the object rests on a "
                f"shelf/stove/workbench that lives in the background, not "
                f"on its own table). Randomizing the room would swap out "
                f"that support out from under the object; the object falls "
                f"SILENTLY (no error, just a worse return). To run this "
                f"task, turn off randomization.render.background."
            )
        if _needs_bg is None:
            raise ValueError(
                f"randomization.render.background.enable=true, but sub_task "
                f"{sub_task_name!r} is not listed in "
                f"task_background.TASK_NEEDS_BACKGROUND at all -- nobody has "
                f"verified whether its background is a physical support or "
                f"purely decorative, and guessing wrong fails the same way "
                f"as the physical-support case above: silently, as a worse "
                f"return with no error. Add it to TASK_NEEDS_BACKGROUND "
                f"(verify whether the foreground brings its own supporting "
                f"surface) before enabling this dimension for it."
            )
        # Passed the gate (the table explicitly marks it False, decorative). ``seed`` is the
        # top-level task-init seed used for the one-time assignment at build time (not a
        # reset seed, since there isn't one yet at this point); its absence is a config
        # problem -- without it there's no way to derive a per-env value deterministically,
        # so refusing loudly is better than silently not taking effect.
        if seed is None:
            raise ValueError(
                f"randomization.render.background.enable=true for sub_task "
                f"{sub_task_name!r}, but no top-level task-init `seed` was "
                f"given to build_stage_manifest -- per-env background room "
                f"assignment happens once at stage-build time (there is no "
                f"reset(seeds=[...]) yet at this point), so it needs the "
                f"task-init payload's top-level `seed` to derive a "
                f"deterministic per-env pick. Add `seed:` to the task-init "
                f"payload."
            )
        seed = int(seed)
        if seed < 0:
            # -1 is the sentinel value for the byte-for-byte reproduction contract (see how
            # substreams()/Sampler.sample() handle a negative seed) -- this dimension, like
            # every other physics/sensor/render dimension, produces no plan at all when it's
            # -1. Left as a fully empty list (still an honest record that "this dimension was
            # turned on", not "not configured"), with no error raised -- consistent with
            # Sampler.sample()'s existing convention.
            background_usda_per_env = []
        else:
            _bg_assets_root = Path(assets_root) if assets_root else _assets_root()
            _bg_root = _bg_assets_root / "background"
            _max_footprint_m = background_spec.get("max_footprint_m")
            if _max_footprint_m is None:
                _max_footprint_m = float(env_spacing)
            _candidates = discover_background_rooms(
                _bg_root, max_footprint_m=float(_max_footprint_m), rooms=background_spec.get("rooms") or None
            )
            # Only consumes the spec for this one background sub-key, not
            # lights/material/sensor -- otherwise this call would also draw a lights/sensor
            # plan that never gets consumed, making it easy to mistakenly think this is also
            # managing those two dimensions (their own sampling happens inside
            # isaac_tiled.py's on_reset(), once per reset, a separate path from the "once at
            # build time" happening here).
            _bg_only_spec = RandomizationSpec(physics={}, render={"background": background_spec})
            _seeds_map = {i: seed for i in range(int(num_envs))}
            _plans = Sampler(_bg_only_spec).sample(_seeds_map, background_candidates=_candidates)
            # Kept for readback (see ``_bg_build_plans`` above).
            _bg_build_plans = {str(i): p.to_dict() for i, p in _plans.items()}
            background_usda_per_env = []
            for _i in range(int(num_envs)):
                _plan = _plans.get(_i)
                _room = (_plan.render.get("background") or {}).get("room") if _plan is not None else None
                background_usda_per_env.append(resolve_room_usda(_bg_root, _room) if _room else "")

    # Per-instance robot base poses (instance_switch.robot_base_xy) and benchmark
    # base-pose generalization (randomization.init_base_xy_thresh) write the SAME
    # base_mocap_off[w] slot in engine reset.py, and the randomization step runs
    # second -- so turning both on would silently clobber the per-instance pose
    # while both code paths report success. Raise loudly instead of leaving it
    # to write order to silently pick a winner.
    check_base_xy_channel_conflict(
        task_name=task_name,
        sub_task_name=sub_task_name,
        instance_id=instance_id,
        instance_switch=scene_cfg.get("instance_switch"),
        randomization=scene_cfg.get("randomization"),
    )

    # Contact compliance, in two tiers welded together by the robot side's
    # ``priority``:
    #   * robot                         solref (0.002, 2.5) -> ke 40000, for grasping
    #   * floor / passive / static_prop  solref (0.010, 1.0) -> ke 10000, for settling
    #
    # ke is roughly 1/timeconst^2, so 0.002 is a hundred times stiffer than
    # MuJoCo's default. That extreme is chosen for grasping and simultaneously
    # makes objects unstable when they land on a table. timeconst is the only
    # knob that improves this; time step, solver iterations, friction cone,
    # multi-contact, and primitive table shapes were all tried without effect.
    #
    # Grasping still uses 0.002 via ``priority`` below, which makes the robot
    # side's stiffness structural rather than a value that happens to be
    # right.
    #
    # Restitution and residual motion still differ from PhysX by about two
    # orders of magnitude. That is a soft-constraint versus hard-constraint
    # difference, which no solref value closes — do not chase it here.
    #
    # Robot friction [1.5, 0.1, 0.01] comes from the per-class
    # recommendations. Pairing takes the larger of the two surfaces, so
    # raising the robot side alone lifts fingertip-object tangential friction
    # without affecting object-table.
    _contact = {kind: {"solref": [0.010, 1.0]} for kind in ("floor", "passive", "static_prop", "other")}
    # Overrides the four non-robot classes (floor / passive / static_prop /
    # other); without one the default is (0.010, 1.0). Two entry points, with
    # the environment winning over the yaml.
    #
    # This affects object-table and object-object only: the robot side carries
    # priority=1 and its solref is adopted wholesale on unequal priority, so
    # robot-object is decided entirely by the robot side.
    _object_solref = _resolve_solref(
        env_var="RLINF_OBJECT_SOLREF", yaml_key="object_solref", factory=[0.010, 1.0], label="object"
    )
    for _kind in _contact:
        _contact[_kind]["solref"] = list(_object_solref)
    # Overrides the robot side; without one the default is (0.002, 2.5). Two
    # entry points, with the environment winning over the yaml.
    #
    # Why the knob exists: escape velocity is v = delta * sqrt(ke/m), so
    # multiplying mass and dividing stiffness are equivalent. Changing mass
    # changes the task, so stiffness is the production-safe side. The known
    # side effect is finger penetration.
    #
    # timeconst stays at 0.002: it is bounded below by sub_dt and kd =
    # 2/timeconst follows from it. dampratio is the only axis that moves ke
    # without moving kd, trading a critically damped very stiff contact for an
    # overdamped softer one.
    #
    # At the current dampratio (2.5), the robot side's ke=40000 sits above the
    # passive side's ke=10000, which is the two-tier split this contact model
    # is designed around. A dampratio of 5.0 would flatten both tiers to the
    # same ke, collapsing the split down to a difference in damping alone.
    #
    # The default lives only in ``factory=`` below; physics_overrides.yaml
    # deliberately does not repeat it, since two defaults drift.
    _robot_solref = _resolve_solref(
        env_var="RLINF_ROBOT_SOLREF", yaml_key="robot_solref", factory=[0.002, 2.5], label="robot"
    )
    _contact["robot"] = {"solref": _robot_solref}
    # priority: on unequal priority the higher side's solref, solimp, and
    # condim are adopted wholesale with no solmix blending, so robot-object
    # keeps the stiff contact while object-table gets the softer one. Friction
    # takes the per-component maximum regardless of priority in this kernel,
    # so the robot friction triple below is unaffected.
    _contact["robot"]["priority"] = 1
    # torsional 0.5 / rolling 0.2: a gripped object otherwise rolls along the
    # fingertip arc as the contact point migrates under disturbance, with grip
    # force decaying as it does. Rolling and torsional friction resist that
    # migration torque directly.
    _contact["robot"]["friction"] = [2.0, 0.5, 0.2]
    # Per-class scene policy: explicit task values override generated defaults.
    from copy import deepcopy

    newton_cfg = deepcopy(extra_newton_cfg or {})
    for kind, values in (newton_cfg.pop("mjc_contact", {}) or {}).items():
        _contact.setdefault(kind, {}).update(values)
    scene_cfg["newton"] = {**newton_cfg, "mjc_contact": _contact}
    scene_cfg["newton"]["eval_static_props"] = list(r.get("static_prop_aabbs") or [])
    scene_cfg_path = str(Path(out_path).resolve().with_name(Path(out_path).stem + "_scene_cfg.json"))
    Path(scene_cfg_path).write_text(json.dumps(scene_cfg, indent=2))

    cam_entries: list[dict[str, Any]] = []
    for i, c in enumerate(cameras or []):
        h, w = c.get("image_hw") or [256, 256]
        name = c.get("name", f"cam_{i}")
        intr = c.get("intrinsic", {}) or {}
        # cam_rel_path / render_product_path MUST match what author_render_layer
        # created: it authors /RenderOVRTX/Cameras/<prim_path> and
        # /RenderOVRTX/Cam_<i>, using prim_path = prim_path|link|name.
        cam_rel_path = c.get("prim_path") or c.get("link") or name
        cam_entries.append(
            {
                "name": name,
                "render_product_path": f"/RenderOVRTX/Cam_{i}",
                "topic": c.get("topic", f"/rlinf/{name}/image_raw"),
                "depth_topic": c.get("depth_topic", ""),
                # Fields the OVRTX render node reads from the manifest (mirror the
                # assemble_scene camera-entry schema; missing 'path' crashed it).
                "path": cam_rel_path,
                "frame_id": c.get("frame_id", ""),
                "link": c.get("link", ""),
                "width": int(w),
                "height": int(h),
                # NATIVE benchmark resolution (falls back to tile w/h) — the render
                # node derives each camera's vertical FOV from this aspect so non-1.6
                # cameras stretch, not crop, into the uniform tile.
                "native_width": int((c.get("native_hw") or [h, w])[1]),
                "native_height": int((c.get("native_hw") or [h, w])[0]),
                "fx": float(intr.get("fx", 610.0)),
                "fy": float(intr.get("fy", 610.0)),
                "cx": float(intr.get("cx", w / 2.0)),
                "cy": float(intr.get("cy", h / 2.0)),
                "model": (c.get("model") or "").strip(),
                # Debug freecam (tiled/GridCloner path): if present, the render node
                # AUTHORS an env-anchored look-at camera at these env-local offsets
                # instead of resolving a robot camera. Harmless when absent.
                **(
                    {"freecam_eye": c["freecam_eye"], "freecam_target": c.get("freecam_target")}
                    if c.get("freecam_eye") is not None
                    else {}
                ),
            }
        )

    # Third-person FREE CAM (debug/visualization). author_render_layer already
    # authored the FreeCam RenderProduct (/RenderOVRTX/FreeCam) + camera prim
    # (/RenderOVRTX/Cameras/FreeCam); we add a manifest entry so the render node
    # renders + publishes it, and set free_cam_prim_path so it can be moved live
    # via the node's ~/free_cam_pose topic. is_free_cam=True -> the node skips
    # /tf anchoring. NOT an observation camera (num_cameras = len(observation.
    # cameras), computed separately), so this does not change the policy's obs.
    free_cam_prim_path = ""
    if author_render:
        fcw, fch = 1280, 720  # matches author_task_render_layer free_cam_hw
        fcf = fcw * 18.0 / 36.0  # focalLength 18 / horizontalAperture 36 -> 640
        cam_entries.append(
            {
                "name": "freecam",
                "render_product_path": "/RenderOVRTX/FreeCam",
                "topic": "/rlinf/freecam/image_raw",
                "depth_topic": "",
                "path": "FreeCam",
                "frame_id": "free_cam",
                "link": "",
                "width": fcw,
                "height": fch,
                "fx": fcf,
                "fy": fcf,
                "cx": fcw / 2.0,
                "cy": fch / 2.0,
                "model": "",
                "is_free_cam": True,
            }
        )
        free_cam_prim_path = "/RenderOVRTX/Cameras/FreeCam"

    # Rolling's engine reads this manifest by PATH via ``Manifest.read`` ->
    # ``from_dict``, which raises SchemaVersionError unless top-level ``version``
    # equals the core's MANIFEST_VERSION (4). Source it from the schema when
    # importable (so it stays pinned in lockstep with the engine), else fall
    # back to the known-current 4. Lazy import keeps task_loader free of a
    # hard genie_sim_engine_core dependency at module-import time.
    try:
        from genie_sim_engine_core.schema.manifest import (  # noqa: PLC0415
            MANIFEST_VERSION as _MANIFEST_VERSION,
        )
    except Exception:
        _MANIFEST_VERSION = 4

    manifest = {
        # Schema version the rolling engine validates on read (see above).
        # from_dict ignores unknown top-level keys, so the rlinf extras below
        # (foreground_usda, robot_init_pose, init_joint_pos, _benchmark_task,
        # camera native_*/link) survive validation AND survive on disk.
        "version": _MANIFEST_VERSION,
        "base_path": str(Path(out_path).resolve().parent),
        "scene_usda": scene_usda,
        # RAW foreground (table + manipulable objects, no floor/lights) — the
        # light-less benchmark scene BEFORE _wrap_scene_with_default_env. The
        # multi-env assembler replicates THIS per env under /World/envs/env_<i>
        # (the floor+lights in scene_usda's wrapper stay shared/global), so each
        # env gets its own table+objects. Equals scene_usda when render is off.
        "foreground_usda": r["scene_usda"],
        # The stage the render node OPENS — global env only, no foreground.
        # Empty string = the node falls back to opening ``scene_usda`` and
        # deactivating the strays instead.
        #
        # Deliberately an OPTIONAL field at the SAME schema version: the core
        # manifest schema's own carve-out says adding an optional field the
        # C++ mirror does not read must NOT bump ``MANIFEST_VERSION``, because
        # bumping is what would actually break the engine and the C++ renderer
        # (precedent there: ``background_usda``).
        "env_shell_usda": env_shell_usda,
        # Optional BACKGROUND room (shelf/counter/door). Empty unless
        # ``task_background.resolve_use_background`` said this task needs it —
        # then the render node references it per env so the real workbench
        # supports the (authored-in-place) objects.
        "background_usda": (r.get("background_usda", "") if use_background else ""),
        # Static props the engine should publish world AABBs for (eval
        # checkers only — see the field's note in resolve_benchmark_task
        # above). Tied to the background: with the room absent there are no
        # props to measure, so the list goes out empty too.
        "static_prop_aabbs": (r.get("static_prop_aabbs", []) if use_background else []),
        # Per-env background room override (see the field docs on
        # ``_isaac_render_common.Manifest.background_usda_per_env``). This key only appears
        # when ``randomization.render.background.enable`` is true --
        # ``background_usda_per_env is None`` is the sole criterion for "left at the default
        # off" (see the comment on that handling block above); when it is absent, old
        # manifests and new manifests with this dimension off are indistinguishable, which
        # is the point: a manifest reader sees "this field has a default empty list, this
        # manifest doesn't carry one" rather than a new empty list that means something else.
        **({"background_usda_per_env": background_usda_per_env} if background_usda_per_env is not None else {}),
        "robot_usda": r["robot_usda"],
        "robot_prefix": r["robot_prefix"],
        "robot_from_urdf": r["robot_from_urdf"],
        # Robot BASE world pose (benchmark MOVES the robot here; scene stays at
        # origin). Applied by the engine via scene_yaml.robot.init_base_pose below.
        "robot_init_pose": rip,
        # scene-cfg the engine loads (session.py reads manifest["scene_yaml"]);
        # carries robot.init_base_pose so _apply_init_base_pose places the robot.
        "scene_yaml": scene_cfg_path,
        # Initial JOINT pose benchmark eval applies (PiEnv.reset), reusing
        # benchmark's ROBOT_CONFIGS + TASK_INFO_DICT/TaskInfo. {joint: value}.
        # The engine applies these values when initializing robot joints.
        "init_joint_pos": resolve_robot_init_joints(sub_task_name, r["robot_cfg_name"]),
        # The optional render layer contains camera and RenderProduct definitions.
        "render_layer_usda": render_layer_usda,
        "cameras": cam_entries,
        # third-person FreeCam prim the render node moves via ~/free_cam_pose
        "free_cam_prim_path": free_cam_prim_path,
        # provenance (handy for debugging which task this manifest came from)
        "_benchmark_task": {
            "task_name": task_name,
            "sub_task_name": sub_task_name,
            "instance_id": int(instance_id),
        },
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))

    # Retain the resolved physics/sensor spec and build-time plans for readback.
    # The render specification can be normalized from the accepted init configuration.
    _record_build_randomization(
        resolved={"physics": dict(spec.physics), "sensor": dict(spec.sensor)},
        build_time_plans={
            **({"joint_pd": _jpd_build_plans} if joint_pd_spec.get("enable", False) else {}),
            **({"background": _bg_build_plans} if background_spec.get("enable", False) else {}),
        },
    )
    return str(out)


def _superset_free_body_stats(
    sub_task_name: str,
    instance_ids: list,
    *,
    bench_root: Optional[str] = None,
) -> tuple:
    """Return ``(free_bodies, total_slots)`` for the deduplicated superset.

    Slot counts come from ``resolve_superset_instances``, matching assembly.
    Kinematic slots do not receive free joints and are excluded from the size
    gate. Every occurrence of an asset must agree on its kinematic status;
    inconsistent authoring raises ``ValueError``.
    """
    bench = Path(bench_root) if bench_root else _benchmark_config_root()
    base = bench / "llm_task" / sub_task_name
    instance_ids = list(instance_ids)

    r = resolve_superset_instances(sub_task_name, instance_ids, bench_root=bench_root)
    total_slots = len(next(iter(r["instance_poses"].values())))

    blocks_cache: dict = {}

    def _blocks_for(iid):
        if iid not in blocks_cache:
            scene = base / str(iid) / "scene.usda"
            out: dict = {}
            for name, block, _type_kw in _extract_object_xform_blocks(str(scene)):
                arc = next((m for m in _asset_arc_re().finditer(block) if not m.group(1)), None)
                out[name] = (block, arc)
            blocks_cache[iid] = out
        return blocks_cache[iid]

    # Cross-instance kinematic-consistency guard (see docstring) + the
    # per-slot lookup below share this single scan.
    asset_kinematic: dict = {}  # asset -> (kinematic, iid, member) of the FIRST occurrence seen
    slot_owner: dict = {}  # slot -> (iid, member), first instance in order that claims it
    for iid in instance_ids:
        alias = r["name_alias"][str(iid)]
        for member, (block, arc) in _blocks_for(iid).items():
            if arc is None:
                continue
            asset = arc.group(2).rsplit("/", 1)[0]
            kin = "physics:kinematicEnabled = 1" in block
            if asset in asset_kinematic:
                prev_kin, prev_iid, prev_member = asset_kinematic[asset]
                if prev_kin != kin:
                    raise ValueError(
                        f"asset {asset!r} in {sub_task_name!r} is authored "
                        f"kinematic in one occurrence but not another -- "
                        f"instance {prev_iid}'s {prev_member!r} has "
                        f"physics:kinematicEnabled={prev_kin}, instance "
                        f"{iid}'s {member!r} has kinematicEnabled={kin}. "
                        f"Refusing to guess which is authoritative for the "
                        f"free-body scale gate."
                    )
            else:
                asset_kinematic[asset] = (kin, iid, member)
            slot = alias.get(member)
            if slot is not None:
                slot_owner.setdefault(slot, (iid, member))

    kinematic_slots = 0
    for slot, (iid, member) in slot_owner.items():
        block, _arc = _blocks_for(iid)[member]
        if "physics:kinematicEnabled = 1" in block:
            kinematic_slots += 1
    free_bodies = total_slots - kinematic_slots
    return free_bodies, total_slots


def build_superset_stage_manifest(
    task_name: str,
    sub_task_name: str,
    instance_ids: list,
    cameras: list[dict],
    out_path: str,
    *,
    max_free_bodies: int = 32,
    allow_oversize: bool = False,
    **kw,
) -> str:
    """Build a manifest spanning the selected benchmark instances.

    The foreground has one prim per shared asset slot, and ``instance_switch``
    holds each instance's active and parked poses. ``instance_ids[0]`` supplies
    the robot, cameras and background. Other keywords pass to
    ``build_stage_manifest``.

    Reject more than ``max_free_bodies`` non-kinematic slots unless
    ``allow_oversize`` is set, before scene assembly. The gate limits the dense
    MuJoCo-Warp solver's scaling cost; rejection still permits running an
    individual instance. Log the slot count and estimated step cost.
    """
    instance_ids = list(instance_ids)
    free_bodies, total_slots = _superset_free_body_stats(sub_task_name, instance_ids, bench_root=kw.get("bench_root"))
    nv = 46 + 6 * free_bodies
    est_ms = 0.2568 * nv + 3.379e-6 * nv**3
    oversize = free_bodies > max_free_bodies
    if oversize and not allow_oversize:
        raise ValueError(
            f"superset for {sub_task_name!r} would need {free_bodies} free "
            f"bodies ({total_slots} slots total, nv={nv}), over "
            f"max_free_bodies={max_free_bodies} -- est. {est_ms:.0f} "
            f"ms/physics-step, too slow to run. This task is still usable, "
            f"just not via cross-instance superset switching: fall back to "
            f"one sim process per instance (build_stage_manifest) and "
            f"restart the process to switch instances -- the current eval "
            f"pipeline already runs this way. If you understand the cost "
            f"and want the superset anyway, pass allow_oversize=True."
        )
    print(
        f"[task_loader] superset {sub_task_name!r} over "
        f"{len(instance_ids)} instances: {free_bodies} free bodies "
        f"({total_slots} slots total, nv={nv}, est. {est_ms:.0f} "
        f"ms/physics-step), max_free_bodies={max_free_bodies}"
        + (" -- OVER THRESHOLD, proceeding via allow_oversize=True" if oversize else ""),
        flush=True,
    )
    return build_stage_manifest(
        task_name, sub_task_name, int(instance_ids[0]), cameras, out_path, superset_instance_ids=instance_ids, **kw
    )


__all__ = [
    "resolve_benchmark_task",
    "build_stage_manifest",
    "build_superset_stage_manifest",
]
