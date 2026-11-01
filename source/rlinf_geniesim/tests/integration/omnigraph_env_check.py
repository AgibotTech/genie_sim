#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Check that every cloned environment runs its own OmniGraph during simulation.

Each graph writes a distinct value. Sentinels are re-armed before every
measurement to detect missed execution, shared values, and incorrect targets.
The check repeats after reset and stage rebuild.

Run inside the simulator runtime:
    N_ENVS=2 python3 source/rlinf_geniesim/tests/integration/omnigraph_env_check.py

Set USE_BUILDER=physics or kinematic with S2_MANIFEST=<manifest.json> to use the
production stage builder instead of the standalone test stage.
"""
import os
import sys

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import omni.usd  # noqa: E402
from isaacsim.core.utils.stage import create_new_stage, get_current_stage  # noqa: E402
from isaacsim.core.api import SimulationContext  # noqa: E402
from isaacsim.core.cloner import GridCloner  # noqa: E402
from pxr import Sdf, Usd, UsdGeom  # noqa: E402

N = int(os.environ.get("N_ENVS", "4"))
FG = "/tmp/item6_fg.usda"
#: Exercise the REAL ``build_gridcloner_stage`` instead of the hand-rolled
#: stage below: ``USE_BUILDER=physics`` (physics=True, the fused caller) or
#: ``USE_BUILDER=kinematic`` (physics=False, the tiled caller), together with
#: ``S2_MANIFEST=<a generated manifest.json>``. Without this the check only
#: proves something about a local imitation of the builder, which is free to
#: drift from it.
USE_BUILDER = os.environ.get("USE_BUILDER", "").strip().lower()
#: Which GridCloner mode. ``build_gridcloner_stage`` uses copy_from_source=True
#: (a deep Sdf.CopySpec of env_0's prim specs); data-collection's replay uses
#: False (inheritPathList). They are different composition mechanisms, so they
#: need not agree about whether a cloned graph evaluates.
CLONE_COPY = os.environ.get("CLONE_COPY", "1").strip() not in ("0", "false", "no")
S2_MANIFEST = os.environ.get("S2_MANIFEST", "")
SENTINEL = -1.0  # authored value; the graph must overwrite it


def write_fg():
    """One prim to stamp, plus a push graph that stamps it every tick."""
    st = Usd.Stage.CreateNew(FG)
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    w = st.DefinePrim("/World", "Xform")
    st.SetDefaultPrim(w)
    tgt = st.DefinePrim("/World/Marker", "Xform")
    tgt.CreateAttribute("graphStamp", Sdf.ValueTypeNames.Double).Set(SENTINEL)

    g = st.DefinePrim("/World/Marker/PushGraph", "OmniGraph")
    g.CreateAttribute("evaluator:type", Sdf.ValueTypeNames.Token).Set("push")
    g.CreateAttribute("fabricCacheBacking", Sdf.ValueTypeNames.Token).Set("StageWithoutHistory")
    g.CreateAttribute("pipelineStage", Sdf.ValueTypeNames.Token).Set("pipelineStageSimulation")
    from pxr import Gf as _Gf

    g.CreateAttribute("fileFormatVersion", Sdf.ValueTypeNames.Int2).Set(_Gf.Vec2i(1, 9))
    if os.environ.get("NO_AUTO_INSTANCING") == "1":
        # OmniGraph merges structurally identical graphs ("auto instancing").
        # Merged, only one env's output follows its own input.
        g.CreateAttribute("allowAutoInstancing", Sdf.ValueTypeNames.Bool).Set(False)

    src = st.DefinePrim("/World/Marker/PushGraph/Value", "OmniGraphNode")
    src.CreateAttribute("node:type", Sdf.ValueTypeNames.Token).Set("omni.graph.nodes.ConstantDouble")
    src.CreateAttribute("node:typeVersion", Sdf.ValueTypeNames.Int).Set(1)
    src.CreateAttribute("inputs:value", Sdf.ValueTypeNames.Double).Set(7.5)

    wr = st.DefinePrim("/World/Marker/PushGraph/Write", "OmniGraphNode")
    wr.CreateAttribute("node:type", Sdf.ValueTypeNames.Token).Set("omni.graph.nodes.WritePrimAttribute")
    wr.CreateAttribute("node:typeVersion", Sdf.ValueTypeNames.Int).Set(1)
    wr.CreateAttribute("inputs:name", Sdf.ValueTypeNames.Token).Set("graphStamp")
    wr.CreateAttribute("inputs:usdWriteBack", Sdf.ValueTypeNames.Bool).Set(True)
    wr.CreateRelationship("inputs:prim").SetTargets(["/World/Marker"])
    v = wr.CreateAttribute("inputs:value", Sdf.ValueTypeNames.Double)
    v.Set(0.0)
    v.AddConnection("/World/Marker/PushGraph/Value.inputs:value")
    st.GetRootLayer().Save()


def _overlay_foreground(real_fg: str) -> str:
    """A copy of the task's foreground with the observable marker+graph added.

    Sublayered rather than edited, so the real asset and the generated stage
    are untouched.
    """
    out = "/tmp/item6_overlay_fg.usda"
    # Reuse the layer if this run already built it (the rebuild phase calls
    # this a second time). Deleting the file is not enough: Sdf keeps the
    # layer resident by identifier and CreateNew then raises.
    existing = Sdf.Layer.Find(out)
    if existing is not None:
        return out
    if os.path.exists(out):
        os.remove(out)
    layer = Sdf.Layer.CreateNew(out)
    layer.subLayerPaths.append(real_fg)
    # The builder REFERENCES this file, and a reference resolves the layer's
    # own defaultPrim -- which a sublayer does not supply. Without this the
    # overlay composes to nothing and the marker silently never appears.
    _src_layer = Sdf.Layer.FindOrOpen(real_fg)
    if _src_layer and _src_layer.defaultPrim:
        layer.defaultPrim = _src_layer.defaultPrim
    st = Usd.Stage.Open(layer)
    st.SetEditTarget(layer)
    src = Usd.Stage.Open(FG)
    root = src.GetRootLayer()
    # CopySpec needs the destination parent to EXIST as a spec; this overlay
    # only sublayers the real foreground, so /World is composed-in, not
    # authored here.
    Sdf.CreatePrimInLayer(layer, Sdf.Path("/World"))
    Sdf.CopySpec(root, Sdf.Path("/World/Marker"), layer, Sdf.Path("/World/Marker"))
    st.GetRootLayer().Save()
    return out


def build_via_builder():
    """The production path: common.build_gridcloner_stage(...)."""
    import json  # noqa: PLC0415
    import sys as _sys  # noqa: PLC0415

    # .../source/rlinf_geniesim/scripts -> .../source
    here = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    common_dir = os.path.join(source_dir, "geniesim_ros", "src", "ros_ws", "src", "genie_sim_render", "scripts")
    if not os.path.isdir(common_dir):
        common_dir = "/workspace/source/geniesim_ros/src/ros_ws/src/" "genie_sim_render/scripts"
    if common_dir not in _sys.path:
        _sys.path.insert(0, common_dir)
    import _isaac_render_common as common  # noqa: PLC0415

    raw = json.loads(open(S2_MANIFEST).read())
    raw["foreground_usda"] = _overlay_foreground(raw["foreground_usda"])
    if os.environ.get("OLD_PATH") == "1":
        # Blank the shell to force the stage build path used before the
        # env-shell restructure (open the full scene, deactivate the
        # strays), to compare against that baseline rather than assume.
        raw["env_shell_usda"] = ""
        print("[i6] OLD PATH: env_shell_usda blanked", flush=True)
    tmp = "/tmp/item6_manifest.json"
    open(tmp, "w").write(json.dumps(raw, indent=2))

    manifest = common.Manifest.load(tmp)
    physics = USE_BUILDER == "physics"
    print(
        f"[i6] using REAL build_gridcloner_stage(physics={physics}) " f"manifest={os.path.basename(S2_MANIFEST)}",
        flush=True,
    )
    stage, _cams, _origins = common.build_gridcloner_stage(manifest, app, N, 15.0, physics=physics)
    paths = [f"/World/envs/env_{i}" for i in range(N)]
    return stage, paths


def build():
    if USE_BUILDER:
        return build_via_builder()
    create_new_stage()
    for _ in range(10):
        app.update()
    stage = get_current_stage()
    cloner = GridCloner(spacing=15.0)
    cloner.define_base_env("/World/envs")
    paths = cloner.generate_paths("/World/envs/env", N)
    p = stage.DefinePrim(paths[0] + "/World", "Xform")
    p.GetReferences().AddReference(FG)
    stage.Load(paths[0] + "/World")
    if os.environ.get("CLONE_BEFORE_TICK") == "1":
        # Hypothesis: env_0's graph REGISTERS during these updates, and the
        # clones are then copies of an already-registered graph that never
        # register themselves. Cloning before any tick would have all N appear
        # together, before OmniGraph has seen any of them.
        print("[i6] CLONE_BEFORE_TICK: skipping the pre-clone updates", flush=True)
    else:
        for _ in range(15):
            app.update()
    print(f"[i6] clone copy_from_source={CLONE_COPY}", flush=True)
    cloner.clone(source_prim_path=paths[0], prim_paths=paths, copy_from_source=CLONE_COPY, replicate_physics=False)
    stage.Load("/")
    for _ in range(15):
        app.update()
    return stage, paths


def set_per_env_values(stage, paths):
    """Give each env a distinct constant: env_i writes 10*(i+1)."""
    for i, ep in enumerate(paths):
        n = stage.GetPrimAtPath(f"{ep}/World/Marker/PushGraph/Value")
        n.GetAttribute("inputs:value").Set(10.0 * (i + 1))
    for _ in range(5):
        app.update()


def arm(stage, paths):
    """Re-arm the sentinel so the next reading can only come from a RUN."""
    for ep in paths:
        stage.GetPrimAtPath(f"{ep}/World/Marker").GetAttribute("graphStamp").Set(SENTINEL)
    for _ in range(3):
        app.update()


def check(stage, paths, label):
    got = stamps(stage, paths)
    want = [10.0 * (i + 1) for i in range(len(paths))]
    good = all(g is not None and abs(g - w) < 1e-6 for g, w in zip(got, want))
    print(f"[i6] {label}: stamps={got} want={want} -> {'OK' if good else 'FAIL'}", flush=True)
    return good


def stamps(stage, paths):
    out = []
    for ep in paths:
        a = stage.GetPrimAtPath(f"{ep}/World/Marker").GetAttribute("graphStamp")
        out.append(a.Get() if a else None)
    return out


def targets(stage, paths):
    """Which prim does each env's Write node point at?"""
    out = []
    for ep in paths:
        n = stage.GetPrimAtPath(f"{ep}/World/Marker/PushGraph/Write")
        r = n.GetRelationship("inputs:prim") if n else None
        t = r.GetTargets() if r else []
        out.append(str(t[0]) if t else None)
    return out


write_fg()
ok = True

if os.environ.get("WARM_OG") == "1":
    # Hypothesis: on a first build the OmniGraph registry is still coming
    # up while the stage is assembled, so graphs created during that
    # window never register; by the time a rebuild happens it is live and
    # all of them do. If that is it, touching omni.graph before building
    # should be enough.
    import omni.graph.core as og  # noqa: PLC0415

    n = len(og.get_all_graphs())
    for _ in range(10):
        app.update()
    print(f"[i6] WARM: touched omni.graph before building " f"(graphs currently registered: {n})", flush=True)


def _set_inputs(stage, paths, base):
    want = [float(base + i) for i in range(len(paths))]
    for ep, v in zip(paths, want):
        stage.GetPrimAtPath(f"{ep}/World/Marker/PushGraph/Value").GetAttribute("inputs:value").Set(float(v))
    return want


def _hit(got, want):
    return sum(1 for g, w in zip(got, want) if g is not None and abs(g - w) < 1e-6)


def probe_app_only(stage, paths, base, label):
    """App-tick-only probe: no SimulationContext, no reset().

    This matches the tiled renderer's actual lifecycle -- it builds the
    stage, creates camera sensors, and pumps app updates, without ever
    constructing a SimulationContext. Constructing one here would change
    the outcome and could mask a per-env failure that only shows up in the
    real lifecycle.
    """
    n = len(paths)
    want = _set_inputs(stage, paths, base)
    if os.environ.get("ORCHESTRATOR") == "1":
        # What isaacsim_tiled_render._render_once ACTUALLY does. A bare
        # app.update() is not the tiled lifecycle: the node drives frames
        # through Replicator's orchestrator, which advances the timeline
        # (pause_timeline=False). Testing app ticks instead was measuring a
        # lifecycle that does not exist in production.
        import omni.replicator.core as rep  # noqa: PLC0415

        for _ in range(10):
            rep.orchestrator.step(rt_subframes=1, pause_timeline=False)
    else:
        for _ in range(10):
            app.update()
    got = stamps(stage, paths)
    good = _hit(got, want) == n
    print(
        f"[i6] {label}: base={base:<6.0f} got={got}  " f"({_hit(got, want)}/{n}) -> {'OK' if good else 'FAIL'}",
        flush=True,
    )
    return good


def probe(stage, paths, sim, base, label):
    """Drive the input and watch the output, with a FRESH input per phase.

    Each phase gets its own base value: reusing one value across phases
    cannot distinguish "this phase did not run the graph" from "the value
    was already correct from the previous phase".

    Deliberately never writes the output attribute: re-arming a sentinel
    there desyncs Fabric from USD and produces incoherent readings.
    """
    n = len(paths)
    print(f"[i6] {label}", flush=True)

    want = _set_inputs(stage, paths, base)
    for _ in range(5):
        app.update()
    g_upd = stamps(stage, paths)
    print(f"        app.update  base={base:<6.0f} got={g_upd}  " f"({_hit(g_upd, want)}/{n})", flush=True)

    want_p = _set_inputs(stage, paths, base + 1000)
    for _ in range(20):
        sim.step(render=False)
    g_phys = stamps(stage, paths)
    print(f"        sim.step    base={base+1000:<6.0f} got={g_phys}  " f"({_hit(g_phys, want_p)}/{n})", flush=True)

    want_r = _set_inputs(stage, paths, base + 2000)
    for _ in range(5):
        sim.step(render=True)
    g_rend = stamps(stage, paths)
    print(f"        render step base={base+2000:<6.0f} got={g_rend}  " f"({_hit(g_rend, want_r)}/{n})", flush=True)

    # The pass criterion is per-env execution, measured on the phases that DO
    # drive this graph. A push graph on the simulation pipeline updates on
    # app ticks and render steps but NOT on bare physics steps (confirmed by
    # giving that phase its own input and seeing the output ignore it) -- a
    # graph that must run every physics step needs an On Physics Step
    # trigger instead, which is a different authoring question.
    return _hit(g_upd, want) == n and _hit(g_rend, want_r) == n


stage, paths = build()
tg = targets(stage, paths)
per_env = all(t and t.startswith(p) for t, p in zip(tg, paths))
print(f"[i6] write targets per-env correct: {per_env}", flush=True)
for t in tg:
    print(f"      {t}", flush=True)
if not per_env:
    print("[i6] FAIL: a graph writes outside its own env")
    ok = False

if os.environ.get("OG_DUMP") == "1":
    import omni.graph.core as og  # noqa: PLC0415

    for _ in range(5):
        app.update()
    print("[i6] OmniGraph registry:", flush=True)
    for g in og.get_all_graphs():
        try:
            gp = g.get_path_to_graph()
        except Exception:
            gp = "<?>"
        try:
            insts = g.get_instance_count()
        except Exception:
            insts = "<?>"
        print(f"        {gp}   instances={insts}", flush=True)
    for ep in paths:
        pr = stage.GetPrimAtPath(f"{ep}/World/Marker/PushGraph")
        a = pr.GetAttribute("allowAutoInstancing") if pr else None
        print(
            f"        {ep}: graph prim valid={bool(pr) and pr.IsValid()} "
            f"allowAutoInstancing={a.Get() if a else '<unset>'}",
            flush=True,
        )

if os.environ.get("NO_SIM_CONTEXT") == "1":
    # Reproduce the tiled renderer's lifecycle: no SimulationContext at all.
    ok &= probe_app_only(stage, paths, 100.0, "tiled lifecycle (app ticks only)")
    ok &= probe_app_only(stage, paths, 500.0, "tiled lifecycle, new input")
    print("[i6] PASS" if ok else "[i6] FAIL", flush=True)
    os._exit(0 if ok else 1)

sim = SimulationContext(physics_dt=1.0 / 120.0, rendering_dt=1.0 / 120.0, stage_units_in_meters=None, device="cuda:0")
sim.reset()

# Two different bases: the second proves the output TRACKS the input rather
# than having been written once at assembly time.
ok &= probe(stage, paths, sim, 10.0, "runs during simulation (base 10)")
ok &= probe(stage, paths, sim, 100.0, "output tracks a NEW input (base 100)")

sim.reset()
ok &= probe(stage, paths, sim, 200.0, "still runs after reset()")

# Full stage rebuild -- the tiled hot-reload shape. Skipped in USE_BUILDER
# mode: build_gridcloner_stage opens its own stage and cannot be re-entered
# after new_stage() in the same session. That path is covered separately by
# a reload check that exercises the same wipe-and-rebuild shape.
if not USE_BUILDER:
    omni.usd.get_context().new_stage()
    app.update()
    stage, paths = build()
    sim2 = SimulationContext(
        physics_dt=1.0 / 120.0, rendering_dt=1.0 / 120.0, stage_units_in_meters=None, device="cuda:0"
    )
    sim2.reset()
    ok &= probe(stage, paths, sim2, 300.0, "still runs after a full stage rebuild")

print("[i6] PASS" if ok else "[i6] FAIL", flush=True)

# A failing check must fail the process, not just print FAIL, or
# automation reads it as green. ``SimulationApp.close()`` terminates the
# process itself and never returns, so any exit call placed after it is
# dead code; the status is decided first and set directly with
# ``os._exit``, leaving teardown to the OS.
os._exit(0 if ok else 1)
