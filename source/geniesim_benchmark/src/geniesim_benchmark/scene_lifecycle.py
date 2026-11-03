"""Synchronous asset-script retirement before a USD subtree is replaced."""

import sys

from pxr import Sdf, Usd

from geniesim_assets.behavior_lifecycle import retire_scope


def retire_asset_scripts(stage: Usd.Stage, root_path: str | Sdf.Path) -> None:
    """Release callbacks and forget the old SDK instances before same-path reuse.

    Kit scripting 107 and behavior scripting 110 defer USD resync processing.
    Their resync handlers retain instances if the path already exists again.
    Neither exposes a synchronous public unload operation. Keep the verified
    manager contract in this adapter, and reject an incompatible manager before
    the caller deletes any USD objects. Do not reload the extension or create
    script instances ourselves; the SDK loads the newly composed scripts.
    """
    if stage is None:
        return
    usd_module = sys.modules.get("omni.usd")
    if usd_module is not None and usd_module.get_context().get_stage() != stage:
        raise RuntimeError("Cannot retire scripts belonging to an inactive USD stage")
    root = Sdf.Path(root_path)
    managers = []
    seen = set()
    for name in ("omni.kit.scripting", "omni.behavior.scripting.core"):
        module = sys.modules.get(name)
        if module is None:
            continue
        manager_type = getattr(module, "ScriptManager", None)
        if manager_type is None:
            raise RuntimeError(f"Unsupported script manager interface: {name}")
        manager = manager_type.get_instance()
        if manager is None or id(manager) in seen:
            continue
        scripts = getattr(manager, "_prim_to_scripts", None)
        destroy = getattr(manager, "_destroy_script", None)
        if not isinstance(scripts, dict) or not callable(destroy):
            raise RuntimeError(f"Unsupported synchronous script retirement: {name}")
        managers.append((manager, scripts, destroy))
        seen.add(id(manager))

    retire_scope(stage, root_path)
    # ScriptNode state has its own timeline/async subscriptions. Its SDK
    # cleanup must run while the database is still valid, as in WheelParking.
    prim = stage.GetPrimAtPath(root)
    if prim:
        script_nodes = [item for item in Usd.PrimRange(prim)
                        if item.GetTypeName() == "OmniGraphNode"
                        and item.GetAttribute("node:type").Get() == "omni.graph.scriptnode.ScriptNode"]
        if script_nodes:
            import omni.graph.core as og
            from omni.graph.scriptnode.ogn.python.nodes.OgnScriptNode import OgnScriptNode

            for item in script_nodes:
                node = og.Controller.node(str(item.GetPath()))
                if node.is_valid():
                    OgnScriptNode.try_cleanup(node)
    for _manager, scripts, destroy in managers:
        pending = [(path, script) for path, entries in list(scripts.items())
                   if Sdf.Path(path).HasPrefix(root) for script in list(entries)]
        for path, script in pending:
            destroy(path, script)
        if any(Sdf.Path(path).HasPrefix(root) for path in scripts):
            raise RuntimeError(f"Script instances remain below {root}")
