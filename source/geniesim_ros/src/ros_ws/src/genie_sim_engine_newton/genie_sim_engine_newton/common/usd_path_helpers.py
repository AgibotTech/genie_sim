# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""USD asset-path utilities shared across the runtime-USD dump paths.

The two dump callsites (``engine/newton/setup/runtime.py:_dump_runtime_usd``
for the newton-standalone path, and ``kit/stage.py:_dump_robot_runtime_usd``
for the isaac-wrapper path) both produce ``robot_runtime.usda`` files
that embed ABSOLUTE asset paths into the dump when the source stage
is opened with absolute paths.  Result: ``robot_runtime.usda`` carries
lines like::

    subLayers = [@/geniesim_assets/scenes/scene_flat_g2_sp_fsvbd/newton_scene.usda@]
    prepend references = @/geniesim_assets/scenes/scene_flat_g2_sp_fsvbd/robot.usda@

That makes the dump non-portable: if the entire ``/geniesim_assets``
tree is moved or mounted at a different prefix, the references no
longer resolve.  We want the dump to look like::

    subLayers = [@./newton_scene.usda@]
    prepend references = @./robot.usda@

so the file's references are anchored to whatever directory it lives
in.  This module provides that rewrite as a single reusable function
so both dump paths share the same behaviour.

Strategy
--------

USD's ``Sdf.Layer.UpdateExternalReference(oldAsset, newAsset)`` updates
every place the layer references the old asset path — references,
payloads, sublayers, AND ``asset``-typed attribute values (the latter
covers texture paths inside material networks, where each
``inputs:*_texture`` carries an ``Sdf.AssetPath``).  We enumerate every
absolute asset path the layer carries via
``Sdf.Layer.GetExternalReferences()``, decide whether each should be
rewritten relative to the dump's parent directory, and call
``UpdateExternalReference`` per path.

We use ``./<rel>`` form (anchored-relative) consistently — USD treats
``@./foo@`` as relative to the layer's own location, which is the
behaviour we want.  Pure-relative ``@foo@`` (no leading ``./``) works
the same way in modern USD but the explicit anchor reads cleaner
and avoids any ambiguity with search-path resolution.

Out of scope
------------

* Asset paths whose target lives OUTSIDE the dump's parent tree don't
  always have a sensible relative form (you'd end up with
  ``../../../somewhere`` chains that obscure intent).  We only rewrite
  paths whose target is reachable via ``relpath`` without leaving the
  configured ``anchor_root`` (defaults to the parent dir of the dump,
  expandable via the ``anchor_root`` parameter when the operator wants
  to anchor higher up — e.g. at ``/geniesim_assets`` so cross-scene
  texture references go relative through ``../blank/...``).
* Asset paths that are already relative are left alone.
* Asset paths that don't resolve to an existing file are left alone
  (we treat that as "operator authored something exotic; don't touch").
"""

from __future__ import annotations

import os
from typing import Optional


def make_layer_asset_paths_relative(
    layer_path: str,
    anchor_root: Optional[str] = None,
    *,
    source_dir: Optional[str] = None,
    logger=None,
) -> int:
    """Open the layer at ``layer_path`` and rewrite every absolute
    asset path it carries to a path anchored to the dump's parent
    directory.

    Parameters
    ----------
    layer_path :
        Absolute path on disk to the USD layer to rewrite.  Layer is
        opened, modified, and saved in place.
    anchor_root :
        Optional ancestor directory.  Asset paths whose target lives
        under this root will be rewritten as ``./relpath`` from
        ``dirname(layer_path)``.  Asset paths whose target lives
        outside ``anchor_root`` are left as-is.  Defaults to
        ``dirname(layer_path)`` itself (only same-directory
        references get rewritten).

        Pass a higher root (e.g. ``/geniesim_assets``) to also
        rewrite cross-scene references like
        ``@/geniesim_assets/scenes/blank/texture/bc.png@`` →
        ``@../blank/texture/bc.png@``.
    source_dir :
        Optional directory of the layer's ORIGINAL location, used
        only to resolve bare / relative payload references when the
        layer was produced by ``Stage.GetRootLayer().Export(other_path)``
        (e.g. the engine's ``robot_runtime.usda`` dump copies its
        scene USD's content to a different dump directory).  In that
        case bare strings like ``@supermarket_shelf_002/Aligned.usda@``
        no longer resolve at the dump's location, but they DID resolve
        at the source's location.  We compose ``source_dir`` with the
        bare path to get the original absolute target, then
        re-relativize against ``dirname(layer_path)``.

        When ``None`` (default) we fall back to the heuristic
        ``_try_resolve_bare`` filesystem scan, which only works when
        the bare basename is unique under ``anchor_root``.  Pass an
        explicit ``source_dir`` whenever the caller knows where the
        layer originated — it sidesteps the ambiguity entirely.
    logger :
        Optional logger; falls back to ``print`` for the summary
        line.  Per-path rewrite details only go to the logger
        (debug-level conceptually, but we use ``info`` so the
        operator can verify).

    Returns
    -------
    Number of asset paths rewritten.

    Failures
    --------
    Best-effort.  If ``Sdf.Layer.FindOrOpen`` fails, returns 0
    without raising — the dump file already exists, the worst case
    is "carries absolute paths".  Per-path rewrite errors are caught
    individually so one bad reference doesn't drop the others.
    """

    def _info(msg: str) -> None:
        if logger is not None and hasattr(logger, "info"):
            logger.info(msg)
        else:
            print(msg, flush=True)

    def _warn(msg: str) -> None:
        if logger is not None and hasattr(logger, "warn"):
            logger.warn(msg)
        else:
            print(f"WARN: {msg}", flush=True)

    if not layer_path or not os.path.isfile(layer_path):
        return 0

    try:
        from pxr import Sdf  # noqa: PLC0415
    except ImportError:
        # pxr isn't on the path (e.g. running outside Kit / mjpython
        # context).  Nothing to do — the dump simply keeps absolute
        # paths.  This is a tooling-time helper, not load-bearing.
        _warn(f"[usd-paths] pxr import failed; leaving {layer_path} unchanged")
        return 0

    layer = Sdf.Layer.FindOrOpen(layer_path)
    if layer is None:
        _warn(f"[usd-paths] could not open {layer_path}; leaving as-is")
        return 0

    dump_dir = os.path.dirname(os.path.abspath(layer_path))
    if anchor_root is None:
        # Default to the dump's PARENT directory, not the dump dir
        # itself.  Two reasons:
        #
        #   1. Cross-scene references (e.g. ``/geniesim_assets/scenes/
        #      blank/T_Shirt_fold.usd`` referenced from a dump at
        #      ``/geniesim_assets/scenes/scene_flat_g2_sp_fsvbd/``) live
        #      one level up — ``../blank/T_Shirt_fold.usd`` is the
        #      natural relative form, and the user's "scenes/" tree
        #      moves as one unit.
        #   2. Same-dir references still work — ``commonpath`` against
        #      the parent dir matches anything underneath, including
        #      the dump_dir itself.
        #
        # If an operator wants to anchor even higher (e.g. cover
        # ``/geniesim_assets/textures`` from a deeply-nested dump),
        # they can still pass ``anchor_root`` explicitly.
        anchor_root = os.path.dirname(dump_dir)
    else:
        anchor_root = os.path.abspath(anchor_root)

    def _try_relativize(asset_path: str) -> Optional[str]:
        """Return a ``./rel/path`` form when the absolute asset
        target lives under ``anchor_root``, or ``None`` to leave
        the path alone."""
        if not asset_path:
            return None
        # Already relative — USD anchors it to the layer's location,
        # which is what we want.  Don't touch.
        if not os.path.isabs(asset_path):
            return None
        # Skip URL-style asset paths (omniverse://, http://, etc.) —
        # ``isabs`` returns True for some of those on Linux but
        # there's nothing meaningful to do with relpath.
        if "://" in asset_path:
            return None
        # Outside the anchor root → leave alone.  ``commonpath`` is
        # the correct check; ``startswith`` would false-positive on
        # paths that share a prefix string but not a directory
        # (``/foo/bar`` vs ``/foo/barbaz``).
        try:
            common = os.path.commonpath([asset_path, anchor_root])
        except ValueError:
            return None
        if common != anchor_root and not common.startswith(anchor_root + os.sep):
            return None
        try:
            rel = os.path.relpath(asset_path, dump_dir)
        except ValueError:
            return None
        # Normalize: ``./foo`` beats bare ``foo``; ``../foo`` we
        # leave as ``../foo`` (USD reads it relative to the layer).
        if not rel.startswith(".."):
            rel = "./" + rel
        return rel

    def _try_resolve_bare_via_source(asset_path: str) -> Optional[str]:
        """Resolve a bare/relative reference inherited from the source layer.

        Used when ``source_dir`` is provided (typically because the dump
        was produced by ``Stage.GetRootLayer().Export(other_path)`` and
        the bare references were authored relative to the source layer's
        original location).  Composes ``source_dir`` with ``asset_path``
        to recover the original absolute target, then relativizes that
        against ``dump_dir``.

        Returns the rewritten ``./rel/path`` (or ``../rel/path``) form,
        or ``None`` if the composed target doesn't exist on disk (we'd
        rather leave a broken bare path than emit a confidently-wrong
        rewrite that masks the real problem).
        """
        if not source_dir:
            return None
        if not asset_path or os.path.isabs(asset_path):
            return None
        if "://" in asset_path:
            return None
        original_abs = os.path.normpath(os.path.join(source_dir, asset_path))
        if not os.path.isfile(original_abs):
            return None
        try:
            rel = os.path.relpath(original_abs, dump_dir)
        except ValueError:
            return None
        if not rel.startswith(".."):
            rel = "./" + rel
        return rel

    def _try_resolve_bare(asset_path: str) -> Optional[str]:
        """Handle bare/relative paths whose target doesn't exist in
        ``dump_dir`` but exists somewhere under ``anchor_root``.

        Typical case: ``@gridroom_black.usd@`` authored in
        ``/scenes/blank/blank.usda`` (where it resolves to a sibling)
        gets Exported into a dump at ``/scenes/scene_X/`` (where the
        same bare path would resolve to a non-existent
        ``/scenes/scene_X/gridroom_black.usd``).  We scan the
        anchor_root subtree for a file with the same basename and,
        if exactly one match is found, rewrite the bare path to the
        anchored-relative path pointing at that match.

        Returns the rewritten path, or ``None`` to leave the
        original bare path in place.  We're CONSERVATIVE: bare paths
        that already resolve correctly (file exists in dump_dir) are
        left alone, and ambiguous matches (multiple files with the
        same basename under anchor_root) are left alone with a warn.
        """
        if not asset_path or os.path.isabs(asset_path):
            return None
        if "://" in asset_path:
            return None
        # Does the bare path already resolve? (USD would anchor it
        # at dump_dir.)  If yes, leave alone.
        candidate_at_layer = os.path.normpath(os.path.join(dump_dir, asset_path))
        if os.path.isfile(candidate_at_layer):
            return None
        # Doesn't resolve at the layer's location.  Search the
        # anchor_root subtree for a file with the same basename.
        basename = os.path.basename(asset_path)
        matches = []
        for root_dir, _dirs, files in os.walk(anchor_root):
            if basename in files:
                matches.append(os.path.join(root_dir, basename))
                # Cap the scan so a pathological subtree doesn't
                # stall startup — two matches is enough to know we're
                # ambiguous.
                if len(matches) > 1:
                    break
        if not matches:
            return None
        if len(matches) > 1:
            _warn(
                f"[usd-paths] bare reference {asset_path!r} in {layer_path} "
                f"could match multiple files under {anchor_root}: "
                f"{matches[:2]}...  Leaving unresolved — author an "
                f"absolute path or move the file."
            )
            return None
        # Single match — rewrite as anchored relative from dump_dir.
        rel = os.path.relpath(matches[0], dump_dir)
        if not rel.startswith(".."):
            rel = "./" + rel
        return rel

    # Sdf.Layer.GetExternalReferences() returns every distinct
    # asset path the layer mentions — sublayers, references,
    # payloads, AND asset-typed attribute values.  Iterating once
    # is enough; ``UpdateExternalReference`` propagates the rename
    # to every site that uses the old path.
    all_paths = list(layer.GetExternalReferences())
    abs_paths = [p for p in all_paths if os.path.isabs(p)]
    bare_paths = [p for p in all_paths if p and not os.path.isabs(p) and "://" not in p]
    n_rewritten = 0
    for old_path in abs_paths:
        new_path = _try_relativize(old_path)
        if new_path is None or new_path == old_path:
            continue
        try:
            layer.UpdateExternalReference(old_path, new_path)
            n_rewritten += 1
            _info(f"[usd-paths] {layer_path}: {old_path!r} → {new_path!r}")
        except Exception as exc:  # noqa: BLE001 — best-effort
            _warn(f"[usd-paths] {layer_path}: failed to rewrite {old_path!r}: " f"{exc!r}; leaving as absolute")
    # Bare/relative path resolution — separate pass because the
    # ``_try_resolve_bare`` heuristic walks the anchor_root subtree
    # and only fires for paths that don't already resolve, so it
    # can't be folded into ``_try_relativize`` cleanly.  We prefer the
    # source-dir lookup when ``source_dir`` was passed (deterministic,
    # no ambiguity); otherwise fall back to the basename scan.
    for old_path in bare_paths:
        new_path = _try_resolve_bare_via_source(old_path)
        if new_path is None:
            new_path = _try_resolve_bare(old_path)
        if new_path is None or new_path == old_path:
            continue
        try:
            layer.UpdateExternalReference(old_path, new_path)
            n_rewritten += 1
            _info(f"[usd-paths] {layer_path}: bare {old_path!r} → {new_path!r}")
        except Exception as exc:  # noqa: BLE001
            _warn(f"[usd-paths] {layer_path}: failed to rewrite bare {old_path!r}: {exc!r}")

    if n_rewritten > 0:
        layer.Save()
        _info(
            f"[usd-paths] {layer_path}: rewrote {n_rewritten} asset "
            f"path(s) → relative (dump_dir={dump_dir}, anchor_root={anchor_root})"
        )
    return n_rewritten
