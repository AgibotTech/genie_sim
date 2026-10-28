# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Selective hand contacts, shared by the single- and multi-world MJ-Warp adapters.

Newton 1.4/1.6.1 import disabled articulation self-collision as both geom
bitmasks and body excludes. Match Newton Next's hand policy before MJ-Warp
freezes its broadphase pairs: open hand contacts, retain non-hand exclusions
and the hand's parent/grandparent mesh-overlap guards.
"""

from contextlib import contextmanager
from functools import wraps
from pathlib import Path
import re
from threading import RLock
import xml.etree.ElementTree as ET

_HAND_BODY_RE = re.compile(r"(?:^|/|_)hand_[lr]_")
_PUT_MODEL_LOCK = RLock()


def _keep_exclude(first, second, hands, parents):
    if not (hands[first] or hands[second]):
        return True
    return (
        parents[first] == second
        or parents[second] == first
        or parents[parents[first]] == second
        or parents[parents[second]] == first
    )


def enable_hand_self_collision(mj_model, logger=None):
    """Mutate a compiled CPU model before ``mujoco_warp.put_model`` copies it.

    Return (hand geoms, cleared excludes, kept excludes). Visual-only geoms
    keep their zero masks; original contype bits and all non-hand masks stay
    intact. Clearing signatures preserves the compiled array's capacity.
    """
    import mujoco

    hands = [
        bool(_HAND_BODY_RE.search(mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i) or ""))
        for i in range(mj_model.nbody)
    ]
    if not any(hands):
        return (0, 0, 0)
    parents = mj_model.body_parentid
    geoms = cleared = kept = 0
    for i in range(mj_model.ngeom):
        if hands[mj_model.geom_bodyid[i]] and (mj_model.geom_contype[i] or mj_model.geom_conaffinity[i]):
            mj_model.geom_conaffinity[i] = -1
            geoms += 1
    for i, signature in enumerate(mj_model.exclude_signature):
        if not signature:
            continue
        first, second = int(signature) >> 16, int(signature) & 0xFFFF
        if _keep_exclude(first, second, hands, parents):
            kept += 1
        else:
            mj_model.exclude_signature[i] = 0
            cleared += 1
    if cleared:
        mj_model.exclude_signature.sort()
    if logger is not None:
        logger.info(
            f"[mjwarp-adapter] hand self-collision: {geoms} contact geom(s), "
            f"{cleared} hand exclude(s) cleared, {kept} non-hand/near-neighbour exclude(s) retained"
        )
    return (geoms, cleared, kept)


def enable_hand_self_collision_in_mjcf(path, logger=None):
    """Apply the same policy to SolverMuJoCo's dump, which comes from MjSpec.

    The solver exports its original spec after put_model, so mutations of the
    compiled model alone do not reach the file. Post-processing retains mesh
    paths, drive values, and every unrelated XML element.
    """
    tree = ET.parse(path)
    root = tree.getroot()
    bodies, parents = [], []

    def collect(body, parent):
        index = len(bodies)
        bodies.append(body)
        parents.append(parent)
        for child in body.findall("body"):
            collect(child, index)

    collect(root.find("worldbody"), 0)
    names = {body.get("name", ""): i for i, body in enumerate(bodies)}
    hands = [bool(_HAND_BODY_RE.search(body.get("name", ""))) for body in bodies]
    if not any(hands):
        return (0, 0, 0)
    geoms = cleared = kept = 0
    for body, is_hand in zip(bodies, hands):
        if not is_hand:
            continue
        for geom in body.findall("geom"):
            if int(geom.get("contype", "1")) or int(geom.get("conaffinity", "1")):
                geom.set("conaffinity", "-1")
                geoms += 1
    contact = root.find("contact")
    if contact is not None:
        for exclude in contact.findall("exclude"):
            first, second = names[exclude.get("body1")], names[exclude.get("body2")]
            if _keep_exclude(first, second, hands, parents):
                kept += 1
            else:
                contact.remove(exclude)
                cleared += 1
    tree.write(path, encoding="unicode")
    return (geoms, cleared, kept)


@contextmanager
def hand_self_collision_at_put_model(logger, mjcf_path=""):
    """Scope the Newton Next pre-put_model policy to solver construction.

    Both supported Newton versions call the module function dynamically.
    Serialize our constructors and restore it on success or failure; no
    permanent monkey patch or per-step work is introduced.
    """
    import mujoco_warp

    with _PUT_MODEL_LOCK:
        original = mujoco_warp.put_model
        hand_geoms = 0

        @wraps(original)
        def put_model(model, *args, **kwargs):
            nonlocal hand_geoms
            hand_geoms, _, _ = enable_hand_self_collision(model, logger)
            return original(model, *args, **kwargs)

        mujoco_warp.put_model = put_model
        try:
            yield
        finally:
            mujoco_warp.put_model = original
        if hand_geoms and mjcf_path and Path(mjcf_path).is_file():
            enable_hand_self_collision_in_mjcf(mjcf_path)
