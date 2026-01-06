# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from __future__ import annotations
from typing import NamedTuple, Any, Callable, Literal, Generator, Tuple, List

import os, math
import random
import numpy as np
from scipy.spatial.transform import Rotation as R

from geniesim.generator.scene_language.shape_utils import *
from geniesim.generator.scene_language.math_utils import *
from geniesim.generator.scene_language.type_utils import *
from geniesim.generator.scene_language.engine_utils import *
from geniesim.generator.scene_language.dsl_utils import *
from geniesim.generator.scene_language.flow_utils import *
from geniesim.generator.scene_language.calc_utils import *
from geniesim.generator.scene_language.assert_utils import *

from geniesim.assets import ASSETS_INDEX  # If this fails please check assets folder at source/geniesim/assets

SEED = int.from_bytes(os.urandom(4), "big")  # draw a fresh seed
print("Random seed:", SEED)  # print it for tracing
np.random.seed(SEED)
random.seed(SEED)

from utils import *
import uuid
import networkx as nx
from networkx.readwrite import json_graph
from itertools import groupby


def quaternion_to_rotation_matrix(q):
    x, y, z, w = q
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0:
        return np.eye(3)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return R


def make_pose_matrix(rot: np.ndarray, trans: list[float]) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = trans
    return T


def quaternion_to_angle_direction(xyzw_quaternion):
    """
    Convert xyzw quaternion [x, y, z, w] to axis-angle representation.

    Returns:
        tuple: (angle_in_radians, axis_vector)
    """
    x, y, z, w = xyzw_quaternion

    # Normalize the quaternion to ensure it's a unit quaternion
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0:
        return 0.0, np.array([1.0, 0.0, 0.0])

    x, y, z, w = x / norm, y / norm, z / norm, w / norm

    # Calculate the angle
    angle = 2 * math.acos(w)

    # Calculate the axis (handle the case where w is very close to 1)
    s = math.sqrt(1 - w * w)
    if s < 1e-8:  # threshold to avoid division by zero
        # If w is close to 1, the angle is very small, use default axis
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis = np.array([x / s, y / s, z / s])

    return angle, axis


def matrix_to_xyz_xyzw(matrix):
    """Convert a 4×4 homogeneous matrix to [x, y, z, qx, qy, qz, qw]."""
    matrix = np.asarray(matrix, dtype=np.float64)
    xyz = matrix[:3, 3]
    quat = R.from_matrix(matrix[:3, :3]).as_quat()  # [qx, qy, qz, qw]
    return np.hstack((xyz, quat))


@register("build asset from ASSETS_INDEX metadata")
def usd(oid: str, keywords: List[str] = []) -> Shape:
    parts = ASSETS_INDEX[oid]["shapes"]
    shapes = []
    for sub in parts:
        primitive_type = sub["type"]
        shape_kwargs = {}

        if primitive_type == "cube":
            shape_kwargs = {"scale": tuple(sub.get("size", (0.05, 0.05, 0.05)))}
        elif primitive_type == "sphere":
            r = sub.get("size", (0.05, 0.05, 0.05))[0] / 2
            shape_kwargs = {"radius": r}
        elif primitive_type == "cylinder":
            size = sub.get("size", (0.05, 0.05, 0.1))
            shape_kwargs = {
                "radius": size[0] / 2,
                "p0": (0, 0, -size[2] / 2),
                "p1": (0, 0, size[2] / 2),
            }
        else:
            continue

        s = primitive_call(
            primitive_type,
            shape_kwargs=shape_kwargs,
            info={"id": oid, "name": sub["name"], "keywords": keywords},
            color=(np.random.rand(), np.random.rand(), np.random.rand()),
        )

        pos = sub.get("position", (0, 0, 0))
        quat = sub.get("quaternion", (0, 0, 0, 1))
        s = transform_shape(s, translation_matrix(pos))

        angle, axis = quaternion_to_angle_direction(quat)
        center = compute_shape_center(s)
        s = transform_shape(s, rotation_matrix(angle, axis, center))

        shapes.append(s)

    shapes = concat_shapes(*shapes)
    shapes = transform_shape(shapes, translation_matrix((0, 0, get_object_info(shapes)["z_offset"])))

    # print("oid", oid)
    # pprint(shapes)

    return shapes  # concat_shapes(*shapes)


def get_subpart_info(object_id: str, subpart_id: str):
    # print(object_id, subpart_id)
    # pprint(ASSETS_INDEX[object_id]["shapes"])
    if subpart_id not in {v["name"] for v in ASSETS_INDEX[object_id]["shapes"]}:
        subpart_id = "bbox"
    for sub in ASSETS_INDEX[object_id]["shapes"]:
        if subpart_id.lower() in sub["name"].lower():
            center = np.array(sub["position"], float)
            size = np.array(sub["size"], float)
            size = np.array([size[0], size[1], size[2]])
            quat = sub.get("quaternion", (0, 0, 0, 1))
            half = size / 2.0
            xyz_min = center - half
            xyz_max = center + half
            return {
                "center": center,
                "quat": quat,
                "size": size,
                "xyz_max": xyz_max,
                "xyz_min": xyz_min,
            }
    raise ValueError(f"Part '{subpart_id}' not found in {object_id}")


def get_object_info(shape: Shape):
    info = {
        "size": compute_shape_sizes(shape),
        "max": compute_shape_max(shape),
        "min": compute_shape_min(shape),
        "center": [0, 0, 0],
        "z_offset": 0.0,
    }
    objects = [s["info"]["info"]["id"] for s in shape if "origin" == s["info"]["info"]["name"]]
    num_object = len(objects)
    info["num_object"] = num_object
    info["objects"] = [s["info"]["info"]["id"] for s in shape if "origin" == s["info"]["info"]["name"]]
    if 1 == num_object:
        for s in shape:
            if s["info"]["info"]["name"] == "origin":
                info["center"] = matrix_to_xyz_xyzw(s["to_world"])[:3]
                info["z_offset"] = (info["center"] - info["min"])[2]
    elif 2 <= num_object:
        pass
        # info["size"] = [0, 0, 0]
        # info["max"] = [0, 0, 0]
        # info["min"] = [0, 0, 0]
    else:
        info["center"] = (info["max"] - info["min"]) / 2.0
        info["z_offset"] = info["size"][2] / 2.0
    # pprint(info)
    return info


def gen_scene_layout_info(scene_data) -> dict:
    G = nx.DiGraph()
    G.graph["rankdir"] = "LR"
    layout_info = {
        "scene_id": "deepseek_gen",
        "seed": SEED,
        "layout": {},
        "relations": {},
    }

    for i, obj in enumerate(scene_data):
        # name
        _id = obj["info"]["info"]["id"]
        _id = "".join([char for char in _id if not char.isdigit()])
        if "_" == _id[-1]:
            _id = _id[:-1]

        # uuid
        _uuid = [str(o[1]) for o in obj["info"]["stack"]]  #
        _uuid_new = uuid.uuid5(uuid.NAMESPACE_DNS, "".join(str(u) for u in _uuid))

        # id
        _id += f"_{str(_uuid_new).split('-')[0]}"
        _paths = [f"{o[0]}_{str(o[1]).split('-')[0]}" for o in obj["info"]["stack"]]
        _paths.reverse()
        _paths = [k for k, _ in groupby(_paths)]
        _tags = _paths[:-1] + [_id]

        if _id not in layout_info["layout"]:
            layout_info["layout"][_id] = {}

        if "origin" == obj["info"]["info"]["name"]:
            pose = matrix_to_xyz_xyzw(obj["to_world"])
            layout_info["layout"][_id] = {
                "usd": obj["info"]["info"]["id"],
                "xyz": [float(round(xyz, 3)) for xyz in pose[:3]],
                "xyzw": [float(round(xyzw, 6)) for xyzw in pose[3:]],
                "tags": _tags,
                "keywords": obj["info"]["info"]["keywords"],
                "description": ASSETS_INDEX[obj["info"]["info"]["id"]]["description"],
                # "path": _paths,
            }

            for i in range(len(_tags) - 1):
                G.add_edge(_tags[i], _tags[i + 1])
            G.nodes[_tags[-1]]["tags"] = obj["info"]["info"]["keywords"]

    def scene_graph_find_root_node(G):
        root_node = ""
        try:
            root_node = next(nx.topological_sort(G))

        except nx.NetworkXUnfeasible:
            print("The graph has a cycle and cannot be topologically sorted.")
        return root_node

    def compress_degree2_paths(G):
        """
        Return a new DiGraph in which every maximal 1-in/1-out chain is
        replaced by a single edge (head -> tail).  The original node names of
        the head and tail are preserved.
        """
        G = G.copy()
        removed = set()

        for v in list(G.nodes()):
            if v in removed:
                continue

            # build the forward chain
            chain = [v]
            cur = v
            while G.out_degree(cur) == 1 and G.in_degree(cur) == 1 and cur not in removed:
                nxt = next(G.successors(cur))
                if nxt == v:  # simple cycle: stop
                    break
                chain.append(nxt)
                cur = nxt

            # build the backward chain
            cur = v
            while G.in_degree(cur) == 1 and G.out_degree(cur) == 1 and cur not in removed:
                prv = next(G.predecessors(cur))
                if prv == cur:  # simple cycle: stop
                    break
                chain.insert(0, prv)
                cur = prv

            # now chain = [..., tail, a, b, ..., head]
            head, tail = chain[-1], chain[0]
            if head == tail:  # isolated cycle; ignore for this task
                continue

            # remove internal nodes
            internals = chain[1:-1]
            if internals:  # at least one node to collapse
                G.remove_nodes_from(internals)
                removed.update(internals)
                G.add_edge(tail, head)  # keep original head & tail names

        return G

    G_simple = G  # compress_degree2_paths(G)
    layout_info["relations"]["graph"] = json_graph.node_link_data(G_simple)
    root_node = scene_graph_find_root_node(G)
    for succ in G.successors(root_node):
        layout_info["scene_id"] = "_".join(str(succ).split("_")[:-1])

    # tree_dict = to_rich_tree_dict(layout_info["layout"])
    # tree = get_rich_tree(tree_dict)
    # layout_info["relations"]["tree"] = tree_dict

    # rprint(tree)

    # exit()
    return layout_info, G_simple
