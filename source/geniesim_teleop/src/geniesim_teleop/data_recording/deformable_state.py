# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Decode Newton deformable-state messages into replay frames."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_TOPIC_PREFIX = "/genie_sim_engine/deformable"
CHEF_TOPIC_PREFIX = "/genie_sim_engine/chef_food"


def topic_names(prefix: str = DEFAULT_TOPIC_PREFIX) -> dict[str, str]:
    """Return the four topics that make up one deformable-state stream."""
    normalized = "/" + str(prefix).strip("/")
    if normalized == "/":
        raise ValueError("deformable topic prefix must not be empty")
    return {
        "info": f"{normalized}/info",
        "mapping": f"{normalized}/mapping",
        "state": f"{normalized}/state",
        "rigid_state": f"{normalized}/rigid_state",
    }


_CHEF_TOPICS = topic_names(CHEF_TOPIC_PREFIX)
INFO_TOPIC = _CHEF_TOPICS["info"]
MAPPING_TOPIC = _CHEF_TOPICS["mapping"]
STATE_TOPIC = _CHEF_TOPICS["state"]
RIGID_STATE_TOPIC = _CHEF_TOPICS["rigid_state"]

STATE_DTYPE = np.dtype(
    {
        "names": ("x", "y", "z", "semantic_type", "instance_id", "local_index", "global_index"),
        "formats": ("<f4", "<f4", "<f4", "u1", "<u2", "<u4", "<u4"),
        "offsets": (0, 4, 8, 12, 14, 16, 20),
        "itemsize": 24,
    }
)

MAPPING_DTYPE = np.dtype(
    {
        "names": (
            "binding_index",
            "vertex_index",
            "particle_0",
            "particle_1",
            "particle_2",
            "particle_3",
            "weight_0",
            "weight_1",
            "weight_2",
            "weight_3",
        ),
        "formats": ("<u2", "<u4", "<u4", "<u4", "<u4", "<u4", "<f4", "<f4", "<f4", "<f4"),
        "offsets": (0, 4, 8, 12, 16, 20, 24, 28, 32, 36),
        "itemsize": 40,
    }
)

RIGID_DTYPE = np.dtype(
    {
        "names": ("body_index", "x", "y", "z", "qw", "qx", "qy", "qz"),
        "formats": ("<u4", "<f4", "<f4", "<f4", "<f4", "<f4", "<f4", "<f4"),
        "offsets": (0, 4, 8, 12, 16, 20, 24, 28),
        "itemsize": 32,
    }
)


def stamp_to_nanoseconds(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _message_rows(message: Any, dtype: np.dtype, label: str) -> np.ndarray:
    if bool(message.is_bigendian):
        raise ValueError(f"{label} must use little-endian PointCloud2 data")
    if int(message.height) != 1:
        raise ValueError(f"{label} must be an unorganized PointCloud2")
    if int(message.point_step) != dtype.itemsize:
        raise ValueError(f"{label} point_step is {message.point_step}; expected {dtype.itemsize}")
    expected_bytes = int(message.width) * dtype.itemsize
    payload = bytes(message.data)
    if len(payload) != expected_bytes:
        raise ValueError(f"{label} has {len(payload)} bytes; expected {expected_bytes}")
    return np.frombuffer(payload, dtype=dtype, count=int(message.width))


def _pose_matrix(row: np.void) -> list[list[float]]:
    w, x, y, z = (float(row[name]) for name in ("qw", "qx", "qy", "qz"))
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if not np.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("deformable rigid snapshot contains an invalid quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    pose = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), float(row["x"])],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), float(row["y"])],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), float(row["z"])],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return pose.tolist()


class DeformableStateDecoder:
    """Validated static mapping plus per-frame reconstruction/writer."""

    def __init__(self, info: dict[str, Any], mapping_message: Any):
        if int(info.get("schema_version", -1)) != 2:
            raise ValueError(f"unsupported deformable-state schema {info.get('schema_version')!r}")
        if info.get("points_coordinate_space") not in {"mesh_local", "world"}:
            raise ValueError("unsupported deformable-state points coordinate space")

        mapping = _message_rows(mapping_message, MAPPING_DTYPE, "deformable mapping")
        if len(mapping) != int(info.get("mapping_point_count", -1)):
            raise ValueError("deformable mapping count differs from info metadata")

        self.info = info
        self.instances = sorted(info.get("instances") or [], key=lambda item: int(item["binding_index"]))
        self._bindings: list[tuple[dict[str, Any], np.ndarray, np.ndarray]] = []
        seen_paths = set()
        for expected_index, item in enumerate(self.instances):
            binding_index = int(item["binding_index"])
            if binding_index != expected_index:
                raise ValueError("deformable binding indices must be contiguous and ordered")
            mesh_path = str(item["mesh_prim_path"])
            parent_path = str(item["parent_prim_path"])
            if not mesh_path.startswith("/") or not parent_path.startswith("/"):
                raise ValueError("deformable Prim paths must be absolute")
            if mesh_path in seen_paths:
                raise ValueError(f"duplicate deformable Mesh path {mesh_path}")
            seen_paths.add(mesh_path)

            rows = mapping[mapping["binding_index"] == binding_index]
            vertex_count = int(item["vertex_count"])
            if len(rows) != vertex_count:
                raise ValueError(f"deformable mapping {mesh_path} has {len(rows)} rows; expected {vertex_count}")
            order = np.argsort(rows["vertex_index"], kind="stable")
            rows = rows[order]
            if not np.array_equal(rows["vertex_index"], np.arange(vertex_count, dtype=np.uint32)):
                raise ValueError(f"deformable mapping {mesh_path} has invalid vertex indices")
            mapping_kind = str(item.get("mapping_kind", "tet_barycentric_v1"))
            if mapping_kind == "tet_barycentric_v1":
                particle_indices = np.column_stack(tuple(rows[f"particle_{column}"] for column in range(4))).astype(
                    np.uint32, copy=False
                )
                weights = np.column_stack(tuple(rows[f"weight_{column}"] for column in range(4))).astype(
                    np.float32, copy=False
                )
                if (
                    not np.isfinite(weights).all()
                    or np.any(weights < -1.0e-5)
                    or not np.allclose(weights.sum(axis=1), 1.0, atol=1.0e-5)
                ):
                    raise ValueError(f"deformable mapping {mesh_path} has invalid weights")
            elif mapping_kind == "particle_direct_v1":
                direct = np.asarray(rows["particle_0"], dtype=np.uint32)
                particle_indices = np.repeat(direct[:, None], 4, axis=1)
                weights = np.zeros((len(rows), 4), dtype=np.float32)
                weights[:, 0] = 1.0
            else:
                raise ValueError(f"deformable mapping {mesh_path} uses unsupported kind {mapping_kind!r}")
            self._bindings.append((item, particle_indices, weights))

    @classmethod
    def from_messages(cls, info_message: Any, mapping_message: Any) -> "DeformableStateDecoder":
        return cls(json.loads(str(info_message.data)), mapping_message)

    def reconstruct(self, state_message: Any) -> tuple[int, float, list[dict[str, Any]]]:
        rows = _message_rows(state_message, STATE_DTYPE, "deformable state")
        if len(rows) != int(self.info.get("point_count", -1)):
            raise ValueError("deformable state count differs from info metadata")
        global_indices = np.asarray(rows["global_index"], dtype=np.int64)
        if len(np.unique(global_indices)) != len(global_indices):
            raise ValueError("deformable state contains duplicate global particle indices")
        max_index = max(
            int(self.info.get("model_particle_count", 0)) - 1,
            int(global_indices.max(initial=-1)),
        )
        global_to_row = np.full(max_index + 1, -1, dtype=np.int64)
        global_to_row[global_indices] = np.arange(len(rows), dtype=np.int64)
        positions = np.column_stack((rows["x"], rows["y"], rows["z"])).astype(
            np.float32,
            copy=False,
        )

        reconstructed = []
        for item, particle_indices, weights in self._bindings:
            if np.any(particle_indices > max_index):
                raise ValueError(f"deformable state misses particles for {item['mesh_prim_path']}")
            state_rows = global_to_row[particle_indices]
            if np.any(state_rows < 0):
                raise ValueError(f"deformable state misses particles for {item['mesh_prim_path']}")
            world_points = np.sum(positions[state_rows] * weights[:, :, None], axis=1)
            # Replay consumes recording-world points; keep mesh transforms as metadata.
            reconstructed.append(
                {
                    "metadata": item,
                    "points": np.asarray(world_points, dtype=np.float32),
                }
            )

        stamp_ns = stamp_to_nanoseconds(state_message.header.stamp)
        return stamp_ns, stamp_ns / 1_000_000_000.0, reconstructed

    def write_frame(
        self,
        state_message: Any,
        rigid_message: Any,
        output_dir: str | Path,
        frame_number: int,
    ) -> tuple[int, dict[str, dict[str, Any]], list[dict[str, Any]]]:
        stamp_ns, timestamp, reconstructed = self.reconstruct(state_message)
        root = Path(output_dir) / "deformable"
        rigid_stamp_ns = stamp_to_nanoseconds(rigid_message.header.stamp)
        if rigid_stamp_ns != stamp_ns:
            raise ValueError(f"deformable rigid/soft timestamp mismatch: {rigid_stamp_ns} != {stamp_ns}")
        rigid_rows = _message_rows(rigid_message, RIGID_DTYPE, "deformable rigid state")
        body_paths = list(self.info.get("body_paths") or [])
        if len(rigid_rows) != len(body_paths):
            raise ValueError(f"deformable rigid state has {len(rigid_rows)} bodies; expected {len(body_paths)}")
        order = np.argsort(rigid_rows["body_index"], kind="stable")
        rigid_rows = rigid_rows[order]
        if not np.array_equal(rigid_rows["body_index"], np.arange(len(body_paths), dtype=np.uint32)):
            raise ValueError("deformable rigid state has invalid body indices")
        objects: dict[str, dict[str, Any]] = {
            path.lstrip("/").replace("/", "_"): {"pose": _pose_matrix(row)} for path, row in zip(body_paths, rigid_rows)
        }
        frame_entries = []
        for result in reconstructed:
            item = result["metadata"]
            name = str(item["name"])
            relative_file = f"{name}/{int(frame_number):06d}.npz"
            destination = root / relative_file
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                destination,
                points=result["points"],
                prim_path=np.asarray(str(item["mesh_prim_path"])),
                time_stamp=np.float64(timestamp),
            )
            objects[str(item["object_key"])] = {"pose": item["parent_pose_world"]}
            frame_entries.append(
                {
                    "name": name,
                    "prim_path": str(item["mesh_prim_path"]),
                    "file": relative_file,
                    "n_nodes": int(item["vertex_count"]),
                }
            )
        return stamp_ns, objects, frame_entries


# Backward-compatible name for callers that adopted the original chef entrypoint.
ChefDeformableDecoder = DeformableStateDecoder
