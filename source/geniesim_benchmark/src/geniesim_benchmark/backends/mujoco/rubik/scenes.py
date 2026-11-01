# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Hash-verified import of the supplied scene-only Rubik USD archive."""

import ast
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path, PurePosixPath
import re
import stat
from zipfile import ZipFile

import numpy as np

from .catalog_metadata import public_catalog_metadata
from scipy.spatial.transform import Rotation


@dataclass
class SceneState:
    identifier: str
    turns: int
    relative_path: str
    sha256: str
    root: np.ndarray
    cubies: dict
    source_metadata: dict
    facelets: str = ""

    def receipt(self):
        return {
            "id": self.identifier,
            "turns": self.turns,
            "scene_usda": self.relative_path,
            "sha256": self.sha256,
            "initial_facelets": self.facelets,
            "source_metadata": self.source_metadata,
        }


def read_matrix(text, name):
    pattern = r'over "' + re.escape(name) + r'"\s*\{\s*matrix4d xformOp:transform\s*=\s*(\([^\n]+\))'
    matches = re.findall(pattern, text)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one initial matrix for {name}")
    matrix = np.asarray(ast.literal_eval(matches[0]), dtype=float)
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[:, 3], [0, 0, 0, 1], atol=1e-10)
    ):
        raise ValueError(f"Invalid USD pose matrix for {name}")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-8) or abs(np.linalg.det(rotation) - 1) > 1e-8:
        raise ValueError(f"USD pose contains scaling or reflection for {name}")
    return matrix


class SceneCatalog:
    def __init__(self, archive):
        self.archive = Path(archive).expanduser().resolve()
        self.archive_sha256 = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        self.scenes = []
        with ZipFile(self.archive) as zipfile:
            names = zipfile.namelist()
            manifests = [name for name in names if name.endswith("/manifest.json")]
            if len(manifests) != 1:
                raise ValueError("Scene archive requires one manifest.json")
            self.prefix = manifests[0].removesuffix("manifest.json")
            for info in zipfile.infolist():
                path = PurePosixPath(info.filename)
                if path.is_absolute() or ".." in path.parts or stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError("Unsafe path in scene archive")
            self.manifest = json.loads(zipfile.read(manifests[0]))
            for entry in self.manifest:
                path = entry["scene_usda"]
                if PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts:
                    raise ValueError("Unsafe scene path in manifest")
                content = zipfile.read(self.prefix + path)
                digest = hashlib.sha256(content).hexdigest()
                if digest != entry["sha256"]:
                    raise ValueError(f"Scene checksum mismatch: {path}")
                turns = entry["turns"]
                if type(turns) is not int or turns not in (1, 2, 3):
                    raise ValueError("Scenes must declare one, two or three turns")
                text = content.decode("utf8")
                names = re.findall(r'over "(cube_cubelet_[^"]+)"', text)
                expected = {
                    "cube_cubelet_" + "_".join(("pos_" if v > 0 else "neg_") + a for a, v in zip("xyz", c) if v)
                    for c in itertools.product((-1, 0, 1), repeat=3)
                    if any(c)
                }
                if len(names) != 26 or set(names) != expected:
                    raise ValueError(f"Scene must contain all 26 named cubies: {path}")
                if not np.allclose(read_matrix(text, "cube_middle"), np.eye(4), atol=1e-10):
                    raise ValueError("Nonidentity cube_middle transform is unsupported")
                root = read_matrix(text, "rubik_cube")
                cubies = {name: read_matrix(text, name) for name in names}
                public_source = public_catalog_metadata(entry)
                source = {
                    key: public_source.get(key)
                    for key in (
                        "training_overlap",
                        "source_scene_training_overlap_before_robot_removal",
                        "source_run_training_overlap",
                        "generalization",
                    )
                }
                self.scenes.append(SceneState(entry["replay_uuid"], turns, path, digest, root, cubies, source))
        if len({scene.identifier for scene in self.scenes}) != len(self.scenes):
            raise ValueError("Duplicate scene identifiers")

    def select(self, task_name, seed, count):
        match = re.fullmatch(r"rubik_cube_solve(?:_([123])turn)?", task_name)
        if not match:
            raise ValueError("Unsupported Rubik scene task")
        turns = int(match[1]) if match[1] else None
        eligible = [scene for scene in self.scenes if turns is None or scene.turns == turns]
        if not eligible:
            raise ValueError(f"No scenes for {task_name}")
        order = np.random.default_rng(seed).permutation(len(eligible))
        return [eligible[order[index % len(order)]] for index in range(count)]

    def extract(self, directory, *, assets_root=None):
        directory.mkdir(parents=True, exist_ok=False)
        with ZipFile(self.archive) as zipfile:
            zipfile.extractall(directory)
        if assets_root is not None:
            root = Path(assets_root).expanduser().resolve()
            # Resolve the archive's public-asset mount in the rendering copy only.
            # The original archive and native scene/checksum receipts stay unchanged.
            def resolve_asset(match):
                target = root / match[1]
                if not target.is_file():
                    raise RuntimeError("Scene USD asset is missing: " + str(target))
                return "@" + target.as_posix() + "@"

            for usd in directory.rglob("*.usda"):
                original = usd.read_text()
                resolved = re.sub(r"@/workspace/assets/([^@]+)@", resolve_asset, original)
                if resolved != original:
                    usd.write_text(resolved)
        return directory / self.prefix


def initialize_scene(model, data, cube, scene):
    """Initialize native joint coordinates from USD poses before the first tick."""
    if data.time != 0:
        raise RuntimeError("Cannot initialize a running scene")
    root = cube.root
    rotation = scene.root[:3, :3].T
    joints = range(model.body_jntadr[root], model.body_jntadr[root] + model.body_jntnum[root])
    for joint in joints:
        q = model.jnt_qposadr[joint]
        if int(model.jnt_type[joint]) == 2:
            axis = model.jnt_axis[joint]
            data.qpos[q] = np.dot(scene.root[3, :3] - model.body_pos[root], axis)
        elif int(model.jnt_type[joint]) == 1:
            data.qpos[q : q + 4] = np.roll(Rotation.from_matrix(rotation).as_quat(), 1)
        else:
            raise ValueError("Unexpected native cube root joint")
    for body, coord in cube.items:
        name = model.body(body).name.replace(":", "_")
        pose = scene.cubies[name]
        R = pose[:3, :3].T
        parent = model.body_parentid[body]
        for joint in range(model.body_jntadr[parent], model.body_jntadr[parent] + model.body_jntnum[parent]):
            q = model.jnt_qposadr[joint]
            kind = int(model.jnt_type[joint])
            if kind == 1:
                data.qpos[q : q + 4] = np.roll(Rotation.from_matrix(R).as_quat(), 1)
            elif kind == 3:
                data.qpos[q] = np.dot(Rotation.from_matrix(R).as_rotvec(), model.jnt_axis[joint])
            elif kind == 2:
                data.qpos[q] = np.dot(R.T @ pose[3, :3] - model.body_pos[body], model.jnt_axis[joint])
                if not model.jnt_range[joint, 0] - 1e-9 <= data.qpos[q] <= model.jnt_range[joint, 1] + 1e-9:
                    raise ValueError(f"Scene radial displacement exceeds native bounds: {name}")
            else:
                raise ValueError("Unexpected native cubie joint")
    import mujoco

    mujoco.mj_forward(model, data)
    if not np.allclose(data.xpos[root], scene.root[3, :3], atol=1e-8):
        raise ValueError("Scene core translation did not map to native physics")
    for body, _ in cube.items:
        pose = scene.cubies[model.body(body).name.replace(":", "_")]
        position = data.xmat[root].reshape(3, 3).T @ (data.xpos[body] - data.xpos[root])
        relative = data.xmat[root].reshape(3, 3).T @ data.xmat[body].reshape(3, 3)
        if not np.allclose(position, pose[3, :3], atol=1e-8) or not np.allclose(relative, pose[:3, :3].T, atol=1e-8):
            raise ValueError(f"Scene cubie pose did not map to native physics: {model.body(body).name}")
    scene.facelets, error = cube.decode(data)
    if scene.facelets is None or error > 3:
        raise ValueError("Scene initialization is not aligned")
