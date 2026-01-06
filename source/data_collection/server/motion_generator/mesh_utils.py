# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Mesh utilities extracted from motion_gen_reacher.

Provides:
- simplify_obstacles_from_stage
- robust mesh simplifiers (_simplify_* helpers)
- get_mesh_attrs (triangulating wrapper that aligns with usd_helper.get_mesh_attrs)

Keep optional heavy imports local to functions to avoid import errors in non-ISAAC envs.
"""

import time

import numpy as np
from curobo.geom.types import Mesh
from curobo.types.math import Pose
from curobo.util.usd_helper import get_prim_world_pose

from common.base_utils.logger import logger

try:
    import torch
except Exception:
    torch = None


def simplify_obstacles_from_stage(
    world_config, max_faces=1000, min_faces=50, usd_path=None, save=False
):
    if not world_config.mesh:
        return world_config

    new_meshes = []

    for i, mesh in enumerate(world_config.mesh):
        try:
            logger.info(f"Processing mesh {i+1}/{len(world_config.mesh)}: {mesh.name}")

            vertices = np.array(mesh.vertices)
            faces = mesh.faces

            if len(vertices) == 0 or len(faces) == 0:
                logger.info(f"Skipping empty mesh: {mesh.name}")
                new_meshes.append(mesh)
                continue

            current_face_count = len(faces)
            logger.info(f"Original face count: {current_face_count}")

            if current_face_count <= max_faces:
                logger.info("Face count is within reasonable range, no simplification needed")
                new_meshes.append(mesh)
                continue

            target_faces = max(min_faces, min(max_faces, current_face_count // 2))
            simplified_mesh = None
            if simplified_mesh is None:
                logger.info("Starting mesh simplification")
                time0 = time.time()
                simplified_mesh = _simplify_mesh_robust(vertices, faces, target_faces, mesh.name)
                time1 = time.time()
                logger.info(f"Simplification time: {time1 - time0} seconds")

            if simplified_mesh is not None:
                mesh.vertices = simplified_mesh["vertices"]
                mesh.faces = simplified_mesh["faces"]
                logger.info(f"Simplification successful: {len(simplified_mesh['faces'])} faces")
            else:
                logger.info("Simplification failed, keeping original mesh")

            new_meshes.append(mesh)

        except Exception as e:
            logger.info(f"Error processing mesh {mesh.name}: {e}")
            new_meshes.append(mesh)

    world_config.mesh = new_meshes
    return world_config


def _simplify_mesh_robust(vertices, faces, target_faces, mesh_name):
    try:
        pass

        result = _simplify_with_pymeshlab(vertices, faces, target_faces)
        if result is not None:
            logger.info("Simplification with pymeshlab successful")
            return result
    except ImportError:
        pass
    except Exception as e:
        logger.info(f"pymeshlab simplification failed: {e}")

    try:
        result = _simplify_with_openmesh_robust(vertices, faces, target_faces)
        if result is not None:
            logger.info("Simplification with OpenMesh successful")
            return result
    except Exception as e:
        logger.info(f"OpenMesh simplification failed: {e}")

    try:
        result = _simplify_with_face_sampling(vertices, faces, target_faces)
        if result is not None:
            logger.info("Simplification with face sampling successful")
            return result
    except Exception as e:
        logger.info(f"Face sampling simplification failed: {e}")

    try:
        result = _simplify_with_uniform_sampling(vertices, faces, target_faces)
        if result is not None:
            logger.info("Simplification with uniform sampling successful")
            return result
    except Exception as e:
        logger.info(f"Uniform sampling simplification failed: {e}")

    return None


def _simplify_with_pymeshlab(vertices, faces, target_faces):
    try:
        import pymeshlab

        ms = pymeshlab.MeshSet()

        faces_array = []
        for face in faces:
            if len(face) == 3:
                faces_array.append(face)
            elif len(face) == 4:
                faces_array.append([face[0], face[1], face[2]])
                faces_array.append([face[0], face[2], face[3]])
            else:
                for i in range(1, len(face) - 1):
                    faces_array.append([face[0], face[i], face[i + 1]])

        faces_array = np.array(faces_array, dtype=np.int32)

        mesh = pymeshlab.Mesh(vertices.astype(np.float64), faces_array)
        ms.add_mesh(mesh)

        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces)

        simplified_mesh = ms.current_mesh()
        new_vertices = simplified_mesh.vertex_matrix()
        new_faces = simplified_mesh.face_matrix()

        return {"vertices": new_vertices, "faces": new_faces.tolist()}
    except Exception as e:
        logger.info(f"pymeshlab processing error: {e}")
        return None


def _simplify_with_openmesh_robust(vertices, faces, target_faces):
    try:
        import openmesh as om

        mesh = om.TriMesh()
        vertex_handles = []
        vertex_map = {}
        tolerance = 1e-6

        for i, point in enumerate(vertices):
            point_key = tuple(np.round(point / tolerance) * tolerance)
            if point_key in vertex_map:
                vertex_handles.append(vertex_map[point_key])
            else:
                vh = mesh.add_vertex([float(point[0]), float(point[1]), float(point[2])])
                vertex_handles.append(vh)
                vertex_map[point_key] = vh

        valid_faces = 0
        for face_indices in faces:
            if len(face_indices) < 3:
                continue
            try:
                valid_indices = []
                for idx in face_indices:
                    if 0 <= idx < len(vertex_handles):
                        valid_indices.append(idx)
                if len(valid_indices) < 3:
                    continue
                if len(valid_indices) == 3:
                    vhs = [vertex_handles[i] for i in valid_indices]
                    if len(set(vhs)) == 3:
                        mesh.add_face(vhs)
                        valid_faces += 1
                elif len(valid_indices) == 4:
                    vhs = [vertex_handles[i] for i in valid_indices]
                    if len(set(vhs)) == 4:
                        try:
                            mesh.add_face([vhs[0], vhs[1], vhs[2]])
                            mesh.add_face([vhs[0], vhs[2], vhs[3]])
                            valid_faces += 2
                        except Exception:
                            mesh.add_face([vhs[0], vhs[1], vhs[3]])
                            mesh.add_face([vhs[1], vhs[2], vhs[3]])
                            valid_faces += 2
                else:
                    vhs = [vertex_handles[i] for i in valid_indices]
                    center_vh = vhs[0]
                    for i in range(1, len(vhs) - 1):
                        if len(set([center_vh, vhs[i], vhs[i + 1]])) == 3:
                            mesh.add_face([center_vh, vhs[i], vhs[i + 1]])
                            valid_faces += 1
            except Exception:
                continue

        logger.info(f"Successfully added {valid_faces} faces")
        if valid_faces == 0:
            return None

        mesh.garbage_collection()

        current_faces = mesh.n_faces()
        if current_faces > target_faces:
            decimater = om.TriMeshDecimater(mesh)
            hqem = om.TriMeshModQuadricHandle()
            normal_flipping = om.TriMeshModNormalFlippingHandle()
            roundness = om.TriMeshModRoundnessHandle()
            decimater.add(hqem)
            decimater.add(normal_flipping)
            decimater.add(roundness)
            if decimater.initialize():
                decimater.module(normal_flipping).set_max_normal_deviation(25.0)
                decimater.module(roundness).set_min_roundness(0.1)
                decimater.decimate_to_faces(0, target_faces)
                mesh.garbage_collection()

        new_vertices = []
        for vh in mesh.vertices():
            point = mesh.point(vh)
            new_vertices.append([point[0], point[1], point[2]])

        new_faces = []
        for fh in mesh.faces():
            face_verts = [vh.idx() for vh in mesh.fv(fh)]
            new_faces.append(face_verts)

        return {"vertices": np.array(new_vertices), "faces": new_faces}
    except Exception as e:
        logger.info(f"OpenMesh processing error: {e}")
        return None


def _simplify_with_face_sampling(vertices, faces, target_faces):
    try:
        if len(faces) <= target_faces:
            return {"vertices": vertices, "faces": faces}

        face_importance = []
        for i, face_indices in enumerate(faces):
            if len(face_indices) < 3:
                face_importance.append(0)
                continue
            try:
                v0, v1, v2 = (
                    vertices[face_indices[0]],
                    vertices[face_indices[1]],
                    vertices[face_indices[2]],
                )
                edge1 = v1 - v0
                edge2 = v2 - v0
                area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
                face_importance.append(area)
            except (IndexError, ValueError):
                face_importance.append(0)

        importance_indices = np.argsort(face_importance)[::-1]
        selected_indices = importance_indices[:target_faces]
        selected_faces = [faces[i] for i in selected_indices]

        used_vertices = set()
        for face in selected_faces:
            used_vertices.update(face)

        old_to_new = {}
        new_vertices = []
        for i, old_idx in enumerate(sorted(used_vertices)):
            old_to_new[old_idx] = i
            new_vertices.append(vertices[old_idx])

        new_faces = []
        for face in selected_faces:
            try:
                new_face = [old_to_new[old_idx] for old_idx in face if old_idx in old_to_new]
                if len(new_face) >= 3:
                    new_faces.append(new_face)
            except KeyError:
                continue

        return {"vertices": np.array(new_vertices), "faces": new_faces}
    except Exception as e:
        logger.info(f"Face sampling simplification error: {e}")
        return None


def _simplify_with_uniform_sampling(vertices, faces, target_faces):
    try:
        if len(faces) <= target_faces:
            return {"vertices": vertices, "faces": faces}

        step = max(1, len(faces) // target_faces)
        selected_faces = faces[::step][:target_faces]

        valid_faces = []
        for face in selected_faces:
            if len(face) >= 3:
                valid_faces.append(face[:3])

        if not valid_faces:
            return None

        used_vertices = set()
        for face in valid_faces:
            used_vertices.update(face)

        old_to_new = {}
        new_vertices = []
        for i, old_idx in enumerate(sorted(used_vertices)):
            if old_idx < len(vertices):
                old_to_new[old_idx] = i
                new_vertices.append(vertices[old_idx])

        new_faces = []
        for face in valid_faces:
            try:
                new_face = [old_to_new[old_idx] for old_idx in face if old_idx in old_to_new]
                if len(new_face) == 3:
                    new_faces.append(new_face)
            except (KeyError, IndexError):
                continue

        return {"vertices": np.array(new_vertices), "faces": new_faces}
    except Exception as e:
        logger.info(f"Uniform sampling simplification error: {e}")
        return None


def get_mesh_attrs(prim, cache=None, transform=None):
    try:
        import curobo.util.usd_helper as _usd_helper
    except Exception:
        _usd_helper = None

    try:
        if _usd_helper and hasattr(_usd_helper, "get_mesh_attrs"):
            mesh = _usd_helper.get_mesh_attrs(prim, cache=cache, transform=transform)
        else:
            mesh = None
    except Exception as e:
        logger.error(
            f"usd_helper.get_mesh_attrs failed for {prim.GetPath() if hasattr(prim, 'GetPath') else prim}: {e}"
        )
        mesh = None

    if mesh is None:
        try:
            from pxr import UsdGeom

            if prim and prim.IsA(UsdGeom.Mesh):
                usd_mesh = UsdGeom.Mesh(prim)
                points = usd_mesh.GetPointsAttr().Get() or []
                fv_counts = usd_mesh.GetFaceVertexCountsAttr().Get() or []
                fv_indices = usd_mesh.GetFaceVertexIndicesAttr().Get() or []

                vertices = np.array([[p[0], p[1], p[2]] for p in points], dtype=np.float64)
                faces = []
                idx = 0
                for cnt in fv_counts:
                    if cnt <= 0:
                        continue
                    face = fv_indices[idx : idx + cnt]
                    idx += cnt
                    if len(face) == 3:
                        faces.append([int(face[0]), int(face[1]), int(face[2])])
                    elif len(face) > 3:
                        for i in range(1, len(face) - 1):
                            faces.append([int(face[0]), int(face[i]), int(face[i + 1])])

                class _M:
                    pass

                m = _M()
                m.name = str(prim.GetPath())
                m.vertices = vertices
                m.faces = faces
                m.pose = None
                mesh = m
        except Exception:
            mesh = None

    if mesh is None:
        return None

    try:
        vertices = np.array(mesh.vertices)
    except Exception:
        vertices = np.array([])

    faces = mesh.faces
    new_faces = []
    try:
        if isinstance(faces, np.ndarray) and faces.ndim == 2 and faces.shape[1] == 3:
            for f in faces:
                new_faces.append([int(f[0]), int(f[1]), int(f[2])])
        else:
            for f in faces:
                if isinstance(f, (list, tuple, np.ndarray)):
                    idxs = [int(x) for x in f]
                else:
                    continue
                if len(idxs) == 3:
                    new_faces.append(idxs)
                elif len(idxs) > 3:
                    for i in range(1, len(idxs) - 1):
                        a, b, c = idxs[0], idxs[i], idxs[i + 1]
                        if a == b or b == c or a == c:
                            continue
                        if len(vertices) == 0 or max(a, b, c) >= len(vertices) or min(a, b, c) < 0:
                            continue
                        new_faces.append([a, b, c])
                else:
                    continue
    except Exception as e:
        logger.error(f"Triangulation error for mesh {getattr(mesh, 'name', '<unknown>')}: {e}")
        return None

    if len(new_faces) == 0:
        return None

    try:
        mat, t_scale = get_prim_world_pose(cache, prim)
    except Exception:
        mat = None
        t_scale = None

    if mat is not None and transform is not None:
        try:
            mat = transform @ mat
        except Exception:
            pass

    mesh_pose = None
    try:
        if mat is not None and torch is not None:
            tensor_mat = torch.as_tensor(mat, device=torch.device("cuda", 0))
            mesh_pose = Pose.from_matrix(tensor_mat).tolist()
    except Exception:
        mesh_pose = getattr(mesh, "pose", None)

    mesh_scale = t_scale if t_scale is not None else getattr(mesh, "scale", None)

    try:
        if isinstance(mesh, Mesh):
            mesh.vertices = vertices
            mesh.faces = new_faces
            mesh.pose = mesh_pose
            mesh.scale = mesh_scale
            return mesh
    except Exception:
        pass

    try:
        mesh_name = getattr(mesh, "name", "<mesh>")
        new_mesh = Mesh(
            name=mesh_name,
            pose=mesh_pose,
            vertices=vertices,
            faces=new_faces,
            scale=mesh_scale,
        )
        return new_mesh
    except Exception as e:
        logger.error(f"Failed to construct Mesh for {getattr(mesh, 'name', '<mesh>')}: {e}")
        return None
