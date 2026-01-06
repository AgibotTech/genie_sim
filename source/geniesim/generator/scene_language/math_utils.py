import logging
import numpy as np
from scipy.spatial.transform import Rotation
from transforms3d._gohlketransforms import (
    translation_matrix as _translation_matrix,
    rotation_matrix as _rotation_matrix,
    reflection_matrix as _reflection_matrix,
    identity_matrix as _identity_matrix,
)
from geniesim.generator.scene_language.type_utils import P, T
from typing import Union

logger = logging.getLogger(__name__)


__all__ = [
    "translation_matrix",
    "rotation_matrix",
    # 'align_vectors',
    "scale_matrix",
    "reflection_matrix",
    "identity_matrix",
]


def rotation_matrix(angle: float, direction: P, point: P) -> T:
    """
    Args:
        angle (float) : the angle of rotation in radians
        direction (P) : the axis of rotation
        point (P) : the point about which the rotation is performed
    """
    # positive for counter-clockwise rotation if looking along the direction vector towards the origin
    return _rotation_matrix(angle, direction=direction, point=point)


def align_vectors(direction_to: P, direction_from: P) -> T:
    """
    Returns a rotation matrix that transforms `direction_from` to align with `direction_to`.
    """
    if np.linalg.norm(direction_to) < 1e-6 or np.linalg.norm(direction_from) < 1e-6:
        print(f"[ERROR] Aligning vectors with zero length: {direction_to=} {direction_from=}")
        return np.eye(4)
    ret = np.eye(4)
    rot, *_ = Rotation.align_vectors(direction_to, direction_from)
    ret[:3, :3] = rot.as_matrix()
    return ret


def translation_matrix(offset: P) -> T:
    """
    Args:
        offset (P) : the translation vector
    """
    return _translation_matrix(offset)


def scale_matrix(scale: float, origin: P) -> T:
    """
    Args:
        scale (float) - the scaling factor, only uniform scaling is supported
        origin (P) - the origin of the scaling operation
    """
    return _scale_matrix(scale, origin=origin)


def _scale_matrix(scale: Union[float, P], enforce_uniform: bool = True, origin: Union[P, None] = None) -> T:
    scale = np.asarray(scale)
    if scale.size == 1:
        scale = np.ones(3) * scale.item()
    else:
        assert scale.size == 3, f"Expected a scalar or a 3D vector, got {scale}"
        if enforce_uniform and not np.all(scale == scale.mean()):  # soft
            pass
            # logger.warning(f"Non-uniform scaling is not supported; {scale=}")
    out = np.diag(np.append(scale, 1))
    if origin is not None:
        origin = np.asarray(origin)
        out = translation_matrix(origin) @ out @ translation_matrix(-origin)
    return out


def reflection_matrix(point: P, normal: P) -> T:
    """
    Args:
        point: P - a point on the mirror plane
        normal: P - the normal vector of the mirror plane
    """
    return _reflection_matrix(point, normal)


def identity_matrix() -> T:
    """
    Returns the identity matrix in SE(3).
    """
    return _identity_matrix()
