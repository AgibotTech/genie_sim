from geniesim.generator.scene_language.type_utils import Shape, P
import numpy as np
from typing import Literal, Tuple
from geniesim.generator.scene_language._shape_utils import compute_bbox, compute_bboxes
from geniesim.generator.scene_language.math_utils import translation_matrix
from geniesim.generator.scene_language.shape_utils import transform_shape, concat_shapes


# __all__ = ["attach", "align_with_min", "align_with_center"]
__all__ = ["compute_shape_center", "compute_shape_min", "compute_shape_max", "compute_shape_sizes"]


def attach(direction: P, shapes: list[Shape]) -> list[Shape]:
    """
    Iteratively attach each input shape along the given direction, with the first shape attached to the second, the second to the third, and so on.
    The LAST shape is the base and will not be transformed.
    Returns a shape list with the input order after attachment.
    """
    return _attach(direction, shapes)[0]


def _attach(direction: P, shapes: list[Shape], atol: float = 1e-2) -> Tuple[list[Shape], bool]:
    dir = np.asarray(direction) / np.linalg.norm(direction)
    out_shapes = []
    s2 = None
    proj_lens = []
    for s1 in reversed(shapes):
        if len(s1) == 0:
            out_shapes.append(s1)
            continue
        if s2 is not None:
            d1 = np.max([max(b.max @ dir, b.min @ dir) for b in compute_bboxes(s1)])
            d2 = np.min([min(b.max @ dir, b.min @ dir) for b in compute_bboxes(s2)])
            proj_lens.append(d1 - d2)
            s1 = transform_shape(s1, translation_matrix((d2 - d1) * dir))  # move s1 to attach to s2
        s2 = s1
        out_shapes.append(s2)
    # print('attach assertion', proj_lens)
    return list(reversed(out_shapes)), 0 if len(proj_lens) == 0 else np.abs(proj_lens).max() > atol


def align_with_min(normal: P, shapes: list[Shape]) -> list[Shape]:
    """
    Align shapes such that their bottom lie on some shared plane perp to the given normal, if the normal points up.
    The LAST shape is the base and will not be transformed.
    Returns a shape list with the input order after alignment.
    """
    return _align("min", normal, shapes)[0]


def align_with_center(normal: P, shapes: list[Shape]) -> list[Shape]:
    """
    Align shapes such that their center lie on some shared plane perp to the given normal.
    The LAST shape is the base and will not be transformed.
    Returns a shape list with the input order after alignment.
    """
    return _align("center", normal, shapes)[0]


def _align(
    key: Literal["min", "max", "center"], normal: P, shapes: list[Shape], atol: float = 1e-2
) -> Tuple[list[Shape], bool]:
    out_shapes = []
    normal = np.array(normal) / np.linalg.norm(normal)
    proj_lens = []
    for shape in shapes:
        box = compute_bbox(shape)
        proj_len = getattr(box, key) @ normal
        proj_lens.append(proj_len)
        trans = translation_matrix(-proj_len * normal)
        out_shapes.append(transform_shape(shape, trans))
    # set the last shape as the base  # TODO note this in docstring
    # out_shapes = [transform_shape(shape, -trans) for shape in out_shapes]
    # print('align assertion', np.asarray(proj_lens) - proj_lens[-1])
    return out_shapes, np.abs(np.asarray(proj_lens) - proj_lens[-1]).max() > atol


def compute_shape_center(shape: Shape) -> P:
    """
    Returns the shape center.
    """
    if isinstance(shape, dict):  # hack
        shape = [shape]
    return compute_bbox(shape).center


def compute_shape_min(shape: Shape) -> P:
    """
    Returns the min corner of the shape.
    """
    if isinstance(shape, dict):  # hack
        shape = [shape]
    box = compute_bbox(shape)
    return box.min


def compute_shape_max(shape: Shape) -> P:
    """
    Returns the max corner of the shape.
    """
    if isinstance(shape, dict):  # hack
        shape = [shape]
    box = compute_bbox(shape)
    return box.max


def compute_shape_sizes(shape: Shape) -> P:
    """
    Returns the shape sizes along x, y, and z axes.
    """
    if isinstance(shape, dict):  # hack
        shape = [shape]
    return compute_bbox(shape).sizes
