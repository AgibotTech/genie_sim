from contextlib import contextmanager
from geniesim.generator.scene_language.type_utils import Box, ShapeSampler, Shape, T, P
from geniesim.generator.scene_language._shape_utils import (
    Hole,
    library,
    _children,
    placeholder,
    compute_bbox,
    transform_shape as _transform_shape,
)
import numpy as np


__all__ = ["transform_shape", "concat_shapes"]
# __all__ = ["transform_shape", "concat_shapes", "create_hole", "compute_bbox_sizes", "compute_bbox_center"]

_CHECK_SHAPE = False
_REPLACE_SHAPE = False


def create_hole(name: str, docstring: str, check: Box) -> ShapeSampler:
    """
    Creates a placeholder for a shape sampling function. The function will be later specified by 'impl_{name}'.
     Useful for defining recursive or complex shape functions and for function reuse.

    Args:
        name: str - a unique name
        docstring: str - a detailed function docstring
        check: Box - a 3D box that approximates function outputs
    Returns:
        ShapeSampler
    """

    if name in library:
        print(f"Warning: hole {name} already exists in {library.keys()}")
    else:
        hole = Hole(name=name, docstring=docstring, check=check, normalize=False)
        library[name] = hole
    hole = library[name]
    _children.add(hole)
    return hole


def concat_shapes(*shapes: Shape) -> Shape:
    """
    Combines multiple shapes into a single shape.
    """
    out = []
    for s in shapes:
        if isinstance(s, dict):
            # FIXME hack, so that GPT outputs run into compilation error less often
            out.append(s)
        else:
            out.extend(s)
    return out


def transform_shape(shape: Shape, pose: T) -> Shape:
    """
    Args:
        shape: Shape
        pose: T - If pose is A @ B, then B is applied first, followed by A.

    Returns:
        The input shape transformed by the given pose.
    """
    shape = _transform_shape(shape, pose)
    check = Box((0, 0, 0), 1)  # hack
    if _CHECK_SHAPE:
        _ = _check_shape(shape, check)
    if _REPLACE_SHAPE:
        shape = placeholder(center=check.center, scale=check.size, color=(0, 0, np.random.rand()))
    return shape


def compute_bbox_sizes(shape: Shape) -> P:
    """
    Returns the bounding box sizes along x, y, and z axes.
    """
    return compute_bbox(shape).sizes


def compute_bbox_center(shape: Shape) -> P:
    """
    Returns the bounding box center.
    """
    return compute_bbox(shape).center


@contextmanager
def _replace_shape_context(flag: bool):
    global _REPLACE_SHAPE
    orig_flag = _REPLACE_SHAPE
    _REPLACE_SHAPE = flag
    try:
        yield
    finally:
        _REPLACE_SHAPE = orig_flag


def _check_shape(shape: Shape, check: Box) -> bool:
    box = compute_bbox_sizes(shape)
    print(box, check)
    assert box == check, f"Expected {check}, got {box}"  # FIXME
    return True  # TODO
