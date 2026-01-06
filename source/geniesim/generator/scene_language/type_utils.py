from typing import NamedTuple, Any, Callable, Union
import numpy as np
import numpy.typing

__all__ = ["T", "P", "Shape"]

P = np.typing.ArrayLike  # 3D vector, e.g., a point or direction
T = np.typing.ArrayLike  # 4x4 transformation matrix
Shape = list[dict[str, Any]]  # a shape is a list of primitive shapes
ShapeSampler = Callable[..., Shape]  # A shape sampling function


class Box(NamedTuple):
    """
    A 3D box.

    Examples:
        - `Box((0, 0, 0), 1)`
            A box with corner positions (-0.5, -0.5, -0.5) and (0.5, 0.5, 0.5).
        - `Box((0, 0, 0), (1, 2, 3))`
            A box with corner positions (-0.5, -1, -1.5) and (0.5, 1, 1.5).

    Attributes:
        center: P - Box center.
        size: float | P - Box size.
            If a float, the box is uniformly scaled; if a 3D vector, the box is non-uniformly scaled.
    """

    center: P
    size: Union[float, P]
