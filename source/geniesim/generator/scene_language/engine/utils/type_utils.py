from typing import NamedTuple
from jaxtyping import Float
import numpy as np


class BBox(NamedTuple):
    # A n-dim box.
    center: Float[np.ndarray, "n"]
    min: Float[np.ndarray, "n"]
    max: Float[np.ndarray, "n"]
    sizes: Float[np.ndarray, "n"]
    size: float
