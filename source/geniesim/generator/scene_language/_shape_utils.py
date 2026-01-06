# engine-agnostic
# nonpublic
from __future__ import annotations
from typing import Callable, Union
from geniesim.generator.scene_language.math_utils import translation_matrix, _scale_matrix
from geniesim.generator.scene_language.type_utils import Box, ShapeSampler, Shape, T
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Hole:
    def __init__(self, name: str, docstring: str, normalize: bool, check: Box) -> None:
        """
        Args:
            name: str - a unique name
            docstring: str - a detailed function docstring
            check: Box - a 3D box that approximates function outputs
        """
        self.name: str = name
        self.docstring: str = docstring
        self.children: set[Hole] | None = None
        self.parents: set[Hole] = set()
        self.fn: ShapeSampler | None = None  # it's a sampler

        self.normalize = normalize
        self.box = check
        self.debug = False
        self.normalize_to_box = False

        self.implementation: Union[str, None] = None  # hack: store implementations for `sketch_helper.py`

        self.info = ""

    @property
    def is_implemented(self) -> bool:
        return self.fn is not None

    def __repr__(self):
        return f"Hole({self.name})"
        # return f'Hole({self.id}):\n"""\n{self.docstring}\n"""\n'

    def __call__(self, *args, **kwargs) -> Shape:
        if not self.is_implemented:
            if self.name == "primitive_call":
                print("this should not happen; fix it by adding `import mi_helper` at the start of the script")
                import ipdb

                ipdb.set_trace()  # should never happen
            (shape_,) = placeholder(
                center=self.box.center,
                info=self.info,
                scale=self.box.size,
                color=(0, np.random.rand(), 0),
            )
            shape_["info"] = {"docstring": self.name}
            return [shape_]  # placeholder shape
        assert self.fn is not None, self.fn
        if self.debug:
            from geniesim.generator.scene_language.shape_utils import _replace_shape_context

            with _replace_shape_context(True):
                shape = self.fn(*args, **kwargs)
        else:
            shape = self.fn(*args, **kwargs)

        if self.normalize or self.normalize_to_box:
            if self.normalize and self.normalize_to_box:
                logger.warning(f"{self.name} is already normalized, but normalize_to_box is set")

            # normalize
            box = compute_bbox(shape)
            shape = transform_shape(shape, _scale_matrix(1 / box.sizes.mean()) @ translation_matrix(-box.center))

            if self.normalize_to_box:
                # denormalize
                shape = transform_shape(
                    shape,
                    translation_matrix(np.asarray(self.box.center)) @ _scale_matrix(np.asarray(self.box.size).mean()),
                )

        return shape

    def add_parent(self, parent: Hole):
        self.parents.add(parent)

    def implement(self, impl_fn: Callable[[], ShapeSampler]):
        if self.is_implemented:
            print(f"[WARNING] {self.name} will be re-implemented!!!")
        else:
            print(f"[INFO] Implementing {self.name}...")

        # Comment these out - Matt
        # assert self.fn is None, self.fn
        # assert self.children is None, self.children

        _children.clear()
        fn = impl_fn()
        children = _children.copy()
        _children.clear()

        for child in children:
            child.add_parent(self)

        self.fn = fn
        self.children = children

    def get_descendants(self, visited=None):
        if visited is None:
            visited = {}  # including self!
        if self.name in visited:
            print(f"[ERROR] Unexpected, should debug: {visited}, {self}")
        visited.update({self.name: self})
        for child in self.children:
            if child.name not in visited:
                child.get_descendants(visited)
        return visited

    def get_descendants_by_depth(self, visited=None, cur_depth=0, max_depth=None) -> dict[str, Hole]:
        if max_depth is not None and cur_depth > max_depth:
            return visited if visited is not None else {}
        if visited is None:
            visited = {}
        if self.name in visited:
            print(f"[ERROR] Unexpected, should debug: {visited}, {self}, {cur_depth=}, {max_depth=}")
        visited.update({self.name: self})
        for child in self.children:
            if child.name not in visited:
                child.get_descendants_by_depth(visited, cur_depth=cur_depth + 1, max_depth=max_depth)
        return visited

    def get_ancestors(self, visited=None):
        if visited is None:
            visited = {self.name: self}  # including self!
        for parent in self.parents:
            if parent.name not in visited:
                visited[parent.name] = parent
                parent.get_ancestors(visited)
        return visited


library: dict[str, Hole] = {}  # maps id to holes
_children: set[Hole] = set()

assert "primitive_call" not in library, library.keys()
primitive_call = Hole(
    name="primitive_call", docstring="TODO", normalize=False, check=Box((0, 0, 0), 1)
)  # FIXME normalize??
assert "placeholder" not in library, library.keys()
placeholder = Hole(
    name="placeholder",
    docstring="Returns:\n\tShape - placeholder",
    normalize=False,
    check=Box((0, 0, 0), 1),
)


placeholder.implement(
    lambda: lambda center, info, scale, color=(0, 1, 0): [
        {
            "info": info,
            "type": "cube",
            "to_world": translation_matrix(center) @ _scale_matrix(scale, enforce_uniform=False) @ _scale_matrix(0.5),
            "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": color}},
            # 'bsdf': {'type': 'thindielectric', 'int_ior': 'bk7', 'ext_ior': 'air'},
            # 'bsdf': {'type': 'dielectric', 'int_ior': 'water', 'ext_ior': 'air'},
            # 'interior': {'type': 'homogeneous', 'albedo': {'type': 'rgb', 'value': color}},
            # 'bsdf': {'type': 'ref', 'id': 'green'}
        }
    ]
)


def compute_bbox(shape: Shape) -> "BBox":
    from geniesim.generator.scene_language.mi_helper import _preprocess_shape
    from geniesim.generator.scene_language.engine.utils.mitsuba_utils import (
        compute_bbox as _compute_bbox,
    )

    shape = _preprocess_shape(shape)
    scene_dict = {"type": "scene", **{f"{i:02d}": s for i, s in enumerate(shape)}}
    return _compute_bbox(scene_dict)


def compute_bboxes(shape: Shape) -> list["BBox"]:
    from geniesim.generator.scene_language.mi_helper import _preprocess_shape
    from geniesim.generator.scene_language.engine.utils.mitsuba_utils import (
        compute_bboxes as _compute_bboxes,
    )

    shape = _preprocess_shape(shape)
    scene_dict = {"type": "scene", **{f"{i:02d}": s for i, s in enumerate(shape)}}
    return _compute_bboxes(scene_dict)


def transform_shape(shape: Shape, pose: T) -> Shape:
    return [
        {k: v for k, v in s.items() if k != "to_world"} | {"to_world": np.asarray(pose) @ s["to_world"]} for s in shape
    ]
