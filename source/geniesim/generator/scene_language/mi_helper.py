# engine-specific helper has access to engine-agnostic helper
from geniesim.generator.scene_language.engine.constants import (
    ENGINE_MODE,
    PROJ_DIR,
)  # set mitsuba variant within constants.py to avoid errors in typing mi.Sensor
import tempfile
from typing import Literal, Callable, Union, Optional, List
import time
from pathlib import Path
import mitsuba as mi
from geniesim.generator.scene_language.math_utils import (
    _scale_matrix,
    translation_matrix,
    rotation_matrix,
    identity_matrix,
)
from geniesim.generator.scene_language.type_utils import T, Shape, Box, P
from geniesim.generator.scene_language._shape_utils import (
    placeholder,
    primitive_call,
    transform_shape,
    compute_bbox,
)
from PIL import Image, ImageDraw
from tqdm import tqdm
import numpy as np
import numpy.typing
import xml.etree.ElementTree as ET
import hashlib
import uuid
from contextlib import redirect_stdout, contextmanager
import copy
import sys
import os
from geniesim.generator.scene_language.engine.utils.mitsuba_utils import (
    set_bsdf_refs,
    set_scene_dict_default,
    set_auto_camera,
)
from geniesim.generator.scene_language.engine.utils.type_utils import BBox

# from geniesim.generator.scene_language.engine.utils.camera_utils import orbit_camera

__all__ = ["execute"]


SPP = 32
try:
    import torch
    import torchvision.transforms.functional
    import torchvision.utils

    if torch.cuda.is_available():
        # SPP = {'mi': 4096, 'neural': 4096, 'lmd': 4}[ENGINE_MODE]
        SPP = {"mi": 1024, "neural": 1024, "lmd": 4}[ENGINE_MODE]
except Exception:
    pass


NUM_FRAMES = 6
RESOLUTION = 512
FOV = 49.1
ELEVATION = -20
REL_CAM_RADIUS = 2


def orbit_camera(elevation, azimuth, radius=1, is_degree=True, target=None):
    # left hand tie, negative y above
    # radius: scalar
    # elevation: scalar, in (-90, 90), from +y to -y is (-90, 90)
    # azimuth: scalar, in (-180, 180), from +z to +x is (0, 90)
    # return: [4, 4], camera pose matrix
    if is_degree:
        elevation = np.deg2rad(elevation)
        azimuth = np.deg2rad(azimuth)
    x = radius * np.cos(elevation) * np.sin(azimuth)
    y = -radius * np.sin(elevation)
    z = radius * np.cos(elevation) * np.cos(azimuth)
    if target is None:
        target = np.zeros([3], dtype=np.float32)
    return np.array([x, y, z]) + target  # [3]


def _preprocess_shape(shape: Shape, global_transform: Union[T, None] = None) -> Shape:
    if global_transform is None:
        global_transform = np.eye(4)
    global_transform = mi.scalar_rgb.Transform4f(global_transform)

    return [
        {
            kk: (vv if kk != "to_world" else (global_transform @ mi.scalar_rgb.Transform4f(vv)))
            for kk, vv in v.items()
            if kk != "info"
        }
        for v in shape
    ]


def render_depth(
    shape: Shape,
    save_dir: Union[str, None],
    sensors: dict[str, mi.Sensor],
    normalization: Union[T, None] = None,
) -> tuple[dict[str, np.typing.NDArray[np.bool_]], dict[str, np.typing.NDArray[np.float32]]]:
    save_dir = Path(save_dir)
    if normalization is not None:
        shape = transform_shape(shape, normalization)
    shape = _preprocess_shape(shape)
    scene_dict = {
        "type": "scene",
        "integrator": {"type": "aov", "aovs": "dd:depth"},
        **{f"{i:02d}": s for i, s in enumerate(shape)},
    }
    scene = mi.load_dict(scene_dict)
    segm_maps = {}
    depth_maps = {}
    for sensor_name, sensor in sensors.items():
        image = mi.render(scene, sensor=sensor, spp=4)
        mi.Bitmap(image).write((save_dir / f"sensor_{sensor_name}_depth.exr").as_posix())

        depth: np.typing.NDArray[np.float32] = np.asarray(image[:, :, 0])
        segm: np.typing.NDArray[np.bool_] = depth > 1e-3  # (h, w)

        Image.fromarray((segm * 255).astype(np.uint8)).save((save_dir / f"sensor_{sensor_name}_segm.png").as_posix())

        segm_maps[sensor_name] = segm
        depth_maps[sensor_name] = depth
    return segm_maps, depth_maps


def project(
    shape: Shape,
    save_dir: Union[str, None],
    sensors: dict[str, mi.Sensor],
    normalization: Union[T, None] = None,
) -> tuple[
    dict[str, list[BBox]],
    dict[str, list[np.typing.NDArray[np.bool_]]],
    dict[str, list[np.typing.NDArray[np.float32]]],
]:
    if save_dir is None:
        save_dir = Path("outputs/tmp")
        save_dir.mkdir(exist_ok=True)
    else:
        save_dir = Path(save_dir)

    if normalization is not None:
        shape = transform_shape(shape, normalization)
    shape = _preprocess_shape(shape)
    boxes_all: dict[str, list[BBox]] = {}
    segm_maps_all: dict[str, list[np.typing.NDArray[np.bool_]]] = {}
    depth_maps_all: dict[str, list[np.typing.NDArray[np.float32]]] = {}
    # canvas_all: dict[str, MyCanvas] = {}
    for sensor_name, sensor in sensors.items():
        boxes: list[BBox] = []
        boxes_all[sensor_name] = boxes
        # canvas = copy.deepcopy(global_canvas) if global_canvas is not None else MyCanvas()
        # canvas_all[sensor_name] = canvas
        segm_maps: list[np.typing.NDArray[np.bool_]] = []
        segm_maps_all[sensor_name] = segm_maps
        depth_maps: list[np.typing.NDArray[np.float32]] = []
        depth_maps_all[sensor_name] = depth_maps

        for ind, s in enumerate(shape):
            scene_dict = {
                "type": "scene",
                "integrator": {"type": "aov", "aovs": "dd:depth"},
                "shape": s,
            }
            scene = mi.load_dict(scene_dict)
            image = mi.render(scene, sensor=sensor, spp=4)
            save_to = save_dir / f"sensor_{sensor_name}_shape_{ind:02d}.exr"
            if False:
                mi.Bitmap(image).write(save_to.as_posix())
            if True:  # render_segm:
                depth: np.typing.NDArray[np.float32] = np.asarray(image[:, :, 0])
                segm: np.typing.NDArray[np.bool_] = depth > 1e-2  # (h, w)

                rows = np.any(segm, axis=1)
                cols = np.any(segm, axis=0)
                if not rows.any() or not cols.any():
                    box = BBox(
                        center=np.zeros(2),
                        size=0,
                        min=np.zeros(2),
                        max=np.zeros(2),
                        sizes=np.zeros(2),
                    )
                else:
                    y_min, y_max = np.where(rows)[0][[0, -1]]
                    x_min, x_max = np.where(cols)[0][[0, -1]]

                    if False:
                        disp = Image.fromarray(np.uint8(segm * 255))
                        draw = ImageDraw.Draw(disp)
                        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
                        disp.save(save_to.with_suffix(".png").as_posix())

                    box = BBox(
                        center=np.asarray((x_min + x_max) * 0.5, (y_min + y_max) * 0.5),
                        size=max(x_max - x_min, y_max - y_min),
                        min=np.asarray((x_min, y_min)),
                        max=np.asarray((x_max, y_max)),
                        sizes=np.asarray((x_max - x_min, y_max - y_min)),
                    )

                boxes.append(box)
                segm_maps.append(segm)
                depth_maps.append(depth)

        disp_all = Image.new("RGB", tuple(mi.Bitmap(image).size()), "white")
        draw_all = ImageDraw.Draw(disp_all)
        for box in boxes:
            draw_all.rectangle([box.min[0], box.min[1], box.max[0], box.max[1]], outline="red", width=2)

        disp_all = torchvision.transforms.functional.to_pil_image(
            torchvision.utils.draw_segmentation_masks(
                image=torchvision.transforms.functional.pil_to_tensor(disp_all),
                masks=torch.tensor(np.stack(segm_maps, axis=0)),
            )
        )

        save_to = save_dir / f"sensor_{sensor_name}_shape_all.png"
        disp_all.save(save_to)
    return boxes_all, segm_maps_all, depth_maps_all


XML_PATH_INDOORS = (Path(__file__).parent.parent / "assets/mitsuba/indoors/scene.xml").as_posix()
XML_PATH_INDOORS_NO_WINDOW = (Path(__file__).parent.parent / "assets/mitsuba/indoors_no_window/scene.xml").as_posix()
XML_PATH_OUTDOORS = (Path(__file__).parent.parent / "assets/mitsuba/outdoors/scene.xml").as_posix()
XML_PATH_TABLE = (Path(__file__).parent.parent / "assets/mitsuba/table/scene.xml").as_posix()
XML_PATH_ROVER_BACKGROUND = (Path(__file__).parent.parent / "assets/mitsuba/rover_background/scene.xml").as_posix()
XML_PATH_ROVER = (Path(__file__).parent.parent / "assets/mitsuba/rover/scene.xml").as_posix()


def find_ground_plane_rover() -> float:
    if not Path(XML_PATH_ROVER_BACKGROUND).exists():
        print(f"WARNING: {XML_PATH_ROVER_BACKGROUND} not found")
        return
    return 0
    scene: mi.Scene = mi.load_file(XML_PATH_ROVER_BACKGROUND)
    for shape in scene.shapes():
        print(shape)
    import ipdb

    ipdb.set_trace()
    scene: mi.Scene = mi.load_file(XML_PATH_ROVER)

    ground_plane = float("inf")
    for shape in scene.shapes():
        ground_plane = min(ground_plane, shape.bbox().min[1])
    return ground_plane


def find_box_rover() -> BBox:
    if not Path(XML_PATH_ROVER).exists():
        # print(f'WARNING: {XML_PATH_ROVER} not found')
        min_corner = np.array([-100, 0, -100])
        max_corner = np.array([100, 5.39451838, 100])
        center = (min_corner + max_corner) / 2
        sizes = max_corner - min_corner
        return BBox(center=center, size=max(sizes), min=min_corner, max=max_corner, sizes=sizes)
    scene: mi.Scene = mi.load_file(XML_PATH_ROVER)
    min_corner = np.array([float("inf"), float("inf"), float("inf")])
    max_corner = np.array([-float("inf"), -float("inf"), -float("inf")])
    for shape in scene.shapes():
        bbox = shape.bbox()
        min_corner = np.minimum(min_corner, bbox.min)
        max_corner = np.maximum(max_corner, bbox.max)
    center = (min_corner + max_corner) / 2
    sizes = max_corner - min_corner
    return BBox(center=center, size=max(sizes), min=min_corner, max=max_corner, sizes=sizes)


def find_box_indoors() -> BBox:
    if not Path(XML_PATH_INDOORS).exists():
        print(f"WARNING: {XML_PATH_INDOORS} not found")
        return
    box = find_box_table()
    ground_plane = find_ground_plane_indoors()
    h = box.max[1] - ground_plane
    center = np.array([box.center[0], ground_plane + h / 2, box.center[2]])
    sizes = np.array([box.sizes[0] * 1.5, h, box.sizes[2] * 1.5])
    return BBox(center=center, size=max(sizes), min=center - sizes / 2, max=center + sizes / 2, sizes=sizes)


def find_ground_plane_indoors(scale=1.0, shift=0.0) -> float:
    if not Path(XML_PATH_INDOORS).exists():
        print(f"WARNING: {XML_PATH_INDOORS} not found")
        return
    scene: mi.Scene = mi.load_file(XML_PATH_INDOORS)

    for shape in scene.shapes():
        if shape.id() == "FloorTiles":
            ground_plane = scale * shape.bbox().max[1] + shift
            break
    return ground_plane


def find_box_table() -> BBox:
    if not Path(XML_PATH_TABLE).exists():
        print(f"WARNING: {XML_PATH_TABLE} not found")
        return
    scene: mi.Scene = mi.load_file(XML_PATH_TABLE)

    for shape in scene.shapes():
        if shape.id() == "WhiteMarble":
            xmin, _, zmin = np.asarray(shape.bbox().min)
            xmax, table_plane, zmax = np.asarray(shape.bbox().max)
        if shape.id() == "FloorTiles":
            floor_plane = shape.bbox().max[1]
    h = (table_plane - floor_plane) * 1.5

    return BBox(
        center=np.array(((xmin + xmax) / 2, table_plane + h / 2, (zmin + zmax) / 2)),
        size=max(xmax - xmin, h, zmax - zmin),
        min=np.array((xmin, table_plane, zmin)),
        max=np.array((xmax, table_plane + h, zmax)),
        sizes=np.array((xmax - xmin, h, zmax - zmin)),
    )


def find_ground_plane_table() -> float:
    if not Path(XML_PATH_TABLE).exists():
        print(f"WARNING: {XML_PATH_TABLE} not found")
        return
    scene: mi.Scene = mi.load_file(XML_PATH_TABLE)

    for shape in scene.shapes():
        if shape.id() == "WhiteMarble":
            ground_plane = shape.bbox().max[1]
            break
    return ground_plane


def find_box_outdoors() -> BBox:
    scale = 40
    ground_plane = find_ground_plane_outdoors()
    return BBox(
        center=np.array([0, scale / 2 + ground_plane, -scale / 2]),
        size=scale,
        min=np.array([-scale / 2, ground_plane, -scale]),
        max=np.array([scale / 2, scale / 2 + ground_plane, 0]),
        sizes=np.array([scale, scale / 2, scale]),
    )


def find_ground_plane_outdoors() -> float:
    return -2


def create_coord_system(scale, global_transform: Optional[T] = None):
    r = 0.001 * scale
    h = 1 * scale

    # def create_axis(c: P, t: T):
    #     return transform_shape(primitive_call('cube', scale=[r, h, r], color=c), t @ translation_matrix([0, h / 2, 0]))
    # x = create_axis([1, 0, 0], rotation_matrix(direction=[0, 0, 1], angle=-np.pi / 2, point=(0, 0, 0)))
    # y = create_axis([0, 1, 0], identity_matrix())
    # z = create_axis([0, 0, 1], rotation_matrix(direction=[1, 0, 0], angle=np.pi / 2, point=(0, 0, 0)))

    # x = transform_shape(primitive_call('cube', scale=[h, r, r], color=[1, 0, 0]), translation_matrix([h / 2, 0, 0]))
    # y = transform_shape(primitive_call('cube', scale=[r, h, r], color=[0, 1, 0]), translation_matrix([0, h / 2, 0]))
    # z = transform_shape(primitive_call('cube', scale=[r, r, h], color=[0, 0, 1]), translation_matrix([0, 0, h / 2]))

    x = transform_shape(cube_fn(scale=[h, r, r], color=[1, 0, 0]), translation_matrix([h / 2, 0, 0]))
    y = transform_shape(cube_fn(scale=[r, h, r], color=[0, 1, 0]), translation_matrix([0, h / 2, 0]))
    z = transform_shape(cube_fn(scale=[r, r, h], color=[0, 0, 1]), translation_matrix([0, 0, h / 2]))

    shape = transform_shape(x + y + z, global_transform)
    shape = _preprocess_shape(shape)
    xx, yy, zz = shape
    coord = {
        "type": "scene",
        "integrator": {"type": "aov", "aovs": "alb:albedo,dd:depth"},
        "x": xx,
        "y": yy,
        "z": zz,
    }
    return coord


SCENE_PRESETS = {
    "rover_background": {
        "xml_path": XML_PATH_ROVER_BACKGROUND,
        "ground_plane": find_ground_plane_rover(),
        "box": find_box_rover(),
        "coord_scale": 1,
    },
    # 'indoors_no_window': {
    #     'xml_path': XML_PATH_INDOORS_NO_WINDOW,
    #     'ground_plane': find_ground_plane_indoors(shift=0.),    # shift=2.
    #     'box': find_box_indoors(),
    #     'coord_scale': 1,
    # },
    # 'indoors': {
    #     'xml_path': XML_PATH_INDOORS,
    #     'ground_plane': find_ground_plane_indoors(),
    #     'box': find_box_indoors(),
    #     'coord_scale': 1,
    # },
    # 'outdoors': {
    #     'xml_path': XML_PATH_OUTDOORS,
    #     'ground_plane': find_ground_plane_outdoors(),
    #     'box': find_box_outdoors(),
    #     'coord_scale': 1,
    # },
    # 'table': {
    #     'xml_path': XML_PATH_TABLE,
    #     'ground_plane': find_ground_plane_table(),
    #     'box': find_box_table(),
    #     'coord_scale': 1,
    # },
}


def concatenate_xml_files(orig_path: str, tmp_path: str):
    original_tree = ET.parse(orig_path)
    original_root = original_tree.getroot()
    tmp_tree = ET.parse(tmp_path)
    tmp_root = tmp_tree.getroot()

    # assume that shape IDs won't collide
    existing_ids = {child.get("id") for child in original_root.iterfind(".//*[@id]")}
    for element in tmp_root.iterfind(".//*[@id]"):
        if element.get("id") in existing_ids:
            raise RuntimeError(f"ID collision: {element.get('id')}")
            # element.set('id', element.get('id') + "_new")

    for child in tmp_root:
        original_root.append(child)
    return original_tree


def spherical_interp(P0, P1, N):
    if N == 0:
        return np.zeros(((0, 3)))
    # TODO: fix and improve this
    M = (P0 + P1) / 2
    vec = P1 - P0

    normal = np.cross(P0, P1)
    normal = normal / np.linalg.norm(normal)

    bisector = np.cross(normal, vec)
    bisector = bisector / np.linalg.norm(bisector)

    radius = np.linalg.norm(P0 - M)
    center = M + bisector * radius

    angles = np.linspace(0, np.pi, N)
    interpolated_points = []

    for angle in angles:
        rotation_matrix = np.array(
            [
                [
                    np.cos(angle) + normal[0] ** 2 * (1 - np.cos(angle)),
                    normal[0] * normal[1] * (1 - np.cos(angle)) - normal[2] * np.sin(angle),
                    normal[0] * normal[2] * (1 - np.cos(angle)) + normal[1] * np.sin(angle),
                ],
                [
                    normal[1] * normal[0] * (1 - np.cos(angle)) + normal[2] * np.sin(angle),
                    np.cos(angle) + normal[1] ** 2 * (1 - np.cos(angle)),
                    normal[1] * normal[2] * (1 - np.cos(angle)) - normal[0] * np.sin(angle),
                ],
                [
                    normal[2] * normal[0] * (1 - np.cos(angle)) - normal[1] * np.sin(angle),
                    normal[2] * normal[1] * (1 - np.cos(angle)) + normal[0] * np.sin(angle),
                    np.cos(angle) + normal[2] ** 2 * (1 - np.cos(angle)),
                ],
            ]
        )

        point_on_arc = np.dot(rotation_matrix, (P0 - center)) + center
        interpolated_points.append(point_on_arc)

    return np.array(interpolated_points)


def linear_interp(P0, P1, N):
    t_values = np.linspace(0, 1, N)
    interpolated_points = []

    for t in t_values:
        interpolated_point = (1 - t) * P0 + t * P1
        interpolated_points.append(interpolated_point)

    return np.array(interpolated_points)


def generate_forward_facing_spiral(box, num=30, radius=0.3, rots=1):
    # points = [
    #     np.asarray((box.center[0] - box.sizes[0] * .6, box.center[1] + box.sizes[1] * .6, box.center[2] + box.sizes[2] * 2)),
    #     np.asarray((box.center[0], box.center[1] + box.sizes[1] * .6, box.center[2] + box.sizes[2] * 2)),
    #     np.asarray((box.center[0] + box.sizes[0] * .6, box.center[1] + box.sizes[1] * .6, box.center[2] + box.sizes[2] * 2)),
    #     np.asarray((box.center[0] + box.sizes[0] * .6, box.center[1] + box.sizes[1] * 1, box.center[2] + box.sizes[2] * 2)),
    #     np.asarray((box.center[0], box.center[1] + box.sizes[1] * 1, box.center[2] + box.sizes[2] * 2)),
    #     np.asarray((box.center[0] - box.sizes[0] * .6, box.center[1] + box.sizes[1] * 1, box.center[2] + box.sizes[2] * 2)),
    # ]

    spiral_center = np.asarray(
        (
            box.center[0] - box.sizes[0] * 0.2,
            box.center[1] + box.sizes[1] * 1.2,
            box.center[2] + box.sizes[2] * 1.2,
        )
    )
    points = []
    for theta in np.linspace(0.0, 2.0 * np.pi * rots, num + 1)[:-1]:
        ab = box.center - spiral_center
        u = ab / np.linalg.norm(ab)
        if np.allclose(u, np.array([1, 0, 0])):
            w = np.array([0, 1, 0])
        else:
            w = np.array([1, 0, 0])
        v1 = np.cross(u, w)
        v1 /= np.linalg.norm(v1)
        v2 = np.cross(u, v1)
        v2 /= np.linalg.norm(v2)

        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        v_rotated = rotation_matrix @ np.array([v1, v2])
        # v_rotated = v_rotated[0] * np.cos(theta) + v_rotated[1] * np.sin(theta)
        cam_location = spiral_center + radius * v_rotated[0] + radius * v_rotated[1]
        points.append(cam_location)

    box_points = [
        np.asarray(
            (
                box.center[0] - box.sizes[0] * 0.5,
                box.center[1] - box.sizes[1] * 0.5,
                box.center[2] - box.sizes[2] * 0.5,
            )
        ),
        np.asarray(
            (
                box.center[0] + box.sizes[0] * 0.5,
                box.center[1] - box.sizes[1] * 0.5,
                box.center[2] - box.sizes[2] * 0.5,
            )
        ),
        np.asarray(
            (
                box.center[0] - box.sizes[0] * 0.5,
                box.center[1] + box.sizes[1] * 0.5,
                box.center[2] - box.sizes[2] * 0.5,
            )
        ),
        np.asarray(
            (
                box.center[0] + box.sizes[0] * 0.5,
                box.center[1] + box.sizes[1] * 0.5,
                box.center[2] - box.sizes[2] * 0.5,
            )
        ),
        np.asarray(
            (
                box.center[0] - box.sizes[0] * 0.5,
                box.center[1] - box.sizes[1] * 0.5,
                box.center[2] + box.sizes[2] * 0.5,
            )
        ),
        np.asarray(
            (
                box.center[0] + box.sizes[0] * 0.5,
                box.center[1] - box.sizes[1] * 0.5,
                box.center[2] + box.sizes[2] * 0.5,
            )
        ),
        np.asarray(
            (
                box.center[0] - box.sizes[0] * 0.5,
                box.center[1] + box.sizes[1] * 0.5,
                box.center[2] + box.sizes[2] * 0.5,
            )
        ),
        np.asarray(
            (
                box.center[0] + box.sizes[0] * 0.5,
                box.center[1] + box.sizes[1] * 0.5,
                box.center[2] + box.sizes[2] * 0.5,
            )
        ),
    ]

    fovs = []
    for point in points:
        fov = fov_from_box_cam(point, box_points, box)
        fovs.append(fov)
    return points, fovs


def generate_360_spiral(box, num=30, radius=1.0, rots=1):
    # rotate along y-axis
    # points = [
    #     np.asarray((box.center[0] - box.sizes[0] * .6, box.center[1] + box.sizes[1] * .6, box.center[2] + box.sizes[2] * 2)),
    #     np.asarray((box.center[0], box.center[1] + box.sizes[1] * .6, box.center[2] + box.sizes[2] * 2)),
    #     np.asarray((box.center[0] + box.sizes[0] * .6, box.center[1] + box.sizes[1] * .6, box.center[2] + box.sizes[2] * 2)),
    #     np.asarray((box.center[0] + box.sizes[0] * .6, box.center[1] + box.sizes[1] * 1, box.center[2] + box.sizes[2] * 2)),
    #     np.asarray((box.center[0], box.center[1] + box.sizes[1] * 1, box.center[2] + box.sizes[2] * 2)),
    #     np.asarray((box.center[0] - box.sizes[0] * .6, box.center[1] + box.sizes[1] * 1, box.center[2] + box.sizes[2] * 2)),
    # ]

    # spiral_start = np.asarray((box.center[0] - box.sizes[0] * .2, box.center[1] + box.sizes[1] * 1.2, box.center[2] + box.sizes[2] * 1.2))
    # spiral_length = np.asarray((-box.sizes[0] * .2, box.sizes[1] * 1.2, box.sizes[2] * 1.2)) * radius
    spiral_start = np.asarray((box.sizes[0] * 0.2, box.sizes[1] * 1.5, box.sizes[2] * 0.7))
    spiral_length = ((spiral_start[0] ** 2 + (spiral_start[1] * 0) ** 2 + spiral_start[2] ** 2) ** 0.5) * radius
    points = []
    for theta in np.linspace(0.0, 2.0 * np.pi * rots, num + 1)[:-1]:
        cam_location = (
            spiral_length * np.asarray([np.cos(theta), 0.0, np.sin(theta)])
            + np.asarray([0.0, spiral_start[1] * radius, 0.0])
            + box.center
        )
        points.append(cam_location)

    box_points = [
        np.asarray(
            (
                box.center[0] - box.sizes[0] * 0.5,
                box.center[1] - box.sizes[1] * 0.5,
                box.center[2] - box.sizes[2] * 0.5,
            )
        ),
        np.asarray(
            (
                box.center[0] + box.sizes[0] * 0.5,
                box.center[1] - box.sizes[1] * 0.5,
                box.center[2] - box.sizes[2] * 0.5,
            )
        ),
        np.asarray(
            (
                box.center[0] - box.sizes[0] * 0.5,
                box.center[1] + box.sizes[1] * 0.5,
                box.center[2] - box.sizes[2] * 0.5,
            )
        ),
        np.asarray(
            (
                box.center[0] + box.sizes[0] * 0.5,
                box.center[1] + box.sizes[1] * 0.5,
                box.center[2] - box.sizes[2] * 0.5,
            )
        ),
        np.asarray(
            (
                box.center[0] - box.sizes[0] * 0.5,
                box.center[1] - box.sizes[1] * 0.5,
                box.center[2] + box.sizes[2] * 0.5,
            )
        ),
        np.asarray(
            (
                box.center[0] + box.sizes[0] * 0.5,
                box.center[1] - box.sizes[1] * 0.5,
                box.center[2] + box.sizes[2] * 0.5,
            )
        ),
        np.asarray(
            (
                box.center[0] - box.sizes[0] * 0.5,
                box.center[1] + box.sizes[1] * 0.5,
                box.center[2] + box.sizes[2] * 0.5,
            )
        ),
        np.asarray(
            (
                box.center[0] + box.sizes[0] * 0.5,
                box.center[1] + box.sizes[1] * 0.5,
                box.center[2] + box.sizes[2] * 0.5,
            )
        ),
    ]

    fovs = []
    for point in points:
        fov = fov_from_box_cam(point, box_points, box)
        fovs.append(fov)

    return points, fovs


def fov_from_box_cam(cam_position, box_points, box):
    # distances = np.array([np.linalg.norm(box_point - cam_position) for box_point in box_points])
    # min_distance = np.min(distances)
    # min_distance_idx = np.argmin(distances)
    # closest_point = box_points[min_distance_idx]

    angles = []

    for box_point in box_points:
        ab = box.center - cam_position
        ac = box_point - cam_position
        dot_product = np.dot(ab, ac)
        ab_magnitude = np.linalg.norm(ab)
        ac_magnitude = np.linalg.norm(ac)
        cos_theta = dot_product / (ab_magnitude * ac_magnitude)
        angle = np.arccos(cos_theta)
        angle = np.rad2deg(angle)
        angles.append(angle)

    angle = max(angles)
    fov = 2 * angle
    return fov


def compute_best_view_from_z(box, fov=60):
    # https://keep.google.com/u/0/#NOTE/1jRg8-HbrVj2_UJLJwyTQZoWJJknTzybJJvUspDlqWnyK8aaaYiz1UybTVvlVjRs
    _, my, mz = box.min
    cx, _, _ = box.center
    _, sy, sz = box.sizes
    syz = np.sqrt(sy**2 + sz**2)

    beta = np.arctan(sy / sz)
    gamma = np.deg2rad(90 - fov / 2)
    v = syz / 2 / np.cos(gamma)

    theta = np.pi - beta - gamma
    bv_up = (cx, my + v * np.sin(theta), mz + sz + v * np.cos(theta))

    bv_mid = (cx, my + sy / 2, mz + sz + v * np.sin(gamma))

    theta = max(np.pi / 2 - beta - gamma, 0)
    bv_down = (cx, my + v * np.sin(theta), mz + v * np.cos(theta))
    return [bv_up, bv_mid, bv_down], [fov, fov, fov]


def compute_best_view_from_z_from_top(box, fov=60, alpha=-0.1, num_frames=3, pad=None):
    assert fov < 90, fov
    if pad is not None:
        pad = np.asarray(pad) * box.size
        box = BBox(
            center=box.center,
            size=box.size,
            min=np.asarray(box.min) - pad,
            max=np.asarray(box.max) + pad,
            sizes=np.asarray(box.sizes) + 2 * pad,
        )
    _, my, mz = box.min
    cx, _, _ = box.center
    sx, sy, sz = box.sizes
    ((_, bv_y, bv_z), _, _), _ = compute_best_view_from_z(box, fov * 2)  # Circle center
    radius = np.sqrt((bv_y - my) ** 2 + (bv_z - (mz + sz)) ** 2)  # same as v from compute_best_view_from_z
    # radius >= cam_y - bv_y
    # => cam_y <= bv_y + radius
    cam_y = min(bv_y + radius, my + (1 + alpha) * sy)
    cam_z = bv_z + np.sqrt(radius**2 - (cam_y - bv_y) ** 2)
    cam_x_list = np.linspace(cx - sx, cx + sx, num_frames)
    return [(cam_x, cam_y, cam_z) for cam_x in cam_x_list], [fov] * num_frames


def compute_best_views(box, target_box):
    center = np.asarray(box.center)  # [x, y, z]
    sizes = np.asarray(box.sizes)

    # first look at the box center from +z direction
    if sizes[1] <= sizes[2]:
        bv_z = np.asarray(
            [
                center[0],
                (sizes[1] ** 2 + sizes[2] ** 2) / (2 * sizes[1]),
                center[2] + sizes[2] * 0.5,
            ]
        )
        fov_z = np.arctan(sizes[1] / sizes[2])
        fov_z = 2 * np.rad2deg(fov_z)
    else:
        bv_z = np.asarray(
            [
                center[0],
                center[1] + sizes[1] * 0.5,
                (sizes[1] ** 2 + sizes[2] ** 2) / (2 * sizes[2]),
            ]
        )
        fov_z = np.arctan(sizes[2] / sizes[1])
        fov_z = 2 * np.rad2deg(fov_z)

    # then look at the box center from +x direction
    if sizes[1] <= sizes[0]:
        bv_x = np.asarray(
            [
                center[0] + sizes[0] * 0.5,
                (sizes[1] ** 2 + sizes[0] ** 2) / (2 * sizes[1]),
                center[2],
            ]
        )
        fov_x = np.arctan(sizes[1] / sizes[0])
        fov_x = 2 * np.rad2deg(fov_x)
    else:
        bv_x = np.asarray(
            [
                (sizes[1] ** 2 + sizes[0] ** 2) / (2 * sizes[0]),
                center[1] + sizes[1] * 0.5,
                center[2],
            ]
        )
        fov_x = np.arctan(sizes[0] / sizes[1])
        fov_x = 2 * np.rad2deg(fov_x)

    bestviews = [bv_z, bv_x]
    fovs = [fov_z, fov_x]

    # use target_box to limit the range, this case will only be with too large y value
    final_bestviews = []
    final_fovs = []

    for idx, (bv, fov) in enumerate(zip(bestviews, fovs)):
        target_box_min = np.asarray(target_box.center - target_box.sizes / 2)
        target_box_max = np.asarray(target_box.center + target_box.sizes / 2)

        if bv[1] >= target_box_max[1]:
            y_threshold = target_box_max[1]
            if idx == 0:
                # along z axis
                new_bv = np.asarray(
                    [
                        center[0] + 0.5 * sizes[0] * (y_threshold - 0.5 * sizes[1]) / (bv[1] - 0.5 * sizes[1]),
                        y_threshold,
                        bv[2],
                    ]
                )
                new_fov = np.arctan(
                    0.5
                    * np.sqrt(sizes[0] ** 2 + sizes[1] ** 2)
                    / np.sqrt((new_bv[0] - center[0]) ** 2 + (y_threshold - 0.5 * sizes[1]) ** 2)
                )
                new_fov = 2 * np.rad2deg(new_fov)
            elif idx == 1:
                # along x axis
                new_bv = np.asarray(
                    [
                        bv[0],
                        y_threshold,
                        center[2] - 0.5 * sizes[2] * (y_threshold - 0.5 * sizes[1]) / (bv[1] - 0.5 * sizes[1]),
                    ]
                )
                new_fov = np.arctan(
                    0.5
                    * np.sqrt(sizes[2] ** 2 + sizes[1] ** 2)
                    / np.sqrt((center[2] - new_bv[2]) ** 2 + (y_threshold - 0.5 * sizes[1]) ** 2)
                )
                new_fov = 2 * np.rad2deg(new_fov)
            final_bestviews.append(new_bv)
            final_fovs.append(new_fov)
        else:
            final_bestviews.append(bv)
            final_fovs.append(fov)

    return final_bestviews, final_fovs


def compute_normalization(shape, preset_id="rover_background"):
    preset = SCENE_PRESETS[preset_id]
    box = compute_bbox(shape)
    target_box = preset["box"]
    scale = min(target_box.sizes / box.sizes)
    normalization = (
        translation_matrix((target_box.center[0], preset["ground_plane"], target_box.center[2]))
        @ _scale_matrix(scale)
        @ translation_matrix((-box.center[0], box.sizes[1] / 2 - box.center[1], -box.center[2]))
    )
    return normalization


def execute_from_preset(
    shape: Shape,
    save_dir: Optional[str],
    preset_id: Literal["rover_background"] = "rover_background",
    # normalization: Union[None, T] = None,
    # sensors: Union[None, dict[str, mi.Sensor]] = None,
    prev_out: Optional[dict] = None,
    timestep: Optional[tuple[int, int]] = None,
) -> dict:
    out = dict()
    normalization: Union[None, T] = None if prev_out is None else prev_out["normalization"]
    sensors: Union[None, dict[str, mi.Sensor]] = None if prev_out is None else prev_out["sensors"]
    sensor_info = None if prev_out is None else prev_out["sensor_info"]
    preset = SCENE_PRESETS[preset_id]
    if normalization is None:
        normalization = compute_normalization(shape, preset_id)

    out["normalization"] = normalization

    shape = transform_shape(shape, normalization)
    # print('after', compute_bbox(shape))
    # print('target', target_box)

    shape = _preprocess_shape(shape)

    ply_path_to_tmp_ply_path: dict[str, str] = {}
    tmp_shape = []
    tmp_ply_dir = Path(preset["xml_path"]).parent / "meshes"
    tmp_ply_dir.mkdir(exist_ok=True, parents=True)
    mesh_shape = []
    for s in shape:
        if s["type"] == "ply":
            ply_path = Path(s["filename"]).absolute().as_posix()
            if ply_path not in ply_path_to_tmp_ply_path:
                tmp_ply_path = tmp_ply_dir / f"tmp_{uuid.uuid4()}.ply"
                # print('creating soft link', tmp_ply_path, '->', s['filename'])
                tmp_ply_path.symlink_to(s["filename"])
                ply_path_to_tmp_ply_path[ply_path] = tmp_ply_path.as_posix()

            tmp_shape.append(
                {
                    "filename": ply_path_to_tmp_ply_path[ply_path],
                    **{kk: vv for kk, vv in s.items() if kk != "filename"},
                }
            )
            mesh_shape.append(
                {
                    "filename": ply_path_to_tmp_ply_path[ply_path],
                    **{kk: vv for kk, vv in s.items() if kk != "filename"},
                }
            )
        else:
            tmp_shape.append(s)
    shape = tmp_shape

    if False:  # engine_mode == 'neural':
        from optimize_utils import layout_optimize, layout_optimize_mi

        # scene, tmp_xml_file = layout_optimize(mesh_shape, prompt, save_dir, preset['xml_path'], preset['box'])
        scene, tmp_xml_file = layout_optimize_mi(mesh_shape, prompt, save_dir, preset["xml_path"], preset["box"])

    # if you want to rescale the vertex color
    # scene_dict = {'type': 'scene'}
    # need_rescale_ids = []
    # for i, s in enumerate(shape):
    #     scene_dict.update({**{f'{i:02d}': s}})
    #     if 'filename' in s.keys() and 'tmp' in s['filename']:
    #         need_rescale_ids.append(f'{i:02d}')
    else:
        tmp_xml_file = Path(preset["xml_path"]).with_name(f"tmp_{uuid.uuid4()}.xml")
        scene_dict = {"type": "scene", **{f"{i:02d}": s for i, s in enumerate(shape)}}
        with open(os.devnull, "w") as f, redirect_stdout(f):
            mi.xml.dict_to_xml(scene_dict, tmp_xml_file)
        tree = concatenate_xml_files(preset["xml_path"], tmp_xml_file.as_posix())

        tree.write(tmp_xml_file.as_posix())
        with suppress_output():
            scene: mi.Scene = mi.load_file(tmp_xml_file.as_posix())
    # out['sensors'] = {'rendering': scene.sensors()[0]}

    # also for rescale the vertex color
    # for shape_tmp in scene.shapes():
    #     if shape_tmp.id() in need_rescale_ids:
    #         shape_params = mi.traverse(shape_tmp)
    #         tmp = shape_params['vertex_color'] / 255
    #         shape_params['vertex_color'] = tmp
    #         shape_params.update()

    if sensors is None:
        sensors = {}
        box = compute_bbox(shape)  # box **after** normalization
        canon_sensor = scene.sensors()[0]
        # canon_transform = canon_sensor.world_transform()

        # time_step = timestep[0]
        # all_time_step = timestep[1]
        # # TODO always 6 views for now
        #
        # # points = [
        # #     np.asarray((box.center[0] - box.sizes[0] * .6, box.center[1] + box.sizes[1] * .6, box.center[2] + box.sizes[2] * 2)),
        # #     np.asarray((box.center[0], box.center[1] + box.sizes[1] * .6, box.center[2] + box.sizes[2] * 2)),
        # #     np.asarray((box.center[0] + box.sizes[0] * .6, box.center[1] + box.sizes[1] * .6, box.center[2] + box.sizes[2] * 2)),
        # #     np.asarray((box.center[0] + box.sizes[0] * .6, box.center[1] + box.sizes[1] * 1, box.center[2] + box.sizes[2] * 2)),
        # #     np.asarray((box.center[0], box.center[1] + box.sizes[1] * 1, box.center[2] + box.sizes[2] * 2)),
        # #     np.asarray((box.center[0] - box.sizes[0] * .6, box.center[1] + box.sizes[1] * 1, box.center[2] + box.sizes[2] * 2)),
        # # ]
        #
        # traj_campos = list()
        #
        # campos, fovs = generate_forward_facing_spiral(box, num=all_time_step)
        # sample_fov = max(fovs) * 1.1
        # if sample_fov > 180:
        #     # from pdb import set_trace; set_trace()
        #     sample_fov = 60  # FIXME
        #     fovs = [sample_fov] * len(fovs)
        # sample_fovs = [sample_fov for _ in fovs]     # uncomment this if need to use the same fovs
        # traj_campos.append({
        #     'campos': campos,
        #     'fovs_changing': fovs,
        #     'fovs_fixing': sample_fovs,
        #     'traj_name': 'forward_facing'
        # })
        #
        # campos, fovs = generate_360_spiral(box, num=all_time_step)
        # sample_fov = max(fovs) * 1.1
        # if sample_fov > 180:
        #     # from pdb import set_trace; set_trace()
        #     sample_fov = 60  # FIXME
        #     fovs = [sample_fov] * len(fovs)
        # sample_fovs = [sample_fov for _ in fovs]     # uncomment this if need to use the same fovs
        # traj_campos.append({
        #     'campos': campos,
        #     'fovs_changing': fovs,
        #     'fovs_fixing': sample_fovs,
        #     'traj_name': '360_view'
        # })
        #
        # for traj in traj_campos:
        #     points = traj['campos']
        #     all_cam_pos = np.stack(points)
        #     all_cam_pos = all_cam_pos[time_step:]
        #
        #     fovs_changing = traj['fovs_changing']
        #     fovs_fixing = traj['fovs_fixing']
        #     traj_name = traj['traj_name']
        #
        #     for cidx in range(all_cam_pos.shape[0]):
        #         cam_pos = all_cam_pos[cidx]
        #         sensor_changing: mi.Sensor = mi.load_dict({
        #             'type': 'perspective',
        #             'to_world': mi.scalar_rgb.Transform4f.look_at(
        #                 origin=cam_pos,
        #                 target=box.center,
        #                 up=[0, 1, 0]
        #             ),
        #             'fov': fovs_changing[cidx],
        #             'film': {
        #                 'type': 'hdrfilm',
        #                 'width': 512,
        #                 'height': 512,
        #             }
        #         })
        #         # sensors[f'rendering_traj_{traj_name}_changing_{cidx:03d}'] = sensor_changing
        #     for cidx in range(all_cam_pos.shape[0]):
        #         cam_pos = all_cam_pos[cidx]
        #         sensor_fixing: mi.Sensor = mi.load_dict({
        #             'type': 'perspective',
        #             'to_world': mi.scalar_rgb.Transform4f.look_at(
        #                 origin=cam_pos,
        #                 target=box.center,
        #                 up=[0, 1, 0]
        #             ),
        #             'fov': fovs_fixing[cidx],
        #             'film': {
        #                 'type': 'hdrfilm',
        #                 'width': 512,
        #                 'height': 512,
        #             }
        #         })
        #         # sensors[f'rendering_traj_{traj_name}_fixing_{cidx:03d}'] = sensor_fixing
        #
        # sensor: mi.Sensor = mi.load_dict({
        #     'type': 'perspective',
        #     # 'type': 'orthographic',
        #     # 'to_world': mi.scalar_rgb.Transform4f.translate(
        #     #     box.size * np.asarray(rel_offset),
        #     # ) @ canon_sensor.world_transform(),
        #     'to_world': mi.scalar_rgb.Transform4f.look_at(
        #         # origin=np.array(canon_transform.translation()) + box.size * np.asarray([0, .4, -2]),
        #         # origin=np.array([target_box.center[0], target_box.max[1], target_box.max[2]]),
        #         origin=target_box.max,
        #         # target=np.array(canon_transform.matrix)[:3, :3] @ np.array([0, 0, 1]) + np.array(canon_transform.translation()),
        #         target=box.center,
        #         up=[0, 1, 0]
        #     ),
        #     'fov': np.array(mi.traverse(canon_sensor)['x_fov']).item(),
        #     'film': {
        #         'type': 'hdrfilm',
        #         'width': 512,
        #         'height': 512,
        #         'pixel_format': 'rgba',
        #     }
        # })
        # # sensors['rendering_up'] = sensor
        #
        # bestviews, bestview_fovs = compute_best_view_from_z(box)  # compute_best_views(box, target_box)
        # bestviews, bestview_fovs = compute_best_view_from_z_from_top(box, fov=40, num_frames=6, alpha=-0.1, pad=.1)  # compute_best_views(box, target_box)
        # # take the first and last view, interpolate the trajectory to be of fixed radius towards box.center
        # first_view = np.asarray(bestviews[0])
        target = box.center
        # dir1 = first_view - target
        # elev = np.rad2deg(np.arctan2(dir1[1], np.linalg.norm([dir1[0], dir1[2]])))
        # radius = np.linalg.norm(dir1)
        # print(f'[INFO] elev: {elev}, radius: {radius}')
        num_frames = NUM_FRAMES  # 6
        azims = np.linspace(0, 360, num_frames, endpoint=False).tolist()
        # bestviews = [orbit_camera(-elev, azim, radius=radius, target=target) for azim in azims]
        #
        #     # d = dir1 * (i / (num_frames - 1)) + dir2 * (1 - i / (num_frames - 1))
        #     # d = d / np.linalg.norm(d) * np.linalg.norm(dir1)
        #     # bestviews.append(target + d)

        elev = ELEVATION  # -20
        radius = np.linalg.norm(box.sizes) / 2 * REL_CAM_RADIUS  # * 2
        bestviews = [orbit_camera(elev, azim, radius=radius, target=target) for azim in azims]
        fov = FOV
        bestview_fovs = [fov] * len(bestviews)

        sensor_info = {
            "elev": [elev] * num_frames,
            "radius": [radius] * num_frames,
            "azim": azims,
            "eyes": bestviews,
            "targets": [target] * num_frames,
            "fov": bestview_fovs,
        }

        for vidx in range(len(bestviews)):
            sensor_bestview: mi.Sensor = mi.load_dict(
                {
                    "type": "perspective",
                    "to_world": mi.ScalarTransform4f().look_at(
                        origin=bestviews[vidx],
                        # target=np.array(canon_transform.matrix)[:3, :3] @ np.array([0, 0, 1]) + np.array(canon_transform.translation()),
                        target=box.center,
                        up=[0, 1, 0],
                    ),
                    "fov": bestview_fovs[vidx],
                    "film": {
                        "type": "hdrfilm",
                        "width": RESOLUTION,
                        "height": RESOLUTION,
                        "pixel_format": "rgba",
                    },
                }
            )
            sensors[f"bestview_{vidx:02d}"] = sensor_bestview

    # rename sensors
    sensors_cat = {}
    for sensor_ind, (_, sensor) in enumerate(sensors.items()):
        sensors_cat[f"rendering_traj_{sensor_ind:03d}"] = sensor
    out["sensors"] = sensors_cat
    out["sensor_info"] = sensor_info

    if save_dir is None:
        return out
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    # for k in tqdm(out['sensors'].keys(), desc='rendering RGBs...'):  # cause misformatted outputs in execute_err.txt
    for k in out["sensors"].keys():
        image = mi.render(scene, sensor=out["sensors"][k], spp=SPP)
        image = mi.util.convert_to_bitmap(image)
        image = Image.fromarray(np.asarray(image))
        image.save(save_dir / f"{k}.png")

    # coord = mi.load_dict(create_coord_system(preset['coord_scale'], out['normalization']) | {f'{i:02d}': s for i, s in enumerate(shape)})
    coord_dict = mi.load_dict(create_coord_system(preset["coord_scale"], out["normalization"]))
    for k in []:  # ['rendering_up']: #out['sensors'].keys():
        coord = mi.render(coord_dict, sensor=out["sensors"][k], spp=16)
        # mi.Bitmap(coord).write((save_dir / f'{k}_coord.exr').as_posix())
        depth: np.typing.NDArray[np.float32] = np.asarray(coord[:, :, -1])  # (h, w)
        segm: np.typing.NDArray[np.bool_] = depth > 1e-2  # (h, w)
        coord = np.asarray(coord[:, :, :3]).clip(0, 1)
        # coord[~segm] = 0
        coord_save_path = save_dir / f"{k}_coord.png"
        Image.fromarray(np.dstack([(coord * 255).astype(np.uint8), segm.astype(np.uint8) * 255])).save(
            coord_save_path.as_posix()
        )
        Image.blend(
            Image.open(save_dir / f"{k}.png").convert("RGBA"),
            Image.open(coord_save_path).convert("RGBA"),
            alpha=0.3,
        ).save((save_dir / f"{k}_coord_overlay.png"))

    render_depth = False
    if render_depth:
        depth_scene_dict = {
            "type": "scene",
            "integrator": {"type": "depth"},
            **{f"{i:02d}": s for i, s in enumerate(shape)},
        }
        depth_scene = mi.load_dict(depth_scene_dict)
        depth_save_dir = save_dir / "depth"
        depth_save_dir.mkdir(exist_ok=True)
        for k in out["sensors"].keys():
            image = mi.render(depth_scene, sensor=out["sensors"][k], spp=4)

            depth: np.typing.NDArray[np.float32] = np.asarray(image)
            segm: np.typing.NDArray[np.bool_] = depth > 0  # (h, w)
            if segm.any():
                valid_depth = depth[segm]
                dmin = valid_depth.min()
                dmax = valid_depth.max()
                depth_normalized = np.zeros_like(depth)
                depth_normalized[segm] = 0.2 + 0.8 * (depth[segm] - dmin) / (dmax - dmin)
            else:
                depth_normalized = np.zeros_like(depth)
            Image.fromarray((depth_normalized * 255).astype(np.uint8)).save((depth_save_dir / f"{k}.png").as_posix())

    # clean up
    tmp_xml_file.unlink()
    for tmp_ply_path in ply_path_to_tmp_ply_path.values():
        Path(tmp_ply_path).unlink()

    # we should do optimization after loading sensors?
    # from optimize_utils import debug_layout_optimize
    # scene = debug_layout_optimize(scene, keys=[f'{i:02d}' for i in range(len(shape))])

    return out


def execute(
    shape: Shape,
    save_dir: Union[str, None] = None,
    save_prefix: Union[str, None] = None,
    sensor: Union[dict, None, int] = None,
    extra_scene_dict: Union[dict, None] = None,
) -> dict:
    if save_dir is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(time.time()))
        save_dir = Path(__file__).parent.parent / f"outputs/helper_{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    save_prefix = "" if save_prefix is None else save_prefix + "_"

    if extra_scene_dict is None:
        extra_scene_dict = {}
    shape = _preprocess_shape(shape)
    scene_dict = extra_scene_dict | {f"{i:02d}": s for i, s in enumerate(shape)}
    scene_dict = set_scene_dict_default(scene_dict)
    scene_dict = set_bsdf_refs(scene_dict)
    if sensor is None:
        scene_dict = set_auto_camera(scene_dict)
        sensors = list(range(5))
    elif isinstance(sensor, int):
        scene_dict = set_auto_camera(scene_dict)
        sensors = [sensor]
    else:
        scene_dict["sensor"] = sensor
        sensors = [0]
    scene = mi.load_dict(scene_dict)
    for sensor in sensors:
        image = mi.render(scene, sensor=sensor)
        image = mi.util.convert_to_bitmap(image)
        image = Image.fromarray(np.asarray(image))
        image.save(save_dir / f"{save_prefix}sensor_{sensor:02d}.png")
    return scene_dict


cube_fn: Callable[[Union[float, P], P], Shape] = lambda info, scale, color=(1, 1, 1): [
    {
        "type": "cube",
        "to_world": _scale_matrix(scale, enforce_uniform=False) @ _scale_matrix(0.5),
        "bsdf": {
            "type": "diffuse",
            "reflectance": {"type": "rgb", "value": np.asarray(color[:3]).clip(0, 1)},
        },
        "info": {"stack": [], "info": info},
    }
]

# https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#cylinder-cylinder
cylinder_fn: Callable[[float, P, P], Shape] = lambda info, radius, p0, p1, color=(1, 1, 1): [
    {
        "type": "cylinder",
        "p0": mi.ScalarPoint3f(*p0),
        "p1": mi.ScalarPoint3f(*p1),
        "radius": radius,
        "to_world": identity_matrix(),
        "bsdf": {
            "type": "diffuse",
            "reflectance": {"type": "rgb", "value": np.asarray(color[:3]).clip(0, 1)},
        },
        "info": {"stack": [], "info": info},
    }
]

sphere_fn: Callable[[P, Union[float, P]], Shape] = lambda info, color=(1, 1, 1), radius=1: [
    {
        "type": "sphere",  # l if np.all(np.asarray(scale) == np.mean(scale)) else 'cube',  # FIXME hack
        "to_world": _scale_matrix(radius),
        # 'to_world': _scale_matrix(0.5 * np.asarray(scale)),
        "bsdf": {
            "type": "diffuse",
            "reflectance": {"type": "rgb", "value": np.asarray(color[:3]).clip(0, 1)},
        },
        # 'bsdf': {'type': 'ref', 'id': 'red'},
        "info": {"stack": [], "info": info},
    }
]


def cone_fn(radius: float, p0: P, p1: P, color=(1, 1, 1)) -> Shape:
    num_segments = max(1, int(np.linalg.norm(np.asarray(p1) - np.asarray(p0)) * 10))
    segments = []
    for i in range(num_segments):
        # p0_segment, p1_segment should be along the centerline of the cone
        p0_segment = np.asarray(p0) + (np.asarray(p1) - np.asarray(p0)) * (i / num_segments)
        p1_segment = np.asarray(p0) + (np.asarray(p1) - np.asarray(p0)) * ((i + 1) / num_segments)
        radius_segment = radius * (1 - i / num_segments)
        segments.append(
            {
                "type": "cylinder",
                "p0": mi.ScalarPoint3f(*p0_segment),
                "p1": mi.ScalarPoint3f(*p1_segment),
                "radius": radius_segment,
                "to_world": identity_matrix(),
                "bsdf": {
                    "type": "diffuse",
                    "reflectance": {"type": "rgb", "value": np.asarray(color[:3]).clip(0, 1)},
                },
                "info": {"stack": []},
            }
        )
    return segments


def curve_fn(name: str, control_points: List[P], radius: Union[float, List[float]], color: P) -> Shape:
    if not isinstance(radius, (list, tuple)):
        radius = [radius] * len(control_points)
    if len(radius) != len(control_points):
        print(f"[ERROR] len({radius=}) != len({control_points=})")
        radius = [np.mean(radius)] * len(control_points)
    tmpdir = Path(PROJ_DIR) / "tmp"
    tmpdir.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, dir=tmpdir.as_posix()) as f:
        for point, r in zip(control_points, radius):
            f.write(f"{float(point[0])} {float(point[1])} {float(point[2])} {float(r)}\n")
        fn = f.name

    return [
        {
            "type": name,
            "filename": fn,
            "to_world": identity_matrix(),
            "bsdf": {
                "type": "diffuse",
                "reflectance": {"type": "rgb", "value": np.asarray(color[:3]).clip(0, 1)},
            },
            "info": {"stack": []},
        }
    ]


def bsplinecurve_fn(control_points: List[P], radius: Union[float, List[float]], color: P) -> Shape:
    return curve_fn("bsplinecurve", control_points, radius, color)


def linearcurve_fn(control_points: List[P], radius: Union[float, List[float]], color: P) -> Shape:
    return curve_fn("linearcurve", control_points, radius, color)


_from_minecraft_cuboid_fn: Callable[[P, Union[float, P]], Shape] = lambda block_type, scale, fill: [
    {
        "type": "cube",
        "to_world": _scale_matrix(scale, enforce_uniform=False) @ _scale_matrix(0.5),
        "bsdf": {
            "type": "diffuse",
            "reflectance": {"type": "rgb", "value": np.asarray(color[:3]).clip(0, 1)},
        },
        "info": {"stack": []},
    }
]
# unit_cube.implement(lambda: lambda s: [{'type': 'cube', 'to_world': mi.scalar_rgb.Transform4f.scale(s).scale(0.5)}])
# unit_sphere.implement(lambda: lambda: [{'type': 'sphere', 'to_world': mi.scalar_rgb.Transform4f(np.eye(4)), 'bsdf': {'type': 'ref', 'id': 'red'}}])


def filename_to_color(filename: str):
    """Generate a consistent RGB color for a given filename using hashing."""
    # Create a hash of the filename
    hash_object = hashlib.sha256(filename.encode())
    hex_hash = hash_object.hexdigest()

    # Convert the first 6 characters of the hash to an integer
    # and normalize it to the range [0, 1]
    r = int(hex_hash[0:2], 16) / 255.0
    g = int(hex_hash[2:4], 16) / 255.0
    b = int(hex_hash[4:6], 16) / 255.0

    return [r, g, b]


def box_fn(
    prompt: str,
    kwargs: dict,
    scale: P,
    center: Union[P, None] = None,
    enforce_centered_origin: bool = True,
    shape_type: Literal["cube", "sphere"] = "cube",
    **extra_info,
):
    if enforce_centered_origin and center is not None:
        print(f"enforce_centered_origin is True but {center=}")
    if center is None:
        center = (0, 0, 0)
    return [
        {
            "type": shape_type,
            "to_world": translation_matrix(center) @ _scale_matrix(scale, enforce_uniform=False) @ _scale_matrix(0.5),
            "bsdf": {
                "type": "diffuse",
                "reflectance": {"type": "rgb", "value": filename_to_color(prompt)},
            },
            "info": {"docstring": prompt, "stack": [], "kwargs": kwargs, **extra_info},
        }
    ]


def primitive_box_fn(prompt: str, kwargs: dict, shape: Shape, **extra_info):
    assert len(shape) == 1, shape
    return [
        {
            **shape[0],
            "bsdf": {
                "type": "diffuse",
                "reflectance": {"type": "rgb", "value": filename_to_color(prompt)},
            },
            "info": {"docstring": prompt, "stack": [], "kwargs": kwargs, **extra_info},
        }
    ]


@contextmanager
def suppress_output():
    # Save the original file descriptors
    original_stdout_fd = sys.__stdout__.fileno()
    original_stderr_fd = sys.__stderr__.fileno()

    # Open /dev/null
    with open(os.devnull, "w") as devnull:
        # Duplicate the original file descriptors
        saved_stdout_fd = os.dup(original_stdout_fd)
        saved_stderr_fd = os.dup(original_stderr_fd)

        try:
            # Redirect stdout and stderr to /dev/null
            os.dup2(devnull.fileno(), original_stdout_fd)
            os.dup2(devnull.fileno(), original_stderr_fd)

            yield
        finally:
            # Restore original file descriptors
            os.dup2(saved_stdout_fd, original_stdout_fd)
            os.dup2(saved_stderr_fd, original_stderr_fd)

            # Close the duplicated file descriptors
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)


def shap_e_fn(
    prompt: str,
    scale: Union[float, P, None] = None,
    allow_rotate_y: bool = True,
    allow_rotate_x: bool = False,
    allow_rotate_z: bool = False,
    center: Union[P, None] = None,
    enforce_centered_origin: bool = True,
):
    if enforce_centered_origin and center is not None:
        print(f"enforce_centered_origin is True but {center=}")
    if center is None:
        center = (0, 0, 0)

    from neural_helper import run_pipe, get_cache_save_dir, run_TripoSR

    prompt_save_dir = get_cache_save_dir(prompt)
    run_pipe(prompt=prompt, save_dir=prompt_save_dir, overwrite=False)
    # run_TripoSR(prompt=prompt, save_dir=prompt_save_dir, overwrite=False)

    ind = 0  # FIXME
    ply_save_path = Path(prompt_save_dir) / f"{ind:02d}.ply"
    # print(f'{ply_save_path=}')
    # node.normalize_to_box = True  # hack
    shape = [
        {
            "type": "ply",
            "filename": ply_save_path.as_posix(),
            "to_world": np.eye(4),
            "bsdf": {
                "type": "diffuse",
                "reflectance": {
                    "type": "rgb",
                    "value": filename_to_color(ply_save_path.as_posix()),
                },
            },
            "info": {"docstring": prompt, "stack": []},
        }
    ]
    if scale is not None:
        with suppress_output():
            box = compute_bbox(shape)
        shape = transform_shape(shape, translation_matrix(-box.center))

        # angles = [0, np.pi / 2]
        angles = [0, np.pi / 2]  # , np.pi, 3 * np.pi / 2]
        axes = []
        if allow_rotate_y:
            axes.append((0, 1, 0))
        if allow_rotate_x:
            axes.append((1, 0, 0))
        if allow_rotate_z:
            axes.append((0, 0, 1))
        best_fit = None
        min_difference = float("inf")

        scale_to = np.broadcast_to(np.asarray(scale), (3,))
        for axis in axes:
            for angle in angles:
                rot_matrix = rotation_matrix(angle, axis, point=(0, 0, 0))
                scale_from = np.abs(np.dot(rot_matrix[:3, :3], box.sizes))
                # print(f'{axis=}, {angle=}, {scale_from=}, {scale_to=}')
                difference = np.abs(np.log(scale_to) - np.log(scale_from)).sum()

                if difference < min_difference:
                    # print('best fit axis', axis, 'angle', angle)
                    min_difference = difference
                    best_fit = (rot_matrix, scale_from)
        if best_fit is None:
            rot_matrix = identity_matrix()
            scale_from = np.abs(np.dot(rot_matrix[:3, :3], box.sizes))
            best_fit = (rot_matrix, scale_from)
        shape = transform_shape(shape, best_fit[0])
        shape = transform_shape(shape, _scale_matrix(scale_to / best_fit[1], enforce_uniform=False))

        # scale_from = box.sizes
        # scale_from_alt = np.array([scale_from[2], scale_from[1], scale_from[0]])
        # scale_to = np.broadcast_to(np.asarray(scale), (3,))
        # if np.abs(np.log(scale_to) - np.log(scale_from_alt)).sum() < np.abs(np.log(scale_to) - np.log(scale_from)).sum():
        #     print('rotate by 90 degrees')
        #     shape = transform_shape(shape, rotation_matrix(np.pi / 2, (0, 1, 0)))
        #     scale_from = scale_from_alt

        # shape = transform_shape(shape, _scale_matrix(scale_to / scale_from, enforce_uniform=False))
    shape = transform_shape(shape, translation_matrix(center))

    return shape


def impl_primitive_call():
    def fn(name: Literal["sphere", "cube", "cylinder"], **kwargs):

        # print(kwargs)
        return {
            "cube": cube_fn,
            "sphere": sphere_fn,
            "cylinder": cylinder_fn,
            "cone": cone_fn,
            "bsplinecurve": bsplinecurve_fn,
            "linearcurve": linearcurve_fn,
            "run": shap_e_fn,
            "box": box_fn,
            "_from_minecraft_cuboid": _from_minecraft_cuboid_fn,
        }.get(name, cube_fn)(**kwargs)

    return fn


if not primitive_call.is_implemented:  # FIXME not sure.. otherwise it errors in `_shape_utils.py` under `compute_bbox`
    primitive_call.implement(impl_primitive_call)


if __name__ == "__main__":
    # print(compute_normalization(unit_cube()))
    # print(compute_normalization(unit_sphere()))
    # compute_normalization(create_shape(unit_sphere(), scale_matrix(2)), verbose=True)

    # mat = np.eye(4)
    # mat[1, 3] = 1
    # toy_shape = [{'type': 'cube', 'to_world': mat}]
    # for k in ['table']: #SCENE_PRESETS.keys():
    #     execute_from_preset(toy_shape, (Path(__file__).parent.parent / 'outputs/mi_helper' / k).as_posix(), k)
    pass
