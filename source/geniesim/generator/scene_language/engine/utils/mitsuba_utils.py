# import os
# os.environ['MI_DEFAULT_VARIANT'] = 'scalar_rgb'
import mitsuba as mi
from PIL import Image
from pathlib import Path
import numpy as np
from .type_utils import BBox
from typing import Optional

T = mi.scalar_rgb.Transform4f


def create_default_scene_dict() -> dict:
    scene_dict = {"type": "scene", "integrator": {"type": "path", "max_depth": 4}}

    light = {
        "type": "rectangle",
        "to_world": T.translate([0.0, 0.99, 0.01]).rotate([1, 0, 0], 90).scale([0.23, 0.19, 0.19]),
        "emitter": {
            "type": "area",
            "radiance": {
                "type": "rgb",
                "value": [18.387, 13.9873, 6.75357],
            },
        },
    }
    scene_dict["light"] = light

    background = {
        "floor": {
            "type": "rectangle",
            "to_world": T.translate([0.0, -1.0, 0.0]).rotate([1, 0, 0], -90),
        },
        "ceiling": {
            "type": "rectangle",
            "to_world": T.translate([0.0, 1.0, 0.0]).rotate([1, 0, 0], 90),
        },
        "back": {
            "type": "rectangle",
            "to_world": T.translate([0.0, 0.0, -1.0]),
        },
        "right-wall": {
            "type": "rectangle",
            "to_world": T.translate([1.0, 0.0, 0.0]).rotate([0, 1, 0], -90),
        },
        "left-wall": {
            "type": "rectangle",
            "to_world": T.translate([-1.0, 0.0, 0.0]).rotate([0, 1, 0], 90),
        },
    }
    scene_dict.update(background)

    sensor = {
        "type": "perspective",
        "to_world": T.look_at(origin=[0, 2, 4], target=[0, 0, 0], up=[0, 1, 0]),
        # 'sampler': {
        #     'type': 'independent',
        #     'sample_count': 64,
        # },
        # 'film': {
        #     'type': 'hdrfilm',
        #     'width': 1024,
        #     'height': 1024,
        #     'filter': {'type': 'tent'}
        # }
    }
    scene_dict["sensor"] = sensor
    return scene_dict


def set_scene_dict_default(scene_dict: dict) -> dict:
    if "type" in scene_dict:
        assert scene_dict["type"] == "scene", scene_dict["type"]
    else:
        scene_dict["type"] = "scene"
    if "integrator" not in scene_dict:
        scene_dict["integrator"] = {"type": "path", "max_depth": 4}
    # scene_dict = add_eager_shape_template(scene_dict)
    return scene_dict


def compute_bbox(scene_dict: dict) -> BBox:
    boxes = compute_bboxes(scene_dict)
    if len(boxes) == 0:
        box_min = np.ones((3,)) * -0.5
        box_max = np.ones((3,)) * 0.5
    else:
        box_min = np.min([box.min for box in boxes], axis=0)
        box_max = np.max([box.max for box in boxes], axis=0)

    box_center = (box_min + box_max) / 2
    box_sizes = box_max - box_min
    return BBox(center=box_center, sizes=box_sizes, min=box_min, max=box_max, size=float(max(box_sizes)))


def compute_bboxes(scene_dict: dict) -> list[BBox]:
    scene_dict = preprocess_scene_dict(scene_dict)
    boxes = []
    for shape in mi.load_dict(scene_dict).shapes():
        box_min = np.asarray(shape.bbox().min)
        box_max = np.asarray(shape.bbox().max)
        box_center = (box_min + box_max) / 2
        box_sizes = box_max - box_min
        boxes.append(BBox(center=box_center, sizes=box_sizes, min=box_min, max=box_max, size=float(max(box_sizes))))
    return boxes


def set_auto_camera(
    scene_dict: dict, add_sensor: bool = True, add_background: bool = True, add_light: bool = True
) -> dict:
    box = compute_bbox(scene_dict=scene_dict)
    box_center = box.center
    sq_box_size = box.size

    def get_sensor(rel_offset: tuple[float, float, float]) -> dict:
        return {
            "type": "perspective",
            "to_world": T.look_at(
                origin=box_center + sq_box_size * np.asarray(rel_offset), target=box_center, up=[0, 1, 0]
            ),
        }

    if add_sensor:
        scene_dict["auto_sensor"] = get_sensor(rel_offset=(0.0, 2.0, 4.0))
        scene_dict["auto_sensor_from_left"] = get_sensor(rel_offset=(-1.0, 0.5, 1.0))
        scene_dict["auto_sensor_from_right"] = get_sensor(rel_offset=(1.0, 0.5, 1.0))
        scene_dict["auto_sensor_from_up"] = get_sensor(rel_offset=(0, 0.2, 0.2))
        scene_dict["auto_sensor_from_down"] = get_sensor(rel_offset=(0, -1, 1))

    if add_background:
        background = {
            "auto_floor": {
                "type": "rectangle",
                "to_world": T.translate(box_center)
                .translate([0.0, -sq_box_size, 0.0])
                .rotate([1, 0, 0], -90)
                .scale(sq_box_size),
            },
            "auto_ceiling": {
                "type": "rectangle",
                "to_world": T.translate(box_center)
                .translate([0.0, sq_box_size, 0.0])
                .rotate([1, 0, 0], 90)
                .scale(sq_box_size),
            },
            "auto_back_wall": {
                "type": "rectangle",
                "to_world": T.translate(box_center).translate([0.0, 0.0, -sq_box_size]).scale(sq_box_size),
            },
            "auto_right_wall": {
                "type": "rectangle",
                "to_world": T.translate(box_center)
                .translate([sq_box_size, 0.0, 0.0])
                .rotate([0, 1, 0], -90)
                .scale(sq_box_size),
            },
            "auto_left_wall": {
                "type": "rectangle",
                "to_world": T.translate(box_center)
                .translate([-sq_box_size, 0.0, 0.0])
                .rotate([0, 1, 0], 90)
                .scale(sq_box_size),
            },
        }
        scene_dict.update(background)

    if add_light:
        light = {
            "type": "rectangle",
            "to_world": T.translate(box_center)
            .translate([0.0, sq_box_size, 0.0])
            .rotate([1, 0, 0], 90)
            .scale(sq_box_size * 0.3),
            "emitter": {
                "type": "area",
                "radiance": {
                    "type": "rgb",
                    "value": [18.387, 13.9873, 6.75357],
                },
            },
        }
        scene_dict["auto_light"] = light
    return scene_dict


def set_bsdf_refs(scene_dict: dict) -> dict:
    if "white" in scene_dict and "green" in scene_dict and "red" in scene_dict:
        return scene_dict
    temp_scene_dict = {}
    while len(scene_dict) > 0:
        k, v = scene_dict.popitem()
        temp_scene_dict[k] = v

    scene_dict.update(
        {
            "white": {
                "type": "diffuse",
                "reflectance": {
                    "type": "rgb",
                    "value": [0.885809, 0.698859, 0.666422],
                },
            },
            "green": {
                "type": "diffuse",
                "reflectance": {
                    "type": "rgb",
                    "value": [0.105421, 0.37798, 0.076425],
                },
            },
            "red": {
                "type": "diffuse",
                "reflectance": {
                    "type": "rgb",
                    "value": [0.570068, 0.0430135, 0.0443706],
                },
            },
        }
    )

    while len(temp_scene_dict) > 0:
        k, v = temp_scene_dict.popitem()
        scene_dict[k] = v
    return scene_dict


def preprocess_scene_dict(d: dict) -> dict:
    return {
        k: (
            v
            if not (isinstance(v, dict) and v["type"] in ["cube", "sphere"])
            else {kk: vv for kk, vv in v.items() if kk != "info"}
        )
        for k, v in d.items()
    }


def render_scene_dict(
    scene_dict: dict, verbose: bool = True, save_to: Optional[str] = None, sensor: int = 0
) -> Image.Image:
    scene_dict = preprocess_scene_dict(scene_dict)

    scene_dict = set_scene_dict_default(scene_dict)
    scene_dict = set_bsdf_refs(scene_dict)
    scene_dict = set_auto_camera(scene_dict)
    if verbose:
        print(scene_dict)
    scene = mi.load_dict(scene_dict)
    image = mi.render(scene, sensor=sensor)
    # mi.Bitmap(image).write(f'outputs/{filename}.exr')  # debug
    image = mi.util.convert_to_bitmap(image)
    image = Image.fromarray(np.asarray(image))
    if save_to is not None:
        Path(save_to).parent.mkdir(exist_ok=True)
        image.save(save_to)
    # mi.Bitmap(img).write('cbox.exr')
    # im: Float[mi.Bitmap, "H W 3"] = mi.Bitmap(img)

    # mi.write_bitmap('train_primitives.png', image)
    return image


def add_eager_shape_template(scene_dict: dict) -> dict:
    group = {
        "type": "shapegroup",
        "body": {
            "type": "cube",
            "to_world": T(),
        },
    }
    scene_dict["placeholder"] = group

    # move the key `placeholder` to the beginning
    # assuming python > 3.6 so that the dictionary is ordered by default
    for key in list(scene_dict.keys()):
        if key == "placeholder":
            break
        scene_dict[key] = scene_dict.pop(key)  # FIXME order is flipped?
    return scene_dict


def add_eager_shape(
    scene_dict: dict, location: tuple[float, float, float], scale: tuple[float, float, float], apply_gravity_flag: bool
) -> dict:
    shape = {
        "type": "instance",
        "to_world": T.translate(location).scale(scale),
        "shapegroup": {"type": "ref", "id": "placeholder"},
    }
    if apply_gravity_flag:
        aux_scene_dict = {}
        add_eager_shape_template(aux_scene_dict)
        shape = apply_gravity(shape, aux_scene_dict=aux_scene_dict)
    scene_dict[f"placeholder_{len(scene_dict):03d}"] = shape
    return scene_dict


def apply_gravity(shape_dict: dict, aux_scene_dict: dict = None) -> dict:
    assert shape_dict["type"] in ["shape", "instance"], shape_dict
    if aux_scene_dict is None:
        aux_scene_dict = {}
    (sh,) = mi.load_dict({"type": "scene", **aux_scene_dict, "shape": shape_dict}).shapes()
    y_offset = np.asarray(sh.bbox().min)[1]
    shape_dict["to_world"] = T.translate([0, -y_offset, 0]) @ shape_dict["to_world"]
    return shape_dict
