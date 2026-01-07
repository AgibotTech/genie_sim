# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import json
import os
import sys

import cv2
import numpy as np
import open3d as o3d

from common.base_utils.transform_utils import get_pose_wxyz as get_pose
from common.base_utils.transform_utils import world_to_camera

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(project_root)


def undistort(img, K, D):
    h, w = img.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0)
    mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, new_K, (w, h), cv2.CV_32FC1)
    return cv2.remap(img, mapx, mapy, cv2.INTER_NEAREST), new_K


def get_semantic_dict_by_msgs(label_dict: dict, msgs: dict):
    index = 0
    camera_label_dict = {"hand_left": None, "hand_right": None, "head": None}
    for camera_name, camera_msgs in msgs.items():
        if not len(camera_msgs):
            continue
        for label_msg in camera_msgs:
            origin_label_dict = json.loads(label_msg)
            for key, value in origin_label_dict.items():
                if isinstance(value, dict) and "class" in value and "_body" in value["class"]:
                    origin_label_dict[key]["class"] = value["class"].replace("_body", "")
            for key, value in origin_label_dict.items():
                if key == "time_stamp":
                    continue
                semantic = next(iter(value.values()))
                existing = False
                for entry in label_dict.values():
                    if entry["class"] == semantic:
                        existing = True
                        break
                if not existing:
                    label_dict[str(index)] = {"class": semantic}
                    index += 1
            if camera_label_dict[camera_name] is None:
                camera_label_dict[camera_name] = origin_label_dict
            else:
                camera_label_dict[camera_name].update(origin_label_dict)
        camera_label_dict[camera_name].pop("time_stamp", None)
    return label_dict, camera_label_dict


def get_group_id_by_semantic(label_dict, semantic):
    for key, value in label_dict.items():
        if value["class"] == semantic:
            return int(key) - 2
    return 0


def get_semantic_by_group_id(label_dict, id):
    for key, value in label_dict.items():
        if id + 2 == int(key):
            return value["class"]
    return ""


def get_class_id_by_semantic(semantic):
    if "background" in semantic:
        return 0
    elif "barcode" in semantic:
        return 10
    elif "qrcode" in semantic:
        return 11
    elif "robot" in semantic:
        return 1
    else:
        return 2


def get_global_id_by_camera_id(global_dict, camera_dict, camera_id):
    if str(camera_id) in camera_dict:
        semantic = camera_dict[str(camera_id)]["class"]
        for key, value in global_dict.items():
            if value["class"] == semantic:
                return int(key)
    return 0


def get_camera_id_by_global_id(global_dict, camera_dict, global_id):
    semantic = global_dict[str(global_id)]["class"]
    for key, value in camera_dict.items():
        if value["class"] == semantic:
            return int(key)
    return None


def label_to_color(img, label_dict, depth_dir, file_name, camera_label_dict):
    num_classes = len(label_dict)
    unique_labels = np.unique(img[~np.isnan(img)]).astype(int)
    label_num = np.arange(num_classes, dtype=int)
    hsv_colors = np.zeros((num_classes, 3), dtype=np.uint8)
    hsv_colors[:, 0] = np.linspace(0, 180, num_classes, endpoint=False)
    hsv_colors[:, 1] = 255
    hsv_colors[:, 2] = 200
    bgr_colors = cv2.cvtColor(hsv_colors[np.newaxis, :, :], cv2.COLOR_HSV2BGR)[0]
    color_map = {label: color for label, color in zip(label_num, bgr_colors)}
    colored = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for label in unique_labels:
        mask = img == label
        transformed_id = get_global_id_by_camera_id(
            label_dict, camera_label_dict[file_name.rsplit("_", 1)[0]], label
        )
        colored[mask] = color_map[int(transformed_id)]
    id_to_polys = dict()
    for color_label, labels in label_dict.items():
        semantic = next(iter(labels.values()))
        if color_label == "time_stamp" or semantic == "UNLABELLED" or semantic == "BACKGROUND":
            continue
        camera_label = get_camera_id_by_global_id(
            label_dict, camera_label_dict[file_name.rsplit("_", 1)[0]], color_label
        )
        if camera_label is None:
            continue
        mask = (img == camera_label).astype(np.uint8) * 255
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            id_to_polys[semantic] = {"polys": {"hierarchy": [], "poly": []}}
            holes = []
            current_index = 0
            while current_index != -1:
                poly = contours[current_index]
                arc_len = cv2.arcLength(poly, True)
                if arc_len > 10:
                    parent_idx = hierarchy[0][current_index][3]
                    id_to_polys[semantic]["polys"]["hierarchy"].append(
                        {"parent": int(parent_idx), "self_index": int(current_index)}
                    )
                    id_to_polys[semantic]["polys"]["poly"].append(poly)
                    current_child_index = hierarchy[0][current_index][2]
                    while current_child_index != -1:
                        poly = contours[current_child_index]
                        arc_len = cv2.arcLength(poly, True)
                        if arc_len > 10:
                            holes.append(poly)
                        current_child_index = hierarchy[0][current_child_index][0]
                current_index = hierarchy[0][current_index][0]
    return colored, id_to_polys


def get_sort_indices(points):
    points_array = np.array([p for p in points])
    y_coords = -points_array[:, 1]
    x_coords = points_array[:, 0]
    sorted_indices = np.lexsort((x_coords, y_coords))
    top_two = sorted_indices[:2]
    bottom_two = sorted_indices[2:]

    return np.concatenate(
        [
            top_two[np.argsort(points_array[top_two, 0])],
            bottom_two[np.argsort(points_array[bottom_two, 0])],
        ]
    )


def change_bbox_order(part_bbox, single_frame_state, camera_file):
    camera_matrix_pose = single_frame_state["cameras"][camera_file]["pose"]
    camera_part_bbox = world_to_camera(part_bbox, camera_matrix_pose)
    order = get_sort_indices(camera_part_bbox)
    return [part_bbox[i] for i in order]


def get_polys_bounding(polys):
    min_x = np.inf
    max_x = -np.inf
    min_y = np.inf
    max_y = -np.inf
    if polys == []:
        return [0, 0, 0, 0]
    for poly in polys:
        points = poly[:, 0, :]

        min_x = min(min_x, np.min(points[:, 0]))
        max_x = max(max_x, np.max(points[:, 0]))
        min_y = min(min_y, np.min(points[:, 1]))
        max_y = max(max_y, np.max(points[:, 1]))

    return [int(min_x), int(min_y), int(max_x), int(max_y)]


def generate_pointcloud(
    save_dir,
    camera_name,
    robot_init_position,
    robot_init_rotation,
    camera_info,
    depth_image_file,
    rgb_image_file=None,
    camera_pose=None,
):
    ## rgb : colorful pointcloud, camera_pose : transform camera axis to robor base axis
    scale_factor = 1
    depth = cv2.imread(depth_image_file, cv2.IMREAD_UNCHANGED)
    depth_small = cv2.resize(
        depth,
        None,
        fx=scale_factor,
        fy=scale_factor,
        interpolation=cv2.INTER_NEAREST,
    )
    if rgb_image_file is not None:
        rgb = cv2.imread(rgb_image_file)
        rgb_small = cv2.resize(
            rgb,
            None,
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_LINEAR,
        )
        rgb_small = cv2.cvtColor(rgb_small, cv2.COLOR_BGR2RGB)
    else:
        rgb_small = None
    d = camera_info[camera_name]["intrinsic"]
    K = np.array(
        [
            [d["fx"] * scale_factor, 0, d["ppx"] * scale_factor],
            [0, d["fy"] * scale_factor, d["ppy"] * scale_factor],
            [0, 0, 1],
        ],
        np.float32,
    )
    D = np.array([0.0, 0.0, 0.0, 0.0, 0.0], np.float32)
    depth_ud, new_K = undistort(depth_small, K, D)
    rgb_ud = undistort(rgb_small, K, D)[0] if rgb_small is not None else None
    if camera_name == "head":
        scale, dmax = 1000.0, 1000
    else:
        scale, dmax = 10000.0, 10000
    mask = (depth_ud > 0) & (depth_ud < dmax)
    v, u = np.where(mask)
    Z = depth_ud[v, u].astype(np.float32) / scale
    X = (u - new_K[0, 2]) * Z / new_K[0, 0]
    Y = (v - new_K[1, 2]) * Z / new_K[1, 1]
    pts = np.vstack((X, Y, Z)).T
    cols = (rgb_ud[v, u] / 255.0) if rgb_ud is not None else None

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    if cols is not None:
        cloud.colors = o3d.utility.Vector3dVector(cols)

    # Write PCD
    if camera_pose is not None:
        robot_pose = get_pose(robot_init_position, robot_init_rotation)
        robot_pose_inv = np.linalg.inv(robot_pose)
        transform_cam_to_robot = robot_pose_inv @ camera_pose
        cloud.transform(transform_cam_to_robot)
    out_pcd = f"{save_dir}/{camera_name}_pointcloud.pcd"
    o3d.io.write_point_cloud(out_pcd, cloud, write_ascii=False, compressed=True)
