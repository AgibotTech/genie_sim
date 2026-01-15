# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import numpy as np
from scipy.spatial.transform import Rotation as R
import json
import argparse
from pathlib import Path
import subprocess
import struct
import collections
import os
import glob
import sys
import logging
import sqlite3
import shutil
import json
from typing import Any, Dict, Optional
from datetime import datetime
import os.path as osp
import time
import cv2
from tool import CameraPatchConfig, COLMAPDatabase


gpu_id=0

def log_config(path):
    filename = osp.join(path, 'process.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(filename, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
            ]
    )
    logging.info(args)

def ParseIntrinsic(path):
    with open(os.path.join(path, "info/calibration.json")) as f:
        tfs = json.load(f)
    cam_intri = tfs['cameras'][0]['intrinsic']
    K = np.array([
                [cam_intri['fl_x'], 0, cam_intri['cx']],
                [0, cam_intri['fl_y'], cam_intri['cy']],
                [0, 0, 1]
                 ], dtype=np.float32)
    D = tfs['cameras'][0]['distortion']['params']
    dist_coeffs = np.array([D['k1'], D['k2'], D['k3'], D['k4']])
    return K, dist_coeffs


def compute_intersection(path):
    extract_img_list = os.listdir(os.path.join(path, 'camera/left'))
    with open(path / "transforms.json") as f:
        tfs = json.load(f)
    frames = tfs['frames']
    cam_pose_list = []
    for frame in frames:
        name = GetImageName(frame)[-1]
        cam_pose_list.append(name)
    img_set = set(extract_img_list)
    pose_set = set(cam_pose_list)
    inter_set = img_set.intersection(pose_set)
    diff_list = list(img_set - inter_set)
    inter_list = list(inter_set)
    return inter_list


def is_static_pose(prev_pose, current_pose, trans_thresh=0.1, rot_thresh=0.5):
    relative_pose = np.linalg.inv(prev_pose) @ current_pose
    translation = relative_pose[:3, 3]
    translation_norm = np.linalg.norm(translation)

    rotation_matrix = relative_pose[:3, :3]
    rotation = R.from_matrix(rotation_matrix)
    rotation_vector = rotation.as_rotvec()
    rotation_angle_deg = np.degrees(np.linalg.norm(rotation_vector))
    is_trans_static = translation_norm < trans_thresh
    is_rot_static = rotation_angle_deg < rot_thresh
    logging.info(f"translation: {translation_norm:.6f}m, rotation: {rotation_angle_deg:.3f}°")
    logging.info(f"translate static: {is_trans_static}, rotation static: {is_rot_static}")

    return is_trans_static and is_rot_static


# fisheye fov threshold: 0.5m  17
# static threshold:  0.1m 0.5
def should_keep_frame(prev_pose, current_pose, trans_thresh=0.5, rot_thresh=17):
    relative_pose = np.linalg.inv(prev_pose) @ current_pose

    translation = relative_pose[:3, 3]
    translation_norm = np.linalg.norm(translation)

    rotation_matrix = relative_pose[:3, :3]
    rotation = R.from_matrix(rotation_matrix)
    rotation_vector = rotation.as_rotvec()
    rotation_angle_deg = np.degrees(np.linalg.norm(rotation_vector))
    trans_keep = translation_norm > trans_thresh
    rot_keep = rotation_angle_deg > rot_thresh
    logging.info(f"translation: {translation_norm:.6f}m, rotation: {rotation_angle_deg:.3f}°")
    logging.info(f"translate static: {trans_keep}, rotation static: {rot_keep}")

    return trans_keep or rot_keep

def GetImageName(frame):
    if '/' in frame['file_path']:
        str_name = frame['file_path'].split('/')
    else:
        str_name = frame['file_path'].split('\\')
    return str_name

def FilterFrame(frames, extract_img_list):
    filtered_frames = []
    for (i, frame) in enumerate(frames):
        str_name = GetImageName(frame)
        name = str_name[-1]
        if name not in extract_img_list:
            continue
        if str_name[0] == "right":
            break
        filtered_frames.append(frame)
    flitered_dup = FilterDuplicate(filtered_frames)
    return flitered_dup

def FilterDuplicate(frames):
    filtered_frames = []
    prev_frame = frames[0]
    filtered_frames.append(prev_frame)
    for (i, frame) in enumerate(frames):
        if (i+1) == len(frames):
            break
        next_frame = frames[i+1]
        next_frame_pose = np.array(next_frame['transform_matrix'])
        prev_pose = np.array(prev_frame['transform_matrix'])
        if should_keep_frame(prev_pose, next_frame_pose):
            filtered_frames.append(next_frame)
            prev_frame = next_frame
    return filtered_frames

def AddPerFrameInfo(pose, db, img_id, cid, name, image_info, vrot):
    qw, qx, qy, qz, tx, ty, tz = priorpose2colmapformat(pose, vrot)
    image_info += f'{img_id+1} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {cid} {name}\n\n'
    db.add_image(name, cid, np.array((qw, qx, qy, qz)), np.array((tx, ty, tz)), img_id+1)
    return image_info

def priorpose2colmapformat(c2w, vrot):

    w2c = np.identity(4)
    w2c[0:3, 0:3] = c2w[0:3, 0:3].T
    w2c[0:3, 3] = - c2w[0:3, 0:3].T @ c2w[0:3, 3]
    # CG 2 CV
    w2c[1:3] *= -1
    w2c[0:3, 0:3] = vrot @ w2c[0:3, 0:3]
    qx, qy, qz, qw = R.from_matrix(w2c[0:3, 0:3]).as_quat()
    tx, ty, tz = w2c[0:3, 3]
    return qw, qx, qy, qz, tx, ty, tz

def CreateDirectory(path):
    colmap_path = (path / 'colmap' / 'sparse' / '0')
    colmap_output_path = (path / 'sparse')
    colmap_path.mkdir(parents=True, exist_ok=True)
    colmap_output_path.mkdir(parents=True, exist_ok=True)
    (colmap_path / 'points3D.txt').touch()

def GenColmapDataFormat(path):
    # compute intersection images
    image_out_path = osp.join(path, 'images')
    if not os.path.exists(image_out_path):
        os.makedirs(image_out_path, exist_ok=True)
    extract_img_list = compute_intersection(path)
    with open(path / "transforms.json") as f:
        tfs = json.load(f)
    frames = tfs['frames']

    colmap_path = (path / 'colmap' / 'sparse' / '0')
    colmap_output_path = (path / 'sparse' / '0')
    log_path = (path / 'sparse' / 'log')
    log_path.mkdir(parents=True, exist_ok=True)
    colmap_path.mkdir(parents=True, exist_ok=True)
    colmap_output_path.mkdir(parents=True, exist_ok=True)
    (colmap_path / 'points3D.txt').touch()

    # Database
    db = COLMAPDatabase.connect(path / 'colmap' / 'sparse' / '0' / 'database.db')
    db.create_tables()
    K, dist_coeffs = ParseIntrinsic(path)
    # fisheye undistort
    cam_patch_cfg = CameraPatchConfig()
    mapxs = []
    mapys = []
    cam = {}
    img_id = 0
    image_info = ''
    for i, rot in enumerate(cam_patch_cfg.rotations):
        mapx, mapy = cv2.fisheye.initUndistortRectifyMap(
                        K, dist_coeffs, rot, cam_patch_cfg.ideal_intrinsic,
                        (cam_patch_cfg.width, cam_patch_cfg.height), cv2.CV_32FC1
        )
        mapxs.append(mapx)
        mapys.append(mapy)
    cam['mapx'] = np.array(mapxs)
    cam['mapy'] = np.array(mapys)
    filtered_frames = FilterFrame(frames, extract_img_list)
    print("filtered_frames:  ", len(filtered_frames))
    for (j, frame) in enumerate(filtered_frames):
        pose = np.array(frame['transform_matrix'])
        img_name = GetImageName(frame)
        if img_name[0] == 'right':
            break
        image_in = cv2.imread(osp.join(path, 'camera', img_name[0], img_name[-1]))
        for i in range(cam_patch_cfg.rotations.shape[0]):
            vrot = cam_patch_cfg.rotations[i]
            # undistort the image
            mapx = cam['mapx'][i]
            mapy = cam['mapy'][i]
            out_fname = img_name[-1].split('.')[0] + f"_{i:1d}" + '.png'

            # save camera pose
            image_info = AddPerFrameInfo(pose, db, img_id, i+1, out_fname, image_info, vrot)
            img_id += 1
            # save undistort image
            image = cv2.remap(image_in, mapx, mapy, interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(image_out_path, out_fname), image)
    with open(colmap_path / 'images.txt', 'w') as f:
        f.write(image_info)

    # ideal intrinsic
    with open(colmap_path / 'cameras.txt', 'w') as f:
        intr = cam_patch_cfg.ideal_intrinsic
        h, w = cam_patch_cfg.height, cam_patch_cfg.width
        for i in range(cam_patch_cfg.rotations.shape[0]):
            f.write(f"{i+1} PINHOLE {h} {w} {intr[0][0]} {intr[1][1]} {intr[0][2]} {intr[1][2]}\n")
            db.add_camera(1, w, h, np.array((intr[0][0], intr[1][1], intr[0][2], intr[1][2])))
    db.commit()
    db.close()
    logging.info("Preprocess data is done!")

def feature_extract_match(path):
    cmd = f'python3 feature_tools.py --image_dir {path}/images/ --colmap_dir {path}/colmap \
           --feature_type superpoint_inloc --matcher_type superpoint+lightglue'
    logging.info(cmd)
    exit_code = os.system(cmd)
    if exit_code != 0:
        logging.error(f"Extract feature and match  failed with code {exit_code}. Exiting.")
        exit()
    logging.info('Extract feature and match is done')

def point_data_preprocess(path):
    cmd = f'xvfb-run -a CloudCompare -SILENT   -AUTO_SAVE OFF   -O {path}/colorized.las \
            -COMPUTE_NORMALS -OCTREE_NORMALS 0.1 -MODEL LS -ORIENT_NORMS_MST 6  \
            -SS SPATIAL 0.025   -C_EXPORT_FMT PLY -PLY_EXPORT_FMT BINARY_LE \
             -SAVE_CLOUDS FILE {path}/normal_subsample.ply'
    logging.info(cmd)
    exit_code = os.system(cmd)
    if exit_code != 0:
        logging.error(f"Point data preprocess failed.")
        exit()
    logging.info('Process Point cloud data is done')


def OptimizePose(path, args):

    cmd = f'colmap-pcd mapper \
        --database_path {path}/colmap/sparse/0/database.db \
        --image_path {path}/images \
        --output_path {path}/sparse \
        --Mapper.if_add_lidar_constraint 1 \
        --Mapper.init_image_id1 1 \
        --Mapper.init_image_id2 {args.id2} \
        --Mapper.if_import_pose_prior 1 \
        --Mapper.image_pose_prior_path {path}/colmap/sparse/0/ \
        --Mapper.lidar_pointcloud_path {path}/normal_subsample.ply \
        '

    # block reconstruction
    block_cmd = f'colmap-pcd hierarchical_mapper  \
    --database_path {path}/colmap/sparse/0/database.db \
    --image_path {path}/images \
    --output_path {path}/sparse-gba \
    --Mapper.if_add_lidar_constraint 1 \
    --Mapper.init_image_id1 -1 \
    --Mapper.init_image_id2 -1 \
    --leaf_max_num_images 400 \
    --Mapper.if_import_pose_prior 1 \
    --Mapper.image_pose_prior_path {path}/colmap/sparse/0/ \
    --Mapper.lidar_pointcloud_path {path}/normal_subsample.ply \
    '

    # global ba
    gba_cmd = f'colmap-pcd bundle_adjuster \
                --input_path {path}/sparse-gba/0 \
                --output_path {path}/sparse/0 \
                --BundleAdjustment.max_num_iterations 300'

    if len(os.listdir(f'{path}/images')) > 600:
        os.makedirs(os.path.join(path, 'sparse-gba'), exist_ok=True)
        os.system(block_cmd)
        exit_code = os.system(gba_cmd)
        if exit_code != 0:
            logging.error(f"Calc pose failed!.")
            exit()
    else:
        exit_code = os.system(cmd)
    if exit_code != 0:
        logging.error(f"Calc pose failed!.")
        exit()
    logging.info(f"Calc pose is done!")

def Train(path, args):
    # align image and pose data filter static pictures and convert rotation matrix to quan
    try:
        GenColmapDataFormat(path)
    except Exception as e:
        logging.info(f"generate colmap error: {e}")
        exit()

    # compute point cloud normal and subsample
    point_data_preprocess(path)

    # ss lightglue
    feature_extract_match(path)

    # Optimize prior cam pose
    OptimizePose(path, args)
    shutil.move(os.path.join(path, 'normal_subsample.ply'), os.path.join(path, 'sparse/0/sparse.ply'))
    logging.info("---------------------Reconstructe done!!!---------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some path.')
    parser.add_argument('--path', type=str, help='The path to be processed.')
    parser.add_argument('--id2', type=int, default=13, help='The image id of the first pair image.')
    parser.add_argument('--max_depth', type=float, default=10.0, help='To generate  max depth of the mesh.')
    args = parser.parse_args()
    path = Path(args.path)
    log_dir = osp.join(path, 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_config(log_dir)
    if not osp.exists(args.path):
        logging.info(f'Path {path}not exists.')
        exit()
    Train(path, args)
