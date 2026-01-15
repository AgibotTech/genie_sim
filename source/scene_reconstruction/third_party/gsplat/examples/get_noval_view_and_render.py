# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os
import sys
import struct
import shutil
from typing import List, Dict, Tuple, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from PIL import Image
import torch
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from gsplat.rendering import rasterization
from torch import Tensor

sys.path.insert(0, os.path.join(project_root, "examples", "gsplat"))
from utils import rgb_to_sh


def read_images_binary_as_c2w(
    filepath: str,
) -> Tuple[List[Dict], List[int]]:
    images = {}

    with open(filepath, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]

        for _ in range(num_images):
            image_id = struct.unpack("<i", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<dddd", f.read(8 * 4))
            tx, ty, tz = struct.unpack("<ddd", f.read(8 * 3))
            camera_id = struct.unpack("<i", f.read(4))[0]

            name_bytes = bytearray()
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_bytes += c
            image_name = name_bytes.decode("utf-8")

            num_points2D = struct.unpack("<Q", f.read(8))[0]
            f.seek(num_points2D * 24, 1)

            images[image_id] = {
                "qvec_w2c": np.array([qw, qx, qy, qz]),
                "tvec_w2c": np.array([tx, ty, tz]),
                "camera_id": camera_id,
                "name": image_name,
            }

    poses_c2w: List[Dict] = []
    image_ids = sorted(images.keys())

    for img_id in image_ids:
        img = images[img_id]
        qw, qx, qy, qz = img["qvec_w2c"]
        tx, ty, tz = img["tvec_w2c"]

        q_w2c = R.from_quat([qx, qy, qz, qw])
        R_w2c = q_w2c.as_matrix()
        t_w2c = np.array([tx, ty, tz]).reshape((3, 1))

        T_w2c = np.eye(4)
        T_w2c[:3, :3] = R_w2c
        T_w2c[:3, 3:] = t_w2c

        T_c2w = np.linalg.inv(T_w2c)
        R_c2w = T_c2w[:3, :3]
        t_c2w = T_c2w[:3, 3]

        q_c2w = R.from_matrix(R_c2w).as_quat()
        poses_c2w.append(
            {
                "qvec": [q_c2w[3], q_c2w[0], q_c2w[1], q_c2w[2]],
                "tvec": t_c2w.tolist(),
                "name": img["name"],
                "camera_id": img["camera_id"],
            }
        )

    return poses_c2w, image_ids


def interpolate_poses(
    poses_c2w: List[Dict],
    interp_per_segment: int,
) -> List[Dict]:
    if len(poses_c2w) < 2:
        return poses_c2w.copy()

    interp_results: List[Dict] = []

    for i in range(len(poses_c2w) - 1):
        data0 = poses_c2w[i]
        data1 = poses_c2w[i + 1]

        q0_xyzw = [data0["qvec"][1], data0["qvec"][2], data0["qvec"][3], data0["qvec"][0]]
        q1_xyzw = [data1["qvec"][1], data1["qvec"][2], data1["qvec"][3], data1["qvec"][0]]
        t0 = np.array(data0["tvec"], dtype=float)
        t1 = np.array(data1["tvec"], dtype=float)

        key_times = [0.0, 1.0]
        key_rots = R.from_quat([q0_xyzw, q1_xyzw])
        slerp = Slerp(key_times, key_rots)

        alphas = np.linspace(0.0, 1.0, interp_per_segment, endpoint=False)

        for alpha in alphas:
            q_interp = slerp([alpha])[0]
            t_interp = (1.0 - alpha) * t0 + alpha * t1

            q_interp_xyzw = q_interp.as_quat()
            q_interp_wxyz = [
                q_interp_xyzw[3],
                q_interp_xyzw[0],
                q_interp_xyzw[1],
                q_interp_xyzw[2],
            ]

            interp_results.append(
                {
                    "qvec": q_interp_wxyz,
                    "tvec": t_interp.tolist(),
                    "name": data0["name"],
                    "camera_id": data0["camera_id"],
                }
            )

    interp_results.append(poses_c2w[-1])

    return interp_results


def downsample_poses(
    poses_c2w: List[Dict],
    interval: int,
) -> List[Dict]:
    if interval <= 0:
        raise ValueError(f"Downsample interval must be greater than 0, got: {interval}")

    if len(poses_c2w) <= 1:
        return poses_c2w.copy()

    sampled_indices = list(range(0, len(poses_c2w), interval))

    if sampled_indices[-1] != len(poses_c2w) - 1:
        sampled_indices.append(len(poses_c2w) - 1)

    sampled_indices = sorted(list(set(sampled_indices)))

    sampled_poses = [poses_c2w[i] for i in sampled_indices]

    return sampled_poses


def expand_with_axis_rotations(
    poses_c2w: List[Dict],
    x_rot_deg_list,
    rot_deg_list,
    rot_axis: str,
) -> List[Dict]:
    if rot_axis not in ("x", "y", "z"):
        raise ValueError(f"Invalid rot_axis: {rot_axis}, only supports x / y / z")

    expanded: List[Dict] = []

    for data in poses_c2w:
        qw, qx, qy, qz = data["qvec"]
        tx, ty, tz = data["tvec"]

        R_c2w = R.from_quat([qx, qy, qz, qw]).as_matrix()
        t_c2w = np.array([tx, ty, tz], dtype=float)

        base_name, ext = os.path.splitext(data["name"])
        if not ext:
            ext = ".jpg"

        for x_rot_deg in x_rot_deg_list:
            for rot_deg in rot_deg_list:
                R_new = R_c2w.copy()

                if abs(x_rot_deg) > 1e-8:
                    R_x = R.from_euler("x", x_rot_deg, degrees=True).as_matrix()
                    R_new = R_new @ R_x

                if abs(rot_deg) > 1e-8:
                    R_delta = R.from_euler(rot_axis, rot_deg, degrees=True).as_matrix()
                    R_new = R_new @ R_delta

                q_new = R.from_matrix(R_new).as_quat()
                q_new_wxyz = [q_new[3], q_new[0], q_new[1], q_new[2]]

                x_rot_suffix = f"{x_rot_deg}".replace("-", "neg")
                rot_suffix = f"{rot_deg}".replace("-", "neg")
                new_name = f"{base_name}_xrot{x_rot_suffix}_rot{rot_axis}{rot_suffix}{ext}"

                expanded.append(
                    {
                        "qvec": q_new_wxyz,
                        "tvec": t_c2w.tolist(),
                        "name": new_name,
                        "camera_id": data["camera_id"],
                    }
                )

    return expanded


def write_images_binary(filepath: str, poses_c2w: List[Dict]):
    with open(filepath, "wb") as f:
        f.write(struct.pack("<Q", len(poses_c2w)))

        for idx, data in enumerate(poses_c2w):
            image_id = idx + 1
            qw, qx, qy, qz = data["qvec"]
            tx, ty, tz = data["tvec"]
            camera_id = int(data.get("camera_id", 1))
            name = data["name"]

            R_c2w = R.from_quat([qx, qy, qz, qw]).as_matrix()
            t_c2w = np.array([tx, ty, tz], dtype=float).reshape(3, 1)

            T_c2w = np.eye(4)
            T_c2w[:3, :3] = R_c2w
            T_c2w[:3, 3:] = t_c2w

            T_w2c = np.linalg.inv(T_c2w)
            R_w2c = T_w2c[:3, :3]
            t_w2c = T_w2c[:3, 3]

            q_w2c = R.from_matrix(R_w2c).as_quat()
            qw_final, qx_final, qy_final, qz_final = (
                q_w2c[3],
                q_w2c[0],
                q_w2c[1],
                q_w2c[2],
            )

            f.write(struct.pack("<i", image_id))
            f.write(struct.pack("<dddd", qw_final, qx_final, qy_final, qz_final))
            f.write(struct.pack("<ddd", t_w2c[0], t_w2c[1], t_w2c[2]))
            f.write(struct.pack("<i", camera_id))
            f.write(name.encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", 0))


def load_ply_to_splats(
    ply_path: str,
    sh_degree: int = 3,
    device: str = "cuda",
    f_rest_format: str = "channel",
):
    try:
        from plyfile import PlyData
    except ImportError:
        raise ImportError("Please install plyfile: pip install plyfile")

    plydata = PlyData.read(ply_path)
    vertex_element = plydata["vertex"]
    vertex = vertex_element.data

    print(f"PLY file fields: {vertex.dtype.names}")

    means = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)

    quats = np.stack(
        [vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]], axis=1
    ).astype(np.float32)

    scales = np.stack(
        [vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=1
    ).astype(np.float32)

    opacities = vertex["opacity"].astype(np.float32)

    f_dc_raw = np.stack(
        [vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1
    ).astype(np.float32)
    f_dc_rgb = 1.0 / (1.0 + np.exp(-f_dc_raw))
    f_dc_tensor = torch.from_numpy(f_dc_rgb).float()
    f_dc_sh = rgb_to_sh(f_dc_tensor).numpy()
    f_dc = f_dc_sh

    num_sh_coefs = (sh_degree + 1) ** 2 - 1
    total_f_rest = num_sh_coefs * 3

    f_rest_list = []
    for idx in range(total_f_rest):
        f_rest_list.append(vertex[f"f_rest_{idx}"])
    f_rest = np.stack(f_rest_list, axis=1).astype(np.float32)

    if f_rest_format == "coef":
        f_rest = f_rest.reshape(-1, num_sh_coefs, 3)
    elif f_rest_format == "channel":
        f_rest = f_rest.reshape(-1, 3, num_sh_coefs).transpose(0, 2, 1)
    else:
        raise ValueError(f"Unknown f_rest_format: {f_rest_format}")

    sh0 = f_dc[:, None, :]
    shN = f_rest

    splats = {
        "means": torch.nn.Parameter(torch.from_numpy(means).to(device)),
        "quats": torch.nn.Parameter(torch.from_numpy(quats).to(device)),
        "scales": torch.nn.Parameter(torch.from_numpy(scales).to(device)),
        "opacities": torch.nn.Parameter(torch.from_numpy(opacities).to(device)),
        "sh0": torch.nn.Parameter(torch.from_numpy(sh0).to(device)),
        "shN": torch.nn.Parameter(torch.from_numpy(shN).to(device)),
    }

    return splats


def create_intrinsic_matrix(fovx, fovy, width, height):
    fx = width / (2 * np.tan(fovx / 2))
    fy = height / (2 * np.tan(fovy / 2))
    cx = width / 2
    cy = height / 2
    K = np.array(
        [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    return K


def rasterize_splats_simple(
    splats: Dict[str, torch.nn.Parameter],
    camtoworlds: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    sh_degree: int = 3,
    device: str = "cuda",
    **kwargs,
):
    means = splats["means"]
    quats = splats["quats"]
    scales = torch.exp(splats["scales"])
    opacities = torch.sigmoid(splats["opacities"])

    colors = torch.cat([splats["sh0"], splats["shN"]], 1)

    render_colors, render_alphas, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=torch.linalg.inv(camtoworlds),
        Ks=Ks,
        width=width,
        height=height,
        packed=True,
        absgrad=False,
        sparse_grad=False,
        rasterize_mode="classic",
        distributed=False,
        camera_model="pinhole",
        sh_degree=sh_degree,
        **kwargs,
    )

    return render_colors, render_alphas, info


def write_pinhole_cameras_binary(
    filepath: str,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    camera_id: int = 1,
):
    model_id = 1
    num_cameras = 1
    params = (float(fx), float(fy), float(cx), float(cy))

    with open(filepath, "wb") as f:
        f.write(struct.pack("<Q", num_cameras))
        f.write(struct.pack("<I", int(camera_id)))
        f.write(struct.pack("<i", model_id))
        f.write(struct.pack("<Q", int(width)))
        f.write(struct.pack("<Q", int(height)))
        f.write(struct.pack("<" + "d" * len(params), *params))


def parse_float_list(s: str) -> List[float]:
    if s is None:
        return None
    return [float(x.strip()) for x in s.split(",")]


def main(
    dataset_path: str,
    images_bin_path: str,
    checkpt_ply_path: str,
    points3d_ply_path: Optional[str] = None,
    use_downsample: bool = False,
    downsample_interval: int = 2,
    interp_per_segment: int = 1,
    rot_deg_list: Optional[List[float]] = None,
    rot_axis: str = "y",
    x_rot_deg_list: Optional[List[float]] = None,
    width: int = 1600,
    height: int = 1600,
    fovx_deg: float = 90.0,
    fovy_deg: float = 90.0,
    if_save_novel_image: bool = True,
    if_render_image: bool = False,
    batch_size: int = 16,
):
    if rot_deg_list is None:
        rot_deg_list = [30, 60, 0, -30, -60]
    if x_rot_deg_list is None:
        x_rot_deg_list = [-60, -30, 0, 30, 60]

    dataset_path = os.path.abspath(dataset_path)
    images_bin_path = os.path.abspath(images_bin_path)
    checkpt_ply_path = os.path.abspath(checkpt_ply_path)

    sparse_dir = os.path.join(dataset_path, "sparse", "0")
    output_images_novel_bin = os.path.join(sparse_dir, "images.bin")
    output_novel_image_dir = os.path.join(dataset_path, "images_novel_view")

    os.makedirs(sparse_dir, exist_ok=True)
    os.makedirs(output_novel_image_dir, exist_ok=True)

    if points3d_ply_path is not None:
        points3d_ply_path = os.path.abspath(points3d_ply_path)
        if not os.path.exists(points3d_ply_path):
            raise FileNotFoundError(f"points3D.ply file not found: {points3d_ply_path}")
        dst_points3d_path = os.path.join(sparse_dir, "points3D.ply")
        print(f"Copying points3D.ply from {points3d_ply_path} to {dst_points3d_path}")
        shutil.copy2(points3d_ply_path, dst_points3d_path)
        print(f"Copied points3D.ply to {dst_points3d_path}")

    print(f"Reading original images.bin: {images_bin_path}")
    poses_c2w, image_ids = read_images_binary_as_c2w(images_bin_path)
    print(f"Read {len(poses_c2w)} original poses (sorted by image_id)")

    if use_downsample:
        processed_poses = downsample_poses(poses_c2w, downsample_interval)
        print(
            f"Downsampling completed: interval={downsample_interval}, "
            f"from {len(poses_c2w)} poses to {len(processed_poses)} poses"
        )
    elif interp_per_segment > 1:
        processed_poses = interpolate_poses(poses_c2w, interp_per_segment)
        print(f"Interpolation completed, total poses: {len(processed_poses)}")
    else:
        processed_poses = poses_c2w.copy()
        print(f"Using original poses, total: {len(processed_poses)}")

    expanded = expand_with_axis_rotations(
        processed_poses,
        x_rot_deg_list=x_rot_deg_list,
        rot_deg_list=rot_deg_list,
        rot_axis=rot_axis,
    )
    print(
        f"Rotation expansion completed: each pose expanded by {len(x_rot_deg_list)}×{len(rot_deg_list)}="
        f"{len(x_rot_deg_list) * len(rot_deg_list)} poses, total: {len(expanded)}"
    )

    print(f"Writing new images_novel.bin to: {output_images_novel_bin}")
    write_images_binary(output_images_novel_bin, expanded)
    print(f"Saved {len(expanded)} poses to {output_images_novel_bin}")

    poses_mats: List[np.ndarray] = []
    image_names: List[str] = []
    for data in expanded:
        qw, qx, qy, qz = data["qvec"]
        tx, ty, tz = data["tvec"]

        q = R.from_quat([qx, qy, qz, qw])
        R_c2w = q.as_matrix()
        t_c2w = np.array([tx, ty, tz]).reshape((3, 1))

        T_c2w = np.eye(4, dtype=np.float32)
        T_c2w[:3, :3] = R_c2w
        T_c2w[:3, 3] = t_c2w[:, 0]

        poses_mats.append(T_c2w)
        image_names.append(data["name"])

    total_images = len(poses_mats)
    print(f"Rendering cameras: {total_images}")

    print("Loading PLY file...")
    splats = load_ply_to_splats(
        checkpt_ply_path, sh_degree=3, device="cuda", f_rest_format="channel"
    )
    print(f"Loaded {len(splats['means'])} Gaussian points")

    fovx = np.radians(fovx_deg)
    fovy = np.radians(fovy_deg)
    K = create_intrinsic_matrix(fovx, fovy, width, height)
    K_tensor = torch.from_numpy(K).float().to("cuda")

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    cameras_bin_path = os.path.join(sparse_dir, "cameras.bin")
    print(f"Writing cameras.bin to: {cameras_bin_path}")
    write_pinhole_cameras_binary(
        cameras_bin_path,
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        camera_id=1,
    )

    images_path = os.path.join(dataset_path, "images")

    if if_render_image:
        plt.ion()
        fig, (ax_render, ax_orig) = plt.subplots(1, 2, figsize=(10, 5))

    print(f"Starting batch rendering, batch size: {batch_size}")
    c2w_tensors = [torch.from_numpy(pose).float() for pose in poses_mats]
    K_batch_tensors = [K_tensor for _ in range(total_images)]

    with torch.no_grad():
        for batch_start in range(0, total_images, batch_size):
            batch_end = min(batch_start + batch_size, total_images)
            batch_indices = range(batch_start, batch_end)

            print(f"Rendering batch: {batch_start}-{batch_end - 1} / {total_images}")

            c2w_batch = torch.stack(
                [c2w_tensors[i].to("cuda") for i in batch_indices]
            )
            K_batch = torch.stack([K_batch_tensors[i] for i in batch_indices])

            render_colors, render_alphas, info = rasterize_splats_simple(
                splats=splats,
                camtoworlds=c2w_batch,
                Ks=K_batch,
                width=width,
                height=height,
                sh_degree=3,
                device="cuda",
            )

            for idx, i in enumerate(batch_indices):
                name = image_names[i]
                img_path = os.path.join(images_path, name)

                render_colors_clamped = torch.clamp(
                    render_colors[idx].detach(), 0.0, 1.0
                )
                render_np = (render_colors_clamped.cpu().numpy() * 255).astype(
                    np.uint8
                )

                if if_save_novel_image:
                    save_path = os.path.join(output_novel_image_dir, name)
                    Image.fromarray(render_np).save(save_path)

                if if_render_image:
                    if os.path.exists(img_path):
                        orig_img = Image.open(img_path)
                        orig_np = np.array(orig_img)
                    else:
                        orig_np = np.zeros_like(render_np)
                        print(f"Warning: original image not found: {img_path}")

                    ax_render.clear()
                    ax_render.imshow(render_np)
                    ax_render.set_title(f"Rendered: {name}")

                    ax_orig.clear()
                    ax_orig.imshow(orig_np)
                    ax_orig.set_title(f"Original: {name}")

                    plt.pause(0.00001)

    print("Rendering completed!")

    if if_render_image:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate novel view images.bin and render with Gaussian splats"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Dataset root directory (will create images_novel_view and sparse/0)",
    )
    parser.add_argument(
        "--images_bin_path",
        type=str,
        required=True,
        help="Input COLMAP images.bin path (original camera poses)",
    )
    parser.add_argument(
        "--checkpt_ply_path",
        type=str,
        required=True,
        help="Trained Gaussian splats PLY file path",
    )
    parser.add_argument(
        "--points3d_ply_path",
        type=str,
        default=None,
        help="points3D.ply file path (optional, will be copied to sparse/0 if provided)",
    )
    parser.add_argument(
        "--use_downsample",
        action="store_true",
        default=False,
        help="Enable downsampling to reduce pose count (mutually exclusive with interpolation)",
    )
    parser.add_argument(
        "--downsample_interval",
        type=int,
        default=2,
        help="Downsampling interval (select every N-th pose, e.g., 2 means select every other pose)",
    )
    parser.add_argument(
        "--interp_per_segment",
        type=int,
        default=1,
        help="Interpolation points per segment (1 means no interpolation, >1 means interpolate between adjacent poses)",
    )
    parser.add_argument(
        "--rot_deg_list",
        type=str,
        default="30,60,0,-30,-60",
        help="Rotation angles in degrees for axis rotation (comma-separated, e.g., '30,60,0,-30,-60')",
    )
    parser.add_argument(
        "--rot_axis",
        type=str,
        default="y",
        choices=["x", "y", "z"],
        help="Rotation axis for additional rotation sampling",
    )
    parser.add_argument(
        "--x_rot_deg_list",
        type=str,
        default="-60,-30,0,30,60",
        help="X-axis rotation angles in degrees for pitch (comma-separated, e.g., '-60,-30,0,30,60')",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1600,
        help="Render image width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1600,
        help="Render image height",
    )
    parser.add_argument(
        "--fovx_deg",
        type=float,
        default=90.0,
        help="Horizontal field of view in degrees",
    )
    parser.add_argument(
        "--fovy_deg",
        type=float,
        default=90.0,
        help="Vertical field of view in degrees",
    )
    parser.add_argument(
        "--if_save_novel_image",
        action="store_true",
        default=True,
        help="Save rendered novel view images",
    )
    parser.add_argument(
        "--no_save_novel_image",
        dest="if_save_novel_image",
        action="store_false",
        help="Do not save rendered novel view images",
    )
    parser.add_argument(
        "--if_render_image",
        action="store_true",
        default=False,
        help="Display rendered images vs original images comparison",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for rendering (number of cameras per batch)",
    )

    args = parser.parse_args()

    rot_deg_list = parse_float_list(args.rot_deg_list)
    x_rot_deg_list = parse_float_list(args.x_rot_deg_list)

    main(
        dataset_path=args.dataset_path,
        images_bin_path=args.images_bin_path,
        checkpt_ply_path=args.checkpt_ply_path,
        points3d_ply_path=args.points3d_ply_path,
        use_downsample=args.use_downsample,
        downsample_interval=args.downsample_interval,
        interp_per_segment=args.interp_per_segment,
        rot_deg_list=rot_deg_list,
        rot_axis=args.rot_axis,
        x_rot_deg_list=x_rot_deg_list,
        width=args.width,
        height=args.height,
        fovx_deg=args.fovx_deg,
        fovy_deg=args.fovy_deg,
        if_save_novel_image=args.if_save_novel_image,
        if_render_image=args.if_render_image,
        batch_size=args.batch_size,
    )
