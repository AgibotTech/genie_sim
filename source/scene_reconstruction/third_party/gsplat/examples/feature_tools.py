# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""
Code that uses the hierarchical localization toolbox (hloc)
to extract and match image features, estimate camera poses,
and do sparse reconstruction.
Requires hloc module from : GitHub - cvg/Hierarchical-Localization: Visual localization made easy with hloc
"""

# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from pathlib import Path
from typing import Literal
from rich.console import Console
import argparse
from enum import Enum

CONSOLE = Console(width=120)
# from nerfstudio.process_data.process_data_utils import CameraModel
# from nerfstudio.utils.rich_utils import CONSOLE

class CameraModel(Enum):
    """Enum for camera types."""

    SIMPLE_PINHOLE = "SIMPLE_PINHOLE"
    PINHOLE = "PINHOLE"
    SIMPLE_RADIAL = "SIMPLE_RADIAL"
    OPENCV = "OPENCV"
    OPENCV_FISHEYE = "OPENCV_FISHEYE"
    EQUIRECTANGULAR = "EQUIRECTANGULAR"

def run_hloc(
    image_dir: Path,
    colmap_dir: Path,
    camera_model: CameraModel,
    verbose: bool = False,
    matching_method: Literal["vocab_tree", "exhaustive", "sequential"] = "vocab_tree",
    feature_type: Literal[
        "sift", "superpoint_aachen", "superpoint_max", "superpoint_inloc", "r2d2", "d2net-ss", "sosnet", "disk"
    ] = "superpoint_aachen",
    matcher_type: Literal[
        "superglue",
        "superglue-fast",
        "NN-superpoint",
        "NN-ratio",
        "NN-mutual",
        "adalam",
        "disk+lightglue",
        "superpoint+lightglue",
    ] = "superglue",
    num_matched: int = 50,
    refine_pixsfm: bool = False,
    use_single_camera_mode: bool = True,
) -> None:
    """Runs hloc on the images.

    Args:
        image_dir: Path to the directory containing the images.
        colmap_dir: Path to the output directory.
        camera_model: Camera model to use.
        gpu: If True, use GPU.
        verbose: If True, logs the output of the command.
        matching_method: Method to use for matching images.
        feature_type: Type of visual features to use.
        matcher_type: Type of feature matcher to use.
        num_matched: Number of image pairs for loc.
        refine_pixsfm: If True, refine the reconstruction using pixel-perfect-sfm.
        use_single_camera_mode: If True, uses one camera for all frames. Otherwise uses one camera per frame.
    """

    try:
        # TODO(1480) un-hide pycolmap import
        import pycolmap
        from hloc import (  # type: ignore
            extract_features,
            match_features,
            pairs_from_exhaustive,
            pairs_from_retrieval,
            reconstruction,
        )
    except ImportError:
        _HAS_HLOC = False
    else:
        _HAS_HLOC = True

    try:
        from pixsfm.refine_hloc import PixSfM  # type: ignore
    except ImportError:
        _HAS_PIXSFM = False
    else:
        _HAS_PIXSFM = True

    if not _HAS_HLOC:
        CONSOLE.print(
            f"[bold red]Error: To use this set of parameters ({feature_type}/{matcher_type}/hloc), "
            "you must install hloc toolbox!!"
        )
        sys.exit(1)

    if refine_pixsfm and not _HAS_PIXSFM:
        CONSOLE.print("[bold red]Error: use refine_pixsfm, you must install pixel-perfect-sfm toolbox!!")
        sys.exit(1)

    outputs = colmap_dir
    sfm_pairs = outputs / "pairs-netvlad.txt"
    sfm_dir = outputs / "sparse" / "0"
    features = outputs / "features.h5"
    matches = outputs / "matches.h5"

    retrieval_conf = extract_features.confs["netvlad"]  # type: ignore
    feature_conf = extract_features.confs[feature_type]  # type: ignore
    matcher_conf = match_features.confs[matcher_type]  # type: ignore

    references = [p.relative_to(image_dir).as_posix() for p in image_dir.iterdir()]
    extract_features.main(feature_conf, image_dir, image_list=references, feature_path=features)  # type: ignore
    if matching_method == "exhaustive":
        pairs_from_exhaustive.main(sfm_pairs, image_list=references)  # type: ignore
    else:
        retrieval_path = extract_features.main(retrieval_conf, image_dir, outputs)  # type: ignore
        num_matched = min(len(references), num_matched)
        pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=num_matched)  # type: ignore
    match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)  # type: ignore
    image_options = pycolmap.ImageReaderOptions(camera_model=camera_model.value)  # type: ignore

    if use_single_camera_mode:  # one camera per all frames
        camera_mode = pycolmap.CameraMode.SINGLE  # type: ignore
    else:  # one camera per frame
        camera_mode = pycolmap.CameraMode.PER_IMAGE  # type: ignore

    if refine_pixsfm:
        sfm = PixSfM(  # type: ignore
            conf={
                "dense_features": {"use_cache": True},
                "KA": {"dense_features": {"use_cache": True}, "max_kps_per_problem": 1000},
                "BA": {"strategy": "costmaps"},
            }
        )
        refined, _ = sfm.reconstruction(
            sfm_dir,
            image_dir,
            sfm_pairs,
            features,
            matches,
            image_list=references,
            camera_mode=camera_mode,  # type: ignore
            image_options=image_options,
            verbose=verbose,
        )
        print("Refined", refined.summary())

    else:
        reconstruction.main(  # type: ignore
            sfm_dir,
            image_dir,
            sfm_pairs,
            features,
            matches,
            camera_mode=camera_mode,  # type: ignore
            image_options=image_options,
            verbose=verbose,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--colmap_dir", type=Path, required=True)
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--feature_type", type=str, required=True)
    parser.add_argument("--matcher_type", type=str, required=True)
    args = parser.parse_args()

    image_dir: Path = Path(args.image_dir)
    colmap_dir: Path = Path(args.colmap_dir)
    camera_model = CameraModel.PINHOLE
    verbose = False
    matching_method = "vocab_tree"
    feature_type = args.feature_type
    matcher_type = args.matcher_type
    refine_pixsfm = False
    use_single_camera_mode = True

    run_hloc(image_dir=image_dir,
             colmap_dir=colmap_dir,
             camera_model=camera_model,
             verbose=verbose,
             matching_method=matching_method,
             feature_type=feature_type,
             matcher_type=matcher_type,
             refine_pixsfm=refine_pixsfm,
             use_single_camera_mode=use_single_camera_mode)
