# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import argparse
import os
import sys
from typing import List, Optional

_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument(
    "--difix_src_path",
    type=str,
    default="Difix3D/src",
    help="Path to Difix3D src directory containing pipeline_difix module.",
)
_pre_args, _ = _pre_parser.parse_known_args()

if _pre_args.difix_src_path and os.path.isdir(_pre_args.difix_src_path):
    sys.path.insert(0, _pre_args.difix_src_path)
else:
    default_path = "Difix3D/src"
    if os.path.isdir(default_path):
        sys.path.insert(0, default_path)

from diffusers.utils import load_image
from pipeline_difix import DifixPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch run DifixPipeline with configurable device and IO paths."
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save fixed images.",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="Difix3D/hf_model",
        help="Path to the pretrained Difix model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for inference, e.g. cuda:0, cuda:1, or cpu.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of images per batch (adjust per GPU memory).",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip processing images that already exist in output directory.",
    )
    parser.add_argument(
        "--image_extensions",
        type=str,
        nargs="+",
        default=[".png", ".jpg", ".jpeg"],
        help="Image file extensions to process.",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="remove degradation",
        help="Text prompt for image restoration.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="Negative prompt for image generation.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=1,
        help="Number of denoising steps.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        nargs="+",
        default=[199],
        help="Custom timesteps to use for denoising.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=0.0,
        help="Guidance scale for classifier-free guidance.",
    )
    parser.add_argument(
        "--guidance_rescale",
        type=float,
        default=0.0,
        help="Guidance rescale factor.",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="Eta parameter for DDIM scheduler.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Height of output image. If None, uses model default.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Width of output image. If None, uses model default.",
    )
    parser.add_argument(
        "--ref_image",
        type=str,
        default=None,
        help="Path to reference image for image-to-image generation.",
    )
    parser.add_argument(
        "--difix_src_path",
        type=str,
        default="Difix3D/src",
        help="Path to Difix3D src directory containing pipeline_difix module.",
    )

    return parser.parse_args()


def get_image_files(
    input_dir: str,
    output_dir: str,
    extensions: List[str],
    skip_existing: bool = True,
) -> List[str]:
    valid_files = []
    extensions = [ext.lower() if not ext.startswith(".") else ext.lower() for ext in extensions]

    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")

    for f in sorted(os.listdir(input_dir)):
        if not any(f.lower().endswith(ext) for ext in extensions):
            continue

        if skip_existing:
            out_path = os.path.join(output_dir, f)
            if os.path.exists(out_path):
                continue

        valid_files.append(f)

    return valid_files


def main() -> None:
    args = parse_args()

    pipe = DifixPipeline.from_pretrained(args.model_path, trust_remote_code=True)
    pipe.to(args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    valid_files = get_image_files(
        args.input_dir,
        args.output_dir,
        args.image_extensions,
        args.skip_existing,
    )

    total = len(valid_files)
    if total == 0:
        print("No files to process.")
        return

    ref_image = None
    if args.ref_image:
        ref_image = load_image(args.ref_image)

    for start_idx in range(0, total, args.batch_size):
        batch_files = valid_files[start_idx : start_idx + args.batch_size]

        batch_images = []
        batch_prompts = []
        batch_negative_prompts = []

        for fname in batch_files:
            input_path = os.path.join(args.input_dir, fname)
            batch_images.append(load_image(input_path))
            batch_prompts.append(args.prompt)
            if args.negative_prompt:
                batch_negative_prompts.append(args.negative_prompt)

        pipeline_kwargs = {
            "prompt": batch_prompts,
            "image": batch_images,
            "num_inference_steps": args.num_inference_steps,
            "timesteps": args.timesteps,
            "guidance_scale": args.guidance_scale,
            "guidance_rescale": args.guidance_rescale,
            "eta": args.eta,
        }

        if args.negative_prompt:
            pipeline_kwargs["negative_prompt"] = batch_negative_prompts

        if args.ref_image:
            pipeline_kwargs["ref_image"] = ref_image

        if args.height:
            pipeline_kwargs["height"] = args.height

        if args.width:
            pipeline_kwargs["width"] = args.width

        outputs = pipe(**pipeline_kwargs).images

        for idx, (fname, out_img) in enumerate(zip(batch_files, outputs), start=1):
            output_path = os.path.join(args.output_dir, fname)
            out_img.save(output_path)
            current = start_idx + idx
            print(f"Processed: {fname} -> {output_path} ({current}/{total})")


if __name__ == "__main__":
    main()
