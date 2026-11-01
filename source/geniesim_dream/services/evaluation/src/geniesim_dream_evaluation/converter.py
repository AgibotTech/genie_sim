#!/usr/bin/env python3
"""Convert Dream canonical 2-row concat videos to horizontal top|left|right.

Source layout (canonical Dream export), fixed panel sizes:

    ┌──────── top 640×400 ────────┐
    ├─ left 320×200 ─┬─ right 320×200 ─┤
    → concat frame 640×600

Raw FD 640×592 videos are rejected here; Dream finish must canonicalize first.

Output: ``{stem}_top_left_right.mp4`` with three equal 640×400 panels → 1920×400.

Examples:
  python cosmos_concat_to_triplet.py /path/to/vision.mp4 -o /tmp/vision_top_left_right.mp4
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

LAYOUT_SUFFIX = "_top_left_right"
DEFAULT_CRF = 18
CONVERTER_VERSION = "dream-concat-600-200-v1"

# Dream canonical concat (vision.py CONCAT_SIZE / panel sizes).
TOP_W, TOP_H = 640, 400
WRIST_W, WRIST_H = 320, 200
EXPECTED_W = TOP_W
EXPECTED_H = TOP_H + WRIST_H  # 600
OUTPUT_W, OUTPUT_H = 1920, 400

# Exact crops + explicit wrist scale to 640x400 (no scale2ref guessing).
FILTER_COMPLEX = (
    "[0:v]split=3[s0][s1][s2];"
    f"[s0]crop={TOP_W}:{TOP_H}:0:0,setsar=1[top];"
    f"[s1]crop={WRIST_W}:{WRIST_H}:0:{TOP_H},scale={TOP_W}:{TOP_H}:flags=bicubic,setsar=1[left];"
    f"[s2]crop={WRIST_W}:{WRIST_H}:{WRIST_W}:{TOP_H},scale={TOP_W}:{TOP_H}:flags=bicubic,setsar=1[right];"
    "[top][left][right]hstack=inputs=3[vout]"
)


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH")
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe not found in PATH")


def probe_size(path: Path) -> Tuple[int, int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace")[-500:]
        raise RuntimeError(f"ffprobe failed for {path}:\n{err}")
    data = json.loads(proc.stdout.decode("utf-8"))
    stream = data["streams"][0]
    return int(stream["width"]), int(stream["height"])


def probe_video(path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames,r_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace")[-500:]
        raise RuntimeError(f"ffprobe failed for {path}:\n{err}")
    stream = json.loads(proc.stdout.decode("utf-8"))["streams"][0]
    rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
    num, den = rate.split("/")
    fps = float(num) / float(den) if float(den) else 0.0
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "nb_frames": stream.get("nb_frames"),
        "converter_version": CONVERTER_VERSION,
    }


def assert_source_layout(path: Path) -> None:
    w, h = probe_size(path)
    if w != EXPECTED_W or h != EXPECTED_H:
        raise ValueError(
            f"{path}: expected Dream canonical concat {EXPECTED_W}x{EXPECTED_H} "
            f"(top {TOP_W}x{TOP_H} + wrists {WRIST_W}x{WRIST_H}), got {w}x{h}. "
            "Raw 640x592 must be canonicalized by Dream finish before conversion."
        )


def is_source_mp4(path: Path) -> bool:
    if path.suffix.lower() != ".mp4" or not path.is_file():
        return False
    if path.stem.endswith(LAYOUT_SUFFIX):
        return False
    return True


def find_videos(source: Path) -> List[Path]:
    if source.is_file():
        if not is_source_mp4(source):
            raise ValueError(f"not a cosmos source mp4: {source}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"input not found: {source}")
    videos = sorted(
        p for p in source.iterdir() if p.is_file() and is_source_mp4(p)
    )
    if not videos:
        raise FileNotFoundError(f"no source mp4 files in {source}")
    return videos


def default_out_dir(source: Path) -> Path:
    if source.is_file():
        return source.parent / "lcr"
    return source / "lcr"


def output_path_for(src: Path, out_dir: Path) -> Path:
    return out_dir / f"{src.stem}{LAYOUT_SUFFIX}.mp4"


def convert_one(
    src: Path,
    dst: Path,
    crf: int,
    overwrite: bool,
) -> str:
    if dst.exists() and not overwrite:
        return "skip"
    require_ffmpeg()
    assert_source_layout(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    temporary = dst.with_suffix(dst.suffix + ".tmp.mp4")
    if temporary.exists():
        temporary.unlink()
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-filter_complex",
        FILTER_COMPLEX,
        "-map",
        "[vout]",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        temporary.unlink(missing_ok=True)
        err = proc.stderr.decode("utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"ffmpeg failed for {src}:\n{err}")
    probe = probe_video(temporary)
    if probe["width"] != OUTPUT_W or probe["height"] != OUTPUT_H:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"converter probe mismatch for {src}: expected {OUTPUT_W}x{OUTPUT_H}, got "
            f"{probe['width']}x{probe['height']}"
        )
    temporary.replace(dst)
    return "ok"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Dream canonical concat (top 640x400 over left|right 320x200) "
            "to horizontal top|left|right mp4. Does not overwrite source videos."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="cosmos mp4 file(s) and/or a directory of ep*.mp4",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="output mp4 (only valid with a single input file)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output directory (default: <input>/lcr)",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=DEFAULT_CRF,
        help=f"libx264 CRF (default: {DEFAULT_CRF})",
    )
    parser.add_argument(
        "-y",
        "--overwrite",
        action="store_true",
        help="overwrite existing *_top_left_right.mp4",
    )
    return parser.parse_args()


def iter_jobs(
    inputs: Iterable[Path],
    output: Optional[Path],
    out_dir: Optional[Path],
) -> Iterable[tuple[Path, Path]]:
    inputs = list(inputs)
    if output is not None:
        if len(inputs) != 1 or not inputs[0].is_file():
            raise ValueError("-o / --output requires exactly one mp4 file input")
        yield inputs[0], output
        return

    for raw in inputs:
        videos = find_videos(raw)
        dest_dir = out_dir if out_dir is not None else default_out_dir(raw)
        for src in videos:
            yield src, output_path_for(src, dest_dir)


def main() -> int:
    args = parse_args()
    n_ok = n_skip = 0
    for src, dst in iter_jobs(args.inputs, args.output, args.out_dir):
        status = convert_one(src, dst, args.crf, args.overwrite)
        print(f"[{status}] {src} -> {dst}")
        if status == "ok":
            n_ok += 1
        else:
            n_skip += 1
    print(f"[done] ok={n_ok} skip={n_skip}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
