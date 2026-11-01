"""Apply temporal Savitzky-Golay filtering to episode deformable NPZ points.
Suppress single-step cloth jitter while preserving larger-scale movement.
CLI defaults to dry-run; --in-place writes files. The callable defaults to
in-place processing and does nothing when deformable cloth data is absent."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter


def load_episode(deformable_dir: Path):
    """Load point arrays and metadata for all frames in a cloth directory."""
    files = sorted(deformable_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no npz found under {deformable_dir}")
    sample = np.load(files[0], allow_pickle=True)
    P = sample["points"].shape[0]
    pts = np.empty((len(files), P, 3), dtype=np.float32)
    ts = np.empty(len(files), dtype=np.float64)
    prim_path = str(sample["prim_path"])
    for i, f in enumerate(files):
        d = np.load(f, allow_pickle=True)
        if d["points"].shape != (P, 3):
            raise ValueError(
                f"frame {f.name} has {d['points'].shape}, expected ({P},3)"
            )
        pts[i] = d["points"]
        ts[i] = float(d["time_stamp"])
    return files, pts, ts, prim_path


def smooth_points(pts: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """SG along time axis. pts: (T, P, 3)."""
    T = pts.shape[0]
    if T < window:
        raise ValueError(f"T={T} < window={window}")
    out = savgol_filter(
        pts, window_length=window, polyorder=polyorder, axis=0, mode="nearest"
    )
    return out.astype(np.float32, copy=False)


def stats(pts: np.ndarray, dt: float) -> dict:
    v = np.diff(pts, axis=0) / dt
    speed = np.linalg.norm(v, axis=-1)
    a = np.diff(v, axis=0) / dt
    acc = np.linalg.norm(a, axis=-1)
    return {
        "speed_mean": float(speed.mean()),
        "speed_p95": float(np.percentile(speed, 95)),
        "speed_max": float(speed.max()),
        "acc_mean": float(acc.mean()),
        "acc_p95": float(np.percentile(acc, 95)),
        "acc_max": float(acc.max()),
    }


def fmt(d: dict) -> str:
    return (
        f"v: mean={d['speed_mean']:.4f}  p95={d['speed_p95']:.4f}  max={d['speed_max']:.4f}    "
        f"|a|: mean={d['acc_mean']:.3f}  p95={d['acc_p95']:.3f}  max={d['acc_max']:.3f}"
    )


def _process_one_cloth(
    cloth_dir: Path,
    window: int,
    polyorder: int,
    in_place: bool,
    log,
) -> None:
    files, pts, ts, prim_path = load_episode(cloth_dir)
    dt = float(np.median(np.diff(ts)))
    log(f"[smooth_deformable] {cloth_dir}: frames={len(files)} P={pts.shape[1]} dt={dt:.6f}s")

    before = stats(pts, dt)
    pts_s = smooth_points(pts, window=window, polyorder=polyorder)
    after = stats(pts_s, dt)
    log(f"[smooth_deformable]   before | {fmt(before)}")
    log(f"[smooth_deformable]   after  | {fmt(after)}")

    if not in_place:
        log("[smooth_deformable]   [dry-run] not writing. add --in-place to overwrite.")
        return

    for i, f in enumerate(files):
        np.savez(f, points=pts_s[i], prim_path=prim_path, time_stamp=ts[i])
    log(f"[smooth_deformable]   [written] {len(files)} npz updated in place")


def smooth_episode_deformable(
    episode_dir,
    window: int = 7,
    polyorder: int = 2,
    in_place: bool = True,
    log=None,
) -> bool:
    """Filter episode_dir/deformable/<cloth_id>/*.npz, writing in place by default.
    Back up deformable to deformable_raw once; restore raw data before later runs
    to avoid cumulative smoothing. Skip cloth sequences shorter than window.
    Return True if at least one cloth was processed, otherwise False."""
    if log is None:
        log = print  # noqa: T202
    episode_dir = Path(episode_dir)
    deformable_root = episode_dir / "deformable"
    raw_root = episode_dir / "deformable_raw"
    if not deformable_root.is_dir():
        return False

    if in_place:
        # Keep runs repeatable: back up once, then restore original data before filtering.
        if not raw_root.exists():
            shutil.copytree(deformable_root, raw_root)
            log(f"[smooth_deformable] backed up {deformable_root} -> {raw_root}")
        else:
            shutil.rmtree(deformable_root)
            shutil.copytree(raw_root, deformable_root)
            log(f"[smooth_deformable] restored from {raw_root} (idempotent rerun)")

    cloth_dirs = sorted(p for p in deformable_root.iterdir() if p.is_dir())
    if not cloth_dirs:
        return False

    log(
        f"[smooth_deformable] episode={episode_dir} objects={[p.name for p in cloth_dirs]} "
        f"SG(window={window}, polyorder={polyorder}) in_place={in_place}"
    )
    processed = False
    for d in cloth_dirs:
        try:
            _process_one_cloth(d, window, polyorder, in_place, log)
            processed = True
        except (FileNotFoundError, ValueError) as e:
            log(f"[smooth_deformable]   skip {d}: {e}")
    return processed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-dir", required=True, type=Path)
    ap.add_argument("--window", type=int, default=7)
    ap.add_argument("--polyorder", type=int, default=2)
    ap.add_argument("--in-place", action="store_true")
    args = ap.parse_args()

    if not args.episode_dir.is_dir():
        print(f"episode_dir not a dir: {args.episode_dir}", file=sys.stderr)
        sys.exit(1)

    ok = smooth_episode_deformable(
        args.episode_dir,
        window=args.window,
        polyorder=args.polyorder,
        in_place=args.in_place,
    )
    if not ok:
        print(
            f"[smooth_deformable] no deformable/ under {args.episode_dir}; nothing to do.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
