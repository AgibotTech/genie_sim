#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Preview writing control points using a scene's target_letter or an explicit letter."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DATA_GEN_PATH = HERE / "data_gen.py"
STROKE_COLORS = ("#1479d1", "#ef8a17", "#2ca02c", "#9467bd")


def load_literal_assignments(path: Path, names: set[str]) -> dict[str, object]:
    """Read selected literal module constants without importing robot code."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in names:
            values[target.id] = ast.literal_eval(node.value)

    missing = names - values.keys()
    if missing:
        raise ValueError(f"missing literal constants in {path}: {sorted(missing)}")
    return values


def sample_polyline(points: np.ndarray, spacing: float) -> np.ndarray:
    """Mirror data_gen.sample_polyline for a dependency-free preview."""
    sampled = [points[0]]
    for start, end in zip(points[:-1], points[1:]):
        segment = end - start
        steps = max(1, int(np.ceil(np.linalg.norm(segment) / spacing)))
        sampled.extend(
            start + segment * ratio
            for ratio in np.linspace(0.0, 1.0, steps + 1)[1:]
        )
    return np.asarray(sampled)


def to_robot_view_metres(
    normalized_points: np.ndarray,
    letter_width: float,
    letter_height: float,
) -> np.ndarray:
    """Map glyph coordinates to robot-view right/up axes in metres."""
    return np.column_stack(
        (
            normalized_points[:, 0] * letter_width,
            normalized_points[:, 1] * letter_height,
        )
    )


def add_direction_arrow(ax, points: np.ndarray, color: str, fraction: float) -> None:
    index = min(max(0, int((len(points) - 1) * fraction)), len(points) - 2)
    start = points[index]
    end = points[index + 1]
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "->", "color": color, "lw": 1.8},
    )


def render(letter: str | None, output: Path | None, scene_info: Path | None = None) -> Path:
    constants = load_literal_assignments(
        DATA_GEN_PATH,
        {
            "LETTER_STROKES",
            "LETTER_WIDTH",
            "LETTER_HEIGHT",
            "STROKE_POINT_SPACING",
        },
    )
    if scene_info is not None:
        letter = json.loads(scene_info.read_text(encoding="utf-8"))["target_letter"]
    if letter is None:
        raise ValueError("provide a scene_info.json or an explicit letter")
    letter_key = str(letter).upper()
    all_strokes = constants["LETTER_STROKES"]
    if letter_key not in all_strokes:
        raise ValueError(
            f"unsupported letter {letter_key!r}; choose from {sorted(all_strokes)}"
        )

    strokes = all_strokes[letter_key]
    letter_width = float(constants["LETTER_WIDTH"])
    letter_height = float(constants["LETTER_HEIGHT"])
    spacing = float(constants["STROKE_POINT_SPACING"])
    output = output or HERE / f"{letter_key.lower()}_trajectory_preview.png"

    figure, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    sampled_strokes = []
    for stroke_index, normalized_stroke in enumerate(strokes, start=1):
        color = STROKE_COLORS[(stroke_index - 1) % len(STROKE_COLORS)]
        control = to_robot_view_metres(
            np.asarray(normalized_stroke, dtype=float),
            letter_width,
            letter_height,
        )
        sampled = sample_polyline(control, spacing)
        sampled_strokes.append(sampled)

        ax.plot(
            sampled[:, 0],
            sampled[:, 1],
            color=color,
            linewidth=4,
            label=f"stroke {stroke_index}",
            zorder=2,
        )
        ax.scatter(
            sampled[:, 0],
            sampled[:, 1],
            s=14,
            color="#6a3d9a",
            alpha=0.55,
            label="sampled trajectory points" if stroke_index == 1 else None,
            zorder=3,
        )
        ax.scatter(
            control[:, 0],
            control[:, 1],
            s=55,
            facecolors="white",
            edgecolors=color,
            linewidths=1.5,
            label="declared control points" if stroke_index == 1 else None,
            zorder=4,
        )
        add_direction_arrow(ax, sampled, color, 0.45)
        print(
            f"stroke {stroke_index}: {len(control)} control points, "
            f"{len(sampled)} sampled points"
        )

    start = sampled_strokes[0][0]
    end = sampled_strokes[-1][-1]
    ax.scatter(
        [start[0]], [start[1]], marker="^", s=130, color="#2ca02c",
        label="start", zorder=5,
    )
    ax.scatter(
        [end[0]], [end[1]], marker="s", s=100, color="#d62728",
        label="end", zorder=5,
    )

    ax.set_title(
        f"Current uppercase {letter_key} trajectory (robot view)\n"
        f"sampling spacing: {spacing * 1000:.1f} mm"
    )
    ax.set_xlabel("robot-view right (m)")
    ax.set_ylabel("robot-view up (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.legend(loc="upper right", fontsize=9)
    ax.margins(0.18)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    print(f"wrote {output}")
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--letter", help="explicit uppercase letter to preview")
    selection.add_argument("--scene-info", type=Path, help="read target_letter from this scene_info.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve() if args.output else None
    render(args.letter, output, args.scene_info)


if __name__ == "__main__":
    main()
