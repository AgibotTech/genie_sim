#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""
Generate reasoning ability radar chart
Read data from CSV file and generate radar chart based on cognitive_label
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys
import csv
import os
from collections import defaultdict
import argparse

# Set matplotlib font parameters
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]
plt.rcParams["font.size"] = 10
plt.rcParams["axes.unicode_minus"] = False

# Define cognitive ability category mapping
COGNITIVE_CATEGORIES = {
    "Semantic Understanding": [
        "Color",
        "Size",
        "Shape",
        "Target",
        "Category",
    ],
    "Spatial Understanding": [
        "Absolute position",
        "Relative position",
    ],
    "Common Sense Understanding": [
        "Common Sense",
    ],
    "Physical Laws": [
        "Physics",
    ],
    "Memory": [
        "Memory",
    ],
    "Reasoning": [
        "Logic",
        "Arithmetic",
    ],
}

# Define color scheme
COLORS = {
    "Semantic Understanding": "#FFC53D",
    "Spatial Understanding": "#FF668C",
    "Common Sense Understanding": "#8F73E6",
    "Physical Laws": "#174BE5",
    "Memory": "#44C9C1",
    "Reasoning": "#13CEFF",
}

# Category order for legend
CATEGORY_ORDER = [
    "Semantic Understanding",
    "Spatial Understanding",
    "Common Sense Understanding",
    "Physical Laws",
    "Memory",
    "Reasoning",
]


def load_data_from_csv(csv_file):
    """Load data from CSV file and group by cognitive_label

    Args:
        csv_file: Path to CSV file

    Returns:
        dict: {cognitive_label: average_score}
    """
    cognitive_scores = defaultdict(list)

    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    with open(csv_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Find header line
    header_line = None
    header_idx = None
    for i, line in enumerate(lines):
        if "task_name" in line and "avg_score" in line and "cognitive_label" in line:
            header_line = line.strip()
            header_idx = i
            break

    if header_line is None:
        raise ValueError("CSV file does not contain expected header")

    # Parse header
    header_fields = [field.strip() for field in header_line.split(",")]
    try:
        task_name_idx = header_fields.index("task_name")
        avg_score_idx = header_fields.index("avg_score")
        cognitive_label_idx = header_fields.index("cognitive_label")
    except ValueError as e:
        raise ValueError(f"CSV header missing required fields: {e}")

    # Process data lines
    i = header_idx + 1
    while i < len(lines):
        line = lines[i].strip()

        # Skip empty lines
        if not line:
            i += 1
            continue

        # Skip lines with "Statistics saved to"
        if "Statistics saved to" in line:
            i += 1
            continue

        fields = [f.strip() for f in line.split(",")]

        # Check if this line has 3+ fields: avg_score,operation_label,cognitive_label
        if len(fields) >= 3:
            # Try to parse: first field should be numeric (avg_score)
            # third field should be cognitive_label
            try:
                avg_score = float(fields[0])
                cognitive_label = fields[2] if len(fields) > 2 else ""

                # Validate cognitive_label doesn't contain error messages
                if (
                    cognitive_label
                    and "Statistics" not in cognitive_label
                    and "saved to" not in cognitive_label.lower()
                ):
                    cognitive_scores[cognitive_label].append(avg_score)
            except (ValueError, TypeError):
                # First field is not a number, might be malformed
                # Try to find a numeric field
                for j, field in enumerate(fields):
                    try:
                        avg_score = float(field)
                        # Found score, cognitive_label should be 2 fields after
                        if j + 2 < len(fields):
                            cognitive_label = fields[j + 2]
                            if cognitive_label and "Statistics" not in cognitive_label:
                                cognitive_scores[cognitive_label].append(avg_score)
                        break
                    except (ValueError, TypeError):
                        continue

        i += 1

    # Calculate average score for each cognitive_label
    cognitive_averages = {}
    for cognitive, scores in cognitive_scores.items():
        if scores:
            cognitive_averages[cognitive] = sum(scores) / len(scores)

    return cognitive_averages


def generate_radar_chart(csv_file, output_file=None):
    """Generate radar chart from CSV data

    Args:
        csv_file: Path to CSV file
        output_file: Output file path (default: cognition_radar_chart)
    """
    # Load data from CSV
    cognitive_averages = load_data_from_csv(csv_file)

    # Build ability list and category mapping
    # Include ALL defined abilities, fill missing ones with 0.0
    ability_to_category = {}
    ability_to_score = {}

    # First, build category and score mappings for all abilities
    for category, abilities in COGNITIVE_CATEGORIES.items():
        for ability in abilities:
            ability_to_category[ability] = category
            # Use score from data if available, otherwise use 0.0
            ability_to_score[ability] = cognitive_averages.get(ability, 0.0)

    # Sort abilities by category order to maintain consistent layout
    all_abilities = []
    for category in CATEGORY_ORDER:
        for ability in COGNITIVE_CATEGORIES[category]:
            all_abilities.append(ability)

    if not all_abilities:
        raise ValueError("No abilities defined in COGNITIVE_CATEGORIES")

    # Calculate angles for radar chart
    num_abilities = len(all_abilities)
    angles = np.linspace(0, 2 * np.pi, num_abilities, endpoint=False).tolist()
    angles += angles[:1]  # Close the shape

    # Get scores for each ability
    scores = [ability_to_score[ability] for ability in all_abilities]
    scores += scores[:1]  # Close the data

    # Create figure and subplot
    fig, ax = plt.subplots(figsize=(18, 14), subplot_kw=dict(projection="polar"))
    fig.patch.set_facecolor("white")

    # Plot data
    ax.fill(angles, scores, alpha=0.4, color="#8F73E6", label="Model", linewidth=0)
    ax.plot(
        angles,
        scores,
        color="#8F73E6",
        linewidth=2.5,
        marker="o",
        markersize=8,
        markerfacecolor="#9b7fc7",
        markeredgecolor="white",
        markeredgewidth=1.5,
    )

    # Set angle ticks (hide default labels)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([""] * num_abilities)

    # Add arrows and labels for each ability
    arrow_start_radius = 0.05
    arrow_end_radius = 1.15
    ability_label_radius = 1.25

    for i, ability in enumerate(all_abilities):
        angle = angles[i]

        # Draw arrow
        ax.annotate(
            "",
            xy=(angle, arrow_end_radius),
            xytext=(angle, arrow_start_radius),
            arrowprops=dict(arrowstyle="->", lw=1.5, color="#95a5a6", alpha=0.3),
        )

        # Draw ability label
        ax.text(
            angle,
            ability_label_radius,
            ability,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=18,
            fontweight="bold",
            color="#1a1a1a",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#2c3e50", linewidth=2.5, alpha=0.95),
        )

    # Add category color indicators
    category_ranges = {}
    for i, ability in enumerate(all_abilities):
        category = ability_to_category[ability]
        if category not in category_ranges:
            category_ranges[category] = {"indices": [], "angles": []}
        category_ranges[category]["indices"].append(i)
        category_ranges[category]["angles"].append(angles[i])

    category_arc_radius = 1.45

    for category, cat_info in category_ranges.items():
        cat_angles = sorted(cat_info["angles"])
        if len(cat_angles) > 1:
            # Calculate angle range for this category
            min_angle = min(cat_angles)
            max_angle = max(cat_angles)
            # Handle angle wrap-around
            if max_angle - min_angle > np.pi:
                for i in range(len(cat_angles) - 1):
                    if cat_angles[i + 1] - cat_angles[i] > np.pi:
                        min_angle = cat_angles[i + 1]
                        max_angle = cat_angles[i] + 2 * np.pi
                        break

            # Draw category color arc
            arc_angles = np.linspace(min_angle, max_angle, 100)
            arc_radius = category_arc_radius * np.ones_like(arc_angles)
            ax.plot(arc_angles, arc_radius, color=COLORS[category], linewidth=10, alpha=0.75, solid_capstyle="round")
        else:
            # Single angle, draw point
            ax.plot(
                cat_angles[0],
                category_arc_radius,
                "o",
                color=COLORS[category],
                markersize=12,
                alpha=0.75,
                markeredgecolor="white",
                markeredgewidth=2,
            )

    # Set radial range
    ax.set_ylim(0, 1.55)
    # Set grid lines
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([""] * 6)

    # Add radial labels on the right (0 degrees)
    label_angle = 0
    radial_labels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for r_val in radial_labels:
        ax.text(
            label_angle,
            r_val,
            f"{r_val:.1f}" if r_val > 0 else "0",
            horizontalalignment="left",
            verticalalignment="center",
            fontsize=14,
            fontweight="bold",
            color="#666666",
        )

    # Set grid line style
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.8, color="#d3d3d3")
    # Remove outer circle
    ax.spines["polar"].set_visible(False)

    # Add legend
    legend = ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=1, fontsize=14, frameon=False, markerscale=1.5
    )
    for text in legend.get_texts():
        text.set_fontsize(14)
        text.set_fontweight("bold")
    for handle in legend.legend_handles:
        handle.set_alpha(0.8)

    # Add category labels in bottom left
    legend_y_start = 0.28
    legend_x = 0.05
    line_height = 0.04

    for i, category in enumerate(CATEGORY_ORDER):
        if category not in category_ranges:
            continue
        color = COLORS[category]
        y_pos = legend_y_start - i * line_height
        # Draw color marker
        fig.text(legend_x, y_pos, "■", fontsize=20, color=color, verticalalignment="center", horizontalalignment="left")
        # Draw category name
        fig.text(
            legend_x + 0.025,
            y_pos,
            category,
            fontsize=15,
            fontweight="bold",
            color="#333333",
            verticalalignment="center",
            horizontalalignment="left",
        )

    # Set title
    title = plt.title(
        "REASONING ABILITY ASSESSMENT", pad=20, fontsize=28, fontweight="bold", color="#000000", alpha=1.0
    )

    # Adjust layout
    plt.tight_layout()

    # Save image
    if output_file is None:
        output_file = "cognition_radar_chart"

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Radar chart saved as: {output_file}")
    print(f"Score range: {min(scores[:-1]):.2f} - {max(scores[:-1]):.2f}")
    print(f"Total abilities: {num_abilities}")

    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate reasoning ability radar chart from CSV data")
    parser.add_argument(
        "--csv",
        type=str,
        default="task_scores.csv",
        help="Path to CSV file (default: task_scores.csv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file path (default: cognition_radar_chart)",
    )

    args = parser.parse_args()

    # Get absolute path for CSV file
    if not os.path.isabs(args.csv):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(os.path.dirname(script_dir), args.csv)
    else:
        csv_path = args.csv

    print("Generating reasoning ability radar chart...")
    print("=" * 50)
    try:
        generate_radar_chart(csv_path, args.output)
        print("=" * 50)
        print("Done!")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
