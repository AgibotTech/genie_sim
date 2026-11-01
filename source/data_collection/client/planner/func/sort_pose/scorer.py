# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import numpy as np


class PiecewiseScorer:
    """
    Score calculator supporting piecewise functions, each segment can be linear/quadratic/custom,
    with adjustable weights and guaranteed score continuity.

    During initialization, offset can be set for each segment's start to make scores continuous between segments.
    """

    def __init__(self, segments):
        """
        Args:
            segments: list of dict, each item configures one segment. Each dict contains:
                - "range": (a, b)       # Interval start and end (closed interval), a, b are float
                - "type": "linear" | "quadratic" | callable, computation method
                - "weight": float       # Weight for this segment
                - "offset": float optional, default 0 # Score offset for this segment (auto-propagated)

            To maintain score continuity, you can:
                1. Only set the first segment's offset=0, leave others as None or omit, auto-propagate backward for continuity (recommended).
                2. Or manually set offset for all segments (if special continuity requirements).
                3. Or allow the last segment to be discontinuous (not recommended).
        """
        self.segments = []
        for i, seg in enumerate(segments):
            a, b = seg["range"]
            fn_type = seg["type"]
            weight = seg.get("weight", 1.0)
            if isinstance(fn_type, str):
                if fn_type == "linear":

                    def kernel(x):
                        return x

                elif fn_type == "quadratic":

                    def kernel(x):
                        return x**2

                elif fn_type == "abs":

                    def kernel(x):
                        return abs(x)

                else:
                    raise ValueError(f"Unknown type: {fn_type}")
            elif callable(fn_type):
                kernel = fn_type
            else:
                raise ValueError("type should be str or callable")
            offset = seg.get("offset", None)
            self.segments.append(
                {
                    "range": (a, b),
                    "kernel": kernel,
                    "weight": weight,
                    "orig_offset": offset,  # Record original offset for later adjustment
                }
            )
        # Auto-propagate offsets to ensure continuity
        self._compute_offsets()

    def _compute_offsets(self):
        """
        Calculate offset for each segment based on segments and weights/functions to ensure score continuity.
        Supports unordered segments, using the first segment with explicitly specified offset as base (if none specified, first segment offset=0).
        """
        # Find first segment with explicitly specified offset as base segment
        base_idx = None
        for i, seg in enumerate(self.segments):
            if seg["orig_offset"] is not None:
                base_idx = i
                break

        # If none explicitly specified, use first segment, offset=0
        if base_idx is None:
            base_idx = 0

        # Initialize all segment offsets to None
        offsets = [None] * len(self.segments)

        # Set base segment offset
        offsets[base_idx] = (
            self.segments[base_idx]["orig_offset"]
            if self.segments[base_idx]["orig_offset"] is not None
            else 0.0
        )

        # Sort by range start value to find numerical adjacency
        sorted_indices = sorted(
            range(len(self.segments)), key=lambda i: self.segments[i]["range"][0]
        )

        # Starting from base segment, expand offset calculation in both directions according to sorted order
        # First process segments before base segment (expand left)
        base_sorted_pos = sorted_indices.index(base_idx)
        for pos in range(base_sorted_pos - 1, -1, -1):
            curr_idx = sorted_indices[pos]
            next_idx = sorted_indices[pos + 1]

            curr_seg = self.segments[curr_idx]
            next_seg = self.segments[next_idx]

            # If current segment already has explicitly specified offset, use it
            if curr_seg["orig_offset"] is not None:
                offsets[curr_idx] = curr_seg["orig_offset"]
            else:
                # Next segment start score (next segment's offset is already determined)
                next_a = next_seg["range"][0]
                next_score = next_seg["kernel"](next_a) * next_seg["weight"] + offsets[next_idx]

                # Current segment end score (need to calculate offset so end score equals next segment start score)
                curr_b = curr_seg["range"][1]
                curr_score_raw = curr_seg["kernel"](curr_b) * curr_seg["weight"]

                # Make current segment end score equal to next segment start score
                offsets[curr_idx] = next_score - curr_score_raw

        # Process segments after base segment (expand right)
        for pos in range(base_sorted_pos + 1, len(sorted_indices)):
            curr_idx = sorted_indices[pos]
            prev_idx = sorted_indices[pos - 1]

            curr_seg = self.segments[curr_idx]
            prev_seg = self.segments[prev_idx]

            # If current segment already has explicitly specified offset, use it
            if curr_seg["orig_offset"] is not None:
                offsets[curr_idx] = curr_seg["orig_offset"]
            else:
                # Previous segment end score
                prev_b = prev_seg["range"][1]
                prev_score = prev_seg["kernel"](prev_b) * prev_seg["weight"] + offsets[prev_idx]

                # Current segment start score
                curr_a = curr_seg["range"][0]
                curr_score_raw = curr_seg["kernel"](curr_a) * curr_seg["weight"]

                # Make current segment start score equal to previous segment end score
                offsets[curr_idx] = prev_score - curr_score_raw

        # Write
        for i in range(len(self.segments)):
            self.segments[i]["offset"] = offsets[i]

    def score(self, value):
        """
        Given input value, return score (guaranteed piecewise continuous).
        """
        for seg in self.segments:
            a, b = seg["range"]
            if a <= value <= b:
                v = value
                score = seg["kernel"](v) * seg["weight"] + seg["offset"]
                return score
        # No match, return 0
        return 0.0

    def batch_score(self, values):
        """
        Batch input, return score array, maintaining same shape as values.
        """
        values = np.asarray(values)
        results = np.zeros_like(values, dtype=float)
        mask_total = np.zeros_like(values, dtype=bool)
        for seg in self.segments:
            a, b = seg["range"]
            mask = (values >= a) & (values <= b)
            results[mask] = seg["kernel"](values[mask]) * seg["weight"] + seg["offset"]
            mask_total |= mask
        # Values not in any interval are 0
        return results

    def plot_segments(self, ax=None, x_min=None, x_max=None):
        """
        Plot piecewise function graph, including each segment interval, kernel, weight, offset, etc.
        Supports intervals containing infinity, user can specify x-axis min/max values.

        Args:
            ax: matplotlib.axes.Axes, if provided, plot on this subplot, otherwise auto-create
            x_min: float, optional, x-axis minimum value
            x_max: float, optional, x-axis maximum value
        """
        import matplotlib.pyplot as plt
        import numpy as np

        # Determine all interval endpoints
        seg_a_list = [seg["range"][0] for seg in self.segments]
        seg_b_list = [seg["range"][1] for seg in self.segments]

        finite_a = [a for a in seg_a_list if np.isfinite(a)]
        finite_b = [b for b in seg_b_list if np.isfinite(b)]

        auto_x_min = min(finite_a) if finite_a else -10.0
        auto_x_max = max(finite_b) if finite_b else 10.0

        # Set actual plotting range based on input values
        draw_x_min = x_min if x_min is not None else auto_x_min
        draw_x_max = x_max if x_max is not None else auto_x_max

        # Provide appropriate range for infinite intervals
        margin = 0.1 * (draw_x_max - draw_x_min)
        x_plot_min = draw_x_min - margin
        x_plot_max = draw_x_max + margin

        x_all = np.linspace(x_plot_min, x_plot_max, 1000)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
            own_fig = True
        else:
            own_fig = False

        for seg in self.segments:
            a, b = seg["range"]
            # Clip infinite intervals to user/auto-given min/max values
            seg_a = max(a, draw_x_min) if np.isfinite(a) else draw_x_min
            seg_b = min(b, draw_x_max) if np.isfinite(b) else draw_x_max
            if seg_a > seg_b:
                continue  # Skip empty interval
            idx = (x_all >= seg_a) & (x_all <= seg_b)
            if np.any(idx):
                y = seg["kernel"](x_all[idx]) * seg["weight"] + seg["offset"]
                ax.plot(
                    x_all[idx],
                    y,
                    label=f'[{a if np.isfinite(a) else "-∞"}, {b if np.isfinite(b) else "+∞"}], '
                    f'w={seg["weight"]:.2f}, offset={seg["offset"]:.2f}',
                )
                # Fill interval range with color
                ax.axvspan(seg_a, seg_b, alpha=0.07, color="b")
        ax.set_xlim([x_plot_min, x_plot_max])
        ax.set_xlabel("Value")
        ax.set_ylabel("Score")
        ax.set_title("Piecewise Scoring Segments")
        ax.legend()
        ax.grid(True)
        if own_fig:
            plt.show()
