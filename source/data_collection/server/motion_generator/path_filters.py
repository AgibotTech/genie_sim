# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import torch


def sort_by_difference_js(paths, weights=None):
    """
    Sorts a list of joint space paths based on the cumulative difference between consecutive waypoints.

    Args:
        paths (List(JointState): A list of JointState, each path contains a position of shape (T, D) where
                              T is the number of waypoints per path, and D is the dimensionality of each waypoint.
        weights (torch.tensor, optional): A tensor of shape (D,) representing weights for each dimension.
                                          If None, all dimensions are weighted equally.
    Returns:
        torch.tensor: Indices that would sort the paths based on the cumulative difference.
    """

    assert len(paths) > 0, "The paths list should not be empty."

    assert (
        weights is None or weights.shape[0] == paths[0].position.shape[-1]
    ), "Weights must be of shape (D,) where D is the dimensionality of each waypoint."

    device = paths[0].position.device
    if weights is None:
        weights = torch.ones(paths[0].position.shape[-1], device=device)
    else:
        weights = weights.to(device)

    # Calculate the absolute differences between consecutive waypoints
    diffs = []
    for path in paths:
        diff = torch.abs(path.position[1:, :] - path.position[:-1, :])  # Shape: (T-1, D)
        diff = diff.sum(dim=0)  # Average over waypoints, Shape: (D,)
        diffs.append(diff)
    diffs = torch.stack(diffs)  # Shape: (N, D)

    # Apply weights to the differences
    weighted_diffs = diffs * weights  # Broadcasting weights over the last dimension

    # Sum the weighted differences over all waypoints and dimensions
    cumulative_diffs = weighted_diffs.sum(dim=(1))  # Shape: (N,)

    # Get the indices that would sort the paths based on cumulative differences
    sorted_indices = torch.argsort(cumulative_diffs)

    return sorted_indices


def filter_paths_by_position_error(paths, position_errors):
    """
    Filters out paths whose position error exceeds one sigma threshold.

    Args:
        paths (List(JointState): A list of JointState, each path contains a position of shape (T, D).
        position_errors (torch.tensor): A tensor of shape (N,) representing the position error for each path.

    Returns:
        List(bool): A filtered list of bool where each path's position error is below the threshold.
    """
    assert (
        len(paths) == position_errors.shape[0]
    ), "The number of paths must match the number of position errors."

    mean_error = torch.mean(position_errors)
    torch.std(position_errors)
    threshold = mean_error  # + std_error  # one sigma threshold
    res = [position_error <= threshold for position_error in position_errors]

    return res


def filter_paths_by_rotation_error(paths, rotation_errors):
    """
    Filters out paths whose rotation error exceeds two sigma threshold.

    Args:
        paths (List(JointState): A list of JointState, each path contains a position of shape (T, D).
        rotation_errors (torch.tensor): A tensor of shape (N,) representing the rotation error for each path.

    Returns:
        List(bool): A filtered list of bool where each path's rotation error is below the threshold.
    """
    assert (
        len(paths) == rotation_errors.shape[0]
    ), "The number of paths must match the number of rotation errors."

    mean_error = torch.mean(rotation_errors)
    torch.std(rotation_errors)
    threshold = mean_error  # + std_error  # one sigma threshold

    res = [rotation_error <= threshold for rotation_error in rotation_errors]
    return res
