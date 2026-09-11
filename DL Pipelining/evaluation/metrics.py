"""Evaluation metrics for correspondence / registration experiments."""

from __future__ import annotations

from typing import Optional

import numpy as np


def compute_spatial_coverage(
    keypoints: np.ndarray,
    image_shape: tuple[int, int],
    grid: tuple[int, int] = (4, 4),
) -> float:
    """
    Fraction of grid cells that contain at least one keypoint.
    keypoints: Nx2 in (x, y) image coordinates
    image_shape: (H, W)
    """
    if keypoints is None or len(keypoints) == 0:
        return 0.0

    h, w = image_shape[:2]
    rows, cols = grid
    occupied = np.zeros((rows, cols), dtype=bool)

    xs = np.clip(keypoints[:, 0], 0, w - 1e-6)
    ys = np.clip(keypoints[:, 1], 0, h - 1e-6)
    cell_w = w / cols
    cell_h = h / rows

    for x, y in zip(xs, ys):
        c = int(x / cell_w)
        r = int(y / cell_h)
        occupied[r, c] = True

    return float(occupied.sum() / occupied.size)


def p95_error(errors: np.ndarray) -> float:
    if errors is None or len(errors) == 0:
        return 0.0
    return float(np.percentile(errors, 95))


def summarize_match_counts(
    total_matches: int,
    inliers: int,
) -> dict:
    ratio = (inliers / total_matches) if total_matches > 0 else 0.0
    return {
        "total_matches": int(total_matches),
        "inliers": int(inliers),
        "inlier_ratio": float(ratio),
    }