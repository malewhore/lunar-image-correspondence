from pathlib import Path
import json

import cv2
import numpy as np


# ============================================================
# SIFT / ROOTSIFT MATCHING BENCHMARK
#
# Compares:
#
#   OHRC normalized   -> LROC normalized
#   OHRC normalized   -> LROC structural
#   OHRC structural  -> LROC normalized
#   OHRC structural  -> LROC structural
#
# For each combination:
#
#   RootSIFT descriptors
#       ↓
#   KNN matching
#       ↓
#   Lowe ratio tests
#       ↓
#   mutual consistency
#       ↓
#   duplicate filtering
#       ↓
#   affine RANSAC
#   homography RANSAC
#       ↓
#   metrics
#
# This is a DIAGNOSTIC benchmark.
#
# It does not assume the current OHRC→LROC geographic
# mapping is ground truth.
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

RATIO_VALUES = [
    0.70,
    0.75,
    0.80,
    0.85,
]

RANSAC_THRESHOLD = 5.0
RANSAC_CONFIDENCE = 0.995
RANSAC_MAX_ITERS = 5000

MIN_INLIERS = 6

GRID_ROWS = 8
GRID_COLS = 8


# ============================================================
# LOAD FEATURES
# ============================================================

def load_features(path):

    data = np.load(
        path,
        allow_pickle=False
    )

    return {
        "points": data["points"].astype(
            np.float32
        ),
        "descriptors": data["descriptors"].astype(
            np.float32
        ),
        "responses": data["responses"].astype(
            np.float32
        ),
    }


# ============================================================
# MUTUAL RATIO MATCHING
# ============================================================

def mutual_ratio_match(
    src,
    dst,
    ratio
):
    """
    Perform bidirectional RootSIFT matching.

    src:
        moving-image features

    dst:
        reference-image features
    """

    src_desc = src["descriptors"]
    dst_desc = dst["descriptors"]

    if (
        len(src_desc) < 2
        or len(dst_desc) < 2
    ):
        return []

    matcher = cv2.BFMatcher(
        cv2.NORM_L2,
        crossCheck=False
    )

    # --------------------------------------------------------
    # Source -> destination
    # --------------------------------------------------------

    forward = matcher.knnMatch(
        src_desc,
        dst_desc,
        k=2
    )

    forward_good = {}

    for pair in forward:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < ratio * n.distance:

            forward_good[
                m.queryIdx
            ] = m.trainIdx

    # --------------------------------------------------------
    # Destination -> source
    # --------------------------------------------------------

    backward = matcher.knnMatch(
        dst_desc,
        src_desc,
        k=2
    )

    backward_good = {}

    for pair in backward:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < ratio * n.distance:

            backward_good[
                m.queryIdx
            ] = m.trainIdx

    # --------------------------------------------------------
    # Mutual consistency
    # --------------------------------------------------------

    matches = []

    for src_idx, dst_idx in forward_good.items():

        if backward_good.get(
            dst_idx
        ) == src_idx:

            matches.append(
                (
                    src_idx,
                    dst_idx
                )
            )

    return matches


# ============================================================
# DUPLICATE DESTINATION FILTER
# ============================================================

def remove_duplicate_destinations(
    src,
    dst,
    matches
):
    """
    Keep only one source→destination match per destination.

    The lowest RootSIFT distance is retained.
    """

    if not matches:
        return []

    src_desc = src["descriptors"]
    dst_desc = dst["descriptors"]

    matcher = cv2.BFMatcher(
        cv2.NORM_L2
    )

    best = {}

    for src_idx, dst_idx in matches:

        distance = cv2.norm(
            src_desc[src_idx],
            dst_desc[dst_idx],
            cv2.NORM_L2
        )

        if (
            dst_idx not in best
            or distance < best[dst_idx][0]
        ):

            best[dst_idx] = (
                distance,
                src_idx
            )

    return [
        (
            src_idx,
            dst_idx
        )
        for dst_idx, (
            distance,
            src_idx
        ) in best.items()
    ]


# ============================================================
# GET MATCH POINTS
# ============================================================

def match_points(
    src,
    dst,
    matches
):
    """
    Convert feature indices into coordinate arrays.
    """

    if not matches:

        return (
            np.empty(
                (0, 2),
                dtype=np.float32
            ),
            np.empty(
                (0, 2),
                dtype=np.float32
            )
        )

    src_points = np.array(
        [
            src["points"][i]
            for i, j in matches
        ],
        dtype=np.float32
    )

    dst_points = np.array(
        [
            dst["points"][j]
            for i, j in matches
        ],
        dtype=np.float32
    )

    return (
        src_points,
        dst_points
    )


# ============================================================
# AFFINE RANSAC
# ============================================================

def estimate_affine(
    src_points,
    dst_points
):
    """
    Estimate partial affine transformation.
    """

    if len(src_points) < 3:

        return None, None

    matrix, mask = cv2.estimateAffinePartial2D(
        src_points,
        dst_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_THRESHOLD,
        maxIters=RANSAC_MAX_ITERS,
        confidence=RANSAC_CONFIDENCE,
        refineIters=10
    )

    if matrix is None or mask is None:

        return None, None

    return matrix, mask.ravel().astype(
        bool
    )


# ============================================================
# HOMOGRAPHY RANSAC
# ============================================================

def estimate_homography(
    src_points,
    dst_points
):
    """
    Estimate projective homography.
    """

    if len(src_points) < 4:

        return None, None

    matrix, mask = cv2.findHomography(
        src_points,
        dst_points,
        cv2.RANSAC,
        RANSAC_THRESHOLD,
        maxIters=RANSAC_MAX_ITERS,
        confidence=RANSAC_CONFIDENCE
    )

    if matrix is None or mask is None:

        return None, None

    return matrix, mask.ravel().astype(
        bool
    )


# ============================================================
# REPROJECTION ERROR
# ============================================================

def reprojection_errors(
    matrix,
    src_points,
    dst_points,
    model
):
    """
    Calculate Euclidean reprojection error.
    """

    if matrix is None:

        return np.empty(
            (0,),
            dtype=np.float32
        )

    if model == "affine":

        projected = cv2.transform(
            src_points.reshape(
                -1,
                1,
                2
            ),
            matrix
        ).reshape(
            -1,
            2
        )

    elif model == "homography":

        projected_h = cv2.perspectiveTransform(
            src_points.reshape(
                -1,
                1,
                2
            ),
            matrix
        ).reshape(
            -1,
            2
        )

        projected = projected_h

    else:

        raise ValueError(
            f"Unknown model: {model}"
        )

    return np.linalg.norm(
        projected - dst_points,
        axis=1
    )


# ============================================================
# SPATIAL COVERAGE
# ============================================================

def spatial_statistics(
    points,
    image_shape
):
    """
    Calculate spatial distribution of source matches.

    Returns:
        occupied cells
        coefficient of variation
    """

    if len(points) == 0:

        return 0, None

    h, w = image_shape[:2]

    rows = np.clip(
        (
            points[:, 1]
            / h
            * GRID_ROWS
        ).astype(int),
        0,
        GRID_ROWS - 1
    )

    cols = np.clip(
        (
            points[:, 0]
            / w
            * GRID_COLS
        ).astype(int),
        0,
        GRID_COLS - 1
    )

    counts = np.zeros(
        GRID_ROWS * GRID_COLS,
        dtype=np.float32
    )

    for r, c in zip(
        rows,
        cols
    ):

        counts[
            r * GRID_COLS + c
        ] += 1

    occupied = int(
        np.count_nonzero(
            counts
        )
    )

    mean = float(
        np.mean(counts)
    )

    if mean <= 0:

        cv = None

    else:

        cv = float(
            np.std(counts)
            /
            mean
        )

    return occupied, cv


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    matrix,
    inlier_mask,
    src_points,
    dst_points,
    model,
    image_shape
):
    """
    Calculate registration diagnostics.
    """

    if matrix is None:

        return {
            "inliers": 0,
            "inlier_ratio": 0.0,
            "rmse_px": None,
            "median_error_px": None,
            "p95_error_px": None,
            "spatial_cells": 0,
            "spatial_cv": None,
        }

    errors = reprojection_errors(
        matrix,
        src_points,
        dst_points,
        model
    )

    inlier_errors = errors[
        inlier_mask
    ]

    inlier_points = src_points[
        inlier_mask
    ]

    if len(inlier_errors) == 0:

        return {
            "inliers": 0,
            "inlier_ratio": 0.0,
            "rmse_px": None,
            "median_error_px": None,
            "p95_error_px": None,
            "spatial_cells": 0,
            "spatial_cv": None,
        }

    rmse = float(
        np.sqrt(
            np.mean(
                inlier_errors ** 2
            )
        )
    )

    median = float(
        np.median(
            inlier_errors
        )
    )

    p95 = float(
        np.percentile(
            inlier_errors,
            95
        )
    )

    cells, cv = spatial_statistics(
        inlier_points,
        image_shape
    )

    return {
        "inliers": int(
            np.count_nonzero(
                inlier_mask
            )
        ),

        "inlier_ratio": float(
            np.count_nonzero(
                inlier_mask
            )
            /
            max(
                len(src_points),
                1
            )
        ),

        "rmse_px": rmse,

        "median_error_px": median,

        "p95_error_px": p95,

        "spatial_cells": cells,

        "spatial_cv": cv,
    }


# ============================================================
# TEST ONE CONFIGURATION
# ============================================================

def test_configuration(
    src,
    dst,
    ratio,
    image_shape
):
    """
    Test one matching ratio and both geometric models.
    """

    raw_matches = mutual_ratio_match(
        src,
        dst,
        ratio
    )

    filtered_matches = (
        remove_duplicate_destinations(
            src,
            dst,
            raw_matches
        )
    )

    src_points, dst_points = (
        match_points(
            src,
            dst,
            filtered_matches
        )
    )

    result = {
        "ratio": ratio,

        "raw_mutual_matches": len(
            raw_matches
        ),

        "filtered_matches": len(
            filtered_matches
        ),

        "affine": None,

        "homography": None,
    }

    # --------------------------------------------------------
    # Affine
    # --------------------------------------------------------

    affine_matrix, affine_mask = (
        estimate_affine(
            src_points,
            dst_points
        )
    )

    result["affine"] = evaluate_model(
        affine_matrix,
        affine_mask
        if affine_mask is not None
        else np.zeros(
            len(src_points),
            dtype=bool
        ),
        src_points,
        dst_points,
        "affine",
        image_shape
    )

    # --------------------------------------------------------
    # Homography
    # --------------------------------------------------------

    homography_matrix, homography_mask = (
        estimate_homography(
            src_points,
            dst_points
        )
    )

    result["homography"] = evaluate_model(
        homography_matrix,
        homography_mask
        if homography_mask is not None
        else np.zeros(
            len(src_points),
            dtype=bool
        ),
        src_points,
        dst_points,
        "homography",
        image_shape
    )

    return result


# ============================================================
# MAIN BENCHMARK
# ============================================================

def main():

    base = Path(
        "experiments"
        "/features"
    )

    ohrc_normalized = load_features(
        base
        / "pair_001_ohrc_sift"
        / "sift_features.npz"
    )

    ohrc_structural = load_features(
        base
        / "pair_001_ohrc_structural_sift"
        / "sift_features.npz"
    )

    lroc_normalized = load_features(
        base
        / "pair_001_lroc_sift"
        / "sift_features.npz"
    )

    lroc_structural = load_features(
        base
        / "pair_001_lroc_structural_sift"
        / "sift_features.npz"
    )

    combinations = [
        (
            "ohrc_normalized",
            ohrc_normalized,
            "lroc_normalized",
            lroc_normalized
        ),
        (
            "ohrc_normalized",
            ohrc_normalized,
            "lroc_structural",
            lroc_structural
        ),
        (
            "ohrc_structural",
            ohrc_structural,
            "lroc_normalized",
            lroc_normalized
        ),
        (
            "ohrc_structural",
            ohrc_structural,
            "lroc_structural",
            lroc_structural
        ),
    ]

    output_dir = Path(
        "experiments"
        "/matching"
        "/pair_001_sift_benchmark"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    all_results = {}

    print("=" * 70)
    print(
        "PAIR 001 — SIFT / ROOTSIFT MATCHING BENCHMARK"
    )
    print("=" * 70)

    print()

    for (
        src_name,
        src,
        dst_name,
        dst
    ) in combinations:

        name = (
            f"{src_name}"
            f"_TO_"
            f"{dst_name}"
        )

        print("-" * 70)
        print(name)
        print("-" * 70)

        print(
            f"Source features    : "
            f"{len(src['points'])}"
        )

        print(
            f"Reference features : "
            f"{len(dst['points'])}"
        )

        results = []

        for ratio in RATIO_VALUES:

            print(
                f"Testing ratio "
                f"{ratio:.2f} ..."
            )

            result = test_configuration(
                src,
                dst,
                ratio,
                image_shape=(
                    7980,
                    1200
                )
            )

            results.append(
                result
            )

            print(
                f"  mutual   = "
                f"{result['raw_mutual_matches']}"
            )

            print(
                f"  filtered = "
                f"{result['filtered_matches']}"
            )

            print(
                f"  affine   = "
                f"{result['affine']['inliers']} "
                f"inliers"
            )

            print(
                f"  homography = "
                f"{result['homography']['inliers']} "
                f"inliers"
            )

        all_results[name] = results

        print()

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    json_path = (
        output_dir
        / "benchmark_results.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_results,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for name, results in all_results.items():

        print()
        print(name)

        best_affine = max(
            results,
            key=lambda x:
                x["affine"]["inliers"]
        )

        best_homography = max(
            results,
            key=lambda x:
                x["homography"]["inliers"]
        )

        print(
            "  Best affine:"
        )

        print(
            f"    ratio={best_affine['ratio']:.2f} "
            f"inliers={best_affine['affine']['inliers']} "
            f"ratio={best_affine['affine']['inlier_ratio']:.3f} "
            f"RMSE={best_affine['affine']['rmse_px']}"
        )

        print(
            "  Best homography:"
        )

        print(
            f"    ratio={best_homography['ratio']:.2f} "
            f"inliers={best_homography['homography']['inliers']} "
            f"ratio={best_homography['homography']['inlier_ratio']:.3f} "
            f"RMSE={best_homography['homography']['rmse_px']}"
        )

    print()
    print(
        f"Saved: {json_path}"
    )

    print()
    print(
        "Benchmark complete."
    )


if __name__ == "__main__":
    main()