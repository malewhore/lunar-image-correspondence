from pathlib import Path
import sys
import cv2
import numpy as np

# ============================================================
# PROJECT IMPORT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC))

from preprocessing.preprocess import (
    preprocess_image,
)


# ============================================================
# OUTPUT
# ============================================================

OUT = (
    PROJECT_ROOT
    / "experiments"
    / "preprocessing"
    / "synthetic_tests"
    / "T6_correspondence"
)


# ============================================================
# 1. SYNTHETIC LUNAR-LIKE TERRAIN
# ============================================================

def make_base_scene(
    height=768,
    width=1024,
    seed=42,
):
    rng = np.random.default_rng(seed)

    y, x = np.mgrid[0:height, 0:width]

    # Broad terrain variation.
    image = (
        75.0
        + 16.0 * np.sin(x / 95.0)
        + 10.0 * np.sin(y / 71.0)
        + 7.0 * np.sin((x + y) / 43.0)
    )

    # Fine texture.
    noise = rng.normal(
        0,
        1,
        (height, width),
    ).astype(np.float32)

    noise = cv2.GaussianBlur(
        noise,
        (0, 0),
        2.0,
    )

    image += noise * 8.0

    # Broad terrain texture.
    broad = rng.normal(
        0,
        1,
        (height, width),
    ).astype(np.float32)

    broad = cv2.GaussianBlur(
        broad,
        (0, 0),
        15.0,
    )

    image += broad * 18.0

    # --------------------------------------------------------
    # Craters
    # --------------------------------------------------------

    yy, xx = np.mgrid[0:height, 0:width]

    craters = [
        (210, 190, 80, 22),
        (510, 180, 125, 28),
        (800, 270, 65, 18),
        (330, 480, 115, 30),
        (720, 510, 150, 35),
        (890, 650, 70, 20),
        (130, 650, 55, 17),
    ]

    for cx, cy, radius, depth in craters:

        distance = np.sqrt(
            (xx - cx) ** 2
            + (yy - cy) ** 2
        )

        # Crater interior.
        interior = np.exp(
            -(distance / (radius * 0.72)) ** 2
        )

        image -= depth * interior

        # Bright rim.
        rim = np.exp(
            -(
                (distance - radius)
                / (radius * 0.10)
            ) ** 2
        )

        image += depth * 0.75 * rim

        # Directional shadow.
        shadow = np.exp(
            -(
                ((xx - (cx + 0.32 * radius))
                 / (0.60 * radius)) ** 2
                +
                ((yy - (cy + 0.50 * radius))
                 / (0.95 * radius)) ** 2
            )
        )

        image -= depth * 0.65 * shadow

    # --------------------------------------------------------
    # Ridges.
    # --------------------------------------------------------

    ridges = [
        (0.22, 240, 18, 8),
        (-0.48, 620, 13, 11),
        (0.72, 850, 10, 7),
    ]

    for angle, offset, strength, width in ridges:

        distance = np.abs(
            yy
            - (
                np.tan(angle) * xx
                + offset
            )
        )

        ridge = np.exp(
            -(distance / width) ** 2
        )

        image += strength * ridge

    # Normalize to uint8.
    image = cv2.normalize(
        image,
        None,
        8,
        245,
        cv2.NORM_MINMAX,
    )

    return image.astype(np.uint8)


# ============================================================
# 2. ILLUMINATION CHANGE
# ============================================================

def add_illumination_change(image):

    height, width = image.shape

    y, x = np.mgrid[
        0:height,
        0:width,
    ]

    # Smooth illumination field.
    field = (
        0.60
        + 0.85
        * (
            0.35 * x / max(width - 1, 1)
            + 0.65 * y / max(height - 1, 1)
        )
    )

    result = (
        image.astype(np.float32)
        * field
    )

    # Mild nonlinear response.
    result = (
        np.power(
            np.clip(
                result / 255.0,
                0,
                1,
            ),
            0.78,
        )
        * 255.0
    )

    return np.clip(
        result,
        0,
        255,
    ).astype(np.uint8)


# ============================================================
# 3. OHRC-LIKE VERTICAL BANDING
# ============================================================

def add_banding(
    image,
    strength=0.35,
    seed=123,
):

    rng = np.random.default_rng(seed)

    height, width = image.shape

    # High-frequency column variation.
    high = rng.normal(
        0,
        1,
        width,
    ).astype(np.float32)

    high = cv2.GaussianBlur(
        high.reshape(1, -1),
        (0, 0),
        1.2,
    ).ravel()

    # Broad column variation.
    broad = rng.normal(
        0,
        1,
        width,
    ).astype(np.float32)

    broad = cv2.GaussianBlur(
        broad.reshape(1, -1),
        (0, 0),
        22,
    ).ravel()

    pattern = (
        0.60 * high
        + 0.40 * broad
    )

    pattern /= max(
        float(np.std(pattern)),
        1e-6,
    )

    gain = (
        1.0
        + strength
        * 0.30
        * pattern
    )

    gain = np.clip(
        gain,
        0.45,
        1.55,
    )

    result = (
        image.astype(np.float32)
        * gain[None, :]
    )

    # Small column offset.
    offset = rng.normal(
        0,
        2.5,
        width,
    ).astype(np.float32)

    offset = cv2.GaussianBlur(
        offset.reshape(1, -1),
        (0, 0),
        0.8,
    ).ravel()

    result += offset[None, :]

    return np.clip(
        result,
        0,
        255,
    ).astype(np.uint8)


# ============================================================
# 4. NOISE
# ============================================================

def add_noise(
    image,
    sigma=7.0,
    seed=321,
):

    rng = np.random.default_rng(seed)

    result = (
        image.astype(np.float32)
        + rng.normal(
            0,
            sigma,
            image.shape,
        ).astype(np.float32)
    )

    return np.clip(
        result,
        0,
        255,
    ).astype(np.uint8)


# ============================================================
# 5. KNOWN GEOMETRIC TRANSFORM
# ============================================================

def make_transform(
    width,
    height,
):

    center = (
        width / 2.0,
        height / 2.0,
    )

    # IMPORTANT:
    #
    # This matrix maps:
    #
    #     reference -> moving
    #
    # Therefore, when evaluating SIFT matches, which are
    # moving -> reference, we MUST use its inverse.
    #

    M = cv2.getRotationMatrix2D(
        center,
        7.0,
        1.12,
    )

    M[:, 2] += [
        28.0,
        -18.0,
    ]

    return M


def warp_with_transform(
    image,
    M,
):

    height, width = image.shape

    return cv2.warpAffine(
        image,
        M,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


# ============================================================
# 6. TRANSFORM POINTS
# ============================================================

def transform_points(
    points,
    M,
):

    points_h = np.hstack([
        points.astype(np.float32),
        np.ones(
            (len(points), 1),
            dtype=np.float32,
        ),
    ])

    transformed = (
        points_h @ M.T
    )

    return transformed


# ============================================================
# 7. SIFT MATCHING
# ============================================================

def run_sift(
    image_a,
    image_b,
    ratio=0.80,
):

    sift = cv2.SIFT_create(
        nfeatures=10000,
        contrastThreshold=0.01,
        edgeThreshold=30,
        sigma=1.6,
    )

    kp_a, des_a = sift.detectAndCompute(
        image_a,
        None,
    )

    kp_b, des_b = sift.detectAndCompute(
        image_b,
        None,
    )

    if des_a is None or des_b is None:

        return {
            "keypoints_a": len(kp_a),
            "keypoints_b": len(kp_b),
            "matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "rmse": None,
            "points_a": None,
            "points_b": None,
        }

    matcher = cv2.BFMatcher(
        cv2.NORM_L2
    )

    knn = matcher.knnMatch(
        des_a,
        des_b,
        k=2,
    )

    good = []

    for pair in knn:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < ratio * n.distance:
            good.append(m)

    if len(good) < 4:

        return {
            "keypoints_a": len(kp_a),
            "keypoints_b": len(kp_b),
            "matches": len(good),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "rmse": None,
            "points_a": None,
            "points_b": None,
        }

    pts_a = np.float32([
        kp_a[m.queryIdx].pt
        for m in good
    ])

    pts_b = np.float32([
        kp_b[m.trainIdx].pt
        for m in good
    ])

    # SIFT direction:
    #
    #     image_a -> image_b
    #
    # Here:
    #
    #     moving -> reference
    #
    H, mask = cv2.findHomography(
        pts_a,
        pts_b,
        cv2.RANSAC,
        5.0,
        maxIters=5000,
        confidence=0.995,
    )

    if H is None or mask is None:

        return {
            "keypoints_a": len(kp_a),
            "keypoints_b": len(kp_b),
            "matches": len(good),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "rmse": None,
            "points_a": pts_a,
            "points_b": pts_b,
        }

    mask = mask.ravel().astype(bool)

    inlier_a = pts_a[mask]
    inlier_b = pts_b[mask]

    projected = cv2.perspectiveTransform(
        inlier_a.reshape(-1, 1, 2),
        H,
    ).reshape(-1, 2)

    errors = np.linalg.norm(
        projected - inlier_b,
        axis=1,
    )

    rmse = float(
        np.sqrt(
            np.mean(errors ** 2)
        )
    )

    return {
        "keypoints_a": len(kp_a),
        "keypoints_b": len(kp_b),
        "matches": len(good),
        "inliers": int(np.sum(mask)),
        "inlier_ratio": float(
            np.sum(mask) / len(good)
        ),
        "rmse": rmse,
        "points_a": pts_a,
        "points_b": pts_b,
        "H": H,
        "inlier_mask": mask,
    }


# ============================================================
# 8. GROUND-TRUTH ERROR
# ============================================================

def ground_truth_error(
    points_a,
    points_b,
    reference_to_moving,
):

    if (
        points_a is None
        or len(points_a) == 0
    ):
        return None

    # --------------------------------------------------------
    # CRITICAL DIRECTION FIX
    #
    # points_a are in MOVING coordinates.
    #
    # The known M maps:
    #
    #     REFERENCE -> MOVING
    #
    # Therefore:
    #
    #     M^-1 maps MOVING -> REFERENCE
    #
    # which is exactly the direction of the SIFT matches.
    # --------------------------------------------------------

    moving_to_reference = (
        cv2.invertAffineTransform(
            reference_to_moving
        )
    )

    expected = transform_points(
        points_a,
        moving_to_reference,
    )

    errors = np.linalg.norm(
        expected - points_b,
        axis=1,
    )

    return {
        "mean": float(
            np.mean(errors)
        ),
        "median": float(
            np.median(errors)
        ),
        "p95": float(
            np.percentile(
                errors,
                95,
            )
        ),
        "rmse": float(
            np.sqrt(
                np.mean(
                    errors ** 2
                )
            )
        ),
    }


# ============================================================
# 9. PREPROCESS BOTH IMAGES
# ============================================================

def preprocess_pair(
    reference,
    moving,
    use_destripe,
):

    ref_path = (
        OUT
        / "reference_input.png"
    )

    mov_path = (
        OUT
        / "moving_input.png"
    )

    cv2.imwrite(
        str(ref_path),
        reference,
    )

    cv2.imwrite(
        str(mov_path),
        moving,
    )

    reference_results = (
        preprocess_image(
            str(ref_path),
            sensor="OHRC",
            use_clahe=True,
            use_gradient=True,
            use_destriping=use_destripe,
            pyramid_levels=3,
        )
    )

    moving_results = (
        preprocess_image(
            str(mov_path),
            sensor="OHRC",
            use_clahe=True,
            use_gradient=True,
            use_destriping=use_destripe,
            pyramid_levels=3,
        )
    )

    return (
        reference_results,
        moving_results,
    )


# ============================================================
# 10. PRINT RESULT
# ============================================================

def print_result(
    title,
    result,
    gt,
):

    print()
    print("-" * 72)
    print(title)

    print(
        f"Keypoints moving : "
        f"{result['keypoints_a']}"
    )

    print(
        f"Keypoints ref    : "
        f"{result['keypoints_b']}"
    )

    print(
        f"Matches          : "
        f"{result['matches']}"
    )

    print(
        f"Inliers          : "
        f"{result['inliers']}"
    )

    print(
        f"Inlier ratio     : "
        f"{result['inlier_ratio']:.6f}"
    )

    print(
        f"Fit RMSE         : "
        f"{result['rmse']}"
    )

    if gt is None:

        print(
            "Ground-truth     : N/A"
        )

    else:

        print(
            f"GT mean error    : "
            f"{gt['mean']:.6f} px"
        )

        print(
            f"GT median error  : "
            f"{gt['median']:.6f} px"
        )

        print(
            f"GT P95 error     : "
            f"{gt['p95']:.6f} px"
        )

        print(
            f"GT RMSE          : "
            f"{gt['rmse']:.6f} px"
        )


# ============================================================
# 11. MAIN
# ============================================================

def main():

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print(
        "T6 - CORRESPONDENCE PRESERVATION TEST"
    )
    print("=" * 72)

    print(
        "Synthetic reference and moving image have "
        "a known geometric transform."
    )

    print(
        "Moving image additionally contains illumination "
        "change, OHRC-like banding and Gaussian noise."
    )

    print()
    print(
        "IMPORTANT: this evaluates preprocessing + SIFT "
        "on synthetic data only."
    )

    # --------------------------------------------------------
    # Generate clean reference.
    # --------------------------------------------------------

    reference = make_base_scene()

    # --------------------------------------------------------
    # Known reference -> moving transform.
    # --------------------------------------------------------

    reference_to_moving = (
        make_transform(
            reference.shape[1],
            reference.shape[0],
        )
    )

    # --------------------------------------------------------
    # Generate geometrically transformed moving image.
    # --------------------------------------------------------

    moving = warp_with_transform(
        reference,
        reference_to_moving,
    )

    # --------------------------------------------------------
    # Add photometric degradation.
    # --------------------------------------------------------

    moving = add_illumination_change(
        moving
    )

    moving = add_banding(
        moving,
        strength=0.35,
    )

    moving = add_noise(
        moving,
        sigma=7.0,
    )

    # --------------------------------------------------------
    # Save synthetic inputs.
    # --------------------------------------------------------

    cv2.imwrite(
        str(
            OUT
            / "reference_clean.png"
        ),
        reference,
    )

    cv2.imwrite(
        str(
            OUT
            / "moving_degraded.png"
        ),
        moving,
    )

    # --------------------------------------------------------
    # RAW
    # --------------------------------------------------------

    raw = run_sift(
        moving,
        reference,
    )

    raw_gt = ground_truth_error(
        raw.get("points_a"),
        raw.get("points_b"),
        reference_to_moving,
    )

    print_result(
        "RAW -> RAW",
        raw,
        raw_gt,
    )

    # --------------------------------------------------------
    # PREPROCESS WITHOUT DESTRIPING
    # --------------------------------------------------------

    (
        ref_no,
        mov_no,
    ) = preprocess_pair(
        reference,
        moving,
        use_destripe=False,
    )

    processed_no = run_sift(
        mov_no["structural"],
        ref_no["structural"],
    )

    gt_no = ground_truth_error(
        processed_no.get("points_a"),
        processed_no.get("points_b"),
        reference_to_moving,
    )

    print_result(
        "PREPROCESSED "
        "(CLAHE + STRUCTURAL, NO DESTRIPE)",
        processed_no,
        gt_no,
    )

    # --------------------------------------------------------
    # PREPROCESS WITH DESTRIPING
    # --------------------------------------------------------

    (
        ref_ds,
        mov_ds,
    ) = preprocess_pair(
        reference,
        moving,
        use_destripe=True,
    )

    processed_ds = run_sift(
        mov_ds["structural"],
        ref_ds["structural"],
    )

    gt_ds = ground_truth_error(
        processed_ds.get("points_a"),
        processed_ds.get("points_b"),
        reference_to_moving,
    )

    print_result(
        "PREPROCESSED "
        "(DESTRIPE + CLAHE + STRUCTURAL)",
        processed_ds,
        gt_ds,
    )

    # ========================================================
    # REPORT
    # ========================================================

    def gt_text(gt):

        if gt is None:
            return "N/A"

        return (
            f"mean={gt['mean']:.6f}, "
            f"median={gt['median']:.6f}, "
            f"P95={gt['p95']:.6f}, "
            f"RMSE={gt['rmse']:.6f}"
        )

    report = []

    report.append(
        "T6 CORRESPONDENCE PRESERVATION"
    )

    report.append(
        "=" * 72
    )

    report.append(
        "Synthetic reference and moving image have "
        "a known reference->moving affine transform."
    )

    report.append(
        "Moving image additionally contains illumination "
        "change, OHRC-like banding and Gaussian noise."
    )

    report.append("")

    report.append(
        "GROUND-TRUTH DIRECTION"
    )

    report.append(
        "Known transform is REFERENCE -> MOVING."
    )

    report.append(
        "SIFT matches are MOVING -> REFERENCE."
    )

    report.append(
        "Therefore ground truth uses inverse affine transform."
    )

    report.append("")

    report.append(
        "RAW -> RAW"
    )

    report.append(
        f"matches={raw['matches']}"
    )

    report.append(
        f"inliers={raw['inliers']}"
    )

    report.append(
        f"inlier_ratio={raw['inlier_ratio']:.6f}"
    )

    report.append(
        f"fit_rmse={raw['rmse']}"
    )

    report.append(
        f"ground_truth={gt_text(raw_gt)}"
    )

    report.append("")

    report.append(
        "PREPROCESSED WITHOUT DESTRIPING"
    )

    report.append(
        f"matches={processed_no['matches']}"
    )

    report.append(
        f"inliers={processed_no['inliers']}"
    )

    report.append(
        f"inlier_ratio={processed_no['inlier_ratio']:.6f}"
    )

    report.append(
        f"fit_rmse={processed_no['rmse']}"
    )

    report.append(
        f"ground_truth={gt_text(gt_no)}"
    )

    report.append("")

    report.append(
        "PREPROCESSED WITH DESTRIPING"
    )

    report.append(
        f"matches={processed_ds['matches']}"
    )

    report.append(
        f"inliers={processed_ds['inliers']}"
    )

    report.append(
        f"inlier_ratio={processed_ds['inlier_ratio']:.6f}"
    )

    report.append(
        f"fit_rmse={processed_ds['rmse']}"
    )

    report.append(
        f"ground_truth={gt_text(gt_ds)}"
    )

    report.append("")

    report.append(
        "INTERPRETATION"
    )

    report.append(
        "Use correspondence count, inlier ratio and "
        "ground-truth geometric error together."
    )

    report.append(
        "A low RANSAC fit RMSE alone does not prove "
        "that the recovered correspondences are correct."
    )

    report.append(
        "Synthetic results do not establish performance "
        "on real Chandrayaan-2/LROC imagery."
    )

    report_path = (
        OUT
        / "T6_report.txt"
    )

    report_path.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print(
        f"Report saved to:\n{report_path}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()