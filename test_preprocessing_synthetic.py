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
    save_preprocessing_results,
)


# ============================================================
# OUTPUT
# ============================================================

OUT = (
    PROJECT_ROOT
    / "experiments"
    / "preprocessing"
    / "synthetic_tests"
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

    # Broad terrain variation
    image = (
        75.0
        + 16.0 * np.sin(x / 95.0)
        + 10.0 * np.sin(y / 71.0)
        + 7.0 * np.sin((x + y) / 43.0)
    )

    # Fine texture
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

    # Broad terrain texture
    broad_noise = rng.normal(
        0,
        1,
        (height, width),
    ).astype(np.float32)

    broad_noise = cv2.GaussianBlur(
        broad_noise,
        (0, 0),
        15.0,
    )

    image += broad_noise * 18.0

    # --------------------------------------------------------
    # Craters
    # --------------------------------------------------------

    craters = [
        (210, 190, 80, 22),
        (510, 180, 125, 28),
        (800, 270, 65, 18),
        (330, 480, 115, 30),
        (720, 510, 150, 35),
        (890, 650, 70, 20),
        (130, 650, 55, 17),
    ]

    yy, xx = np.mgrid[0:height, 0:width]

    for cx, cy, radius, depth in craters:

        distance = np.sqrt(
            (xx - cx) ** 2
            + (yy - cy) ** 2
        )

        # Crater interior
        interior = np.exp(
            -(distance / (radius * 0.72)) ** 2
        )

        image -= depth * interior

        # Bright rim
        rim = np.exp(
            -(
                (distance - radius)
                / (radius * 0.10)
            ) ** 2
        )

        image += depth * 0.75 * rim

        # Directional shadow
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
    # Ridges
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

    # Mild nonlinear response
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

def add_ohrc_banding(
    image,
    strength=0.35,
    seed=123,
):

    rng = np.random.default_rng(seed)

    height, width = image.shape

    # High-frequency detector variation
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

    # Broad column variation
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

    # Small column offset
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
# 5. GEOMETRIC DISTORTION
# ============================================================

def apply_geometric_distortion(
    image,
    scale=1.0,
    angle=0.0,
    tx=0.0,
    ty=0.0,
    perspective=0.0,
):

    height, width = image.shape

    center = (
        width / 2.0,
        height / 2.0,
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        scale,
    )

    matrix[:, 2] += [
        tx,
        ty,
    ]

    rotated = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    if perspective == 0:
        return rotated

    src = np.float32([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ])

    p = perspective

    dst = np.float32([
        [p, 0],
        [
            width - 1 - p * 0.6,
            p * 0.35,
        ],
        [width - 1, height - 1],
        [
            p * 0.4,
            height - 1 - p * 0.25,
        ],
    ])

    transform = cv2.getPerspectiveTransform(
        src,
        dst,
    )

    return cv2.warpPerspective(
        rotated,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


# ============================================================
# DIAGNOSTICS
# ============================================================

def gradient_stats(image):

    image_f = image.astype(np.float32)

    gx = cv2.Sobel(
        image_f,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        image_f,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    magnitude = cv2.magnitude(
        gx,
        gy,
    )

    return {
        "gradient_mean": float(
            np.mean(magnitude)
        ),
        "gradient_p95": float(
            np.percentile(
                magnitude,
                95,
            )
        ),
    }


def stripe_score(image):

    image_f = image.astype(
        np.float32
    )

    profile = np.median(
        image_f,
        axis=0,
    )

    smooth = cv2.GaussianBlur(
        profile.reshape(1, -1),
        (0, 0),
        12,
    ).ravel()

    residual = (
        profile - smooth
    )

    return float(
        np.std(residual)
    )


# ============================================================
# RUN ONE TEST
# ============================================================

def run_test(
    name,
    image,
    sensor="generic",
    destripe=False,
):

    output = (
        OUT / name
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_path = (
        output / "input.png"
    )

    cv2.imwrite(
        str(input_path),
        image,
    )

    results = preprocess_image(
        path=str(input_path),
        sensor=sensor,
        use_clahe=True,
        use_gradient=True,
        use_destriping=destripe,
        pyramid_levels=3,
    )

    save_preprocessing_results(
        results,
        str(output / "preprocessed"),
    )

    cv2.imwrite(
        str(output / "normalized.png"),
        results["normalized"],
    )

    cv2.imwrite(
        str(output / "clahe.png"),
        results["clahe"],
    )

    cv2.imwrite(
        str(output / "structural.png"),
        results["structural"],
    )

    return {
        "input_mean":
            float(np.mean(image)),

        "input_std":
            float(np.std(image)),

        "input_stripe_score":
            stripe_score(image),

        "normalized_stripe_score":
            stripe_score(
                results["normalized"]
            ),

        "clahe_stripe_score":
            stripe_score(
                results["clahe"]
            ),

        "structural_stripe_score":
            stripe_score(
                results["structural"]
            ),

        "structural_gradient_mean":
            gradient_stats(
                results["structural"]
            )["gradient_mean"],
    }


# ============================================================
# DESTRIPING ABLATION
# ============================================================

def run_destriping_ablation(
    name,
    image,
):

    output = (
        OUT
        / name
        / "destripe_comparison"
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_path = (
        output / "input.png"
    )

    cv2.imwrite(
        str(input_path),
        image,
    )

    without = preprocess_image(
        str(input_path),
        sensor="OHRC",
        use_clahe=True,
        use_gradient=True,
        use_destriping=False,
        pyramid_levels=3,
    )

    with_destripe = preprocess_image(
        str(input_path),
        sensor="OHRC",
        use_clahe=True,
        use_gradient=True,
        use_destriping=True,
        pyramid_levels=3,
    )

    cv2.imwrite(
        str(
            output
            / "structural_without_destripe.png"
        ),
        without["structural"],
    )

    cv2.imwrite(
        str(
            output
            / "structural_with_destripe.png"
        ),
        with_destripe["structural"],
    )

    score_without = stripe_score(
        without["normalized"]
    )

    score_with = stripe_score(
        with_destripe["normalized"]
    )

    return (
        score_without,
        score_with,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("SYNTHETIC PREPROCESSING TEST")
    print("=" * 72)

    clean = make_base_scene()

    # --------------------------------------------------------
    # T1: baseline
    # --------------------------------------------------------

    t1 = clean.copy()

    # --------------------------------------------------------
    # T2: illumination
    # --------------------------------------------------------

    t2 = add_illumination_change(
        clean
    )

    # --------------------------------------------------------
    # T3: OHRC banding
    # --------------------------------------------------------

    t3 = add_ohrc_banding(
        clean,
        strength=0.35,
    )

    # --------------------------------------------------------
    # T4: geometry
    # --------------------------------------------------------

    t4 = apply_geometric_distortion(
        clean,
        scale=1.28,
        angle=11.0,
        tx=35,
        ty=-22,
        perspective=10,
    )

    # --------------------------------------------------------
    # T5: combined difficult case
    # --------------------------------------------------------

    t5 = add_noise(
        add_ohrc_banding(
            add_illumination_change(
                apply_geometric_distortion(
                    clean,
                    scale=1.18,
                    angle=-8.0,
                    tx=-28,
                    ty=20,
                    perspective=8,
                )
            ),
            strength=0.35,
            seed=999,
        ),
        sigma=8.0,
        seed=777,
    )

    tests = [
        (
            "T1_baseline",
            t1,
            "generic",
            False,
        ),
        (
            "T2_illumination",
            t2,
            "generic",
            False,
        ),
        (
            "T3_ohrc_banding",
            t3,
            "OHRC",
            True,
        ),
        (
            "T4_scale_rotation",
            t4,
            "generic",
            False,
        ),
        (
            "T5_combined",
            t5,
            "OHRC",
            True,
        ),
    ]

    metrics = {}

    for (
        name,
        image,
        sensor,
        destripe,
    ) in tests:

        print()
        print("-" * 72)
        print(name)
        print(
            f"sensor={sensor}, "
            f"destripe={destripe}"
        )

        result = run_test(
            name,
            image,
            sensor=sensor,
            destripe=destripe,
        )

        metrics[name] = result

        for key, value in result.items():
            print(
                f"{key}: {value:.6f}"
            )

    # --------------------------------------------------------
    # Destriping comparison
    # --------------------------------------------------------

    ablation_lines = []

    for name, image in [
        ("T3_ohrc_banding", t3),
        ("T5_combined", t5),
    ]:

        before, after = (
            run_destriping_ablation(
                name,
                image,
            )
        )

        change = (
            after - before
        )

        ablation_lines.append(
            f"{name}: "
            f"without={before:.6f}, "
            f"with={after:.6f}, "
            f"change={change:+.6f}"
        )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = []

    report.append(
        "SYNTHETIC PREPROCESSING TEST REPORT"
    )

    report.append(
        "=" * 72
    )

    report.append("")

    report.append(
        "Existing preprocessing implementation "
        "was tested without modification."
    )

    report.append(
        "These tests evaluate preprocessing behavior "
        "only; they do NOT demonstrate registration success."
    )

    report.append("")

    for name, result in metrics.items():

        report.append(
            "-" * 72
        )

        report.append(name)

        for key, value in result.items():

            report.append(
                f"{key}: {value:.6f}"
            )

    report.append("")

    report.append(
        "-" * 72
    )

    report.append(
        "DESTRIPING ABLATION"
    )

    report.extend(
        ablation_lines
    )

    report.append("")

    report.append(
        "INTERPRETATION"
    )

    report.append(
        "Lower stripe score generally indicates "
        "less high-frequency vertical column variation."
    )

    report.append(
        "Gradient changes must be interpreted carefully: "
        "higher gradient magnitude can indicate useful "
        "edge enhancement or noise amplification."
    )

    report.append(
        "T4/T5 geometric distortion is NOT a registration test."
    )

    report.append(
        "Synthetic results must NOT be presented as evidence "
        "that the method works on real Chandrayaan-2/LROC data."
    )

    report_path = (
        OUT
        / "synthetic_preprocessing_report.txt"
    )

    report_path.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("COMPLETE")
    print("=" * 72)
    print(
        f"Outputs: {OUT}"
    )
    print(
        f"Report : {report_path}"
    )


if __name__ == "__main__":
    main()