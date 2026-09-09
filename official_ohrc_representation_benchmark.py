from pathlib import Path
import numpy as np
import cv2


# ============================================================
# OFFICIAL OHRC REPRESENTATION BENCHMARK
# ============================================================
#
# Purpose:
#   Compare different representations of the official calibrated
#   Chandrayaan-2 OHRC image before we run feature matching.
#
# Input:
#   Official calibrated OHRC .IMG
#
# Output:
#   Several visual representations + quantitative striping report.
#
# IMPORTANT:
#   - The original official_ohrc_preprocessing experiment is untouched.
#   - The full 79,796 x 12,000 image is NEVER loaded into RAM.
#   - numpy.memmap is used.
# ============================================================


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(r"C:\Lunar\lunar-image-correspondence")

INPUT_IMG = (
    PROJECT_ROOT
    / "data"
    / "ohrc_official"
    / "pair_001"
    / "data"
    / "calibrated"
    / "20211228"
    / "ch2_ohr_ncp_20211228T2209123959_d_img_d18.img"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "official_ohrc_representation_benchmark"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# OFFICIAL OHRC DIMENSIONS
# ------------------------------------------------------------

HEIGHT = 79796
WIDTH = 12000

DTYPE = np.uint8

# Sample every 10th native pixel.
# This gives approximately the same scale as the previous
# official calibrated preview:
#
# 79796 / 10 ~= 7980
# 12000 / 10 = 1200
#
SAMPLE_STEP = 10


# ------------------------------------------------------------
# UTILITY FUNCTIONS
# ------------------------------------------------------------

def percentile_range(image, low=1.0, high=99.0):
    """
    Robust intensity range.
    """
    values = image.astype(np.float32)

    lo = float(np.percentile(values, low))
    hi = float(np.percentile(values, high))

    if hi <= lo:
        hi = lo + 1.0

    return lo, hi


def stretch_uint8(image, low=1.0, high=99.0):
    """
    Robustly stretch an image to uint8 using percentile limits.
    """
    image = image.astype(np.float32)

    lo, hi = percentile_range(image, low, high)

    out = (image - lo) * 255.0 / (hi - lo)
    out = np.clip(out, 0, 255)

    return out.astype(np.uint8)


def normalize_float(image):
    """
    Normalize arbitrary floating-point image to [0,255].
    """
    image = image.astype(np.float32)

    lo = float(np.min(image))
    hi = float(np.max(image))

    if hi <= lo:
        return np.zeros_like(image, dtype=np.uint8)

    out = (image - lo) * 255.0 / (hi - lo)
    return np.clip(out, 0, 255).astype(np.uint8)


def save_png(path, image):
    """
    Save image and verify that OpenCV succeeded.
    """
    ok = cv2.imwrite(str(path), image)

    if not ok:
        raise RuntimeError(f"Could not write image: {path}")

    print(f"[SAVED] {path}")


# ------------------------------------------------------------
# STRIPING METRICS
# ------------------------------------------------------------

def column_statistics(image):
    """
    Compute per-column statistics.

    Returns:
        median
        p75
        p90
    """
    img = image.astype(np.float32)

    column_median = np.median(img, axis=0)
    column_p75 = np.percentile(img, 75, axis=0)
    column_p90 = np.percentile(img, 90, axis=0)

    return column_median, column_p75, column_p90


def row_statistics(image):
    """
    Compute per-row median statistics.
    """
    img = image.astype(np.float32)

    return np.median(img, axis=1)


def striping_metrics(image):
    """
    Estimate residual vertical striping.

    We use several measurements because a single metric can
    confuse actual terrain structure with detector striping.

    Metrics:

        column_median_std
            Standard deviation of column medians.

        column_median_range
            P95-P5 of column medians.

        normalized_column_std
            Column median std normalized by global image std.

        neighbor_column_difference
            Median absolute difference between neighboring
            column medians.

    Lower generally means less column-wise variation.
    """

    img = image.astype(np.float32)

    col_median = np.median(img, axis=0)

    global_std = float(np.std(img))

    column_median_std = float(np.std(col_median))

    p5 = float(np.percentile(col_median, 5))
    p95 = float(np.percentile(col_median, 95))

    column_median_range = p95 - p5

    if global_std > 1e-8:
        normalized_column_std = column_median_std / global_std
    else:
        normalized_column_std = 0.0

    neighbor_difference = float(
        np.median(np.abs(np.diff(col_median)))
    )

    return {
        "column_median_std": column_median_std,
        "column_median_p95_minus_p5": column_median_range,
        "normalized_column_median_std": normalized_column_std,
        "neighbor_column_median_difference": neighbor_difference,
        "global_std": global_std,
    }


def structure_metrics(image):
    """
    Measure how much edge/structural information exists.

    This is NOT an accuracy metric.

    It is only useful for comparing representations.
    """

    img = image.astype(np.float32)

    gx = cv2.Sobel(
        img,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        img,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    magnitude = cv2.magnitude(gx, gy)

    return {
        "gradient_mean": float(np.mean(magnitude)),
        "gradient_std": float(np.std(magnitude)),
        "gradient_p95": float(np.percentile(magnitude, 95)),
    }


# ------------------------------------------------------------
# REPRESENTATION 1
# ORIGINAL
# ------------------------------------------------------------

def representation_original(image):
    return image.copy()


# ------------------------------------------------------------
# REPRESENTATION 2
# CLAHE
# ------------------------------------------------------------

def representation_clahe(image):
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    return clahe.apply(image)


# ------------------------------------------------------------
# REPRESENTATION 3
# LOCAL CONTRAST NORMALIZATION
# ------------------------------------------------------------

def representation_local_normalized(image):
    """
    Remove slowly varying illumination.

    We estimate a local background using Gaussian blur and
    normalize the deviation from that background.

    This is deliberately different from simple column
    correction.

    It attacks low-frequency illumination variation rather
    than assuming the artifact is purely column-dependent.
    """

    img = image.astype(np.float32)

    background = cv2.GaussianBlur(
        img,
        ksize=(0, 0),
        sigmaX=25,
        sigmaY=25,
    )

    numerator = img - background

    local_sq = cv2.GaussianBlur(
        numerator * numerator,
        ksize=(0, 0),
        sigmaX=25,
        sigmaY=25,
    )

    local_std = np.sqrt(
        np.maximum(local_sq, 1e-6)
    )

    normalized = numerator / local_std

    return normalize_float(normalized)


# ------------------------------------------------------------
# REPRESENTATION 4
# GRADIENT MAGNITUDE
# ------------------------------------------------------------

def representation_gradient_magnitude(image):
    img = image.astype(np.float32)

    gx = cv2.Sobel(
        img,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        img,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    magnitude = cv2.magnitude(gx, gy)

    return stretch_uint8(
        magnitude,
        low=1,
        high=99,
    )


# ------------------------------------------------------------
# REPRESENTATION 5
# GRADIENT ORIENTATION
# ------------------------------------------------------------

def representation_gradient_orientation(image):
    img = image.astype(np.float32)

    gx = cv2.Sobel(
        img,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        img,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    magnitude = cv2.magnitude(gx, gy)

    angle = cv2.phase(
        gx,
        gy,
        angleInDegrees=False,
    )

    # Encode orientation as uint8.
    orientation = (
        angle / (2.0 * np.pi) * 255.0
    ).astype(np.uint8)

    # Suppress orientation in very weak-gradient regions.
    weak = magnitude < np.percentile(
        magnitude,
        40,
    )

    orientation[weak] = 0

    return orientation


# ------------------------------------------------------------
# REPRESENTATION 6
# MULTI-SCALE STRUCTURE
# ------------------------------------------------------------

def representation_multiscale_structure(image):
    """
    Combine gradient responses at multiple scales.

    The goal is to retain:
        - crater boundaries
        - ridges
        - shadow boundaries
        - larger terrain structures

    while reducing sensitivity to very fine detector noise.
    """

    img = image.astype(np.float32)

    responses = []

    for sigma in (1.0, 2.0, 4.0):

        blurred = cv2.GaussianBlur(
            img,
            ksize=(0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
        )

        gx = cv2.Sobel(
            blurred,
            cv2.CV_32F,
            1,
            0,
            ksize=3,
        )

        gy = cv2.Sobel(
            blurred,
            cv2.CV_32F,
            0,
            1,
            ksize=3,
        )

        magnitude = cv2.magnitude(gx, gy)

        responses.append(magnitude)

    combined = (
        0.20 * responses[0]
        + 0.30 * responses[1]
        + 0.50 * responses[2]
    )

    return stretch_uint8(
        combined,
        low=1,
        high=99,
    )


# ------------------------------------------------------------
# SAVE COLUMN PROFILE
# ------------------------------------------------------------

def save_column_profile(image, filename):
    """
    Save a compact text profile of column statistics.

    We don't generate a matplotlib plot here because the primary
    purpose of this benchmark is numerical comparison.
    """

    col_median, col_p75, col_p90 = column_statistics(image)

    path = OUTPUT_DIR / filename

    with open(path, "w", encoding="utf-8") as f:

        f.write("column_index,median,p75,p90\n")

        for i in range(image.shape[1]):
            f.write(
                f"{i},"
                f"{float(col_median[i]):.6f},"
                f"{float(col_p75[i]):.6f},"
                f"{float(col_p90[i]):.6f}\n"
            )

    print(f"[SAVED] {path}")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    print("=" * 72)
    print("       OFFICIAL OHRC REPRESENTATION BENCHMARK")
    print("=" * 72)

    print()
    print(f"Input:  {INPUT_IMG}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    if not INPUT_IMG.exists():
        raise FileNotFoundError(
            f"Official OHRC IMG not found:\n{INPUT_IMG}"
        )

    # --------------------------------------------------------
    # MEMMAP
    # --------------------------------------------------------

    print("[1/7] Opening official calibrated OHRC with memmap...")

    image = np.memmap(
        INPUT_IMG,
        dtype=DTYPE,
        mode="r",
        shape=(HEIGHT, WIDTH),
        order="C",
    )

    print(f"      Native shape: {image.shape}")
    print(f"      Native dtype: {image.dtype}")

    # --------------------------------------------------------
    # SAMPLE
    # --------------------------------------------------------

    print()
    print("[2/7] Creating 10x working representation...")

    working = np.asarray(
        image[::SAMPLE_STEP, ::SAMPLE_STEP],
        dtype=np.uint8,
    )

    print(f"      Working shape: {working.shape}")
    print(
        f"      Min={int(working.min())} "
        f"Max={int(working.max())} "
        f"Mean={float(working.mean()):.4f} "
        f"Median={float(np.median(working)):.4f}"
    )

    save_png(
        OUTPUT_DIR / "01_original.png",
        working,
    )

    # --------------------------------------------------------
    # REPRESENTATIONS
    # --------------------------------------------------------

    print()
    print("[3/7] Generating representations...")

    representations = {}

    print("      - Original")
    representations["original"] = representation_original(
        working
    )

    print("      - CLAHE")
    representations["clahe"] = representation_clahe(
        working
    )

    print("      - Local normalized")
    representations["local_normalized"] = (
        representation_local_normalized(
            working
        )
    )

    print("      - Gradient magnitude")
    representations["gradient_magnitude"] = (
        representation_gradient_magnitude(
            working
        )
    )

    print("      - Gradient orientation")
    representations["gradient_orientation"] = (
        representation_gradient_orientation(
            working
        )
    )

    print("      - Multi-scale structure")
    representations["multiscale_structure"] = (
        representation_multiscale_structure(
            working
        )
    )

    # --------------------------------------------------------
    # SAVE IMAGES
    # --------------------------------------------------------

    print()
    print("[4/7] Saving representations...")

    filenames = {
        "original": "01_original.png",
        "clahe": "02_clahe.png",
        "local_normalized": "03_local_normalized.png",
        "gradient_magnitude": "04_gradient_magnitude.png",
        "gradient_orientation": "05_gradient_orientation.png",
        "multiscale_structure": "06_multiscale_structure.png",
    }

    for name, rep in representations.items():

        if name == "original":
            continue

        save_png(
            OUTPUT_DIR / filenames[name],
            rep,
        )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    print()
    print("[5/7] Computing representation metrics...")

    metrics = {}

    for name, rep in representations.items():

        print(f"      {name}")

        metrics[name] = {}

        metrics[name]["height"] = int(rep.shape[0])
        metrics[name]["width"] = int(rep.shape[1])

        metrics[name]["min"] = int(rep.min())
        metrics[name]["max"] = int(rep.max())
        metrics[name]["mean"] = float(np.mean(rep))
        metrics[name]["median"] = float(np.median(rep))

        metrics[name]["striping"] = striping_metrics(
            rep
        )

        metrics[name]["structure"] = structure_metrics(
            rep
        )

    # --------------------------------------------------------
    # COLUMN PROFILES
    # --------------------------------------------------------

    print()
    print("[6/7] Saving column profiles...")

    for name, rep in representations.items():

        save_column_profile(
            rep,
            f"{name}_column_profile.csv",
        )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print()
    print("[7/7] Writing benchmark report...")

    report_path = OUTPUT_DIR / "representation_benchmark_report.txt"

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "OFFICIAL OHRC REPRESENTATION BENCHMARK\n"
        )

        f.write("=" * 72 + "\n\n")

        f.write(
            f"Input:\n{INPUT_IMG}\n\n"
        )

        f.write(
            f"Native shape: {HEIGHT} x {WIDTH}\n"
        )

        f.write(
            f"Sampling step: {SAMPLE_STEP}\n"
        )

        f.write(
            f"Working shape: "
            f"{working.shape[0]} x {working.shape[1]}\n\n"
        )

        f.write(
            "IMPORTANT:\n"
        )

        f.write(
            "These metrics are diagnostic only. They do NOT "
            "measure registration accuracy.\n\n"
        )

        for name in representations:

            m = metrics[name]

            f.write("-" * 72 + "\n")
            f.write(f"REPRESENTATION: {name}\n")
            f.write("-" * 72 + "\n")

            f.write(
                f"min:              {m['min']}\n"
            )

            f.write(
                f"max:              {m['max']}\n"
            )

            f.write(
                f"mean:             {m['mean']:.6f}\n"
            )

            f.write(
                f"median:           {m['median']:.6f}\n"
            )

            f.write("\n")

            s = m["striping"]

            f.write(
                "STRIPING METRICS\n"
            )

            f.write(
                f"column_median_std: "
                f"{s['column_median_std']:.6f}\n"
            )

            f.write(
                f"column_median_p95_minus_p5: "
                f"{s['column_median_p95_minus_p5']:.6f}\n"
            )

            f.write(
                f"normalized_column_median_std: "
                f"{s['normalized_column_median_std']:.6f}\n"
            )

            f.write(
                f"neighbor_column_median_difference: "
                f"{s['neighbor_column_median_difference']:.6f}\n"
            )

            f.write(
                f"global_std: "
                f"{s['global_std']:.6f}\n"
            )

            f.write("\n")

            st = m["structure"]

            f.write(
                "STRUCTURE METRICS\n"
            )

            f.write(
                f"gradient_mean: "
                f"{st['gradient_mean']:.6f}\n"
            )

            f.write(
                f"gradient_std: "
                f"{st['gradient_std']:.6f}\n"
            )

            f.write(
                f"gradient_p95: "
                f"{st['gradient_p95']:.6f}\n"
            )

            f.write("\n")

    print(f"[SAVED] {report_path}")

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("BENCHMARK COMPLETE")
    print("=" * 72)

    print()
    print("Output directory:")
    print(OUTPUT_DIR)

    print()
    print("Generated files:")

    for p in sorted(OUTPUT_DIR.iterdir()):

        if p.is_file():
            print(
                f"  {p.name:<45} "
                f"{p.stat().st_size:,} bytes"
            )

    print()
    print(
        "Next step: inspect the six PNG representations "
        "and the numerical report."
    )


if __name__ == "__main__":
    main()