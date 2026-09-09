from pathlib import Path
import numpy as np
import cv2
import tifffile


# ============================================================
# OFFICIAL OHRC + LROC 1 m/pixel WORKING PAIR
# ============================================================

PROJECT_ROOT = Path(r"C:\Lunar\lunar-image-correspondence")

OHRC_IMG = (
    PROJECT_ROOT
    / "data"
    / "ohrc_official"
    / "pair_001"
    / "data"
    / "calibrated"
    / "20211228"
    / "ch2_ohr_ncp_20211228T2209123959_d_img_d18.img"
)

LROC_TIF = (
    PROJECT_ROOT
    / "data"
    / "lroc_official"
    / "pair_001"
    / "fullres"
    / "NAC_POLE_SOUTH_CM_AVG_P892S2250.TIF"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "official_1m_working_pair"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# OFFICIAL IMAGE PARAMETERS
# ------------------------------------------------------------

OHRC_HEIGHT = 79796
OHRC_WIDTH = 12000

OHRC_GSD_M = 0.28
LROC_GSD_M = 1.0

# Target physical working scale.
TARGET_GSD_M = 1.0


# ------------------------------------------------------------
# OHRC READING
# ------------------------------------------------------------

def load_ohrc_at_approx_1m():

    print("[OHRC] Opening official calibrated image...")

    img = np.memmap(
        OHRC_IMG,
        dtype=np.uint8,
        mode="r",
        shape=(OHRC_HEIGHT, OHRC_WIDTH),
        order="C",
    )

    print(f"[OHRC] Native shape: {img.shape}")

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # We cannot simply use ::3 and call it 1m/pixel.
    #
    # 0.28 m * 3 = 0.84 m
    #
    # Better approach:
    #
    # Read the image in a manageable array and resize using
    # AREA interpolation to the physically appropriate scale.
    #
    # The full OHRC image is too large to load at once.
    #
    # Therefore we first create the same 10x working preview
    # that we already know fits comfortably in memory.
    #
    # This is an APPROXIMATE 2.8 m/pixel diagnostic image,
    # not the final 1m representation.
    #
    # We will then use the geographic crop in the next stage.
    # --------------------------------------------------------

    preview = np.asarray(
        img[::10, ::10],
        dtype=np.uint8,
    )

    print(
        f"[OHRC] 10x preview shape: {preview.shape}"
    )

    return preview


# ------------------------------------------------------------
# LROC READING
# ------------------------------------------------------------

def load_lroc_reference():

    print("[LROC] Opening GeoTIFF with tifffile memmap...")

    lroc = tifffile.memmap(
        str(LROC_TIF)
    )

    print(
        f"[LROC] Full image shape: {lroc.shape}"
    )

    # Existing scientifically selected P892S2250 crop.
    #
    # This is the crop we previously established from the OHRC
    # geometry footprint.
    #
    # Full-resolution coordinates:
    #
    # x = 27014 : 45488
    # y = 0     : 16491
    #
    x0 = 27014
    x1 = 45488

    y0 = 0
    y1 = 16491

    crop = np.asarray(
        lroc[y0:y1, x0:x1],
        dtype=np.uint8,
    )

    print(
        f"[LROC] Crop shape: {crop.shape}"
    )

    return crop


# ------------------------------------------------------------
# APPROXIMATE PHYSICAL RESAMPLING
# ------------------------------------------------------------

def resize_to_target_gsd(image, current_gsd, target_gsd):

    scale = current_gsd / target_gsd

    new_width = max(
        1,
        int(round(image.shape[1] * scale))
    )

    new_height = max(
        1,
        int(round(image.shape[0] * scale))
    )

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )


# ------------------------------------------------------------
# REPRESENTATIONS
# ------------------------------------------------------------

def clahe(image):

    c = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    return c.apply(image)


def gradient_magnitude(image):

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

    mag = cv2.magnitude(gx, gy)

    lo = np.percentile(mag, 1)
    hi = np.percentile(mag, 99)

    if hi <= lo:
        hi = lo + 1

    out = (
        (mag - lo)
        * 255.0
        / (hi - lo)
    )

    return np.clip(
        out,
        0,
        255,
    ).astype(np.uint8)


def multiscale_structure(image):

    img = image.astype(np.float32)

    responses = []

    for sigma in (1.0, 2.0, 4.0):

        blurred = cv2.GaussianBlur(
            img,
            (0, 0),
            sigma,
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

        responses.append(
            cv2.magnitude(gx, gy)
        )

    combined = (
        0.20 * responses[0]
        + 0.30 * responses[1]
        + 0.50 * responses[2]
    )

    lo = np.percentile(combined, 1)
    hi = np.percentile(combined, 99)

    if hi <= lo:
        hi = lo + 1

    out = (
        (combined - lo)
        * 255.0
        / (hi - lo)
    )

    return np.clip(
        out,
        0,
        255,
    ).astype(np.uint8)


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

def save(path, image):

    ok = cv2.imwrite(
        str(path),
        image,
    )

    if not ok:
        raise RuntimeError(
            f"Failed to save {path}"
        )

    print(
        f"[SAVED] {path.name} "
        f"{image.shape}"
    )


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    print("=" * 72)
    print("       OFFICIAL 1 m WORKING PAIR")
    print("=" * 72)

    print()

    if not OHRC_IMG.exists():
        raise FileNotFoundError(
            f"OHRC file not found:\n{OHRC_IMG}"
        )

    if not LROC_TIF.exists():
        raise FileNotFoundError(
            f"LROC file not found:\n{LROC_TIF}"
        )

    # --------------------------------------------------------
    # OHRC
    # --------------------------------------------------------

    ohrc_preview = load_ohrc_at_approx_1m()

    # --------------------------------------------------------
    # LROC
    # --------------------------------------------------------

    lroc_crop = load_lroc_reference()

    # --------------------------------------------------------
    # IMPORTANT
    # --------------------------------------------------------
    #
    # At this stage the OHRC preview is approximately:
    #
    # 0.28m * 10 = 2.8m/pixel
    #
    # while LROC is:
    #
    # 1m/pixel
    #
    # Therefore we DO NOT pretend these are already aligned
    # physically.
    #
    # We are creating comparable representations only.
    #
    # The actual geographic registration will happen later.
    # --------------------------------------------------------

    print()
    print("[INFO] Creating representations...")

    # OHRC representations
    ohrc_original = ohrc_preview

    ohrc_clahe = clahe(
        ohrc_preview
    )

    ohrc_gradient = gradient_magnitude(
        ohrc_preview
    )

    ohrc_multiscale = multiscale_structure(
        ohrc_preview
    )

    # LROC representations
    lroc_original = lroc_crop

    lroc_clahe = clahe(
        lroc_crop
    )

    lroc_gradient = gradient_magnitude(
        lroc_crop
    )

    lroc_multiscale = multiscale_structure(
        lroc_crop
    )

    print()
    print("[INFO] Saving outputs...")

    save(
        OUTPUT_DIR / "ohrc_original.png",
        ohrc_original,
    )

    save(
        OUTPUT_DIR / "ohrc_clahe.png",
        ohrc_clahe,
    )

    save(
        OUTPUT_DIR / "ohrc_gradient_magnitude.png",
        ohrc_gradient,
    )

    save(
        OUTPUT_DIR / "ohrc_multiscale_structure.png",
        ohrc_multiscale,
    )

    save(
        OUTPUT_DIR / "lroc_original.png",
        lroc_original,
    )

    save(
        OUTPUT_DIR / "lroc_clahe.png",
        lroc_clahe,
    )

    save(
        OUTPUT_DIR / "lroc_gradient_magnitude.png",
        lroc_gradient,
    )

    save(
        OUTPUT_DIR / "lroc_multiscale_structure.png",
        lroc_multiscale,
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    report = OUTPUT_DIR / "working_pair_report.txt"

    with open(
        report,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "OFFICIAL OHRC + LROC WORKING PAIR\n"
        )

        f.write(
            "=" * 72 + "\n\n"
        )

        f.write(
            "OHRC:\n"
        )

        f.write(
            f"Native GSD: {OHRC_GSD_M} m/pixel\n"
        )

        f.write(
            f"10x preview effective GSD: "
            f"{OHRC_GSD_M * 10:.3f} m/pixel\n"
        )

        f.write(
            f"Preview shape: "
            f"{ohrc_preview.shape}\n\n"
        )

        f.write(
            "LROC:\n"
        )

        f.write(
            f"GSD: {LROC_GSD_M} m/pixel\n"
        )

        f.write(
            f"Crop shape: "
            f"{lroc_crop.shape}\n"
        )

        f.write(
            "Crop full-resolution coordinates:\n"
        )

        f.write(
            "x=27014:45488, y=0:16491\n\n"
        )

        f.write(
            "NOTE:\n"
        )

        f.write(
            "This experiment creates comparable visual "
            "representations but does NOT establish pixel-level "
            "registration. The OHRC preview is approximately "
            "2.8 m/pixel while the LROC crop is 1 m/pixel.\n"
        )

    print()
    print(
        f"[SAVED] {report}"
    )

    print()
    print("=" * 72)
    print("COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()