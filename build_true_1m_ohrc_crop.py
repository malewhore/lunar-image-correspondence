from pathlib import Path
import csv
import math

import numpy as np
import cv2


# ============================================================
# TRUE ~1 m/pixel OHRC GEOGRAPHIC CROP
# ============================================================
#
# Uses:
#   1. Official calibrated OHRC IMG
#   2. Official OHRC geometry CSV
#
# Purpose:
#   Find the OHRC native-pixel region that geographically
#   corresponds to the LROC P892S2250 region, then resample
#   that OHRC crop from ~0.28 m/pixel to ~1 m/pixel.
#
# IMPORTANT:
#   The geometry is used as a GEOGRAPHIC OVERLAP PRIOR.
#   It is NOT treated as pixel-level ground truth.
#
# ============================================================


PROJECT_ROOT = Path(
    r"C:\Lunar\lunar-image-correspondence"
)

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

GEOMETRY_CSV = (
    PROJECT_ROOT
    / "data"
    / "ohrc_official"
    / "pair_001"
    / "geometry"
    / "calibrated"
    / "20211228"
    / "ch2_ohr_ncp_20211228T2209123959_g_grd_d18.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "true_1m_ohrc_crop"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------
# OFFICIAL OHRC IMAGE
# ------------------------------------------------------------

HEIGHT = 79796
WIDTH = 12000

OHRC_GSD = 0.28
TARGET_GSD = 1.0


# ------------------------------------------------------------
# LROC P892S2250 REGION
# ------------------------------------------------------------
#
# From our previous geometry investigation, the OHRC points
# mapped into this LROC tile.
#
# These are LROC full-resolution coordinates.
#
# We do NOT assume that they provide exact pixel registration.
#
# ------------------------------------------------------------

LROC_X_MIN = 27314.310
LROC_X_MAX = 45456.780

LROC_Y_MIN = 93.586
LROC_Y_MAX = 16201.852


# ------------------------------------------------------------
# GEOMETRY CSV PARSER
# ------------------------------------------------------------

def find_column(fieldnames, candidates):

    lower_map = {
        name.lower().strip(): name
        for name in fieldnames
    }

    for candidate in candidates:

        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    return None


def load_geometry():

    print("[1/8] Reading OHRC geometry CSV...")

    if not GEOMETRY_CSV.exists():
        raise FileNotFoundError(
            f"Geometry CSV not found:\n{GEOMETRY_CSV}"
        )

    rows = []

    with open(
        GEOMETRY_CSV,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        fieldnames = reader.fieldnames

        if fieldnames is None:
            raise RuntimeError(
                "Geometry CSV has no header."
            )

        print(
            "      Columns:",
            ", ".join(fieldnames)
        )

        lon_col = find_column(
            fieldnames,
            [
                "Longitude",
                "Lon",
                "longitude",
            ],
        )

        lat_col = find_column(
            fieldnames,
            [
                "Latitude",
                "Lat",
                "latitude",
            ],
        )

        pixel_col = find_column(
            fieldnames,
            [
                "OHRC_Pixel",
                "Pixel",
            ],
        )

        scan_col = find_column(
            fieldnames,
            [
                "OHRC_Scan",
                "Scan",
            ],
        )

        if not all(
            [lon_col, lat_col, pixel_col, scan_col]
        ):
            raise RuntimeError(
                "Could not identify required geometry columns."
            )

        for row in reader:

            try:

                lon = float(row[lon_col])
                lat = float(row[lat_col])
                pixel = float(row[pixel_col])
                scan = float(row[scan_col])

            except (
                ValueError,
                TypeError,
            ):
                continue

            rows.append(
                (
                    lon,
                    lat,
                    pixel,
                    scan,
                )
            )

    print(
        f"      Geometry points: {len(rows):,}"
    )

    return np.asarray(
        rows,
        dtype=np.float64,
    )


# ------------------------------------------------------------
# SOUTH POLAR PROJECTION
# ------------------------------------------------------------
#
# This is used ONLY to construct a common approximate metric
# coordinate system from the latitude/longitude geometry.
#
# The purpose is to determine the OHRC geographic footprint
# relative to the previously established LROC crop.
#
# It is NOT used as pixel-level truth.
#
# ------------------------------------------------------------

MOON_RADIUS = 1737400.0


def south_polar_stereo(lon_deg, lat_deg):

    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)

    # Convert longitude to [-180, 180]
    lon = (lon + np.pi) % (
        2.0 * np.pi
    ) - np.pi

    # South-polar stereographic.
    #
    # The exact orientation convention is retained from the
    # earlier project geometry work.
    #
    k = (
        2.0 * MOON_RADIUS
        / (
            1.0
            - np.sin(lat)
        )
    )

    x = (
        k
        * np.cos(lat)
        * np.sin(lon)
    )

    y = (
        k
        * np.cos(lat)
        * np.cos(lon)
    )

    return x, y


# ------------------------------------------------------------
# LOAD PREVIOUS PROJECT GEOGRAPHIC MAPPING
# ------------------------------------------------------------

def load_previous_projection():

    path = (
        PROJECT_ROOT
        / "experiments"
        / "projected_points_v2.csv"
    )

    if not path.exists():

        print(
            "[WARNING] Previous projected_points_v2.csv "
            "not found."
        )

        return None

    print(
        "[2/8] Loading previous geographic projection..."
    )

    data = []

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            try:

                bx = float(
                    row["OHRC_Browse_X"]
                )

                by = float(
                    row["OHRC_Browse_Y"]
                )

                mx = float(
                    row["LROC_Mosaic_X"]
                )

                my = float(
                    row["LROC_Mosaic_Y"]
                )

                tile = row.get(
                    "LROC_Tile",
                    "",
                )

            except (
                ValueError,
                TypeError,
                KeyError,
            ):
                continue

            data.append(
                (
                    bx,
                    by,
                    mx,
                    my,
                    tile,
                )
            )

    print(
        f"      Previous mapping points: "
        f"{len(data):,}"
    )

    return data


# ------------------------------------------------------------
# SELECT P892S2250 POINTS
# ------------------------------------------------------------

def select_tile_points(previous):

    print(
        "[3/8] Selecting points mapped to P892S2250..."
    )

    selected = []

    for (
        bx,
        by,
        mx,
        my,
        tile,
    ) in previous:

        if (
            tile == "P892S2250"
            and LROC_X_MIN <= mx <= LROC_X_MAX
            and LROC_Y_MIN <= my <= LROC_Y_MAX
        ):

            selected.append(
                (
                    bx,
                    by,
                    mx,
                    my,
                )
            )

    print(
        f"      Selected points: {len(selected):,}"
    )

    return np.asarray(
        selected,
        dtype=np.float64,
    )


# ------------------------------------------------------------
# BROWSE → NATIVE OHRC COORDINATES
# ------------------------------------------------------------
#
# The actual browse image is:
#
#   1200 x 7980
#
# Native:
#
#   12000 x 79796
#
# Approximate scale:
#
#   x ≈ browse_x * 10
#   y ≈ browse_y * 10
#
# We preserve the row relationship based on the actual
# dimensions rather than assuming an exact 10x factor.
#
# ------------------------------------------------------------

def browse_to_native(
    browse_x,
    browse_y,
):

    native_x = (
        browse_x
        * (WIDTH / 1200.0)
    )

    native_y = (
        browse_y
        * (HEIGHT / 7980.0)
    )

    return native_x, native_y


# ------------------------------------------------------------
# CROP BOUNDS
# ------------------------------------------------------------

def calculate_crop(selected):

    print(
        "[4/8] Calculating native OHRC crop..."
    )

    bx = selected[:, 0]
    by = selected[:, 1]

    native_x, native_y = browse_to_native(
        bx,
        by,
    )

    x_min = float(np.min(native_x))
    x_max = float(np.max(native_x))

    y_min = float(np.min(native_y))
    y_max = float(np.max(native_y))

    print(
        f"      Native footprint X: "
        f"{x_min:.2f} → {x_max:.2f}"
    )

    print(
        f"      Native footprint Y: "
        f"{y_min:.2f} → {y_max:.2f}"
    )

    # --------------------------------------------------------
    # Margin
    # --------------------------------------------------------
    #
    # Add ~300 native pixels ≈ 84 m.
    #
    # This is deliberately modest.
    #
    margin = 300

    x0 = max(
        0,
        int(math.floor(x_min)) - margin,
    )

    x1 = min(
        WIDTH,
        int(math.ceil(x_max)) + margin,
    )

    y0 = max(
        0,
        int(math.floor(y_min)) - margin,
    )

    y1 = min(
        HEIGHT,
        int(math.ceil(y_max)) + margin,
    )

    print()
    print(
        "      Crop:"
    )

    print(
        f"      X = {x0}:{x1}"
    )

    print(
        f"      Y = {y0}:{y1}"
    )

    print(
        f"      Size = "
        f"{x1-x0} x {y1-y0}"
    )

    return (
        x0,
        x1,
        y0,
        y1,
    )


# ------------------------------------------------------------
# READ NATIVE CROP
# ------------------------------------------------------------

def read_native_crop(
    x0,
    x1,
    y0,
    y1,
):

    print(
        "[5/8] Reading native OHRC crop..."
    )

    image = np.memmap(
        OHRC_IMG,
        dtype=np.uint8,
        mode="r",
        shape=(HEIGHT, WIDTH),
        order="C",
    )

    crop = np.asarray(
        image[
            y0:y1,
            x0:x1,
        ]
    )

    print(
        f"      Crop shape: {crop.shape}"
    )

    print(
        f"      min={int(crop.min())}, "
        f"max={int(crop.max())}, "
        f"mean={float(crop.mean()):.4f}, "
        f"median={float(np.median(crop)):.4f}"
    )

    return crop


# ------------------------------------------------------------
# RESAMPLE TO ~1 m/pixel
# ------------------------------------------------------------

def resample_to_1m(
    crop,
):

    print(
        "[6/8] Resampling OHRC crop toward 1 m/pixel..."
    )

    scale = OHRC_GSD / TARGET_GSD

    new_width = max(
        1,
        int(
            round(
                crop.shape[1]
                * scale
            )
        ),
    )

    new_height = max(
        1,
        int(
            round(
                crop.shape[0]
                * scale
            )
        ),
    )

    print(
        f"      Physical scale factor: "
        f"{scale:.6f}"
    )

    print(
        f"      Output shape: "
        f"{new_height} x {new_width}"
    )

    working = cv2.resize(
        crop,
        (
            new_width,
            new_height,
        ),
        interpolation=cv2.INTER_AREA,
    )

    return working


# ------------------------------------------------------------
# REPRESENTATIONS
# ------------------------------------------------------------

def make_gradient(image):

    img = image.astype(
        np.float32
    )

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

    mag = cv2.magnitude(
        gx,
        gy,
    )

    lo = np.percentile(
        mag,
        1,
    )

    hi = np.percentile(
        mag,
        99,
    )

    if hi <= lo:
        hi = lo + 1.0

    out = (
        mag - lo
    ) * 255.0 / (
        hi - lo
    )

    return np.clip(
        out,
        0,
        255,
    ).astype(np.uint8)


def make_multiscale(image):

    img = image.astype(
        np.float32
    )

    responses = []

    for sigma in (
        1.0,
        2.0,
        4.0,
    ):

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
            cv2.magnitude(
                gx,
                gy,
            )
        )

    combined = (
        0.20 * responses[0]
        + 0.30 * responses[1]
        + 0.50 * responses[2]
    )

    lo = np.percentile(
        combined,
        1,
    )

    hi = np.percentile(
        combined,
        99,
    )

    if hi <= lo:
        hi = lo + 1.0

    out = (
        combined - lo
    ) * 255.0 / (
        hi - lo
    )

    return np.clip(
        out,
        0,
        255,
    ).astype(np.uint8)


def make_clahe(image):

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    return clahe.apply(image)


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

def save_image(
    path,
    image,
):

    ok = cv2.imwrite(
        str(path),
        image,
    )

    if not ok:
        raise RuntimeError(
            f"Failed to save:\n{path}"
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
    print(
        "       TRUE ~1 m OHRC GEOGRAPHIC CROP"
    )
    print("=" * 72)

    print()

    if not OHRC_IMG.exists():
        raise FileNotFoundError(
            f"OHRC IMG not found:\n{OHRC_IMG}"
        )

    if not GEOMETRY_CSV.exists():
        raise FileNotFoundError(
            f"Geometry CSV not found:\n{GEOMETRY_CSV}"
        )

    # --------------------------------------------------------
    # Geometry
    # --------------------------------------------------------

    geometry = load_geometry()

    previous = load_previous_projection()

    if previous is None:
        raise RuntimeError(
            "The previous projected_points_v2.csv "
            "is required for this first crop."
        )

    selected = select_tile_points(
        previous
    )

    if len(selected) < 10:
        raise RuntimeError(
            "Too few P892S2250 points were found."
        )

    # --------------------------------------------------------
    # Crop
    # --------------------------------------------------------

    x0, x1, y0, y1 = calculate_crop(
        selected
    )

    native_crop = read_native_crop(
        x0,
        x1,
        y0,
        y1,
    )

    # --------------------------------------------------------
    # Native crop preview
    # --------------------------------------------------------

    save_image(
        OUTPUT_DIR
        / "01_native_crop.png",
        native_crop,
    )

    # --------------------------------------------------------
    # 1m resampling
    # --------------------------------------------------------

    working = resample_to_1m(
        native_crop
    )

    save_image(
        OUTPUT_DIR
        / "02_ohrc_approx_1m.png",
        working,
    )

    # --------------------------------------------------------
    # Representations
    # --------------------------------------------------------

    print(
        "[7/8] Creating 1m representations..."
    )

    gradient = make_gradient(
        working
    )

    multiscale = make_multiscale(
        working
    )

    clahe = make_clahe(
        working
    )

    save_image(
        OUTPUT_DIR
        / "03_ohrc_1m_clahe.png",
        clahe,
    )

    save_image(
        OUTPUT_DIR
        / "04_ohrc_1m_gradient.png",
        gradient,
    )

    save_image(
        OUTPUT_DIR
        / "05_ohrc_1m_multiscale.png",
        multiscale,
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print(
        "[8/8] Writing report..."
    )

    report = (
        OUTPUT_DIR
        / "true_1m_ohrc_crop_report.txt"
    )

    with open(
        report,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "TRUE ~1 m OHRC GEOGRAPHIC CROP\n"
        )

        f.write(
            "=" * 72 + "\n\n"
        )

        f.write(
            f"Input:\n{OHRC_IMG}\n\n"
        )

        f.write(
            f"Geometry:\n{GEOMETRY_CSV}\n\n"
        )

        f.write(
            "LROC region:\n"
        )

        f.write(
            f"X: {LROC_X_MIN} -> "
            f"{LROC_X_MAX}\n"
        )

        f.write(
            f"Y: {LROC_Y_MIN} -> "
            f"{LROC_Y_MAX}\n\n"
        )

        f.write(
            "OHRC native crop:\n"
        )

        f.write(
            f"X: {x0}:{x1}\n"
        )

        f.write(
            f"Y: {y0}:{y1}\n"
        )

        f.write(
            f"Shape: "
            f"{native_crop.shape}\n\n"
        )

        f.write(
            "Physical sampling:\n"
        )

        f.write(
            f"OHRC native GSD: "
            f"{OHRC_GSD} m/pixel\n"
        )

        f.write(
            f"Target GSD: "
            f"{TARGET_GSD} m/pixel\n"
        )

        f.write(
            "Resampling: "
            "INTER_AREA\n\n"
        )

        f.write(
            f"Working shape: "
            f"{working.shape}\n\n"
        )

        f.write(
            "IMPORTANT:\n"
        )

        f.write(
            "The OHRC geometry is used only to establish "
            "geographic overlap with the LROC region. "
            "The resulting geographic mapping is NOT "
            "treated as pixel-level ground truth.\n"
        )

    print(
        f"[SAVED] {report}"
    )

    print()
    print("=" * 72)
    print("COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()