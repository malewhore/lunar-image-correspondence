from pathlib import Path
import csv
import math

import numpy as np


# ============================================================
# OHRC GEOMETRY DIAGNOSTIC
# ============================================================
#
# Purpose:
#   Inspect the official OHRC geometry CSV directly.
#
#   We will:
#       1. Load all 96,679 geometry points.
#       2. Convert longitude convention to [-180, 180].
#       3. Compute approximate South Polar stereographic
#          coordinates.
#       4. Report geographic footprint.
#       5. Report Pixel/Scan ranges.
#       6. Divide the footprint into longitude sectors so we
#          can understand where the strip actually lies.
#
# IMPORTANT:
#   This is a GEOMETRY DIAGNOSTIC.
#
#   We are NOT claiming that the projected coordinates are
#   pixel-level registration ground truth.
#
# ============================================================


PROJECT_ROOT = Path(
    r"C:\Lunar\lunar-image-correspondence"
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
    / "ohrc_geometry_diagnostic"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------
# MOON
# ------------------------------------------------------------

MOON_RADIUS = 1737400.0


# ------------------------------------------------------------
# LOAD CSV
# ------------------------------------------------------------

def load_geometry():

    print("[1/6] Loading geometry CSV...")

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

        print(
            "      Columns:",
            reader.fieldnames
        )

        for row in reader:

            try:
                lon = float(row["Longitude"])
                lat = float(row["Latitude"])
                pixel = float(row["Pixel"])
                scan = float(row["Scan"])

            except (
                ValueError,
                TypeError,
                KeyError,
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

    data = np.asarray(
        rows,
        dtype=np.float64,
    )

    print(
        f"      Valid points: {len(data):,}"
    )

    return data


# ------------------------------------------------------------
# LONGITUDE CONVERSION
# ------------------------------------------------------------

def convert_longitude(lon):

    """
    Convert 0..360 longitude to -180..180.
    """

    return (
        (lon + 180.0)
        % 360.0
    ) - 180.0


# ------------------------------------------------------------
# SOUTH POLAR STEREOGRAPHIC
# ------------------------------------------------------------

def south_polar_stereo(
    lon_deg,
    lat_deg,
):

    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)

    k = (
        2.0
        * MOON_RADIUS
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
# REPORT RANGE
# ------------------------------------------------------------

def report_range(name, values):

    print(
        f"      {name}: "
        f"{np.min(values):.6f} "
        f"→ "
        f"{np.max(values):.6f}"
    )


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    print("=" * 72)
    print(
        "       OFFICIAL OHRC GEOMETRY DIAGNOSTIC"
    )
    print("=" * 72)

    print()

    data = load_geometry()

    lon = data[:, 0]
    lat = data[:, 1]
    pixel = data[:, 2]
    scan = data[:, 3]

    # --------------------------------------------------------
    # BASIC GEOMETRY
    # --------------------------------------------------------

    print()
    print(
        "[2/6] Basic geographic / image ranges..."
    )

    report_range(
        "Original longitude",
        lon,
    )

    report_range(
        "Latitude",
        lat,
    )

    report_range(
        "Pixel",
        pixel,
    )

    report_range(
        "Scan",
        scan,
    )

    lon180 = convert_longitude(
        lon
    )

    print()
    report_range(
        "Longitude (-180..180)",
        lon180,
    )

    # --------------------------------------------------------
    # CORNERS / EXTREMES
    # --------------------------------------------------------

    print()
    print(
        "[3/6] Finding geographic extremes..."
    )

    indices = {
        "min_lat": np.argmin(lat),
        "max_lat": np.argmax(lat),
        "min_lon_original": np.argmin(lon),
        "max_lon_original": np.argmax(lon),
        "min_lon_180": np.argmin(lon180),
        "max_lon_180": np.argmax(lon180),
    }

    for name, idx in indices.items():

        print(
            f"      {name}: "
            f"lon={lon[idx]:.6f}, "
            f"lon180={lon180[idx]:.6f}, "
            f"lat={lat[idx]:.6f}, "
            f"pixel={pixel[idx]:.1f}, "
            f"scan={scan[idx]:.1f}"
        )

    # --------------------------------------------------------
    # POLAR PROJECTION
    # --------------------------------------------------------

    print()
    print(
        "[4/6] Computing approximate South Polar "
        "stereographic coordinates..."
    )

    x, y = south_polar_stereo(
        lon180,
        lat,
    )

    report_range(
        "Projected X (m)",
        x,
    )

    report_range(
        "Projected Y (m)",
        y,
    )

    # radial distance from pole
    radial = np.sqrt(
        x * x + y * y
    )

    report_range(
        "Radial distance from pole (m)",
        radial,
    )

    # --------------------------------------------------------
    # LONGITUDE SECTORS
    # --------------------------------------------------------

    print()
    print(
        "[5/6] Counting geometry points by longitude sector..."
    )

    sectors = [
        (-180, -135),
        (-135, -90),
        (-90, -45),
        (-45, 0),
        (0, 45),
        (45, 90),
        (90, 135),
        (135, 180),
    ]

    sector_counts = []

    for lo, hi in sectors:

        mask = (
            (lon180 >= lo)
            & (lon180 < hi)
        )

        count = int(
            np.sum(mask)
        )

        sector_counts.append(
            (
                lo,
                hi,
                count,
            )
        )

        print(
            f"      {lo:>4} to {hi:>4}: "
            f"{count:,}"
        )

    # --------------------------------------------------------
    # IMAGE GRID STRUCTURE
    # --------------------------------------------------------

    print()
    print(
        "[6/6] Inspecting image-grid structure..."
    )

    unique_pixel = np.unique(
        pixel
    )

    unique_scan = np.unique(
        scan
    )

    print(
        f"      Unique Pixel values: "
        f"{len(unique_pixel):,}"
    )

    print(
        f"      Unique Scan values: "
        f"{len(unique_scan):,}"
    )

    if len(unique_pixel) > 1:

        pixel_steps = np.diff(
            unique_pixel
        )

        print(
            "      Pixel step values:",
            np.unique(
                pixel_steps
            )
        )

    if len(unique_scan) > 1:

        scan_steps = np.diff(
            unique_scan
        )

        print(
            "      Scan step values:",
            np.unique(
                scan_steps
            )
        )

    # --------------------------------------------------------
    # SAVE PROJECTED POINTS
    # --------------------------------------------------------

    projected_path = (
        OUTPUT_DIR
        / "ohrc_projected_geometry.csv"
    )

    print()
    print(
        f"[SAVE] {projected_path}"
    )

    with open(
        projected_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "LongitudeOriginal",
                "Longitude180",
                "Latitude",
                "Pixel",
                "Scan",
                "ProjectedX_m",
                "ProjectedY_m",
                "RadialDistance_m",
            ]
        )

        for i in range(
            len(data)
        ):

            writer.writerow(
                [
                    lon[i],
                    lon180[i],
                    lat[i],
                    pixel[i],
                    scan[i],
                    x[i],
                    y[i],
                    radial[i],
                ]
            )

    # --------------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------------

    report_path = (
        OUTPUT_DIR
        / "geometry_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "OFFICIAL OHRC GEOMETRY DIAGNOSTIC\n"
        )

        f.write(
            "=" * 72 + "\n\n"
        )

        f.write(
            f"Input:\n{GEOMETRY_CSV}\n\n"
        )

        f.write(
            f"Valid geometry points: "
            f"{len(data):,}\n\n"
        )

        f.write(
            "IMAGE GRID\n"
        )

        f.write(
            f"Pixel range: "
            f"{pixel.min()} -> {pixel.max()}\n"
        )

        f.write(
            f"Scan range: "
            f"{scan.min()} -> {scan.max()}\n"
        )

        f.write(
            f"Unique Pixel values: "
            f"{len(unique_pixel)}\n"
        )

        f.write(
            f"Unique Scan values: "
            f"{len(unique_scan)}\n\n"
        )

        f.write(
            "GEOGRAPHIC RANGE\n"
        )

        f.write(
            f"Original longitude: "
            f"{lon.min()} -> {lon.max()}\n"
        )

        f.write(
            f"Converted longitude: "
            f"{lon180.min()} -> {lon180.max()}\n"
        )

        f.write(
            f"Latitude: "
            f"{lat.min()} -> {lat.max()}\n\n"
        )

        f.write(
            "POLAR PROJECTION\n"
        )

        f.write(
            f"X: "
            f"{x.min()} -> {x.max()} m\n"
        )

        f.write(
            f"Y: "
            f"{y.min()} -> {y.max()} m\n"
        )

        f.write(
            f"Radial distance: "
            f"{radial.min()} -> "
            f"{radial.max()} m\n\n"
        )

        f.write(
            "LONGITUDE SECTORS\n"
        )

        for lo, hi, count in sector_counts:

            f.write(
                f"{lo} to {hi}: "
                f"{count:,}\n"
            )

        f.write(
            "\nIMPORTANT:\n"
        )

        f.write(
            "The polar projection is a diagnostic geographic "
            "coordinate system. It must not be interpreted as "
            "pixel-level correspondence ground truth.\n"
        )

    print(
        f"[SAVE] {report_path}"
    )

    print()
    print("=" * 72)
    print(
        "GEOMETRY DIAGNOSTIC COMPLETE"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()