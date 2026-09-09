"""
PAIR 001 — EXACT OFFICIAL LROC / OHRC OVERLAP DIAGNOSTIC

Purpose
-------
1. Use the actual OHRC geographic footprint.
2. Identify which official LROC South Pole tiles can contain it.
3. Load the official 1421 x 1421 LROC browse PNGs.
4. Create a diagnostic contact mosaic.
5. Project the OHRC footprint into the lunar South Polar
   Stereographic coordinate system.
6. Save diagnostic files for the next exact pixel-overlap step.

IMPORTANT
---------
This script does NOT run SIFT.

We are validating the official LROC reference geometry first.

PAIR 001 OHRC footprint:
    55.564452E -> 233.745958E

Expected relevant LROC tiles:
    P892S0450  : 0-90E
    P892S1350  : 90-180E
    P892S2250  : 180-270E

Excluded:
    P892S3150  : 270-360E


LROC South Polar Stereographic projection:

    R = 1,737,400 m

    k = 2R / (1 - sin(phi))

    X = k * cos(phi) * sin(lambda)
    Y = k * cos(phi) * cos(lambda)

The cos(phi) term is essential.
"""


# ============================================================================
# IMPORTS
# ============================================================================

from pathlib import Path
import math
import csv

import cv2
import numpy as np


# ============================================================================
# PROJECT PATHS
# ============================================================================

BASE_DIR = Path(
    r"C:\Lunar\lunar-image-correspondence"
)

# Official LROC files downloaded earlier.
LROC_DIR = (
    BASE_DIR
    / "data"
    / "lroc_official"
    / "pair_001"
)

BROWSE_DIR = (
    LROC_DIR
    / "browse"
)

METADATA_DIR = (
    LROC_DIR
    / "metadata"
)

# IMPORTANT:
# Results are directly under the project root,
# NOT inside an experiments folder.
OUTPUT_DIR = (
    BASE_DIR
    / "official_lroc_exact_overlap"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================================
# CONSTANTS
# ============================================================================

# Lunar radius used by the official LROC polar stereographic CRS.
R = 1_737_400.0

# All four official browse PNGs were verified earlier as 1421 x 1421.
BROWSE_SIZE = 1421


# ============================================================================
# PAIR 001 — OHRC FOOTPRINT
# ============================================================================

OHRC_CORNERS = [
    {
        "name": "Upper Left",
        "lat": -89.923132,
        "lon": 55.564452,
    },
    {
        "name": "Upper Right",
        "lat": -89.850542,
        "lon": 110.416030,
    },
    {
        "name": "Lower Left",
        "lat": -89.257559,
        "lon": 233.745958,
    },
    {
        "name": "Lower Right",
        "lat": -89.252796,
        "lon": 224.351932,
    },
]


# ============================================================================
# OFFICIAL LROC TILE INFORMATION
# ============================================================================

TILES = {
    "P892S0450": {
        "west": 0.0,
        "east": 90.0,
        "south": -90.0,
        "north": -88.5,

        "ul_x": -0.5,
        "ul_y": 45487.5,

        "samples": 45488,
        "lines": 45488,
    },

    "P892S1350": {
        "west": 90.0,
        "east": 180.0,
        "south": -90.0,
        "north": -88.5,

        "ul_x": -0.5,
        "ul_y": -0.5,

        "samples": 45488,
        "lines": 45488,
    },

    "P892S2250": {
        "west": 180.0,
        "east": 270.0,
        "south": -90.0,
        "north": -88.5,

        "ul_x": 45487.5,
        "ul_y": -0.5,

        "samples": 45489,
        "lines": 45488,
    },

    "P892S3150": {
        "west": 270.0,
        "east": 360.0,
        "south": -90.0,
        "north": -88.5,

        "ul_x": 45487.5,
        "ul_y": 45487.5,

        "samples": 45488,
        "lines": 45489,
    },
}


# ============================================================================
# LUNAR SOUTH POLAR STEREOGRAPHIC PROJECTION
# ============================================================================

def project_lunar(lat_deg, lon_deg):
    """
    Project lunar latitude/longitude into the LROC
    South Polar Stereographic coordinate system.

    Correct equations:

        k = 2R / (1 - sin(phi))

        X = k * cos(phi) * sin(lambda)
        Y = k * cos(phi) * cos(lambda)

    Parameters
    ----------
    lat_deg : float
        Latitude in degrees.

    lon_deg : float
        East longitude in degrees.

    Returns
    -------
    x : float
        Projected X coordinate in metres.

    y : float
        Projected Y coordinate in metres.
    """

    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)

    denominator = (
        1.0
        - math.sin(lat)
    )

    # Exact South Pole.
    if abs(denominator) < 1e-15:
        return 0.0, 0.0

    k = (
        2.0
        * R
        / denominator
    )

    x = (
        k
        * math.cos(lat)
        * math.sin(lon)
    )

    y = (
        k
        * math.cos(lat)
        * math.cos(lon)
    )

    return x, y


# ============================================================================
# LOAD OFFICIAL BROWSE IMAGE
# ============================================================================

def load_browse(tile_name):
    """
    Load one official LROC browse PNG.
    """

    filename = (
        f"{tile_name}.BROWSE.PNG"
    )

    path = (
        BROWSE_DIR
        / filename
    )

    if not path.exists():

        raise FileNotFoundError(
            "\nOfficial LROC browse image not found:\n"
            f"{path}\n\n"
            "Expected file names are:\n"
            "  P892S0450.BROWSE.PNG\n"
            "  P892S1350.BROWSE.PNG\n"
            "  P892S2250.BROWSE.PNG\n"
            "  P892S3150.BROWSE.PNG"
        )

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:

        raise RuntimeError(
            f"OpenCV could not read:\n{path}"
        )

    return image


# ============================================================================
# VERIFY OFFICIAL BROWSE IMAGES
# ============================================================================

def verify_browse_files():

    print()
    print("=" * 90)
    print("VERIFYING OFFICIAL LROC BROWSE IMAGES")
    print("=" * 90)

    images = {}

    for tile_name in TILES:

        image = load_browse(
            tile_name
        )

        images[tile_name] = image

        height, width = image.shape

        print()
        print(tile_name)

        print(
            f"  Browse dimensions: "
            f"{width} x {height}"
        )

        print(
            f"  Data type: "
            f"{image.dtype}"
        )

        print(
            f"  Minimum value: "
            f"{int(image.min())}"
        )

        print(
            f"  Maximum value: "
            f"{int(image.max())}"
        )

        if (
            width == BROWSE_SIZE
            and
            height == BROWSE_SIZE
        ):

            print(
                "  STATUS: OK — 1421 x 1421"
            )

        else:

            print(
                "  WARNING — unexpected dimensions"
            )

    return images


# ============================================================================
# DETERMINE GEOGRAPHICALLY RELEVANT TILES
# ============================================================================

def determine_tiles():

    min_lon = min(
        p["lon"]
        for p in OHRC_CORNERS
    )

    max_lon = max(
        p["lon"]
        for p in OHRC_CORNERS
    )

    selected = []

    print()
    print("=" * 90)
    print("GEOGRAPHIC TILE SELECTION")
    print("=" * 90)

    print()
    print(
        f"OHRC longitude range: "
        f"{min_lon:.6f}°E -> "
        f"{max_lon:.6f}°E"
    )

    print()

    for tile_name, tile in TILES.items():

        # Simple longitude-overlap test.
        #
        # This is intentional here:
        # it identifies candidate tiles before the exact
        # footprint/pixel mapping stage.

        overlaps = (
            max_lon > tile["west"]
            and
            min_lon < tile["east"]
        )

        if overlaps:

            selected.append(
                tile_name
            )

            print(
                f"  {tile_name}: YES"
            )

        else:

            print(
                f"  {tile_name}: NO"
            )

    return selected


# ============================================================================
# PROJECT OHRC FOOTPRINT
# ============================================================================

def project_ohrc():

    projected = []

    for corner in OHRC_CORNERS:

        x, y = project_lunar(
            corner["lat"],
            corner["lon"],
        )

        projected.append(
            {
                "name": corner["name"],
                "lat": corner["lat"],
                "lon": corner["lon"],
                "x": x,
                "y": y,
            }
        )

    return projected


# ============================================================================
# PRINT PROJECTED OHRC FOOTPRINT
# ============================================================================

def print_projected_ohrc(points):

    print()
    print("=" * 90)
    print("OHRC PROJECTED FOOTPRINT")
    print("=" * 90)

    for point in points:

        print()

        print(
            f"  {point['name']}"
        )

        print(
            f"    Latitude:  "
            f"{point['lat']:.6f}°"
        )

        print(
            f"    Longitude: "
            f"{point['lon']:.6f}°E"
        )

        print(
            f"    X: "
            f"{point['x']:.3f} m"
        )

        print(
            f"    Y: "
            f"{point['y']:.3f} m"
        )

    xs = [
        p["x"]
        for p in points
    ]

    ys = [
        p["y"]
        for p in points
    ]

    print()
    print(
        "  Projected bounding box:"
    )

    print(
        f"    X: "
        f"{min(xs):.3f} "
        f"-> "
        f"{max(xs):.3f} m"
    )

    print(
        f"    Y: "
        f"{min(ys):.3f} "
        f"-> "
        f"{max(ys):.3f} m"
    )

    print()
    print(
        f"    Width: "
        f"{max(xs) - min(xs):.3f} m"
    )

    print(
        f"    Height: "
        f"{max(ys) - min(ys):.3f} m"
    )


# ============================================================================
# SAVE PROJECTED OHRC CSV
# ============================================================================

def save_projected_csv(points):

    output = (
        OUTPUT_DIR
        / "ohrc_projected_footprint.csv"
    )

    with open(
        output,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "name",
                "latitude_deg",
                "longitude_deg",
                "x_m",
                "y_m",
            ]
        )

        for point in points:

            writer.writerow(
                [
                    point["name"],
                    point["lat"],
                    point["lon"],
                    point["x"],
                    point["y"],
                ]
            )

    return output


# ============================================================================
# CREATE PHYSICAL PROJECTION PLOT
# ============================================================================

def create_projection_plot(points):

    import matplotlib.pyplot as plt

    xs = [
        p["x"]
        for p in points
    ]

    ys = [
        p["y"]
        for p in points
    ]

    # Close polygon.
    plot_x = xs + [xs[0]]
    plot_y = ys + [ys[0]]

    fig = plt.figure(
        figsize=(8, 8)
    )

    ax = fig.add_subplot(111)

    ax.plot(
        plot_x,
        plot_y,
        "-o",
    )

    for point in points:

        ax.annotate(
            point["name"],
            (
                point["x"],
                point["y"],
            ),
        )

    # South Pole.
    ax.plot(
        0,
        0,
        marker="x",
        markersize=12,
    )

    ax.annotate(
        "South Pole",
        (
            0,
            0,
        ),
    )

    ax.set_xlabel(
        "LROC polar stereographic X (m)"
    )

    ax.set_ylabel(
        "LROC polar stereographic Y (m)"
    )

    ax.set_title(
        "PAIR 001 — OHRC Projected Footprint"
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    output = (
        OUTPUT_DIR
        / "ohrc_projected_footprint.png"
    )

    fig.savefig(
        output,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output


# ============================================================================
# CREATE CONTACT MOSAIC
# ============================================================================

def create_contact_mosaic(images):

    """
    Create a 2x2 contact sheet showing all four official
    LROC South Pole browse images.

    Layout based on the official tile arrangement:

        P892S0450 | P892S3150
        ----------+----------
        P892S1350 | P892S2250

    NOTE:
    This is a CONTACT SHEET for visual inspection.
    It is not yet being used as a mathematically exact
    image-coordinate mosaic.
    """

    canvas = np.zeros(
        (
            BROWSE_SIZE * 2,
            BROWSE_SIZE * 2,
        ),
        dtype=np.uint8,
    )

    positions = {
        "P892S0450": (
            0,
            0,
        ),

        "P892S3150": (
            0,
            BROWSE_SIZE,
        ),

        "P892S1350": (
            BROWSE_SIZE,
            0,
        ),

        "P892S2250": (
            BROWSE_SIZE,
            BROWSE_SIZE,
        ),
    }

    for tile_name, (
        row,
        col,
    ) in positions.items():

        image = images[tile_name]

        canvas[
            row:row + BROWSE_SIZE,
            col:col + BROWSE_SIZE,
        ] = image

    return canvas


# ============================================================================
# LABEL CONTACT MOSAIC
# ============================================================================

def label_contact_mosaic(image):

    output = cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2BGR,
    )

    labels = {
        "P892S0450": (
            30,
            50,
        ),

        "P892S3150": (
            BROWSE_SIZE + 30,
            50,
        ),

        "P892S1350": (
            30,
            BROWSE_SIZE + 50,
        ),

        "P892S2250": (
            BROWSE_SIZE + 30,
            BROWSE_SIZE + 50,
        ),
    }

    for name, (
        x,
        y,
    ) in labels.items():

        cv2.putText(
            output,
            name,
            (
                x,
                y,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # Vertical boundary.
    cv2.line(
        output,
        (
            BROWSE_SIZE,
            0,
        ),
        (
            BROWSE_SIZE,
            BROWSE_SIZE * 2,
        ),
        (255, 255, 255),
        1,
    )

    # Horizontal boundary.
    cv2.line(
        output,
        (
            0,
            BROWSE_SIZE,
        ),
        (
            BROWSE_SIZE * 2,
            BROWSE_SIZE,
        ),
        (255, 255, 255),
        1,
    )

    return output


# ============================================================================
# SAVE REPORT
# ============================================================================

def save_report(
    selected_tiles,
    points,
    generated_files,
):

    output = (
        OUTPUT_DIR
        / "README.txt"
    )

    xs = [
        p["x"]
        for p in points
    ]

    ys = [
        p["y"]
        for p in points
    ]

    with open(
        output,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "PAIR 001 — OFFICIAL LROC EXACT OVERLAP DIAGNOSTIC\n"
        )

        f.write(
            "=" * 80
            + "\n\n"
        )

        f.write(
            "Projection:\n"
        )

        f.write(
            "Lunar South Polar Stereographic\n"
        )

        f.write(
            "Moon radius = 1737400 m\n\n"
        )

        f.write(
            "Equations:\n"
        )

        f.write(
            "k = 2R / (1 - sin(phi))\n"
        )

        f.write(
            "X = k * cos(phi) * sin(lambda)\n"
        )

        f.write(
            "Y = k * cos(phi) * cos(lambda)\n\n"
        )

        f.write(
            "OHRC projected bounding box:\n"
        )

        f.write(
            f"X: {min(xs):.3f} -> "
            f"{max(xs):.3f} m\n"
        )

        f.write(
            f"Y: {min(ys):.3f} -> "
            f"{max(ys):.3f} m\n\n"
        )

        f.write(
            "OHRC geographic footprint:\n"
        )

        for point in points:

            f.write(
                f"{point['name']}: "
                f"{point['lat']:.6f}, "
                f"{point['lon']:.6f}E -> "
                f"{point['x']:.3f}, "
                f"{point['y']:.3f} m\n"
            )

        f.write("\n")

        f.write(
            "Candidate official LROC tiles:\n"
        )

        for tile in selected_tiles:

            f.write(
                f"  {tile}\n"
            )

        f.write("\n")

        f.write(
            "P892S3150 excluded because "
            "the OHRC footprint ends at "
            "233.745958E.\n\n"
        )

        f.write(
            "Generated files:\n"
        )

        for file_path in generated_files:

            f.write(
                f"  {file_path}\n"
            )

    return output


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 90)
    print("PAIR 001 — EXACT OFFICIAL LROC / OHRC OVERLAP DIAGNOSTIC")
    print("=" * 90)

    # ------------------------------------------------------------------------
    # Check important directories.
    # ------------------------------------------------------------------------

    print()
    print("Checking directories...")

    print(
        f"  Project:  {BASE_DIR}"
    )

    print(
        f"  LROC:     {LROC_DIR}"
    )

    print(
        f"  Browse:   {BROWSE_DIR}"
    )

    print(
        f"  Metadata: {METADATA_DIR}"
    )

    print(
        f"  Output:   {OUTPUT_DIR}"
    )

    if not BROWSE_DIR.exists():

        raise FileNotFoundError(
            f"\nLROC browse directory does not exist:\n"
            f"{BROWSE_DIR}"
        )

    # ------------------------------------------------------------------------
    # Determine relevant tiles.
    # ------------------------------------------------------------------------

    selected_tiles = determine_tiles()

    # ------------------------------------------------------------------------
    # Load official browse images.
    # ------------------------------------------------------------------------

    images = verify_browse_files()

    # ------------------------------------------------------------------------
    # Create contact mosaic.
    # ------------------------------------------------------------------------

    contact = create_contact_mosaic(
        images
    )

    contact_labeled = label_contact_mosaic(
        contact
    )

    contact_path = (
        OUTPUT_DIR
        / "official_lroc_browse_contact_mosaic.png"
    )

    success = cv2.imwrite(
        str(contact_path),
        contact_labeled,
    )

    if not success:

        raise RuntimeError(
            f"Could not save:\n{contact_path}"
        )

    print()
    print(
        "Contact mosaic created:"
    )

    print(
        f"  {contact_path}"
    )

    # ------------------------------------------------------------------------
    # Project OHRC footprint.
    # ------------------------------------------------------------------------

    points = project_ohrc()

    print_projected_ohrc(
        points
    )

    # ------------------------------------------------------------------------
    # Save CSV.
    # ------------------------------------------------------------------------

    csv_path = save_projected_csv(
        points
    )

    # ------------------------------------------------------------------------
    # Create projection plot.
    # ------------------------------------------------------------------------

    projection_path = create_projection_plot(
        points
    )

    # ------------------------------------------------------------------------
    # Save report.
    # ------------------------------------------------------------------------

    generated_files = [
        contact_path,
        projection_path,
        csv_path,
    ]

    report_path = save_report(
        selected_tiles,
        points,
        generated_files,
    )

    # ------------------------------------------------------------------------
    # Final output.
    # ------------------------------------------------------------------------

    print()
    print("=" * 90)
    print("FINAL RESULT")
    print("=" * 90)

    print()

    print(
        "Candidate official LROC tiles:"
    )

    for tile in selected_tiles:

        print(
            f"  ✓ {tile}"
        )

    print()

    print(
        "Excluded:"
    )

    print(
        "  ✗ P892S3150"
    )

    print()

    print(
        "Generated files:"
    )

    print(
        f"  1. {contact_path}"
    )

    print(
        f"  2. {projection_path}"
    )

    print(
        f"  3. {csv_path}"
    )

    print(
        f"  4. {report_path}"
    )

    print()
    print("=" * 90)
    print("NEXT STEP")
    print("=" * 90)

    print()
    print(
        "Do NOT run SIFT yet."
    )

    print()
    print(
        "We will next map the actual OHRC footprint"
    )

    print(
        "into the official LROC browse-image pixels."
    )

    print()
    print(
        "Then we will visually verify that both images"
    )

    print(
        "show the same lunar terrain."
    )

    print()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()