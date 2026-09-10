from __future__ import annotations

"""
P0 RAW overlap extraction for Chandrayaan-2 OHRC / LROC pairs.

P0 definition:

    native calibrated OHRC .img
        ->
    geographic overlap crop
        ->
    raw uint8 grayscale

No:
    - CLAHE
    - contrast enhancement
    - normalization
    - gradient representation
    - denoising
    - destriping
    - resampling

The crop is permitted because it is a geographic overlap selection,
not an image representation transformation.

The OHRC geometry CSV is sampled. Therefore we do not require sampled
geometry points to land strictly inside the reference AABB.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from PIL import Image


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT = Path(r"C:\Lunar\lunar-image-correspondence")

PAIRS = {
    "pair_001": {
        "year": "20241115",
        "reference_vrt": (
            PROJECT
            / "data"
            / "real_pairs"
            / "pair_001"
            / "reference"
            / "quickmap-lroc.vrt"
        ),
    },
    "pair_002": {
        "year": "20260103",
        "reference_vrt": (
            PROJECT
            / "data"
            / "real_pairs"
            / "pair_002"
            / "reference"
            / "quickmap-lroc.vrt"
        ),
    },
}

MOON_RADIUS_M = 1_737_400.0

REFERENCE_MARGIN_METERS = 500.0

NATIVE_MARGIN_PIXELS = 500

MIN_GEOMETRY_SAMPLES = 10

MAX_CROP_FRACTION = 0.50


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ImageMetadata:
    width: int
    height: int
    datatype: str
    dimension_source: str
    pixel_resolution_m: float | None
    projection: str | None
    logical_identifier: str | None
    acquisition_start: str | None
    acquisition_stop: str | None
    sun_azimuth_deg: float | None
    sun_elevation_deg: float | None
    solar_incidence_deg: float | None


@dataclass
class ReferenceMetadata:
    width: int
    height: int
    geotransform: tuple[float, float, float, float, float, float]
    srs: str
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    source_raster: Path


@dataclass
class CropBounds:
    pixel_min: int
    pixel_max: int
    scan_min: int
    scan_max: int
    width: int
    height: int
    method: str
    geometry_samples_used: int
    reference_margin_m: float


# =============================================================================
# XML HELPERS
# =============================================================================

def strip_tag(tag: str) -> str:
    return tag.split("}", 1)[-1]


def normalize_name(name: str) -> str:
    return (
        name
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def find_text(root: ET.Element, name: str) -> str | None:
    target = normalize_name(name)

    for elem in root.iter():
        tag = normalize_name(strip_tag(elem.tag))

        if tag == target and elem.text:
            value = elem.text.strip()

            if value:
                return value

    return None


def parse_int(value: str | int | float | None) -> int | None:
    if value is None:
        return None

    try:
        return int(float(str(value).strip()))
    except (ValueError, TypeError):
        return None


def parse_float(value: str | int | float | None) -> float | None:
    if value is None:
        return None

    try:
        return float(str(value).strip())
    except (ValueError, TypeError):
        return None


def find_dimension(
    root: ET.Element,
    aliases: set[str],
) -> int | None:
    """
    Find a dimension from either element text or attributes.

    This is deliberately broad because PDS XML versions can encode the
    same physical dimension under different field names.
    """

    aliases = {
        normalize_name(a)
        for a in aliases
    }

    for elem in root.iter():

        tag = normalize_name(
            strip_tag(elem.tag)
        )

        if tag in aliases:

            if elem.text:
                value = parse_int(
                    elem.text
                )

                if value is not None and value > 0:
                    return value

            for attr_value in elem.attrib.values():

                value = parse_int(
                    attr_value
                )

                if value is not None and value > 0:
                    return value

        for attr_name, attr_value in elem.attrib.items():

            attr_name = normalize_name(
                attr_name
            )

            if attr_name in aliases:

                value = parse_int(
                    attr_value
                )

                if value is not None and value > 0:
                    return value

    return None


# =============================================================================
# NATIVE IMAGE METADATA
# =============================================================================

def read_image_metadata(
    xml_path: Path,
    img_path: Path,
) -> ImageMetadata:

    root = ET.parse(
        xml_path
    ).getroot()

    width_aliases = {
        "sample_count",
        "samplecount",
        "samples",
        "sample",
        "width",
        "image_width",
        "imagewidth",
        "number_of_samples",
        "numberofsamples",
        "column_count",
        "columns",
        "num_samples",
        "n_samples",
    }

    height_aliases = {
        "line_count",
        "linecount",
        "lines",
        "line",
        "height",
        "image_height",
        "imageheight",
        "number_of_lines",
        "numberoflines",
        "row_count",
        "rows",
        "num_lines",
        "n_lines",
        "scan_count",
    }

    width = find_dimension(
        root,
        width_aliases,
    )

    height = find_dimension(
        root,
        height_aliases,
    )

    dimension_source = "XML"

    # -------------------------------------------------------------------------
    # Fallback: use exact native .img byte size.
    #
    # These products are known unsigned-byte OHRC products. For the selected
    # products the native width is 12000 samples. The height is therefore
    # obtained from:
    #
    #       IMG bytes / 12000
    #
    # and validated exactly.
    # -------------------------------------------------------------------------

    if width is None:
        width = 12000
        dimension_source = "IMG_BYTE_SIZE_FALLBACK"

    actual_bytes = img_path.stat().st_size

    if height is None:

        if actual_bytes % width != 0:

            raise RuntimeError(
                "Could not determine image dimensions.\n"
                f"XML: {xml_path}\n"
                f"IMG: {img_path}\n"
                f"IMG bytes: {actual_bytes:,}\n"
                f"Resolved width: {width:,}\n"
                "IMG byte size is not divisible by width."
            )

        height = actual_bytes // width

        dimension_source = "IMG_BYTE_SIZE_FALLBACK"

    expected_bytes = (
        width * height
    )

    if expected_bytes != actual_bytes:

        raise RuntimeError(
            "Native image dimension validation failed.\n"
            f"IMG: {img_path}\n"
            f"Resolved dimensions: {width} x {height}\n"
            f"Expected bytes: {expected_bytes:,}\n"
            f"Actual bytes:   {actual_bytes:,}"
        )

    datatype = (
        find_text(root, "data_type")
        or find_text(root, "datatype")
        or find_text(root, "sample_type")
        or "Unknown"
    )

    logical_identifier = (
        find_text(
            root,
            "logical_identifier",
        )
        or find_text(
            root,
            "logical_id",
        )
        or find_text(
            root,
            "lid",
        )
    )

    acquisition_start = (
        find_text(
            root,
            "start_time",
        )
        or find_text(
            root,
            "start_date_time",
        )
        or find_text(
            root,
            "acquisition_start_time",
        )
    )

    acquisition_stop = (
        find_text(
            root,
            "stop_time",
        )
        or find_text(
            root,
            "stop_date_time",
        )
        or find_text(
            root,
            "acquisition_stop_time",
        )
    )

    pixel_resolution = (
        parse_float(
            find_text(
                root,
                "pixel_resolution",
            )
        )
        or parse_float(
            find_text(
                root,
                "pixel_resolution_m",
            )
        )
        or parse_float(
            find_text(
                root,
                "ground_resolution",
            )
        )
    )

    projection = (
        find_text(
            root,
            "map_projection",
        )
        or find_text(
            root,
            "projection",
        )
    )

    sun_azimuth = (
        parse_float(
            find_text(
                root,
                "sun_azimuth",
            )
        )
        or parse_float(
            find_text(
                root,
                "sun_azimuth_angle",
            )
        )
    )

    sun_elevation = (
        parse_float(
            find_text(
                root,
                "sun_elevation",
            )
        )
        or parse_float(
            find_text(
                root,
                "sun_elevation_angle",
            )
        )
    )

    solar_incidence = (
        parse_float(
            find_text(
                root,
                "solar_incidence",
            )
        )
        or parse_float(
            find_text(
                root,
                "solar_incidence_angle",
            )
        )
    )

    return ImageMetadata(
        width=width,
        height=height,
        datatype=datatype,
        dimension_source=dimension_source,
        pixel_resolution_m=pixel_resolution,
        projection=projection,
        logical_identifier=logical_identifier,
        acquisition_start=acquisition_start,
        acquisition_stop=acquisition_stop,
        sun_azimuth_deg=sun_azimuth,
        sun_elevation_deg=sun_elevation,
        solar_incidence_deg=solar_incidence,
    )


# =============================================================================
# VRT
# =============================================================================

def parse_geotransform(
    text: str,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    float,
]:

    values = [
        float(v)
        for v in re.split(
            r"[,\s]+",
            text.strip(),
        )
        if v
    ]

    if len(values) != 6:
        raise RuntimeError(
            f"Invalid GeoTransform: {text}"
        )

    return tuple(values)  # type: ignore[return-value]


def read_vrt(
    vrt_path: Path,
) -> ReferenceMetadata:

    root = ET.parse(
        vrt_path
    ).getroot()

    width = int(
        root.attrib[
            "rasterXSize"
        ]
    )

    height = int(
        root.attrib[
            "rasterYSize"
        ]
    )

    geotransform_text = None
    srs_text = None
    source_filename = None

    for elem in root.iter():

        tag = strip_tag(
            elem.tag
        ).lower()

        if (
            tag == "geotransform"
            and elem.text
        ):
            geotransform_text = (
                elem.text.strip()
            )

        elif (
            tag == "srs"
            and elem.text
        ):
            srs_text = (
                elem.text.strip()
            )

        elif (
            tag == "sourcefilename"
            and elem.text
            and source_filename is None
        ):
            source_filename = (
                elem.text.strip()
            )

    if geotransform_text is None:
        raise RuntimeError(
            f"GeoTransform not found:\n{vrt_path}"
        )

    if srs_text is None:
        raise RuntimeError(
            f"SRS not found:\n{vrt_path}"
        )

    geotransform = parse_geotransform(
        geotransform_text
    )

    x0, xres, xrot, y0, yrot, yres = (
        geotransform
    )

    if (
        abs(xrot) > 1e-12
        or abs(yrot) > 1e-12
    ):
        raise RuntimeError(
            "Rotated VRT is not supported."
        )

    x1 = (
        x0
        + width
        * xres
    )

    y1 = (
        y0
        + height
        * yres
    )

    xmin = min(
        x0,
        x1,
    )

    xmax = max(
        x0,
        x1,
    )

    ymin = min(
        y0,
        y1,
    )

    ymax = max(
        y0,
        y1,
    )

    source_raster = None

    if source_filename:

        candidate = Path(
            source_filename
        )

        if candidate.is_absolute():

            source_raster = candidate

        else:

            source_raster = (
                vrt_path.parent
                / candidate
            )

    if (
        source_raster is None
        or not source_raster.exists()
    ):

        fallback = (
            vrt_path.parent
            / "quickmap-lroc.png"
        )

        if fallback.exists():
            source_raster = fallback

    if (
        source_raster is None
        or not source_raster.exists()
    ):

        raise RuntimeError(
            "Could not resolve VRT source raster.\n"
            f"VRT: {vrt_path}"
        )

    return ReferenceMetadata(
        width=width,
        height=height,
        geotransform=geotransform,
        srs=srs_text,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        source_raster=source_raster,
    )


# =============================================================================
# FILE DISCOVERY
# =============================================================================

def find_exactly_one(
    paths: list[Path],
    description: str,
) -> Path:

    if len(paths) != 1:

        listing = (
            "\n".join(
                str(p)
                for p in paths
            )
            if paths
            else "(none)"
        )

        raise RuntimeError(
            f"Expected exactly one {description}.\n"
            f"Found {len(paths)}:\n"
            f"{listing}"
        )

    return paths[0]


def find_pair_files(
    pair_id: str,
    year: str,
) -> tuple[
    Path,
    Path,
    Path,
]:

    moving_root = (
        PROJECT
        / "data"
        / "real_pairs"
        / pair_id
        / "moving"
    )

    img_candidates = sorted(
        p
        for p in moving_root.rglob(
            "*.img"
        )
        if (
            "_d_img_d18" in p.name
            and year in p.name
        )
    )

    if not img_candidates:

        img_candidates = sorted(
            p
            for p in moving_root.rglob(
                "*.img"
            )
            if "_d_img_d18" in p.name
        )

    xml_candidates = sorted(
        p
        for p in moving_root.rglob(
            "*.xml"
        )
        if (
            "_d_img_d18.xml"
            in p.name
            and year in p.name
        )
    )

    if not xml_candidates:

        xml_candidates = sorted(
            p
            for p in moving_root.rglob(
                "*.xml"
            )
            if "_d_img_d18.xml"
            in p.name
        )

    geometry_candidates = sorted(
        p
        for p in moving_root.rglob(
            "*.csv"
        )
        if (
            "_g_grd_d18.csv"
            in p.name
            and year in p.name
        )
    )

    if not geometry_candidates:

        geometry_candidates = sorted(
            p
            for p in moving_root.rglob(
                "*.csv"
            )
            if "_g_grd_d18.csv"
            in p.name
        )

    return (
        find_exactly_one(
            img_candidates,
            f"{pair_id} native OHRC IMG",
        ),
        find_exactly_one(
            xml_candidates,
            f"{pair_id} native OHRC XML",
        ),
        find_exactly_one(
            geometry_candidates,
            f"{pair_id} geometry CSV",
        ),
    )


# =============================================================================
# LUNAR SOUTH POLAR STEREOGRAPHIC PROJECTION
# =============================================================================

def south_polar_stereo(
    longitude_deg: np.ndarray,
    latitude_deg: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:

    lon = np.deg2rad(
        longitude_deg.astype(
            np.float64
        )
    )

    lat = np.deg2rad(
        latitude_deg.astype(
            np.float64
        )
    )

    denominator = (
        1.0
        - np.sin(lat)
    )

    denominator = np.maximum(
        denominator,
        1e-15,
    )

    k = (
        2.0
        * MOON_RADIUS_M
        / denominator
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


def load_geometry_csv(
    path: Path,
) -> pd.DataFrame:

    frame = pd.read_csv(
        path
    )

    frame.columns = [
        str(c)
        .strip()
        .lower()
        for c in frame.columns
    ]

    aliases = {
        "longitude": [
            "longitude",
            "lon",
        ],
        "latitude": [
            "latitude",
            "lat",
        ],
        "pixel": [
            "pixel",
            "sample",
            "x",
        ],
        "scan": [
            "scan",
            "line",
            "y",
        ],
    }

    rename_map = {}

    for target, candidates in aliases.items():

        for candidate in candidates:

            if candidate in frame.columns:

                rename_map[
                    candidate
                ] = target

                break

    frame = frame.rename(
        columns=rename_map
    )

    required = {
        "longitude",
        "latitude",
        "pixel",
        "scan",
    }

    missing = (
        required
        - set(frame.columns)
    )

    if missing:

        raise RuntimeError(
            "Geometry CSV missing required columns:\n"
            f"{sorted(missing)}\n"
            f"Available:\n"
            f"{list(frame.columns)}"
        )

    frame = frame[
        [
            "longitude",
            "latitude",
            "pixel",
            "scan",
        ]
    ].copy()

    for column in (
        "longitude",
        "latitude",
        "pixel",
        "scan",
    ):

        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame = frame.dropna().copy()

    frame["pixel"] = (
        frame["pixel"]
        .astype(np.int64)
    )

    frame["scan"] = (
        frame["scan"]
        .astype(np.int64)
    )

    frame["x"], frame["y"] = (
        south_polar_stereo(
            frame["longitude"].to_numpy(),
            frame["latitude"].to_numpy(),
        )
    )

    return frame


# =============================================================================
# CROP SELECTION
# =============================================================================

def clamp_bounds(
    pixel_min: int,
    pixel_max: int,
    scan_min: int,
    scan_max: int,
    width: int,
    height: int,
) -> tuple[
    int,
    int,
    int,
    int,
]:

    pixel_min = max(
        0,
        min(
            pixel_min,
            width - 1,
        ),
    )

    pixel_max = max(
        0,
        min(
            pixel_max,
            width - 1,
        ),
    )

    scan_min = max(
        0,
        min(
            scan_min,
            height - 1,
        ),
    )

    scan_max = max(
        0,
        min(
            scan_max,
            height - 1,
        ),
    )

    if pixel_min > pixel_max:
        pixel_min, pixel_max = (
            pixel_max,
            pixel_min,
        )

    if scan_min > scan_max:
        scan_min, scan_max = (
            scan_max,
            scan_min,
        )

    return (
        pixel_min,
        pixel_max,
        scan_min,
        scan_max,
    )


def make_crop(
    selected: pd.DataFrame,
    image_meta: ImageMetadata,
    method: str,
    reference_margin_m: float,
) -> CropBounds:

    pixel_min = (
        int(
            selected["pixel"].min()
        )
        - NATIVE_MARGIN_PIXELS
    )

    pixel_max = (
        int(
            selected["pixel"].max()
        )
        + NATIVE_MARGIN_PIXELS
    )

    scan_min = (
        int(
            selected["scan"].min()
        )
        - NATIVE_MARGIN_PIXELS
    )

    scan_max = (
        int(
            selected["scan"].max()
        )
        + NATIVE_MARGIN_PIXELS
    )

    (
        pixel_min,
        pixel_max,
        scan_min,
        scan_max,
    ) = clamp_bounds(
        pixel_min,
        pixel_max,
        scan_min,
        scan_max,
        image_meta.width,
        image_meta.height,
    )

    return CropBounds(
        pixel_min=pixel_min,
        pixel_max=pixel_max,
        scan_min=scan_min,
        scan_max=scan_max,
        width=(
            pixel_max
            - pixel_min
            + 1
        ),
        height=(
            scan_max
            - scan_min
            + 1
        ),
        method=method,
        geometry_samples_used=int(
            len(selected)
        ),
        reference_margin_m=reference_margin_m,
    )


def reasonable_crop(
    crop: CropBounds,
    image_meta: ImageMetadata,
) -> bool:

    return (
        crop.width
        / image_meta.width
        <= MAX_CROP_FRACTION
        and
        crop.height
        / image_meta.height
        <= MAX_CROP_FRACTION
    )


def calculate_crop(
    geometry: pd.DataFrame,
    reference: ReferenceMetadata,
    image_meta: ImageMetadata,
) -> CropBounds:

    xmin = reference.xmin
    xmax = reference.xmax
    ymin = reference.ymin
    ymax = reference.ymax

    # -------------------------------------------------------------------------
    # 1. Exact AABB
    # -------------------------------------------------------------------------

    exact = (
        (geometry["x"] >= xmin)
        & (geometry["x"] <= xmax)
        & (geometry["y"] >= ymin)
        & (geometry["y"] <= ymax)
    )

    selected = geometry[
        exact
    ]

    if len(selected) >= MIN_GEOMETRY_SAMPLES:

        crop = make_crop(
            selected,
            image_meta,
            "EXACT_AABB_GEOMETRY",
            0.0,
        )

        if reasonable_crop(
            crop,
            image_meta,
        ):
            return crop

    # -------------------------------------------------------------------------
    # 2. Expanded AABB
    # -------------------------------------------------------------------------

    margin = (
        REFERENCE_MARGIN_METERS
    )

    expanded = (
        (geometry["x"] >= xmin - margin)
        & (geometry["x"] <= xmax + margin)
        & (geometry["y"] >= ymin - margin)
        & (geometry["y"] <= ymax + margin)
    )

    selected = geometry[
        expanded
    ]

    if len(selected) >= MIN_GEOMETRY_SAMPLES:

        crop = make_crop(
            selected,
            image_meta,
            "EXPANDED_AABB_GEOMETRY",
            margin,
        )

        if reasonable_crop(
            crop,
            image_meta,
        ):
            return crop

    # -------------------------------------------------------------------------
    # 3. Nearest samples to reference corners + center
    # -------------------------------------------------------------------------

    target_points = np.array(
        [
            [xmin, ymin],
            [xmin, ymax],
            [xmax, ymin],
            [xmax, ymax],
            [
                (xmin + xmax) / 2.0,
                (ymin + ymax) / 2.0,
            ],
        ],
        dtype=np.float64,
    )

    gx = geometry["x"].to_numpy(
        dtype=np.float64
    )

    gy = geometry["y"].to_numpy(
        dtype=np.float64
    )

    indices = []

    for tx, ty in target_points:

        distance_sq = (
            (gx - tx) ** 2
            +
            (gy - ty) ** 2
        )

        indices.append(
            int(
                np.argmin(
                    distance_sq
                )
            )
        )

    indices = sorted(
        set(indices)
    )

    selected = geometry.iloc[
        indices
    ].copy()

    # How far did the nearest selected geometry point get from the targets?
    sx = selected["x"].to_numpy(
        dtype=np.float64
    )

    sy = selected["y"].to_numpy(
        dtype=np.float64
    )

    distance_matrix = np.sqrt(
        (
            sx[:, None]
            - target_points[:, 0][None, :]
        ) ** 2
        +
        (
            sy[:, None]
            - target_points[:, 1][None, :]
        ) ** 2
    )

    nearest_distance_m = float(
        distance_matrix.min()
    )

    crop = make_crop(
        selected,
        image_meta,
        "NEAREST_REFERENCE_POINTS",
        nearest_distance_m,
    )

    if not reasonable_crop(
        crop,
        image_meta,
    ):

        raise RuntimeError(
            "Nearest-geometry fallback produced an implausibly large crop.\n"
            f"Crop: {crop.width} x {crop.height}\n"
            f"Native: {image_meta.width} x {image_meta.height}\n"
            "Projection or geometry mapping should be investigated."
        )

    return crop


# =============================================================================
# NATIVE CROP EXTRACTION
# =============================================================================

def read_native_crop(
    img_path: Path,
    image_meta: ImageMetadata,
    crop: CropBounds,
) -> np.ndarray:

    dtype = (
        image_meta.datatype
        .lower()
        .replace(
            "_",
            "",
        )
        .replace(
            " ",
            "",
        )
    )

    if dtype not in {
        "unsignedbyte",
        "byte",
        "uint8",
    }:

        raise RuntimeError(
            "P0 currently expects uint8 OHRC data.\n"
            f"XML datatype: {image_meta.datatype}"
        )

    native = np.memmap(
        img_path,
        dtype=np.uint8,
        mode="r",
        shape=(
            image_meta.height,
            image_meta.width,
        ),
    )

    result = np.array(
        native[
            crop.scan_min :
            crop.scan_max + 1,
            crop.pixel_min :
            crop.pixel_max + 1,
        ],
        copy=True,
    )

    del native

    return result


# =============================================================================
# REFERENCE
# =============================================================================

def load_reference_grayscale(
    reference: ReferenceMetadata,
) -> np.ndarray:

    image = Image.open(
        reference.source_raster
    )

    if image.mode != "L":

        image = image.convert(
            "L"
        )

    array = np.asarray(
        image,
        dtype=np.uint8,
    )

    expected = (
        reference.height,
        reference.width,
    )

    if array.shape != expected:

        raise RuntimeError(
            "Reference raster shape does not match VRT.\n"
            f"VRT: {reference.width} x {reference.height}\n"
            f"Raster: {array.shape[1]} x {array.shape[0]}"
        )

    return array


# =============================================================================
# OUTPUT
# =============================================================================

def save_uint8_png(
    array: np.ndarray,
    path: Path,
) -> None:

    if array.dtype != np.uint8:

        raise RuntimeError(
            f"Expected uint8 image; got {array.dtype}"
        )

    Image.fromarray(
        array,
        mode="L",
    ).save(path)


def save_metadata(
    path: Path,
    pair_id: str,
    img_path: Path,
    xml_path: Path,
    geometry_path: Path,
    reference_vrt: Path,
    image_meta: ImageMetadata,
    reference: ReferenceMetadata,
    crop: CropBounds,
    geometry: pd.DataFrame,
) -> None:

    metadata = {

        "pair_id":
            pair_id,

        "preprocessing":
            "P0_RAW",

        "moving": {

            "img":
                str(img_path),

            "xml":
                str(xml_path),

            "width":
                image_meta.width,

            "height":
                image_meta.height,

            "datatype":
                image_meta.datatype,

            "dimension_source":
                image_meta.dimension_source,

            "pixel_resolution_m":
                image_meta.pixel_resolution_m,

            "projection":
                image_meta.projection,

            "logical_identifier":
                image_meta.logical_identifier,

            "acquisition_start":
                image_meta.acquisition_start,

            "acquisition_stop":
                image_meta.acquisition_stop,

            "sun_azimuth_deg":
                image_meta.sun_azimuth_deg,

            "sun_elevation_deg":
                image_meta.sun_elevation_deg,

            "solar_incidence_deg":
                image_meta.solar_incidence_deg,
        },

        "representation": {

            "grayscale":
                "uint8",

            "resampling":
                "NONE",

            "normalization":
                "NONE",

            "contrast_enhancement":
                "NONE",

            "clahe":
                "NONE",

            "gradient":
                "NONE",

            "denoising":
                "NONE",

            "destriping":
                "NONE",
        },

        "geometry": {

            "csv":
                str(geometry_path),

            "records":
                int(
                    len(geometry)
                ),
        },

        "reference": {

            "vrt":
                str(reference_vrt),

            "source_raster":
                str(
                    reference.source_raster
                ),

            "width":
                reference.width,

            "height":
                reference.height,

            "geotransform":
                list(
                    reference.geotransform
                ),

            "srs":
                reference.srs,

            "xmin":
                reference.xmin,

            "xmax":
                reference.xmax,

            "ymin":
                reference.ymin,

            "ymax":
                reference.ymax,
        },

        "crop":
            asdict(crop),

        "method_note":
            "The OHRC geometry CSV is sampled. "
            "Exact AABB, expanded AABB, and nearest-reference-point "
            "selection are attempted in that order.",
    }

    path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )


# =============================================================================
# PROCESS PAIR
# =============================================================================

def process_pair(
    pair_id: str,
) -> None:

    config = PAIRS[
        pair_id
    ]

    print()
    print(
        "=" * 80
    )
    print(
        pair_id
    )
    print(
        "=" * 80
    )

    (
        img_path,
        xml_path,
        geometry_path,
    ) = find_pair_files(
        pair_id,
        config["year"],
    )

    reference_vrt = Path(
        config["reference_vrt"]
    )

    image_meta = read_image_metadata(
        xml_path,
        img_path,
    )

    reference = read_vrt(
        reference_vrt
    )

    geometry = load_geometry_csv(
        geometry_path
    )

    print(
        f"IMG      : {img_path}"
    )

    print(
        f"XML      : {xml_path}"
    )

    print(
        f"Geometry : {geometry_path}"
    )

    print(
        f"Reference: {reference_vrt}"
    )

    print(
        f"Dimensions: "
        f"{image_meta.width} x "
        f"{image_meta.height}"
    )

    print(
        f"Dimension source: "
        f"{image_meta.dimension_source}"
    )

    print(
        f"Datatype: "
        f"{image_meta.datatype}"
    )

    print(
        f"Reference raster: "
        f"{reference.width} x "
        f"{reference.height}"
    )

    print(
        f"Reference X: "
        f"{reference.xmin:.3f} .. "
        f"{reference.xmax:.3f}"
    )

    print(
        f"Reference Y: "
        f"{reference.ymin:.3f} .. "
        f"{reference.ymax:.3f}"
    )

    print(
        f"Geometry records: "
        f"{len(geometry):,}"
    )

    exact = (
        (geometry["x"] >= reference.xmin)
        &
        (geometry["x"] <= reference.xmax)
        &
        (geometry["y"] >= reference.ymin)
        &
        (geometry["y"] <= reference.ymax)
    )

    margin = (
        REFERENCE_MARGIN_METERS
    )

    expanded = (
        (geometry["x"] >= reference.xmin - margin)
        &
        (geometry["x"] <= reference.xmax + margin)
        &
        (geometry["y"] >= reference.ymin - margin)
        &
        (geometry["y"] <= reference.ymax + margin)
    )

    print(
        "Geometry samples inside exact AABB    : "
        f"{int(exact.sum()):,}"
    )

    print(
        "Geometry samples inside expanded AABB : "
        f"{int(expanded.sum()):,}"
        f" (margin={margin:.1f} m)"
    )

    crop = calculate_crop(
        geometry,
        reference,
        image_meta,
    )

    print(
        f"Crop selection method: "
        f"{crop.method}"
    )

    print(
        "Native crop pixel : "
        f"{crop.pixel_min} .. "
        f"{crop.pixel_max}"
    )

    print(
        "Native crop scan  : "
        f"{crop.scan_min} .. "
        f"{crop.scan_max}"
    )

    print(
        f"Native crop size  : "
        f"{crop.width} x "
        f"{crop.height}"
    )

    print(
        f"Geometry samples used: "
        f"{crop.geometry_samples_used}"
    )

    moving = read_native_crop(
        img_path,
        image_meta,
        crop,
    )

    reference_gray = (
        load_reference_grayscale(
            reference
        )
    )

    output_dir = (
        PROJECT
        / "experiments"
        / "baseline"
        / pair_id
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    moving_path = (
        output_dir
        / "moving_raw_overlap.png"
    )

    reference_path = (
        output_dir
        / "reference_raw.png"
    )

    metadata_path = (
        output_dir
        / "crop_metadata.json"
    )

    save_uint8_png(
        moving,
        moving_path,
    )

    save_uint8_png(
        reference_gray,
        reference_path,
    )

    save_metadata(
        metadata_path,
        pair_id,
        img_path,
        xml_path,
        geometry_path,
        reference_vrt,
        image_meta,
        reference,
        crop,
        geometry,
    )

    print(
        f"Saved moving    : {moving_path}"
    )

    print(
        f"Saved reference : {reference_path}"
    )

    print(
        f"Saved metadata  : {metadata_path}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print(
        "=" * 80
    )

    print(
        "P0 RAW OVERLAP EXTRACTION"
    )

    print(
        "=" * 80
    )

    for pair_id in PAIRS:

        process_pair(
            pair_id
        )

    print()
    print(
        "=" * 80
    )

    print(
        "ALL P0 PAIRS COMPLETE"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()