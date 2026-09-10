from __future__ import annotations

"""
Production preprocessing module for Chandrayaan-2 OHRC / LROC
image correspondence.

Purpose
-------
Provide one reusable, configuration-driven preprocessing API for the
downstream CV / DL pipelines.

Selected operating points from the Pair-002 sensitivity experiment:

    Illumination sigma = 20.0
    CLAHE clipLimit    = 2.0
    CLAHE tileGridSize = (8, 8)

Available representations:

    P0_RAW
    P1_ROBUST_NORMALIZED
    P2_ILLUMINATION_CORRECTED
    P3_GRADIENT
    P4_COMBINED
    P5_CLAHE
    P6_ILLUMINATION_CLAHE

Primary recommended representations for downstream testing:

    P2_ILLUMINATION_CORRECTED
    P6_ILLUMINATION_CLAHE

Important
---------
This module performs NO image resampling.

The preprocessing operations are deterministic.

The moving and reference images receive the same representation transform
where that is meaningful, so downstream correspondence algorithms can compare
like-for-like representations.

This file can be imported by:
    - classical CV pipeline
    - SuperPoint / LightGlue pipeline
    - backend integration
    - future experiment runners
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal
import json

import cv2
import numpy as np


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT = Path(
    r"C:\Lunar\lunar-image-correspondence"
)

DEFAULT_PAIR_ID = "pair_002"

DEFAULT_INPUT_ROOT = (
    PROJECT
    / "experiments"
    / "baseline"
)

DEFAULT_OUTPUT_ROOT = (
    PROJECT
    / "experiments"
    / "preprocessing"
)


# =============================================================================
# FROZEN PREPROCESSING PARAMETERS
# =============================================================================

LOW_PERCENTILE = 1.0
HIGH_PERCENTILE = 99.0

ILLUMINATION_SIGMA = 20.0

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (
    8,
    8,
)

EPSILON = 1e-6


# =============================================================================
# REPRESENTATIONS
# =============================================================================

RepresentationName = Literal[
    "P0_RAW",
    "P1_ROBUST_NORMALIZED",
    "P2_ILLUMINATION_CORRECTED",
    "P3_GRADIENT",
    "P4_COMBINED",
    "P5_CLAHE",
    "P6_ILLUMINATION_CLAHE",
]


ALL_REPRESENTATIONS = [
    "P0_RAW",
    "P1_ROBUST_NORMALIZED",
    "P2_ILLUMINATION_CORRECTED",
    "P3_GRADIENT",
    "P4_COMBINED",
    "P5_CLAHE",
    "P6_ILLUMINATION_CLAHE",
]


RECOMMENDED_REPRESENTATIONS = [
    "P2_ILLUMINATION_CORRECTED",
    "P6_ILLUMINATION_CLAHE",
]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class PreprocessingConfig:
    """
    Frozen preprocessing configuration.

    Keeping parameters in one object makes experiments reproducible and
    prevents different downstream pipelines from silently using different
    settings.
    """

    low_percentile: float = LOW_PERCENTILE
    high_percentile: float = HIGH_PERCENTILE

    illumination_sigma: float = ILLUMINATION_SIGMA

    clahe_clip_limit: float = CLAHE_CLIP_LIMIT
    clahe_tile_grid_size: tuple[int, int] = (
        CLAHE_TILE_GRID_SIZE
    )

    resampling: str = "NONE"


@dataclass
class PreprocessedPair:
    """
    Result returned by preprocess_pair().
    """

    pair_id: str
    representation: str

    moving: np.ndarray
    reference: np.ndarray

    metadata: dict


# =============================================================================
# VALIDATION
# =============================================================================

def validate_gray_uint8(
    image: np.ndarray,
    name: str = "image",
) -> None:
    """
    Validate an input/output image.
    """

    if not isinstance(
        image,
        np.ndarray,
    ):
        raise TypeError(
            f"{name} must be a numpy.ndarray."
        )

    if image.ndim != 2:
        raise ValueError(
            f"{name} must be a single-channel grayscale image. "
            f"Shape: {image.shape}"
        )

    if image.dtype != np.uint8:
        raise ValueError(
            f"{name} must be uint8. "
            f"Got: {image.dtype}"
        )


# =============================================================================
# IO
# =============================================================================

def load_gray(
    path: Path | str,
) -> np.ndarray:
    """
    Load an image as uint8 grayscale.
    """

    path = Path(path)

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise RuntimeError(
            f"Could not read image:\n{path}"
        )

    validate_gray_uint8(
        image,
        str(path),
    )

    return image


def save_gray(
    image: np.ndarray,
    path: Path | str,
) -> None:
    """
    Save an uint8 grayscale image.
    """

    validate_gray_uint8(
        image,
        "image",
    )

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ok = cv2.imwrite(
        str(path),
        image,
    )

    if not ok:
        raise RuntimeError(
            f"Failed to write image:\n{path}"
        )


# =============================================================================
# P1 - ROBUST NORMALIZATION
# =============================================================================

def robust_normalize(
    image: np.ndarray,
    config: PreprocessingConfig,
) -> np.ndarray:
    """
    Robust percentile normalization.

        P_low  = percentile(image, low_percentile)
        P_high = percentile(image, high_percentile)

        output =
            clip(
                (image - P_low) /
                (P_high - P_low),
                0,
                1
            )

    The absolute min/max are deliberately not used.
    """

    validate_gray_uint8(
        image,
        "image",
    )

    image_f = image.astype(
        np.float32
    )

    low = float(
        np.percentile(
            image_f,
            config.low_percentile,
        )
    )

    high = float(
        np.percentile(
            image_f,
            config.high_percentile,
        )
    )

    if high <= low:

        return np.zeros_like(
            image,
            dtype=np.uint8,
        )

    normalized = (
        image_f - low
    ) / (
        high - low
    )

    normalized = np.clip(
        normalized,
        0.0,
        1.0,
    )

    return np.round(
        normalized * 255.0
    ).astype(
        np.uint8
    )


# =============================================================================
# P2 - ILLUMINATION CORRECTION
# =============================================================================

def illumination_correct(
    image: np.ndarray,
    config: PreprocessingConfig,
) -> np.ndarray:
    """
    Correct slowly varying illumination.

    Model:

        B = GaussianBlur(I + 1)

        I_corr = (I + 1) / max(B, epsilon)

    followed by robust percentile normalization.

    The selected production sigma is 20.0.
    """

    validate_gray_uint8(
        image,
        "image",
    )

    image_f = (
        image.astype(
            np.float32
        )
        + 1.0
    )

    background = cv2.GaussianBlur(
        image_f,
        ksize=(0, 0),
        sigmaX=config.illumination_sigma,
        sigmaY=config.illumination_sigma,
        borderType=cv2.BORDER_REFLECT,
    )

    corrected = (
        image_f
        / np.maximum(
            background,
            EPSILON,
        )
    )

    low = float(
        np.percentile(
            corrected,
            config.low_percentile,
        )
    )

    high = float(
        np.percentile(
            corrected,
            config.high_percentile,
        )
    )

    if high <= low:

        return np.zeros_like(
            image,
            dtype=np.uint8,
        )

    corrected = (
        corrected - low
    ) / (
        high - low
    )

    corrected = np.clip(
        corrected,
        0.0,
        1.0,
    )

    return np.round(
        corrected * 255.0
    ).astype(
        np.uint8
    )


# =============================================================================
# P3 - GRADIENT REPRESENTATION
# =============================================================================

def gradient_magnitude(
    image: np.ndarray,
    config: PreprocessingConfig,
) -> np.ndarray:
    """
    Calculate Sobel gradient magnitude and robustly normalize it.
    """

    validate_gray_uint8(
        image,
        "image",
    )

    image_f = image.astype(
        np.float32
    )

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

    low = float(
        np.percentile(
            magnitude,
            config.low_percentile,
        )
    )

    high = float(
        np.percentile(
            magnitude,
            config.high_percentile,
        )
    )

    if high <= low:

        return np.zeros_like(
            image,
            dtype=np.uint8,
        )

    magnitude = (
        magnitude - low
    ) / (
        high - low
    )

    magnitude = np.clip(
        magnitude,
        0.0,
        1.0,
    )

    return np.round(
        magnitude * 255.0
    ).astype(
        np.uint8
    )


# =============================================================================
# P5 - CLAHE
# =============================================================================

def clahe_enhance(
    image: np.ndarray,
    config: PreprocessingConfig,
) -> np.ndarray:
    """
    Apply fixed CLAHE parameters.

        clipLimit = 2.0
        tileGridSize = (8, 8)
    """

    validate_gray_uint8(
        image,
        "image",
    )

    clahe = cv2.createCLAHE(
        clipLimit=config.clahe_clip_limit,
        tileGridSize=config.clahe_tile_grid_size,
    )

    return clahe.apply(
        image
    )


# =============================================================================
# IMAGE STATISTICS
# =============================================================================

def image_statistics(
    image: np.ndarray,
) -> dict:
    """
    Reproducible summary statistics.
    """

    validate_gray_uint8(
        image,
        "image",
    )

    image_f = image.astype(
        np.float32
    )

    return {
        "shape": [
            int(image.shape[0]),
            int(image.shape[1]),
        ],
        "dtype": str(
            image.dtype
        ),
        "min": int(
            image.min()
        ),
        "max": int(
            image.max()
        ),
        "mean": float(
            image_f.mean()
        ),
        "median": float(
            np.median(
                image_f
            )
        ),
        "std": float(
            image_f.std()
        ),
        "p01": float(
            np.percentile(
                image_f,
                1,
            )
        ),
        "p05": float(
            np.percentile(
                image_f,
                5,
            )
        ),
        "p25": float(
            np.percentile(
                image_f,
                25,
            )
        ),
        "p75": float(
            np.percentile(
                image_f,
                75,
            )
        ),
        "p95": float(
            np.percentile(
                image_f,
                95,
            )
        ),
        "p99": float(
            np.percentile(
                image_f,
                99,
            )
        ),
        "dark_fraction": float(
            np.mean(
                image <= 5
            )
        ),
        "high_fraction": float(
            np.mean(
                image >= 250
            )
        ),
    }


# =============================================================================
# METADATA
# =============================================================================

def representation_metadata(
    name: str,
    config: PreprocessingConfig,
) -> dict:
    """
    Return the frozen description of a preprocessing representation.
    """

    common = {
        "representation": name,
        "resampling": config.resampling,
        "input_dtype": "uint8",
        "output_dtype": "uint8",
    }

    if name == "P0_RAW":

        return {
            **common,
            "method": "raw",
            "normalization": "NONE",
            "illumination_correction": "NONE",
            "gradient": "NONE",
            "clahe": "NONE",
        }

    if name == "P1_ROBUST_NORMALIZED":

        return {
            **common,
            "method":
                "percentile_normalization",
            "low_percentile":
                config.low_percentile,
            "high_percentile":
                config.high_percentile,
        }

    if name == "P2_ILLUMINATION_CORRECTED":

        return {
            **common,
            "method":
                "division_by_gaussian_background",
            "illumination_sigma":
                config.illumination_sigma,
            "post_normalization":
                "percentile_normalization",
            "low_percentile":
                config.low_percentile,
            "high_percentile":
                config.high_percentile,
        }

    if name == "P3_GRADIENT":

        return {
            **common,
            "method":
                "gradient_magnitude",
            "operator":
                "Sobel_3x3",
            "low_percentile":
                config.low_percentile,
            "high_percentile":
                config.high_percentile,
        }

    if name == "P4_COMBINED":

        return {
            **common,
            "method":
                "normalization"
                "+illumination_correction"
                "+gradient_magnitude",
            "low_percentile":
                config.low_percentile,
            "high_percentile":
                config.high_percentile,
            "illumination_sigma":
                config.illumination_sigma,
            "gradient":
                "Sobel_3x3_magnitude",
        }

    if name == "P5_CLAHE":

        return {
            **common,
            "method":
                "CLAHE",
            "clip_limit":
                config.clahe_clip_limit,
            "tile_grid_size":
                list(
                    config.clahe_tile_grid_size
                ),
        }

    if name == "P6_ILLUMINATION_CLAHE":

        return {
            **common,
            "method":
                "illumination_correction+CLAHE",
            "illumination_sigma":
                config.illumination_sigma,
            "clip_limit":
                config.clahe_clip_limit,
            "tile_grid_size":
                list(
                    config.clahe_tile_grid_size
                ),
        }

    raise ValueError(
        f"Unknown representation: {name}"
    )


# =============================================================================
# SINGLE IMAGE PREPROCESSING
# =============================================================================

def preprocess_image(
    image: np.ndarray,
    representation: RepresentationName,
    config: PreprocessingConfig | None = None,
) -> np.ndarray:
    """
    Preprocess a single image using a named representation.
    """

    validate_gray_uint8(
        image,
        "input image",
    )

    if config is None:
        config = PreprocessingConfig()

    # -------------------------------------------------------------------------
    # P0
    # -------------------------------------------------------------------------

    if representation == "P0_RAW":

        return image.copy()

    # -------------------------------------------------------------------------
    # P1
    # -------------------------------------------------------------------------

    if representation == "P1_ROBUST_NORMALIZED":

        return robust_normalize(
            image,
            config,
        )

    # -------------------------------------------------------------------------
    # P2
    # -------------------------------------------------------------------------

    if representation == "P2_ILLUMINATION_CORRECTED":

        return illumination_correct(
            image,
            config,
        )

    # -------------------------------------------------------------------------
    # P3
    # -------------------------------------------------------------------------

    if representation == "P3_GRADIENT":

        return gradient_magnitude(
            image,
            config,
        )

    # -------------------------------------------------------------------------
    # P4
    # -------------------------------------------------------------------------

    if representation == "P4_COMBINED":

        normalized = robust_normalize(
            image,
            config,
        )

        corrected = illumination_correct(
            normalized,
            config,
        )

        return gradient_magnitude(
            corrected,
            config,
        )

    # -------------------------------------------------------------------------
    # P5
    # -------------------------------------------------------------------------

    if representation == "P5_CLAHE":

        return clahe_enhance(
            image,
            config,
        )

    # -------------------------------------------------------------------------
    # P6
    # -------------------------------------------------------------------------

    if representation == "P6_ILLUMINATION_CLAHE":

        corrected = illumination_correct(
            image,
            config,
        )

        return clahe_enhance(
            corrected,
            config,
        )

    raise ValueError(
        f"Unknown representation: {representation}"
    )


# =============================================================================
# PAIR PREPROCESSING
# =============================================================================

def preprocess_pair(
    moving: np.ndarray,
    reference: np.ndarray,
    pair_id: str,
    representation: RepresentationName,
    config: PreprocessingConfig | None = None,
) -> PreprocessedPair:
    """
    Apply the same named representation to a moving/reference pair.
    """

    validate_gray_uint8(
        moving,
        "moving",
    )

    validate_gray_uint8(
        reference,
        "reference",
    )

    if config is None:
        config = PreprocessingConfig()

    moving_output = preprocess_image(
        moving,
        representation,
        config,
    )

    reference_output = preprocess_image(
        reference,
        representation,
        config,
    )

    metadata = {
        "pair_id":
            pair_id,

        "preprocessing":
            representation_metadata(
                representation,
                config,
            ),

        "moving_statistics":
            image_statistics(
                moving_output
            ),

        "reference_statistics":
            image_statistics(
                reference_output
            ),
    }

    return PreprocessedPair(
        pair_id=pair_id,
        representation=representation,
        moving=moving_output,
        reference=reference_output,
        metadata=metadata,
    )


# =============================================================================
# SAVE PROCESSED PAIR
# =============================================================================

def save_preprocessed_pair(
    result: PreprocessedPair,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
) -> Path:
    """
    Save one processed pair.

    Output:

        <output_root>/<pair_id>/<representation>/
            moving.png
            reference.png
            metadata.json
    """

    output_root = Path(
        output_root
    )

    output_dir = (
        output_root
        / result.pair_id
        / result.representation
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_gray(
        result.moving,
        output_dir
        / "moving.png",
    )

    save_gray(
        result.reference,
        output_dir
        / "reference.png",
    )

    metadata_path = (
        output_dir
        / "metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            result.metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    return output_dir


# =============================================================================
# SAVE ALL REPRESENTATIONS
# =============================================================================

def generate_all_representations(
    pair_id: str,
    moving: np.ndarray,
    reference: np.ndarray,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    representations: list[str] | None = None,
    config: PreprocessingConfig | None = None,
) -> dict[str, PreprocessedPair]:
    """
    Generate and save selected representations.

    By default all seven representations are produced.
    """

    if config is None:
        config = PreprocessingConfig()

    if representations is None:
        representations = list(
            ALL_REPRESENTATIONS
        )

    results: dict[
        str,
        PreprocessedPair,
    ] = {}

    for representation in representations:

        if representation not in ALL_REPRESENTATIONS:

            raise ValueError(
                f"Unknown representation: "
                f"{representation}"
            )

        result = preprocess_pair(
            moving,
            reference,
            pair_id,
            representation,
            config,
        )

        save_preprocessed_pair(
            result,
            output_root,
        )

        results[
            representation
        ] = result

    return results


# =============================================================================
# CONFIGURATION MANIFEST
# =============================================================================

def save_config_manifest(
    output_root: Path | str,
    config: PreprocessingConfig,
) -> Path:
    """
    Save the frozen production configuration.
    """

    output_root = Path(
        output_root
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = {

        "module":
            "production_preprocessing",

        "configuration":
            asdict(config),

        "all_representations":
            ALL_REPRESENTATIONS,

        "recommended_representations":
            RECOMMENDED_REPRESENTATIONS,

        "scientific_note":
            "P2 and P6 use illumination sigma=20 based on "
            "the controlled Pair-002 sigma sensitivity experiment. "
            "This is a preprocessing-selection result; "
            "correspondence performance must be established "
            "by the downstream CV/DL evaluations.",

        "resampling":
            "NONE",
    }

    path = (
        output_root
        / "production_preprocessing_manifest.json"
    )

    path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    return path


# =============================================================================
# COMMAND-LINE DEMO / GENERATION
# =============================================================================

def main() -> None:
    """
    Generate all frozen representations for Pair-002 from the P0 inputs.

    This provides both a reusable module and a convenient command-line entry
    point while the project is under active development.
    """

    pair_id = DEFAULT_PAIR_ID

    moving_path = (
        DEFAULT_INPUT_ROOT
        / pair_id
        / "moving_raw_overlap.png"
    )

    reference_path = (
        DEFAULT_INPUT_ROOT
        / pair_id
        / "reference_raw.png"
    )

    print(
        "=" * 80
    )

    print(
        "PRODUCTION PREPROCESSING"
    )

    print(
        f"Pair: {pair_id}"
    )

    print(
        "=" * 80
    )

    if not moving_path.exists():

        raise FileNotFoundError(
            f"Missing moving input:\n"
            f"{moving_path}"
        )

    if not reference_path.exists():

        raise FileNotFoundError(
            f"Missing reference input:\n"
            f"{reference_path}"
        )

    moving = load_gray(
        moving_path
    )

    reference = load_gray(
        reference_path
    )

    print(
        f"Moving   : "
        f"{moving.shape[1]} x "
        f"{moving.shape[0]}"
    )

    print(
        f"Reference: "
        f"{reference.shape[1]} x "
        f"{reference.shape[0]}"
    )

    config = PreprocessingConfig()

    print()
    print(
        "Frozen production parameters:"
    )

    print(
        f"  Percentiles: "
        f"{config.low_percentile} / "
        f"{config.high_percentile}"
    )

    print(
        f"  Illumination sigma: "
        f"{config.illumination_sigma}"
    )

    print(
        f"  CLAHE clipLimit: "
        f"{config.clahe_clip_limit}"
    )

    print(
        f"  CLAHE tileGridSize: "
        f"{config.clahe_tile_grid_size}"
    )

    print(
        "  Resampling: NONE"
    )

    results = generate_all_representations(
        pair_id,
        moving,
        reference,
        DEFAULT_OUTPUT_ROOT,
        ALL_REPRESENTATIONS,
        config,
    )

    manifest_path = save_config_manifest(
        DEFAULT_OUTPUT_ROOT
        / pair_id,
        config,
    )

    print()

    for name, result in results.items():

        output_dir = (
            DEFAULT_OUTPUT_ROOT
            / pair_id
            / name
        )

        moving_stats = (
            result.metadata[
                "moving_statistics"
            ]
        )

        reference_stats = (
            result.metadata[
                "reference_statistics"
            ]
        )

        print(
            "-" * 80
        )

        print(
            name
        )

        print(
            f"  Moving output    : "
            f"{output_dir / 'moving.png'}"
        )

        print(
            f"  Reference output : "
            f"{output_dir / 'reference.png'}"
        )

        print(
            f"  Moving mean/std  : "
            f"{moving_stats['mean']:.3f} / "
            f"{moving_stats['std']:.3f}"
        )

        print(
            f"  Reference mean/std: "
            f"{reference_stats['mean']:.3f} / "
            f"{reference_stats['std']:.3f}"
        )

    print()
    print(
        "=" * 80
    )

    print(
        "PRODUCTION PREPROCESSING COMPLETE"
    )

    print(
        f"Manifest: {manifest_path}"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()