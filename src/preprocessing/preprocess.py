from pathlib import Path
from typing import Optional, Dict, Any

import cv2
import numpy as np


# ============================================================
# LUNAR IMAGE PREPROCESSING
#
# Supported sensors / products:
#   - OHRC
#   - TMC-2
#   - IIRS-derived image/band products
#   - LROC
#   - generic
#
# Pipeline:
#
#   ORIGINAL
#       |
#       v
#   grayscale / single band
#       |
#       v
#   valid-pixel mask
#       |
#       v
#   optional sensor-specific correction
#       |
#       v
#   robust normalization
#       |
#       v
#   CLAHE
#       |
#       v
#   structural representation
#       |
#       v
#   multi-scale pyramid
#
# OHRC-specific optional path:
#
#   OHRC
#      |
#   column-gain destriping
#      |
#   normalization
#      |
#   CLAHE
#
# IMPORTANT:
#   The original input file is NEVER modified.
# ============================================================


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(path: str) -> np.ndarray:
    """
    Load an image while preserving its original data as much
    as possible.

    The source file itself is never modified.
    """

    image = cv2.imread(
        str(path),
        cv2.IMREAD_UNCHANGED
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not read image:\n{path}"
        )

    return image


# ============================================================
# GRAYSCALE / SINGLE BAND
# ============================================================

def to_grayscale(
    image: np.ndarray
) -> np.ndarray:
    """
    Convert an image to a single-channel representation.

    Already-grayscale images are copied unchanged.

    For multi-channel images, standard OpenCV BGR→gray
    conversion is used.
    """

    if image.ndim == 2:

        return image.copy()

    if image.ndim == 3:

        if image.shape[2] == 1:

            return image[:, :, 0].copy()

        return cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    raise ValueError(
        f"Unsupported image shape: {image.shape}"
    )


# ============================================================
# VALID PIXEL MASK
# ============================================================

def create_valid_mask(
    image: np.ndarray,
    invalid_value: Optional[float] = 0
) -> np.ndarray:
    """
    Create a binary valid-pixel mask.

    Returns:
        255 = valid
        0   = invalid

    By default, zero-valued pixels are considered invalid.
    """

    image_f = image.astype(
        np.float32
    )

    valid = np.isfinite(
        image_f
    )

    if invalid_value is not None:

        valid &= (
            image_f != invalid_value
        )

    return (
        valid.astype(np.uint8)
        * 255
    )


# ============================================================
# ROBUST NORMALIZATION
# ============================================================

def robust_normalize(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0
) -> np.ndarray:
    """
    Robustly normalize an image to uint8 [0,255].

    Percentile normalization prevents a small number of
    extreme/saturated pixels from controlling the entire
    contrast range.
    """

    image_f = image.astype(
        np.float32
    )

    if mask is None:

        valid = np.isfinite(
            image_f
        )

    else:

        valid = (
            mask > 0
        ) & np.isfinite(
            image_f
        )

    if not np.any(valid):

        return np.zeros(
            image.shape,
            dtype=np.uint8
        )

    values = image_f[
        valid
    ]

    low = np.percentile(
        values,
        low_percentile
    )

    high = np.percentile(
        values,
        high_percentile
    )

    if high <= low:

        high = low + 1.0

    result = (
        (image_f - low)
        /
        (high - low)
        * 255.0
    )

    result = np.clip(
        result,
        0,
        255
    )

    result[
        ~valid
    ] = 0

    return result.astype(
        np.uint8
    )


# ============================================================
# OHRC COLUMN-GAIN DESTRIPING
# ============================================================

def destripe_ohrc(
    image: np.ndarray,
    smooth_kernel: int = 151,
    strength: float = 0.85
) -> np.ndarray:
    """
    Suppress vertical column-wise striping in OHRC imagery.

    The observed OHRC intensity can contain a column-dependent
    multiplicative response. A logarithmic representation
    converts this multiplicative variation into an additive
    column bias.

    Procedure:

        1. Determine valid pixels.
        2. Convert image to log domain.
        3. Estimate a robust median value for each column.
        4. Smooth the column profile.
        5. Estimate high-frequency stripe component.
        6. Remove a controlled fraction of the component.
        7. Restore robust global intensity range.

    Parameters
    ----------
    image:
        Single-channel image.

    smooth_kernel:
        Horizontal smoothing scale for the column profile.

    strength:
        Correction strength.

        0.0 = no correction
        1.0 = full estimated correction

        Default = 0.85.

    Returns
    -------
    uint8 corrected image.
    """

    if image.ndim != 2:

        raise ValueError(
            "OHRC destriping requires a single-channel image."
        )

    if not (
        0.0 <= strength <= 1.0
    ):

        raise ValueError(
            "strength must be between 0 and 1."
        )

    image_f = image.astype(
        np.float32
    )

    h, w = image_f.shape

    # --------------------------------------------------------
    # Valid pixels
    # --------------------------------------------------------

    valid = (
        np.isfinite(image_f)
        &
        (image_f > 0)
    )

    if not np.any(valid):

        return np.zeros(
            image.shape,
            dtype=np.uint8
        )

    # --------------------------------------------------------
    # Prevent log(0).
    #
    # Use a small fraction of the robust low percentile as
    # the floor.
    # --------------------------------------------------------

    positive_values = image_f[
        valid
    ]

    p_low = float(
        np.percentile(
            positive_values,
            0.5
        )
    )

    floor = max(
        1.0,
        p_low * 0.05
    )

    work = np.maximum(
        image_f,
        floor
    )

    # --------------------------------------------------------
    # Log transformation.
    # --------------------------------------------------------

    log_image = np.log(
        work
    )

    # --------------------------------------------------------
    # Robust column profile.
    #
    # Median is preferred over mean because craters and
    # shadows should have less influence on the estimated
    # detector pattern.
    # --------------------------------------------------------

    column_profile = np.zeros(
        w,
        dtype=np.float32
    )

    global_median = float(
        np.median(
            log_image[valid]
        )
    )

    for x in range(w):

        column_values = log_image[
            valid[:, x],
            x
        ]

        if column_values.size > 0:

            column_profile[x] = np.median(
                column_values
            )

        else:

            column_profile[x] = global_median

    # --------------------------------------------------------
    # Valid smoothing kernel.
    # --------------------------------------------------------

    smooth_kernel = int(
        smooth_kernel
    )

    if smooth_kernel < 31:

        smooth_kernel = 31

    if smooth_kernel % 2 == 0:

        smooth_kernel += 1

    if smooth_kernel > w:

        smooth_kernel = (
            w
            if w % 2 == 1
            else w - 1
        )

    if smooth_kernel < 3:

        smooth_kernel = 3

    # --------------------------------------------------------
    # Smooth column profile.
    #
    # Broad changes are retained as illumination. The
    # high-frequency residual is treated as striping.
    # --------------------------------------------------------

    smooth_profile = cv2.GaussianBlur(
        column_profile.reshape(1, -1),
        (smooth_kernel, 1),
        0
    ).reshape(-1)

    # --------------------------------------------------------
    # Stripe component.
    # --------------------------------------------------------

    stripe = (
        column_profile
        - smooth_profile
    )

    # --------------------------------------------------------
    # Controlled correction.
    # --------------------------------------------------------

    stripe *= float(
        strength
    )

    corrected_log = (
        log_image
        - stripe[None, :]
    )

    corrected = np.exp(
        corrected_log
    )

    # --------------------------------------------------------
    # Restore invalid pixels.
    # --------------------------------------------------------

    corrected[
        ~valid
    ] = 0

    # --------------------------------------------------------
    # Robust output normalization.
    # --------------------------------------------------------

    corrected_values = corrected[
        valid
    ]

    low = np.percentile(
        corrected_values,
        1.0
    )

    high = np.percentile(
        corrected_values,
        99.0
    )

    if high <= low:

        high = low + 1.0

    corrected = (
        (corrected - low)
        /
        (high - low)
        * 255.0
    )

    corrected = np.clip(
        corrected,
        0,
        255
    )

    corrected[
        ~valid
    ] = 0

    return corrected.astype(
        np.uint8
    )


# ============================================================
# CLAHE
# ============================================================

def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size=(8, 8)
) -> np.ndarray:
    """
    Apply CLAHE local contrast enhancement.

    CLAHE is useful when illumination differs spatially.
    """

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    return clahe.apply(
        image
    )


# ============================================================
# GRADIENT MAGNITUDE
# ============================================================

def gradient_magnitude(
    image: np.ndarray
) -> np.ndarray:
    """
    Calculate normalized gradient magnitude.

    Emphasizes terrain structure:

        - crater rims
        - ridges
        - boundaries
        - shadow edges
        - local texture
    """

    image_f = image.astype(
        np.float32
    )

    gx = cv2.Sobel(
        image_f,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        image_f,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    magnitude = cv2.magnitude(
        gx,
        gy
    )

    return cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(
        np.uint8
    )


# ============================================================
# STRUCTURAL REPRESENTATION
# ============================================================

def structural_image(
    normalized: np.ndarray,
    gradient_weight: float = 0.30
) -> np.ndarray:
    """
    Combine intensity and gradient information.

    Default:
        70% intensity
        30% gradient
    """

    if not (
        0.0 <= gradient_weight <= 1.0
    ):

        raise ValueError(
            "gradient_weight must be between 0 and 1."
        )

    gradient = gradient_magnitude(
        normalized
    )

    intensity_weight = (
        1.0
        - gradient_weight
    )

    result = cv2.addWeighted(
        normalized,
        intensity_weight,
        gradient,
        gradient_weight,
        0
    )

    return result


# ============================================================
# MULTI-SCALE PYRAMID
# ============================================================

def build_pyramid(
    image: np.ndarray,
    levels: int = 3
):
    """
    Build a Gaussian image pyramid.

    Level 0:
        original working resolution

    Level 1:
        approximately 1/2 resolution

    Level 2:
        approximately 1/4 resolution

    etc.
    """

    if levels < 1:

        raise ValueError(
            "levels must be >= 1."
        )

    pyramid = [
        image
    ]

    current = image

    for _ in range(
        levels - 1
    ):

        current = cv2.pyrDown(
            current
        )

        pyramid.append(
            current
        )

    return pyramid


# ============================================================
# COMPLETE PREPROCESSING PIPELINE
# ============================================================

def preprocess_image(
    path: str,
    sensor: str = "generic",
    use_clahe: bool = True,
    use_gradient: bool = True,
    use_destriping: bool = False,
    pyramid_levels: int = 3
) -> Dict[str, Any]:
    """
    Complete sensor-aware preprocessing pipeline.

    Parameters
    ----------
    path:
        Input image path.

    sensor:
        Sensor/product name:
            OHRC
            TMC-2
            IIRS
            LROC
            generic

    use_clahe:
        Apply CLAHE.

    use_gradient:
        Generate structural representation.

    use_destriping:
        Apply OHRC-specific column-gain correction.

        Must only be enabled for OHRC.

    pyramid_levels:
        Number of multi-scale pyramid levels.

    Returns
    -------
    Dictionary containing all intermediate representations.
    """

    # --------------------------------------------------------
    # ORIGINAL
    # --------------------------------------------------------

    original = load_image(
        path
    )

    # --------------------------------------------------------
    # SINGLE CHANNEL
    # --------------------------------------------------------

    grayscale = to_grayscale(
        original
    )

    # --------------------------------------------------------
    # ORIGINAL VALID MASK
    # --------------------------------------------------------

    valid_mask = create_valid_mask(
        grayscale
    )

    # --------------------------------------------------------
    # WORKING IMAGE
    # --------------------------------------------------------

    working = grayscale.copy()

    # --------------------------------------------------------
    # SENSOR-SPECIFIC CORRECTION
    # --------------------------------------------------------

    if use_destriping:

        if sensor.upper() != "OHRC":

            raise ValueError(
                "Destriping is currently implemented "
                "only for OHRC."
            )

        working = destripe_ohrc(
            working,
            smooth_kernel=151,
            strength=0.85
        )

        working_mask = create_valid_mask(
            working
        )

    else:

        working_mask = valid_mask.copy()

    # --------------------------------------------------------
    # ROBUST NORMALIZATION
    # --------------------------------------------------------

    normalized = robust_normalize(
        working,
        working_mask,
        low_percentile=1.0,
        high_percentile=99.0
    )

    # --------------------------------------------------------
    # CLAHE
    # --------------------------------------------------------

    if use_clahe:

        enhanced = apply_clahe(
            normalized,
            clip_limit=2.0,
            tile_grid_size=(8, 8)
        )

    else:

        enhanced = normalized.copy()

    # --------------------------------------------------------
    # STRUCTURAL REPRESENTATION
    # --------------------------------------------------------

    if use_gradient:

        structural = structural_image(
            enhanced,
            gradient_weight=0.30
        )

    else:

        structural = enhanced.copy()

    # --------------------------------------------------------
    # MULTI-SCALE PYRAMID
    # --------------------------------------------------------

    pyramid = build_pyramid(
        structural,
        levels=pyramid_levels
    )

    return {
        "original": original,

        "grayscale": grayscale,

        "valid_mask": valid_mask,

        "working": working,

        "working_mask": working_mask,

        "normalized": normalized,

        "clahe": enhanced,

        "structural": structural,

        "pyramid": pyramid,
    }


# ============================================================
# SAVE PREPROCESSING PRODUCTS
# ============================================================

def save_preprocessing_results(
    results: Dict[str, Any],
    output_dir: str
):
    """
    Save derived preprocessing products.

    The original source image is deliberately NOT copied.
    """

    output = Path(
        output_dir
    )

    output.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Masks
    # --------------------------------------------------------

    cv2.imwrite(
        str(
            output / "valid_mask.png"
        ),
        results["valid_mask"]
    )

    cv2.imwrite(
        str(
            output / "working_mask.png"
        ),
        results["working_mask"]
    )

    # --------------------------------------------------------
    # Main representations
    # --------------------------------------------------------

    cv2.imwrite(
        str(
            output / "working.png"
        ),
        results["working"]
    )

    cv2.imwrite(
        str(
            output / "normalized.png"
        ),
        results["normalized"]
    )

    cv2.imwrite(
        str(
            output / "clahe.png"
        ),
        results["clahe"]
    )

    cv2.imwrite(
        str(
            output / "structural.png"
        ),
        results["structural"]
    )

    # --------------------------------------------------------
    # Pyramid
    # --------------------------------------------------------

    for i, level in enumerate(
        results["pyramid"]
    ):

        cv2.imwrite(
            str(
                output
                / f"pyramid_{i}.png"
            ),
            level
        )


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Sensor-aware lunar image preprocessing"
        )
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Input image path"
    )

    parser.add_argument(
        "--sensor",
        default="generic",
        help=(
            "Sensor/product: OHRC, TMC-2, "
            "IIRS, LROC or generic"
        )
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output directory"
    )

    parser.add_argument(
        "--destripe",
        action="store_true",
        help=(
            "Enable OHRC column-gain destriping"
        )
    )

    args = parser.parse_args()

    print("=" * 70)
    print(
        "LUNAR IMAGE PREPROCESSING"
    )
    print("=" * 70)

    print(
        f"Image    : {args.image}"
    )

    print(
        f"Sensor   : {args.sensor}"
    )

    print(
        f"Destripe : {args.destripe}"
    )

    print()

    # --------------------------------------------------------
    # Run pipeline
    # --------------------------------------------------------

    results = preprocess_image(
        path=args.image,
        sensor=args.sensor,
        use_clahe=True,
        use_gradient=True,
        use_destriping=args.destripe,
        pyramid_levels=3
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_preprocessing_results(
        results,
        args.output
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print()
    print(
        f"Original shape   : "
        f"{results['original'].shape}"
    )

    print(
        f"Working shape    : "
        f"{results['working'].shape}"
    )

    print(
        f"Structural shape : "
        f"{results['structural'].shape}"
    )

    print()

    print(
        "Pyramid:"
    )

    for i, level in enumerate(
        results["pyramid"]
    ):

        print(
            f"  Level {i}: "
            f"{level.shape}"
        )

    print()

    print(
        "Saved preprocessing outputs to:"
    )

    print(
        args.output
    )

    print()

    print(
        "Preprocessing complete."
    )