from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import cv2
import numpy as np


# ============================================================
# SIFT FEATURE EXTRACTION
#
# This module provides a clean baseline interface for:
#
#     image + optional mask
#             ↓
#           SIFT
#             ↓
#       keypoints
#       RootSIFT descriptors
#
# The interface is deliberately independent of Pair 001.
#
# Later SuperPoint can expose the same basic outputs:
#
#     keypoints
#     descriptors
#     scores
#
# ============================================================


# ============================================================
# ROOTSIFT
# ============================================================

def rootsift(
    descriptors: np.ndarray,
    eps: float = 1e-7
) -> np.ndarray:
    """
    Convert standard SIFT descriptors into RootSIFT.

    Procedure:

        1. L1-normalize descriptor.
        2. Take element-wise square root.

    RootSIFT is useful because it changes the descriptor
    representation to a Hellinger-kernel-like form, which
    can improve matching robustness.

    Parameters
    ----------
    descriptors:
        Standard SIFT descriptors.

    eps:
        Numerical stability constant.

    Returns
    -------
    RootSIFT descriptors as float32.
    """

    if descriptors is None:

        return np.empty(
            (0, 128),
            dtype=np.float32
        )

    descriptors = descriptors.astype(
        np.float32
    )

    # L1 normalization
    norms = np.sum(
        descriptors,
        axis=1,
        keepdims=True
    )

    descriptors = (
        descriptors
        /
        (norms + eps)
    )

    # Hellinger / square-root transform
    descriptors = np.sqrt(
        np.maximum(
            descriptors,
            0
        )
    )

    return descriptors.astype(
        np.float32
    )


# ============================================================
# CREATE SIFT
# ============================================================

def create_sift(
    nfeatures: int = 12000,
    n_octave_layers: int = 3,
    contrast_threshold: float = 0.01,
    edge_threshold: float = 20.0,
    sigma: float = 1.6
) -> cv2.SIFT:
    """
    Create the SIFT detector.

    The parameters are deliberately centralized here so
    that we don't scatter tuning values throughout the
    project.
    """

    return cv2.SIFT_create(
        nfeatures=nfeatures,
        nOctaveLayers=n_octave_layers,
        contrastThreshold=contrast_threshold,
        edgeThreshold=edge_threshold,
        sigma=sigma
    )


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_sift(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    nfeatures: int = 12000,
    contrast_threshold: float = 0.01,
    edge_threshold: float = 20.0,
    sigma: float = 1.6
) -> Dict[str, Any]:
    """
    Extract SIFT features and convert descriptors to RootSIFT.

    Parameters
    ----------
    image:
        Single-channel uint8 image.

    mask:
        Optional uint8 mask:
            255 = valid
            0   = invalid

    Returns
    -------
    Dictionary:

        keypoints
        descriptors
        points
        responses
        sizes
        angles
    """

    if image is None:

        raise ValueError(
            "Input image is None."
        )

    # --------------------------------------------------------
    # Ensure single-channel uint8
    # --------------------------------------------------------

    if image.ndim == 3:

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    if image.dtype != np.uint8:

        image = cv2.normalize(
            image,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(
            np.uint8
        )

    # --------------------------------------------------------
    # Validate mask
    # --------------------------------------------------------

    if mask is not None:

        if mask.shape != image.shape:

            raise ValueError(
                "Mask shape does not match image shape."
            )

        if mask.dtype != np.uint8:

            mask = mask.astype(
                np.uint8
            )

    # --------------------------------------------------------
    # SIFT
    # --------------------------------------------------------

    sift = create_sift(
        nfeatures=nfeatures,
        contrast_threshold=contrast_threshold,
        edge_threshold=edge_threshold,
        sigma=sigma
    )

    keypoints, descriptors = (
        sift.detectAndCompute(
            image,
            mask
        )
    )

    # --------------------------------------------------------
    # No features
    # --------------------------------------------------------

    if keypoints is None:

        keypoints = []

    if descriptors is None:

        descriptors = np.empty(
            (0, 128),
            dtype=np.float32
        )

    # --------------------------------------------------------
    # RootSIFT
    # --------------------------------------------------------

    rootsift_descriptors = rootsift(
        descriptors
    )

    # --------------------------------------------------------
    # Coordinates
    # --------------------------------------------------------

    if len(keypoints) > 0:

        points = np.array(
            [
                kp.pt
                for kp in keypoints
            ],
            dtype=np.float32
        )

        responses = np.array(
            [
                kp.response
                for kp in keypoints
            ],
            dtype=np.float32
        )

        sizes = np.array(
            [
                kp.size
                for kp in keypoints
            ],
            dtype=np.float32
        )

        angles = np.array(
            [
                kp.angle
                for kp in keypoints
            ],
            dtype=np.float32
        )

    else:

        points = np.empty(
            (0, 2),
            dtype=np.float32
        )

        responses = np.empty(
            (0,),
            dtype=np.float32
        )

        sizes = np.empty(
            (0,),
            dtype=np.float32
        )

        angles = np.empty(
            (0,),
            dtype=np.float32
        )

    return {
        "keypoints": keypoints,
        "descriptors": rootsift_descriptors,
        "points": points,
        "responses": responses,
        "sizes": sizes,
        "angles": angles,
    }


# ============================================================
# SAVE FEATURES
# ============================================================

def save_features(
    features: Dict[str, Any],
    output_path: str
):
    """
    Save extracted feature data to NPZ.

    Keypoint objects themselves are converted into numerical
    arrays so the file remains easy to load later.
    """

    output = Path(
        output_path
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    keypoints = features[
        "keypoints"
    ]

    if len(keypoints) > 0:

        keypoint_data = np.array(
            [
                [
                    kp.pt[0],
                    kp.pt[1],
                    kp.size,
                    kp.angle,
                    kp.response,
                    kp.octave,
                    kp.class_id
                ]
                for kp in keypoints
            ],
            dtype=np.float32
        )

    else:

        keypoint_data = np.empty(
            (0, 7),
            dtype=np.float32
        )

    np.savez_compressed(
        output,
        keypoints=keypoint_data,
        points=features["points"],
        descriptors=features["descriptors"],
        responses=features["responses"],
        sizes=features["sizes"],
        angles=features["angles"]
    )


# ============================================================
# DRAW FEATURE VISUALIZATION
# ============================================================

def draw_features(
    image: np.ndarray,
    features: Dict[str, Any],
    max_display: int = 3000
) -> np.ndarray:
    """
    Draw detected SIFT keypoints.

    Only a subset is displayed so visualization remains
    manageable for large OHRC images.
    """

    keypoints = features[
        "keypoints"
    ]

    if len(keypoints) == 0:

        return cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR
        )

    # --------------------------------------------------------
    # Select strongest keypoints for visualization.
    # --------------------------------------------------------

    if len(keypoints) > max_display:

        order = np.argsort(
            [
                kp.response
                for kp in keypoints
            ]
        )[::-1]

        selected = [
            keypoints[i]
            for i in order[:max_display]
        ]

    else:

        selected = keypoints

    return cv2.drawKeypoints(
        image,
        selected,
        None,
        flags=(
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
    )


# ============================================================
# COMMAND LINE TEST
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="SIFT / RootSIFT feature extraction"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Input image"
    )

    parser.add_argument(
        "--mask",
        default=None,
        help="Optional mask image"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output directory"
    )

    parser.add_argument(
        "--nfeatures",
        type=int,
        default=12000
    )

    args = parser.parse_args()

    print("=" * 70)
    print("SIFT / ROOTSIFT FEATURE EXTRACTION")
    print("=" * 70)

    print(
        f"Image     : {args.image}"
    )

    print(
        f"Features  : {args.nfeatures}"
    )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = cv2.imread(
        args.image,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:

        raise FileNotFoundError(
            args.image
        )

    # --------------------------------------------------------
    # Load mask
    # --------------------------------------------------------

    mask = None

    if args.mask is not None:

        mask = cv2.imread(
            args.mask,
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:

            raise FileNotFoundError(
                args.mask
            )

    print()
    print(
        f"Image shape: {image.shape}"
    )

    # --------------------------------------------------------
    # Extract
    # --------------------------------------------------------

    features = extract_sift(
        image,
        mask=mask,
        nfeatures=args.nfeatures
    )

    print()
    print(
        "Keypoints:"
    )

    print(
        f"  {len(features['keypoints'])}"
    )

    print()
    print(
        "Descriptor shape:"
    )

    print(
        f"  {features['descriptors'].shape}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_dir = Path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    save_features(
        features,
        str(
            output_dir
            / "sift_features.npz"
        )
    )

    visualization = draw_features(
        image,
        features
    )

    cv2.imwrite(
        str(
            output_dir
            / "sift_keypoints.png"
        ),
        visualization
    )

    print()
    print(
        "Saved:"
    )

    print(
        output_dir
        / "sift_features.npz"
    )

    print(
        output_dir
        / "sift_keypoints.png"
    )

    print()
    print(
        "SIFT extraction complete."
    )