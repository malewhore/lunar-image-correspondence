"""Load a ProcessedPair from image paths or a SIH pair package layout."""

from __future__ import annotations

import os
from typing import Optional

import cv2

from backend.schemas.processed_pair import (
    GeometricInfo,
    ImageMetadata,
    ProcessedPair,
    ScaleInfo,
)

# Known variants inside SIH_DL_Pair001/SIH_DL_Pair001/
VARIANT_FILES = {
    "01_normalized": {
        "moving": "moving_ohrc_normalized.png",
        "reference": "reference_lroc.png",
        "config": {"percentile_norm": True, "clahe": False},
    },
    "02_clahe": {
        "moving": "moving_ohrc_clahe.png",
        "reference": "reference_lroc_clahe.png",
        "config": {"percentile_norm": True, "clahe": True},
    },
    "03_structure_gradient": {
        "moving": "moving_ohrc_structural.png",
        "reference": "reference_lroc_gradient.png",
        "config": {"structure_gradient": True},
    },
}


def load_pair_from_paths(
    moving_path: str,
    reference_path: str,
    pair_id: str = "custom",
) -> ProcessedPair:
    """Load a ProcessedPair from two image paths (ordinary or lunar)."""
    moving = cv2.imread(moving_path, cv2.IMREAD_GRAYSCALE)
    reference = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
    if moving is None:
        raise FileNotFoundError(f"Failed to load moving image: {moving_path}")
    if reference is None:
        raise FileNotFoundError(f"Failed to load reference image: {reference_path}")

    return ProcessedPair(
        pair_id=pair_id,
        moving_image=moving,
        reference_image=reference,
        moving_metadata=ImageMetadata(
            role="moving",
            path=os.path.abspath(moving_path),
            width=int(moving.shape[1]),
            height=int(moving.shape[0]),
            dtype=str(moving.dtype),
        ),
        reference_metadata=ImageMetadata(
            role="reference",
            path=os.path.abspath(reference_path),
            width=int(reference.shape[1]),
            height=int(reference.shape[0]),
            dtype=str(reference.dtype),
        ),
        preprocessing_config={"source": "raw_paths"},
        geometric_info=GeometricInfo(already_registered=False),
        variant=None,
    )


def load_processed_pair(
    pair_root: str,
    variant: str = "02_clahe",
    pair_id: Optional[str] = None,
) -> ProcessedPair:
    """
    Load a ProcessedPair from a SIH pair package.

    Expected layout:
      SIH_DL_Pair001/
        SIH_DL_Pair001/
          02_clahe/
            moving_ohrc_clahe.png
            reference_lroc_clahe.png
    """
    if variant not in VARIANT_FILES:
        raise ValueError(
            f"Unknown variant '{variant}'. Choose from: {sorted(VARIANT_FILES)}"
        )

    basename = os.path.basename(pair_root.rstrip("\\/"))
    inner = os.path.join(pair_root, basename)
    if not os.path.isdir(inner):
        inner = pair_root

    variant_dir = os.path.join(inner, variant)
    if not os.path.isdir(variant_dir):
        raise FileNotFoundError(f"Variant folder not found: {variant_dir}")

    files = VARIANT_FILES[variant]
    moving_path = os.path.join(variant_dir, files["moving"])
    reference_path = os.path.join(variant_dir, files["reference"])

    pair = load_pair_from_paths(
        moving_path,
        reference_path,
        pair_id=pair_id or basename,
    )
    pair.variant = variant
    pair.preprocessing_config = dict(files["config"])
    pair.moving_metadata.sensor = "OHRC"
    pair.moving_metadata.mission = "Chandrayaan-2"
    pair.moving_metadata.native_resolution_m_per_px = 0.28
    pair.moving_metadata.coordinate_system = "browse_image"
    pair.moving_metadata.normalized = True
    pair.reference_metadata.sensor = "LROC_NAC"
    pair.reference_metadata.mission = "LRO"
    pair.reference_metadata.coordinate_system = "geographic_overlap_v4"
    pair.scale_info = ScaleInfo(
        moving_m_per_px=0.28,
        notes=(
            "OHRC ~0.28 m/px; LROC overlap browse scale differs substantially."
        ),
    )
    pair.geometric_info = GeometricInfo(
        overlap_hint="LROC South Pole NAC browse mosaic V4 geographic-overlap",
        already_registered=False,
    )
    return pair
