"""
ProcessedPair — CV/preprocessing → matcher contract (provisional).

Freeze this input contract before matching / training work.
If the team later publishes a shared schema, adapt fields here —
do not invent a second conflicting pair type inside matchers/.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class ImageMetadata:
    """Sensor / product metadata for one side of a pair."""

    role: str  # "moving" | "reference"
    sensor: Optional[str] = None  # e.g. "OHRC", "LROC_NAC"
    mission: Optional[str] = None  # e.g. "Chandrayaan-2", "LRO"
    path: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    dtype: Optional[str] = None
    channels: int = 1
    grayscale: bool = True
    normalized: bool = False
    resized: bool = False
    native_resolution_m_per_px: Optional[float] = None
    coordinate_system: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScaleInfo:
    """Relative scale / resolution relationship between moving and reference."""

    moving_m_per_px: Optional[float] = None
    reference_m_per_px: Optional[float] = None
    scale_ratio_moving_to_reference: Optional[float] = None
    notes: Optional[str] = None


@dataclass
class GeometricInfo:
    """Any known geometric / overlap hints from preprocessing."""

    overlap_hint: Optional[str] = None
    already_registered: bool = False
    transform_hint: Optional[np.ndarray] = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessedPair:
    """
    Verified input unit for matching.

    Images are expected as 2D grayscale uint8 (or float32 in [0, 1]) arrays.
    Coordinates are in the pixel space of these arrays.
    """

    pair_id: str
    moving_image: np.ndarray
    reference_image: np.ndarray
    moving_metadata: ImageMetadata
    reference_metadata: ImageMetadata
    scale_info: ScaleInfo = field(default_factory=ScaleInfo)
    preprocessing_config: dict[str, Any] = field(default_factory=dict)
    geometric_info: GeometricInfo = field(default_factory=GeometricInfo)
    variant: Optional[str] = None  # e.g. "02_clahe"

    def summarize(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "variant": self.variant,
            "moving_shape": tuple(self.moving_image.shape),
            "reference_shape": tuple(self.reference_image.shape),
            "moving_dtype": str(self.moving_image.dtype),
            "reference_dtype": str(self.reference_image.dtype),
            "moving_grayscale": self.moving_image.ndim == 2,
            "reference_grayscale": self.reference_image.ndim == 2,
            "moving_minmax": (
                float(np.min(self.moving_image)),
                float(np.max(self.moving_image)),
            ),
            "reference_minmax": (
                float(np.min(self.reference_image)),
                float(np.max(self.reference_image)),
            ),
            "preprocessing_config": self.preprocessing_config,
            "scale_info": {
                "moving_m_per_px": self.scale_info.moving_m_per_px,
                "reference_m_per_px": self.scale_info.reference_m_per_px,
                "scale_ratio_moving_to_reference": (
                    self.scale_info.scale_ratio_moving_to_reference
                ),
            },
            "already_registered": self.geometric_info.already_registered,
            "moving_path": self.moving_metadata.path,
            "reference_path": self.reference_metadata.path,
        }
