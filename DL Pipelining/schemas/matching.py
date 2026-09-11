"""Intermediate matching payload (before RANSAC / AlgorithmResult)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class MatchingResult:
    method: str

    keypoints_a: np.ndarray
    keypoints_b: np.ndarray

    matches: np.ndarray
    confidence: Optional[np.ndarray]

    num_keypoints_a: int
    num_keypoints_b: int
    num_matches: int

    runtime_seconds: float
