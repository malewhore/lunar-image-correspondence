"""
AlgorithmResult — matcher module → orchestrator / API / frontend contract.

Internal LightGlue details may change; this outbound schema stays stable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class AlgorithmResult:
    method: str
    status: str  # "success" | "unreliable" | "error"

    total_matches: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0

    rmse: float = 0.0
    median_error: float = 0.0
    p95_error: float = 0.0

    spatial_coverage: float = 0.0

    transformation_type: Optional[str] = None
    transformation_matrix: Optional[list] = None

    runtime_ms: float = 0.0

    matches: Optional[list] = None
    inlier_matches: Optional[list] = None

    artifacts: dict[str, Any] = field(default_factory=dict)
    failure_reason: Optional[str] = None

    pair_id: Optional[str] = None
    num_keypoints_moving: int = 0
    num_keypoints_reference: int = 0

    def to_dict(self, include_matches: bool = False) -> dict[str, Any]:
        data = asdict(self)
        if not include_matches:
            data.pop("matches", None)
            data.pop("inlier_matches", None)
        return _json_safe(data)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj
