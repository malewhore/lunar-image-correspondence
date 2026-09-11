"""Compare pretrained vs fine-tuned experiment records (later phases)."""

from __future__ import annotations

from typing import Any


def delta_percent(baseline: float, improved: float) -> float:
    if baseline == 0:
        return 0.0
    return ((baseline - improved) / abs(baseline)) * 100.0


def compare_records(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Compute deltas for core metrics. Positive RMSE delta = candidate improved."""
    return {
        "pair_id": baseline.get("pair_id") or candidate.get("pair_id"),
        "delta_inlier_ratio": float(
            candidate.get("inlier_ratio", 0) - baseline.get("inlier_ratio", 0)
        ),
        "delta_rmse": float(baseline.get("rmse", 0) - candidate.get("rmse", 0)),
        "delta_rmse_pct": delta_percent(
            float(baseline.get("rmse", 0)), float(candidate.get("rmse", 0))
        ),
        "delta_median_error": float(
            baseline.get("median_error", 0) - candidate.get("median_error", 0)
        ),
        "delta_p95_error": float(
            baseline.get("p95_error", 0) - candidate.get("p95_error", 0)
        ),
        "delta_coverage": float(
            candidate.get("spatial_coverage", 0) - baseline.get("spatial_coverage", 0)
        ),
        "delta_runtime": float(
            candidate.get("runtime", 0) - baseline.get("runtime", 0)
        ),
    }
