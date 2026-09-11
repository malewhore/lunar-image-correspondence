"""Helpers for real OHRC/LROC pair evaluation (later phases)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def list_pair_roots(pairs_dir: str | Path) -> list[Path]:
    root = Path(pairs_dir)
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()])


def expected_real_pair_ids() -> list[str]:
    """Placeholder benchmark IDs — replace with team's held-out list."""
    return ["P01", "P02", "P03", "P04", "P05"]
