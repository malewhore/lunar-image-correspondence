"""PHASE 3 — pretrained SuperPoint + LightGlue on ordinary images.

    python scripts/run_ordinary_baseline.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.matchers import LightGlueBaselineMatcher
from backend.schemas import load_pair_from_paths


ORDINARY_URLS = {
    "ordinary_01.jpg": "https://raw.githubusercontent.com/cvg/LightGlue/main/assets/sacre_coeur1.jpg",
    "ordinary_02.jpg": "https://raw.githubusercontent.com/cvg/LightGlue/main/assets/sacre_coeur2.jpg",
}


def ensure_ordinary_images(raw_dir: Path) -> tuple[Path, Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, url in ORDINARY_URLS.items():
        path = raw_dir / name
        if not path.exists():
            print(f"Downloading {name}...")
            urllib.request.urlretrieve(url, path)
        paths.append(path)
    return paths[0], paths[1]


def draw_matches(pair, result, out_path: Path) -> None:
    img0 = pair.moving_image
    img1 = pair.reference_image
    if img0.ndim == 2:
        img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
    if img1.ndim == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

    if result.num_matches == 0:
        canvas = np.hstack([img0, img1])
        cv2.imwrite(str(out_path), canvas)
        return

    idx0 = result.matches[:, 0]
    idx1 = result.matches[:, 1]
    pts0 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in result.keypoints_a[idx0]]
    pts1 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in result.keypoints_b[idx1]]
    dmatches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0) for i in range(len(pts0))]
    vis = cv2.drawMatches(img0, pts0, img1, pts1, dmatches, None, flags=2)
    cv2.imwrite(str(out_path), vis)


def main() -> int:
    raw_dir = ROOT / "data" / "raw"
    out_dir = ROOT / "outputs" / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    img_a, img_b = ensure_ordinary_images(raw_dir)
    pair = load_pair_from_paths(str(img_a), str(img_b), pair_id="ordinary_demo")

    print("Loading pretrained SuperPoint + LightGlue...")
    matcher = LightGlueBaselineMatcher()
    print(f"Device: {matcher.device}")
    result = matcher.match(pair)

    viz_path = out_dir / "matches.png"
    draw_matches(pair, result, viz_path)

    metrics = {
        "pair_id": pair.pair_id,
        "method": result.method,
        "num_keypoints_a": result.num_keypoints_a,
        "num_keypoints_b": result.num_keypoints_b,
        "total_matches": result.num_matches,
        "runtime_seconds": result.runtime_seconds,
        "status": "success" if result.num_matches > 0 else "error",
        "visualization": str(viz_path),
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    matches_payload = {
        "matches": result.matches.tolist() if result.num_matches else [],
        "confidence": result.confidence.tolist() if result.confidence is not None else [],
        "keypoints_a": result.keypoints_a.tolist(),
        "keypoints_b": result.keypoints_b.tolist(),
    }
    (out_dir / "matches.json").write_text(
        json.dumps(matches_payload, indent=2), encoding="utf-8"
    )

    print(json.dumps(metrics, indent=2))
    if result.num_matches > 0:
        print(f"\nCHECKPOINT PASS — ordinary baseline OK. See {out_dir}")
        return 0
    print("\nCHECKPOINT FAIL — zero matches on ordinary images.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
