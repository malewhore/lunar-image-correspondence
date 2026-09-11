"""PHASE 0 checkpoint — load and inspect a ProcessedPair.

Examples (from repo root):

    python scripts/inspect_processed_pair.py --moving data/raw/ordinary_01.jpg --reference data/raw/ordinary_02.jpg

    python scripts/inspect_processed_pair.py --pair-root path/to/SIH_DL_Pair001 --variant 02_clahe
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.schemas import load_pair_from_paths, load_processed_pair


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect ProcessedPair contract")
    parser.add_argument("--moving", type=str, help="Moving / source image path")
    parser.add_argument("--reference", type=str, help="Reference / fixed image path")
    parser.add_argument("--pair-root", type=str, help="SIH pair package root")
    parser.add_argument("--variant", type=str, default="02_clahe")
    parser.add_argument("--pair-id", type=str, default="inspect")
    args = parser.parse_args()

    if args.pair_root:
        pair = load_processed_pair(args.pair_root, variant=args.variant, pair_id=args.pair_id)
    elif args.moving and args.reference:
        pair = load_pair_from_paths(args.moving, args.reference, pair_id=args.pair_id)
    else:
        parser.error("Provide --pair-root OR both --moving and --reference")
        return 2

    summary = pair.summarize()
    print("=== ProcessedPair checkpoint ===")
    print(json.dumps(summary, indent=2))
    print("\nDirect field access:")
    print(f"  pair.moving_image.shape    = {pair.moving_image.shape}")
    print(f"  pair.reference_image.shape = {pair.reference_image.shape}")
    print(f"  pair.moving_image.dtype    = {pair.moving_image.dtype}")
    print(f"  pair.reference_image.dtype = {pair.reference_image.dtype}")
    print("\nCHECKPOINT PASS — ProcessedPair load/inspect OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
