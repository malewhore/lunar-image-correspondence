"""PHASE 2 — Environment verification. Run from repo root:

    python scripts/verify_environment.py
"""

from __future__ import annotations

import importlib
import sys


def check(mod: str) -> tuple[bool, str]:
    try:
        m = importlib.import_module(mod)
        version = getattr(m, "__version__", "unknown")
        return True, str(version)
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def main() -> int:
    print("=== LUNARA environment check ===")
    print(f"Python: {sys.version}")

    ok_all = True
    for name in ("torch", "cv2", "numpy", "matplotlib", "scipy"):
        ok, info = check(name)
        mark = "OK" if ok else "FAIL"
        print(f"  [{mark}] {name}: {info}")
        ok_all = ok_all and ok

    try:
        import torch

        print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] torch CUDA probe: {exc}")
        ok_all = False

    lg_ok, lg_info = check("lightglue")
    print(f"  [{'OK' if lg_ok else 'FAIL'}] lightglue: {lg_info}")
    ok_all = ok_all and lg_ok

    if ok_all:
        print("\nCHECKPOINT PASS — environment ready.")
        return 0

    print("\nCHECKPOINT FAIL — fix imports before LightGlue work.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
