#!/usr/bin/env python3
"""Emit the trtexec build list (onnx|engine|flags) from release_manifest.yaml.

Also verifies all 21 ONNX exist before engines are built. The flags column
carries the Myelin workspace pool size; override per host with
PANSPLAT_POOL_MB (default 12288 MB, documented in docs/04_kernels.md).
"""
from pathlib import Path
import os
import sys

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
MANIFEST = REPO / "scripts" / "release_manifest.yaml"

def main():
    lines = []
    missing = []
    for row in MANIFEST.read_text().splitlines():
        row = row.strip()
        if row.startswith("- key:"):
            key = row.split("key:", 1)[1].strip()
        elif row.startswith("engine:"):
            engine = row.split("engine:", 1)[1].strip()
        elif row.startswith("onnx:"):
            onnx = row.split("onnx:", 1)[1].strip()
            p = REPO / onnx
            if not p.exists():
                missing.append(onnx)
            pool = os.environ.get("PANSPLAT_POOL_MB", "12288")
            lines.append(f"{Path(onnx).name}|{engine}|--memPoolSize=workspace:{pool}")
    if missing:
        print("missing ONNX: " + ", ".join(missing), file=sys.stderr)
        sys.exit(1)
    print("\n".join(lines))

if __name__ == "__main__":
    main()
