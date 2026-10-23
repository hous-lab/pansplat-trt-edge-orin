#!/usr/bin/env python3
"""Derive the engine platform directory name for this host.

export/engines/<platform>/ is keyed by architecture + the TensorRT version of
the trtexec on PATH (e.g. x86-trt10.4), so engines built under one TRT pairing
are never silently reused under another. aarch64 = Orin branch (expects TRT
10.3, JetPack 6.2); x86 covers the A6000/4090 hosts.
"""
import platform
import re
import shutil
import subprocess
import sys


def trt_version():
    exe = shutil.which("trtexec")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--version"], capture_output=True,
                             text=True, timeout=60)
    except subprocess.TimeoutExpired:
        first = None
    else:
        first = out.stdout + out.stderr
    if not first:
        return None
    for line in first.splitlines():
        if "TensorRT" in line:
            for tok in line.split():
                m = re.match(r"v(\d+)(\d{2})(\d{2})$", tok.strip(",[]"))
                if m:
                    return (m.group(1), m.group(2).lstrip("0") or "0")
    return None


def main():
    machine = platform.machine()
    trt = trt_version()
    if trt is None:
        sys.exit("trtexec not found on PATH - cannot derive the platform "
                 "directory (source the TensorRT env; see docs/)")
    ver = f"{trt[0]}.{trt[1]}"
    prefix = "orin" if machine == "aarch64" else "x86"
    if machine == "aarch64" and ver != "10.3":
        # Orin engines in this release were built under JetPack 6.2 / TRT 10.3
        sys.exit(f"Orin branch expects JetPack 6.2 / TRT 10.3, found {ver}")
    plat = f"{prefix}-trt{ver}"
    print(f"TensorRT {ver} on {machine} -> export/engines/{plat}",
          file=sys.stderr)
    return plat


if __name__ == "__main__":
    print(main())
