#!/usr/bin/env python3
"""Regression checks for scripts/make_gallery.py.

  1. A renderer that is present but fails must make the script exit
     non-zero (it used to be reported as "skipped" with exit 0, so a
     gallery smoke test could never fail on render breakage).
  2. Two configs with the same file stem in different directories must
     get distinct work directories and PNG names (they used to share
     both and overwrite each other, racing under -j).

Usage: gallery_script_check.py --rvegen BIN --examples DIR
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "make_gallery.py"


def run(args: list[str]) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(SCRIPT), *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    print("$", " ".join(cmd), f"-> {proc.returncode}")
    print(proc.stdout + proc.stderr)
    return proc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rvegen", required=True)
    ap.add_argument("--examples", required=True)
    a = ap.parse_args()
    ex = Path(a.examples)
    errors: list[str] = []

    with tempfile.TemporaryDirectory(prefix="gallery-check-") as tmp:
        tmp = Path(tmp)

        # 1. `false` stands in for a gmsh that is installed but broken.
        false = shutil.which("false")
        if false is None:
            errors.append("no `false` binary to stand in for a failing gmsh")
        else:
            p = run(["--rvegen", a.rvegen, "--out", str(tmp / "fail"), "--browser", "none",
                     "--gmsh", false, "-j", "1", str(ex / "periodic_circles.json")])
            if p.returncode == 0:
                errors.append("render failure (gmsh exits 1) still gives exit 0")
            if "FAILED periodic_circles.geo" not in p.stdout:
                errors.append("render failure not reported as FAILED")
            if not (tmp / "fail" / "periodic_circles-vtk_legacy.png").is_file():
                errors.append("other outputs of a config with a failing render were not rendered")

        # 2. a/c.json and b/c.json: same stem, different configs.
        for sub, src in (("a", "periodic_circles.json"), ("b", "circles_in_unit_box.json")):
            (tmp / sub).mkdir()
            shutil.copy(ex / src, tmp / sub / "c.json")
        same = [str(tmp / "a" / "c.json"), str(tmp / "b" / "c.json")]
        out = tmp / "same"
        p = run(["--rvegen", a.rvegen, "--out", str(out), "--browser", "none",
                 "--gmsh", "none", "-j", "2", *same])
        if p.returncode != 0:
            errors.append(f"same-stem run exited {p.returncode}")
        pngs = sorted(f.name for f in out.glob("*.png"))
        want = ["a_c-vtk_legacy.png", "b_c-voxel.png"]
        if pngs != want:
            errors.append(f"same-stem PNGs {pngs}, expected {want}")

        # 3. --index lists both; a later subset run without --index must
        #    leave that index alone instead of replacing it.
        out = tmp / "index"
        p = run(["--rvegen", a.rvegen, "--out", str(out), "--browser", "none",
                 "--gmsh", "none", "--index", *same])
        readme = out / "README.md"
        index = readme.read_text() if readme.is_file() else ""
        if p.returncode != 0 or "## a_c" not in index or "## b_c" not in index:
            errors.append("--index run does not list both same-stem configs separately")
        out.mkdir(parents=True, exist_ok=True)
        readme.write_text("sentinel")
        run(["--rvegen", a.rvegen, "--out", str(out), "--browser", "none",
             "--gmsh", "none", same[1]])
        if readme.read_text() != "sentinel":
            errors.append("subset run overwrote the existing index")

    for e in errors:
        print("FAIL:", e)
    if errors:
        return 1
    print("gallery script checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
