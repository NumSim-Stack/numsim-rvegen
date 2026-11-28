#!/usr/bin/env python3
"""Regression checks for scripts/make_gallery.py.

  1. A renderer that is present but fails must make the script exit
     non-zero (it used to be reported as "skipped" with exit 0, so a
     gallery smoke test could never fail on render breakage).
  2. Two configs with the same file stem in different directories must
     get distinct work directories and PNG names (they used to share
     both and overwrite each other, racing under -j).

  3. A renderer that exits 0 with garbage must still fail the entry: a
     gmsh log with an `Error` line, a mesh without matrix elements and a
     blank three.js screenshot (what a page without network gives).
  4. `--check` must notice drift between the configs, the index and the
     PNGs, without rendering.

Usage: gallery_script_check.py --rvegen BIN --examples DIR
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "make_gallery.py"


def load_script():
    spec = importlib.util.spec_from_file_location("make_gallery", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def run(args: list[str]) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(SCRIPT), *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    print("$", " ".join(cmd), f"-> {proc.returncode}")
    print(proc.stdout + proc.stderr)
    return proc


def write_exe(path: Path, text: str) -> str:
    path.write_text(text)
    path.chmod(0o755)
    return str(path)


def check_garbage(a, ex: Path, tmp: Path, errors: list[str]) -> None:
    from PIL import Image

    gal = load_script()

    # Unit level: blank PNG, noisy PNG, gmsh log, meshes.
    blank = tmp / "blank.png"
    Image.new("RGB", (200, 150), (255, 255, 255)).save(blank)
    if not gal.png_blank_reason(blank):
        errors.append("an all-white PNG is not reported as blank")
    noisy = tmp / "noisy.png"
    Image.frombytes("RGB", (100, 100), os.urandom(30000)).save(noisy)
    if gal.png_blank_reason(noisy):
        errors.append("a PNG with thousands of colours is reported as blank")
    if not gal.gmsh_log_errors("Info    : Meshing 1D...\nError   : Unknown Entity 3\n"):
        errors.append("a gmsh log with an Error line is not detected")
    if gal.gmsh_log_errors("Info    : Done\nWarning : Mesh.Format is not msh2\n"):
        errors.append("a gmsh log without Error lines is reported as failing")
    tet = (3, 1, 1, [1, 2, 3, 4])
    both = {(3, 1): "matrix", (3, 2): "inclusion"}
    if gal.mesh_problem([tet, (3, 2, 2, [1, 2, 3, 4])], both, True):
        errors.append("a complete mesh is reported as incomplete")
    if not gal.mesh_problem([(3, 2, 2, [1, 2, 3, 4])], both, True):
        errors.append("a mesh whose matrix has no elements is accepted")
    if not gal.mesh_problem([(2, 1, 1, [1, 2, 3])], both, True):
        errors.append("a mesh without elements of the domain dimension is accepted")
    if not gal.mesh_problem([(3, 2, 2, [1, 2, 3, 4])], {(3, 2): "inclusion"}, True):
        errors.append("a mesh without a matrix group is accepted")

    # End to end: renderers that exit 0 but write garbage.
    liar = write_exe(tmp / "gmsh_liar", "#!/bin/sh\necho 'Error   : Unknown Entity 3'\nexit 0\n")
    p = run(["--rvegen", a.rvegen, "--out", str(tmp / "liar"), "--browser", "none",
             "--gmsh", liar, "-j", "1", str(ex / "periodic_circles.json")])
    if p.returncode == 0 or "FAILED periodic_circles.geo" not in p.stdout:
        errors.append("gmsh that exits 0 after an Error line is accepted")

    shot = write_exe(tmp / "browser_blank", f"""#!{sys.executable}
import sys
from PIL import Image
for arg in sys.argv:
    if arg.startswith("--screenshot="):
        Image.new("RGB", (64, 48), (240, 240, 240)).save(arg.split("=", 1)[1])
""")
    p = run(["--rvegen", a.rvegen, "--out", str(tmp / "blankshot"), "--browser", shot,
             "--gmsh", "none", "-j", "1", str(ex / "mixed_3d.json")])
    if p.returncode == 0 or "blank" not in p.stdout:
        errors.append("a blank three.js screenshot is accepted")
    if (tmp / "blankshot" / "mixed_3d-three_js.png").exists():
        errors.append("a blank three.js screenshot is left in the output directory")


def check_index_drift(a, ex: Path, tmp: Path, errors: list[str]) -> None:
    gallery = SCRIPT.parent.parent / "docs" / "gallery"
    p = run(["--check", "--out", str(gallery), str(ex)])
    if p.returncode != 0:
        errors.append("--check fails on the committed gallery")

    copy = tmp / "gallery_copy"
    shutil.copytree(gallery, copy)
    readme = copy / "README.md"
    text = readme.read_text()
    drifted = {
        "caption edited": text.replace("> ", "> edited: ", 1),
        "entry removed": text.replace("## spheres_in_cube", "## spheres_in_cube_old", 1),
        "image renamed": text.replace('src="periodic_circles-vtk_legacy.png"',
                                      'src="periodic_circles-vtk.png"', 1),
    }
    for what, content in drifted.items():
        if content == text:
            errors.append(f"index drift test '{what}' did not change the README")
            continue
        readme.write_text(content)
        if run(["--check", "--out", str(copy), str(ex)]).returncode == 0:
            errors.append(f"--check accepts a README with {what}")
    readme.write_text(text)
    (copy / "stray-svg.png").write_bytes(b"x")
    if run(["--check", "--out", str(copy), str(ex)]).returncode == 0:
        errors.append("--check accepts a PNG that no config generates")
    (copy / "stray-svg.png").unlink()
    (copy / "periodic_circles-vtk_legacy.png").unlink()
    if run(["--check", "--out", str(copy), str(ex)]).returncode == 0:
        errors.append("--check accepts a README image missing on disk")


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

        check_garbage(a, ex, tmp, errors)
        check_index_drift(a, ex, tmp, errors)

    for e in errors:
        print("FAIL:", e)
    if errors:
        return 1
    print("gallery script checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
