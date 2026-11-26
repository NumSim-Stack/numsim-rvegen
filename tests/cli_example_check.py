#!/usr/bin/env python3
"""Run the rvegen CLI on one example config and check what it wrote.

Exit status alone says little: a writer can succeed and emit an empty,
truncated or unmeshable file. For every postprocess entry the config
declares, this script verifies the output file itself:

  .vox   header present, data count == nx*ny*nz, ids in range, not all matrix
  .vtk   DIMENSIONS/POINT_DATA consistent, value count matches, not all matrix
  .geo   gmsh (when available) parses and meshes it
  .svg   parses as XML and has an <svg> root with drawing elements
  .html  non-empty, contains a script and at least one shape
  .csv   (statistics) rows present, S2(r) / g(r) non-trivial

Usage: cli_example_check.py --rvegen BIN --examples DIR [--gmsh BIN] CONFIG
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


def fail(msg):
    print(f"FAIL: {msg}")
    sys.exit(1)


def check_vox(path, nshapes):
    nx = ny = nz = None
    vals = []
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            m = re.match(r"# nx ny nz:\s+(\d+) (\d+) (\d+)", line)
            if m:
                nx, ny, nz = map(int, m.groups())
        elif line.strip():
            vals.append(int(line))
    if nx is None:
        fail(f"{path.name}: missing '# nx ny nz:' header")
    if len(vals) != nx * ny * nz:
        fail(f"{path.name}: {len(vals)} voxels, header says {nx * ny * nz}")
    if min(vals) < 0 or max(vals) > nshapes:
        fail(f"{path.name}: voxel ids outside 0..{nshapes}")
    inclusion = sum(1 for v in vals if v > 0) / len(vals)
    if inclusion <= 0.0:
        fail(f"{path.name}: no voxel belongs to an inclusion")
    return f"{len(vals)} voxels, inclusion fraction {inclusion:.3f}"


def check_vtk(path, nshapes):
    lines = path.read_text().splitlines()
    if not lines or not lines[0].startswith("# vtk DataFile"):
        fail(f"{path.name}: not a legacy VTK file")
    dims = npts = None
    start = None
    for i, line in enumerate(lines):
        if line.startswith("DIMENSIONS"):
            dims = list(map(int, line.split()[1:4]))
        elif line.startswith("POINT_DATA"):
            npts = int(line.split()[1])
        elif line.startswith("LOOKUP_TABLE"):
            start = i + 1
            break
    if dims is None or npts is None or start is None:
        fail(f"{path.name}: missing DIMENSIONS / POINT_DATA / LOOKUP_TABLE")
    if npts != dims[0] * dims[1] * dims[2]:
        fail(f"{path.name}: POINT_DATA {npts} != DIMENSIONS {dims}")
    vals = [int(x) for x in " ".join(lines[start:]).split()]
    if len(vals) != npts:
        fail(f"{path.name}: {len(vals)} values, POINT_DATA says {npts}")
    if min(vals) < 0 or max(vals) > nshapes:
        fail(f"{path.name}: ids outside 0..{nshapes}")
    if not any(v > 0 for v in vals):
        fail(f"{path.name}: no voxel belongs to an inclusion")
    return f"{npts} points"


def check_geo(path, gmsh, three_d, workdir):
    text = path.read_text()
    if "Point(" not in text and "Circle(" not in text and "Sphere(" not in text \
            and "Disk(" not in text and "Box(" not in text:
        fail(f"{path.name}: no geometry entities")
    if not gmsh:
        return "geometry present (gmsh not available, not meshed)"
    msh = workdir / (path.stem + ".msh")
    cmd = [gmsh, str(path), "-3" if three_d else "-2", "-nopopup", "-o", str(msh)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    out = r.stdout + r.stderr
    if r.returncode != 0 or "Error" in out:
        fail(f"{path.name}: gmsh failed (rc={r.returncode}):\n{out[-2000:]}")
    if not msh.exists() or msh.stat().st_size == 0:
        fail(f"{path.name}: gmsh wrote no mesh")
    return "meshed by gmsh"


def check_svg(path):
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        fail(f"{path.name}: not valid XML: {e}")
    if not root.tag.endswith("svg"):
        fail(f"{path.name}: root element is {root.tag}, not svg")
    n = sum(1 for _ in root.iter())
    if n < 2:
        fail(f"{path.name}: svg has no drawing elements")
    return f"{n} elements"


def check_html(path):
    t = path.read_text()
    if "<script" not in t or "kind:" not in t:
        fail(f"{path.name}: three.js page has no script / shapes")
    return f"{t.count('kind:')} shapes"


def check_csv(path):
    rows = [l for l in path.read_text().splitlines() if l and not l.startswith("#")]
    if len(rows) < 2:
        fail(f"{path.name}: fewer than 2 data rows")
    cols = [[float(x) for x in r.split(",")] for r in rows]
    s2 = [c[1] for c in cols]
    g = [c[2] for c in cols]
    if max(s2) - min(s2) <= 1e-9:
        fail(f"{path.name}: S2(r) is constant, statistics are trivial")
    if max(g) <= 0.0:
        fail(f"{path.name}: g(r) is identically zero")
    return f"{len(rows)} bins"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rvegen", required=True)
    ap.add_argument("--examples", required=True)
    ap.add_argument("--gmsh", default="")
    ap.add_argument("config")
    a = ap.parse_args()

    config_path = Path(a.config).resolve()
    config = json.loads(config_path.read_text())
    gmsh = a.gmsh if a.gmsh and a.gmsh.upper() not in ("NOTFOUND", "NONE") else None
    if gmsh and not shutil.which(gmsh):
        gmsh = None

    with tempfile.TemporaryDirectory(prefix="rvegen_cli_") as tmp:
        work = Path(tmp)
        # Configs reference assets such as data/octahedron.stl relative to cwd.
        data = Path(a.examples) / "data"
        if data.is_dir():
            (work / "data").symlink_to(data.resolve())
        r = subprocess.run([str(Path(a.rvegen).resolve()), str(config_path)], cwd=work,
                           capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            fail(f"rvegen exited {r.returncode}\n{r.stdout}{r.stderr}")
        m = re.search(r"generated (\d+) shapes?", r.stdout)
        if not m:
            fail(f"no 'generated N shapes' line in output:\n{r.stdout}")
        nshapes = int(m.group(1))
        if nshapes <= 0:
            fail("generated 0 shapes")
        print(f"{config_path.name}: {nshapes} shapes")

        three_d = len(config.get("domain_box", [1, 1, 0])) >= 3 and \
            config["domain_box"][2] > 0
        pps = config.get("postprocess", [])
        if not pps:
            fail("config has no postprocess entries, nothing to check")
        for pp in pps:
            out = pp.get("output_path")
            if not out:
                fail(f"postprocess '{pp['type']}' has no output_path")
            p = work / out
            if not p.is_file() or p.stat().st_size == 0:
                fail(f"{pp['type']}: output {out} missing or empty")
            ext = p.suffix.lower()
            if ext == ".vox":
                msg = check_vox(p, nshapes)
            elif ext == ".vtk":
                msg = check_vtk(p, nshapes)
            elif ext == ".geo":
                msg = check_geo(p, gmsh, three_d, work)
            elif ext == ".svg":
                msg = check_svg(p)
            elif ext == ".html":
                msg = check_html(p)
            elif ext == ".csv":
                msg = check_csv(p)
            else:
                fail(f"{out}: no checker for extension {ext}")
            print(f"  {out}: {msg}")
    print("PASS")


if __name__ == "__main__":
    main()
