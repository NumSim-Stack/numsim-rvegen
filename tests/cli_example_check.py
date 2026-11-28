#!/usr/bin/env python3
"""Run the rvegen CLI on one example config and check what it wrote.

Exit status alone says little: a writer can succeed and emit an empty,
truncated or unmeshable file. For every postprocess entry the config
declares, this script verifies the output file itself:

  .vox   header grid and extent match the config, data count == nx*ny*nz,
         id checks below
  .vtk   POINT_DATA layout (one point per voxel, DIMENSIONS == grid,
         SPACING*DIMENSIONS == domain) or CELL_DATA layout (one cell per
         voxel, DIMENSIONS == grid+1, SPACING*(DIMENSIONS-1) == domain);
         a flat 2D axis has DIMENSIONS 1 in both; ORIGIN 0, value count ==
         the data count, id checks below
  .geo   gmsh meshes it; the mesh has >= 2 top-dimension volumes/surfaces
         (matrix + inclusions), its elements tile the domain exactly (total
         area/volume == domain, so no gaps or overlapping fragments) and,
         for periodic configs, boundary nodes match on opposite faces
  .svg   valid XML <svg>; 2D: viewBox == domain; >= N shape elements
  .html  contains a script and >= N shapes
  .csv   (statistics) header volume fraction in (0, 1); the columns the
         header names in every row; a first row r = 0 that holds the volume
         fraction of the grid in every S2 column; r increasing; every S2
         value in [0, 1] and, in a periodic box, not above the value at
         r = 0; S2 not constant; g(r) >= 0 and not 0. A field may be empty
         (a bin without a lag; g(r) at r = 0), but g(r) is empty in every
         bin only when fewer than two inclusions leave it undefined

Voxel ids (.vox / .vtk) are read in one of three modes:
  shapes        ids in 0..N, both matrix and inclusion voxels, at least
                half of the N shape ids present
  phases        the config declares phases / phase_assignment, so ids are
                phase ids: in 0..P (P = number of phases) and more than one
                distinct value; the file's id-scheme tag must say so too
  tessellation  generator voronoi_2d / voronoi_3d: grains fill the domain,
                so no voxel is matrix (0); ids in 1..N and the number of
                distinct ids is within 10% of N (N <= n_seeds, also within
                10%)

Termination target: the achieved fill must reach what the config asks for.
  volume_fraction       the inclusion fraction, measured by re-running the
                        config (same seed) with an extra voxel post-process,
                        must lie in [max(t - 0.02, 0.9 t), 1.3 t + 0.01]
                        for target t (the voxel estimate and the last shape
                        placed account for the slack)
  number_of_inclusions  the CLI's shape count must equal the target; skipped
                        for the periodic generator, whose count includes
                        wrap-around copies

N is the shape count the CLI reports. The checks rely on the exit status
only (no "PASS" line matching), so any failure or crash fails the test.
Exit code 77 (CTest SKIP_RETURN_CODE) means every other check passed but a
.geo could not be meshed because gmsh is not available.

Usage: cli_example_check.py --rvegen BIN [--gmsh BIN] CONFIG
       cli_example_check.py --self-test
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


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def expected_grid(pp):
    """(nx, ny, nz) the postprocess entry asks for, or None if it does not say."""
    if all(k in pp for k in ("nx", "ny", "nz")):
        return [int(pp["nx"]), int(pp["ny"]), int(pp["nz"])]
    return None


TESSELLATION_GENERATORS = ("voronoi_2d", "voronoi_3d")


class IdMode:
    """How the voxel values of a config are to be read (see module doc)."""

    def __init__(self, config, nshapes):
        self.nshapes = nshapes
        gen = config.get("generator", {})
        self.tessellation = gen.get("type") in TESSELLATION_GENERATORS
        self.n_seeds = gen.get("n_seeds") if self.tessellation else None
        # A periodic tessellation also returns the copies of the grains
        # that cross a face; they carry the id of their grain, so the
        # number of shapes is larger than the number of grains.
        self.periodic = bool(self.tessellation and gen.get("periodic"))
        self.phases = None
        if "phases" in config or "phase_assignment" in config:
            phases = config.get("phases")
            if not isinstance(phases, list) or not phases:
                fail("config has phase_assignment but no non-empty 'phases' list")
            self.phases = [ph.get("name") for ph in phases]

    def check(self, name, vals):
        """Shared voxel-id checks for .vox and .vtk grids."""
        if self.phases is not None:
            return self._check_phases(name, vals)
        if self.tessellation:
            return self._check_tessellation(name, vals)
        return self._check_shapes(name, vals)

    def _check_phases(self, name, vals):
        nph = len(self.phases)
        if min(vals) < 0 or max(vals) > nph:
            fail(f"{name}: phase ids outside 0..{nph} "
                 f"(found {min(vals)}..{max(vals)})")
        distinct = set(vals)
        if len(distinct) < 2:
            fail(f"{name}: every voxel has phase id {min(vals)}, "
                 "the phase map is uniform")
        if self.tessellation and 0 in distinct:
            fail(f"{name}: tessellation left untagged (0) voxels")
        return (f"phase ids {sorted(distinct)} of 0..{nph}", len(distinct))

    def _check_tessellation(self, name, vals):
        if self.periodic:
            return self._check_periodic_tessellation(name, vals)
        n = self.nshapes
        if self.n_seeds is not None:
            seeds = int(self.n_seeds)
            if n > seeds or n < 0.9 * seeds:
                fail(f"{name}: {n} grains for n_seeds = {seeds}")
        if min(vals) < 1 or max(vals) > n:
            fail(f"{name}: grain ids outside 1..{n} "
                 f"(found {min(vals)}..{max(vals)}); a tessellation has no matrix")
        present = len(set(vals))
        # A grain smaller than a voxel may be missed by every sample, but
        # most must show up. Mapping all grains onto a few ids fails here.
        if present < 0.9 * n:
            fail(f"{name}: only {present} of {n} grain ids appear in the grid")
        return ("tessellation, no matrix voxel", present)

    def _check_periodic_tessellation(self, name, vals):
        n = self.nshapes
        if min(vals) < 1 or max(vals) > n:
            fail(f"{name}: grain ids outside 1..{n} "
                 f"(found {min(vals)}..{max(vals)}); a tessellation has no matrix")
        present = len(set(vals))
        # Every shape is a grain or a copy of one, and a grain has at most
        # 2^d - 1 copies; fewer distinct ids than that means grains merged.
        if present > n or present < n / 8:
            fail(f"{name}: {present} grain ids for {n} shapes of a periodic "
                 "tessellation")
        if self.n_seeds is not None:
            seeds = int(self.n_seeds)
            if present > seeds or present < 0.9 * seeds:
                fail(f"{name}: {present} grains for n_seeds = {seeds}")
        return ("periodic tessellation, no matrix voxel", present)

    def _check_shapes(self, name, vals):
        inclusion, present = check_ids(name, vals, self.nshapes)
        return (f"inclusion fraction {inclusion:.3f}", present)


def check_id_scheme(name, tag, mode):
    """A file's id-scheme tag (if it has one) must match the config's mode."""
    if tag is None:
        return
    says_phases = "phase" in tag.lower()
    if says_phases != (mode.phases is not None):
        fail(f"{name}: id scheme {tag!r}, but the config "
             f"{'declares' if mode.phases is not None else 'does not declare'} phases")


def check_ids(name, vals, nshapes):
    """Voxel-id checks for shape-index grids (matrix + inclusions)."""
    if min(vals) < 0 or max(vals) > nshapes:
        fail(f"{name}: ids outside 0..{nshapes}")
    inclusion = sum(1 for v in vals if v > 0) / len(vals)
    if inclusion <= 0.0:
        fail(f"{name}: no voxel belongs to an inclusion")
    if inclusion >= 1.0:
        fail(f"{name}: no matrix voxel left, every voxel is an inclusion")
    # Every example resolves its shapes at several voxels each, so most
    # shape ids must appear. A writer that maps all shapes onto one id, or
    # drops the second half of the list, fails here.
    present = len(set(vals) - {0})
    if present < 0.5 * nshapes:
        fail(f"{name}: only {present} of {nshapes} shape ids appear in the grid")
    return inclusion, present


def check_vox(path, mode, pp, domain):
    nx = ny = nz = None
    extent = None
    scheme = None
    vals = []
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            m = re.match(r"# id scheme:\s*(.*)", line)
            if m:
                scheme = m.group(1)
            m = re.match(r"# nx ny nz:\s+(\d+) (\d+) (\d+)", line)
            if m:
                nx, ny, nz = map(int, m.groups())
            m = re.match(r"# Lx Ly Lz:\s+(\S+) (\S+) (\S+)", line)
            if m:
                extent = list(map(float, m.groups()))
        elif line.strip():
            vals.append(int(line))
    if nx is None:
        fail(f"{path.name}: missing '# nx ny nz:' header")
    want = expected_grid(pp)
    if want and [nx, ny, nz] != want:
        fail(f"{path.name}: grid {nx}x{ny}x{nz}, config asks for {want}")
    if extent is None:
        fail(f"{path.name}: missing '# Lx Ly Lz:' header")
    if not all(close(e, d) for e, d in zip(extent, domain)):
        fail(f"{path.name}: extent {extent} != domain_box {domain}")
    if len(vals) != nx * ny * nz:
        fail(f"{path.name}: {len(vals)} voxels, header says {nx * ny * nz}")
    check_id_scheme(path.name, scheme, mode)
    what, present = mode.check(path.name, vals)
    return f"{len(vals)} voxels, {what}, {present} ids"


def check_vtk(path, mode, pp, domain):
    lines = path.read_text().splitlines()
    if not lines or not lines[0].startswith("# vtk DataFile"):
        fail(f"{path.name}: not a legacy VTK file")
    dims = origin = spacing = None
    layout = count = field = None
    start = None
    for i, line in enumerate(lines):
        if line.startswith("DIMENSIONS"):
            dims = list(map(int, line.split()[1:4]))
        elif line.startswith("ORIGIN"):
            origin = list(map(float, line.split()[1:4]))
        elif line.startswith("SPACING"):
            spacing = list(map(float, line.split()[1:4]))
        elif line.startswith(("POINT_DATA", "CELL_DATA")):
            if layout is not None:
                fail(f"{path.name}: both POINT_DATA and CELL_DATA sections")
            layout, count = line.split()[0], int(line.split()[1])
        elif line.startswith("SCALARS"):
            field = line.split()[1]
        elif line.startswith("LOOKUP_TABLE"):
            start = i + 1
            break
    if dims is None or layout is None or start is None:
        fail(f"{path.name}: missing DIMENSIONS / POINT_DATA or CELL_DATA / LOOKUP_TABLE")
    if len(dims) != 3 or min(dims) < 1:
        fail(f"{path.name}: DIMENSIONS {dims}")
    cell = layout == "CELL_DATA"
    # Voxels per axis. POINT_DATA puts one point per voxel; CELL_DATA one
    # cell per voxel, so DIMENSIONS (point counts) are one more. A flat
    # (2D) axis is a single point in both layouts (VTK pixel cells).
    for axis in range(3):
        if domain[axis] <= 0 and dims[axis] != 1:
            fail(f"{path.name}: DIMENSIONS {dims}, flat axis {axis} must be 1")
        if cell and domain[axis] > 0 and dims[axis] < 2:
            fail(f"{path.name}: CELL_DATA needs >= 2 points on axis {axis}, "
                 f"DIMENSIONS {dims}")
    grid = [d - 1 if cell and d > 1 else d for d in dims]
    if count != grid[0] * grid[1] * grid[2]:
        fail(f"{path.name}: {layout} {count} != {grid[0] * grid[1] * grid[2]} "
             f"for DIMENSIONS {dims}")
    want = expected_grid(pp)
    if want and grid != want:
        fail(f"{path.name}: {layout} grid {grid} (DIMENSIONS {dims}), "
             f"config asks for {want}")
    if origin is None or spacing is None:
        fail(f"{path.name}: missing ORIGIN / SPACING")
    if not all(close(o, 0.0) for o in origin):
        fail(f"{path.name}: ORIGIN {origin}, expected 0 0 0")
    for axis in range(3):
        # A flat (2D) axis has a single cell; its spacing is a placeholder.
        if domain[axis] > 0 and not close(spacing[axis] * grid[axis], domain[axis]):
            fail(f"{path.name}: extent on axis {axis} is SPACING {spacing[axis]} "
                 f"* {grid[axis]} voxels = {spacing[axis] * grid[axis]}, "
                 f"domain is {domain[axis]}")
    if field in ("phase_id", "shape_index"):
        check_id_scheme(path.name, field, mode)
    vals = [int(x) for x in " ".join(lines[start:]).split()]
    if len(vals) != count:
        fail(f"{path.name}: {len(vals)} values, {layout} says {count}")
    what, present = mode.check(path.name, vals)
    return f"{layout} {'x'.join(map(str, grid))}, {what}, {present} ids"


# gmsh MSH 2.2 element type -> (dimension, number of corner nodes).
# Higher-order types are measured on their corner nodes, which tile the
# domain exactly when the outer boundary is straight (it is a box).
MSH_TYPES = {2: (2, 3), 3: (2, 4), 9: (2, 3), 10: (2, 4), 16: (2, 4),
             4: (3, 4), 11: (3, 4)}
MSH_3D_UNSUPPORTED = {5, 6, 7, 12, 13, 14, 17, 18, 19}


def read_msh22(path):
    nodes, elements = {}, []
    lines = iter(path.read_text().splitlines())
    for line in lines:
        if line == "$Nodes":
            for _ in range(int(next(lines))):
                f = next(lines).split()
                nodes[int(f[0])] = (float(f[1]), float(f[2]), float(f[3]))
        elif line == "$Elements":
            for _ in range(int(next(lines))):
                f = list(map(int, next(lines).split()))
                etype, ntags = f[1], f[2]
                tags = f[3:3 + ntags]
                elementary = tags[1] if ntags >= 2 else 0
                elements.append((etype, elementary, f[3 + ntags:]))
    return nodes, elements


def measure(etype, pts):
    if MSH_TYPES[etype][0] == 2:
        # Shoelace on the corner polygon (triangle or quad).
        a = 0.0
        for i in range(len(pts)):
            x0, y0, _ = pts[i]
            x1, y1, _ = pts[(i + 1) % len(pts)]
            a += x0 * y1 - x1 * y0
        return abs(a) / 2.0
    (ax, ay, az), (bx, by, bz), (cx, cy, cz), (dx, dy, dz) = pts
    ux, uy, uz = bx - ax, by - ay, bz - az
    vx, vy, vz = cx - ax, cy - ay, cz - az
    wx, wy, wz = dx - ax, dy - ay, dz - az
    det = ux * (vy * wz - vz * wy) - uy * (vx * wz - vz * wx) + uz * (vx * wy - vy * wx)
    return abs(det) / 6.0


def check_periodic_nodes(name, nodes, domain, dim):
    """Mesh nodes on each low face must have a partner on the high face."""
    tol = 1e-6 * max(domain)
    for axis in range(dim):
        lo, hi = set(), set()
        for p in nodes.values():
            key = tuple(round(p[k] / tol) for k in range(dim) if k != axis)
            if abs(p[axis]) <= tol:
                lo.add(key)
            elif abs(p[axis] - domain[axis]) <= tol:
                hi.add(key)
        if not lo or lo != hi:
            fail(f"{name}: periodic mesh, {len(lo)} nodes on face {axis}=0 vs "
                 f"{len(hi)} on face {axis}=L, {len(lo ^ hi)} unmatched")


def check_geo(path, gmsh, three_d, workdir, pp, domain):
    text = path.read_text()
    if "Point(" not in text and "Circle(" not in text and "Sphere(" not in text \
            and "Disk(" not in text and "Box(" not in text:
        fail(f"{path.name}: no geometry entities")
    if not gmsh:
        return None
    dim = 3 if three_d else 2
    msh = workdir / (path.stem + ".msh")
    cmd = [gmsh, str(path), f"-{dim}", "-nopopup", "-format", "msh22", "-o", str(msh)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    out = r.stdout + r.stderr
    if r.returncode != 0 or "Error" in out:
        fail(f"{path.name}: gmsh failed (rc={r.returncode}):\n{out[-2000:]}")
    if not msh.exists() or msh.stat().st_size == 0:
        fail(f"{path.name}: gmsh wrote no mesh")
    nodes, elements = read_msh22(msh)
    total, entities, count = 0.0, set(), 0
    for etype, elementary, conn in elements:
        if etype in MSH_3D_UNSUPPORTED:
            fail(f"{path.name}: element type {etype} not handled by this check")
        if etype not in MSH_TYPES or MSH_TYPES[etype][0] != dim:
            continue
        corners = conn[:MSH_TYPES[etype][1]]
        total += measure(etype, [nodes[n] for n in corners])
        entities.add(elementary)
        count += 1
    if count == 0:
        fail(f"{path.name}: mesh has no {dim}D elements")
    if len(entities) < 2:
        fail(f"{path.name}: mesh has {len(entities)} {dim}D region(s), "
             "expected the matrix plus at least one inclusion")
    want = domain[0] * domain[1] * (domain[2] if dim == 3 else 1.0)
    if not close(total, want):
        fail(f"{path.name}: elements cover {total!r}, domain is {want!r} "
             "(gaps or overlapping fragments)")
    if pp.get("periodic"):
        check_periodic_nodes(path.name, nodes, domain, dim)
    return (f"meshed by gmsh: {count} elements in {len(entities)} regions, "
            f"measure {total:.9g}" + (", periodic nodes match" if pp.get("periodic") else ""))


SVG_SHAPES = ("circle", "ellipse", "rect", "polygon", "path")


def check_svg(path, nshapes, three_d, domain):
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        fail(f"{path.name}: not valid XML: {e}")
    if not root.tag.endswith("svg"):
        fail(f"{path.name}: root element is {root.tag}, not svg")
    if not three_d:
        vb = list(map(float, root.get("viewBox", "").split() or [0, 0, 0, 0]))
        if len(vb) != 4 or not all(close(a, b) for a, b in zip(vb, [0, 0, domain[0], domain[1]])):
            fail(f"{path.name}: viewBox {root.get('viewBox')!r} != domain {domain[:2]}")
    n = sum(1 for e in root.iter() if e.tag.rsplit("}", 1)[-1] in SVG_SHAPES)
    # The 2D writer draws the domain frame as one <rect>.
    drawn = n - (0 if three_d else 1)
    if drawn < nshapes:
        fail(f"{path.name}: {drawn} shape elements for {nshapes} shapes")
    return f"{n} shape elements"


def check_html(path, nshapes):
    t = path.read_text()
    if "<script" not in t:
        fail(f"{path.name}: three.js page has no script")
    n = t.count("kind:")
    if n < nshapes:
        fail(f"{path.name}: {n} shapes in the page for {nshapes} generated")
    return f"{n} shapes"


def check_csv(path):
    text = path.read_text()
    m = re.search(r"^# volume_fraction = (\S+)", text, re.M)
    if not m:
        fail(f"{path.name}: missing '# volume_fraction' header")
    phi = float(m.group(1))
    if not 0.0 < phi < 1.0:
        fail(f"{path.name}: volume fraction {phi} outside (0, 1)")
    rows = [l for l in text.splitlines() if l and not l.startswith("#")]
    if len(rows) < 2:
        fail(f"{path.name}: fewer than 2 data rows")
    m = re.search(r"^# columns: (.+)$", text, re.M)
    if not m:
        fail(f"{path.name}: missing '# columns' header")
    columns = [c.strip() for c in m.group(1).split(",")]
    if columns[:3] != ["r", "S2(r)", "g(r)"] or len(columns) < 5:
        fail(f"{path.name}: columns must start r, S2(r), g(r) and go on "
             f"with S2 along the axes, got {columns}")
    if any(not c.startswith("S2") for c in columns[3:]):
        fail(f"{path.name}: a column after g(r) is not an S2 column")
    periodic = re.search(r"^# S2 = .*periodic box", text, re.M) is not None
    fields = [r.split(",") for r in rows]
    if any(len(f) != len(columns) for f in fields):
        fail(f"{path.name}: a row does not have the {len(columns)} columns "
             "of the header")

    def number(field):
        return float(field) if field.strip() else None

    if any(not f[0].strip() for f in fields):
        fail(f"{path.name}: empty r field")
    r = [float(f[0]) for f in fields]
    if r[0] != 0.0:
        fail(f"{path.name}: the first row is not r = 0")
    if any(b <= a for a, b in zip(r, r[1:])):
        fail(f"{path.name}: r is not strictly increasing")
    # r = 0: both points are the same one, so every S2 column of a set holds
    # its volume fraction on the grid, and a pair of phases holds 0.
    zero = [number(f) for f in fields[0]]
    if zero[1] is None or not 0.0 < zero[1] < 1.0:
        fail(f"{path.name}: S2(0) = {zero[1]} is not a volume fraction")
    m = re.search(r"^# voxel_volume_fraction = (\S+)", text, re.M)
    if not m or abs(float(m.group(1)) - zero[1]) > 1e-9:
        fail(f"{path.name}: '# voxel_volume_fraction' is not S2(0)")
    if zero[2] is not None:
        fail(f"{path.name}: g(r) has a value at r = 0")
    for c in [1] + list(range(3, len(columns))):
        column = [number(f[c]) for f in fields]
        if column[0] is None:
            fail(f"{path.name}: {columns[c]} is empty at r = 0")
        values = [v for v in column if v is not None]
        # A probability; in a periodic box every lag has the same number of
        # pairs, and none can beat lag 0.
        top = column[0] + 1e-9 if periodic and "|" not in columns[c] else 1.0
        if min(values) < 0.0 or max(values) > top:
            fail(f"{path.name}: {columns[c]} outside [0, {top:.6g}]")
    s2 = [number(f[1]) for f in fields]
    s2 = [v for v in s2 if v is not None]
    if len(s2) < 3:
        fail(f"{path.name}: S2(r) has fewer than 3 values")
    if max(s2) - min(s2) <= 1e-9:
        fail(f"{path.name}: S2(r) is constant, statistics are trivial")
    # g(r) is empty when the pack has fewer than two inclusions.
    g_all = [number(f[2]) for f in fields[1:]]
    g = [v for v in g_all if v is not None]
    if g and len(g) != len(g_all):
        fail(f"{path.name}: g(r) is empty in {len(g_all) - len(g)} of {len(g_all)} bins")
    if g and (min(g) < 0.0 or max(g) <= 0.0):
        fail(f"{path.name}: g(r) negative or identically zero")
    return (f"{len(rows) - 1} bins, {len(columns)} columns, phi {phi:.3f}"
            + ("" if g else ", g(r) empty"))


def fill_window(target):
    """Accepted range of the achieved inclusion fraction for a target."""
    return max(target - 0.02, 0.9 * target), 1.3 * target + 0.01


def measured_fraction(rvegen, config, config_dir, work, domain):
    """Inclusion fraction of the config's RVE, from a voxel grid.

    Runs the config again (the seed fixes the shapes) with its post-processes
    replaced by one voxel writer, so the number does not depend on which
    outputs the example happens to write.
    """
    three_d = domain[2] > 0
    n = 96 if three_d else 512
    derived = dict(config)
    # With phases declared the grid would hold phase ids, the matrix phase
    # included; without them it holds 0 for matrix and a shape id otherwise.
    derived.pop("phases", None)
    derived.pop("phase_assignment", None)
    derived["postprocess"] = [{"type": "voxel", "nx": n, "ny": n, "nz": n if three_d else 1,
                               "output_path": "fill_check.vox"}]
    path = work / "fill_check.json"
    path.write_text(json.dumps(derived))
    # The derived config lives in the work directory; its inputs (meshes)
    # are still the ones next to the original config.
    r = subprocess.run([rvegen, str(path), "--input-dir", str(config_dir)], cwd=work,
                       capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        fail(f"fill check run exited {r.returncode}\n{r.stdout}{r.stderr}")
    vals = [int(l) for l in (work / "fill_check.vox").read_text().splitlines()
            if l.strip() and not l.startswith("#")]
    if not vals:
        fail("fill check: empty voxel grid")
    return sum(1 for v in vals if v > 0) / len(vals)


def check_target(rvegen, config, config_dir, nshapes, work, domain):
    """The RVE must reach the termination target the config asks for."""
    term = config.get("termination", {})
    kind = term.get("type")
    if kind == "volume_fraction":
        target = float(term["target_fraction"])
        got = measured_fraction(rvegen, config, config_dir, work, domain)
        lo, hi = fill_window(target)
        if not lo <= got <= hi:
            fail(f"volume fraction {got:.4f} not within [{lo:.4f}, {hi:.4f}] "
                 f"of the target {target} (RVE under- or over-filled)")
        return f"volume fraction {got:.4f} for target {target}"
    if kind == "number_of_inclusions" and config.get("generator", {}).get("type") != "periodic":
        target = int(term["target"])
        if nshapes != target:
            fail(f"{nshapes} shapes generated, target is {target}")
        return f"{nshapes} inclusions for target {target}"
    return None


def self_test():
    """Checks of the checker: the parts that need no CLI run."""
    with tempfile.TemporaryDirectory(prefix="rvegen_cli_selftest_") as tmp:
        csv = Path(tmp) / "s.csv"
        head = ("# volume_fraction = 0.1\n"
                "# S2 = pair counts of the 8 x 8 voxel grid, exact (FFT); "
                "periodic box, lags wrap around\n"
                "# voxel_volume_fraction = 0.11\n"
                "# columns: r, S2(r), g(r), S2_x(r), S2_y(r)\n")

        def expect(text, ok, what):
            csv.write_text(head + text)
            try:
                check_csv(csv)
                good = True
            except SystemExit:
                good = False
            if good != ok:
                print(f"self-test FAILED: {what}")
                sys.exit(1)

        z = "0,0.11,,0.11,0.11\n"
        expect(z + "0.1,0.10,1.0,0.1,0.1\n0.2,0.05,2.0,,0.04\n", True, "full rows")
        expect(z + "0.1,0.10,,0.1,0.1\n0.2,0.05,,0.05,0.05\n", True,
               "empty g(r) in every bin")
        expect(z + "0.1,0.10,,0.1,0.1\n0.2,0.05,2.0,0.05,0.05\n", False,
               "g(r) empty in only some bins")
        expect(z + "0.1,0.10,0,0.1,0.1\n0.2,0.05,0,0.05,0.05\n", False,
               "g(r) identically zero")
        expect("0.1,0.10,1.0,0.1,0.1\n0.2,0.05,2.0,0.05,0.05\n", False,
               "no row for r = 0")
        expect(z + "0.1,0.10,1.0,0.1\n0.2,0.05,2.0,0.05\n", False,
               "a column short")
        expect(z + "0.1,0.12,1.0,0.1,0.1\n0.2,0.05,2.0,0.05,0.05\n", False,
               "S2 above S2(0) in a periodic box")
        expect("0,0.12,,0.12,0.12\n0.1,0.10,1.0,0.1,0.1\n0.2,0.05,2,0.05,0.05\n",
               False, "S2(0) is not the voxel volume fraction of the header")
    lo, hi = fill_window(0.35)
    if lo > 0.35 or hi < 0.35 or 0.223 >= lo or not lo <= 0.3546 <= hi:
        print("self-test FAILED: fill window")
        sys.exit(1)
    lo, hi = fill_window(0.05)
    if 0.0261 >= lo or not lo <= 0.0517 <= hi:
        print("self-test FAILED: fill window at 0.05")
        sys.exit(1)
    print("self-test passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rvegen")
    ap.add_argument("--gmsh", default="")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("config", nargs="?")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    if not (a.rvegen and a.config):
        ap.error("--rvegen and CONFIG are required")

    config_path = Path(a.config).resolve()
    config = json.loads(config_path.read_text())
    gmsh = a.gmsh if a.gmsh and a.gmsh.upper() not in ("NOTFOUND", "NONE") else None
    if gmsh and not shutil.which(gmsh):
        gmsh = None

    with tempfile.TemporaryDirectory(prefix="rvegen_cli_") as tmp:
        work = Path(tmp)
        # Run from an empty directory: outputs land there, inputs such as
        # data/octahedron.stl are found next to the config.
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

        mode = IdMode(config, nshapes)
        domain = [float(v) for v in config.get("domain_box", [1, 1, 0])] + [0.0]
        domain = domain[:3]
        if msg := check_target(str(Path(a.rvegen).resolve()), config, config_path.parent,
                              nshapes, work, domain):
            print(f"  target: {msg}")
        # A shipped example runs clean: it reaches its target (the measured
        # check above is independent of rvegen's own report) and prints no
        # warning of any kind.
        if "warning" in r.stderr:
            fail(f"rvegen printed a warning:\n{r.stderr.strip()}")
        domain = [float(v) for v in config.get("domain_box", [1, 1, 0])] + [0.0]
        domain = domain[:3]
        three_d = domain[2] > 0
        unmeshed = []
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
                msg = check_vox(p, mode, pp, domain)
            elif ext == ".vtk":
                msg = check_vtk(p, mode, pp, domain)
            elif ext == ".geo":
                msg = check_geo(p, gmsh, three_d, work, pp, domain)
                if msg is None:
                    unmeshed.append(out)
                    msg = "geometry present, NOT meshed (gmsh not available)"
            elif ext == ".svg":
                msg = check_svg(p, nshapes, three_d, domain)
            elif ext == ".html":
                msg = check_html(p, nshapes)
            elif ext == ".csv":
                msg = check_csv(p)
            else:
                fail(f"{out}: no checker for extension {ext}")
            print(f"  {out}: {msg}")
    if unmeshed:
        print(f"SKIP: gmsh not available, not meshed: {', '.join(unmeshed)}")
        sys.exit(77)
    print("all checks passed")


if __name__ == "__main__":
    main()
