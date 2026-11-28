#!/usr/bin/env python3
"""Render every rvegen example config into a picture gallery.

For each JSON config the script runs the `rvegen` CLI in a scratch
directory, then turns every file the config's post-processes wrote into
a PNG:

  .svg   (svg, svg_3d)        -> Inkscape, falling back to ImageMagick
  .html  (three_js)           -> headless Chrome/Chromium screenshot
  .vox   (voxel)              -> matplotlib voxel slice(s); composite voxels of a
                                 v4 file in the mixed colour of their ids
  .vtk   (vtk_legacy)         -> matplotlib voxel slice(s), v2 and v3 layouts
  .geo   (gmsh_geo)           -> gmsh meshes it, matplotlib draws the mesh
  .csv   (statistics)         -> matplotlib S2(r) and g(r) plots

and writes `<out>/README.md`, an index that GitHub renders as the gallery.

    scripts/make_gallery.py                      # all of examples/
    scripts/make_gallery.py examples/mixed_*.json --out /tmp/g --index
    cmake --build build --target gallery         # same, via CMake

Only the rvegen binary, Python 3, numpy and matplotlib are required.
Inkscape (or ImageMagick), Chrome and gmsh are optional: outputs that
need a missing tool are reported as skipped. The three.js pages load
three.js from unpkg, so screenshots of them need network access.

`--check` renders nothing: it compares the entries the script would
generate (names, PNG files, captions) with docs/gallery/README.md and the
PNGs on disk and exits non-zero on drift (the `gallery_index_check` test).

A renderer that exits 0 is still checked: a gmsh log with an `Error` line,
a mesh without matrix elements and a blank three.js screenshot (which is
what a page without network access produces) fail the entry.

Exit status is non-zero when rvegen fails on a config, when a config's
post-process does not write its output, or when a renderer that is
available fails on an output (a missing tool is a skip, not a failure).

A config may carry a top-level "_comment" string; it becomes the
caption. The rvegen CLI resolves relative paths against its working
directory; the gallery runs each config as if from the config's own
directory (`cd examples && rvegen x.json`), so relative "*_path" inputs
(e.g. "mesh_path") are resolved against the config's directory. Run by
hand, the CLI needs the same: examples/mesh_inclusions_3d.json reads
"data/octahedron.stl" and only works from inside examples/.
"output_path" values are reduced to their basename so every output
lands in the scratch directory. Configs whose file names share a stem
(e.g. a/x.json and b/x.json) get distinct output names.

The README.md index is written when the whole default examples/
directory is rendered, or with --index; a run on a subset leaves an
existing index alone so it does not drop the other configs.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

MATRIX_RGB = (0.86, 0.86, 0.86)
PALETTE = [
    (0.122, 0.467, 0.706), (1.000, 0.498, 0.055), (0.173, 0.627, 0.173),
    (0.839, 0.153, 0.157), (0.580, 0.404, 0.741), (0.549, 0.337, 0.294),
    (0.890, 0.467, 0.761), (0.737, 0.741, 0.133), (0.090, 0.745, 0.812),
    (0.200, 0.300, 0.500),
]
# Keep PNG bytes stable across runs so regenerated images only show up in
# a diff when the picture actually changed.
PNG_META = {"Software": None}


def colour(index: int) -> tuple[float, float, float]:
    """Colour for inclusion/phase id `index` (> 0); 0 is the matrix."""
    return MATRIX_RGB if index <= 0 else PALETTE[(index - 1) % len(PALETTE)]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@dataclass
class Tools:
    rvegen: Path
    gmsh: str | None
    svg: list[str] | None  # command prefix; input and output appended
    svg_kind: str | None
    browser: str | None
    width: int
    timeout: int
    no_sandbox: bool = False


def run_tool(cmd: list[str], timeout: int, **kw) -> subprocess.CompletedProcess:
    """Run an external tool; on failure raise with the tail of its output."""
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kw)
    if proc.returncode != 0:
        tail = (proc.stderr.strip() or proc.stdout.strip())[-400:]
        raise RuntimeError(f"{Path(cmd[0]).name} exited {proc.returncode}: {tail}")
    return proc


def find_rvegen(explicit: str | None) -> Path:
    candidates = [explicit, os.environ.get("RVEGEN")]
    candidates += [str(REPO / d / "rvegen") for d in ("build", "build-release", "build-debug")]
    candidates.append(shutil.which("rvegen"))
    for c in candidates:
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            return Path(c).resolve()
    sys.exit("make_gallery: rvegen binary not found; pass --rvegen or set $RVEGEN")


def find_browser(explicit: str | None) -> str | None:
    if explicit == "none":
        return None
    if explicit:
        return explicit
    for name in ("google-chrome", "chromium", "chromium-browser", "chrome"):
        if path := shutil.which(name):
            return path
    return None


def gmsh_version(gmsh: str | None) -> str | None:
    if not gmsh:
        return None
    try:
        out = subprocess.run([gmsh, "--version"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    m = re.search(r"\d+\.\d+\.\d+", out.stdout + out.stderr)
    return m.group(0) if m else None


def find_svg_converter() -> tuple[str | None, list[str] | None]:
    if shutil.which("inkscape"):
        return "inkscape", ["inkscape", "--export-type=png"]
    if shutil.which("magick"):
        return "magick", ["magick", "-density", "150"]
    if shutil.which("convert"):
        return "convert", ["convert", "-density", "150"]
    return None, None


# ---------------------------------------------------------------------------
# Running a config
# ---------------------------------------------------------------------------
@dataclass
class Result:
    config: Path
    cfg: dict
    ok: bool = False
    message: str = ""
    shapes: int | None = None
    images: list[tuple[str, str]] = field(default_factory=list)  # (png name, label)
    skipped: list[str] = field(default_factory=list)  # a needed tool is missing
    failed: list[str] = field(default_factory=list)  # an output could not be rendered
    slug: str = ""


def assign_slugs(configs: list[Path]) -> list[tuple[Path, str]]:
    """Give every config a unique name for its work dir and PNGs.

    The stem is used as is unless another config shares it; then the
    parent directory names are prepended until the names differ, with a
    numeric suffix as the last resort.
    """
    by_stem: dict[str, list[Path]] = {}
    for c in configs:
        by_stem.setdefault(c.stem, []).append(c)
    used: set[str] = set()
    out = []
    for c in configs:
        if len(by_stem[c.stem]) == 1:
            slug = c.stem
        else:
            parts = [p for p in c.parent.parts if p not in ("/", "")]
            slug = f"{parts[-1]}_{c.stem}" if parts else c.stem
        base, n = slug, 2
        while slug in used:
            slug = f"{base}-{n}"
            n += 1
        used.add(slug)
        out.append((c, slug))
    return out


def unique_png(stem: str, kind: str, used: set[str]) -> str:
    """PNG name for post-process `kind` of config `stem`; records it in `used`."""
    name = f"{stem}-{kind}.png"
    n = 2
    while name in used:
        name = f"{stem}-{kind}-{n}.png"
        n += 1
    used.add(name)
    return name


def planned_pngs(cfg: dict, slug: str) -> list[str]:
    """The PNG names a successful run of `cfg` writes, without rendering."""
    used: set[str] = set()
    names = []
    for pp in cfg.get("postprocess", []):
        output = pp.get("output_path")
        if not output:
            continue
        kind = pp.get("type", Path(output).suffix.lstrip("."))
        if Path(output).suffix.lower() in (".svg", ".html", ".vox", ".vtk", ".geo", ".csv"):
            names.append(unique_png(slug, kind, used))
    return names


def process(cfg_path: Path, slug: str, out_dir: Path, work_root: Path, tools: Tools) -> Result:
    cfg = json.loads(cfg_path.read_text())
    res = Result(cfg_path, cfg, slug=slug)
    stem = slug
    work = work_root / stem
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    # The config is run where it is: rvegen finds its inputs next to it and
    # writes the (relative) outputs into the work directory.
    try:
        proc = subprocess.run([str(tools.rvegen), str(cfg_path.resolve()),
                               "--output-dir", str(work)],
                              capture_output=True, text=True, timeout=tools.timeout)
    except subprocess.TimeoutExpired:
        res.message = f"rvegen timed out after {tools.timeout}s"
        return res
    if proc.returncode != 0:
        res.message = f"rvegen exited {proc.returncode}: {proc.stderr.strip()[-400:]}"
        return res
    if m := re.search(r"generated (\d+) shapes", proc.stdout + proc.stderr):
        res.shapes = int(m.group(1))

    is_3d = len(cfg.get("domain_box", [])) >= 3 and float(cfg["domain_box"][2]) > 0
    used: set[str] = set()
    for pp in cfg.get("postprocess", []):
        output = pp.get("output_path")
        if not output:
            continue
        src = work / output
        kind = pp.get("type", src.suffix.lstrip("."))
        if not src.is_file():
            res.failed.append(f"{output}: not written by rvegen")
            continue
        name = unique_png(stem, kind, used)
        dst = out_dir / name
        try:
            label = render(src, dst, kind, is_3d, work, tools)
        except Exception as e:  # noqa: BLE001 - report, keep going, fail at the end
            res.failed.append(f"{output}: {type(e).__name__}: {e}")
            continue
        if label is None:
            continue
        if label.startswith("skip:"):
            res.skipped.append(f"{output}: {label[5:].strip()}")
        else:
            if kind not in SHADED_KINDS:
                shrink_png(dst)
            res.images.append((name, label))
    res.ok = True
    return res


# Outputs with smooth shading; a 256-colour palette would band them.
SHADED_KINDS = {"three_js", "svg_3d"}


def shrink_png(path: Path) -> None:
    """Re-encode a flat-colour image with a 256-colour palette (~3-4x smaller)."""
    from PIL import Image  # matplotlib already depends on Pillow

    with Image.open(path) as im:
        q = im.convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT,
                                       dither=Image.Dither.NONE)
    q.save(path, optimize=True)


def render(src: Path, dst: Path, kind: str, is_3d: bool, work: Path, tools: Tools) -> str | None:
    ext = src.suffix.lower()
    if ext == ".svg":
        return render_svg(src, dst, tools)
    if ext == ".html":
        return render_html(src, dst, tools)
    if ext == ".vox":
        grid, extent, names, shares = read_vox(src)
        if shares is None:
            plot_grid(grid, extent, names, dst, "voxel grid")
            return "voxel grid (.vox)"
        plot_grid(grid, extent, names, dst, "voxel grid, partial volumes", shares)
        return "voxel grid with partial volumes (.vox)"
    if ext == ".vtk":
        grid, extent = read_vtk(src)
        plot_grid(grid, extent, {}, dst, "VTK voxel grid")
        return "voxel grid (.vtk)"
    if ext == ".geo":
        return render_geo(src, dst, is_3d, work, tools)
    if ext == ".csv":
        plot_statistics(src, dst)
        return "statistics: S2(r) and g(r)"
    return f"skip: no renderer for {ext} ({kind})"


# ---------------------------------------------------------------------------
# SVG / HTML
# ---------------------------------------------------------------------------
def render_svg(src: Path, dst: Path, tools: Tools) -> str:
    if not tools.svg:
        return "skip: no SVG converter (install inkscape or imagemagick)"
    if tools.svg_kind == "inkscape":
        cmd = tools.svg + [f"--export-filename={dst}", f"--export-width={tools.width}",
                           "--export-background=white", str(src)]
    else:
        cmd = tools.svg + [str(src), "-background", "white", "-flatten",
                           "-resize", f"{tools.width}x", str(dst)]
    run_tool(cmd, tools.timeout)
    if not dst.is_file():
        raise RuntimeError(f"{tools.svg_kind} produced no PNG")
    return "SVG"


def render_html(src: Path, dst: Path, tools: Tools) -> str:
    if not tools.browser:
        return "skip: no headless Chrome/Chromium (or --browser none)"
    height = round(tools.width * 0.78)
    # A private profile keeps headless Chrome from touching the user's.
    with tempfile.TemporaryDirectory(prefix="rvegen-gallery-chrome-") as profile:
        cmd = [tools.browser, "--headless=new", "--hide-scrollbars", "--no-first-run",
               f"--user-data-dir={profile}",
               *(["--no-sandbox"] if tools.no_sandbox else []),
               "--enable-unsafe-swiftshader", "--use-angle=swiftshader",
               f"--window-size={tools.width},{height}",
               "--virtual-time-budget=15000",
               f"--screenshot={dst}", src.resolve().as_uri()]
        run_tool(cmd, tools.timeout)
    if not dst.is_file():
        raise RuntimeError("browser produced no screenshot")
    # Offline, the page still renders: an empty viewer with its info box.
    if blank := png_blank_reason(dst):
        dst.unlink()
        raise RuntimeError(f"browser screenshot is blank ({blank}); is unpkg.com reachable?")
    return "three.js viewer (screenshot)"


# A real three.js render has thousands of colours (shading) and the
# background covers well under 95 % of it; the empty viewer with only its
# info box has a few hundred colours and about 97 % background.
BLANK_MIN_COLOURS = 2000
BLANK_MAX_DOMINANT = 0.95


def png_blank_reason(path: Path) -> str | None:
    """Describe why an image is (nearly) one colour, or None if it is not."""
    from PIL import Image  # matplotlib already depends on Pillow

    with Image.open(path) as im:
        rgb = im.convert("RGB")
        colours = rgb.getcolors(rgb.size[0] * rgb.size[1]) or []
    total = sum(n for n, _ in colours)
    top = max((n for n, _ in colours), default=0)
    if len(colours) < BLANK_MIN_COLOURS:
        return f"{len(colours)} distinct colours"
    if total and top / total > BLANK_MAX_DOMINANT:
        return f"{100 * top / total:.1f} % one colour"
    return None


# ---------------------------------------------------------------------------
# Voxel grids
# ---------------------------------------------------------------------------
def read_vox(path: Path):
    """Grid (z, y, x), extent, phase names and, for a file with partial
    volumes (format v4), the shares of the composite voxels:
    {flat voxel index: [(id, share), ...]}; None for a v3 file."""
    n = extent = samples = None
    names: dict[int, str] = {}
    rows = []
    with path.open() as f:
        for line in f:
            if line.startswith("#"):
                if m := re.match(r"#\s*nx ny nz:\s*(\d+)\s+(\d+)\s+(\d+)", line):
                    n = tuple(int(g) for g in m.groups())
                elif m := re.match(r"#\s*Lx Ly Lz:\s*(\S+)\s+(\S+)\s+(\S+)", line):
                    extent = tuple(float(g) for g in m.groups())
                elif m := re.match(r"#\s*phase\s+(\d+)\s*=\s*(.+)", line):
                    names[int(m.group(1))] = m.group(2).strip()
                elif m := re.match(r"#\s*subsamples:\s*\d+\s*\((\d+) sample points", line):
                    samples = int(m.group(1))
            elif line.strip():
                rows.append(line)
    if n is None or extent is None:
        raise ValueError("missing nx/ny/nz or Lx/Ly/Lz header")
    count = n[0] * n[1] * n[2]
    grid = np.asarray([int(v) for v in rows[:count]], dtype=np.int64).reshape(n[2], n[1], n[0])
    if samples is None:
        if len(rows) != count:
            raise ValueError(f"{len(rows)} values for {count} voxels")
        return grid, extent, names, None
    shares = {}
    for line in rows[count:]:
        t = [int(v) for v in line.split()]
        pairs = list(zip(t[1::2], t[2::2]))
        if sum(c for _, c in pairs) != samples:
            raise ValueError(f"composite voxel {t[0]}: samples do not add up to {samples}")
        shares[t[0]] = [(i, c / samples) for i, c in pairs]
    return grid, extent, names, shares


def read_vtk(path: Path):
    tokens = path.read_text().split("\n")
    dims = spacing = layout = None
    count = 0
    i = 0
    while i < len(tokens):
        t = tokens[i].split()
        i += 1
        if not t:
            continue
        if t[0] == "DIMENSIONS":
            dims = [int(v) for v in t[1:4]]
        elif t[0] == "SPACING":
            spacing = [float(v) for v in t[1:4]]
        elif t[0] in ("POINT_DATA", "CELL_DATA"):
            layout, count = t[0], int(t[1])
        elif t[0] == "LOOKUP_TABLE":
            break
    if dims is None or spacing is None or layout is None:
        raise ValueError("not an rvegen STRUCTURED_POINTS file")
    values = np.asarray(" ".join(tokens[i:]).split()[:count], dtype=np.int64)
    # v2 wrote one sample per point; v3 writes one per cell, with n+1 points
    # per sampled axis (a 2D grid keeps a single z point).
    n = dims if layout == "POINT_DATA" else [max(d - 1, 1) for d in dims]
    grid = values.reshape(n[2], n[1], n[0])
    extent = tuple(spacing[a] * n[a] for a in range(3))
    if n[2] == 1:
        extent = (extent[0], extent[1], 0.0)
    return grid, extent


def project(grid, lut, axis: int):
    """First inclusion hit looking along -axis from the + side, depth-shaded.

    `grid` is indexed (z, y, x); the result keeps the other two axes in
    order, so it can be drawn with the same extents as a slice.
    """
    g = np.flip(grid, axis=axis)
    solid = g > 0
    first = solid.argmax(axis=axis)
    ids = np.take_along_axis(g, np.expand_dims(first, axis), axis).squeeze(axis)
    shade = 1.0 - 0.6 * first / g.shape[axis]
    rgb = lut[ids] * shade[..., None]
    rgb[~solid.any(axis=axis)] = (1.0, 1.0, 1.0)
    return rgb


def plot_grid(grid, extent, names: dict[int, str], dst: Path, title: str,
              shares: dict | None = None) -> None:
    top = int(grid.max())
    if shares:
        top = max(top, max(i for row in shares.values() for i, _ in row))
    lut = np.array([colour(k) for k in range(top + 1)])
    rgb = lut[grid]
    if shares:
        # A composite voxel takes the colours of its ids, mixed by share.
        flat = rgb.reshape(-1, 3)
        for voxel, row in shares.items():
            flat[voxel] = sum(share * lut[i] for i, share in row)
    nz, ny, nx = grid.shape
    Lx, Ly, Lz = extent
    if nz == 1:
        fig, ax = plt.subplots(figsize=(6, 6 * Ly / Lx if Lx else 6))
        panels = [(ax, rgb[0], (0, Lx, 0, Ly), "x", "y", f"{nx}×{ny}")]
    else:
        # Top row: mid-plane slices. Bottom row: what an observer outside
        # the box sees with the matrix made transparent, darker = deeper.
        fig, axes = plt.subplots(2, 3, figsize=(13, 8.8))
        panels = [
            (axes[0, 0], rgb[nz // 2], (0, Lx, 0, Ly), "x", "y", f"slice z = {Lz / 2:g}"),
            (axes[0, 1], rgb[:, ny // 2, :], (0, Lx, 0, Lz), "x", "z", f"slice y = {Ly / 2:g}"),
            (axes[0, 2], rgb[:, :, nx // 2], (0, Ly, 0, Lz), "y", "z", f"slice x = {Lx / 2:g}"),
            (axes[1, 0], project(grid, lut, 0), (0, Lx, 0, Ly), "x", "y", "seen from +z"),
            (axes[1, 1], project(grid, lut, 1), (0, Lx, 0, Lz), "x", "z", "seen from +y"),
            (axes[1, 2], project(grid, lut, 2), (0, Ly, 0, Lz), "y", "z", "seen from +x"),
        ]
    for ax, img, ext, xl, yl, sub in panels:
        ax.imshow(img, origin="lower", extent=ext, interpolation="nearest")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(sub, fontsize=10)
    shape = f"{nx}×{ny}" + (f"×{nz}" if nz > 1 else "")
    fig.suptitle(f"{title} ({shape} voxels)")
    if names:
        handles = [Patch(color=colour(k), label=v) for k, v in sorted(names.items())]
        fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 6), frameon=False)
    fig.tight_layout()
    fig.savefig(dst, dpi=110, metadata=PNG_META)
    plt.close(fig)


# ---------------------------------------------------------------------------
# gmsh
# ---------------------------------------------------------------------------
# MSH 2.2 element type -> (dimension, number of corner nodes)
MSH_TYPES = {2: (2, 3), 3: (2, 4), 9: (2, 3), 10: (2, 4), 16: (2, 4),
             4: (3, 4), 11: (3, 4), 29: (3, 4)}


def render_geo(src: Path, dst: Path, is_3d: bool, work: Path, tools: Tools) -> str:
    if not tools.gmsh:
        return "skip: gmsh not found"
    msh = work / (src.stem + ".msh")
    proc = subprocess.run([tools.gmsh, "-3" if is_3d else "-2", "-format", "msh22",
                           "-o", str(msh), str(src)],
                          capture_output=True, text=True, timeout=tools.timeout, cwd=work)
    # The pip `gmsh` launcher exits 0 after script errors, so the log is
    # checked as well as the exit status.
    errors = gmsh_log_errors(proc.stdout + proc.stderr)
    if proc.returncode != 0 or not msh.is_file() or errors:
        raise RuntimeError("gmsh failed: " + ("; ".join(errors[:3]) or f"exit {proc.returncode}"))
    nodes, elements, names = read_msh(msh)
    if problem := mesh_problem(elements, names, is_3d):
        raise RuntimeError("gmsh mesh is incomplete: " + problem)
    if is_3d:
        plot_mesh_3d(nodes, elements, names, dst)
    else:
        plot_mesh_2d(nodes, elements, names, dst)
    return "gmsh mesh of the .geo"


def gmsh_log_errors(log: str) -> list[str]:
    """Lines of a gmsh log that report an error ("Error   : ...")."""
    return [ln.strip() for ln in log.splitlines() if re.match(r"\s*Error\b", ln)]


def mesh_problem(elements, names, is_3d: bool) -> str | None:
    """Why a parsed mesh cannot be a complete RVE, or None if it looks fine.

    Needs elements of the domain's dimension, no named physical group
    without elements, and (when groups are named) one called "matrix".
    """
    dim = 3 if is_3d else 2
    mine = [e for e in elements if e[0] == dim]
    if not mine:
        return f"no {dim}D elements"
    used = {e[1] for e in mine}
    groups = {tag: name for (d, tag), name in names.items() if d == dim}
    empty = sorted(name for tag, name in groups.items() if tag not in used)
    if empty:
        return "physical group(s) without elements: " + ", ".join(empty)
    if groups and not any(n.lower() == "matrix" for n in groups.values()):
        return "no matrix physical group"
    return None


def read_msh(path: Path):
    lines = path.read_text().split("\n")
    nodes: dict[int, tuple[float, float, float]] = {}
    elements = []  # (dim, physical, elementary, corner node ids)
    names: dict[tuple[int, int], str] = {}
    i = 0
    while i < len(lines):
        head = lines[i].strip()
        if head == "$PhysicalNames":
            for ln in lines[i + 2:i + 2 + int(lines[i + 1])]:
                dim, tag, name = ln.split(maxsplit=2)
                names[(int(dim), int(tag))] = name.strip().strip('"')
        elif head == "$Nodes":
            for ln in lines[i + 2:i + 2 + int(lines[i + 1])]:
                t = ln.split()
                nodes[int(t[0])] = (float(t[1]), float(t[2]), float(t[3]))
        elif head == "$Elements":
            for ln in lines[i + 2:i + 2 + int(lines[i + 1])]:
                t = [int(v) for v in ln.split()]
                etype, ntags = t[1], t[2]
                if etype not in MSH_TYPES:
                    continue
                dim, corners = MSH_TYPES[etype]
                tags = t[3:3 + ntags]
                conn = t[3 + ntags:3 + ntags + corners]
                elements.append((dim, tags[0] if tags else 0, tags[1] if len(tags) > 1 else 0, conn))
        i += 1
    return nodes, elements, names


def group_colours(elements, names, dim, measure):
    """Map each element's group key to a colour; the matrix stays grey.

    With physical groups the key is the physical tag and a group named
    "matrix" is grey. Without them it is the elementary entity, and the
    entity with the largest measure is taken to be the matrix.
    """
    has_phys = any(e[1] > 0 for e in elements if e[0] == dim)
    key = (lambda e: e[1]) if has_phys else (lambda e: e[2])
    keys = sorted({key(e) for e in elements if e[0] == dim})
    labels: dict[int, str] = {}
    if has_phys:
        matrix = {k for k in keys if names.get((dim, k), "").lower() == "matrix"}
        for k in keys:
            labels[k] = names.get((dim, k), f"physical {k}")
    else:
        sizes: dict[int, float] = {}
        for e in elements:
            if e[0] == dim:
                sizes[key(e)] = sizes.get(key(e), 0.0) + measure(e)
        matrix = {max(sizes, key=sizes.get)} if sizes else set()
    colours, n = {}, 0
    for k in keys:
        if k in matrix:
            colours[k] = MATRIX_RGB
        else:
            n += 1
            colours[k] = colour(n)
    legend = labels if has_phys else {}
    return key, colours, legend


def plot_mesh_2d(nodes, elements, names, dst: Path) -> None:
    pts = lambda e: [nodes[n][:2] for n in e[3]]  # noqa: E731

    def area(e):
        p = np.asarray(pts(e))
        x, y = p[:, 0], p[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    faces = [e for e in elements if e[0] == 2]
    key, colours, legend = group_colours(elements, names, 2, area)
    xy = np.asarray([nodes[n][:2] for n in nodes])
    Lx, Ly = np.ptp(xy[:, 0]) or 1.0, np.ptp(xy[:, 1]) or 1.0
    fig, ax = plt.subplots(figsize=(6.5, 6.5 * Ly / Lx))
    ax.add_collection(PolyCollection([pts(e) for e in faces],
                                     facecolors=[colours[key(e)] for e in faces],
                                     edgecolors=(0.15, 0.15, 0.15), linewidths=0.12))
    ax.set_xlim(xy[:, 0].min(), xy[:, 0].max())
    ax.set_ylim(xy[:, 1].min(), xy[:, 1].max())
    ax.set_aspect("equal")
    ax.set_title(f"gmsh mesh: {len(faces)} elements, {len(nodes)} nodes")
    add_legend(fig, ax, legend, colours)
    fig.tight_layout()
    fig.savefig(dst, dpi=130, metadata=PNG_META)
    plt.close(fig)


def plot_mesh_3d(nodes, elements, names, dst: Path) -> None:
    tets = [e for e in elements if e[0] == 3]

    def volume(e):
        a, b, c, d = (np.asarray(nodes[n]) for n in e[3])
        return abs(np.dot(b - a, np.cross(c - a, d - a))) / 6.0

    key, colours, legend = group_colours(elements, names, 3, volume)
    xyz = np.asarray(list(nodes.values()))
    hi = xyz.max(axis=0)
    tol = 1e-7 * float(np.ptp(xyz, axis=0).max() or 1.0)
    polys, cols = [], []
    # The view below looks from +x, +y, +z, so the three max faces are visible.
    for e in tets:
        for face in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
            p = np.asarray([nodes[e[3][j]] for j in face])
            if any(np.all(np.abs(p[:, a] - hi[a]) < tol) for a in range(3)):
                polys.append(p)
                cols.append(colours[key(e)])
    fig = plt.figure(figsize=(7, 6.4))
    ax = fig.add_subplot(projection="3d")
    ax.add_collection3d(Poly3DCollection(polys, facecolors=cols,
                                         edgecolors=(0.15, 0.15, 0.15), linewidths=0.08))
    lo = xyz.min(axis=0)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(tuple(hi - lo))
    ax.view_init(elev=24, azim=40)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"gmsh mesh: {len(tets)} tetrahedra, {len(nodes)} nodes (outer faces)")
    add_legend(fig, ax, legend, colours)
    fig.tight_layout()
    fig.savefig(dst, dpi=120, metadata=PNG_META)
    plt.close(fig)


def add_legend(fig, ax, legend: dict[int, str], colours) -> None:
    if not legend:
        return
    handles = [Patch(color=colours[k], label=v) for k, v in sorted(legend.items(), key=lambda kv: kv[1])]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)


# ---------------------------------------------------------------------------
# Statistics CSV
# ---------------------------------------------------------------------------
def plot_statistics(src: Path, dst: Path) -> None:
    phi = None
    columns: list[str] = []
    rows = []
    for line in src.read_text().splitlines():
        if line.startswith("#"):
            if m := re.match(r"#\s*volume_fraction\s*=\s*(\S+)", line):
                phi = float(m.group(1))
            elif m := re.match(r"#\s*columns:\s*(.+)", line):
                columns = [c.strip() for c in m.group(1).split(",")]
        elif line.strip():
            rows.append([float(v) for v in line.split(",")])
    data = np.asarray(rows)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError("expected r, S2(r), g(r) columns")
    r, s2, g = data[:, 0], data[:, 1], data[:, 2]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(r, s2, "o-", ms=3, label=columns[1] if len(columns) > 1 else "S2(r)")
    if phi is not None:
        a1.axhline(phi, ls="--", c="grey", lw=1, label=f"φ = {phi:.3f}")
        a1.axhline(phi * phi, ls=":", c="grey", lw=1, label=f"φ² = {phi * phi:.3f}")
    a1.set_xlabel("r")
    a1.set_title("two-point correlation S2(r)")
    a1.legend(fontsize=8)
    a2.plot(r, g, "o-", ms=3, c=PALETTE[1])
    a2.axhline(1.0, ls="--", c="grey", lw=1)
    a2.set_xlabel("r")
    a2.set_title("radial distribution g(r)")
    fig.tight_layout()
    fig.savefig(dst, dpi=110, metadata=PNG_META)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Gallery index
# ---------------------------------------------------------------------------
def describe(cfg: dict, shapes: int | None) -> str:
    box = cfg.get("domain_box", [1, 1, 0])
    dim = "3D" if len(box) >= 3 and float(box[2]) > 0 else "2D"
    size = "×".join(f"{float(v):g}" for v in (box if dim == "3D" else box[:2]))
    inputs = sorted({s.get("type", "?") for s in cfg.get("shapes", [])})
    parts = [f"{dim} {size}", ", ".join(inputs),
             f"generator `{cfg.get('generator', {}).get('type', '?')}`"]
    if shapes is not None:
        parts.append(f"{shapes} shapes")
    return " · ".join(parts)


def write_index(results: list[Result], out_dir: Path, gmsh_version: str | None = None) -> None:
    lines = [
        "# rvegen example gallery",
        "",
        "<!-- Generated by scripts/make_gallery.py; do not edit by hand. -->",
        "",
        "Every picture below is rendered from a config in `examples/`. Regenerate with",
        "`scripts/make_gallery.py` or `cmake --build build --target gallery`.",
        "",
        "Each config runs from any directory: rvegen resolves relative input paths",
        "(`mesh_path`) against the config's directory.",
        "",
    ]
    if gmsh_version:
        lines += [f"The `gmsh_geo` pictures were meshed with gmsh {gmsh_version}; other gmsh",
                  "versions mesh differently, so the element counts in their titles change.", ""]
    for res in sorted(results, key=lambda r: r.slug):
        rel = os.path.relpath(res.config, out_dir)
        lines.append(f"## {res.slug}")
        lines.append("")
        lines.append(f"[`{res.config.name}`]({rel}) · {describe(res.cfg, res.shapes)}")
        lines.append("")
        if comment := res.cfg.get("_comment"):
            lines.append(f"> {comment}")
            lines.append("")
        if not res.ok:
            lines.append(f"**Not rendered:** {res.message}")
            lines.append("")
            continue
        for name, label in res.images:
            lines.append(f'<img src="{name}" alt="{res.slug}: {label}" width="440">')
        if res.images:
            lines.append("")
        for note in res.failed:
            lines.append(f"- **failed** {note}")
        for note in res.skipped:
            lines.append(f"- _skipped_ {note}")
        if res.skipped or res.failed:
            lines.append("")
    (out_dir / "README.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
def check_index(configs: list[Path], out_dir: Path) -> list[str]:
    """Compare the entries this script generates with the committed index.

    No rendering: the entry list (slugs, PNG names, captions, config
    summary lines) comes from the configs, the other side is
    `<out_dir>/README.md` and the PNGs in `out_dir`. Returns the problems.
    """
    problems: list[str] = []
    readme = out_dir / "README.md"
    if not readme.is_file():
        return [f"{readme} does not exist"]
    sections: dict[str, list[str]] = {}
    current = None
    for line in readme.read_text().splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            if current in sections:
                problems.append(f"README lists '{current}' twice")
            sections[current] = []
        elif current is not None:
            sections[current].append(line)

    entries = {slug: c for c, slug in assign_slugs(configs)}
    for slug in sorted(set(entries) - set(sections)):
        problems.append(f"{slug}: generated from {entries[slug].name} but missing in README")
    for slug in sorted(set(sections) - set(entries)):
        problems.append(f"{slug}: in README but no config generates it")

    expected_files: set[str] = set()
    for slug in sorted(set(entries) & set(sections)):
        path = entries[slug]
        cfg = json.loads(path.read_text())
        body = sections[slug]
        pngs = planned_pngs(cfg, slug)
        expected_files.update(pngs)
        listed = [m.group(1) for ln in body if (m := re.match(r'<img src="([^"]+)"', ln))]
        if not any(ln.startswith("**Not rendered:**") for ln in body):
            # A missing tool is recorded in the index as "_skipped_", so the
            # image list may legitimately be shorter, never longer.
            extra = sorted(set(listed) - set(pngs))
            if extra:
                problems.append(f"{slug}: README shows {', '.join(extra)}, not generated by the config")
            skipped = [ln for ln in body if ln.startswith("- _skipped_")]
            if not skipped:
                missing = sorted(set(pngs) - set(listed))
                if missing:
                    problems.append(f"{slug}: README does not show {', '.join(missing)}")
        for name in listed:
            if not (out_dir / name).is_file():
                problems.append(f"{slug}: README shows {name}, which is not on disk")
        caption = [ln[2:] for ln in body if ln.startswith("> ")]
        want = [cfg["_comment"]] if cfg.get("_comment") else []
        if caption != want:
            problems.append(f"{slug}: caption differs from the _comment of {path.name}")
        info = next((ln for ln in body if ln.startswith("[`")), "")
        head = f"[`{path.name}`]({os.path.relpath(path, out_dir)}) · {describe(cfg, None)}"
        if not (info == head or info.startswith(head + " · ")):
            problems.append(f"{slug}: config summary line differs from the config ({path.name})")
    for f in sorted(out_dir.glob("*.png")):
        if f.name not in expected_files:
            problems.append(f"{f.name}: on disk but no config generates it")
    return problems


# ---------------------------------------------------------------------------
def collect_configs(paths: list[str]) -> list[Path]:
    configs: list[Path] = []
    for p in map(Path, paths or [str(REPO / "examples")]):
        if p.is_dir():
            configs += sorted(p.glob("*.json"))
        elif p.is_file():
            configs.append(p)
        else:
            sys.exit(f"make_gallery: no such file or directory: {p}")
    # The same file named twice (e.g. a dir and a file in it) runs once.
    return list(dict.fromkeys(c.resolve() for c in configs))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("configs", nargs="*", help="config files or directories (default: examples/)")
    ap.add_argument("--out", default=str(REPO / "docs" / "gallery"), help="output directory")
    ap.add_argument("--rvegen", help="rvegen binary (default: $RVEGEN, build/rvegen, PATH)")
    ap.add_argument("--gmsh", default="gmsh", help="gmsh binary for .geo outputs, or 'none'")
    ap.add_argument("--browser", help="Chrome/Chromium for .html outputs, or 'none'")
    ap.add_argument("--width", type=int, default=900, help="width of SVG/HTML renders in px")
    ap.add_argument("--timeout", type=int, default=600, help="per-step timeout in seconds")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--work", help="keep intermediate files in this directory")
    ap.add_argument("--browser-no-sandbox", action="store_true",
                    help="pass --no-sandbox to Chrome (implied when running as root)")
    ap.add_argument("--check", action="store_true",
                    help="render nothing; fail if the README index or the PNGs in --out "
                         "differ from what the configs would generate")
    idx = ap.add_mutually_exclusive_group()
    idx.add_argument("--index", action="store_true",
                     help="write README.md even for an explicit list of configs")
    idx.add_argument("--no-index", action="store_true", help="do not write README.md")
    args = ap.parse_args(argv)
    if args.check:
        out_dir = Path(args.out).resolve()
        problems = check_index(collect_configs(args.configs), out_dir)
        for p in problems:
            print(f"make_gallery --check: {p}", file=sys.stderr)
        if problems:
            print(f"make_gallery: {out_dir / 'README.md'} is out of date; "
                  "run scripts/make_gallery.py", file=sys.stderr)
            return 1
        print(f"gallery index up to date: {out_dir / 'README.md'}")
        return 0
    # A subset run would replace the full index with a partial one.
    write_idx = args.index or (not args.no_index and not args.configs)

    gmsh = None if args.gmsh in (None, "none") else shutil.which(args.gmsh)
    if args.gmsh not in (None, "none") and gmsh is None:
        print(f"make_gallery: gmsh '{args.gmsh}' not found; .geo outputs will be skipped")
    svg_kind, svg_cmd = find_svg_converter()
    # Chrome refuses to start its sandbox as root (e.g. in containers).
    is_root = hasattr(os, "geteuid") and os.geteuid() == 0
    tools = Tools(find_rvegen(args.rvegen), gmsh, svg_cmd, svg_kind,
                  find_browser(args.browser), args.width, args.timeout,
                  no_sandbox=args.browser_no_sandbox or is_root)
    configs = collect_configs(args.configs)
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp = None
    if args.work:
        work_root = Path(args.work).resolve()
        work_root.mkdir(parents=True, exist_ok=True)
    else:
        tmp = tempfile.TemporaryDirectory(prefix="rvegen-gallery-")
        work_root = Path(tmp.name)

    print(f"rvegen: {tools.rvegen}\ngmsh: {tools.gmsh or '-'} · svg: {tools.svg_kind or '-'}"
          f" · browser: {tools.browser or '-'}\nconfigs: {len(configs)} -> {out_dir}")
    try:
        with futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            jobs = {pool.submit(process, c, slug, out_dir, work_root, tools): c
                    for c, slug in assign_slugs(configs)}
            results = []
            for fut in futures.as_completed(jobs):
                res = fut.result()
                results.append(res)
                status = "ERR" if not res.ok or res.failed else "ok "
                detail = ", ".join(n for n, _ in res.images) if res.ok else res.message
                name = res.config.name
                if res.slug != res.config.stem:
                    name += f" (as {res.slug})"
                print(f"[{status}] {name}: {detail}")
                for note in res.failed:
                    print(f"      FAILED {note}")
                for note in res.skipped:
                    print(f"      skipped {note}")
    finally:
        if tmp:
            tmp.cleanup()

    if write_idx:
        write_index(results, out_dir, gmsh_version(tools.gmsh))
        print(f"index: {out_dir / 'README.md'}")
    bad = [r for r in results if not r.ok or r.failed]
    if bad:
        print(f"make_gallery: {len(bad)} of {len(results)} configs failed: "
              + ", ".join(sorted(r.config.name for r in bad)), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
