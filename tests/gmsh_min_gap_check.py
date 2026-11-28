#!/usr/bin/env python3
"""Periodic packs generated with a minimum gap mesh in gmsh (#111).

For 20 seeds each, a periodic 2D pack of circles and a periodic 3D pack of
spheres are generated with min_gap = wall_gap = the largest mesh size, the
.geo file is meshed with gmsh (periodic constraints on), and the run fails
on a non-zero exit status, an "Error" line or an empty mesh.

The shortest element edge is printed per dimension. It is not checked by
default: the gaps remove the slivers between inclusions and at the faces,
but an OpenCASCADE seam vertex next to a face still leaves a short edge now
and then (#150). --min-edge 0.2 turns the check on.

usage: gmsh_min_gap_check.py --rvegen <exe> --gmsh <exe> --workdir <dir>
                             [--seeds N] [--min-gap G] [--wall-gap G]
"""
import argparse
import json
import math
import pathlib
import subprocess
import sys

CLMAX = {2: 0.03, 3: 0.09}


def config(dim: int, seed: int, min_gap: float, wall_gap: float) -> dict:
    three = dim == 3
    shape = {"type": "sphere_input" if three else "circle_input",
             "pos_x_dist": "x", "pos_y_dist": "x", "radius_dist": "r"}
    if three:
        shape["pos_z_dist"] = "x"
    return {
        "domain_box": [1.0, 1.0, 1.0] if three else [1.0, 1.0],
        "seed": seed,
        "distributions": {
            "x": {"type": "uniform_real", "a": 0.0, "b": 1.0},
            "r": ({"type": "uniform_real", "a": 0.12, "b": 0.18} if three else
                  {"type": "uniform_real", "a": 0.04, "b": 0.08}),
        },
        "shapes": [shape],
        "generator": {"type": "periodic", "max_attempts": 200000,
                      "min_gap": min_gap, "wall_gap": wall_gap},
        "termination": {"type": "number_of_inclusions",
                        "target": 12 if three else 25},
        "postprocess": [{"type": "gmsh_geo", "output_path": "pack.geo",
                         "periodic": True,
                         "char_length_min": CLMAX[dim] / 3,
                         "char_length_max": CLMAX[dim]}],
    }


def shortest_edge(msh: pathlib.Path, dim: int):
    """(number of triangles or tetrahedra, their shortest edge), MSH 2."""
    lines = msh.read_text(encoding="utf-8", errors="replace").splitlines()
    nodes = {}
    i = lines.index("$Nodes")
    for line in lines[i + 2:i + 2 + int(lines[i + 1])]:
        n, x, y, z = line.split()
        nodes[int(n)] = (float(x), float(y), float(z))
    wanted, corners = (2, 3) if dim == 2 else (4, 4)   # triangle / tetrahedron
    i = lines.index("$Elements")
    count, best = 0, math.inf
    for line in lines[i + 2:i + 2 + int(lines[i + 1])]:
        f = line.split()
        if int(f[1]) != wanted:
            continue
        count += 1
        pts = [nodes[int(n)] for n in f[3 + int(f[2]):][:corners]]
        for a in range(corners):
            for b in range(a + 1, corners):
                best = min(best, math.dist(pts[a], pts[b]))
    return count, best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rvegen", required=True)
    ap.add_argument("--gmsh", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--min-gap", type=float, default=None,
                    help="default: the largest mesh size")
    ap.add_argument("--wall-gap", type=float, default=None,
                    help="default: the largest mesh size")
    ap.add_argument("--dims", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--min-edge", type=float, default=0.0,
                    help="shortest allowed element edge, in units of the "
                         "smallest mesh size (default 0: not checked)")
    args = ap.parse_args()

    work = pathlib.Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    failed = []
    for dim in args.dims:
        min_gap = CLMAX[dim] if args.min_gap is None else args.min_gap
        wall_gap = CLMAX[dim] if args.wall_gap is None else args.wall_gap
        worst = math.inf
        for seed in range(1, args.seeds + 1):
            tag = f"{dim}D seed {seed}"
            (work / "pack.json").write_text(
                json.dumps(config(dim, seed, min_gap, wall_gap)),
                encoding="utf-8")
            for stale in ("pack.geo", "pack.msh"):
                (work / stale).unlink(missing_ok=True)
            gen = subprocess.run([args.rvegen, "pack.json"], cwd=work,
                                 capture_output=True, text=True)
            if gen.returncode != 0:
                failed.append(f"{tag}: rvegen exit {gen.returncode}: "
                              f"{(gen.stderr or gen.stdout).strip()[-300:]}")
                continue
            mesh = subprocess.run(
                [args.gmsh, "pack.geo", f"-{dim}", "-o", "pack.msh",
                 "-format", "msh2", "-v", "2"],
                cwd=work, capture_output=True, text=True, timeout=600)
            out = mesh.stdout + mesh.stderr
            errors = [l for l in out.splitlines() if "Error" in l]
            if mesh.returncode != 0 or errors:
                failed.append(f"{tag}: gmsh exit {mesh.returncode}: "
                              f"{' | '.join(errors)[:300]}")
                continue
            msh = work / "pack.msh"
            n, edge = shortest_edge(msh, dim) if msh.exists() else (0, 0.0)
            worst = min(worst, edge / (CLMAX[dim] / 3))
            if n == 0:
                failed.append(f"{tag}: gmsh wrote no elements")
            elif edge < args.min_edge * CLMAX[dim] / 3:
                failed.append(f"{tag}: shortest element edge {edge:.3g} is "
                              f"below {args.min_edge} x the smallest mesh "
                              f"size {CLMAX[dim] / 3:.3g}")
        print(f"{dim}D: shortest element edge over all seeds = "
              f"{worst:.3g} x the smallest mesh size")
    if failed:
        print("\n".join(failed))
        print(f"gmsh_min_gap_check: {len(failed)} run(s) failed")
        return 1
    print(f"gmsh_min_gap_check: OK ({args.seeds} seeds, dims {args.dims})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
