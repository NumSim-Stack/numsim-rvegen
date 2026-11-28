"""Benchmark cases for scripts/run_benchmarks.py.

Each case is one whole pipeline run of the rvegen binary: a config, built
here for a size, plus the input files it needs. Cases of one family differ
only in size, which is what the scaling check compares.

A case is a dict:
  name     unique, used for file names and for --filter
  family   cases that differ in size only share it
  size     the number the config was built for (inclusions, seeds, voxels)
  work     what the run time is divided by in the scaling check:
           "attempts" (placement attempts of the run), "shapes" (shapes
           generated) or None (no scaling check: writers, single sizes)
  config   the JSON config; output paths are relative, the runner gives each
           case its own directory
  files    {relative path: text} written next to the config before the run
  large    True for the sizes --quick leaves out
"""

import math

ATTEMPTS = 50_000_000  # never the limit; the termination ends the run


def _uniform(a, b):
    return {"type": "uniform_real", "a": a, "b": b}


def _case(name, family, size, work, config, files=None, large=False):
    return {"name": name, "family": family, "size": size, "work": work,
            "config": config, "files": files or {}, "large": large}


def circles(n, fill, generator="periodic", gap=0.0, post=()):
    """About n circles filling `fill` of the unit square."""
    r = math.sqrt(fill / (math.pi * n))
    gen = {"type": generator, "max_attempts": ATTEMPTS}
    if gap:
        gen["min_gap"] = gap * r
    return {
        "domain_box": [1.0, 1.0],
        "seed": 1,
        "distributions": {"pos": _uniform(0.0, 1.0),
                          "radius": _uniform(0.8 * r, 1.2 * r)},
        "shapes": [{"type": "circle_input", "pos_x_dist": "pos",
                    "pos_y_dist": "pos", "radius_dist": "radius"}],
        "generator": gen,
        "termination": {"type": "volume_fraction", "target_fraction": fill},
        "postprocess": list(post),
    }


def spheres(n, fill, generator="periodic", post=()):
    """About n spheres filling `fill` of the unit cube."""
    r = (fill / (4.0 / 3.0 * math.pi * n)) ** (1.0 / 3.0)
    return {
        "domain_box": [1.0, 1.0, 1.0],
        "seed": 1,
        "distributions": {"pos": _uniform(0.0, 1.0),
                          "radius": _uniform(0.8 * r, 1.2 * r)},
        "shapes": [{"type": "sphere_input", "pos_x_dist": "pos",
                    "pos_y_dist": "pos", "pos_z_dist": "pos",
                    "radius_dist": "radius"}],
        "generator": {"type": generator, "max_attempts": ATTEMPTS},
        "termination": {"type": "volume_fraction", "target_fraction": fill},
        "postprocess": list(post),
    }


def mixed_2d(n, fill=0.25):
    """Circles, rectangles and rotated ellipses, about n in all."""
    s = math.sqrt(fill / n)  # side of the square one inclusion has to itself
    return {
        "domain_box": [1.0, 1.0],
        "seed": 1,
        "distributions": {
            "pos": _uniform(0.0, 1.0),
            "radius": _uniform(0.45 * s, 0.65 * s),
            "width": _uniform(0.9 * s, 1.3 * s),
            "height": _uniform(0.6 * s, 0.9 * s),
            "semi_a": _uniform(0.7 * s, 0.9 * s),
            "semi_b": _uniform(0.3 * s, 0.45 * s),
            "angle": _uniform(0.0, math.pi),
        },
        "shapes": [
            {"type": "circle_input", "pos_x_dist": "pos", "pos_y_dist": "pos",
             "radius_dist": "radius"},
            {"type": "rectangle_input", "pos_x_dist": "pos",
             "pos_y_dist": "pos", "width_dist": "width",
             "height_dist": "height"},
            {"type": "ellipse_input", "pos_x_dist": "pos", "pos_y_dist": "pos",
             "radius_a_dist": "semi_a", "radius_b_dist": "semi_b",
             "rotation_dist": "angle"},
        ],
        "generator": {"type": "periodic", "max_attempts": ATTEMPTS},
        "termination": {"type": "volume_fraction", "target_fraction": fill},
        "postprocess": [],
    }


def mixed_3d(n, fill=0.15):
    """Spheres and boxes in the cube, about n in all."""
    s = (fill / n) ** (1.0 / 3.0)
    return {
        "domain_box": [1.0, 1.0, 1.0],
        "seed": 1,
        "distributions": {"pos": _uniform(0.0, 1.0),
                          "radius": _uniform(0.5 * s, 0.7 * s),
                          "edge": _uniform(0.8 * s, 1.2 * s)},
        "shapes": [
            {"type": "sphere_input", "pos_x_dist": "pos", "pos_y_dist": "pos",
             "pos_z_dist": "pos", "radius_dist": "radius"},
            {"type": "box_input", "pos_x_dist": "pos", "pos_y_dist": "pos",
             "pos_z_dist": "pos", "width_dist": "edge", "height_dist": "edge",
             "depth_dist": "edge"},
        ],
        "generator": {"type": "periodic", "max_attempts": ATTEMPTS},
        "termination": {"type": "volume_fraction", "target_fraction": fill},
        "postprocess": [],
    }


def fibres(n, fill=0.10):
    """n nearly vertical straight fibres through the cube."""
    r = math.sqrt(fill / (math.pi * n))
    return {
        "domain_box": [1.0, 1.0, 1.0],
        "seed": 1,
        "distributions": {"lateral": _uniform(2 * r, 1.0 - 2 * r),
                          "bottom": _uniform(0.02, 0.06),
                          "top": _uniform(0.94, 0.98),
                          "radius": _uniform(0.9 * r, 1.1 * r)},
        "shapes": [{"type": "polyline_tube_input",
                    "start_x_dist": "lateral", "start_y_dist": "lateral",
                    "start_z_dist": "bottom", "end_x_dist": "lateral",
                    "end_y_dist": "lateral", "end_z_dist": "top",
                    "radius_dist": "radius"}],
        "generator": {"type": "only_inside", "max_attempts": ATTEMPTS},
        "termination": {"type": "number_of_inclusions", "target": n},
        "postprocess": [],
    }


def icosphere_ply(subdivisions):
    """A unit sphere of 20 * 4^subdivisions triangles as an ASCII PLY file."""
    t = (1.0 + math.sqrt(5.0)) / 2.0
    raw = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t),
           (0, 1, t), (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1),
           (-t, 0, -1), (-t, 0, 1)]

    def unit(p):
        length = math.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2)
        return (p[0] / length, p[1] / length, p[2] / length)

    vertices = [unit(p) for p in raw]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
             (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
             (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
             (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    for _ in range(subdivisions):
        middle = {}

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in middle:
                pa, pb = vertices[a], vertices[b]
                vertices.append(unit(((pa[0] + pb[0]) / 2,
                                      (pa[1] + pb[1]) / 2,
                                      (pa[2] + pb[2]) / 2)))
                middle[key] = len(vertices) - 1
            return middle[key]

        split = []
        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            split += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = split
    lines = ["ply", "format ascii 1.0",
             f"element vertex {len(vertices)}", "property double x",
             "property double y", "property double z",
             f"element face {len(faces)}",
             "property list uchar int vertex_indices", "end_header"]
    lines += [f"{x:.17g} {y:.17g} {z:.17g}" for x, y, z in vertices]
    lines += [f"3 {a} {b} {c}" for a, b, c in faces]
    return "\n".join(lines) + "\n"


def meshes(n, subdivisions, fill=0.10):
    """n rotated copies of a triangulated sphere."""
    r = (fill / (4.0 / 3.0 * math.pi * n)) ** (1.0 / 3.0)
    config = {
        "domain_box": [1.0, 1.0, 1.0],
        "seed": 1,
        "distributions": {"pos": _uniform(1.3 * r, 1.0 - 1.3 * r),
                          "unit": _uniform(0.0, 1.0),
                          "scale": _uniform(0.8 * r, 1.2 * r)},
        "shapes": [{"type": "mesh_inclusion_input", "mesh_path": "sphere.ply",
                    "position_x_dist": "pos", "position_y_dist": "pos",
                    "position_z_dist": "pos", "orientation": "uniform",
                    "orientation_dist": "unit", "scale_dist": "scale"}],
        "generator": {"type": "only_inside", "max_attempts": ATTEMPTS},
        "termination": {"type": "number_of_inclusions", "target": n},
        "postprocess": [],
    }
    return config, {"sphere.ply": icosphere_ply(subdivisions)}


def voronoi(dim, n):
    return {
        "domain_box": [1.0] * dim,
        "generator": {"type": f"voronoi_{dim}d", "n_seeds": n, "seed": 1,
                      "periodic": True},
        "postprocess": [],
    }


def weave(yarns, post=()):
    return {
        "domain_box": [1.0, 1.0, 0.2],
        "generator": {"type": "weave", "pattern": "twill_2_2",
                      "n_warp_yarns": yarns, "n_weft_yarns": yarns,
                      "yarn_radius": 0.22 / yarns, "amplitude": 0.28 / yarns,
                      "n_segments_per_yarn": 64,
                      "warp_phase_name": "warp", "weft_phase_name": "weft"},
        "phases": [{"name": "matrix"}, {"name": "warp"}, {"name": "weft"}],
        "postprocess": list(post),
    }


def cases():
    out = []

    def add(*args, **kwargs):
        out.append(_case(*args, **kwargs))

    # ---- placement -----------------------------------------------------------
    for n in (1_000, 10_000, 100_000):
        add(f"circles_periodic_{n}", "circles, periodic, 40 %", n, "attempts",
            circles(n, 0.40))
    for n in (10_000, 100_000):
        add(f"circles_inside_{n}", "circles, only_inside, 30 %", n, "attempts",
            circles(n, 0.30, generator="only_inside"))
    for n in (10_000, 100_000):
        add(f"circles_min_gap_{n}", "circles, periodic, 40 %, min_gap", n,
            "attempts", circles(n, 0.40, gap=0.2))
    for n in (10_000, 100_000):
        add(f"mixed_2d_{n}", "circles + rectangles + ellipses, 25 %", n,
            "attempts", mixed_2d(n))
    for n in (1_000, 10_000, 100_000):
        add(f"spheres_periodic_{n}", "spheres, periodic, 30 %", n, "attempts",
            spheres(n, 0.30), large=n > 10_000)
    for n in (10_000, 100_000):
        add(f"mixed_3d_{n}", "spheres + boxes, 15 %", n, "attempts",
            mixed_3d(n))
    for n in (100, 400):
        add(f"fibres_{n}", "straight fibres, 5 %", n, "attempts",
            fibres(n, 0.05))
    for n in (200, 2_000):
        config, files = meshes(n, 5)
        add(f"mesh_inclusions_{n}", "mesh inclusions, 20 480 triangles each",
            n, "attempts", config, files=files)

    # ---- tessellations and weaves -------------------------------------------
    for n in (1_000, 10_000, 100_000):
        add(f"voronoi_2d_{n}", "voronoi_2d, periodic", n, "shapes",
            voronoi(2, n), large=n > 10_000)
    for n in (500, 2_000, 8_000):
        add(f"voronoi_3d_{n}", "voronoi_3d, periodic", n, "shapes",
            voronoi(3, n), large=n > 2_000)
    add("weave_twill_16", "twill weave, 16 x 16 yarns", 16, None, weave(16))

    # ---- writers: the whole run, generation included (compare with the
    # ---- placement case of the same size) ------------------------------------
    def voxel(nx, ny, nz):
        return {"type": "voxel", "nx": nx, "ny": ny, "nz": nz,
                "output_path": "out.vox"}

    add("write_voxel_2d_2048", "writers", 2048 * 2048, None,
        circles(10_000, 0.40, post=[voxel(2048, 2048, 1)]))
    add("write_voxel_3d_128", "writers", 128 ** 3, None,
        spheres(1_000, 0.30, post=[voxel(128, 128, 128)]))
    add("write_voxel_3d_256", "writers", 256 ** 3, None,
        spheres(1_000, 0.30, post=[voxel(256, 256, 256)]), large=True)
    add("write_voxel_3d_256_dense", "writers", 256 ** 3, None,
        spheres(10_000, 0.30, post=[voxel(256, 256, 256)]), large=True)
    add("write_vtk_3d_128", "writers", 128 ** 3, None,
        spheres(1_000, 0.30, post=[{"type": "vtk_legacy", "nx": 128,
                                    "ny": 128, "nz": 128,
                                    "output_path": "out.vtk"}]))
    add("write_gmsh_geo_2d", "writers", 1_000, None,
        circles(1_000, 0.40, post=[{"type": "gmsh_geo",
                                    "output_path": "out.geo"}]))
    add("write_svg_2d", "writers", 10_000, None,
        circles(10_000, 0.40, post=[{"type": "svg", "canvas_width": 800.0,
                                     "output_path": "out.svg"}]))
    add("write_statistics_2d", "writers", 10_000, None,
        circles(10_000, 0.40, post=[{"type": "statistics", "nx": 512,
                                     "ny": 512, "nbins": 64,
                                     "output_path": "out.csv"}]))
    add("write_three_js_3d", "writers", 1_000, None,
        spheres(1_000, 0.30, post=[{"type": "three_js",
                                    "output_path": "out.html"}]))
    add("write_voxel_weave", "writers", 256 * 256 * 64, None,
        weave(16, post=[voxel(256, 256, 64)]))
    return out
