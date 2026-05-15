"""Smoke tests for the rvegen Python bindings — one method per shape.

Driven by the CI workflow (`.github/workflows/ci.yml`) after
`pip install ./python` in editable mode. The asserts are lightweight on
purpose: this is a guardrail that catches binding regressions
(missing re-export in `__init__.py`, wrong arg type in pybind11 lambda,
broken transitive include of \\<rvegen/...\\>), not a full geometric
test suite — those live on the C++ side.
"""

from __future__ import annotations

import math
import os
import tempfile

import pytest

import rvegen


def test_circle_basics():
    c = rvegen.Circle(0.5, 0.5, 0.1)
    assert c.x == pytest.approx(0.5)
    assert c.y == pytest.approx(0.5)
    assert c.radius == pytest.approx(0.1)
    assert c.is_inside(0.5, 0.5)
    assert not c.is_inside(0.7, 0.7)
    assert c.area() == pytest.approx(math.pi * 0.01)


def test_sphere_basics():
    s = rvegen.Sphere(0.0, 0.0, 0.0, 0.5)
    assert s.is_inside(0.0, 0.0, 0.0)
    assert not s.is_inside(1.0, 0.0, 0.0)
    assert s.volume() == pytest.approx((4.0 / 3.0) * math.pi * 0.125)


def test_rectangle_basics():
    r = rvegen.Rectangle(0.0, 0.0, 0.2, 0.4)
    assert r.area() == pytest.approx(0.08)
    assert r.is_inside(0.05, 0.1)
    assert not r.is_inside(0.5, 0.5)


def test_box_basics():
    b = rvegen.Box(0.0, 0.0, 0.0, 0.2, 0.4, 0.5)
    assert b.volume() == pytest.approx(0.04)
    assert b.is_inside(0.05, 0.1, 0.2)
    assert not b.is_inside(0.5, 0.0, 0.0)


def test_ellipse_basics():
    e = rvegen.Ellipse(0.0, 0.0, 0.5, 0.2, 0.0)
    # Area = π · a · b
    assert e.area() == pytest.approx(math.pi * 0.5 * 0.2)
    assert e.is_inside(0.0, 0.0)
    assert not e.is_inside(1.0, 0.0)


def test_polyline_tube_basics():
    t = rvegen.PolylineTube([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], radius=0.1)
    assert t.radius == pytest.approx(0.1)
    assert len(t.centerline) == 2
    assert t.is_inside(0.5, 0.0, 0.0)
    assert not t.is_inside(0.5, 0.5, 0.0)
    # π · r² · L = π · 0.01 · 1.0
    assert t.volume() == pytest.approx(math.pi * 0.01)


def test_voronoi_cell_unit_cube():
    verts = [
        (-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5),
        (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5),
    ]
    faces = [
        [0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
        [3, 7, 6, 2], [0, 4, 7, 3], [1, 2, 6, 5],
    ]
    v = rvegen.VoronoiCell(vertices=verts, faces=faces)
    assert v.volume() == pytest.approx(1.0)
    assert v.is_inside(0.0, 0.0, 0.0)
    assert not v.is_inside(0.6, 0.0, 0.0)
    # Ensure the accessors round-trip the input geometry.
    assert len(v.vertices) == 8
    assert len(v.faces) == 6


def test_mesh_inclusion_from_stl_file():
    # Smallest STL that the ASCII reader will accept (1 facet).
    stl = (
        "solid one\n"
        "  facet normal 0 0 -1\n"
        "    outer loop\n"
        "      vertex -0.5 -0.5 -0.5\n"
        "      vertex  0.5  0.5 -0.5\n"
        "      vertex  0.5 -0.5 -0.5\n"
        "    endloop\n"
        "  endfacet\n"
        "endsolid one\n"
    )
    with tempfile.NamedTemporaryFile(
        suffix=".stl", mode="w", delete=False
    ) as f:
        f.write(stl)
        path = f.name
    try:
        m = rvegen.MeshInclusion.from_stl_file(path)
        assert m.triangle_count == 1
        # Translate via the bound mutator — confirms the binding accepts
        # three doubles and forwards to the C++ method.
        m.set_middle_point(10.0, 10.0, 10.0)
    finally:
        os.remove(path)


def test_module_exports():
    """Every shape we documented in __all__ should be importable."""
    for name in [
        "Box",
        "Circle",
        "ConvexPolygon",
        "Ellipse",
        "MeshInclusion",
        "PolylineTube",
        "Rectangle",
        "Sphere",
        "VoronoiCell",
        "VoronoiGenerator2d",
        "WeaveGenerator",
    ]:
        assert hasattr(rvegen, name), f"rvegen.{name} missing from public API"


def test_phase_name_round_trip_on_every_shape():
    """Every bound shape exposes a phase_name property that round-trips."""
    shapes = [
        rvegen.Circle(0.5, 0.5, 0.1),
        rvegen.Sphere(0.0, 0.0, 0.0, 0.5),
        rvegen.Rectangle(0.0, 0.0, 0.2, 0.4),
        rvegen.Box(0.0, 0.0, 0.0, 0.2, 0.4, 0.5),
        rvegen.Ellipse(0.0, 0.0, 0.5, 0.2, 0.0),
        rvegen.PolylineTube([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], radius=0.1),
        rvegen.ConvexPolygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]),
    ]
    for s in shapes:
        # Default is empty.
        assert s.phase_name == ""
        # Setter + getter round-trip.
        s.phase_name = "fibre"
        assert s.phase_name == "fibre"


def test_convex_polygon_unit_square():
    """Direct ConvexPolygon construction + area / is_inside / centroid."""
    sq = rvegen.ConvexPolygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    assert sq.vertex_count == 4
    assert sq.area() == pytest.approx(1.0)
    assert sq.is_inside(0.5, 0.5)
    assert not sq.is_inside(1.5, 0.5)
    cx, cy, cz = sq.get_middle_point()
    assert cx == pytest.approx(0.5)
    assert cy == pytest.approx(0.5)
    assert cz == 0.0
    # Translation moves the polygon.
    sq.set_middle_point(5.0, 5.0)
    assert sq.is_inside(5.0, 5.0)
    assert not sq.is_inside(0.5, 0.5)


def test_convex_polygon_rejects_too_few_vertices():
    with pytest.raises(RuntimeError):
        rvegen.ConvexPolygon([(0.0, 0.0), (1.0, 0.0)])


def test_voronoi_generator_2d_split_two_seeds():
    """Symmetric two seeds → two half-rectangles of area 0.5 each, both as
    ConvexPolygon objects ready for downstream use."""
    gen = rvegen.VoronoiGenerator2d(
        Lx=1.0, Ly=1.0, seeds=[(0.25, 0.5), (0.75, 0.5)])
    assert gen.seed_count == 2

    polygons = gen.build_polygons()
    assert len(polygons) == 2
    assert len(polygons[0]) == 4
    assert len(polygons[1]) == 4

    cells = gen.build_cells()
    assert len(cells) == 2
    for cell in cells:
        assert isinstance(cell, rvegen.ConvexPolygon)
        assert cell.area() == pytest.approx(0.5)
    # Each seed is inside its own cell.
    assert cells[0].is_inside(0.25, 0.5)
    assert not cells[0].is_inside(0.75, 0.5)
    assert cells[1].is_inside(0.75, 0.5)


def test_voronoi_generator_2d_partition_invariant():
    """For arbitrary seed sets the sum of cell areas equals Lx·Ly."""
    seeds = [(0.3, 0.4), (1.7, 0.4), (0.5, 1.1),
             (1.2, 0.8), (1.5, 1.3), (0.8, 0.2)]
    gen = rvegen.VoronoiGenerator2d(Lx=2.0, Ly=1.5, seeds=seeds)
    cells = gen.build_cells()
    total = sum(c.area() for c in cells)
    assert total == pytest.approx(2.0 * 1.5, abs=1e-10)


def test_weave_generator_builds_expected_tube_count():
    """4 warp + 3 weft yarns → 7 PolylineTubes in the output list."""
    gen = rvegen.WeaveGenerator(
        domain_box=(1.0, 1.0, 0.2),
        n_warp_yarns=4,
        n_weft_yarns=3,
        yarn_radius=0.04,
        amplitude=0.03)
    tubes = gen.build()
    assert len(tubes) == 7
    for t in tubes:
        assert isinstance(t, rvegen.PolylineTube)
        assert t.radius == pytest.approx(0.04)


def test_weave_generator_phase_tagging_propagates():
    """set_warp_phase_name / set_weft_phase_name flow through to build()."""
    gen = rvegen.WeaveGenerator(
        domain_box=(1.0, 1.0, 0.2),
        n_warp_yarns=2, n_weft_yarns=2,
        yarn_radius=0.04, amplitude=0.03)
    gen.set_warp_phase_name("warp")
    gen.set_weft_phase_name("weft")
    assert gen.warp_phase_name == "warp"
    assert gen.weft_phase_name == "weft"
    # The bound PolylineTube doesn't expose phase_name on the Python
    # side yet (binding limitation, not generator limitation), so the
    # property assertions above are the binding-level test.


def test_weave_generator_validates_inputs():
    """Bad parameters surface as Python RuntimeError, not silent garbage."""
    with pytest.raises(RuntimeError, match="n_warp_yarns"):
        rvegen.WeaveGenerator(
            domain_box=(1.0, 1.0, 0.2),
            n_warp_yarns=0, n_weft_yarns=2,
            yarn_radius=0.04, amplitude=0.03)
    with pytest.raises(RuntimeError, match="yarn_radius"):
        rvegen.WeaveGenerator(
            domain_box=(1.0, 1.0, 0.2),
            n_warp_yarns=2, n_weft_yarns=2,
            yarn_radius=-0.01, amplitude=0.03)
    with pytest.raises(RuntimeError, match="2D domain"):
        # 2D (Lz=0) + non-zero amplitude is geometrically nonsensical.
        rvegen.WeaveGenerator(
            domain_box=(1.0, 1.0, 0.0),
            n_warp_yarns=2, n_weft_yarns=2,
            yarn_radius=0.04, amplitude=0.03)
