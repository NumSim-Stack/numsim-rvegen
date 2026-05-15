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
        "Ellipse",
        "MeshInclusion",
        "PolylineTube",
        "Rectangle",
        "Sphere",
        "VoronoiCell",
        "homogenization",
    ]:
        assert hasattr(rvegen, name), f"rvegen.{name} missing from public API"


def test_homogenization_mori_tanaka():
    """Stiff fibre in soft matrix raises the effective moduli above the
    matrix and stays under the Voigt arithmetic mean."""
    K_eff, G_eff = rvegen.homogenization.mori_tanaka_moduli(
        K_matrix=1.0e9, G_matrix=0.5e9,
        K_inclusions=[5.0e10], G_inclusions=[3.0e10],
        volume_fractions=[0.3])
    assert K_eff > 1.0e9
    assert G_eff > 0.5e9
    # Voigt upper bound for K = (1-f)·K_m + f·K_inc.
    K_voigt = 0.7 * 1.0e9 + 0.3 * 5.0e10
    assert K_eff < K_voigt


def test_homogenization_hashin_shtrikman_brackets_mt():
    """HS lower ≤ MT ≤ HS upper for a 2-phase well-ordered mixture."""
    f = 0.3
    K_m, G_m = 1.0e9, 0.5e9
    K_f, G_f = 5.0e10, 3.0e10
    K_mt, G_mt = rvegen.homogenization.mori_tanaka_moduli(
        K_matrix=K_m, G_matrix=G_m,
        K_inclusions=[K_f], G_inclusions=[G_f],
        volume_fractions=[f])
    K_lo, G_lo = rvegen.homogenization.hashin_shtrikman_lower(
        K_m, G_m, K_f, G_f, f)
    K_hi, G_hi = rvegen.homogenization.hashin_shtrikman_upper(
        K_m, G_m, K_f, G_f, f)
    # MT-with-soft-as-matrix equals HS lower analytically; allow a
    # small numerical slop.
    assert K_lo <= K_mt * (1 + 1e-9) + 1e-9
    assert K_mt <= K_hi * (1 + 1e-9) + 1e-9
    assert G_lo <= G_mt * (1 + 1e-9) + 1e-9
    assert G_mt <= G_hi * (1 + 1e-9) + 1e-9


def test_homogenization_self_consistent_within_hs():
    """Universal property: SC estimate lies within [HS-, HS+]."""
    K = [1.0e9, 5.0e10]
    G = [0.5e9, 3.0e10]
    v = [0.7, 0.3]
    K_sc, G_sc = rvegen.homogenization.self_consistent_moduli(K, G, v)
    K_lo, G_lo = rvegen.homogenization.hashin_shtrikman_lower_n(K, G, v)
    K_hi, G_hi = rvegen.homogenization.hashin_shtrikman_upper_n(K, G, v)
    assert K_lo <= K_sc <= K_hi
    assert G_lo <= G_sc <= G_hi


def test_homogenization_self_consistent_raises_on_non_convergence():
    """max_iter=1 with tight tolerance forces a non-convergence throw."""
    with pytest.raises(RuntimeError, match="not converge"):
        rvegen.homogenization.self_consistent_moduli(
            [1.0e9, 5.0e10], [0.5e9, 3.0e10], [0.7, 0.3],
            max_iter=1, tol=1e-15)
