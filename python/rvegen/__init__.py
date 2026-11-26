"""rvegen — RVE generator for FFT/FEM homogenization.

Phase-2 Python surface: shape primitives (Circle, Sphere, Rectangle, Box,
Ellipse, PolylineTube, MeshInclusion, VoronoiCell). Distributions,
generators, and the JSON-driven config path land in follow-up PRs.
"""

from ._core import (
    Box,
    Circle,
    ConvexPolygon,
    Ellipse,
    MeshInclusion,
    PolylineTube,
    Rectangle,
    Sphere,
    VoronoiCell,
    VoronoiGenerator2d,
    VoronoiGenerator3d,
    WeaveGenerator,
    lognormal_grain_seeds_3d,
    voro_pp_available,
)

__all__ = [
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
    "VoronoiGenerator3d",
    "WeaveGenerator",
    "lognormal_grain_seeds_3d",
    "voro_pp_available",
]
__version__ = "0.1.0"
