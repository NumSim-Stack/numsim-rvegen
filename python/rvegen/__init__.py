"""rvegen — RVE generator for FFT/FEM homogenization.

Phase-2 Python surface: shape primitives (Circle, Sphere, Rectangle, Box,
Ellipse, PolylineTube, MeshInclusion, VoronoiCell). Distributions,
generators, and the JSON-driven config path land in follow-up PRs.
"""

from ._core import (
    Box,
    Circle,
    Ellipse,
    MeshInclusion,
    PolylineTube,
    Rectangle,
    Sphere,
    VoronoiCell,
    WeaveGenerator,
)

__all__ = [
    "Box",
    "Circle",
    "Ellipse",
    "MeshInclusion",
    "PolylineTube",
    "Rectangle",
    "Sphere",
    "VoronoiCell",
    "WeaveGenerator",
]
__version__ = "0.1.0"
