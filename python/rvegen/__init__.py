"""rvegen — RVE generator for FFT/FEM homogenization.

Phase-2 Python surface:
  * Shape primitives: Circle, Sphere, Rectangle, Box, Ellipse,
    PolylineTube, MeshInclusion, VoronoiCell.
  * Analytical mean-field homogenization: rvegen.homogenization
    submodule with mori_tanaka_moduli, hashin_shtrikman_*, and
    self_consistent_moduli.

Distributions, generators, and the JSON-driven config path land in
follow-up PRs.
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
    homogenization,
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
    "homogenization",
]
__version__ = "0.1.0"
