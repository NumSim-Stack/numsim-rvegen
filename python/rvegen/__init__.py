"""rvegen — RVE generator for FFT/FEM homogenization.

Phase-1 Python surface: shape primitives (Circle, Sphere, Rectangle, Box).
Distributions, generators, and the JSON-driven config path land in
follow-up PRs.
"""

from ._core import Box, Circle, Rectangle, Sphere

__all__ = ["Box", "Circle", "Rectangle", "Sphere"]
__version__ = "0.1.0"
