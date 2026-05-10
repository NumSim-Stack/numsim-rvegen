"""rvegen — RVE generator for FFT/FEM homogenization.

Phase-1 Python surface: shape primitives (Circle, Sphere). Distributions,
generators, and the JSON-driven config path land in follow-up PRs.
"""

from ._core import Circle, Sphere

__all__ = ["Circle", "Sphere"]
__version__ = "0.1.0"
