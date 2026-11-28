"""rvegen — RVE geometry generator.

Python surface:

* shape primitives (``Circle``, ``Sphere``, ``Rectangle``, ``Box``,
  ``Ellipse``, ``PolylineTube``, ``MeshInclusion``, ``VoronoiCell``,
  ``ConvexPolygon``) over a common ``Shape`` base;
* ``run(config, seed=None)`` -- the CLI pipeline on a config dict or a JSON
  file path; returns a ``RunResult`` (sequence of shapes + ``.summary``);
* ``check(config)`` -- validate a config without generating anything;
* ``voxelize(shapes, domain, nx, ny, nz)`` -- numpy ``uint32`` grid;
* ``voxel_fractions(shapes, domain, nx, ny, nz, subsamples)`` -- the grid
  with partial volumes; ``voxel_resolution(...)`` -- voxels across the
  thinnest shape and the grid that would resolve it;
* registry introspection: ``rvegen.distributions``,
  ``direction_distributions``, ``inputs`` (alias ``shapes``), ``generators``,
  ``terminations``, ``post_processes``, each with ``keys()`` and
  ``schema(name)``.
"""

from ._core import (
    Box,
    Circle,
    ConvexPolygon,
    Ellipse,
    MeshInclusion,
    PolylineTube,
    Rectangle,
    RunResult,
    Shape,
    Sphere,
    VoronoiCell,
    VoronoiGenerator2d,
    VoronoiGenerator3d,
    WeaveGenerator,
    WeavePattern,
    TargetNotReached,
    lognormal_grain_seeds_3d,
    voro_pp_available,
    direction_distributions,
    check,
    distributions,
    generators,
    inputs,
    post_processes,
    run,
    shapes,
    terminations,
    voxel_fractions,
    voxel_resolution,
    voxelize,
)

__all__ = [
    "Box",
    "Circle",
    "ConvexPolygon",
    "Ellipse",
    "MeshInclusion",
    "PolylineTube",
    "Rectangle",
    "RunResult",
    "Shape",
    "Sphere",
    "TargetNotReached",
    "VoronoiCell",
    "VoronoiGenerator2d",
    "VoronoiGenerator3d",
    "WeaveGenerator",
    "WeavePattern",
    "lognormal_grain_seeds_3d",
    "voro_pp_available",
    "check",
    "direction_distributions",
    "distributions",
    "generators",
    "inputs",
    "post_processes",
    "run",
    "shapes",
    "terminations",
    "voxel_fractions",
    "voxel_resolution",
    "voxelize",
]
__version__ = "0.1.0"
