# rvegen Python bindings

Phase-1 surface: shape primitives only. Distributions, generators, and the
JSON-driven config path land in follow-up PRs.

## Build & install (editable, from this directory)

```bash
cd python
pip install -e .
```

scikit-build-core invokes the CMake build, which fetches pybind11 if not
found and links against the parent `rvegen_lib` interface target.

## Usage

```python
import rvegen

c = rvegen.Circle(0.5, 0.5, 0.1)
print(c.x, c.y, c.radius)        # 0.5 0.5 0.1
print(c.is_inside(0.55, 0.55))   # True
print(c.area())                  # ≈ π · 0.01

s = rvegen.Sphere(0.0, 0.0, 0.0, 0.5)
print(s.volume())                # ≈ (4/3)π · 0.125

r = rvegen.Rectangle(0.0, 0.0, 0.2, 0.4)
print(r.area())                  # 0.08

b = rvegen.Box(0.0, 0.0, 0.0, 0.2, 0.4, 0.5)
print(b.volume())                # 0.04
```

## What's NOT exposed yet

- Ellipse, polyline_tube, mesh_inclusion, voronoi_cell (trivial follow-up — same binding pattern)
- Distributions (need engine reference + GIL handling)
- Generators, terminations, post-processes
- JSON config path (`rvegen.run_from_json(...)`)
- Wheels via cibuildwheel (planned alongside the conda-forge feedstock)
