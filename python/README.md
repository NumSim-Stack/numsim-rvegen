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
```

## What's NOT exposed yet

- Rectangle / box / ellipse / polyline_tube / mesh_inclusion (trivial follow-up — same binding pattern as Circle / Sphere)
- Distributions (need engine reference + GIL handling)
- Generators, terminations, post-processes
- JSON config path (`rvegen.run_from_json(...)`)
- Wheels via cibuildwheel (planned alongside the conda-forge feedstock)
