# rvegen Python bindings

Phase-2 shape surface: all eight shape primitives. Distributions, generators,
and the JSON-driven config path land in follow-up PRs.

## Build & install (editable, from this directory)

Use a virtual environment — modern Linux distros (Debian, Ubuntu, …)
mark the system Python "externally managed" (PEP 668) and refuse a
plain `pip install`.

```bash
cd python
python3 -m venv .venv
source .venv/bin/activate
pip install -e .            # editable install — picks up source edits
```

scikit-build-core invokes the CMake build, which fetches pybind11 if not
found and links against the parent `rvegen_lib` interface target.

If your numsim-core checkout isn't at `../../numsim-core/include` (the
default sibling layout), point CMake at it:

```bash
CMAKE_ARGS="-DCMAKE_CXX_FLAGS=-I/path/to/numsim-core/include" \
  pip install -e .
```

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

e = rvegen.Ellipse(0.0, 0.0, 0.5, 0.2, 0.0)
print(e.area())                  # ≈ π · 0.5 · 0.2

t = rvegen.PolylineTube([(0, 0, 0), (1, 0, 0)], radius=0.1)
print(t.is_inside(0.5, 0, 0))    # True

m = rvegen.MeshInclusion.from_stl_file("cube.stl")
print(m.triangle_count, m.volume())

# Voronoi cell — convex polyhedron via vertices + face-index lists.
v = rvegen.VoronoiCell(
    vertices=[(-0.5,-0.5,-0.5),(0.5,-0.5,-0.5),(0.5,0.5,-0.5),(-0.5,0.5,-0.5),
              (-0.5,-0.5,0.5),(0.5,-0.5,0.5),(0.5,0.5,0.5),(-0.5,0.5,0.5)],
    faces=[[0,3,2,1],[4,5,6,7],[0,1,5,4],[3,7,6,2],[0,4,7,3],[1,2,6,5]],
)
print(v.volume())                # 1.0
```

## Tests

A pytest suite under `python/tests/` exercises one method per bound
shape. After an editable install (in your venv):

```bash
pip install pytest
pytest python/tests/
```

Running the suite without `pip install -e .` first will fail with
`ModuleNotFoundError: No module named 'rvegen'` — the editable install
is what links the compiled `_core.so` next to the package.

The CI workflow runs this on every PR (see `.github/workflows/ci.yml`).

## What's NOT exposed yet

- Distributions (need engine reference + GIL handling)
- Generators, terminations, post-processes
- JSON config path (`rvegen.run_from_json(...)`)
- Wheels via cibuildwheel (planned alongside the conda-forge feedstock)
