# rvegen Python bindings

Shape primitives, a JSON-driven `rvegen.run()`, `rvegen.voxelize()` and
registry introspection.

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

### Editable vs. non-editable install

- `pip install -e .` — editable, recommended for **development**.
  Source edits are picked up without re-installing. Used by the local
  pytest workflow above.
- `pip install ./python` — non-editable, matches what users will get
  from a wheel. **CI uses this** so a regression in the install path
  itself is caught. Requires re-install after every source edit.

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
pip install pytest numpy
pytest python/tests/
```

Running the suite without `pip install -e .` first will fail with
`ModuleNotFoundError: No module named 'rvegen'` — the editable install
is what links the compiled `_core.so` next to the package.

The CI workflow runs this on every PR (see `.github/workflows/ci.yml`).

## Pipeline, voxelization, introspection

```python
import rvegen

# Same JSON schema as the CLI; pass a dict or a path to a JSON file.
config = {
    "domain_box": [1.0, 1.0, 0.0],
    "distributions": {"x": {"type": "uniform_real", "a": 0.1, "b": 0.9},
                      "r": {"type": "uniform_real", "a": 0.03, "b": 0.05}},
    "shapes": [{"type": "circle_input", "pos_x_dist": "x",
                "pos_y_dist": "x", "radius_dist": "r"}],
    "generator": {"type": "only_inside", "max_attempts": 100000},
    "termination": {"type": "number_of_inclusions", "target": 50},
}
res = rvegen.run(config, seed=2026)   # RunResult: sequence of shapes + summary
print(len(res), res.summary)          # seed, n_shapes, domain_box, volume_fraction, ...
for s in res:
    print(s.kind, s.center, s.area())

# Validate without generating: raises on the first problem, writes nothing.
rvegen.check(config)

# A config file: relative input paths in it (mesh_path) are resolved against
# the file's directory, relative output paths against output_dir (default:
# the working directory). For a dict, input_dir says where inputs are.
res = rvegen.run("examples/mesh_inclusions_3d.json", output_dir="results")

# run() prints nothing. Pass a callable to see the progress lines as they
# are produced; verbose=True adds the defaults used per block, the resolved
# paths and the seed. Warnings are always raised as UserWarning.
res = rvegen.run(config, log=print, verbose=True)
res = rvegen.run(config, log=logging.getLogger("rvegen").info)

# Writers under "postprocess" run too (set postprocess=False to skip them).
# Deterministic generators (voronoi_2d/3d, weave, tow_weave) need no
# "shapes" / "termination" sections.

# numpy uint32 grid indexed [i, j, k]; value = 1-based index of the first
# containing shape, 0 = outside. 2D domains collapse to nz = 1. The array is
# Fortran-ordered (grid.flags.f_contiguous, x fastest, zero-copy); use
# numpy.ascontiguousarray(grid) when C order is required. nx/ny/nz >= 1.
grid = rvegen.voxelize(list(res), [1.0, 1.0, 0.0], nx=64, ny=64, nz=1)

# Registry introspection: distributions, direction_distributions,
# inputs (alias: shapes), generators, terminations, post_processes.
print(rvegen.generators.keys())
print(rvegen.generators.schema("weave"))   # {field: {type, required, description, min, ...}}
```

The Python entry point calls the same `rvegen::run_pipeline`
(`include/rvegen/pipeline.h`) as the `rvegen` CLI.

## Wheels

`.github/workflows/wheels.yml` builds manylinux wheels for CPython
3.10-3.13 with cibuildwheel (on `v*` tags or manual dispatch); settings
live under `[tool.cibuildwheel]` in `pyproject.toml`.

## What's NOT exposed yet

- Distributions / generators / terminations as standalone Python objects (use `rvegen.run` with a config dict)
- Pickle support for shapes
