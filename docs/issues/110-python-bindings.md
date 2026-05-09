# #110 — Python bindings (pybind11)

> Drafted issue + PR text. Branch: `feature/python-bindings`.
> **Status: design only — no implementation in this branch.**

## Issue body

The single biggest adoption-barrier reduction. Researchers script in
Python; right now they have to shell out to the CLI or write a JSON
config and read the result file back. Native pybind11 module would
make rvegen useful in Jupyter notebooks, pipelines that mix it with
scikit-learn / scipy, and as a backend behind FFT solvers driven from
Python (DAMASK has Python bindings, AMITEX is moving that way).

### Proposed module

```python
import rvegen
import numpy as np

# Schema introspection — every registry's keys + parameters().
print(rvegen.distributions.keys())
# ['uniform_real', 'normal', 'constant']

print(rvegen.distributions.schema('uniform_real'))
# {'a': {'type': 'double', 'required': True, 'description': '...'},
#  'b': {'type': 'double', 'required': True, 'description': '...'}}

# Build distributions, inputs, generator, termination — JSON-style or
# direct.
config = {
    "distributions": { "uniform_x": {...}, ... },
    "shapes": [...],
    "generator":   {"type": "only_inside", "max_attempts": 100000},
    "termination": {"type": "number_of_inclusions", "target": 50},
}
shapes = rvegen.run(config, seed=2026)
print(f"placed {len(shapes)} shapes")

# Per-shape attribute access.
for s in shapes:
    print(s.kind, s.center, s.area())

# Voxel grid as numpy array directly — no file round-trip.
grid = rvegen.voxelize(shapes, domain_box=[1.0, 1.0, 0.0],
                      nx=64, ny=64, nz=1)
# grid is a numpy.ndarray of dtype uint32, shape (64, 64, 1).
```

### Build setup

```cmake
option(RVEGEN_BUILD_PYTHON "Build Python bindings via pybind11" OFF)
if(RVEGEN_BUILD_PYTHON)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(rvegen_py python/rvegen_py.cpp)
  target_link_libraries(rvegen_py PRIVATE rvegen_lib)
  set_target_properties(rvegen_py PROPERTIES OUTPUT_NAME rvegen)
endif()
```

`pip install` story: a thin `setup.py` / `pyproject.toml` that uses
scikit-build-core to invoke CMake. Standard pybind11 packaging
pattern.

### Acceptance

- [ ] `pip install -e .` produces a working `rvegen` Python module.
- [ ] All registry types introspectable via `rvegen.<category>.keys()`.
- [ ] `rvegen.run(config_dict)` runs the pipeline and returns shapes.
- [ ] `rvegen.voxelize(...)` returns a numpy array.
- [ ] PyPI-publishable `pyproject.toml` with manylinux wheels via
      cibuildwheel.

### Implementation outline

```cpp
// python/rvegen_py.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/json.hpp>

#include "rvegen/rvegen.h"

PYBIND11_MODULE(rvegen, m) {
    // Registry introspection submodules.
    auto dist_mod = m.def_submodule("distributions");
    dist_mod.def("keys", []() { /* return keys of distribution_registry */ });
    dist_mod.def("schema", [](std::string const& name) { /* convert
        parameter_controller_t to a Python dict */ });
    // ... same for shapes, generators, terminations, post_processes.

    // Pipeline runner.
    m.def("run", [](nlohmann::json const& config, std::uint64_t seed) {
        // Build pipeline from JSON, run, return shapes as Python objects.
    });

    // Direct voxel access.
    m.def("voxelize", [](std::vector<...> const& shapes,
                         std::array<double,3> box,
                         std::size_t nx, std::size_t ny, std::size_t nz) {
        auto grid = rvegen::sample_voxel_grid(shapes, box, nx, ny, nz);
        return py::array_t<std::uint32_t>(...);  // numpy array
    });
}
```

### Out of scope

- Async / concurrency from Python (rvegen pipelines are short enough
  not to need it for the bindings).
- Stub generation for type hints — possible follow-up via `pybind11-
  stubgen`.
- Pickle support for shape objects — useful but niche; add if
  requested.

## PR body

Closes #110. **Currently draft — no implementation in this branch.**

Implementation work requires:
- ~400 LOC for the pybind11 module surface (registries × 5 +
  pipeline runner + voxelize).
- ~80 LOC for `pyproject.toml` + scikit-build configuration.
- ~150 LOC for cibuildwheel CI to produce manylinux wheels.
- ~200 LOC of Python-side tests (round-trip via JSON, schema
  introspection, voxel-array shape checks).
- Documentation: a `docs/python.md` walkthrough.

Estimated 2–3 days of focused work, plus 1 day for packaging /
manylinux wheels. Substantial payoff: drops the adoption barrier
for the 90% of users who script in Python.
