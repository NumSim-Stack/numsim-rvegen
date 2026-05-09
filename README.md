# rvegen

JSON-driven Representative Volume Element (RVE) generator for FFT- and FEM-based
computational homogenization.

The library produces synthetic microstructures — random packings of circles,
spheres, rectangles, boxes, and other shapes — fully described by a JSON config
file. The output is either a gmsh `.geo` script (for FEM meshing) or a regular
voxel grid (for FFT solvers like Moulinec–Suquet).

The architecture is layered, header-only, and designed for extension: every
generator strategy, distribution, shape input, termination predicate, and
output writer is a registered type selectable by name from JSON. Adding a new
type does not require modifying any existing code.

---

## Quick start

```bash
cmake -S . -B build
cmake --build build
./build/rvegen examples/circles_in_unit_box.json /tmp/circles.vox
# → rvegen: generated 10 shapes
# → rvegen: wrote /tmp/circles.vox

ctest --test-dir build
```

By default the build excludes VTK rendering and CGAL voxelization (both
optional, both heavy). Enable either explicitly:

```bash
cmake -S . -B build -DRVEGEN_WITH_VTK=ON -DRVEGEN_WITH_VOXELIZATION=ON
```

The default build needs only Boost headers, nlohmann_json, and the vendored
GeometricTools headers under `external/`.

---

## JSON config schema

Top-level structure:

```json
{
  "domain_box":    [Lx, Ly, Lz],
  "seed":          <integer>,
  "distributions": { "<name>": { "type": "...", ...params }, ... },
  "shapes":        [ { "type": "...", ...params }, ... ],
  "generator":     { "type": "...", ...params },
  "termination":   { "type": "...", ...params },
  "output":        { "type": "...", ...params }
}
```

`Lz = 0` means a 2D RVE (the third axis collapses). `seed` is optional;
defaults to 42. The five sections each select one or more registered types by
name.

### Catalogue of built-in types

| Section         | `"type"` value             | Params                                   |
|-----------------|----------------------------|------------------------------------------|
| **distributions** | `uniform_real`           | `a` (min), `b` (max)                     |
|                 | `normal`                   | `mean`, `stddev`                         |
|                 | `constant`                 | `value`                                  |
| **shapes**      | `circle_input`             | `pos_x_dist`, `pos_y_dist`, `radius_dist` |
|                 | `sphere_input`             | `pos_x_dist`, `pos_y_dist`, `pos_z_dist`, `radius_dist` |
|                 | `rectangle_input`          | `pos_x_dist`, `pos_y_dist`, `width_dist`, `height_dist` |
|                 | `box_input`                | 3 position + 3 extent dist names         |
| **generator**   | `only_inside`              | `max_attempts`                           |
|                 | `periodic`                 | `max_attempts`                           |
|                 | `random`                   | `max_attempts`                           |
| **termination** | `number_of_inclusions`     | `target` (count)                         |
|                 | `volume_fraction`          | `target_fraction` (0..1)                 |
|                 | `until_full`               | (no params; relies on generator's cap)   |
| **output**      | `gmsh_geo`                 | (no params)                              |
|                 | `voxel`                    | `nx`, `ny`, `nz`                         |
|                 | `vtk_legacy`               | `nx`, `ny`, `nz`                         |

The cross-references in shape inputs (e.g. `"pos_x_dist": "uniform_x"`) refer
to keys in the `distributions` block. The orchestrator builds distributions
first, then resolves these names when constructing inputs.

### Worked examples

- `examples/circles_in_unit_box.json` — 2D, 10 circles, voxel output.
- `examples/spheres_in_cube.json` — 3D, ~5% volume fraction, voxel output.
- `examples/periodic_circles.json` — 2D periodic, 20% volume fraction,
  ParaView-readable VTK output.

Run any of them with `./build/rvegen <path-to.json> <output-path>`.

---

## Adding a new type

The library is open for extension. To add (say) a new shape input type
`triangle_input`:

1. **Library code** — implement the shape if it doesn't exist
   (`include/rvegen/shapes/triangle.h`), add `collision_details(triangle, …)`
   overloads in `intersection/collision_details.h` for every existing shape
   pair (the compiler tells you which are missing), and implement
   `is_inside(point)`.

2. **Input** — `include/rvegen/inputs/triangle_input.h` with:
   - direct C++ ctor
   - schema-driven ctor `(parameter_handler_t const&, distribution_map_t<T> const&)`
   - static `parameters()` method declaring the JSON schema
   - `new_shape()` override

3. **Registration** — one line in
   `include/rvegen/registry/register_inputs.h`:
   ```cpp
   inline void register_triangle_input() {
     input_registry_t<T>::instance()
         .template register_type<triangle_input<T>>("triangle_input");
   }
   ```
   plus a call in `register_all_inputs()`.

JSON config can immediately use `"type": "triangle_input"`. Zero edits to the
CLI, the orchestrator, the existing generators, or any other shape.

The same recipe applies to new distributions, terminations, generators, and
output writers — each lives in its own subdirectory with its own
`register_*.h`.

---

## Architecture

```
JSON config file
    │
    ▼  parse + validate
nlohmann::json
    │
    ▼  parameter_visitor_nlohmann + schema.accept(visitor, handler)
parameter_handler_t       (typed values, extracted from JSON)
    │
    ▼  registry.create(type_name, handler[, deps...])
unique_ptr<Base>          (concrete type chosen by JSON name)
    │
    ▼  pipeline orchestration
generator->compute(inputs, *termination, domain_box)
    │
    ▼
vector<unique_ptr<shape_base<T>>>
    │
    ▼  output_writer->write(file, shapes, domain_box)
.geo / .vox / .vtk
```

Five registries handle the heterogeneous-by-name dispatch, one per concept:

| Registry                  | Base                  | Construction args                   |
|---------------------------|-----------------------|-------------------------------------|
| `distribution_registry_t` | `distribution_base`   | `(handler, Engine&)`                |
| `input_registry_t`        | `shape_input_base`    | `(handler, distribution_map<T>)`    |
| `generator_registry_t`    | `rve_generator_base`  | `(handler)`                         |
| `termination_registry_t`  | `termination_base`    | `(handler, std::array<T,3> box)`    |
| `output_registry_t`       | `output_base`         | `(handler)`                         |

Each is a `numsim_core::object_registry` template instantiation. Registered
types provide a static `parameters()` schema and a ctor matching the
registry's `CtorArgs`. The schema lets the JSON visitor validate input before
construction.

The library is header-only (`rvegen_lib` is an `INTERFACE` CMake target).
Consumers `#include "rvegen/rvegen.h"` and link `rvegen_lib`.

---

## Threading

The library has no global mutable state on the hot path. Each thread builds
its own `std::mt19937` engine, its own distributions referencing that engine,
and its own pipeline; threads never share these.

| Component               | Thread safety                                           |
|-------------------------|---------------------------------------------------------|
| `std::mt19937` engine   | Per-thread (do not share — `mt19937` is not threadsafe) |
| Distribution wrappers   | Per-thread (each owns one engine reference)             |
| Registry singletons     | Read-only after `register_all_*()`; safe to query       |
| Generator instance      | Per-thread (cheap to construct; no shared state)        |
| Output writers          | Per-thread for `write()`; the file stream is the user's |

`tests/concurrency_smoke.cpp` verifies:
- Same seed across threads → bit-identical shape sequences (determinism).
- Different seeds → diverging shape sequences (engine isolation).
- 8-thread heavy load → all pipelines run to completion.

For ensemble Monte Carlo or parameter sweeps, the simplest pattern is to run
the CLI in parallel with different seeds — the pipeline is process-level
embarrassingly parallel.

---

## Project layout

```
include/rvegen/
  rvegen.h                  umbrella include
  types.h                   parameter_handler_t, parameter_controller_t aliases

  shapes/                   shape_base + circle, sphere, rectangle, box
                            (header-only; bounding_box_base hierarchy alongside)
  intersection/             collision_details overload set + AABB helpers
                            (compile-time dispatch where possible)
  distributions/            distribution_base + uniform_real, normal, constant
  inputs/                   shape_input_base + circle, sphere, rectangle, box
  termination/              termination_base + number_of_inclusions,
                            volume_fraction, until_full
  generators/               rve_generator_base + only_inside, periodic, random
  output/                   output_base + gmsh_geo, voxel, vtk_legacy
                            + voxel_grid sampling helper
  registry/                 register_*.h per category + build_from_json helper
  json/                     parameter_visitor_nlohmann

apps/rvegen_cli.cpp         the JSON-driven executable (~120 lines)
examples/                   bundled example configs
tests/                      smoke tests, one per layer, all under ctest
```

---

## Build options

| Option                       | Default | Effect                                    |
|------------------------------|---------|-------------------------------------------|
| `RVEGEN_WITH_VTK`            | `OFF`   | Build legacy VTK rendering (dev tool)     |
| `RVEGEN_WITH_VOXELIZATION`   | `OFF`   | Build CGAL-based mesh voxelization        |

The default build pulls in only Boost headers + nlohmann_json. CGAL and VTK
are FetchContent-fallback dependencies if you opt them in but they're not
installed locally.

---

## License

rvegen is **dual-licensed**:

- **GNU General Public License v3.0** (see [`LICENSE`](LICENSE)) — free
  for open-source work whose derivatives are themselves distributed
  under GPL-3.0.
- **Commercial licence** (see [`COMMERCIAL.md`](COMMERCIAL.md)) — for
  proprietary, closed-source use without GPL's copyleft requirements.

The GPL-3.0 covers academic research, open-source projects, and internal
research that isn't redistributed. If you ship a closed-source product
that links rvegen, or your organisation prohibits GPL dependencies in
production, see `COMMERCIAL.md` for the paid licence terms.

The vendored GeometricTools headers under `external/GeometricTools/`
remain governed by their own (Boost Software License) terms; the rvegen
licence does not modify or override them.

---

## Status / roadmap

The current state covers the JSON-driven core, three generator strategies, the
gmsh + voxel + VTK output paths, and a thread-safe distribution layer. See
`REFACTORING_PLAN.md` for the phase-by-phase migration history and remaining
backlog (rotation support, GTE-backed precise collision dispatch, more shape
types).
