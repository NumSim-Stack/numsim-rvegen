# Changelog

All notable changes to rvegen are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
once `1.0.0` is reached. Pre-1.0 versions may break compatibility on
minor bumps; the changelog calls out breaking changes explicitly.

## [Unreleased]

### Added
- MIT LICENSE.
- CMake project version, description, and homepage URL declared via the
  `project()` call.
- This CHANGELOG.

## [0.1.0] — unreleased

First tagged release. Captures the post-import state of the codebase
including the dispatcher rework and the GUI-prep work.

### Added
- Layered header-only library at `include/rvegen/` covering shapes,
  intersection, distributions, inputs, generators, terminations,
  post-processes, JSON visitor, and per-category registries.
- JSON-driven CLI (`rvegen <config.json>`) with `postprocess: [...]`
  fan-out — one config can emit gmsh `.geo`, voxel grids, VTK,
  SVG/SVG-3D, and interactive Three.js HTML in a single run.
- Static-indexing-keyed collision dispatcher: `std::vector<std::vector<fn_t>>`
  indexed by shape id, no RTTI on the hot path.
- Visualization boundary types (`triangle_mesh`, `mesh_dispatcher`,
  per-shape `to_mesh` overloads) for downstream renderers (Tessera).
- Cooperative progress + cancellation hooks on every generator
  (`progress_options` with `cancel`/`on_progress`/`emit_every`).
- Schema metadata (`description`, `min`, `max`, `units`) on every
  registered type; enforced by `schema_audit_test`.
- `BUILD_TESTING` gate so consumers can `add_subdirectory(rvegen)`
  without inheriting rvegen's tests.

[Unreleased]: https://github.com/numsim-stack/rvegen/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/numsim-stack/rvegen/releases/tag/v0.1.0
