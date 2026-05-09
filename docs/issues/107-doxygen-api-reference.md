# #107 — Generate API reference (Doxygen / similar)

> Drafted issue + PR text. Branch: `feature/doxygen-api-reference`.

## Issue body

The headers carry good `@brief` / `@tparam` / `@return` Doxygen-style
comments throughout (the original author was disciplined about this).
Running them through Doxygen produces a browsable, cross-linked
reference site with maybe an hour of CMake glue. No reference exists
today; new contributors have to read headers in dependency order to
understand the type tree.

### Proposed fix

Add a `Doxyfile.in` template + a `RVEGEN_BUILD_DOCS` CMake option
gated behind `find_package(Doxygen REQUIRED)`. Default OFF so the
common build doesn't pull in doxygen. With `cmake
-DRVEGEN_BUILD_DOCS=ON ..` and `make docs`, output lands in
`build/docs/html/index.html`.

Inputs: `include/`, `README.md`, `docs/extending.md`. Excludes:
`external/`, `build/`. README serves as the main page.

### Acceptance

- [ ] `cmake -DRVEGEN_BUILD_DOCS=ON .. && make docs` produces
      browsable HTML at `build/docs/html/index.html`.
- [ ] Default build (without the option) is unchanged.
- [ ] The mainpage links to the contributor doc (`docs/extending.md`)
      and to a per-namespace listing.

## PR body

Closes #107.

### Summary

- Adds `docs/Doxyfile.in` configured for the rvegen header tree with
  C++23 preprocessing hints.
- Adds an opt-in `RVEGEN_BUILD_DOCS` CMake option that requires
  `find_package(Doxygen)` and creates a `docs` custom target.
- README is the Doxygen mainpage; `docs/extending.md` and the entire
  `include/` tree are inputs.
- Default behaviour: docs are NOT built. Existing CI / standard
  developer flow is unchanged.

### Test plan

- [x] `cmake -DRVEGEN_BUILD_DOCS=ON ..` finds doxygen 1.9.x.
- [x] `make docs` produces `build/docs/html/index.html` with the
      class hierarchy populated.
- [x] `cmake -DRVEGEN_BUILD_DOCS=OFF ..` (default) doesn't try to
      find doxygen.
- [x] Existing 15/16 ctest suite still passes.

### Out of scope

- ReadTheDocs / GitHub Pages publishing — deploy hook can be added
  to the CI workflow once rvegen lands on GitHub.
- Custom Doxygen theme (e.g. doxygen-awesome-css) — the default
  theme is functional; theming is a follow-up.
