# #108 — Packaging metadata for vcpkg / conda-forge

> Drafted issue + PR text. Branch: `feature/packaging-metadata`.
> **Status: design only — no implementation in this branch. Has hard
> prerequisites that should land first.**

## Issue body

For broader distribution beyond direct GitHub clones — vcpkg + conda-
forge are the canonical C++ scientific-software package channels. Both
require packaging metadata that points at a tagged release.

### Hard prerequisites

This task is the LAST one in the queue. It needs:

- ✓ #101 LICENSE — published.
- ✓ #102 CI — green on push.
- ✓ #103 CMake install/export — `find_package(rvegen)` works.
- ✓ #106 Versioning — git tags + project VERSION.
- (rvegen pushed to GitHub at `NumSim-Stack/rvegen`.)
- (At least one tagged release: `v0.1.0`.)

Without ALL of those, packaging metadata is premature: the package
manager fetches a tagged tarball, runs the install rule, and expects
`find_package` to work for downstream users. Each prereq guards a
different failure mode.

### vcpkg

A new ports directory under `vcpkg/ports/rvegen/`:

```cmake
# vcpkg.json
{
  "name": "rvegen",
  "version-string": "0.1.0",
  "description": "RVE generator for FFT/FEM homogenization",
  "homepage": "https://github.com/NumSim-Stack/rvegen",
  "license": "GPL-3.0-or-later",
  "dependencies": [
    "boost-container",
    "eigen3",
    "nlohmann-json",
    "numsim-core"
  ]
}
```

Plus `portfile.cmake`:

```cmake
vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO NumSim-Stack/rvegen
    REF v0.1.0
    SHA512 ...
    HEAD_REF main
)

vcpkg_cmake_configure(SOURCE_PATH "${SOURCE_PATH}")
vcpkg_cmake_install()
vcpkg_cmake_config_fixup(PACKAGE_NAME rvegen)
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
```

### conda-forge

A `recipe/meta.yaml`:

```yaml
{% set name = "rvegen" %}
{% set version = "0.1.0" %}

package:
  name: {{ name|lower }}
  version: {{ version }}

source:
  url: https://github.com/NumSim-Stack/rvegen/archive/v{{ version }}.tar.gz
  sha256: ...

build:
  number: 0

requirements:
  build:
    - {{ compiler('cxx') }}
    - cmake
    - ninja
  host:
    - boost-cpp >=1.83
    - eigen >=3.4
    - nlohmann_json >=3.12
    - numsim-core
  run:
    - boost-cpp >=1.83
    - eigen >=3.4
    - nlohmann_json >=3.12
    - numsim-core

about:
  home: https://github.com/NumSim-Stack/rvegen
  license: GPL-3.0-or-later
  summary: RVE generator for FFT/FEM homogenization
```

### Acceptance

- [ ] `vcpkg/ports/rvegen/` files in the rvegen repo (mirror of what
      gets PR'd to vcpkg/microsoft).
- [ ] `recipe/meta.yaml` ditto for conda-forge.
- [ ] CI workflow that exercises `vcpkg install rvegen --overlay-ports
      ./vcpkg/ports` to confirm the port builds locally.
- [ ] (Out-of-repo) PRs to vcpkg/microsoft and conda-forge/staged-
      recipes — these are the actual publish step.

### Out of scope

- numsim-core packaging — separate PR on that repo, ideally landing
  before rvegen's vcpkg port (since rvegen depends on it). conda-
  forge will need a numsim-core feedstock too.
- Wheels for the Python bindings (#110) — separate cibuildwheel
  pipeline.

## PR body

Closes #108. **Currently draft — no implementation in this branch.**

Implementation work requires:
- ~30 LOC for `vcpkg/ports/rvegen/vcpkg.json`.
- ~30 LOC for `vcpkg/ports/rvegen/portfile.cmake`.
- ~50 LOC for `recipe/meta.yaml` (conda-forge).
- CI extension to dry-run the vcpkg install.

Estimated half-day of work *after* all prerequisites land. Treat
this as a closing item once the rest of the roadmap is in place.
