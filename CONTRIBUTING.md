# Contributing to rvegen

## Quick start

```bash
# Sibling-tree layout: rvegen falls back to ../../numsim-core/include when no
# installed numsim-core CMake package is found. To use an installed one
# instead, add -DCMAKE_PREFIX_PATH=<numsim-core prefix> to the configure line.
# Needs g++-14 (or newer) for <print>: add -DCMAKE_CXX_COMPILER=g++-14 if
# your default compiler is older.
mkdir -p numsim-stack && cd numsim-stack
git clone https://github.com/NumSim-Stack/numsim-core.git
git clone https://github.com/NumSim-Stack/numsim-rvegen.git

cd numsim-rvegen/rvegen
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
```

CI also builds Debug with `-fsanitize=address,undefined` and with Clang.
To reproduce the sanitizer job locally add
`-DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-sanitize-recover=undefined"`.
`cli_check_*` tests (Python 3) run the CLI on every `examples/*.json` and
inspect the files written, so a new example config is tested automatically.
They report *Skipped* (not Passed) when a config writes a `.geo` and gmsh is
missing; CI configures with `-DRVEGEN_REQUIRE_GMSH=ON` so that cannot happen
there.

The Clang job needs a numsim-core whose `static_indexing` draws ids lazily.
With the older eager version clang gives every shape type id 0; the
collision and mesh dispatchers then throw at registration ("share
static_indexing id 0") and `shape_id_unique_test` fails, instead of
dispatching circles as box x sphere.
After adding a registered type, refresh the README list with
`tests/readme_types_check.py --rvegen build/rvegen --readme README.md --update`.

## Branch layout

- `main` — release-track. CI must be green.
- `feature/<short-name>` — one PR per feature; rebase before merge if main has moved.

## Bumping the pinned numsim-core ref in CI

`.github/workflows/ci.yml` checks out numsim-core at a specific SHA so
breaking changes there don't cascade into every rvegen PR. To bump:

1. Decide on the target ref:
   ```bash
   gh api repos/NumSim-Stack/numsim-core/commits/main --jq '.sha'
   ```
2. Update the `ref:` line in the **Checkout numsim-core (pinned)** step of
   `.github/workflows/ci.yml`. The pin must stay at or after numsim-core
   `ef6d2ac` ("make the installed package consumable"): CI builds and
   installs numsim-core, and rvegen finds it with
   `find_package(numsim-core CONFIG)`.
3. Push the change on a feature branch and confirm the workflow stays
   green (gcc Release, gcc Debug with ASan/UBSan, clang, install-smoke, Python pytest).
4. Merge.

Bump deliberately — typically when rvegen needs a new numsim-core
feature, or as part of a quarterly sync. Don't bump speculatively.

## Adding a new shape

See `docs/extending.md` for the full walkthrough. Short version:

1. Add the shape header under `include/rvegen/shapes/`. Inherit
   `numsim_core::static_indexing<your_shape<T>, shape_base<T>>` and
   implement the `shape_base<T>` virtuals.
2. Add a unit test in `tests/extra_types_smoke.cpp`, and the type to the
   id set in `tests/shape_id_unique_test.cpp`.
3. If you want it driven from JSON config, add an input under
   `include/rvegen/inputs/` and register it in
   `include/rvegen/registry/register_inputs.h`.
4. Run the schema audit: `ctest -R schema_audit_test`.

## Adding a new metadata field for shapes

The generic `info` container (`include/rvegen/metadata/info.h`) is the
single integration point. To add a new field consumed by every input:

1. Extend `shape_input_base::read_metadata` (one place) with the
   handler-key lookup + the call into `_info`.
2. Add the schema entry to each input's `parameters()` (one line per
   input, mirroring the existing `phase_name` / `metadata` entries).
3. Add a test exercising one input + the new field.

The first two steps are mechanical; the metadata container's design
exists so step 1 only happens once.
