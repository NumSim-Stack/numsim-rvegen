# Packaging metadata for rvegen

This directory holds the packaging metadata for distribution channels
beyond direct GitHub clones. The files here mirror what eventually gets
PR'd to the upstream package-manager registries.

## vcpkg

`vcpkg/ports/rvegen/` — files for the `microsoft/vcpkg` ports tree.

Local install for testing:

```bash
vcpkg install rvegen --overlay-ports ./packaging/vcpkg/ports
```

To publish: PR the `vcpkg/ports/rvegen/` directory contents to
[microsoft/vcpkg](https://github.com/microsoft/vcpkg) under `ports/rvegen/`,
along with a `versions/r-/rvegen.json` baseline-version entry.

**Prerequisites before publishing**:

- [ ] `numsim-core` has a vcpkg port (it's listed as a dep in `vcpkg.json`)
- [ ] A `v0.1.0` tag exists on this repo
- [ ] The placeholder `SHA512 0` in `portfile.cmake` is replaced with the
      real hash (`vcpkg_get_archive_sha512` or equivalent)

## conda-forge

`conda-forge/recipe/meta.yaml` — the conda-forge feedstock recipe.

To publish: open a PR to
[conda-forge/staged-recipes](https://github.com/conda-forge/staged-recipes)
with the contents of `conda-forge/recipe/`. Once merged, conda-forge
admins automatically create the `rvegen-feedstock` repo.

**Prerequisites before publishing**:

- [ ] `numsim-core` has a conda-forge feedstock (it's a `host`/`run` dep)
- [ ] A `v0.1.0` tag exists on this repo
- [ ] The placeholder `sha256: 000...` is replaced with the real hash
      (`curl -L | sha256sum` of the release tarball)

## Why the metadata is committed before the publish step

vcpkg and conda-forge each fetch a tagged release tarball; without a
release tag, the metadata can't be published. Committing it now lets:

- CI exercise `vcpkg install rvegen --overlay-ports …` against a local
  source build.
- Downstream consumers vendor the port file and use it as an overlay
  port even before the upstream PR lands.
- The publish step itself becomes a mechanical task once the release
  cuts (replace placeholder hashes, open the PRs).
