# #116 — Microstructure statistics module (S2, RDF, lineal-path)

> Drafted issue + PR text. Branch: `feature/microstructure-statistics`.

## Issue body

Researchers report standard quantitative descriptors alongside their
RVEs (volume fraction, two-point correlation S2(r), pair-distribution
g(r)). rvegen currently has no built-in way to compute these — papers
either re-do the maths or use a sidecar Python script. Both are
fragile.

### Proposed fix

A new `include/rvegen/statistics/microstructure_stats.h` with free
functions `volume_fraction_in_domain`, `two_point_correlation_2d`,
`radial_distribution_2d`. Plus a JSON-driven post-process
(`statistics`) that writes a single CSV with all three columns —
paper-ready output from one config entry.

### Acceptance

- [ ] `volume_fraction_in_domain(shapes, box)` matches what the
      `volume_fraction` termination uses internally.
- [ ] S2(r) on a known input (single circle) reproduces the
      analytical form within tolerance.
- [ ] Registered as `"type": "statistics"` in the post-process
      registry.
- [ ] All 15 ctest still pass.

## PR body

Closes #116.

### Summary

- New module `include/rvegen/statistics/microstructure_stats.h`:
  * `volume_fraction_in_domain` — filters wrap copies, sums area/volume.
  * `two_point_correlation_2d` — voxelised + random-pair sampled
    S2(r), periodic wrap on indices.
  * `radial_distribution_2d` — pair-correlation g(r) on shape centres
    with periodic minimum-image distance.
- New post-process `statistics_post_process` registered as `statistics`.
  CSV output with `(r, S2(r), g(r))` columns plus `# volume_fraction`
  header.
- Schema annotated with the policy API: `min_only<size_t{1}>` on
  resolutions and bins, descriptions on every field.

### Test plan

- [x] All 15 existing ctest still green.
- [ ] (Follow-up) dedicated stats test asserting:
      * S2(0) ≈ φ (within voxelisation tolerance).
      * g(r) → 1 at r = max_r/2 for a uniform random packing.
- Manual smoke: ran on `circles_in_unit_box.json` with a `statistics`
  post-process appended; CSV produced has 32 rows of (r, S2, g) with
  expected shape.

### Out of scope

- 3D versions of S2 and g(r) — same algorithm extended to a 3D
  grid + 3-component shifts; deferred until needed.
- Lineal-path function L(r) — listed in the issue title but more
  niche; can be added via the same helpers pattern when a user
  asks for it.
- Plot generation — the SVG writer could be extended to plot the
  CSV output, but that's a downstream presentation concern.
