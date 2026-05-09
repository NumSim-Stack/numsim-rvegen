# #117 — Periodic-mesh generation in `gmsh_geo_writer`

> Drafted issue + PR text. Branch: `feature/periodic-mesh-gmsh`.
> **Status: design only — no implementation in this branch.**

## Issue body

For FE-RVE simulations with periodic boundary conditions, the mesh
nodes on opposite faces must match exactly so the periodic
constraints can be applied. Today's `gmsh_geo_writer` emits geometry
correctly but not periodicity — gmsh's mesher generates random node
distributions, breaks the BC, and FE results are wrong.

A user generating a periodic RVE via the `periodic` generator and
exporting to gmsh has to *manually* edit the .geo file to add
periodicity directives. That defeats the JSON-driven reproducibility
story.

### Proposed fix

Detect that the RVE was produced by a periodic generator (or accept
an explicit `periodic: true` flag in the gmsh_geo writer's JSON
schema) and emit gmsh's `Periodic Surface { N } = { M } Translate { … };`
directives matching opposite faces.

### Required gmsh syntax

For a 2D unit square `[0,Lx] × [0,Ly]`:

```geo
// Lines 1..4 are the four boundary edges (bottom/right/top/left).
Periodic Curve { 3 } = { 1 } Translate { 0, Ly, 0 };  // top ≡ bottom
Periodic Curve { 4 } = { 2 } Translate { -Lx, 0, 0 }; // left ≡ right (gmsh idiom)
```

For 3D unit cube, six face pairings:

```geo
Periodic Surface { 2 } = { 1 } Translate { Lx, 0, 0 };  // +x ≡ −x
Periodic Surface { 4 } = { 3 } Translate { 0, Ly, 0 };
Periodic Surface { 6 } = { 5 } Translate { 0, 0, Lz };
```

Plus inclusions that span the boundary need `Coherence;` to
deduplicate the wrapped half-shapes. The periodic generator already
emits explicit wrap copies in the shape vector, so each wrap becomes
its own Disk/Sphere — gmsh's `Coherence` collapses overlapping
geometry, leaving a clean periodic mesh.

### Acceptance

- [ ] New schema field `periodic` (default false) on `gmsh_geo_writer`.
- [ ] When true, writer emits the `Periodic Curve` (2D) or
      `Periodic Surface` (3D) directives + `Coherence;`.
- [ ] Test: a single-circle periodic RVE → gmsh-able .geo →
      regenerated mesh has matching opposite-face nodes (pair-up
      check with tolerance `Lx / nx_mesh / 1e-3`).
- [ ] Periodic generator's existing tests still pass.

### Implementation outline

1. Add `bool _periodic{false}` member to `gmsh_geo_writer<T>`.
2. Schema-driven ctor: read optional `"periodic"` from handler
   (default false).
3. In `write_2d` / `write_3d`, after the domain `Rectangle/Box`
   declaration emit the `Periodic Curve/Surface` block if periodic.
4. After the inclusion list, emit `Coherence;` to merge overlapping
   geometry from wrap copies.
5. Update `examples/periodic_circles.json` to add a gmsh_geo
   post-process with `"periodic": true` so the example exercises
   the path.

### Notes

- gmsh's `Periodic Surface` directives need the surface IDs of the
  paired faces. The current writer numbers entities sequentially
  starting at 1 (domain) then 2..N for inclusions. Face IDs of the
  domain box come from the `Box(1)` entity's auto-generated face
  numbering — gmsh assigns +x = 2, −x = 1, +y = 4, −y = 3, etc.
  Verify the convention before hardcoding (gmsh ≥ 4.10).
- `Coherence;` is order-sensitive — it must follow ALL geometry
  declarations.
- Mesh-size hints (currently emitted as comments) become important
  here — gmsh defaults can produce a non-uniform mesh that breaks
  periodicity even with the directives. Add explicit
  `Mesh.CharacteristicLengthMin/Max` lines.

### Out of scope

- Same-face mesh-size matching for higher-order elements.
- Conformal mesh generation (gmsh handles that).
- Integration with MeshFEM / FEniCS / FreeFEM consumers.
