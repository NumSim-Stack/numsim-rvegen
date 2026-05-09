# #115 — STL/PLY mesh import as inclusion type

> Drafted issue + PR text. Branch: `feature/stl-ply-mesh-import`.
> **Status: design only — no implementation in this branch.**

## Issue body

Real-world particle microstructures from μ-CT (cement, concrete, AM
powder beds, biological tissues) come as triangle-mesh files (STL,
PLY). Currently rvegen only supports analytical shapes — to use a
μ-CT-derived particle, the user has to approximate it with a sphere
or ellipse, losing fidelity.

Adding mesh-based inclusions opens up the cement / concrete / AM
powder communities, where this is a real workflow.

### Proposed fix

```cpp
template <typename T = double>
class mesh_inclusion
    : public numsim_core::static_indexing<mesh_inclusion<T>, shape_base<T>> {
public:
  triangle_mesh<T> mesh;       // imported from STL/PLY
  std::array<T, 3> position;   // translate the mesh
  std::array<T, 3> rotation;   // Euler angles or quaternion
  T scale;                     // uniform scale factor

  bool is_inside(std::array<T,3> const& p) const override {
    // Transform p back to mesh-local coords, then test:
    //   - Ray-casting parity test, OR
    //   - Generalized winding number (more robust)
    // CGAL's Side_of_triangle_mesh<> primitive does this in O(log N).
  }

  void make_bounding_box() override { /* AABB of transformed mesh */ }
};
```

### Loaders

- STL (binary + ASCII): standard format. ~80 LOC.
- PLY (binary + ASCII): also standard. ~150 LOC (more flexible).

Both available off-the-shelf via `tinyply` / `tinystl` headers — could
vendor either. Or use libigl which is heavier.

### Collision

mesh-mesh collision is genuinely hard. Options:
- libigl/CGAL `intersect_other` — exact, expensive.
- BVH of triangle pairs + GJK on overlapping leaves — practical for
  RVE generation where most pairs don't overlap.
- Approximation: convex-hull collision with a `dynamic_cast`-style
  fallback to AABB for non-overlapping pairs.

For the first implementation, BVH + per-leaf triangle-triangle
intersection (Möller-Trumbore) is correct and fast enough for
moderate RVE sizes. fcl (Flexible Collision Library) wraps this
nicely.

### Acceptance

- [ ] `mesh_inclusion<T>` shape registered with is_inside via ray-cast
      and to_mesh that returns a (transformed) copy of the loaded
      mesh.
- [ ] STL loader (binary + ASCII) for common files.
- [ ] PLY loader (binary + ASCII) — can defer if STL is enough.
- [ ] Mesh-mesh collision dispatcher entry.
- [ ] Test: load a single tetrahedron mesh, voxelize; resulting phase
      grid matches an analytical reference within tolerance.

### Out of scope

- Mesh repair (degenerate triangles, non-manifold edges) — assume
  inputs are watertight; a malformed mesh is the user's problem.
- Mesh simplification — provide `to_mesh()` that returns the loaded
  mesh as-is; users wanting low-poly previews handle that downstream.
- Per-mesh anisotropic material — use #112 (FOD) machinery once it
  lands.

## PR body

Closes #115. **Currently draft — no implementation in this branch.**

Implementation work requires:
- ~150 LOC for `mesh_inclusion<T>` (shape header).
- ~80 LOC for STL loader.
- ~150 LOC for PLY loader.
- ~200 LOC for BVH-based mesh-mesh collision.
- ~80 LOC for `mesh_inclusion_input` (parses path-to-mesh + transform).
- ~40 LOC for example JSON exercising μ-CT particle import.
- ~150 LOC of tests (single tetrahedron voxelization, two-mesh
  collision, mesh-vs-sphere collision).
- Vendor `tinystl` (or equivalent header-only STL/PLY loader).

Estimated 1–2 days of focused work. Tackle when the cement/concrete
or AM-powder use case becomes urgent.
