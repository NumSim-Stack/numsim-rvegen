# #114 — Voronoi-based polycrystal generator

> Drafted issue + PR text. Branch: `feature/voronoi-polycrystals`.
> **Status: design only — no implementation in this branch.**

## Issue body

Opens the metals / alloys market — the largest computational-mechanics
community. Direct competition with Neper (the gold standard, free,
mature, dominant). Per `ROADMAP.md` this is a Horizon 3 expansion item;
not the headline feature, but it dramatically increases addressable
users.

### Proposed fix

```cpp
template <typename T = double>
class polycrystal_generator final : public rve_generator_base<T> {
public:
  // Schema fields:
  //   target_grain_count: number of grains in the unit cell
  //   grain_size_distribution: name of a distribution (lognormal)
  //   tessellation: "voronoi" | "laguerre" (radius-weighted)
  //   periodic: bool (use periodic Voronoi)

  shape_vector compute(input_vector& inputs,
                       termination_base<T> const& term,
                       std::array<value_type, 3> const& domain_box,
                       progress_options const& opts = {}) override {
    // 1. Sample seed points (count from termination's target).
    // 2. Compute periodic Voronoi (or Laguerre) tessellation.
    // 3. Convert each cell to a `polyhedron_cell<T>` shape.
    // 4. Assign per-grain orientation (uniform random or A2-driven).
    // 5. Return shape vector.
  }
};
```

### Required dependencies

- **Voro++** or **CGAL** for the tessellation. CGAL is heavier but
  produces topologically clean cells with periodic support; Voro++
  is lighter but the periodic implementation is less robust.
- A new `polyhedron_cell<T>` shape: stores its vertex list and face
  topology, implements `is_inside` via half-space tests.

### Two key challenges

**1. Cell-cell collision** — for polycrystal generation, cells fully
fill the domain (no inter-cell gaps). The collision check is
unnecessary if generation is via tessellation directly. Implement as
a collision_details overload that returns false when the cells share
a face (which they always do in tessellation output).

**2. Periodic boundary handling** — for FFT homogenization, the
tessellation must be periodic. Both CGAL and Voro++ have periodic
modes; Neper uses a custom routine.

### Output extensions

- gmsh emits `Volume(...)` per cell with `Physical Volume("grain_N")`
  per-grain tags.
- DAMASK material.config integration: per-grain orientation +
  material — natural fit for crystal plasticity FFT homogenization.

### Acceptance

- [ ] `polycrystal_generator` registered for `"type": "voronoi"`.
- [ ] Periodic tessellation correctness: opposite faces match.
- [ ] Grain-size statistics control: target lognormal distribution
      with mean and std-dev parameters.
- [ ] Test: generate 100 grains in a unit cube with target grain
      size distribution; assert empirical histogram matches the
      input within tolerance.
- [ ] DAMASK material.config emission for crystal plasticity.

### Out of scope for this PR

- Texture (orientation distribution) — depends on #112 FOD support
  landing.
- Grain growth simulation, recrystallization — material-history
  modelling, not generation.
- Mesh-conformal grain boundaries — use the standard gmsh meshing
  on the tessellation output.

## PR body

Closes #114. **Currently draft — no implementation in this branch.**

Implementation work requires:
- ~300 LOC for `polyhedron_cell<T>` (shape with arbitrary face count).
- ~400 LOC for `polycrystal_generator<T>` (tessellation + assignment).
- ~200 LOC for the half-space-based `is_inside` and convex-cell
  collision dispatch.
- ~150 LOC for gmsh emission with per-grain Physical Volume tags.
- ~250 LOC for the DAMASK material.config post-process.
- Vendor or find_package(CGAL) — CGAL is the heaviest dep we'd take
  on; gated behind `RVEGEN_WITH_VORONOI=ON`.
- ~300 LOC of tests.

Estimated 3–5 days of focused work. Worth tackling once Tessera is
past prototype and the metals/alloys community shows interest.
