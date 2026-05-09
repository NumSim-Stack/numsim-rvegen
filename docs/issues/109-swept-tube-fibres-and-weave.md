# #109 — Swept-tube fibre shape + woven-pattern generator

> Drafted issue + PR text. Branch: `feature/swept-tube-fibres`.
> **Status: design only — no implementation in this branch.**

## Issue body

The headline feature for opening textile composites — currently
under-served by free tools. rvegen can do circles / spheres / ellipses
but not curved fibres or interlaced weaves. Adding this expands the
addressable user base meaningfully (this is the Horizon 2 niche-
dominance feature in `ROADMAP.md`).

### Two pieces

**1. `polyline_tube` shape** — a swept tube along a 3D centerline.

```cpp
template <typename T = double>
class polyline_tube
    : public numsim_core::static_indexing<polyline_tube<T>, shape_base<T>> {
public:
  std::vector<std::array<T, 3>> centerline;  // sampled points
  T radius;
  // ...
  bool is_inside(std::array<T,3> const& p) const override {
    // Closest-point-to-segment over each segment of the centerline.
    // Inside if min-distance ≤ radius.
  }
  void make_bounding_box() override { /* min/max over centerline ± radius */ }
};

// Mesh side (for Tessera viewport):
template <typename T>
triangle_mesh<T> to_mesh(polyline_tube<T> const& t) {
  // Frenet-frame ring extrusion: at each centerline point compute a
  // local frame perpendicular to the tangent, generate ring vertices
  // around it, connect consecutive rings with quad-strip triangles.
}
```

**2. `weave_generator`** — emits N polyline_tubes with correct
over/under interlock topology.

```cpp
class weave_generator final : public rve_generator_base<T> {
public:
  std::size_t warp_count;
  std::size_t weft_count;
  std::string weave_pattern;     // "plain" | "twill_2_2" | "satin_5"
  T fibre_radius;
  T deflection_amplitude;
  T fibre_pitch;                 // centre-to-centre spacing
};
```

### Algorithmic core

For a plain weave (1×1 over/under):
- Warps run along x, wefts along y, both at z = 0
- At each crossing (i, j), warp goes UP if (i+j) is even, DOWN otherwise
- Deflection: warp's z-coordinate at crossing = +deflection if up, −deflection if down
- The deflection pattern between crossings follows a smooth interpolation (sinusoidal, cubic spline, etc.)

For twill: pattern shifts with each row.
For satin: longer floats between crossings.

### Acceptance

- [ ] `polyline_tube<T>` shape registered with `to_mesh` for VTK
      rendering and `is_inside` for voxelization.
- [ ] `weave_generator<T>` registered for `"type": "weave"`.
- [ ] Plain weave produces interlocked fibres that voxelize correctly
      (no fibre-fibre overlap; correct over/under topology).
- [ ] Periodic boundary handling: warp/weft fibres wrap correctly
      across opposite faces.
- [ ] Test: 4×4 plain weave at 0.30 fibre volume fraction; voxelize;
      check that the projected pattern matches a known reference image.

### Implementation notes

**polyline_tube collision**: tube-tube intersection is the hard part.
Approach: discretize each tube into a list of capsules (one per
segment) and pairwise-test capsule-capsule intersection (a known
analytical query). GTE has `gte::Capsule3<T>` and `TIQuery<Capsule3,
Capsule3>`. ~50 LOC plus capsule-capsule registration in
`collision_dispatcher`.

**Weave deflection**: the natural mathematical model is a series of
cubic Hermite splines connecting the over/under crossing points.
Tangents can be set to zero at crossings (no slope discontinuity) or
to follow the local mean fibre direction. Simple and visually
convincing for plain weaves.

**Fibre voxelization**: each polyline_tube voxelizes via its
`is_inside`. The voxel grid resolution needs to be high enough that
a tube of radius `r` registers — rule of thumb `r > 2*dx`.

### Out of scope for this PR

- Knit / braided fabric topologies — separate issue.
- Curved beam-element output for FE solvers — voxel + mesh suffices
  for FFT and gmsh-meshed FE workflows.
- Yarn-level granularity (each fibre as a tow of multiple filaments).
- Mechanical contact between fibres in deflected regions — the
  generator places them with deflection but doesn't model contact
  forces.

## PR body

Closes #109. **Currently draft — no implementation in this branch.**

Implementation work requires:
- ~150 LOC for `polyline_tube<T>` (shape header).
- ~50 LOC for `polyline_tube_input<T>` (input header).
- ~200 LOC for `weave_generator<T>` (with plain / twill / satin
  patterns).
- ~30 LOC for `to_mesh(polyline_tube)` (Frenet-frame ring extrusion).
- ~80 LOC for capsule-capsule collision registration.
- ~40 LOC for at least one new example JSON.
- ~150 LOC of tests (single-fibre placement, plain-weave-4x4,
  periodic-boundary correctness).

Estimated 1–2 days of focused work. Highest-impact remaining feature
per the roadmap; tackle this when the GUI is past prototype and
needs textile-composite use cases to demo.
